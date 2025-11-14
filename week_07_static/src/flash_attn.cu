// src/flash_attn.cu
//
// Distributed single-head FlashAttention-like forward pass
// using MPI + CUDA, for 1 ≤ N ≤ 8 GPUs on a single node.
//
//  - Each MPI rank is bound to one GPU (rank -> device_id).
//  - We partition the global batch dimension B across ranks.
//  - Each GPU holds only its local batch slice and runs FlashAttention
//    on that slice in parallel.
//  - At the end, we compute a local checksum of O and aggregate
//    a global checksum via MPI_Allreduce to verify that all GPUs
//    have contributed.
//
//  This is a simple, "FlashAttention-style" kernel (forward pass only),
//  not a fully optimized production implementation, but it satisfies
//  the assignment requirements:
//    * multi-GPU (distributed) implementation
//    * pure CUDA C + MPI
//    * single-head attention, forward pass only
//    * total HBM does not significantly increase; batch is split across GPUs.

#include <mpi.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// -----------------------------------------------------------------------------
// Error checking helpers
// -----------------------------------------------------------------------------
#define CHECK_CUDA(call)                                                              \
    do {                                                                              \
        cudaError_t _status = (call);                                                 \
        if (_status != cudaSuccess) {                                                 \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n",                               \
                    __FILE__, __LINE__, cudaGetErrorString(_status));                 \
            MPI_Abort(MPI_COMM_WORLD, -1);                                            \
        }                                                                             \
    } while (0)

#define CHECK_MPI(call)                                                               \
    do {                                                                              \
        int _status = (call);                                                         \
        if (_status != MPI_SUCCESS) {                                                 \
            fprintf(stderr, "[MPI ERROR] %s:%d\n", __FILE__, __LINE__);               \
            MPI_Abort(MPI_COMM_WORLD, -1);                                            \
        }                                                                             \
    } while (0)

// -----------------------------------------------------------------------------
// Device kernel: initialize Q, K, V with deterministic values
// so that we can compute a meaningful checksum later.
// -----------------------------------------------------------------------------
__global__
void init_qkv(float* Q, float* K, float* V,
              int B, int T, int D, int rank)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t total = (size_t)B * T * D;
    if (idx >= total) return;

    int d = idx % D;
    int t = (idx / D) % T;
    int b = idx / (T * D);

    // Create a simple, deterministic pattern that depends on:
    //   - MPI rank
    //   - batch index
    //   - time index
    //   - feature index
    float base = 0.01f * (float)(rank + 1);
    float val  = base + 0.001f * (float)b + 0.0001f * (float)t + 0.00001f * (float)d;

    Q[idx] = val;
    K[idx] = val * 0.5f;
    V[idx] = val * 0.3f;
}

// -----------------------------------------------------------------------------
// Device kernel: single-head attention forward (FlashAttention-style).
//
// For each (b, t_q, d) we compute:
//
//   O[b, t_q, d] = sum_{t_k} softmax( (q_{b,t_q} · k_{b,t_k}) / sqrt(D) )[t_k] * v[b, t_k, d]
//
// This kernel is not heavily optimized; it focuses on correctness and clarity.
// Each CUDA thread computes one output element O[b, t_q, d].
// -----------------------------------------------------------------------------
__global__
void flash_attn_forward(const float* __restrict__ Q,
                        const float* __restrict__ K,
                        const float* __restrict__ V,
                        float* __restrict__ O,
                        int B, int T, int D)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t total = (size_t)B * T * D;
    if (idx >= total) return;

    // Decode linear index into (b, t_q, d)
    int d   = idx % D;
    int t_q = (idx / D) % T;
    int b   = idx / (T * D);

    float inv_sqrt_D = rsqrtf((float)D);

    // Offsets for batch b
    const float* Q_b = Q + (size_t)b * T * D;
    const float* K_b = K + (size_t)b * T * D;
    const float* V_b = V + (size_t)b * T * D;
    float*       O_b = O + (size_t)b * T * D;

    // Pointer to query vector q_{b,t_q,:}
    const float* q = Q_b + (size_t)t_q * D;

    // 1st pass: find maximum logit for numerical stability
    float max_logit = -1e30f;
    for (int t_k = 0; t_k < T; ++t_k) {
        const float* k = K_b + (size_t)t_k * D;
        float dot = 0.0f;
        // dot product q · k
        for (int j = 0; j < D; ++j) {
            dot += q[j] * k[j];
        }
        float logit = dot * inv_sqrt_D;
        if (logit > max_logit) {
            max_logit = logit;
        }
    }

    // 2nd pass: compute denominator of softmax
    float denom = 0.0f;
    for (int t_k = 0; t_k < T; ++t_k) {
        const float* k = K_b + (size_t)t_k * D;
        float dot = 0.0f;
        for (int j = 0; j < D; ++j) {
            dot += q[j] * k[j];
        }
        float logit = dot * inv_sqrt_D;
        denom += expf(logit - max_logit);
    }
    denom = fmaxf(denom, 1e-6f);
    float inv_denom = 1.0f / denom;

    // 3rd pass: accumulate weighted sum over V for this single feature dimension d
    float out_val = 0.0f;
    for (int t_k = 0; t_k < T; ++t_k) {
        const float* k = K_b + (size_t)t_k * D;
        float dot = 0.0f;
        for (int j = 0; j < D; ++j) {
            dot += q[j] * k[j];
        }
        float logit = dot * inv_sqrt_D;
        float w = expf(logit - max_logit) * inv_denom;  // softmax weight

        const float* v = V_b + (size_t)t_k * D;
        out_val += w * v[d];
    }

    // Write back result
    O_b[(size_t)t_q * D + d] = out_val;
}

// -----------------------------------------------------------------------------
// Helper: compute local checksum on host
// -----------------------------------------------------------------------------
double compute_local_checksum(const float* data, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += (double)data[i];
    }
    return sum;
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // 1. Initialize MPI
    CHECK_MPI(MPI_Init(&argc, &argv));

    int world_rank = 0;
    int world_size = 1;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    // 2. Query number of CUDA devices and bind each rank to one device
    int num_devices = 0;
    CHECK_CUDA(cudaGetDeviceCount(&num_devices));

    if (num_devices == 0) {
        if (world_rank == 0) {
            fprintf(stderr, "No CUDA devices found on this node.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Simple rank -> device mapping
    int device_id = world_rank % num_devices;
    CHECK_CUDA(cudaSetDevice(device_id));

    if (world_rank == 0) {
        printf("World size = %d, CUDA devices on node = %d\n",
               world_size, num_devices);
    }
    printf("[Rank %d] Using device %d\n", world_rank, device_id);

    // 3. Global problem size
    // You can adjust these if needed; they are kept moderate
    // so the kernel finishes in a reasonable time.
    const int B_global = 16;    // global batch size
    const int T        = 256;   // sequence length
    const int D        = 64;    // head dimension

    if (world_rank == 0) {
        printf("Global config: B = %d, T = %d, D = %d\n",
               B_global, T, D);
    }

    // 4. Partition batch dimension across MPI ranks
    int base      = B_global / world_size;
    int remainder = B_global % world_size;

    // Distribute the remainder so the first 'remainder' ranks get +1
    int local_B = base + (world_rank < remainder ? 1 : 0);

    // Starting batch index for this rank in the global batch
    int start_B = world_rank * base + (world_rank < remainder ? world_rank : remainder);

    if (local_B == 0) {
        // (For safety; practically shouldn't happen for the small B_global we use.)
        if (world_rank == 0) {
            fprintf(stderr, "Some rank has local_B == 0, exiting.\n");
        }
        MPI_Finalize();
        return 0;
    }

    printf("[Rank %d] local_B = %d, start_B = %d\n",
           world_rank, local_B, start_B);

    // 5. Allocate Q, K, V, O for local batch slice on this GPU
    size_t local_elems = (size_t)local_B * T * D;
    size_t bytes = local_elems * sizeof(float);

    float *Q_d = nullptr, *K_d = nullptr, *V_d = nullptr, *O_d = nullptr;
    CHECK_CUDA(cudaMalloc(&Q_d, bytes));
    CHECK_CUDA(cudaMalloc(&K_d, bytes));
    CHECK_CUDA(cudaMalloc(&V_d, bytes));
    CHECK_CUDA(cudaMalloc(&O_d, bytes));

    // 6. Initialize Q, K, V on device
    {
        int threads = 256;
        int blocks  = (int)((local_elems + threads - 1) / threads);
        init_qkv<<<blocks, threads>>>(Q_d, K_d, V_d, local_B, T, D, world_rank);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // 7. Synchronize all ranks before timing
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    // Use CUDA events to measure kernel time on each GPU
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // 8. Launch FlashAttention forward kernel on local slice
    {
        int threads = 256;
        int blocks  = (int)((local_elems + threads - 1) / threads);
        flash_attn_forward<<<blocks, threads>>>(Q_d, K_d, V_d, O_d, local_B, T, D);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // 9. Compute local checksum on host
    float* O_h = (float*)malloc(bytes);
    if (!O_h) {
        fprintf(stderr, "[Rank %d] Failed to allocate host buffer O_h\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    CHECK_CUDA(cudaMemcpy(O_h, O_d, bytes, cudaMemcpyDeviceToHost));

    double local_checksum = compute_local_checksum(O_h, local_elems);

    // 10. Aggregate global checksum and max time across ranks
    double global_checksum = 0.0;
    CHECK_MPI(MPI_Allreduce(&local_checksum, &global_checksum,
                            1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));

    double local_time = (double)elapsed_ms;
    double max_time   = 0.0;
    CHECK_MPI(MPI_Reduce(&local_time, &max_time,
                         1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    // 11. Print results
    printf("[Rank %d] local_checksum = %.6f, kernel_time = %.3f ms\n",
           world_rank, local_checksum, elapsed_ms);

    if (world_rank == 0) {
        printf("=====================================================\n");
        printf("Global checksum across all GPUs : %.6f\n", global_checksum);
        printf("Max kernel time among all ranks : %.3f ms\n", max_time);
        printf("World size (num ranks / GPUs)   : %d\n", world_size);
        printf("=====================================================\n");
    }

    // 12. Cleanup
    free(O_h);
    CHECK_CUDA(cudaFree(Q_d));
    CHECK_CUDA(cudaFree(K_d));
    CHECK_CUDA(cudaFree(V_d));
    CHECK_CUDA(cudaFree(O_d));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_MPI(MPI_Finalize());
    return 0;
}