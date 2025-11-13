// flash_attention.cu
// Week 4 – FlashAttention implementation in CUDA
// Single-head scaled dot-product attention with an IO-aware, tiled kernel
// using online softmax. Includes a naive CPU reference implementation
// and a simple test in main().
//
// Q, K, V, O are all stored as row-major arrays of shape (N, d):
//   - N: sequence length
//   - d: head dimension
//
// Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
//
// This file is fully self-contained: you can compile and run it with:
//
//   nvcc -O3 -std=c++17 flash_attention.cu -o flash_attention
//   ./flash_attention
//
// In your course repository, you will likely only keep the kernel
// (flash_attn_forward_kernel) and its wrapper, and rely on the
// instructor's test harness instead of this main().

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <cassert>

// ---------------------------
// Configurable parameters
// ---------------------------

// Number of query rows per thread block (B_r)
#define BR 32
// Number of key/value rows per tile (B_c)
#define BC 32
// Maximum supported head dimension d.
// If your assignment uses a larger d (e.g., 256), increase this value.
#define MAX_D 128

// Simple CUDA error checker to help with debugging
inline void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error at %s: %s\n", msg, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

// -------------------------------------
// CPU reference: naive attention (O(N^2))
// -------------------------------------

// Compute single-head scaled dot-product attention on the CPU:
//   O = softmax(Q K^T / sqrt(d)) V
//
// Q, K, V, O: shape (N, d), row-major.
void attention_forward_cpu(const float *Q,
                           const float *K,
                           const float *V,
                           float *O,
                           int N, int d) {
    const float inv_sqrt_d = 1.0f / std::sqrt((float)d);

    // Process one query row at a time
    for (int i = 0; i < N; ++i) {
        const float *Qi = Q + i * d;

        // 1) Find max score for numerical stability
        float max_score = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N; ++j) {
            const float *Kj = K + j * d;
            float score = 0.0f;
            for (int k = 0; k < d; ++k) {
                score += Qi[k] * Kj[k];
            }
            score *= inv_sqrt_d;
            if (score > max_score) {
                max_score = score;
            }
        }

        // 2) Compute the normalizer l_i = sum_j exp(score_ij - max_score)
        float l_i = 0.0f;
        for (int j = 0; j < N; ++j) {
            const float *Kj = K + j * d;
            float score = 0.0f;
            for (int k = 0; k < d; ++k) {
                score += Qi[k] * Kj[k];
            }
            score *= inv_sqrt_d;
            l_i += std::exp(score - max_score);
        }

        // 3) Accumulate O_i = sum_j softmax_ij * V_j
        float *Oi = O + i * d;
        for (int k = 0; k < d; ++k) {
            Oi[k] = 0.0f;
        }

        for (int j = 0; j < N; ++j) {
            const float *Kj = K + j * d;
            const float *Vj = V + j * d;
            float score = 0.0f;
            for (int k = 0; k < d; ++k) {
                score += Qi[k] * Kj[k];
            }
            score *= inv_sqrt_d;
            float p_ij = std::exp(score - max_score) / l_i;
            for (int k = 0; k < d; ++k) {
                Oi[k] += p_ij * Vj[k];
            }
        }
    }
}

// -------------------------------------
// CUDA FlashAttention kernel
// -------------------------------------

// This kernel implements a simplified FlashAttention-style algorithm:
//
//   - It uses tiling over keys/values of size B_c (BC) and
//     processes BR query rows per thread block.
//   - Each thread is responsible for exactly one query row i.
//     This guarantees that there are no write conflicts on O_i.
//   - For each query row i, the kernel maintains:
//       m_i : running maximum score over all keys seen so far
//       l_i : running normalizer (denominator) of the softmax
//       O_i : running output row
//   - Keys and values are loaded tile-by-tile into shared memory, and
//     we update (m_i, l_i, O_i) in an online fashion.
//
// The math follows the "online softmax" derivation:
//
//   When a new tile has max score m_ij and contributes sum of exp scores
//   and weighted values, we merge it with the previous state (m_i, l_i, O_i)
//   by moving to the new global max m_new, and rescaling old and new parts.

__global__
void flash_attn_forward_kernel(const float * __restrict__ Q,
                               const float * __restrict__ K,
                               const float * __restrict__ V,
                               float * __restrict__ O,
                               int N, int d) {
    // Each thread controls one query row.
    const int row_in_block = threadIdx.x;          // 0..BR-1
    const int i = blockIdx.x * BR + row_in_block;  // global row index for this thread

    if (i >= N) {
        return; // out-of-bounds threads exit immediately
    }

    // Shared memory tiles for K and V.
    // Each tile has BC rows and at most d columns (up to MAX_D).
    __shared__ float K_tile[BC * MAX_D];
    __shared__ float V_tile[BC * MAX_D];

    const float inv_sqrt_d = rsqrtf((float)d);
    const float neg_inf = -1e30f;

    // State for this query row i
    float m_i = neg_inf;  // running max
    float l_i = 0.0f;     // running normalizer
    float *Oi = O + i * d;

    // Initialize O_i = 0
    for (int k = 0; k < d; ++k) {
        Oi[k] = 0.0f;
    }

    // Pointer to Q_i in global memory
    const float *Qi = Q + i * d;

    // Iterate over tiles of keys/values
    for (int key_start = 0; key_start < N; key_start += BC) {
        // 1. Load K, V tiles into shared memory
        // Each thread loads several rows (stride blockDim.x).
        for (int r = row_in_block; r < BC; r += blockDim.x) {
            int global_row = key_start + r;
            float *K_row_tile = &K_tile[r * MAX_D];
            float *V_row_tile = &V_tile[r * MAX_D];

            if (global_row < N) {
                const float *K_row = K + global_row * d;
                const float *V_row = V + global_row * d;
                for (int kk = 0; kk < d; ++kk) {
                    K_row_tile[kk] = K_row[kk];
                    V_row_tile[kk] = V_row[kk];
                }
            } else {
                // If we are beyond N, pad with zeros (no contribution)
                for (int kk = 0; kk < d; ++kk) {
                    K_row_tile[kk] = 0.0f;
                    V_row_tile[kk] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Determine how many valid rows are actually in this tile
        int tile_rows = min(BC, N - key_start);
        if (tile_rows <= 0) {
            __syncthreads();
            continue;
        }

        // 2. First pass over the tile: compute tile max m_ij
        float m_ij = neg_inf;
        for (int r = 0; r < tile_rows; ++r) {
            const float *Kj = &K_tile[r * MAX_D];
            float score = 0.0f;
            for (int kk = 0; kk < d; ++kk) {
                score += Qi[kk] * Kj[kk];
            }
            score *= inv_sqrt_d;
            if (score > m_ij) {
                m_ij = score;
            }
        }

        // 3. Merge tile max with running max
        float m_new = (m_i > m_ij) ? m_i : m_ij;

        // Temporary accumulator for this tile's contribution to O_i
        float O_tile[MAX_D];
        for (int kk = 0; kk < d; ++kk) {
            O_tile[kk] = 0.0f;
        }

        // 4. Second pass: compute exp(score - m_new) for each key in the tile,
        //    update l_i_new and accumulate the weighted V rows into O_tile.
        float l_new = 0.0f;
        float alpha = (m_i == neg_inf) ? 0.0f : expf(m_i - m_new); // factor for old state

        // Start by adding the scaled previous normalizer
        l_new = alpha * l_i;

        for (int r = 0; r < tile_rows; ++r) {
            const float *Kj = &K_tile[r * MAX_D];
            const float *Vj = &V_tile[r * MAX_D];

            float score = 0.0f;
            for (int kk = 0; kk < d; ++kk) {
                score += Qi[kk] * Kj[kk];
            }
            score *= inv_sqrt_d;

            // unnormalized softmax weight under the new maximum
            float p = expf(score - m_new);
            l_new += p;

            // accumulate p * V_j into O_tile
            for (int kk = 0; kk < d; ++kk) {
                O_tile[kk] += p * Vj[kk];
            }
        }

        // 5. Update O_i using the online softmax formula:
        //
        //    O_i_new = (alpha * l_i / l_new) * O_i + (1 / l_new) * O_tile
        float coeff_prev = 0.0f;
        if (l_new > 0.0f && l_i > 0.0f) {
            coeff_prev = (alpha * l_i) / l_new;
        }
        float coeff_tile = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;

        for (int kk = 0; kk < d; ++kk) {
            Oi[kk] = coeff_prev * Oi[kk] + coeff_tile * O_tile[kk];
        }

        // 6. Update the running state for this query row
        m_i = m_new;
        l_i = l_new;

        __syncthreads();
    }

    // At this point, O_i already contains the final result:
    //   softmax(Q_i K^T / sqrt(d)) V
}

// Host-side wrapper to launch the kernel
void flash_attn_forward_cuda(const float *d_Q,
                             const float *d_K,
                             const float *d_V,
                             float *d_O,
                             int N, int d) {
    dim3 block(BR);
    dim3 grid((N + BR - 1) / BR);

    flash_attn_forward_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, N, d);
    checkCuda(cudaGetLastError(), "flash_attn_forward_kernel launch");
}

// ---------------------------
// Simple test in main()
// ---------------------------

int main() {
    // You can change N and d to experiment with different sizes.
    // Make sure d <= MAX_D.
    const int N = 128;  // sequence length
    const int d = 64;   // head dimension

    assert(d <= MAX_D);

    size_t bytes = static_cast<size_t>(N) * d * sizeof(float);

    // Host buffers
    float *h_Q      = (float*)std::malloc(bytes);
    float *h_K      = (float*)std::malloc(bytes);
    float *h_V      = (float*)std::malloc(bytes);
    float *h_O_cpu  = (float*)std::malloc(bytes);
    float *h_O_gpu  = (float*)std::malloc(bytes);

    if (!h_Q || !h_K || !h_V || !h_O_cpu || !h_O_gpu) {
        std::fprintf(stderr, "Host malloc failed.\n");
        return EXIT_FAILURE;
    }

    // Initialize Q, K, V with random values in [-1, 1]
    std::srand(42);
    auto rand_float = []() {
        return ((float)std::rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    };

    for (int i = 0; i < N * d; ++i) {
        h_Q[i] = rand_float();
        h_K[i] = rand_float();
        h_V[i] = rand_float();
    }

    std::printf("Running CPU reference implementation...\n");
    attention_forward_cpu(h_Q, h_K, h_V, h_O_cpu, N, d);

    // Device buffers
    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_O = nullptr;

    checkCuda(cudaMalloc(&d_Q, bytes), "cudaMalloc d_Q");
    checkCuda(cudaMalloc(&d_K, bytes), "cudaMalloc d_K");
    checkCuda(cudaMalloc(&d_V, bytes), "cudaMalloc d_V");
    checkCuda(cudaMalloc(&d_O, bytes), "cudaMalloc d_O");

    checkCuda(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice), "Memcpy Q H2D");
    checkCuda(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice), "Memcpy K H2D");
    checkCuda(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice), "Memcpy V H2D");

    std::printf("Running CUDA FlashAttention kernel...\n");
    flash_attn_forward_cuda(d_Q, d_K, d_V, d_O, N, d);

    checkCuda(cudaMemcpy(h_O_gpu, d_O, bytes, cudaMemcpyDeviceToHost), "Memcpy O D2H");

    // Compare CPU and GPU results
    float max_diff = 0.0f;
    for (int i = 0; i < N * d; ++i) {
        float diff = std::fabs(h_O_cpu[i] - h_O_gpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    std::printf("Max abs diff between CPU and CUDA: %.6e\n", max_diff);
    if (max_diff < 1e-4f) {
        std::printf("✅ PASS: CUDA FlashAttention matches CPU reference within tolerance.\n");
    } else {
        std::printf("❌ MISMATCH: difference too large.\n");
    }

    // Clean up
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    std::free(h_Q);
    std::free(h_K);
    std::free(h_V);
    std::free(h_O_cpu);
    std::free(h_O_gpu);

    return 0;
}
