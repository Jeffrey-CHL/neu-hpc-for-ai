// gemm.cu
// General Matrix Multiply in CUDA:
// D = alpha * (A @ B) + beta * C
// A: (M x K), B: (K x N), C/D: (M x N)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#ifndef TILE
#define TILE 16  // try 16 or 32 depending on GPU
#endif

// -------------------- CPU reference --------------------
void gemm_cpu(int M, int N, int K,
              const float* A, const float* B, const float* C,
              float* D, float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += A[i*K + k] * B[k*N + j];
            }
            D[i*N + j] = alpha * acc + beta * C[i*N + j];
        }
    }
}

// -------------------- CUDA: naive kernel --------------------
__global__ void gemm_naive(int M, int N, int K,
                           const float* __restrict__ A,
                           const float* __restrict__ B,
                           const float* __restrict__ C,
                           float* D, float alpha, float beta) {
    int row = blockDim.y * blockIdx.y + threadIdx.y; // i
    int col = blockDim.x * blockIdx.x + threadIdx.x; // j
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
            acc += A[row*K + k] * B[k*N + col];
        }
        D[row*N + col] = alpha * acc + beta * C[row*N + col];
    }
}

// -------------------- CUDA: tiled kernel with shared memory --------------------
__global__ void gemm_tiled(int M, int N, int K,
                           const float* __restrict__ A,
                           const float* __restrict__ B,
                           const float* __restrict__ C,
                           float* D, float alpha, float beta) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockDim.y * blockIdx.y + threadIdx.y; // i
    int col = blockDim.x * blockIdx.x + threadIdx.x; // j

    float acc = 0.f;
    // loop over tiles along K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x; // k
        int b_row = t * TILE + threadIdx.y; // k

        // load tiles with bounds check (coalesced)
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row*K + a_col] : 0.f;

        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row*N + col] : 0.f;

        __syncthreads();

        // compute partial dot for this tile
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        D[row*N + col] = alpha * acc + beta * C[row*N + col];
    }
}

// -------------------- utils --------------------
static inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(1);
    }
}
#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void init_random(float* x, int n, unsigned seed=123) {
    srand(seed);
    for (int i = 0; i < n; ++i) {
        x[i] = ((rand() % 2001) - 1000) / 1000.0f; // [-1, 1]
    }
}

float l2_rel_error(const float* a, const float* b, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = (double)a[i] - (double)b[i];
        num += diff * diff;
        den += (double)b[i] * (double)b[i];
    }
    return (float) sqrt((num + 1e-12) / (den + 1e-12));
}

// -------------------- main --------------------
int main(int argc, char** argv) {
    // Sizes via args: M N K [mode] [tile]
    // mode: 0=naive, 1=tiled (default 1)
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    int mode = (argc > 4) ? atoi(argv[4]) : 1;
    int tile_from_arg = (argc > 5) ? atoi(argv[5]) : TILE; // informational

    float alpha = 1.0f, beta = 1.0f;
    printf("GEMM: D = alpha*A@B + beta*C (alpha=%.1f, beta=%.1f)\n", alpha, beta);
    printf("Dims: A(%d x %d), B(%d x %d), C/D(%d x %d)\n", M,K,K,N,M,N);
    printf("Kernel: %s, TILE=%d\n", mode==0?"naive":"tiled", tile_from_arg);

    size_t szA = (size_t)M*K, szB = (size_t)K*N, szC = (size_t)M*N;

    // Host alloc
    float *hA = (float*)malloc(szA*sizeof(float));
    float *hB = (float*)malloc(szB*sizeof(float));
    float *hC = (float*)malloc(szC*sizeof(float));
    float *hD_ref = (float*)malloc(szC*sizeof(float));
    float *hD = (float*)malloc(szC*sizeof(float));

    init_random(hA, szA, 1);
    init_random(hB, szB, 2);
    init_random(hC, szC, 3);

    // CPU reference
    printf("Running CPU reference...\n");
    gemm_cpu(M,N,K,hA,hB,hC,hD_ref,alpha,beta);

    // Device alloc
    float *dA,*dB,*dC,*dD;
    gpuCheck(cudaMalloc(&dA, szA*sizeof(float)));
    gpuCheck(cudaMalloc(&dB, szB*sizeof(float)));
    gpuCheck(cudaMalloc(&dC, szC*sizeof(float)));
    gpuCheck(cudaMalloc(&dD, szC*sizeof(float)));

    gpuCheck(cudaMemcpy(dA, hA, szA*sizeof(float), cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(dB, hB, szB*sizeof(float), cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(dC, hC, szC*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + block.x - 1)/block.x,
              (M + block.y - 1)/block.y);

    // Warmup
    if (mode == 0) {
        gemm_naive<<<grid, block>>>(M,N,K,dA,dB,dC,dD,alpha,beta);
    } else {
        gemm_tiled<<<grid, block>>>(M,N,K,dA,dB,dC,dD,alpha,beta);
    }
    gpuCheck(cudaDeviceSynchronize());

    // Timing
    const int iters = 10;
    cudaEvent_t start, stop;
    gpuCheck(cudaEventCreate(&start));
    gpuCheck(cudaEventCreate(&stop));
    gpuCheck(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        if (mode == 0) {
            gemm_naive<<<grid, block>>>(M,N,K,dA,dB,dC,dD,alpha,beta);
        } else {
            gemm_tiled<<<grid, block>>>(M,N,K,dA,dB,dC,dD,alpha,beta);
        }
    }
    gpuCheck(cudaEventRecord(stop));
    gpuCheck(cudaEventSynchronize(stop));
    float ms = 0.f;
    gpuCheck(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters; // avg per iteration

    gpuCheck(cudaMemcpy(hD, dD, szC*sizeof(float), cudaMemcpyDeviceToHost));

    // Check correctness
    float rel = l2_rel_error(hD, hD_ref, (int)szC);
    printf("Relative L2 error vs CPU: %.3e\n", rel);
    if (rel > 1e-3f) {
        printf("WARNING: relative error > 1e-3\n");
    }

    // Report throughput
    // FLOPs for GEMM ~ 2*M*N*K (mul+add) + M*N (axpby), we report dominant term
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = flops / (ms/1000.0) / 1e9;
    printf("Time: %.3f ms (avg over %d iters)\n", ms, iters);
    printf("Throughput: %.2f GFLOP/s\n", gflops);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
    free(hA); free(hB); free(hC); free(hD); free(hD_ref);
    return 0;
}
