// =============================
// File: gemm.cu
// Week03: GEMM with transpose options, in-place C update, and tiled kernel
// =============================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cassert>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    exit(1); \
  } \
} while(0)
#endif

// Transpose flags
enum Transpose : int { N = 0, T = 1 };

// Device-side access helpers honoring transpose
__device__ __forceinline__ float getA(const float* A, int lda, int row, int col, Transpose ta) {
  return (ta == N) ? A[row * lda + col] : A[col * lda + row];
}

__device__ __forceinline__ float getB(const float* B, int ldb, int row, int col, Transpose tb) {
  return (tb == N) ? B[row * ldb + col] : B[col * ldb + row];
}

// =============================
// Naive kernel
// =============================
__global__ void gemm_naive_kernel(
    int M, int Nn, int K,
    const float* __restrict__ A, int lda, Transpose ta,
    const float* __restrict__ B, int ldb, Transpose tb,
    float* __restrict__ C, int ldc,
    float alpha, float beta)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= Nn) return;

  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    float a = getA(A, lda, row, k, ta);
    float b = getB(B, ldb, k, col, tb);
    acc += a * b;
  }

  float c_old = C[row * ldc + col];
  C[row * ldc + col] = alpha * acc + beta * c_old;
}

// =============================
// Tiled kernel
// =============================
__global__ void gemm_tiled_kernel(
    int M, int Nn, int K,
    const float* __restrict__ A, int lda, Transpose ta,
    const float* __restrict__ B, int ldb, Transpose tb,
    float* __restrict__ C, int ldc,
    float alpha, float beta,
    int TILE)
{
  extern __shared__ float smem[];
  float* As = smem;
  float* Bs = smem + TILE * TILE;

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.f;

  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int k_base = t * TILE;

    int a_row = row;
    int a_col = k_base + threadIdx.x;
    float a_val = 0.f;
    if (a_row < M && a_col < K) {
      a_val = getA(A, lda, a_row, a_col, ta);
    }
    As[threadIdx.y * TILE + threadIdx.x] = a_val;

    int b_row = k_base + threadIdx.y;
    int b_col = col;
    float b_val = 0.f;
    if (b_row < K && b_col < Nn) {
      b_val = getB(B, ldb, b_row, b_col, tb);
    }
    Bs[threadIdx.y * TILE + threadIdx.x] = b_val;

    __syncthreads();

    int kk_max = min(TILE, K - k_base);
    for (int kk = 0; kk < kk_max; ++kk) {
      acc += As[threadIdx.y * TILE + kk] * Bs[kk * TILE + threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < Nn) {
    float c_old = C[row * ldc + col];
    C[row * ldc + col] = alpha * acc + beta * c_old;
  }
}

// =============================
// CPU reference
// =============================
void gemm_cpu(int M, int Nn, int K,
              const std::vector<float>& A, int lda, Transpose ta,
              const std::vector<float>& B, int ldb, Transpose tb,
              std::vector<float>& C, int ldc,
              float alpha, float beta)
{
  std::vector<float> out(M * ldc);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < Nn; ++j) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) {
        float a = (ta == N) ? A[i * lda + k] : A[k * lda + i];
        float b = (tb == N) ? B[k * ldb + j] : B[j * ldb + k];
        acc += a * b;
      }
      out[i * ldc + j] = alpha * acc + beta * C[i * ldc + j];
    }
  }
  C.swap(out);
}

static inline float frand() { return float(rand()) / float(RAND_MAX) - 0.5f; }

// =============================
// Argument parsing and main()
// =============================
struct Args {
  int M = 1024, Nn = 1024, K = 1024;
  std::string mode = "tiled";
  Transpose ta = N;
  Transpose tb = N;
  float alpha = 1.f;
  float beta  = 0.f;
  int TILE = 32;
  bool verify = true;
};

Transpose parse_t(const char* s){ return (s[0]=='T'||s[0]=='t')? T : N; }

Args parse(int argc, char** argv){
  Args a;
  if (argc >= 4) { a.M = atoi(argv[1]); a.Nn = atoi(argv[2]); a.K = atoi(argv[3]); }
  for (int i=4;i<argc;i++){
    std::string k = argv[i];
    if (k=="--mode") { a.mode = argv[++i]; }
    else if (k=="--ta") { a.ta = parse_t(argv[++i]); }
    else if (k=="--tb") { a.tb = parse_t(argv[++i]); }
    else if (k=="--alpha") { a.alpha = atof(argv[++i]); }
    else if (k=="--beta") { a.beta = atof(argv[++i]); }
    else if (k=="--tile") { a.TILE = atoi(argv[++i]); }
    else if (k=="--no-verify") { a.verify = false; }
  }
  return a;
}

int main(int argc, char** argv){
  Args args = parse(argc, argv);
  int M=args.M, Nn=args.Nn, K=args.K;

  printf("GEMM Week03: M=%d N=%d K=%d mode=%s ta=%c tb=%c alpha=%.3f beta=%.3f tile=%d verify=%d\n",
         M, Nn, K, args.mode.c_str(), args.ta==N?'N':'T', args.tb==N?'N':'T',
         args.alpha, args.beta, args.TILE, args.verify);

  int lda = (args.ta==N)? K : M;
  int ldb = (args.tb==N)? Nn: K;
  int ldc = Nn;

  std::vector<float> hA((args.ta==N? M:K) * lda);
  std::vector<float> hB((args.tb==N? K:Nn)* ldb);
  std::vector<float> hC(M * ldc);
  std::vector<float> hC_ref(M * ldc);

  srand(0xC0FFEE);
  for (auto& x: hA) x = frand();
  for (auto& x: hB) x = frand();
  for (auto& x: hC) x = frand();
  hC_ref = hC;

  float *dA,*dB,*dC;
  CHECK_CUDA(cudaMalloc(&dA, hA.size()*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dB, hB.size()*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dC, hC.size()*sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size()*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size()*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, hC.data(), hC.size()*sizeof(float), cudaMemcpyHostToDevice));

  dim3 block((args.mode=="naive")?16:args.TILE, (args.mode=="naive")?16:args.TILE);
  dim3 grid((Nn + block.x -1)/block.x, (M + block.y -1)/block.y);

  if (args.mode=="naive") {
    gemm_naive_kernel<<<grid, block>>>(M,Nn,K, dA,lda,args.ta, dB,ldb,args.tb, dC,ldc, args.alpha,args.beta);
  } else {
    size_t shmem = 2 * args.TILE * args.TILE * sizeof(float);
    gemm_tiled_kernel<<<grid, dim3(args.TILE,args.TILE), shmem>>>(
      M,Nn,K, dA,lda,args.ta, dB,ldb,args.tb, dC,ldc, args.alpha,args.beta, args.TILE);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(hC.data(), dC, hC.size()*sizeof(float), cudaMemcpyDeviceToHost));

  if (args.verify) {
    gemm_cpu(M,Nn,K, hA,lda,args.ta, hB,ldb,args.tb, hC_ref,ldc, args.alpha,args.beta);
    double max_err = 0.0, rms = 0.0;
    for (size_t i=0;i<hC.size();++i){
      double e = double(hC[i]) - double(hC_ref[i]);
      max_err = fmax(max_err, fabs(e));
      rms += e*e;
    }
    rms = sqrt(rms / hC.size());
    printf("Verify: max_abs_err=%.3e  rms=%.3e\n", max_err, rms);
  }

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}