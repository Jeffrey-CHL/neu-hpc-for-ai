#include <cuda.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "flash_attn2.cuh"

// Convert helpers
__device__ inline float to_float(const half x) { return __half2float(x); }
__device__ inline float to_float(const nv_bfloat16 x) { return __bfloat162float(x); }
__device__ inline half from_float_h(float x) { return __float2half(x); }
__device__ inline nv_bfloat16 from_float_bf(float x) { return __float2bfloat16(x); }

// Kernel parameters (kept small/clear for the assignment scale)
static constexpr int Br = 64;   // query rows per block
static constexpr int Bc = 64;   // KV cols per tile
static constexpr int TPB = 128; // threads per block
static constexpr int DMAX = 256; // max D supported by this simple smem layout

template<typename T>
__global__ void fa2_fwd_kernel(const T* __restrict__ Q,
                               const T* __restrict__ K,
                               const T* __restrict__ V,
                               T* __restrict__ O,
                               float* __restrict__ M,
                               float* __restrict__ L,
                               int B, int H, int N, int D,
                               float scale, bool causal) {
  int bh = blockIdx.z;              // b*h index
  int qb = blockIdx.y;              // which Q tile
  int b = bh / H; int h = bh % H;
  int q_row0 = qb * Br;
  if (q_row0 >= N) return;

  // index helper for packed [B,H,N,D]
  auto idxBH = [=] __device__ (int n, int d){ return (((b*H + h)*N + n)*D + d); };

  // shared staging for current KV tile (simple layout)
  __shared__ float sK[Bc][DMAX];
  __shared__ float sV[Bc][DMAX];

  // init per-row stats and output
  for (int r = threadIdx.x; r < Br; r += blockDim.x) {
    int n = q_row0 + r; if (n >= N) continue;
    int mIdx = (b*H + h)*N + n;
    M[mIdx] = -INFINITY;
    L[mIdx] = 0.0f;
    // zero O row (will be rescaled and accumulated before final division by l)
    for (int d=0; d<D; ++d) O[idxBH(n,d)] = (T)0;
  }
  __syncthreads();

  for (int kc = 0; kc < N; kc += Bc) {
    int cols = min(Bc, N - kc);
    // load K,V tile into shared memory
    for (int c = threadIdx.x; c < cols*D; c += blockDim.x) {
      int jj = c / D; int dd = c % D;
      int n = kc + jj;
      sK[jj][dd] = to_float(K[idxBH(n,dd)]);
      sV[jj][dd] = to_float(V[idxBH(n,dd)]);
    }
    __syncthreads();

    // process each query row owned by this block
    for (int r = threadIdx.x; r < Br; r += blockDim.x) {
      int n = q_row0 + r; if (n >= N) continue;
      if (causal && n < kc) continue; // whole tile is masked (above diagonal)
      int mIdx = (b*H + h)*N + n;
      float m_prev = M[mIdx];
      float l_prev = L[mIdx];

      // 1) find row_max over this tile
      float row_max = -INFINITY;
      for (int jj=0; jj<cols; ++jj) {
        if (causal && (kc+jj) > n) break; // masked upper triangle
        float acc = 0.f;
        for (int d=0; d<D; ++d) acc += to_float(Q[idxBH(n,d)]) * sK[jj][d];
        float s = acc * scale;
        row_max = fmaxf(row_max, s);
      }
      float m_new = fmaxf(m_prev, row_max);

      // 2) rescale previous O by exp(m_prev - m_new) and update l
      float rescale = expf(m_prev - m_new);
      float l_new = l_prev * rescale;
      for (int d=0; d<D; ++d) {
        float o_prev = to_float(O[idxBH(n,d)]) * rescale;
        O[idxBH(n,d)] = (T)o_prev;
      }

      // 3) accumulate this tile
      for (int jj=0; jj<cols; ++jj) {
        if (causal && (kc+jj) > n) break;
        float acc = 0.f;
        for (int d=0; d<D; ++d) acc += to_float(Q[idxBH(n,d)]) * sK[jj][d];
        float s = acc * scale;
        float p = expf(s - m_new);
        l_new += p;
        // O += p * V_j
        for (int d=0; d<D; ++d) {
          float o_cur = to_float(O[idxBH(n,d)]) + p * sV[jj][d];
          O[idxBH(n,d)] = (T)o_cur;
        }
      }
      M[mIdx] = m_new;
      L[mIdx] = l_new;
    }
    __syncthreads();
  }

  // 4) finalize: divide O_row by l
  for (int r = threadIdx.x; r < Br; r += blockDim.x) {
    int n = q_row0 + r; if (n >= N) continue;
    float l = L[(b*H + h)*N + n];
    float inv_l = 1.f / max(l, 1e-20f);
    for (int d=0; d<D; ++d) {
      float o_cur = to_float(O[idxBH(n,d)]) * inv_l;
      O[idxBH(n,d)] = (T)o_cur;
    }
  }
}

template<typename T>
void fa2_fwd_dispatch(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
                      at::Tensor m, at::Tensor l, bool causal) {
  const int B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);
  TORCH_CHECK(D <= 256, "This reference kernel supports D <= 256");
  const float scale = 1.f / sqrtf(float(D));

  dim3 grid;
  grid.x = 1;
  grid.y = (N + Br - 1) / Br;
  grid.z = B * H;
  dim3 block(TPB);

  // zero O for safety (we also write all elements)
  AT_CUDA_CHECK(cudaMemset(o.data_ptr(), 0, o.numel() * o.element_size()));
  fa2_fwd_kernel<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
    (const T*)q.data_ptr(), (const T*)k.data_ptr(), (const T*)v.data_ptr(),
    (T*)o.data_ptr(), m.data_ptr<float>(), l.data_ptr<float>(),
    B,H,N,D, scale, causal);
}

void flash_attn2_forward_launcher(at::Tensor q, at::Tensor k, at::Tensor v,
                                  at::Tensor o, at::Tensor m, at::Tensor l,
                                  bool causal) {
  if (q.scalar_type() == at::kHalf) {
    fa2_fwd_dispatch<half>(q,k,v,o,m,l,causal);
  } else if (q.scalar_type() == at::kBFloat16) {
    fa2_fwd_dispatch<nv_bfloat16>(q,k,v,o,m,l,causal);
  } else {
    TORCH_CHECK(false, "Only FP16/BF16 supported for q,k,v");
  }
}
