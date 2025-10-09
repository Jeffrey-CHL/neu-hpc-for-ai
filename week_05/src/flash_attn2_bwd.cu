#include <cuda.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "flash_attn2.cuh"

static constexpr int Br = 64;
static constexpr int Bc = 64;
static constexpr int TPB = 128;
static constexpr int DMAX = 256;

__device__ inline float to_float(const half x) { return __half2float(x); }
__device__ inline float to_float(const nv_bfloat16 x) { return __bfloat162float(x); }

// Backward: recompute P per tile, accumulate into float32 grads
template<typename T>
__global__ void fa2_bwd_kernel(const T* __restrict__ Q,
                               const T* __restrict__ K,
                               const T* __restrict__ V,
                               const T* __restrict__ O,
                               const float* __restrict__ M,
                               const float* __restrict__ L,
                               const T* __restrict__ dO,
                               float* __restrict__ dQ32,
                               float* __restrict__ dK32,
                               float* __restrict__ dV32,
                               int B, int H, int N, int D,
                               float scale, bool causal) {
  int bh = blockIdx.z;
  int qb = blockIdx.y;
  int b = bh / H; int h = bh % H;
  int q_row0 = qb * Br; if (q_row0 >= N) return;
  auto idxBH = [=] __device__ (int n, int d){ return (((b*H + h)*N + n)*D + d); };

  __shared__ float sK[Bc][DMAX];
  __shared__ float sV[Bc][DMAX];

  // zero dQ rows owned by this block (float32 buffer)
  for (int r=threadIdx.x; r<Br; r+=blockDim.x) {
    int n = q_row0 + r; if (n>=N) continue;
    for (int d=0; d<D; ++d) dQ32[idxBH(n,d)] = 0.f;
  }
  __syncthreads();

  for (int kc=0; kc<N; kc+=Bc) {
    int cols = min(Bc, N-kc);
    // stage K,V tile
    for (int c = threadIdx.x; c < cols*D; c += blockDim.x) {
      int jj = c / D; int dd = c % D;
      int n = kc + jj;
      sK[jj][dd] = to_float(K[idxBH(n,dd)]);
      sV[jj][dd] = to_float(V[idxBH(n,dd)]);
    }
    __syncthreads();

    for (int r=threadIdx.x; r<Br; r+=blockDim.x) {
      int n = q_row0 + r; if (n>=N) continue;
      if (causal && n < kc) continue;
      float m = M[(b*H + h)*N + n];
      float l = L[(b*H + h)*N + n];
      float inv_l = 1.f / max(l, 1e-20f);

      // First pass: compute alpha and dV
      float alpha = 0.f;
      for (int jj=0; jj<cols; ++jj) {
        if (causal && (kc+jj) > n) break;
        float acc_s = 0.f;
        for (int d=0; d<D; ++d) acc_s += to_float(Q[idxBH(n,d)]) * sK[jj][d];
        float s = acc_s * scale;
        float P = expf(s - m) * inv_l;

        float dOV = 0.f;
        for (int d=0; d<D; ++d) dOV += to_float(dO[idxBH(n,d)]) * sV[jj][d];
        alpha += P * dOV;

        // dV_j += P * dO_row
        for (int d=0; d<D; ++d) {
          float inc = P * to_float(dO[idxBH(n,d)]);
          atomicAdd(&dV32[idxBH(kc+jj,d)], inc);
        }
      }

      // Second pass: dP and accumulate dQ, dK
      for (int jj=0; jj<cols; ++jj) {
        if (causal && (kc+jj) > n) break;
        float acc_s = 0.f;
        for (int d=0; d<D; ++d) acc_s += to_float(Q[idxBH(n,d)]) * sK[jj][d];
        float s = acc_s * scale;
        float P = expf(s - m) * inv_l;

        float dOV = 0.f;
        for (int d=0; d<D; ++d) dOV += to_float(dO[idxBH(n,d)]) * sV[jj][d];
        float dP = P * (dOV - alpha);

        // dQ_row += dP * K_j * scale
        for (int d=0; d<D; ++d) {
          float inc = dP * sK[jj][d] * scale;
          dQ32[idxBH(n,d)] += inc;
        }
        // dK_j += dP * Q_row * scale
        for (int d=0; d<D; ++d) {
          float inc = dP * to_float(Q[idxBH(n,d)]) * scale;
          atomicAdd(&dK32[idxBH(kc+jj,d)], inc);
        }
      }
    }
    __syncthreads();
  }
}

template<typename T>
void fa2_bwd_dispatch(at::Tensor q, at::Tensor k, at::Tensor v,
                      at::Tensor o, at::Tensor m, at::Tensor l,
                      at::Tensor dout, at::Tensor dq32,
                      at::Tensor dk32, at::Tensor dv32, bool causal) {
  const int B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);
  TORCH_CHECK(D <= 256, "This reference kernel supports D <= 256");
  const float scale = 1.f / sqrtf(float(D));

  dim3 grid;
  grid.y = (N + Br - 1) / Br;
  grid.z = B * H; grid.x = 1;
  dim3 block(TPB);

  AT_CUDA_CHECK(cudaMemset(dq32.data_ptr(), 0, dq32.numel() * dq32.element_size()));
  AT_CUDA_CHECK(cudaMemset(dk32.data_ptr(), 0, dk32.numel() * dk32.element_size()));
  AT_CUDA_CHECK(cudaMemset(dv32.data_ptr(), 0, dv32.numel() * dv32.element_size()));

  fa2_bwd_kernel<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
    (const T*)q.data_ptr(), (const T*)k.data_ptr(), (const T*)v.data_ptr(),
    (const T*)o.data_ptr(), m.data_ptr<float>(), l.data_ptr<float>(),
    (const T*)dout.data_ptr(),
    (float*)dq32.data_ptr(), (float*)dk32.data_ptr(), (float*)dv32.data_ptr(),
    B,H,N,D, scale, causal);
}

void flash_attn2_backward_launcher(at::Tensor q, at::Tensor k, at::Tensor v,
                                   at::Tensor o, at::Tensor m, at::Tensor l,
                                   at::Tensor dout,
                                   at::Tensor dq32, at::Tensor dk32, at::Tensor dv32,
                                   bool causal) {
  if (q.scalar_type() == at::kHalf) {
    fa2_bwd_dispatch<half>(q,k,v,o,m,l,dout,dq32,dk32,dv32,causal);
  } else if (q.scalar_type() == at::kBFloat16) {
    fa2_bwd_dispatch<nv_bfloat16>(q,k,v,o,m,l,dout,dq32,dk32,dv32,causal); // NOTE: order fix below
  } else {
    TORCH_CHECK(false, "Only FP16/BF16 supported for q,k,v");
  }
}
