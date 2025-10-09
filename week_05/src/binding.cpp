#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "flash_attn2.cuh"

static void check_inputs(const at::Tensor& t) {
  TORCH_CHECK(t.is_cuda(), "Tensor must be CUDA");
  TORCH_CHECK(t.is_contiguous(), "Tensor must be contiguous in memory [B,H,N,D]");
}

at::Tensor fa2_forward(at::Tensor q, at::Tensor k, at::Tensor v, bool causal) {
  check_inputs(q); check_inputs(k); check_inputs(v);
  auto o = at::empty_like(q);
  auto sizes = q.sizes();
  auto m = at::empty({sizes[0], sizes[1], sizes[2]}, q.options().dtype(at::kFloat));
  auto l = at::empty({sizes[0], sizes[1], sizes[2]}, q.options().dtype(at::kFloat));
  flash_attn2_forward_launcher(q, k, v, o, m, l, causal);
  return o;
}

std::vector<at::Tensor> fa2_forward_with_stats(at::Tensor q, at::Tensor k, at::Tensor v, bool causal) {
  check_inputs(q); check_inputs(k); check_inputs(v);
  auto o = at::empty_like(q);
  auto sizes = q.sizes();
  auto m = at::empty({sizes[0], sizes[1], sizes[2]}, q.options().dtype(at::kFloat));
  auto l = at::empty({sizes[0], sizes[1], sizes[2]}, q.options().dtype(at::kFloat));
  flash_attn2_forward_launcher(q, k, v, o, m, l, causal);
  return {o, m, l};
}

std::vector<at::Tensor> fa2_backward(at::Tensor q, at::Tensor k, at::Tensor v,
                                     at::Tensor o, at::Tensor m, at::Tensor l,
                                     at::Tensor dout, bool causal) {
  check_inputs(q); check_inputs(k); check_inputs(v);
  check_inputs(o); check_inputs(dout);
  TORCH_CHECK(m.dtype()==at::kFloat && l.dtype()==at::kFloat, "m,l must be float32");

  // Accumulate grads in float32 for safe atomics, then cast back to q.dtype
  auto dq32 = at::zeros_like(q, q.options().dtype(at::kFloat));
  auto dk32 = at::zeros_like(k, k.options().dtype(at::kFloat));
  auto dv32 = at::zeros_like(v, v.options().dtype(at::kFloat));

  flash_attn2_backward_launcher(q,k,v,o,m,l,dout,dq32,dk32,dv32,causal);

  auto dq = dq32.to(q.dtype());
  auto dk = dk32.to(k.dtype());
  auto dv = dv32.to(v.dtype());
  return {dq, dk, dv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fa2_forward, "FlashAttention2 forward");
  m.def("forward_with_stats", &fa2_forward_with_stats, "FlashAttention2 forward (return stats)");
  m.def("backward", &fa2_backward, "FlashAttention2 backward");
}
