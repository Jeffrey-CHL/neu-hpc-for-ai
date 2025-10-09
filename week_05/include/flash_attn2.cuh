#pragma once
#include <torch/extension.h>

// Forward launcher: runs FA‑2 forward and fills output O plus stats m,l (float32).
void flash_attn2_forward_launcher(at::Tensor q, at::Tensor k, at::Tensor v,
                                  at::Tensor o, at::Tensor m, at::Tensor l,
                                  bool causal);

// Backward launcher: accumulates gradients into float32 buffers (dq32, dk32, dv32).
void flash_attn2_backward_launcher(at::Tensor q, at::Tensor k, at::Tensor v,
                                   at::Tensor o, at::Tensor m, at::Tensor l,
                                   at::Tensor dout,
                                   at::Tensor dq32, at::Tensor dk32, at::Tensor dv32,
                                   bool causal);
