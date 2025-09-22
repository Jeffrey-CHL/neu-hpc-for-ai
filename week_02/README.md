# Week 02 - CUDA GEMM

This implements `D = alpha * (A @ B) + beta * C` in CUDA.
Two kernels:
- `gemm_naive`: naive per-element kernel
- `gemm_tiled`: shared-memory tiled kernel (TILE=16 or 32)

## Build

```bash
make                # default TILE=16, arch sm_80
# or
make TILE=32
# or (specify GPU architecture)
make ARCH="-arch=sm_90" TILE=32
