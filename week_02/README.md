# Week 02 - CUDA GEMM

## Description
This project implements **general matrix multiplication (GEMM)** in CUDA:

\[
D = \alpha (A \times B) + \beta C
\]

Two kernels are implemented:
- **Naive**: each thread computes one output element directly.
- **Tiled**: each thread block uses shared memory to improve memory locality.

## Build
```bash
make                # default TILE=16, arch=sm_80
make TILE=32        # try with 32x32 tiles
make ARCH="-arch=sm_90"

## Run
# gemm [M] [N] [K] [mode] [tile]
# mode: 0 = naive, 1 = tiled
./gemm                  # defaults to 1024 1024 1024, tiled
./gemm 512 512 512 0 16 # naive kernel
./gemm 1024 1024 1024 1 32 # tiled kernel with TILE=32

# if running on Modal:
modal run week_02/run_on_modal.py run_gemm --M 1024 --N 1024 --K 1024 --mode 1 --tile 32

## Results

| M=N=K | TILE | Kernel | Time (ms) | GFLOP/s | Rel Error |
|-------|------|--------|-----------|---------|-----------|
| 1024  | 16   | tiled  | 6.984     | 307.5   | 2.3e-07   |

*(Collected from Modal GPU run — shows throughput and correctness for the tiled kernel.)*

## Reflection

- Successfully compiled and executed the CUDA GEMM kernel on Modal with GPU support.  
- The **tiled kernel** achieved ~307 GFLOP/s on a 1024×1024 matrix with TILE=16, which demonstrates significant speedup compared to naive approaches.  
- The **relative L2 error** was ≈2.3e-07, confirming numerical correctness.  
- The main challenge was environment setup (`nvcc` not found, CUDA toolkit issues).  
  - Solved by installing CUDA 12.2 and adjusting `benchmark_modal.py` with `image.add_local_dir`.  
- Learned how to:
  - Launch CUDA kernels on cloud GPUs via Modal.  
  - Benchmark performance with different matrix sizes and tile configurations.  
  - Observe how tiling and shared memory improve memory locality and throughput.  
