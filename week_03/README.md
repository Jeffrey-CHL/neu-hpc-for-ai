# Week 03 - Extended CUDA GEMM

## Description
This project extends the Week 02 CUDA GEMM implementation.  
The kernel now supports:
- Optional transposition of input matrices **A** or **B**.
- In-place update of **C** (instead of writing to a separate matrix **D**).
- A tiled kernel implementation to reduce high-bandwidth memory (HBM) accesses, following the ideas from PMPP Chapter 5.

Formally, the operation is:

\[
C \leftarrow \alpha \cdot op_s(A) \cdot op_t(B) + \beta \cdot C
\]

where:
- \(op_s(A) = A\) if \(s=N\), or \(A^T\) if \(s=T\)  
- \(op_t(B) = B\) if \(t=N\), or \(B^T\) if \(t=T\)  

---

## Build & Run

This project is configured to run on **[Modal](https://modal.com/)** with an NVIDIA GPU.  
You don’t need a local CUDA setup on your laptop (e.g., Mac M1/M2).

### Run on Modal
From the `week_03/` directory:

```bash
# Run with Modal (automatically compiles and executes gemm.cu)
modal run run_gemm.py::run
```

The script will:
1. Build a CUDA container image with `gemm.cu`.
2. Compile the code with `nvcc`.
3. Run the GEMM kernel in both **naive** and **tiled** modes with different tile sizes.
4. Collect and print verification results.

---

## Results

Below are the verification results automatically collected from Modal runs:

| Mode   | TILE | Matrix Size (M=N=K) | Max Abs Error | RMS Error |
|--------|------|----------------------|---------------|-----------|
| naive | 16 | 1024 | 4.768e-06 | 6.159e-07 |
| naive | 32 | 1024 | 4.768e-06 | 6.159e-07 |
| tiled | 16 | 1024 | 4.768e-06 | 6.159e-07 |
| tiled | 32 | 1024 | 4.768e-06 | 6.159e-07 |

---

## Reflection

- Adding transpose and in-place updates required careful handling of matrix dimensions and indexing.  
- The tiled version significantly reduces global memory loads by exploiting shared memory, although performance tuning (e.g., larger sizes, profiling) is still possible.  
- Modal made it convenient to run CUDA on GPUs without needing a local NVIDIA setup.  

---
