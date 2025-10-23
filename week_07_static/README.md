# week_07_static

This package contains a **self-contained** Modal app (`main.py`) that writes
the CUDA project files **inside** the container at runtime (no local directory
mounting). This avoids the "was modified during build process" errors.

## How to run

```bash
cd week_07_static
modal run main.py::auto_run
```
# Week 07 — FlashAttention CUDA + MPI Demo

- Environment: CUDA 12.4.1 + OpenMPI 4.1.2 + Modal GPU (A100 40GB)
- Program: Minimal FlashAttention kernel simulation with matrix ops and softmax.
- Result: OK - checksum: 15.206454
- All 4 MPI workers produced consistent results.