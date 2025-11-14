# Week 07 – Distributed FlashAttention on Modal

This project implements a **distributed FlashAttention** kernel using **CUDA** and **MPI**, and runs it on a GPU instance provisioned via [Modal](https://modal.com).

The code is based on the course starter files for INFO 7375 HPC for AI.

---

## 1. Project Structure

Inside this `week_07_static` directory:

- `main.py`  
  Modal entrypoint.  
  - Builds a Docker image with CUDA, OpenMPI and the project files.  
  - Compiles the CUDA code with `nvcc`.  
  - Launches the distributed FlashAttention executable with `mpirun`.

- `src/flash_attn.cu`  
  CUDA + MPI implementation of the distributed FlashAttention kernel.  
  - Each MPI **rank**:
    - Selects a GPU device.
    - Computes attention on its local chunk of the batch.
    - Prints its **local checksum** and **kernel time**.
  - Rank 0 gathers local checksums and kernel times and prints:
    - **Global checksum across all GPUs**
    - **Max kernel time among all ranks**
    - **World size** (number of ranks / GPUs)

- `Makefile`  
  Builds the CUDA executable `bin/flash_attn` using `nvcc` and links it with MPI.

- `Dockerfile`  
  Base container image for Modal with CUDA toolchain, `nvcc`, OpenMPI, etc.

- `README.md`  
  (This file.) High‑level description and run instructions.

---

## 2. Requirements

You need:

- A working **Python environment** with the `modal` CLI and Python SDK configured  
  (including `modal setup` already done with a valid token).
- Access to a **GPU-capable** Modal workspace (e.g., course-provided account).
- This directory as your current working directory:
  ```bash
  cd /path/to/week_07_static
  ```

---

## 3. How to Run the Project

> The commands below are exactly what was used to produce the working run.

### 3.1 Deploy the app (optional but recommended)

This uploads and builds the Modal app, and prepares the image:

```bash
modal deploy main.py
```

> Note: The code still uses `modal.Stub`, which Modal has deprecated in favor of `modal.App`.  
> You may see a *deprecation warning*, but this does **not** affect correctness for this assignment.

### 3.2 Run the distributed FlashAttention

To actually build and run the CUDA + MPI code on a GPU via Modal:

```bash
modal run main.py::run
```

What `run()` does:

1. Sets `WORKDIR` to `/root/project` (the mounted project directory).
2. Calls `make` / `nvcc` to build:
   ```bash
   nvcc -O3 -arch=sm_80 -I/usr/include/x86_64-linux-gnu/mpi         -o bin/flash_attn src/flash_attn.cu         -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
   ```
3. Launches the distributed job using 4 MPI ranks:
   ```bash
   mpirun -np 4 ./bin/flash_attn
   ```

---

## 4. Example Output

A successful run on Modal looks like this (trimmed for brevity):

```text
==========
== CUDA ==
==========

CUDA Version 12.4.1

== base_dir == /root/project
== Files in project == ['main.py', 'README.md', 'src', 'Makefile', '.DS_Store', '__pycache__', 'Dockerfile']
🔧 Building...
mkdir -p bin
nvcc -O3 -arch=sm_80 -I/usr/include/x86_64-linux-gnu/mpi -o bin/flash_attn src/flash_attn.cu -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
🚀 Running distributed FlashAttention...
World size = 4, CUDA devices on node = 1
[Rank 0] Using device 0
[Rank 1] Using device 0
[Rank 2] Using device 0
[Rank 3] Using device 0
Global config: B = 16, T = 256, D = 64
[Rank 0] local_B = 4, start_B = 0
[Rank 1] local_B = 4, start_B = 4
[Rank 2] local_B = 4, start_B = 8
[Rank 3] local_B = 4, start_B = 12
[Rank 0] local_checksum = 483.073074, kernel_time = 8.994 ms
[Rank 1] local_checksum = 679.724023, kernel_time = 11.112 ms
[Rank 2] local_checksum = 876.374962, kernel_time = 4.758 ms
[Rank 3] local_checksum = 1073.025907, kernel_time = 6.878 ms
=====================================================
Global checksum across all GPUs : 3112.197967
Max kernel time among all ranks : 11.112 ms
World size (num ranks / GPUs)   : 4
=====================================================
```

If you see a similar set of lines:

- All **4 ranks** print their `local_checksum` and `kernel_time`.
- Rank 0 prints the **global checksum** and **max kernel time**.

…then the distributed FlashAttention implementation is running correctly.

---

## 5. Notes on Warnings

You may see some extra warnings in the logs, for example:

- **Modal deprecation warnings** (about `Stub` or `Mount`)  
  These are related to new versions of the Modal API, not to your CUDA/MPI logic.

- **OpenMPI “btl_vader_single_copy_mechanism CMA” warning**  
  Example:
  ```text
  WARNING: The default btl_vader_single_copy_mechanism CMA is
  not available due to different user namespaces.
  ...
  This may result in lower performance.
  ```
  This only indicates that one shared‑memory optimization is not available inside the container.  
  It may slightly affect performance but **does not affect correctness** and is acceptable for this assignment.

---

## 6. Summary

- `main.py` uses Modal to:
  - Build a CUDA + MPI container image.
  - Compile `src/flash_attn.cu` into `bin/flash_attn`.
  - Launch a 4‑rank distributed FlashAttention run with `mpirun`.
- The program prints local and global checksums as a correctness check and reports the max kernel time across ranks.

To reproduce the results:

```bash
cd week_07_static
modal deploy main.py
modal run main.py::run
```

This is all you need to run and submit the assignment.
