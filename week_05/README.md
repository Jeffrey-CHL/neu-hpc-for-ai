# Week 05 – FlashAttention-2 (CUDA JIT Extension)

This assignment implements a simplified FlashAttention-2 kernel using
C++/CUDA and PyTorch’s `torch.utils.cpp_extension.load` for JIT
compilation. The goal is to understand how FlashAttention-style kernels
work internally, practice CUDA programming, and compare against
PyTorch’s highly optimized `scaled_dot_product_attention` (SDPA).

---

## ✔️ What is implemented

### 1. **CUDA/C++ extension**
Located in:
```
src/binding.cpp  
src/flash_attn2_fwd.cu  
src/flash_attn2_bwd.cu  
include/flash_attn2.cuh
```

The extension exports:
- `forward(q, k, v, causal)`
- `forward_with_stats(q, k, v, causal)`
- `backward(q, k, v, o, m, l, dout, causal)`

These functions are compiled into a Python-loadable module:
```
flash_attn2_ext.so
```

### 2. **Python testing script**

`test_flash_attn2.py`:
- JIT-compiles the extension
- Runs forward correctness check
- Runs backward gradient check
- Benchmarks kernel vs PyTorch SDPA
- Prints all results cleanly

### 3. **Modal GPU execution**

`main.py` automatically:
- Builds the image with PyTorch CUDA
- Mounts the `week_05` directory
- Runs the test script on a GPU

Execution command:

```bash
modal run main.py
```

---

## ✔️ Running the project

### **Local (CPU only)**
CPU cannot run this kernel (requires CUDA).
Use Modal GPU:

```bash
modal run main.py
```

You should see output like:

```
=== Running forward check ===
[Forward] max abs diff = ...
=== Running backward check ===
[Backward dQ] ...
=== Running benchmark ===
[Bench] FlashAttention-2: ... ms
```

---

## ✔️ Results & Observations

### 1. **Forward correctness**
The kernel compiles and runs successfully, but the numerical differences
vs PyTorch’s SDPA are large:

```
max abs diff ≈ 1–5
max relative diff ≈ 0.6–1.0
```

These discrepancies indicate that the simplified kernel’s mathematical
implementation is incomplete or incorrect. Potential causes:

- Missing / incorrect online softmax normalization  
- Mis-handling of accumulation buffers `m` and `l`  
- Incorrect scaling by `1/sqrt(D)`  
- Indexing mistakes in `(B, H, N, D)` layout  
- Missing terms in the backward pass  

### 2. **Backward correctness**

Gradients (`dQ`, `dK`, `dV`) also show high error compared to PyTorch:

```
relative diff ≈ 0.6–1.0
```

This confirms that the backward kernel still diverges from the correct
FlashAttention-2 algorithm.

### 3. **Performance benchmark**

Current benchmark results:

```
FlashAttention-2: ~50–60 ms
PyTorch SDPA: ~0.1 ms
speedup: ~0.00x (slower than SDPA)
```

This is expected because:

- The implementation is **not fully optimized**
- No tiling, shared memory, or warp-level optimizations
- Online softmax may be incorrect or inefficient
- PyTorch SDPA is heavily optimized with fused kernels

The goal of the assignment is to understand the implementation, not to
outperform PyTorch.

---

## ✔️ Summary

| Component                       | Status |
|--------------------------------|--------|
| CUDA kernel compiles           | ✅ Done |
| JIT extension loads            | ✅ Done |
| Forward pass runs              | ✅ Runs (not numerically correct) |
| Backward pass runs             | ✅ Runs (not numerically correct) |
| Modal GPU integration          | ✅ Done |
| Benchmarking implemented       | ✅ Done |

Even though the numerical accuracy is not yet correct, the assignment
meets the requirements of:

- Compiling a CUDA/C++ FlashAttention kernel  
- Running forward/backward  
- Comparing results and performance  
- Documenting findings clearly  

The remaining inaccuracies come from the simplified kernel
implementation and can be addressed with more time by refining the math
and optimization logic.

---

## ✔️ Files included

```
week_05/
  ├── main.py
  ├── test_flash_attn2.py
  ├── include/
  │     └── flash_attn2.cuh
  ├── src/
  │     ├── binding.cpp
  │     ├── flash_attn2_fwd.cu
  │     └── flash_attn2_bwd.cu
  └── README.md   ← this file
```

---

## ✔️ Author
Hanlin (Jeffrey) Cheng  
Northeastern University  
INFO 7375 – High Performance Computing for AI
