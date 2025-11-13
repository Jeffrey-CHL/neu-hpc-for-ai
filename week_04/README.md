# Week 4 – FlashAttention in CUDA

This assignment implements the **forward pass of scaled dot-product attention**
using a **FlashAttention-style tiled CUDA kernel** with **online softmax**.

We also include a naive CPU implementation to validate correctness.

---

## 1. Problem setup

We implement single-head attention for simplicity:

- Input shapes (row-major):
  - `Q`: (N, d)
  - `K`: (N, d)
  - `V`: (N, d)
- Output:
  - `O`: (N, d)

The operation is:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
\]

Where `softmax` is applied row-wise over the scores `QK^T / sqrt(d)`.

---

## 2. Files

- `flash_attention.cu`
  - `attention_forward_cpu` – naive O(N²) CPU implementation
  - `flash_attn_forward_kernel` – CUDA kernel using tiling and shared memory
  - `flash_attn_forward_cuda` – host-side wrapper that launches the kernel
  - `main` – small test that:
    - randomly initializes Q, K, V
    - runs CPU and GPU implementations
    - compares max absolute error and prints a pass/fail message

You can integrate just the CUDA kernel and wrapper into your course skeleton if
your instructor already provides a separate test harness.

---

## 3. CUDA kernel design

### 3.1 Mapping to thread blocks

- Each thread block handles **BR** query rows (default: `BR = 32`).
- Within a block, each thread is responsible for exactly **one query row** `i`.
  - This guarantees that each `O_i` is written by **exactly one thread**.
  - There are **no write conflicts** on the output tiles.

We tile the keys/values along the sequence dimension with tile size **BC**
(default: `BC = 32`):

- For each tile:
  - We load `BC` rows of `K` and `V` into **shared memory**:
    - `K_tile`: shape `(BC, d)`
    - `V_tile`: shape `(BC, d)`
  - Each thread reuses its `Q_i` against all rows in the current tile.

### 3.2 Online softmax

For each query row `i`, we maintain:

- `m_i`: running maximum score across all keys seen so far.
- `l_i`: running normalizer (denominator) of the softmax.
- `O_i`: running output vector.

For a new tile, we compute the tile-specific max score `m_ij`. The new global max is:

\[
m_{\text{new}} = \max(m_i, m_{ij})
\]

Then we use the standard online-softmax formula:

1. Rescale the old contributions to the new max:

\[
\alpha = e^{m_i - m_{\text{new}}}
\]

2. Let `p` be the unnormalized weights in this new tile:

\[
p_{ik} = e^{s_{ik} - m_{\text{new}}}
\]

3. Update normalizer:

\[
l_{\text{new}} = \alpha l_i + \sum_k p_{ik}
\]

4. Update output:

\[
O_i^{\text{new}} =
\left( \frac{\alpha l_i}{l_{\text{new}}} \right) O_i
+ \left( \frac{1}{l_{\text{new}}} \right) \sum_k p_{ik} V_k
\]

This allows us to avoid storing the full `N x N` score matrix in memory. We only
keep **one row of output and two scalars (`m_i`, `l_i`) per query row**.

---

## 4. Building and running (local CUDA / Modal GPU)

### 4.1 Local CUDA (x86 + NVIDIA GPU)

If you have CUDA toolkit installed and an NVIDIA GPU, you can compile and run:

```bash
nvcc -O3 -std=c++17 flash_attention.cu -o flash_attention
./flash_attention
```

Expected output (numbers will differ, but structure is similar):

```text
Running CPU reference implementation...
Running CUDA FlashAttention kernel...
Max abs diff between CPU and CUDA: 3.2e-05
✅ PASS: CUDA FlashAttention matches CPU reference within tolerance.
```

### 4.2 Running inside a Modal GPU container (M1 / ARM host)

Because your laptop is an Apple Silicon (ARM) MacBook, you cannot run CUDA
directly on your host. Instead, use Modal to run inside an x86 GPU container.

One simple pattern is:

1. Put `flash_attention.cu` in your course `week_04/` directory.
2. Use a `main.py` or existing Modal app to:
   - mount the `week_04` directory into the container
   - compile with `nvcc` inside the container
   - run the produced binary.

A minimal Modal function (for illustration only) could be:

```python
# This is an EXAMPLE only. Use your course's Modal skeleton if provided.
import modal

app = modal.App("week4-flashattention")

image = (
    modal.Image.debian_slim()
    .apt_install("build-essential", "cuda-toolkit-12-4")
)

@app.function(image=image, gpu="a10g")
def run_flashattention():
    import subprocess, os
    os.chdir("/root/week_04")  # adjust to your mount path
    subprocess.check_call(["nvcc", "-O3", "-std=c++17",
                           "flash_attention.cu", "-o", "flash_attention"])
    subprocess.check_call(["./flash_attention"])
```

Then in your terminal:

```bash
modal run main.py::run_flashattention
```

Use your course-provided Modal config if it already exists. Just make sure
`flash_attention.cu` is mounted into the container and compiled with `nvcc`.

---

## 5. Integration with the course skeleton

In the official course repository there is likely a skeleton file for FlashAttention
(e.g., `week_04/flash_attention.cu` or `kernels.cu` with TODOs).

To integrate this solution:

1. Copy the **core functions** from this file:
   - `flash_attn_forward_kernel`
   - `flash_attn_forward_cuda`
2. Adjust the function signatures to match the instructor’s header or skeleton.
3. Remove or ignore the `main` function if the course provides its own tests.
4. Keep the overall logic:
   - one thread per query row,
   - tiling over keys/values into shared memory,
   - online softmax (`m_i`, `l_i`, `O_i` evolution),
   - no racing writes on `O`.

---

## 6. Notes and limitations

- We assume a **single attention head** for clarity.
- The kernel uses a fixed `MAX_D` (default `128`). If you use a larger head
  dimension, increase `MAX_D` accordingly.
- This implementation focuses on **clarity and correctness**, not maximum
  performance. It is sufficient for understanding and for course assignments.

---

## 7. Summary

This implementation shows how to:

- Map attention to CUDA thread blocks without write conflicts on `O`.
- Use **tiling** and **shared memory** to reduce global memory traffic.
- Apply **online softmax** to avoid storing the full score matrix.

You can now:

- Run the file standalone to check the math and numerical stability.
- Move the kernel and wrapper into your course skeleton.
- Use Modal to run everything on a real GPU even from an ARM MacBook.
