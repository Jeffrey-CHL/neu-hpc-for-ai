# Week 04 — FlashAttention (CUDA) & PMPP Ch.6 Q4

> Course: INFO 7375 — High Performance Computing for AI  
> Student: Hanlin (Jeffrey) Cheng  
> Date: 2025-10-01

## Deliverables
- **PMPP Ch.6 Q4** — written answer (see `pmpq_ch6_q4.md`).
- **FlashAttention in CUDA** — `flash_attn_cuda.cu` (kernel + host wrapper).
- **Reference CPU implementation** — `flash_attn_ref.c` (online-softmax FlashAttention) and `naive_attention.c` (for correctness checks).
- **Tester** — `test_runner.cpp` comparing CUDA with CPU for random cases.
- **Build** — `Makefile` with targets for CPU-only and CUDA.
- **(Optional)** `llama2.cu` hook-in to your GEMM kernel (not included; stub provided).

## How the tiled (online) softmax works
For a sequence length `N` and head dimension `d`, choose tile sizes `B_r` (rows of Q) and `B_c` (cols of K/V). For each Q tile `Q_i`:
1. Initialize running max `m_i ← -inf`, running denom `l_i ← 0`, and partial output `O_i ← 0`.
2. For each KV tile `K_j, V_j`:
   - `S_ij = (Q_i K_j^T) / sqrt(d)`
   - `m_ij = rowmax(S_ij)`
   - `m_i_new = max(m_i, m_ij)` (elementwise)
   - `P_ij = exp(S_ij - m_i_new)`
   - `l_ij = rowsum(P_ij)`
   - `O_i ← diag(exp(m_i - m_i_new)) O_i + P_ij V_j`
   - `l_i ← exp(m_i - m_i_new) ⊙ l_i + l_ij`
   - `m_i ← m_i_new`
3. **Finalize**: `O_i ← diag(l_i)^{-1} O_i`

This avoids forming `N×N` attention and keeps numerical stability with the online normalizer.

### Threadblock mapping (CUDA)
Assign **one threadblock per `O_i` tile** so each output row is written by exactly one block. Within the block:
- Stage `Q_i`, `K_j`, `V_j`, and temporaries (`S_ij`, `P_ij`, `m_ij`, `l_ij`) into shared memory.
- Loop `j` over KV tiles; update `(m_i, l_i, O_i)` in-register or in shared memory.
- Global writes: only the final `O_i` tile.

Shared memory budget (floats):  
`M < 2*B_c*d + 2*B_r*d + 6*B_r + 2*B_r*B_c`  
Multiply by 4 bytes for FP32 to compare with per-SM shared memory.

## Build
```
# CPU-only quick check
make cpu

# CUDA build (requires NVCC & GPU)
make cuda

# Run tests
./build/test_cpu            # CPU vs CPU (naive vs flash)
./build/test_cuda           # CPU (ref) vs CUDA
```

## File Tree
```
week_04/
  Makefile
  README.md
  naive_attention.c
  flash_attn_ref.c
  flash_attn_cuda.cu
  test_runner.cpp
  pmpq_ch6_q4.md
  llama2_cu_stub.cu
```
