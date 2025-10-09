# Week 5 — FlashAttention-2 (Section 3) — Reimplementation

**Course**: INFO 7375 — High Performance Computing for AI  
**Assignment**: Implement FlashAttention-2 as described in **Section 3** of the paper:
- §3.1 Forward & Backward (online softmax, recomputation for backward)
- §3.2 Parallelism (tiled scanning of KV, block-per-Qtile)
- §3.3 Work Partitioning between Warps

This submission provides a **simplified but correct** FA‑2 style implementation that emphasizes algorithmic fidelity over micro-optimizations.
It supports FP16/BF16 with FP32 accumulation, causal masking, and online softmax statistics `(m, l)` saved for backward.

> **Note**: For clarity, this kernel favors correctness over peak FLOPs. Comments highlight where to add `cp.async`, double-buffering,
> tensor-core tiling, and warp specialization to reach paper-level throughput.

## Repository Layout
```
week_05_flashattention2/
├─ README.md
├─ include/
│  └─ flash_attn2.cuh
├─ src/
│  ├─ binding.cpp
│  ├─ flash_attn2_fwd.cu
│  └─ flash_attn2_bwd.cu
└─ test_flash_attn2.py
```

## Build & Run (Python JIT)
```
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python test_flash_attn2.py
```
The script will JIT-compile the CUDA extension and run forward/backward checks and a small benchmark.

## Implementation Overview
**Forward (§3.1)**  
- Tile queries by `Br` rows per CUDA block and stream KV tiles of width `Bc`.
- Maintain per-row `m_i` (running max) and `l_i` (exp-sum) in FP32 for numerical stability.
- Update the output accumulator `O` per tile, reweighting previous `O` by `exp(m_prev - m_new)`.

**Backward (§3.1)**  
- Recompute partial probabilities `P` per tile using saved `(m,l)`; accumulate `dV`, form row scalars `alpha = Σ_j P_ij * (dO·V_j)`,
  then `dP = P ⊙ ((dO·V^T) - alpha)` to produce `dQ` and `dK`.
- For simplicity and correct atomics, gradients are accumulated in **float32 buffers** and cast back to the input dtype in Python.

**Parallelism (§3.2)**  
- Grid over `(B, H, N/Br)`; each block streams KV tiles. (Persistent blocks and split‑KV are noted in comments.)

**Work Partition (§3.3)**  
- Comments mark where to separate **load warps** vs **compute warps** and where to insert **double-buffering** with `cp.async`.

## Notes & Limits
- This reference targets head dimensions `D ≤ 256` to keep shared-memory staging simple.
- Padding masks are omitted for brevity (easy to add with an index list); **causal** is supported.
- Tolerances for FP16 forward: `atol=1e-2, rtol=1e-2`.

## Reflection (fill with your measurements)
- Online softmax avoids materializing `N×N` and preserves numerical stability.
- Backward recomputation trades extra flops for lower memory.
- Main performance levers (not fully exploited here): tensor-core MMA tiling, `cp.async` pipelining, persistent blocks, split‑KV reductions.
