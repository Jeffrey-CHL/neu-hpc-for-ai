# Week 8 – DeepSeek-Style Mixture-of-Experts (MoE) Operator (Unfused, CUDA)

This repository contains a **single-GPU** implementation of a simple
**Mixture-of-Experts (MoE)** operator, inspired by the DeepSeek-V3 MoE layer.

The implementation focuses on:

- **Top-1 Gate / Router (Switch-style)**
- **Multiple MLP Experts**
- **Token → Expert dispatch (logical expert parallelism)**
- **Expert outputs → Original token order combine**
- All implemented with **CUDA kernels** (unfused – communication and
  computation are separate steps).

> Note: This implementation simulates **expert-parallel routing logic on one GPU**.  
> Extending it to truly distributed multi-GPU expert parallelism would require
> adding collective communication (e.g., NCCL AllToAll) on top of this operator.

---

## Files

- `main.cu`  
  Core CUDA implementation:
  - Gate / router (logits + softmax + Top-1)
  - Token counting per expert
  - Position assignment within each expert mini-batch
  - Dispatch into a compact expert input buffer
  - Per-expert MLP (two-layer FFN with GELU)
  - Combine outputs back to the original token order

- `Makefile`  
  Simple build script using `nvcc`.

- `run.sh`  
  Convenience script to build and run the MoE example in any CUDA-capable
  environment (e.g., a Modal GPU container).

- `main.py`  
  Modal entrypoint that:
  - Mounts this directory into a GPU-enabled container.
  - Builds `deepseek_moe` with `make`.
  - Runs the binary on a GPU.

---

## How to Build and Run (CUDA Environment)

You need:

- A CUDA-capable GPU (e.g., A10, A100, etc.)
- CUDA toolkit installed (`nvcc` must be in your `PATH`)

Then:

```bash
make        # build the binary
make run    # build and run, or:
./deepseek_moe
