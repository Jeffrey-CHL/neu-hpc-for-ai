import math
import time
import torch
from torch.utils.cpp_extension import load


# JIT-compile the CUDA extension.
# extra_include_paths ensures we can find "flash_attn2.cuh" in include/.
ext = load(
    name="flash_attn2_ext",
    sources=[
        "src/binding.cpp",
        "src/flash_attn2_fwd.cu",
        "src/flash_attn2_bwd.cu",
    ],
    extra_cflags=[
        "-O3",
        "-std=c++17",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "-lineinfo",
        # Make sure half / bfloat16 operators are available.
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        # 允许 __device__ / __host__ lambda
        "--extended-lambda",
    ],
    extra_include_paths=["include", "src"],
    verbose=True,
)


# ---------------------------------------------------------------------------
# Thin Python wrappers around C++/CUDA extension
# ---------------------------------------------------------------------------

def flash_attn2_forward(q, k, v, causal: bool = False):
    """
    q, k, v: (B, H, L, D)
    causal: whether to use causal masking.
    Returns: output (B, H, L, D)

    NOTE: 对应 binding.cpp 里的 fa2_forward -> 导出名 "forward"
    """
    return ext.forward(q, k, v, causal)


def flash_attn2_forward_with_stats(q, k, v, causal: bool = False):
    """
    调用扩展里带统计量版本：
    返回 (o, m, l)
    o: output (B, H, L, D)
    m, l: softmax 的 log-sum-exp 相关统计 (B, H, L)
    """
    return ext.forward_with_stats(q, k, v, causal)


def flash_attn2_backward(q, k, v, o, m, l, dout, causal: bool = False):
    """
    Backward wrapper; 参数顺序必须匹配 binding.cpp 中的 fa2_backward：

      fa2_backward(q, k, v, o, m, l, dout, causal) -> (dq, dk, dv)

    q, k, v, o, m, l, dout 都是 CUDA tensor。
    """
    return ext.backward(q, k, v, o, m, l, dout, causal)


# ---------------------------------------------------------------------------
# Utilities to create random test inputs
# ---------------------------------------------------------------------------

def _make_inputs(
    B: int = 2,
    H: int = 4,
    L: int = 256,
    D: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """
    Create random Q, K, V tensors with shape (B, H, L, D).
    """
    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype)
    return q, k, v


# ---------------------------------------------------------------------------
# Forward correctness check
# ---------------------------------------------------------------------------

def check_forward(
    B: int = 2,
    H: int = 4,
    L: int = 256,
    D: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    causal: bool = False,
    atol: float = 2e-2,
    rtol: float = 2e-2,
):
    """
    Compare the forward pass of our FlashAttention-2 kernel against
    PyTorch's scaled_dot_product_attention (SDPA).
    """
    q, k, v = _make_inputs(B, H, L, D, device=device, dtype=dtype)

    # Our CUDA kernel: (B, H, L, D)
    out_fa2 = flash_attn2_forward(q, k, v, causal)

    # Reference: PyTorch SDPA expects (B, L, H, D)
    q_ref = q.transpose(1, 2)
    k_ref = k.transpose(1, 2)
    v_ref = v.transpose(1, 2)

    attn_mask = None
    if causal:
        mask = torch.full((L, L), float("-inf"), device=device, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=1)
        attn_mask = mask

    out_ref = torch.nn.functional.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )  # (B, L, H, D)

    out_ref = out_ref.transpose(1, 2)  # back to (B, H, L, D)

    # Compare maximum absolute and relative error
    abs_diff = (out_fa2 - out_ref).abs().max().item()
    rel_diff = abs_diff / (out_ref.abs().max().item() + 1e-6)

    print(f"[Forward] max abs diff = {abs_diff:.4e}, max rel diff = {rel_diff:.4e}")
    if abs_diff > atol and rel_diff > rtol:
        print("Forward check: ❌ FAIL (differences too large)")
    else:
        print("Forward check: ✅ PASS")


# ---------------------------------------------------------------------------
# Backward correctness check
# ---------------------------------------------------------------------------

def check_backward(
    B: int = 2,
    H: int = 4,
    L: int = 256,
    D: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    causal: bool = False,
    atol: float = 5e-2,
    rtol: float = 5e-2,
):
    """
    Compare gradients dQ, dK, dV between our kernel and SDPA.

    注意：这里不会依赖 autograd 去 “反推” 我们的 CUDA kernel，
    而是直接调用扩展里的 backward(q,k,v,o,m,l,dout,causal)。
    """
    # ---- Our kernel path (using custom backward) ----
    q_fa, k_fa, v_fa = _make_inputs(B, H, L, D, device=device, dtype=dtype)

    # forward with stats (o, m, l) 用于 backward
    o_fa, m_fa, l_fa = flash_attn2_forward_with_stats(q_fa, k_fa, v_fa, causal)

    # 简单起见，用全 1 的 dout
    dout = torch.ones_like(o_fa)

    dq_fa, dk_fa, dv_fa = flash_attn2_backward(
        q_fa, k_fa, v_fa, o_fa, m_fa, l_fa, dout, causal
    )

    # ---- Reference SDPA path (autograd) ----
    q_ref = q_fa.clone().detach().requires_grad_(True)
    k_ref = k_fa.clone().detach().requires_grad_(True)
    v_ref = v_fa.clone().detach().requires_grad_(True)

    q_t = q_ref.transpose(1, 2)
    k_t = k_ref.transpose(1, 2)
    v_t = v_ref.transpose(1, 2)

    attn_mask = None
    if causal:
        mask = torch.full((L, L), float("-inf"), device=device, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=1)
        attn_mask = mask

    out_ref = torch.nn.functional.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    out_ref = out_ref.transpose(1, 2)

    # 使用同样的 dout
    (out_ref * dout).sum().backward()

    dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

    def _report(name: str, fa: torch.Tensor, ref: torch.Tensor):
        abs_diff = (fa - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[Backward {name}] max abs diff = {abs_diff:.4e}, max rel diff = {rel_diff:.4e}")
        if abs_diff > atol and rel_diff > rtol:
            print(f"  -> ❌ {name} gradient check FAIL")
        else:
            print(f"  -> ✅ {name} gradient check PASS")

    _report("dQ", dq_fa, dq_ref)
    _report("dK", dk_fa, dk_ref)
    _report("dV", dv_fa, dv_ref)


# ---------------------------------------------------------------------------
# Benchmark: FlashAttention-2 vs PyTorch SDPA
# ---------------------------------------------------------------------------

def bench(
    B: int = 4,
    H: int = 8,
    L: int = 1024,
    D: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    causal: bool = False,
    warmup: int = 5,
    iters: int = 20,
):
    """
    Simple wall-clock benchmark comparing our FlashAttention-2 kernel
    with PyTorch's scaled_dot_product_attention.
    """
    q, k, v = _make_inputs(B, H, L, D, device=device, dtype=dtype)

    # Warm-up for more stable timing
    for _ in range(warmup):
        _ = flash_attn2_forward(q, k, v, causal)
    torch.cuda.synchronize()

    for _ in range(warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
        )
    torch.cuda.synchronize()

    # Measure FlashAttention-2
    start = time.time()
    for _ in range(iters):
        _ = flash_attn2_forward(q, k, v, causal)
    torch.cuda.synchronize()
    t_fa2 = (time.time() - start) * 1000.0 / iters

    # Measure SDPA
    start = time.time()
    for _ in range(iters):
        _ = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
        )
    torch.cuda.synchronize()
    t_sdpa = (time.time() - start) * 1000.0 / iters

    print(
        f"[Bench] FlashAttention-2: {t_fa2:.3f} ms, "
        f"PyTorch SDPA: {t_sdpa:.3f} ms, "
        f"speedup: {t_sdpa / t_fa2:.2f}x"
    )


if __name__ == "__main__":
    print("=== Running forward check ===")
    check_forward()
    print("=== Running backward check ===")
    check_backward()
    print("=== Running benchmark ===")
    bench()