import math, time
import torch
from torch.utils.cpp_extension import load

device = 'cuda'
dtype = torch.float16  # or torch.bfloat16

ext = load(name='flash_attn2_ext',
           sources=[
             'src/binding.cpp',
             'src/flash_attn2_fwd.cu',
             'src/flash_attn2_bwd.cu',
           ],
           extra_cflags=['-O3', '-std=c++17'],
           extra_cuda_cflags=['-O3', '-lineinfo'],
           verbose=False)

def sdpa_ref(q,k,v,causal):
    return torch.nn.functional.scaled_dot_product_attention(q,k,v, attn_mask=None, dropout_p=0.0, is_causal=causal)

@torch.inference_mode()
def check_forward(B=2,H=4,N=256,D=64, causal=True):
    q = torch.randn(B,H,N,D, device=device, dtype=dtype) / math.sqrt(D)
    k = torch.randn(B,H,N,D, device=device, dtype=dtype) / math.sqrt(D)
    v = torch.randn(B,H,N,D, device=device, dtype=dtype)

    o_ref = sdpa_ref(q,k,v,causal)
    o, m, l = ext.forward_with_stats(q,k,v,causal)

    atol, rtol = (1e-2, 1e-2)
    max_abs = (o - o_ref).abs().max().item()
    max_rel = ((o - o_ref).abs() / (o_ref.abs()+1e-6)).max().item()
    print(f"Forward diff: abs={max_abs:.3e} rel={max_rel:.3e}")
    assert max_abs <= atol + 1e-5 and max_rel <= rtol + 1e-5

def check_backward(B=1,H=2,N=64,D=32, causal=True):
    # use float32 reference for stable autograd compare
    q = torch.randn(B,H,N,D, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B,H,N,D, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B,H,N,D, device=device, dtype=torch.float32, requires_grad=True)
    qh, kh, vh = q.to(dtype), k.to(dtype), v.to(dtype)

    with torch.no_grad():
        o, m, l = ext.forward_with_stats(qh,kh,vh,causal)
    o = o.to(torch.float32)

    dout = torch.randn_like(o)
    dq, dk, dv = ext.backward(qh,kh,vh,o.to(dtype), m, l, dout.to(dtype), causal)

    # Reference gradients from PyTorch SDPA
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    o_ref = sdpa_ref(q2,k2,v2,causal)
    o_ref.backward(dout)

    def maxdiff(a,b): return (a.float()-b.float()).abs().max().item()

    print("Grad diffs:",
          "dq", maxdiff(dq, q2.grad),
          "dk", maxdiff(dk, k2.grad),
          "dv", maxdiff(dv, v2.grad))

def bench(B=2,H=8,N=1024,D=128, causal=True, iters=20):
    q = torch.randn(B,H,N,D, device=device, dtype=dtype)
    k = torch.randn(B,H,N,D, device=device, dtype=dtype)
    v = torch.randn(B,H,N,D, device=device, dtype=dtype)

    # warmup
    o = ext.forward(q,k,v,causal)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        o = ext.forward(q,k,v,causal)
    torch.cuda.synchronize()
    t1 = time.time()

    o2 = sdpa_ref(q,k,v,causal)
    torch.cuda.synchronize()
    t2 = time.time()

    print(f"FA2(forward) avg: {(t1-t0)/iters*1000:.2f} ms; SDPA (1 run): {(t2-t1)*1000:.2f} ms")

if __name__ == "__main__":
    torch.cuda.init()
    check_forward()
    check_backward()
    bench()
