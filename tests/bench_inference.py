import sys
sys.path.insert(0, "python")

import time

import litetorch as lt
from litetorch import nn

print("=== Inference Perf Benchmark (I2) ===")

has_cuda = False
try:
    has_cuda = lt.cuda.is_available() if hasattr(lt, "cuda") else False
except Exception:
    pass
if not has_cuda:
    print("SKIP: no CUDA device (GPU-gated)")
    sys.exit(0)

has_torch = False
try:
    import torch
    has_torch = torch.cuda.is_available()
except ImportError:
    pass

device = lt.Device("cuda:0")
B, T, D = 4, 128, 512

block = nn.GPTBlock(d_model=D, n_q_heads=8, n_kv_heads=2, d_ff=1024)
block.to(device)
block.eval()
x = lt.randn([B, T, D], device, False)

for _ in range(3):
    block.forward(x)
lt.cuda_synchronize()

def bench(fn, iters=20):
    lt.cuda_synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    lt.cuda_synchronize()
    dt = time.perf_counter() - t0
    toks = B * T * iters / dt
    return dt / iters * 1000, toks

ms, toks = bench(lambda: block.forward(x))
print(f"eager:          {ms:.2f} ms/iter  {toks:,.0f} tok/s")

try:
    g = lt.inference.capture_cuda_graph(block, x)
    ms_g, toks_g = bench(lambda: g.replay())
    print(f"cuda-graph:     {ms_g:.2f} ms/iter  {toks_g:,.0f} tok/s  ({toks_g/toks:.2f}x)")
except Exception as e:
    print(f"cuda-graph: SKIP ({e})")

if has_torch:
    tblock = torch.nn.TransformerEncoderLayer(d_model=D, nhead=8, dim_feedforward=1024,
                                              batch_first=True).cuda().eval()
    tx = torch.randn(B, T, D, device="cuda")
    for _ in range(3):
        tblock(tx)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        tblock(tx)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ttoks = B * T * iters / dt
    print(f"pytorch eager:  {dt/iters*1000:.2f} ms/iter  {ttoks:,.0f} tok/s")
    try:
        cblock = torch.compile(tblock, mode="reduce-overhead")
        for _ in range(3):
            cblock(tx)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            cblock(tx)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ctoks = B * T * iters / dt
        print(f"torch.compile:  {dt/iters*1000:.2f} ms/iter  {ctoks:,.0f} tok/s")
    except Exception as e:
        print(f"torch.compile: SKIP ({e})")
else:
    print("pytorch: SKIP (not installed)")

print("done")
