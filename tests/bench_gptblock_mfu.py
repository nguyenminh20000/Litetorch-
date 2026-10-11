import litetorch as lt
from litetorch import nn
import time
import sys

use_fp16 = "--fp16" in sys.argv
dtype_name = "FP16" if use_fp16 else "FP32"
peak = 65.0 if use_fp16 else 8.1

print(f"=== GPTBlock MFU Benchmark [{dtype_name}] ===")

device = lt.Device("cuda:0")
B, T = 1, 512

block = nn.GPTBlock(d_model=2560, n_q_heads=20, n_kv_heads=5, d_ff=6912)
block.to(device)
if use_fp16:
    block.half()
    lt.cuda.set_fp16_math_enabled(True)
params = block.parameters()
total_params = sum(p.numel() for p in params)
print(f"Params per block: {total_params:,}")

x = lt.randn([B, T, 2560], device, True)
if use_fp16:
    x = x.cast(lt.DataType.FP16)

for _ in range(3):
    out = block.forward(x)
    loss = lt.Ops.sum(out)
    loss.backward()
    for p in params:
        p.zero_grad()
lt.cuda_synchronize()

n_iters = 20
lt.cuda_synchronize()
t0 = time.perf_counter()
for _ in range(n_iters):
    out = block.forward(x)
    loss = lt.Ops.sum(out)
    loss.backward()
    for p in params:
        p.zero_grad()
lt.cuda_synchronize()
t1 = time.perf_counter()

ms_per_iter = (t1 - t0) / n_iters * 1000
print(f"Forward+backward per iter: {ms_per_iter:.1f} ms")

flops_per_iter = 6 * total_params * B * T
tflops = flops_per_iter / (ms_per_iter / 1000) / 1e12
print(f"Achieved: {tflops:.2f} TFLOPS")
print(f"MFU vs T4 {dtype_name} peak {peak} TFLOPS: {tflops / peak * 100:.1f}%")
