import litetorch as lt
from litetorch import nn
import time

print("=== GPTBlock MFU Benchmark (Wave 2: fused QKV + fused gate/up) ===")

device = lt.Device("cuda:0")
B, T = 2, 2048

block = nn.GPTBlock(d_model=2560, n_q_heads=20, n_kv_heads=5, d_ff=6912)
block.to(device)
params = block.parameters()
total_params = sum(p.numel() for p in params)
print(f"Params per block: {total_params:,}")

x = lt.randn([B, T, 2560], device, True)

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

flops_per_iter = 2 * 2 * 6 * total_params * B * T
tflops = flops_per_iter / (ms_per_iter / 1000) / 1e12
print(f"Achieved: {tflops:.1f} TFLOPS")
print(f"MFU vs T4 peak 65 TFLOPS (FP16): {tflops / 65 * 100:.1f}%")
print(f"MFU vs T4 sustained ~40 TFLOPS: {tflops / 40 * 100:.1f}%")
