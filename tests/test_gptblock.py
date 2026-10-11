import litetorch as lt
from litetorch import nn

print("=== GPTBlock Llama-3 style verification ===")

block = nn.GPTBlock(d_model=2560, n_q_heads=20, n_kv_heads=5, d_ff=6912)
print(f"Config: d_model={block.d_model}, Q heads={block.n_q_heads}, KV heads={block.n_kv_heads}, d_ff={block.d_ff}, head_dim={block.head_dim}, rope_theta={block.rope_theta}")

params = block.parameters()
total = sum(p.numel() for p in params)
print(f"Params per block: {total:,}")
expected_per_layer = 69473536
assert total == expected_per_layer, f"Param mismatch: {total} vs {expected_per_layer}"
print(f"28 layers + embeddings: {(total*28 + 50257*2560 + 2560)/1e9:.3f}B")

print("\n=== Forward pass (small) ===")
sm = nn.GPTBlock(d_model=64, n_q_heads=4, n_kv_heads=2, d_ff=128)
x = lt.Tensor.create([2, 8, 64], lt.Device("cpu"), True)
out = sm.forward(x)
print(f"Input: {x.shape} -> Output: {out.shape}")
assert list(out.shape) == [2, 8, 64], "Shape mismatch"
print("Forward OK")

print("\n=== Causality check ===")
import random
random.seed(0)
lt.manual_seed(0)
x1 = lt.randn([1, 8, 64], lt.Device("cpu"), False)
d1 = x1.to_vector()
d2 = list(d1)
for i in range(4*64, 8*64):
    d2[i] = d1[i] + 1.0
x2 = lt.Tensor.from_vector(d2, [1, 8, 64], lt.Device("cpu"), False)
o1 = sm.forward(x1).to_vector()
o2 = sm.forward(x2).to_vector()
max_diff_first_half = max(abs(a-b) for a,b in zip(o1[:4*64], o2[:4*64]))
print(f"Max diff in first 4 positions (should be ~0): {max_diff_first_half:.2e}")
assert max_diff_first_half < 1e-4, "Causality violated: future tokens affect past!"
print("Causality OK - future tokens do not affect past positions")

print("\n=== Gradient check ===")
xg = lt.Tensor.create([1, 4, 64], lt.Device("cpu"), True)
og = sm.forward(xg)
loss = og.sum()
loss.backward()
has_grad = sum(1 for p in sm.parameters() if p.grad is not None)
print(f"Params with grad: {has_grad}/{len(sm.parameters())}")
assert has_grad == len(sm.parameters()), "Some params missing grad"
print("Backward OK")

print("\n=== ALL GPTBlock TESTS PASSED ===")
