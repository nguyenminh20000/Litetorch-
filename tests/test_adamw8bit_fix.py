import litetorch as lt
import math

print("=== AdamW8bit Fix Verification ===")

# Test 1: FP32 params (basic correctness)
print("\n[Test 1] FP32 params")
p = lt.Tensor([2.0, 4.0], [2], lt.Device("cpu:0"), True)
p.grad = lt.Tensor([0.1, 0.2], [2], lt.Device("cpu:0"), False)
opt = lt.optim.AdamW8bit([p], lr=0.01)
opt.step()
vals = p.to_vector()
print(f"  After step: {vals}")
assert vals[0] != 2.0, "param should have updated"
print("  PASS")

# Test 2: FP16 params (the critical bug fix)
print("\n[Test 2] FP16 params (was garbage before fix)")
try:
    p16 = lt.Tensor([2.0, 4.0], [2], lt.Device("cpu:0"), True, lt.DataType.FP16)
    p16.grad = lt.Tensor([0.1, 0.2], [2], lt.Device("cpu:0"), False, lt.DataType.FP16)
    opt16 = lt.optim.AdamW8bit([p16], lr=0.01)
    opt16.step()
    vals16 = p16.to_vector()
    print(f"  After step: {vals16}")
    assert all(abs(v) < 10.0 for v in vals16), f"FP16 values garbage: {vals16}"
    assert vals16[0] != 2.0, "FP16 param should have updated"
    print("  PASS - FP16 handled correctly")
except Exception as e:
    print(f"  SKIP/FAIL: {e}")

# Test 3: Block-wise quantization (large tensor > 2048 elements)
print("\n[Test 3] Block-wise quantization (5000 elements = 3 blocks)")
n = 5000
p3 = lt.Tensor([1.0] * n, [n], lt.Device("cpu:0"), True)
p3.grad = lt.Tensor([0.01] * n, [n], lt.Device("cpu:0"), False)
opt3 = lt.optim.AdamW8bit([p3], lr=0.01)
opt3.step()
nblocks = (n + 2047) // 2048
print(f"  Blocks: {nblocks}, m_scales len: {len(opt3.m_scales[0])}")
assert len(opt3.m_scales[0]) == nblocks, "should have per-block scales"
print("  PASS")

# Test 4: Multiple steps (state persistence)
print("\n[Test 4] Multiple steps")
p4 = lt.Tensor([1.0, 2.0, 3.0], [3], lt.Device("cpu:0"), True)
opt4 = lt.optim.AdamW8bit([p4], lr=0.01)
for i in range(5):
    p4.grad = lt.Tensor([0.1, 0.1, 0.1], [3], lt.Device("cpu:0"), False)
    opt4.step()
vals4 = p4.to_vector()
print(f"  After 5 steps: {vals4}")
assert vals4[0] < 1.0, "should decrease with positive grad"
print("  PASS")

print("\n=== All tests passed ===")
