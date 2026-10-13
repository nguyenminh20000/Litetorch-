import sys
sys.path.insert(0, "python")
import numpy as np
import litetorch as lt
ops = lt.Ops

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name}")

def to_np(t):
    return np.array(t.tolist(), dtype=np.float64)

print("== FP8 E4M3 cast roundtrip ==")
np.random.seed(0)
x_np = (np.random.randn(4, 8) * 2).astype(np.float32)
x = lt.tensor(x_np.tolist())
scale = float(np.abs(x_np).max() / 240.0)
xq = ops.fp8_cast_scaled(x, scale, lt.FP8_E4M3)
check("fp8_e4m3 dtype", str(xq.dtype) == "FP8_E4M3" or "E4M3" in str(xq.dtype))
xr = ops.fp8_uncast_scaled(xq, scale)
xr_np = to_np(xr)
rel_err = np.abs(xr_np - x_np) / (np.abs(x_np) + 1e-6)
check("fp8_e4m3 roundtrip rel_err<7e-2", float(rel_err.max()) < 7e-2)

print("== FP8 E5M2 cast roundtrip ==")
xq2 = ops.fp8_cast_scaled(x, scale, lt.FP8_E5M2)
xr2 = ops.fp8_uncast_scaled(xq2, scale)
xr2_np = to_np(xr2)
rel_err2 = np.abs(xr2_np - x_np) / (np.abs(x_np) + 1e-6)
check("fp8_e5m2 roundtrip rel_err<1.2e-1", float(rel_err2.max()) < 1.2e-1)

print("== FP8 matmul ==")
a_np = (np.random.randn(4, 8)).astype(np.float32)
b_np = (np.random.randn(8, 6)).astype(np.float32)
a = lt.tensor(a_np.tolist())
b = lt.tensor(b_np.tolist())
a_scale = float(np.abs(a_np).max() / 240.0)
b_scale = float(np.abs(b_np).max() / 240.0)
aq = ops.fp8_cast_scaled(a, a_scale, lt.FP8_E4M3)
bq = ops.fp8_cast_scaled(b, b_scale, lt.FP8_E4M3)
ref = a_np @ b_np
out_scale = float(np.abs(ref).max() / 240.0) + 1e-6
c = ops.fp8_matmul(aq, bq, a_scale, b_scale, out_scale, lt.FP8_E4M3)
c_np = to_np(ops.fp8_uncast_scaled(c, out_scale))
rel_err3 = np.abs(c_np - ref) / (np.abs(ref) + 1e-3)
check("fp8_matmul mean_rel_err<2e-1", float(rel_err3.mean()) < 2e-1)

print(f"\nTOTAL: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
