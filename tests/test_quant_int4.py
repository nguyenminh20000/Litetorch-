import sys
sys.path.insert(0, "python")
import numpy as np
import litetorch as lt
from litetorch import ops, nn

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

print("== INT4 pack/unpack roundtrip ==")
np.random.seed(1)
vals = np.random.randint(-8, 8, size=(16,)).astype(np.int32)
x = lt.tensor(vals.tolist(), dtype=lt.INT8)
packed = ops.int4_pack(x)
check("packed size halved", list(packed.shape) == [8])
unpacked = ops.int4_unpack(packed, 16)
unp_np = to_np(unpacked).astype(np.int32)
check("int4 roundtrip exact", bool((unp_np == vals).all()))

print("== INT4 odd numel ==")
vals_odd = np.random.randint(-8, 8, size=(7,)).astype(np.int32)
x_odd = lt.tensor(vals_odd.tolist(), dtype=lt.INT8)
packed_odd = ops.int4_pack(x_odd)
check("odd packed size", list(packed_odd.shape) == [4])
unp_odd = to_np(ops.int4_unpack(packed_odd, 7)).astype(np.int32)
check("odd roundtrip exact", bool((unp_odd == vals_odd).all()))

print("== INT4 weight-only linear ==")
x_np = (np.random.randn(2, 16)).astype(np.float32)
w_np = (np.random.randn(8, 16)).astype(np.float32)
w_scale_np = np.abs(w_np).max(axis=1) / 7.0
w_q = np.round(w_np / w_scale_np[:, None]).clip(-8, 7).astype(np.int32)
x_t = lt.tensor(x_np.tolist())
wq_t = lt.tensor(w_q.tolist(), dtype=lt.INT8)
packed_w = ops.int4_pack(wq_t.reshape([8 * 8])).reshape([8, 8])
scale_t = lt.tensor(w_scale_np.tolist())
out = ops.int4_weight_only_linear(x_t, packed_w, scale_t)
out_np = to_np(out)
ref = x_np @ (w_q.astype(np.float32) * w_scale_np[:, None]).T
rel_err = np.abs(out_np - ref) / (np.abs(ref) + 1e-3)
check("int4 linear rel_err<1e-3", float(rel_err.max()) < 1e-3)

print(f"\nTOTAL: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
