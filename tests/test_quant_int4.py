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
packed_w = ops.int4_pack(wq_t.reshape([8 * 16])).reshape([8, 8])
scale_t = lt.tensor(w_scale_np.tolist())
out = ops.int4_weight_only_linear(x_t, packed_w, scale_t)
out_np = to_np(out)
ref = x_np @ (w_q.astype(np.float32) * w_scale_np[:, None]).T
rel_err = np.abs(out_np - ref) / (np.abs(ref) + 1e-3)
check("int4 linear rel_err<1e-3", float(rel_err.max()) < 1e-3)

print("== INT4 odd-K weight-only linear (per-row unpack) ==")
def pack_rows_int4(w_q):
    rows = []
    for r in range(w_q.shape[0]):
        rows.append([int(v) for v in ops.int4_pack(lt.tensor(w_q[r].tolist(), dtype=lt.INT8)).tolist()])
    return lt.tensor(rows, dtype=lt.INT8)

def odd_int4_case(name, N, K, seed):
    rng = np.random.RandomState(seed)
    x_np = rng.randn(2, K).astype(np.float32)
    w_np = rng.randn(N, K).astype(np.float32)
    scale_np = np.abs(w_np).max(axis=1) / 7.0
    w_q = np.round(w_np / scale_np[:, None]).clip(-8, 7).astype(np.int32)
    packed_w = pack_rows_int4(w_q)
    check(name + " packed shape", list(packed_w.shape) == [N, (K + 1) // 2])
    out = ops.int4_weight_only_linear(lt.tensor(x_np.tolist()), packed_w, lt.tensor(scale_np.tolist()))
    ref = x_np @ (w_q.astype(np.float32) * scale_np[:, None]).T
    rel = np.abs(to_np(out) - ref) / (np.abs(ref) + 1e-3)
    check(name + " rel_err<1e-3", float(rel.max()) < 1e-3)

odd_int4_case("int4 odd-K N=2 K=5", 2, 5, 11)
odd_int4_case("int4 odd-K N=2 K=3", 2, 3, 12)
odd_int4_case("int4 odd-K N=2 K=1", 2, 1, 13)
odd_int4_case("int4 odd-K N=1 K=5", 1, 5, 14)

print("== INT4 unpack validation ==")
def check_throws(name, fn):
    global passed, failed
    try:
        fn()
    except Exception:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name}")

small = ops.int4_pack(lt.tensor([1, 2, 0, 7, 5, 5, 3, 2], dtype=lt.INT8))
check("int4 packed fixture size", list(small.shape) == [4])
check("int4 packed fixture bytes", to_np(small).astype(np.int32).tolist() == [-87, -8, -35, -85])
check_throws("int4 oversized numel throws", lambda: ops.int4_unpack(small, 10))
check_throws("int4 FP32 packed throws", lambda: ops.int4_unpack(lt.tensor([1.0, 2.0, 3.0, 4.0]), 8))
check_throws("int4 NF4-dtype packed throws", lambda: ops.int4_unpack(lt.tensor([1.0, 2.0, 3.0, 4.0]).cast(lt.NF4), 8))
check_throws("int4 UINT8 packed throws", lambda: ops.int4_unpack(lt.tensor([1, 2, 3, 4], dtype=lt.UINT8), 8))
check("int4 exact ceil numel passes", bool((to_np(ops.int4_pack(ops.int4_unpack(small, 8))).astype(np.int32) == to_np(small).astype(np.int32)).all()))
check("int4 odd ceil numel passes", to_np(ops.int4_pack(ops.int4_unpack(small, 7))).astype(np.int32).tolist() == [-87, -8, -35, -117])

print(f"\nTOTAL: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
