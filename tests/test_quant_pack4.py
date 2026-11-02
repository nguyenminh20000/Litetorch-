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

def to_int_list(t):
    return [int(v) for v in t.tolist()]

print("== NF4 pack/unpack roundtrip ==")
np.random.seed(2)
vals = (np.random.rand(16) * 2.0 - 1.0).astype(np.float32)
w_nf4 = lt.tensor(vals.tolist()).cast(lt.NF4)
packed = ops.nf4_pack(w_nf4)
check("nf4 packed size halved", list(packed.shape) == [8])
unpacked = ops.nf4_unpack(packed, 16)
check("nf4 roundtrip exact", to_np(unpacked.cast(lt.FP32)).tolist() == to_np(w_nf4.cast(lt.FP32)).tolist())
check("nf4 repack identical", to_int_list(ops.nf4_pack(unpacked)) == to_int_list(packed))

print("== NF4 odd numel ==")
vals_odd = (np.random.rand(7) * 2.0 - 1.0).astype(np.float32)
w_odd = lt.tensor(vals_odd.tolist()).cast(lt.NF4)
packed_odd = ops.nf4_pack(w_odd)
check("nf4 odd packed size", list(packed_odd.shape) == [4])
unp_odd = ops.nf4_unpack(packed_odd, 7)
check("nf4 odd roundtrip exact", to_np(unp_odd.cast(lt.FP32)).tolist() == to_np(w_odd.cast(lt.FP32)).tolist())

print("== NF4 nibble order (even -> LOW, odd -> HIGH) ==")
idx = lt.tensor([1, 2, 0, 15, 5, 5], dtype=lt.INT8)
packed_idx = ops.nf4_pack(idx)
b = to_int_list(packed_idx)
check("nf4 nibble order", b == [0x21, -0x10, 0x55])
check("nf4 int8-index roundtrip", to_int_list(ops.nf4_pack(ops.nf4_unpack(packed_idx, 6))) == b)
check("nf4 fp32-input path", to_int_list(ops.nf4_pack(lt.tensor(vals.tolist()))) == to_int_list(packed))

print("== NF4 weight-only linear ==")
x_np = (np.random.randn(2, 16)).astype(np.float32)
w_np = (np.random.randn(8, 16)).astype(np.float32)
w_scale_np = np.abs(w_np).max(axis=1)
w_scaled = w_np / w_scale_np[:, None]
w_nf4_t = lt.tensor(w_scaled.tolist()).cast(lt.NF4)
packed_w = ops.nf4_pack(w_nf4_t.reshape([8 * 16])).reshape([8, 8])
scale_t = lt.tensor(w_scale_np.tolist())
x_t = lt.tensor(x_np.tolist())
out = ops.nf4_weight_only_linear(x_t, packed_w, scale_t)
out_np = to_np(out)
w_deq = to_np(w_nf4_t.cast(lt.FP32))
ref = x_np @ (w_deq * w_scale_np[:, None]).T
rel_err = np.abs(out_np - ref) / (np.abs(ref) + 1e-3)
check("nf4 linear rel_err<1e-3", float(rel_err.max()) < 1e-3)
bias_np = (np.random.randn(8)).astype(np.float32)
bias_t = lt.tensor(bias_np.tolist())
out_b = ops.nf4_weight_only_linear(x_t, packed_w, scale_t, bias_t)
out_b_np = to_np(out_b)
rel_err_b = np.abs(out_b_np - (ref + bias_np)) / (np.abs(ref + bias_np) + 1e-3)
check("nf4 linear bias rel_err<1e-3", float(rel_err_b.max()) < 1e-3)

print("== FP4 pack/unpack roundtrip ==")
vals4 = (np.random.rand(16) * 12.0 - 6.0).astype(np.float32)
w_fp4 = lt.tensor(vals4.tolist()).cast(lt.FP4_E2M1)
packed4 = ops.fp4_pack(w_fp4)
check("fp4 packed size halved", list(packed4.shape) == [8])
unpacked4 = ops.fp4_unpack(packed4, 16)
check("fp4 roundtrip exact", to_np(unpacked4.cast(lt.FP32)).tolist() == to_np(w_fp4.cast(lt.FP32)).tolist())
vals4_odd = (np.random.rand(7) * 12.0 - 6.0).astype(np.float32)
w4_odd = lt.tensor(vals4_odd.tolist()).cast(lt.FP4_E2M1)
packed4_odd = ops.fp4_pack(w4_odd)
check("fp4 odd packed size", list(packed4_odd.shape) == [4])
unp4_odd = ops.fp4_unpack(packed4_odd, 7)
check("fp4 odd roundtrip exact", to_np(unp4_odd.cast(lt.FP32)).tolist() == to_np(w4_odd.cast(lt.FP32)).tolist())

print("== FP4 weight-only linear ==")
w4_np = (np.random.randn(8, 16)).astype(np.float32)
w4_scale_np = np.abs(w4_np).max(axis=1) / 6.0
w4_scaled = w4_np / w4_scale_np[:, None]
w4_t = lt.tensor(w4_scaled.tolist()).cast(lt.FP4_E2M1)
packed4_w = ops.fp4_pack(w4_t.reshape([8 * 16])).reshape([8, 8])
scale4_t = lt.tensor(w4_scale_np.tolist())
out4 = ops.fp4_weight_only_linear(x_t, packed4_w, scale4_t)
out4_np = to_np(out4)
w4_deq = to_np(w4_t.cast(lt.FP32))
ref4 = x_np @ (w4_deq * w4_scale_np[:, None]).T
rel_err4 = np.abs(out4_np - ref4) / (np.abs(ref4) + 1e-3)
check("fp4 linear rel_err<1e-3", float(rel_err4.max()) < 1e-3)

print("== Packed memory halves vs INT8 ==")
orig = lt.tensor(vals.tolist(), dtype=lt.INT8)
check("packed bytes halved", list(packed.shape) == [8] and list(orig.shape) == [16])

print("== unpack validation (QA rework) ==")
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

small = ops.nf4_pack(lt.tensor([1, 2, 0, 15, 5, 5, 3, 2], dtype=lt.INT8))
check("nf4 packed fixture size", list(small.shape) == [4])
check_throws("nf4 oversized numel throws", lambda: ops.nf4_unpack(small, 10))
check_throws("fp4 oversized numel throws", lambda: ops.fp4_unpack(small, 10))
check_throws("nf4 non-INT8 packed throws", lambda: ops.nf4_unpack(lt.tensor([1.0, 2.0, 3.0, 4.0]), 8))
check_throws("fp4 non-INT8 packed throws", lambda: ops.fp4_unpack(lt.tensor([1.0, 2.0, 3.0, 4.0]), 8))
check_throws("nf4 NF4-dtype packed throws", lambda: ops.nf4_unpack(lt.tensor([1.0, 2.0, 3.0, 4.0]).cast(lt.NF4), 8))
check("nf4 exact ceil numel passes", to_int_list(ops.nf4_pack(ops.nf4_unpack(small, 8))) == to_int_list(small))
check("nf4 odd ceil numel passes", to_int_list(ops.nf4_pack(ops.nf4_unpack(small, 7))) == [0x21, -0x10, 0x55, 0x03])

print(f"\nTOTAL: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
