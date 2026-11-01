import os
import sys

TREE = os.environ.get("LITETORCH_TREE")
if not TREE:
    print("BCAST0 GPU GATE FAIL: LITETORCH_TREE not set")
    sys.exit(2)
sys.path.insert(0, os.path.join(TREE, "python"))

try:
    import litetorch as lt
except Exception as e:
    print("BCAST0 GPU GATE FAIL: import litetorch raised %r" % (e,))
    sys.exit(1)


def run_case(name, a_shape, b_shape, expect_shape):
    dev = lt.Device("cuda:0")
    a = lt.Tensor.ones(a_shape, dev, False, lt.DataType.FP16)
    b = lt.Tensor.ones(b_shape, dev, False, lt.DataType.FP16)
    try:
        out = lt.Ops.mul(a, b)
    except Exception as e:
        print("BCAST0 CASE %s FAIL: threw %r" % (name, e))
        return None
    got_shape = list(out.shape)
    if got_shape != list(expect_shape):
        print("BCAST0 CASE %s FAIL: shape %s != %s" % (name, got_shape, list(expect_shape)))
        return None
    if out.dtype != lt.DataType.FP16:
        print("BCAST0 CASE %s FAIL: dtype %s != FP16" % (name, out.dtype))
        return None
    print("BCAST0 CASE %s PASS: shape %s dtype FP16" % (name, got_shape))
    return out


def main():
    backend = lt.get_backend_name()
    print("BCAST0 PREFLIGHT backend=%s cuda_available=%s lt_count=%s" % (
        backend, lt.cuda.is_available(), lt.cuda.device_count()))
    if backend not in ("CUDA", "ROCm"):
        print("BCAST0 PREFLIGHT FAIL: backend is %s, not a native GPU backend" % backend)
        print("BCAST0 GPU GATE FAIL")
        return 1
    if not lt.cuda.is_available():
        print("BCAST0 PREFLIGHT FAIL: lt.cuda.is_available() is False")
        print("BCAST0 GPU GATE FAIL")
        return 1
    if lt.cuda.device_count() != 2:
        print("BCAST0 PREFLIGHT FAIL: lt.cuda.device_count() is %s, expected 2" % lt.cuda.device_count())
        print("BCAST0 GPU GATE FAIL")
        return 1
    lt.cuda.set_device(0)
    print("BCAST0 PREFLIGHT PASS")
    zero_out = run_case("zero-dim", [0], [0, 1, 0], [0, 1, 0])
    sanity_out = run_case("sanity", [2, 3], [3], [2, 3])
    if zero_out is None or sanity_out is None:
        print("BCAST0 GPU GATE FAIL")
        return 1
    vals = sanity_out.to_vector()
    if len(vals) != 6 or any(v != 1.0 for v in vals):
        print("BCAST0 VALUES FAIL: %s" % (vals,))
        print("BCAST0 GPU GATE FAIL")
        return 1
    print("BCAST0 VALUES PASS: all 1.0")
    print("BCAST0 GPU GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
