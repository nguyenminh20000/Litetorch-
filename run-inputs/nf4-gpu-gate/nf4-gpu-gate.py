import os
import struct
import sys

TREE = os.environ.get("LITETORCH_TREE")
if not TREE:
    print("NF4 GPU GATE FAIL: LITETORCH_TREE not set")
    sys.exit(2)
sys.path.insert(0, os.path.join(TREE, "python"))

try:
    import litetorch as lt
except Exception as e:
    print("NF4 GPU GATE FAIL: import litetorch raised %r" % (e,))
    sys.exit(1)

NEW_GRID = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]
NEAREST_IN = 0.85
NEAREST_EXPECT = 0.7229568362236023


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def gpu_device_ok(t, name):
    if t.device.type != lt.DeviceType.GPU:
        print("NF4 GPU GATE FAIL: %s device is %s, not GPU (kernel path did not run)" % (name, t.device))
        return False
    return True


def main():
    backend = lt.get_backend_name()
    print("NF4 PREFLIGHT backend=%s cuda_available=%s lt_count=%s" % (
        backend, lt.cuda.is_available(), lt.cuda.device_count()))
    if backend not in ("CUDA", "ROCm"):
        print("NF4 PREFLIGHT FAIL: backend is %s, not a native GPU backend" % backend)
        print("NF4 GPU GATE FAIL")
        return 1
    if not lt.cuda.is_available():
        print("NF4 PREFLIGHT FAIL: lt.cuda.is_available() is False")
        print("NF4 GPU GATE FAIL")
        return 1
    if lt.cuda.device_count() < 1:
        print("NF4 PREFLIGHT FAIL: lt.cuda.device_count() is %s" % lt.cuda.device_count())
        print("NF4 GPU GATE FAIL")
        return 1
    lt.cuda.set_device(0)
    print("NF4 PREFLIGHT PASS")
    print("NF4 OPENCL NOT_COVERED (no Python-addressable OpenCL device on this backend)")
    dev = lt.Device("cuda:0")
    x = lt.Tensor.from_vector(NEW_GRID, [16], dev, False, lt.DataType.FP32)
    if not gpu_device_ok(x, "roundtrip input"):
        print("NF4 GPU GATE FAIL")
        return 1
    x_nf4 = x.cast(lt.DataType.NF4)
    if x_nf4.dtype != lt.DataType.NF4 or not gpu_device_ok(x_nf4, "NF4 tensor"):
        print("NF4 GPU GATE FAIL")
        return 1
    x_back = x_nf4.cast(lt.DataType.FP32)
    if x_back.dtype != lt.DataType.FP32 or not gpu_device_ok(x_back, "roundtrip output"):
        print("NF4 GPU GATE FAIL")
        return 1
    lt.cuda.synchronize()
    vals = x_back.to_vector()
    expected = [f32(v) for v in NEW_GRID]
    bad = [(i, vals[i], expected[i]) for i in range(16) if vals[i] != expected[i]]
    if bad:
        for (i, got, exp) in bad:
            print("NF4 ROUNDTRIP FAIL: idx %d got %r expected %r" % (i, got, exp))
        print("NF4 GPU GATE FAIL")
        return 1
    print("NF4 ROUNDTRIP PASS: 16/16 exact (idx14 = %r; old grid fails this check at 11 indices)" % (vals[14],))
    y = lt.Tensor.from_vector([NEAREST_IN], [1], dev, False, lt.DataType.FP32)
    y_nf4 = y.cast(lt.DataType.NF4)
    if y_nf4.dtype != lt.DataType.NF4 or not gpu_device_ok(y_nf4, "nearest NF4 tensor"):
        print("NF4 GPU GATE FAIL")
        return 1
    y_back = y_nf4.cast(lt.DataType.FP32)
    if y_back.dtype != lt.DataType.FP32 or not gpu_device_ok(y_back, "nearest output"):
        print("NF4 GPU GATE FAIL")
        return 1
    lt.cuda.synchronize()
    got = y_back.to_vector()[0]
    exp = f32(NEAREST_EXPECT)
    if got != exp:
        print("NF4 NEAREST FAIL: nearest(0.85) got %r expected %r" % (got, exp))
        print("NF4 GPU GATE FAIL")
        return 1
    print("NF4 NEAREST PASS: nearest(0.85) = %r (old grid would give 1.0)" % (got,))
    print("NF4 GPU GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
