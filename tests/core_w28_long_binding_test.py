import sys
import litetorch as lt
import torch

fails = 0

def check(name, cond, detail=""):
    global fails
    if cond:
        print("PASS: %s" % name)
    else:
        print("FAIL: %s %s" % (name, detail))
        fails += 1

cpu = lt.Device("cpu")

t = lt.Tensor.from_vector([1.5, -2.5, 3.0], [3], cpu, False)
tl = t.long()
check("long() dtype is INT64", tl.dtype == lt.DataType.INT64, "got %s" % tl.dtype)
check("long() values truncate like torch", tl.to_vector() == [1.0, -2.0, 3.0], "got %s" % tl.to_vector())
tt = torch.tensor([1.5, -2.5, 3.0]).long()
check("torch parity: long() dtype", tt.dtype == torch.int64, "got %s" % tt.dtype)
check("torch parity: long() values", tl.to_vector() == tt.tolist(), "lt %s torch %s" % (tl.to_vector(), tt.tolist()))

big = lt.Tensor.from_vector([2147483648.0], [1], cpu, False, lt.DataType.INT64)
bl = big.long()
check("long() on INT64 keeps INT64", bl.dtype == lt.DataType.INT64, "got %s" % bl.dtype)
check("long() preserves 2**31 exactly", bl.to_vector() == [2147483648.0], "got %s" % bl.to_vector())

bi = big.int()
check("int() stays INT32", bi.dtype == lt.DataType.INT32, "got %s" % bi.dtype)

i32 = lt.Tensor.from_vector([7.0, -3.0], [2], cpu, False, lt.DataType.INT32)
il = i32.long()
check("long() on INT32 promotes to INT64", il.dtype == lt.DataType.INT64, "got %s" % il.dtype)
check("long() on INT32 keeps values", il.to_vector() == [7.0, -3.0], "got %s" % il.to_vector())

f = lt.Tensor.from_vector([2.0, 4.0], [2], cpu, False)
check("float() stays FP32", f.float().dtype == lt.DataType.FP32, "got %s" % f.float().dtype)
check("double() is FP64", f.double().dtype == lt.DataType.FP64, "got %s" % f.double().dtype)
check("half() is FP16", f.half().dtype == lt.DataType.FP16, "got %s" % f.half().dtype)

if fails == 0:
    print("ALL CORE_W28 LONG BINDING TESTS PASSED")
    sys.exit(0)
print("%d CHECKS FAILED" % fails)
sys.exit(1)
