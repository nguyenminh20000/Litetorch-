#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/hatch/workspace/Litetorch-')
import litetorch as lt

CPU = lt.Device(lt.DeviceType.CPU, 0)
Ops = lt.Ops

def check(cond, name):
    print(("PASS" if cond else "FAIL"), name)
    assert cond, name

def expect_throw(fn, name):
    try:
        fn()
    except Exception:
        print("PASS", name); return
    print("FAIL", name); assert False, name

def T(data, shape, rg=False, dtype=lt.DataType.FP32):
    return lt.Tensor.from_vector(data, shape, CPU, rg, dtype)

# matmul N-D backward shapes
a = T([1.,2.,3.,4.,5.,6.,7.,8.], [2,2,2], True)
b = T([1.,0.,0.,1.], [2,2], True)
y = Ops.matmul(a, b); Ops.sum(y).backward()
check(list(a.grad.shape)==[2,2,2] and list(b.grad.shape)==[2,2], "matmul 3D@2D backward shapes")
check(abs(b.grad.to_vector()[0]-(1+3+5+7))<1e-4, "matmul 3D@2D grad_b broadcast-reduced")

# transpose bounds
expect_throw(lambda: T([1.,2.],[2]).transpose(0,5), "transpose OOB throws")
expect_throw(lambda: T([1.,2.],[2]).transpose(-3,0), "transpose neg OOB throws")

# add_ dtype mismatch
x = T([1.,2.],[2])
expect_throw(lambda: x.add_(T([1,2],[2],dtype=lt.DataType.INT32)), "add_ dtype mismatch throws")

# conv dtype check
expect_throw(lambda: Ops.conv2d(
    T([1.]*16,[1,1,4,4]).cast(lt.DataType.FP16),
    T([1.]*4,[1,1,2,2]).cast(lt.DataType.FP16)), "conv2d FP16 throws")

# pool validation
expect_throw(lambda: Ops.max_pool2d(T([1.]*16,[1,1,4,4]),2,stride=0), "maxpool stride=0 throws")
expect_throw(lambda: Ops.max_pool2d(T([1.]*16,[1,1,4,4]),2,padding=2), "maxpool padding>k/2 throws")
xp = T([float(i) for i in range(16)],[1,1,4,4],True)
o = Ops.max_pool2d(xp,2); Ops.sum(o).backward()
check(o.to_vector()==[5.0,7.0,13.0,15.0], "maxpool2d fwd int indices")
check(abs(sum(xp.grad.to_vector())-4.0)<1e-5, "maxpool2d bwd")

# layernorm validation
expect_throw(lambda: Ops.layer_norm(T([1.]*6,[2,3]),[4]), "layernorm bad normalized_shape throws")
o = Ops.layer_norm(T([1.,2.,3.,4.],[2,2]),[2])
check(abs(o.to_vector()[0]+1.0)<1e-4, "layernorm fwd")

# cat bounds
expect_throw(lambda: Ops.cat([T([1.,2.],[2]),T([3.,4.],[2])],dim=5), "cat OOB dim throws")

# squeeze/unsqueeze non-contiguous
x = T([float(i) for i in range(24)],[2,3,4]).transpose(1,2)
u = x.reshape([2,4,3]).transpose(0,1)
check(u.to_vector()[0]==0.0, "non-contiguous value preserved")

# checkpoint create_graph
x = T([1.,2.,3.,4.],[4],rg=True)
y = lt.checkpoint(lambda v: Ops.mul(v,v), x)
Ops.sum(y).backward(create_graph=True)
check(x.grad.requires_grad, "checkpoint create_graph keeps grad graph")

# w8a8 not in python bindings; skip (covered by C++ audit)

print("ALL BATCH2 TESTS PASSED")
