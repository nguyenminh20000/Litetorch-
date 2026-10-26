import sys

import numpy as np
import torch
import torch.nn as tnn

import litetorch as lt

print("DIAG-SCRIPT diag-opencl-fillzero", flush=True)
print(f"DIAG-BACKEND native_cuda_available={lt.cuda.is_available()}", flush=True)
if lt.cuda.is_available():
    print("DIAG-FAIL native CUDA backend is present; this run must be pure-OpenCL (build with LITETORCH_NO_NATIVE_GPU=1)", flush=True)
    sys.exit(2)

dev = lt.Device(lt.DeviceType.GPU, 0)
cpu = lt.Device("cpu")
rng = np.random.default_rng(0)
TOL = 1e-5
failures = []


def lt_np(t):
    return np.array(t.to(cpu).to_vector(), dtype=np.float64).reshape(list(t.shape))


def check(name, lt_t, torch_t):
    a = lt_np(lt_t)
    b = torch_t.detach().cpu().numpy().astype(np.float64)
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    ok = d <= TOL
    print(f"DIAG-OPENCL-FILLZERO {name} max_abs_diff={d:.6g} {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        failures.append(name)


def lt_tv(arr, requires_grad=False):
    return lt.Tensor.from_vector(arr.reshape(-1).tolist(), list(arr.shape), dev, requires_grad)


def grad_of(out, inp):
    s = out.sum()
    return lt.autograd.grad([s], [inp], retain_graph=True)[0]


x_np = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
x_lt = lt_tv(x_np, requires_grad=True)
x_t = torch.tensor(x_np, device="cuda", requires_grad=True)
z_lt = lt.Tensor.zeros([128, 64], dev)
check("zeros-128x64", z_lt, torch.zeros((128, 64), device="cuda"))
z2_lt = lt.Tensor.zeros([100000], dev)
check("zeros-100k", z2_lt, torch.zeros((100000,), device="cuda"))

p_lt = lt.Ops.max_pool2d(x_lt, 2)
p_t = tnn.functional.max_pool2d(x_t, 2)
check("maxpool-fwd", p_lt, p_t)
check("maxpool-bwd", grad_of(p_lt, x_lt), torch.autograd.grad(p_t.sum(), x_t, retain_graph=True)[0])

w_np = rng.standard_normal((4, 3, 3, 3)).astype(np.float32) * 0.1
b_np = rng.standard_normal((4,)).astype(np.float32) * 0.1
w_lt = lt_tv(w_np, requires_grad=True)
b_lt = lt_tv(b_np, requires_grad=True)
w_t = torch.tensor(w_np, device="cuda", requires_grad=True)
b_t = torch.tensor(b_np, device="cuda", requires_grad=True)
c_lt = lt.Ops.conv2d(x_lt, w_lt, b_lt, 1, 1)
c_t = tnn.functional.conv2d(x_t, w_t, b_t, 1, 1)
check("conv-fwd", c_lt, c_t)
g_lt = lt.autograd.grad([c_lt.sum()], [x_lt, w_lt, b_lt], retain_graph=True)
g_t = torch.autograd.grad(c_t.sum(), [x_t, w_t, b_t], retain_graph=True)
check("conv-bwd-gx", g_lt[0], g_t[0])
check("conv-bwd-gw", g_lt[1], g_t[1])
check("conv-bwd-gb", g_lt[2], g_t[2])

n_np = rng.standard_normal((2, 4, 8)).astype(np.float32)
nw_np = rng.standard_normal((8,)).astype(np.float32)
nb_np = rng.standard_normal((8,)).astype(np.float32)
n_lt = lt_tv(n_np, requires_grad=True)
nw_lt = lt_tv(nw_np, requires_grad=True)
nb_lt = lt_tv(nb_np, requires_grad=True)
n_t = torch.tensor(n_np, device="cuda", requires_grad=True)
nw_t = torch.tensor(nw_np, device="cuda", requires_grad=True)
nb_t = torch.tensor(nb_np, device="cuda", requires_grad=True)
l_lt = lt.Ops.layer_norm(n_lt, [8], nw_lt, nb_lt, 1e-5)
l_t = tnn.functional.layer_norm(n_t, [8], nw_t, nb_t, 1e-5)
check("layernorm-fwd", l_lt, l_t)
lg_lt = lt.autograd.grad([l_lt.sum()], [n_lt, nw_lt, nb_lt], retain_graph=True)
lg_t = torch.autograd.grad(l_t.sum(), [n_t, nw_t, nb_t], retain_graph=True)
check("layernorm-bwd-gx", lg_lt[0], lg_t[0])
check("layernorm-bwd-gw", lg_lt[1], lg_t[1])
check("layernorm-bwd-gb", lg_lt[2], lg_t[2])

if failures:
    print(f"DIAG-OPENCL-FILLZERO RESULT FAIL failed={failures}", flush=True)
    sys.exit(1)
print("DIAG-OPENCL-FILLZERO RESULT PASS", flush=True)
