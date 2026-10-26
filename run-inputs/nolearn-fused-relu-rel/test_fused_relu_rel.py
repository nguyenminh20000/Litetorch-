import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import litetorch as lt

dev = lt.Device("gpu:0")
phase = "B-no-cudnn" if os.environ.get("LITETORCH_NO_CUDNN") else "A-cudnn"
torch.manual_seed(0)
np.random.seed(0)
ATOL = 1e-3
RTOL = 1e-4


def to_np(t, shape):
    return np.array(t.to(lt.Device("cpu")).to_vector(), dtype=np.float32).reshape(shape)


def grad_np(t, shape):
    if t.grad is None:
        return None
    return to_np(t.grad, shape)


def run_case(name, layers, x_np, g_np, stride, padding):
    xt = lt.Tensor.from_vector(x_np.reshape(-1).tolist(), list(x_np.shape), dev, True)
    lt_params = []
    for w_np, b_np in layers:
        w = lt.Tensor.from_vector(w_np.reshape(-1).tolist(), list(w_np.shape), dev, True)
        b = lt.Tensor.from_vector(b_np.tolist(), list(b_np.shape), dev, True)
        lt_params.append((w, b))
    gt = lt.Tensor.from_vector(g_np.reshape(-1).tolist(), list(g_np.shape), dev)
    h = xt
    for w, b in lt_params:
        h = lt.Ops.conv2d_relu(h, w, b, stride, padding)
    out = h
    out.backward(gt)
    xt_t = torch.tensor(x_np, requires_grad=True)
    t_params = []
    for w_np, b_np in layers:
        t_params.append((torch.tensor(w_np, requires_grad=True), torch.tensor(b_np, requires_grad=True)))
    h_t = xt_t
    for w_t, b_t in t_params:
        h_t = F.relu(F.conv2d(h_t, w_t, b_t, stride=stride, padding=padding))
    out_t = h_t
    out_t.backward(torch.tensor(g_np))
    diffs = {}
    ratios = {}
    refmax = {}
    def record(key, lt_arr, t_arr):
        d = np.max(np.abs(lt_arr - t_arr))
        m = np.max(np.abs(t_arr))
        thr = ATOL + RTOL * m
        diffs[key] = d
        refmax[key] = m
        ratios[key] = d / thr
    record("fwd", to_np(out, list(out_t.shape)), out_t.detach().numpy())
    lt_gx = grad_np(xt, x_np.shape)
    record("gx", lt_gx, xt_t.grad.numpy())
    for i, ((w, b), (w_t, b_t)) in enumerate(zip(lt_params, t_params)):
        lt_gw = grad_np(w, list(w_t.shape))
        lt_gb = grad_np(b, list(b_t.shape))
        record(f"gw{i}", lt_gw, w_t.grad.numpy())
        record(f"gb{i}", lt_gb, b_t.grad.numpy())
    parts = " ".join(f"{k}={v:.6g}" for k, v in diffs.items())
    rparts = " ".join(f"{k}={v:.4g}" for k, v in ratios.items())
    mparts = " ".join(f"{k}={v:.6g}" for k, v in refmax.items())
    print(f"DIAG-FUSED-RELU phase={phase} case={name} {parts}", flush=True)
    print(f"DIAG-FUSED-RELU-REL phase={phase} case={name} ratio={rparts} refmax={mparts} atol={ATOL:.6g} rtol={RTOL:.6g}", flush=True)
    return max(ratios.values())


N, C_in, H, W = 4, 1, 12, 12
C1, C2 = 4, 6
x_np = np.random.randn(N, C_in, H, W).astype(np.float32)
w1_np = np.random.randn(C1, C_in, 3, 3).astype(np.float32)
b1_np = np.random.randn(C1).astype(np.float32)
w2_np = np.random.randn(C2, C1, 3, 3).astype(np.float32)
b2_np = np.random.randn(C2).astype(np.float32)
g_np = np.random.randn(N, C2, 8, 8).astype(np.float32)
worst = run_case("2layer", [(w1_np, b1_np), (w2_np, b2_np)], x_np, g_np, 1, 0)

N2, C_in2, H2, W2, C_out2 = 128, 3, 32, 32, 32
x2_np = np.random.randn(N2, C_in2, H2, W2).astype(np.float32)
w2c_np = np.random.randn(C_out2, C_in2, 3, 3).astype(np.float32)
b2c_np = np.random.randn(C_out2).astype(np.float32)
g2_np = np.random.randn(N2, C_out2, H2, W2).astype(np.float32)
worst = max(worst, run_case("cifar-conv1", [(w2c_np, b2c_np)], x2_np, g2_np, 1, 1))

print(f"DIAG-FUSED-RELU phase={phase} worst_ratio={worst:.6g} threshold=1", flush=True)
print("PASS" if worst <= 1 else "FAIL", flush=True)
sys.exit(0 if worst <= 1 else 1)
