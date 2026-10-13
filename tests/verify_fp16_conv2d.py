import litetorch as lt
import torch
import torch.nn.functional as F
import numpy as np
import time

dev = lt.Device("gpu:0")
cpu = lt.Device("cpu")
torch.manual_seed(0)
np.random.seed(0)

N, C_in, H, W = 4, 8, 16, 16
C_out, KH, KW = 16, 3, 3
H_out, W_out = H - KH + 1, W - KW + 1
FP16 = lt.DataType.FP16

def lt_np(t, shape, dtype=np.float32):
    return np.array(t.to(cpu).tolist(), dtype=dtype).reshape(shape)

x_np = (np.random.randn(N, C_in, H, W) * 0.5).astype(np.float32)
w_np = (np.random.randn(C_out, C_in, KH, KW) * 0.5).astype(np.float32)
b_np = (np.random.randn(C_out) * 0.5).astype(np.float32)
g_np = (np.random.randn(N, C_out, H_out, W_out) * 0.5).astype(np.float32)

def lt_half(arr, shape, dev, req=False):
    t32 = lt.Tensor.from_vector(arr.reshape(-1).tolist(), shape, dev, req)
    return t32.to(FP16)

print("=== 1. FP16 forward vs torch fp16 ===")
xt = lt_half(x_np, [N, C_in, H, W], dev)
wt = lt_half(w_np, [C_out, C_in, KH, KW], dev)
bt = lt_half(b_np, [C_out], dev)
out = lt.Ops.conv2d(xt, wt, bt, 1, 1)
assert out.dtype == FP16, f"out dtype {out.dtype} != FP16"
xt_t = torch.tensor(x_np, dtype=torch.float16, device="cuda")
wt_t = torch.tensor(w_np, dtype=torch.float16, device="cuda")
bt_t = torch.tensor(b_np, dtype=torch.float16, device="cuda")
ref = F.conv2d(xt_t, wt_t, bt_t, padding=1).float().cpu().numpy()
got = lt_np(out, [N, C_out, H_out, W_out])
d = np.max(np.abs(got - ref))
print("fwd max diff:", d)
assert d < 1e-2, "FP16 forward FAIL"

print("=== 2. FP16 backward vs torch fp16 ===")
xt = lt_half(x_np, [N, C_in, H, W], dev)
wt = lt_half(w_np, [C_out, C_in, KH, KW], dev, True)
bt = lt_half(b_np, [C_out], dev, True)
out = lt.Ops.conv2d(xt, wt, bt, 1, 1)
gt = lt_half(g_np, [N, C_out, H_out, W_out], dev)
out.backward(gt)
lt_gw = lt_np(wt.grad, [C_out, C_in, KH, KW])
lt_gb = lt_np(bt.grad, [C_out])
xt_t = torch.tensor(x_np, dtype=torch.float16, device="cuda")
wt_t = torch.tensor(w_np, dtype=torch.float16, device="cuda", requires_grad=True)
bt_t = torch.tensor(b_np, dtype=torch.float16, device="cuda", requires_grad=True)
out_t = F.conv2d(xt_t, wt_t, bt_t, padding=1)
out_t.backward(torch.tensor(g_np, dtype=torch.float16, device="cuda"))
pt_gw = wt_t.grad.float().cpu().numpy()
pt_gb = bt_t.grad.float().cpu().numpy()
dgw = np.max(np.abs(lt_gw - pt_gw))
dgb = np.max(np.abs(lt_gb - pt_gb))
print("gw max diff:", dgw, " gb max diff:", dgb)
assert dgw < 2e-2 and dgb < 2e-2, "FP16 backward FAIL"

print("=== 3. conv2d_relu FP16 ===")
xt = lt_half(x_np, [N, C_in, H, W], dev)
wt = lt_half(w_np, [C_out, C_in, KH, KW], dev)
bt = lt_half(b_np, [C_out], dev)
out = lt.Ops.conv2d_relu(xt, wt, bt, 1, 1)
ref = F.relu(F.conv2d(xt_t, wt_t, bt_t, padding=1)).float().cpu().numpy()
got = lt_np(out, [N, C_out, H_out, W_out])
d = np.max(np.abs(got - ref))
print("relu fwd max diff:", d)
assert d < 1e-2, "FP16 conv2d_relu FAIL"

print("=== 4. autocast: FP32 in -> FP16 out ===")
xt32 = lt.Tensor.from_vector(x_np.reshape(-1).tolist(), [N, C_in, H, W], dev)
wt32 = lt.Tensor.from_vector(w_np.reshape(-1).tolist(), [C_out, C_in, KH, KW], dev)
bt32 = lt.Tensor.from_vector(b_np.tolist(), [C_out], dev)
guard = lt.amp.AutocastGuard(True, FP16)
out_ac = lt.Ops.conv2d(xt32, wt32, bt32, 1, 1)
del guard
assert out_ac.dtype == FP16, f"autocast out dtype {out_ac.dtype} != FP16"
got = lt_np(out_ac, [N, C_out, H_out, W_out])
ref32 = F.conv2d(torch.tensor(x_np, device="cuda"), torch.tensor(w_np, device="cuda"),
                 torch.tensor(b_np, device="cuda"), padding=1).cpu().numpy()
d = np.max(np.abs(got - ref32))
print("autocast max diff vs fp32 ref:", d)
assert d < 5e-2, "autocast FAIL"

print("=== 5. CPU FP16 raises clean error ===")
try:
    xc = lt_half(x_np, [N, C_in, H, W], cpu)
    wc = lt_half(w_np, [C_out, C_in, KH, KW], cpu)
    lt.Ops.conv2d(xc, wc, None, 1, 1)
    raise AssertionError("CPU FP16 should have raised")
except RuntimeError as e:
    assert "not supported on CPU" in str(e), f"wrong error: {e}"
    print("CPU error OK:", e)

print("=== 6. FP32 regression (unchanged path) ===")
xt = lt.Tensor.from_vector(x_np.reshape(-1).tolist(), [N, C_in, H, W], dev)
wt = lt.Tensor.from_vector(w_np.reshape(-1).tolist(), [C_out, C_in, KH, KW], dev)
bt = lt.Tensor.from_vector(b_np.tolist(), [C_out], dev)
out = lt.Ops.conv2d(xt, wt, bt, 1, 1)
assert out.dtype == lt.DataType.FP32
got = lt_np(out, [N, C_out, H_out, W_out])
d = np.max(np.abs(got - ref32))
print("fp32 max diff:", d)
assert d < 1e-4, "FP32 regression FAIL"

print("=== 7. benchmark FP16 vs FP32 ===")
BN, BC, BH, BW = 32, 64, 32, 32
BCo, BK = 128, 3
bho, bwo = BH - BK + 1, BW - BK + 1
for dt, tag in [(lt.DataType.FP32, "fp32"), (FP16, "fp16")]:
    xb = lt.Tensor.randn([BN, BC, BH, BW], dev, False, dt)
    wb = lt.Tensor.randn([BCo, BC, BK, BK], dev, False, dt)
    bb = lt.Tensor.randn([BCo], dev, False, dt)
    lt.Ops.conv2d(xb, wb, bb, 1, 1)
    lt.cuda_synchronize()
    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        lt.Ops.conv2d(xb, wb, bb, 1, 1)
    lt.cuda_synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    flops = 2 * BN * BCo * bho * bwo * BC * BK * BK
    print(f"{tag}: {ms:.2f} ms/iter, {flops / ms / 1e9:.2f} TFLOPS")

print("ALL FP16 CONV2D CHECKS PASSED")
