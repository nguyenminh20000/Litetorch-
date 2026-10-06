import litetorch as lt
import torch
import torch.nn.functional as F
import numpy as np

dev = lt.Device("gpu:0")
torch.manual_seed(0)
np.random.seed(0)

N, C_in, H, W = 4, 3, 8, 8
C_out, KH, KW = 5, 3, 3

x_np = np.random.randn(N, C_in, H, W).astype(np.float32)
w_np = np.random.randn(C_out, C_in, KH, KW).astype(np.float32)
b_np = np.random.randn(C_out).astype(np.float32)
g_np = np.random.randn(N, C_out, 6, 6).astype(np.float32)

xt = lt.Tensor.from_vector(x_np.reshape(-1).tolist(), [N, C_in, H, W], dev)
wt = lt.Tensor.from_vector(w_np.reshape(-1).tolist(), [C_out, C_in, KH, KW], dev, True)
bt = lt.Tensor.from_vector(b_np.tolist(), [C_out], dev, True)
gt = lt.Tensor.from_vector(g_np.reshape(-1).tolist(), [N, C_out, 6, 6], dev)

out = lt.Ops.conv2d(xt, wt, bt, 1, 0)
out.backward(gt)

print("wt.grad is None:", wt.grad is None)
print("bt.grad is None:", bt.grad is None)
if wt.grad is not None:
    def to_np(t, shape):
        v = t.to(lt.Device("cpu")).tolist()
        return np.array(v, dtype=np.float32).reshape(shape)
    lt_gw = to_np(wt.grad, w_np.shape)
    lt_gb = to_np(bt.grad, b_np.shape)
    xt_t = torch.tensor(x_np, requires_grad=True)
    wt_t = torch.tensor(w_np, requires_grad=True)
    bt_t = torch.tensor(b_np, requires_grad=True)
    out_t = F.conv2d(xt_t, wt_t, bt_t)
    out_t.backward(torch.tensor(g_np))
    pt_gw = wt_t.grad.numpy()
    pt_gb = bt_t.grad.numpy()
    print("gw max diff:", np.max(np.abs(lt_gw - pt_gw)))
    print("gb max diff:", np.max(np.abs(lt_gb - pt_gb)))
    print("PASS" if np.max(np.abs(lt_gw - pt_gw)) < 1e-3 else "FAIL")
else:
    print("FAIL: grad is None")
