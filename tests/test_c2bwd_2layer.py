import litetorch as lt
import torch
import torch.nn.functional as F
import numpy as np

dev = lt.Device("gpu:0")
torch.manual_seed(0)
np.random.seed(0)

N, C_in, H, W = 4, 1, 12, 12
C1, C2 = 4, 6

x_np = np.random.randn(N, C_in, H, W).astype(np.float32)
w1_np = np.random.randn(C1, C_in, 3, 3).astype(np.float32)
b1_np = np.random.randn(C1).astype(np.float32)
w2_np = np.random.randn(C2, C1, 3, 3).astype(np.float32)
b2_np = np.random.randn(C2).astype(np.float32)
g_np = np.random.randn(N, C2, 8, 8).astype(np.float32)

xt = lt.Tensor.from_vector(x_np.reshape(-1).tolist(), [N, C_in, H, W], dev)
w1 = lt.Tensor.from_vector(w1_np.reshape(-1).tolist(), [C1, C_in, 3, 3], dev, True)
b1 = lt.Tensor.from_vector(b1_np.tolist(), [C1], dev, True)
w2 = lt.Tensor.from_vector(w2_np.reshape(-1).tolist(), [C2, C1, 3, 3], dev, True)
b2 = lt.Tensor.from_vector(b2_np.tolist(), [C2], dev, True)
gt = lt.Tensor.from_vector(g_np.reshape(-1).tolist(), [N, C2, 8, 8], dev)

h = lt.Ops.relu(lt.Ops.conv2d(xt, w1, b1, 1, 0))
out = lt.Ops.conv2d(h, w2, b2, 1, 0)
out.backward(gt)

def to_np(t, shape):
    return np.array(t.to(lt.Device("cpu")).to_vector(), dtype=np.float32).reshape(shape)

lt_gw1 = to_np(w1.grad, w1_np.shape)
lt_gw2 = to_np(w2.grad, w2_np.shape)

xt_t = torch.tensor(x_np, requires_grad=True)
w1_t = torch.tensor(w1_np, requires_grad=True)
b1_t = torch.tensor(b1_np, requires_grad=True)
w2_t = torch.tensor(w2_np, requires_grad=True)
b2_t = torch.tensor(b2_np, requires_grad=True)
h_t = F.relu(F.conv2d(xt_t, w1_t, b1_t))
out_t = F.conv2d(h_t, w2_t, b2_t)
out_t.backward(torch.tensor(g_np))

print("gw1 max diff:", np.max(np.abs(lt_gw1 - w1_t.grad.numpy())))
print("gw2 max diff:", np.max(np.abs(lt_gw2 - w2_t.grad.numpy())))
d1 = np.max(np.abs(lt_gw1 - w1_t.grad.numpy()))
d2 = np.max(np.abs(lt_gw2 - w2_t.grad.numpy()))
print("PASS" if d1 < 1e-3 and d2 < 1e-3 else "FAIL")
