import torch
import litetorch as lt
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
dev = lt.Device("gpu:0")
cpu = lt.Device("cpu")

shapes = [
    ((2, 3, 8, 8), (4, 3, 3, 3), 1, 1),
    ((2, 3, 8, 8), (4, 3, 3, 3), 2, 1),
    ((1, 2, 5, 5), (3, 2, 2, 2), 1, 0),
]

def to_lt(arr):
    v = np.ascontiguousarray(arr, dtype=np.float32).ravel().tolist()
    return lt.Tensor.from_vector(v, list(arr.shape), dev, True)

ok = True
for xs, ws, stride, pad in shapes:
    x = torch.randn(*xs, dtype=torch.float32, requires_grad=True)
    w = torch.randn(*ws, dtype=torch.float32, requires_grad=True)
    b = torch.randn(ws[0], dtype=torch.float32, requires_grad=True)
    out = torch.nn.functional.conv2d(x, w, b, stride=stride, padding=pad)
    out.sum().backward()
    gx_t, gw_t = x.grad.numpy(), w.grad.numpy()

    xl, wl = to_lt(x.detach().numpy()), to_lt(w.detach().numpy())
    bl = to_lt(b.detach().numpy())
    outl = lt.Ops.conv2d(xl, wl, bl, stride, pad)
    g = lt.Tensor.from_vector([1.0] * int(np.prod(outl.shape)), list(outl.shape), dev, False)
    outl.backward(g)
    gx_l = np.array(xl.grad.to(cpu).to_vector(), dtype=np.float32).reshape(xs)
    gw_l = np.array(wl.grad.to(cpu).to_vector(), dtype=np.float32).reshape(ws)

    dgx = np.abs(gx_t - gx_l).max()
    dgw = np.abs(gw_t - gw_l).max()
    s = "PASS" if dgx < 1e-3 and dgw < 1e-3 else "FAIL"
    ok = ok and s == "PASS"
    print(f"x{xs} w{ws} s{stride} p{pad}: gx_diff={dgx:.2e} gw_diff={dgw:.2e} {s}")

print("ALL PASS" if ok else "SOME FAILED")
