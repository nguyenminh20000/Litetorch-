import torch
import litetorch as lt
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

shapes = [
    ((2, 3, 8, 8), (4, 3, 3, 3), 1, 1),
    ((2, 3, 8, 8), (4, 3, 3, 3), 2, 1),
    ((1, 2, 5, 5), (3, 2, 2, 2), 1, 0),
]

ok = True
for xs, ws, stride, pad in shapes:
    x = torch.randn(*xs, dtype=torch.float32, requires_grad=True)
    w = torch.randn(*ws, dtype=torch.float32, requires_grad=True)
    b = torch.randn(ws[0], dtype=torch.float32, requires_grad=True)
    out = torch.nn.functional.conv2d(x, w, b, stride=stride, padding=pad)
    out.sum().backward()
    gx_t, gw_t = x.grad.clone(), w.grad.clone()

    xl = lt.tensor(np.ascontiguousarray(x.detach().numpy()), requires_grad=True).cuda()
    wl = lt.tensor(np.ascontiguousarray(w.detach().numpy()), requires_grad=True).cuda()
    bl = lt.tensor(np.ascontiguousarray(b.detach().numpy()), requires_grad=True).cuda()
    outl = lt.Ops.conv2d(xl, wl, bl, stride=stride, padding=pad)
    outl.backward(lt.ones_like(outl))
    gx_l = np.array(xl.grad.to_vector()).reshape(xs)
    gw_l = np.array(wl.grad.to_vector()).reshape(ws)

    dgx = np.abs(gx_t.numpy() - gx_l).max()
    dgw = np.abs(gw_t.numpy() - gw_l).max()
    status = "PASS" if dgx < 1e-3 and dgw < 1e-3 else "FAIL"
    if status == "FAIL":
        ok = False
    print(f"x{xs} w{ws} s{stride} p{pad}: gx_diff={dgx:.2e} gw_diff={dgw:.2e} {status}")

print("ALL PASS" if ok else "SOME FAILED")
