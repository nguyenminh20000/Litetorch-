import math
import sys

import numpy as np
import torch

import litetorch as lt

dev = lt.Device("cuda:0")
THRESH = 1e-6


def vals(n, lo, hi):
    out = []
    s = 12345
    for _ in range(n):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(lo + (hi - lo) * (s / 0x7FFFFFFF))
    return out


def grad_ok(t):
    g = t.grad
    if g is None:
        return False
    for x in g.to_vector():
        if not math.isfinite(x):
            return False
    return True


def main():
    x_np = np.array(vals(8192, -2.0, 2.0), dtype=np.float32).reshape(128, 64)
    x = lt.Tensor.from_vector(x_np.reshape(-1).tolist(), [128, 64], dev, True)
    lt.Ops.sum(lt.Ops.tanh(x)).backward()
    lt.cuda_synchronize()
    if x.grad is None:
        print("DIAG-TANH-INT64 case=deciding grad=None", flush=True)
        print("FAIL", flush=True)
        return 1
    g_lt = np.array(x.grad.to(lt.Device("cpu")).to_vector(), dtype=np.float64)
    xt = torch.tensor(x_np, device="cuda", dtype=torch.float32, requires_grad=True)
    torch.sum(torch.tanh(xt)).backward()
    torch.cuda.synchronize()
    g_t = xt.grad.detach().cpu().numpy().astype(np.float64).reshape(-1)
    diff = float(np.max(np.abs(g_lt - g_t)))
    print(f"DIAG-TANH-INT64 case=deciding max_abs_diff={diff:.6g} threshold={THRESH:.6g}", flush=True)
    ok = math.isfinite(diff) and diff <= THRESH
    xs = lt.Tensor.from_vector(vals(8, -1.0, 1.0), [2, 4], dev, True)
    lt.Ops.sum(lt.Ops.tanh(xs)).backward()
    lt.cuda_synchronize()
    ok_s = grad_ok(xs)
    print(f"DIAG-TANH-INT64 case=op_tanh grad_ok={int(ok_s)}", flush=True)
    ok = ok and ok_s
    m = lt.nn.Tanh()
    m.to(dev)
    xn = lt.Tensor.from_vector(vals(8, -1.0, 1.0), [2, 4], dev, True)
    lt.Ops.sum(m.forward(xn)).backward()
    lt.cuda_synchronize()
    ok_n = grad_ok(xn)
    print(f"DIAG-TANH-INT64 case=nn_tanh grad_ok={int(ok_n)}", flush=True)
    ok = ok and ok_n
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"DIAG-TANH-INT64 FAIL {type(e).__name__}: {e}", flush=True)
        print("FAIL", flush=True)
        sys.exit(1)
