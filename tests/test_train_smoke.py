import math
import random

import litetorch as lt

LR = 1e-3
CPU = lt.Device("cpu")
DEV = lt.Device("cuda:0") if lt.is_gpu_available() else CPU
_sync = getattr(lt, "cuda_synchronize", lambda: None)


def randn(n, scale=1.0):
    return [random.gauss(0.0, scale) for _ in range(n)]


def scalar(t):
    _sync()
    return t.to(CPU).to_vector()[0]


def vec(t):
    _sync()
    return t.to(CPU).to_vector()


def all_finite(vals):
    return all(math.isfinite(v) for v in vals)


def check_grads(params):
    for i, p in enumerate(params):
        g = p.grad
        if g is None:
            return False, f"param{i} grad is None"
        if not all_finite(vec(g)):
            return False, f"param{i} grad has NaN/Inf"
    return True, ""


def train_loop(params, fwd, xt, yt, steps, loss_fn):
    opt = lt.optim.Adam(params, lr=LR)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        out = fwd(xt)
        loss = loss_fn(out, yt)
        lv = scalar(loss)
        if not math.isfinite(lv):
            return None, "loss NaN/Inf"
        losses.append(lv)
        loss.backward()
        ok, msg = check_grads(params)
        if not ok:
            return None, msg
        opt.step()
    _sync()
    return losses, ""


def test_mlp():
    random.seed(0)
    b, din, dh, dout = 32, 16, 32, 4
    x = randn(b * din, 0.5)
    w_true = randn(din * dout, 0.3)
    y = [sum(x[i * din + k] * w_true[k * dout + j] for k in range(din)) for i in range(b) for j in range(dout)]
    xt = lt.Tensor.from_vector(x, [b, din], DEV)
    yt = lt.Tensor.from_vector(y, [b, dout], DEV)
    model = lt.nn.Sequential([lt.nn.Linear(din, dh), lt.nn.ReLU(), lt.nn.Linear(dh, dout)])
    model.to(DEV)
    losses, msg = train_loop(model.parameters(), model.forward, xt, yt, 10, lt.Ops.mse_loss)
    if losses is None:
        return False, msg
    if not losses[-1] < losses[0]:
        return False, f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    return True, f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_cnn():
    random.seed(1)
    b, c, h, w = 8, 1, 8, 8
    x = randn(b * c * h * w, 0.5)
    w_true = randn(c * h * w * 2, 0.2)
    y = [sum(x[i * c * h * w + k] * w_true[k * 2 + j] for k in range(c * h * w)) for i in range(b) for j in range(2)]
    xt = lt.Tensor.from_vector(x, [b, c, h, w], DEV)
    yt = lt.Tensor.from_vector(y, [b, 2], DEV)
    model = lt.nn.Sequential([
        lt.nn.Conv2d(1, 4, 3, padding=1),
        lt.nn.ReLU(),
        lt.nn.Flatten(),
        lt.nn.Linear(4 * h * w, 2),
    ])
    model.to(DEV)
    losses, msg = train_loop(model.parameters(), model.forward, xt, yt, 10, lt.Ops.mse_loss)
    if losses is None:
        return False, msg
    if not losses[-1] < losses[0]:
        return False, f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    return True, f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_transformer():
    random.seed(2)
    b, t, e = 4, 8, 16
    x = randn(b * t * e, 0.5)
    y = randn(b * t * e, 0.5)
    xt = lt.Tensor.from_vector(x, [b, t, e], DEV)
    yt = lt.Tensor.from_vector(y, [b, t, e], DEV)
    dec = lt.nn.TransformerDecoderLayer(e, 4, 32)
    proj = lt.nn.Linear(e, e)
    dec.to(DEV)
    proj.to(DEV)
    params = dec.parameters() + proj.parameters()
    fwd = lambda inp: proj.forward(dec.forward(inp, inp))
    losses, msg = train_loop(params, fwd, xt, yt, 5, lt.Ops.mse_loss)
    if losses is None:
        return False, msg
    return True, f"loss {losses[0]:.4f} -> {losses[-1]:.4f}, no NaN"


def main():
    print(f"device: {DEV.to_string()}", flush=True)
    results = []
    for name, fn in [("mlp", test_mlp), ("cnn", test_cnn), ("transformer", test_transformer)]:
        try:
            ok, msg = fn()
        except Exception as ex:
            ok, msg = False, f"exception: {ex}"
        results.append((name, ok, msg))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"TRAIN_SMOKE: {n_pass}/{len(results)} passed", flush=True)
    return n_pass == len(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
