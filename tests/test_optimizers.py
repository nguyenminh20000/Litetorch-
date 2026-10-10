import math
import random

import litetorch as lt

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


def norm(params):
    return math.sqrt(sum(v * v for p in params for v in vec(p)))


def snap(params):
    return [vec(p)[:] for p in params]


def make_regression(seed=0, b=32, din=16, dout=4):
    random.seed(seed)
    x = randn(b * din, 0.5)
    w_true = randn(din * dout, 0.3)
    y = [sum(x[i * din + k] * w_true[k * dout + j] for k in range(din)) for i in range(b) for j in range(dout)]
    xt = lt.Tensor.from_vector(x, [b, din], DEV)
    yt = lt.Tensor.from_vector(y, [b, dout], DEV)
    return xt, yt


def make_model(seed=0, din=16, dout=4):
    random.seed(seed)
    model = lt.nn.Sequential([lt.nn.Linear(din, dout)])
    model.to(DEV)
    return model


def train_steps(opt, params, fwd, xt, yt, steps):
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = lt.Ops.mse_loss(fwd(xt), yt)
        lv = scalar(loss)
        if not math.isfinite(lv):
            return None, "loss NaN/Inf"
        losses.append(lv)
        loss.backward()
        opt.step()
    return losses, ""


def test_sgd():
    xt, yt = make_regression(0)
    model = make_model(1)
    before = snap(model.parameters())
    opt = lt.optim.SGD(model.parameters(), lr=1e-2)
    losses, msg = train_steps(opt, model.parameters(), model.forward, xt, yt, 5)
    if losses is None:
        return False, msg
    if not losses[-1] < losses[0]:
        return False, f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    if all(a == b for p, q in zip(before, snap(model.parameters())) for a, b in zip(p, q)):
        return False, "params did not change"
    if abs(opt.get_lr() - 1e-2) > 1e-9:
        return False, "get_lr mismatch"
    return True, f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_sgd_momentum():
    xt, yt = make_regression(10)
    m0 = make_model(11)
    m1 = make_model(11)
    s0 = snap(m0.parameters())
    s1 = snap(m1.parameters())
    if not all(a == b for p, q in zip(s0, s1) for a, b in zip(p, q)):
        return False, "init not identical"
    opt0 = lt.optim.SGD(m0.parameters(), lr=1e-2, momentum=0.0)
    opt1 = lt.optim.SGD(m1.parameters(), lr=1e-2, momentum=0.9)
    for _ in range(2):
        for opt, model in ((opt0, m0), (opt1, m1)):
            opt.zero_grad()
            lt.Ops.mse_loss(model.forward(xt), yt).backward()
            opt.step()
    d0 = sum((a - b) ** 2 for p, q in zip(s0, snap(m0.parameters())) for a, b in zip(p, q))
    d1 = sum((a - b) ** 2 for p, q in zip(s1, snap(m1.parameters())) for a, b in zip(p, q))
    if d1 <= d0:
        return False, f"momentum did not amplify update: {d0:.6f} vs {d1:.6f}"
    return True, f"update norm^2 no-mom {d0:.6f} mom {d1:.6f}"


def test_sgd_weight_decay():
    xt, yt = make_regression(20)
    m0 = make_model(21)
    m1 = make_model(21)
    opt0 = lt.optim.SGD(m0.parameters(), lr=1e-2, weight_decay=0.0)
    opt1 = lt.optim.SGD(m1.parameters(), lr=1e-2, weight_decay=0.5)
    for _ in range(5):
        for opt, model in ((opt0, m0), (opt1, m1)):
            opt.zero_grad()
            lt.Ops.mse_loss(model.forward(xt), yt).backward()
            opt.step()
    n0 = norm(m0.parameters())
    n1 = norm(m1.parameters())
    if not n1 < n0:
        return False, f"weight decay not shrinking: {n0:.6f} vs {n1:.6f}"
    return True, f"norm no-wd {n0:.6f} wd {n1:.6f}"


def test_adam():
    xt, yt = make_regression(30)
    model = make_model(31)
    before = snap(model.parameters())
    opt = lt.optim.Adam(model.parameters(), lr=1e-3)
    losses, msg = train_steps(opt, model.parameters(), model.forward, xt, yt, 5)
    if losses is None:
        return False, msg
    if not losses[-1] < losses[0]:
        return False, f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    if all(a == b for p, q in zip(before, snap(model.parameters())) for a, b in zip(p, q)):
        return False, "params did not change"
    return True, f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_adamw_weight_decay():
    xt, yt = make_regression(40)
    m0 = make_model(41)
    m1 = make_model(41)
    opt0 = lt.optim.Adam(m0.parameters(), lr=1e-3, weight_decay=0.0)
    opt1 = lt.optim.AdamW(m1.parameters(), lr=1e-3, weight_decay=0.1)
    for _ in range(5):
        for opt, model in ((opt0, m0), (opt1, m1)):
            opt.zero_grad()
            lt.Ops.mse_loss(model.forward(xt), yt).backward()
            opt.step()
    n0 = norm(m0.parameters())
    n1 = norm(m1.parameters())
    if not n1 < n0:
        return False, f"adamw decay not shrinking: {n0:.6f} vs {n1:.6f}"
    return True, f"norm adam {n0:.6f} adamw {n1:.6f}"


def test_adamw():
    xt, yt = make_regression(50)
    model = make_model(51)
    opt = lt.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    losses, msg = train_steps(opt, model.parameters(), model.forward, xt, yt, 5)
    if losses is None:
        return False, msg
    if not losses[-1] < losses[0]:
        return False, f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    return True, f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_rmsprop():
    xt, yt = make_regression(60)
    model = make_model(61)
    opt = lt.optim.RMSprop(model.parameters(), lr=1e-2)
    losses, msg = train_steps(opt, model.parameters(), model.forward, xt, yt, 5)
    if losses is None:
        return False, msg
    if not losses[-1] < losses[0]:
        return False, f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    return True, f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_steplr():
    model = make_model(70)
    opt = lt.optim.SGD(model.parameters(), lr=0.1)
    sched = lt.optim.StepLR(opt, 2, 0.1)
    if abs(opt.get_lr() - 0.1) > 1e-9:
        return False, "initial lr wrong"
    sched.step()
    if abs(opt.get_lr() - 0.1) > 1e-9:
        return False, f"lr decayed too early: {opt.get_lr()}"
    sched.step()
    if abs(opt.get_lr() - 0.01) > 1e-7:
        return False, f"lr not decayed after step_size: {opt.get_lr()}"
    sched.step()
    if abs(opt.get_lr() - 0.01) > 1e-7:
        return False, f"lr decayed off-schedule: {opt.get_lr()}"
    return True, f"lr schedule 0.1 -> 0.1 -> {opt.get_lr()}"


def test_cosine():
    model = make_model(80)
    opt = lt.optim.SGD(model.parameters(), lr=0.1)
    sched = lt.optim.CosineAnnealingLR(opt, 10, 0.0)
    for _ in range(5):
        sched.step()
    lr5 = opt.get_lr()
    if abs(lr5 - 0.05) > 1e-4:
        return False, f"lr at t=5 wrong: {lr5}"
    for _ in range(5):
        sched.step()
    lr10 = opt.get_lr()
    if abs(lr10) > 1e-4:
        return False, f"lr at t=T_max not eta_min: {lr10}"
    if not lr10 < lr5 < 0.1:
        return False, "lr not monotonically decreasing"
    return True, f"lr t=5 {lr5:.5f} t=10 {lr10:.6f}"


def test_set_lr():
    model = make_model(90)
    opt = lt.optim.Adam(model.parameters(), lr=1e-3)
    opt.set_lr(5e-4)
    if abs(opt.get_lr() - 5e-4) > 1e-9:
        return False, f"set_lr failed: {opt.get_lr()}"
    return True, "set_lr ok"


def test_grad_scaler():
    scaler = lt.optim.GradScaler(4.0, 2.0, 0.5, 2)
    xt, yt = make_regression(100)
    model = make_model(101)
    opt = lt.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = lt.Ops.mse_loss(model.forward(xt), yt)
    base = scalar(loss)
    scaled = scaler.scale_loss(loss)
    if abs(scalar(scaled) - base * 4.0) > 1e-3 * max(1.0, abs(base)):
        return False, f"scale_loss wrong: {scalar(scaled)} vs {base * 4.0}"
    before = snap(model.parameters())
    scaled.backward()
    scaler.step(opt)
    if all(a == b for p, q in zip(before, snap(model.parameters())) for a, b in zip(p, q)):
        return False, "params did not change after scaler.step"
    scaler.update()
    s0 = scaler.scale
    scaler.update()
    if abs(scaler.scale - s0 * 2.0) > 1e-6:
        return False, f"scale did not grow after interval: {scaler.scale}"
    return True, f"scale 4.0 -> {scaler.scale}"


def test_zero3():
    xt, yt = make_regression(110)
    model = make_model(111)
    before = snap(model.parameters())
    opt = lt.optim.ZeRO3Optimizer(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = lt.Ops.mse_loss(model.forward(xt), yt)
    lv = scalar(loss)
    if not math.isfinite(lv):
        return False, "loss NaN/Inf"
    loss.backward()
    opt.step()
    if all(a == b for p, q in zip(before, snap(model.parameters())) for a, b in zip(p, q)):
        return False, "params did not change"
    return True, f"loss {lv:.4f}, params updated"


def test_adamw8bit():
    xt, yt = make_regression(120)
    model = make_model(121)
    before = snap(model.parameters())
    opt = lt.optim.AdamW8bit(model.parameters(), lr=1e-3)
    losses, msg = train_steps(opt, model.parameters(), model.forward, xt, yt, 5)
    if losses is None:
        return False, msg
    if not losses[-1] < losses[0]:
        return False, f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    if all(a == b for p, q in zip(before, snap(model.parameters())) for a, b in zip(p, q)):
        return False, "params did not change"
    return True, f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_adamwfp8():
    xt, yt = make_regression(130)
    model = make_model(131)
    before = snap(model.parameters())
    opt = lt.optim.AdamWFP8(model.parameters(), lr=1e-3)
    losses, msg = train_steps(opt, model.parameters(), model.forward, xt, yt, 5)
    if losses is None:
        return False, msg
    if not losses[-1] < losses[0]:
        return False, f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    if all(a == b for p, q in zip(before, snap(model.parameters())) for a, b in zip(p, q)):
        return False, "params did not change"
    return True, f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"


def main():
    print(f"device: {DEV.to_string()}", flush=True)
    cases = [
        ("sgd", test_sgd),
        ("sgd_momentum", test_sgd_momentum),
        ("sgd_weight_decay", test_sgd_weight_decay),
        ("adam", test_adam),
        ("adamw", test_adamw),
        ("adamw_weight_decay", test_adamw_weight_decay),
        ("rmsprop", test_rmsprop),
        ("steplr", test_steplr),
        ("cosine", test_cosine),
        ("set_lr", test_set_lr),
        ("grad_scaler", test_grad_scaler),
        ("zero3", test_zero3),
        ("adamw8bit", test_adamw8bit),
        ("adamwfp8", test_adamwfp8),
    ]
    passed = 0
    for name, fn in cases:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"exception: {type(e).__name__}: {e}"
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
        if ok:
            passed += 1
    print(f"TOTAL: {passed}/{len(cases)} passed", flush=True)
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
