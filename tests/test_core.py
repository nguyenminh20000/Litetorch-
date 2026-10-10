import math
import os
import random
import tempfile

import litetorch as lt

CPU = lt.Device("cpu")
DEV = lt.Device("cuda:0") if lt.is_gpu_available() else CPU
_sync = getattr(lt, "cuda_synchronize", lambda: None)


def randn(n, scale=1.0):
    return [random.gauss(0.0, scale) for _ in range(n)]


def vec(t):
    _sync()
    return t.to(CPU).to_vector()


def all_finite(vals):
    return all(math.isfinite(v) for v in vals)


def grad_ok(params):
    for i, p in enumerate(params):
        g = p.grad
        if g is None:
            return False, f"param{i} grad is None"
        if not all_finite(vec(g)):
            return False, f"param{i} grad has NaN/Inf"
    return True, ""


def t_reshape_view():
    a = lt.Tensor.from_vector(randn(12), [12], DEV, True)
    b = a.reshape([3, 4])
    c = b.view([12])
    if list(c.shape) != [12]:
        return False, f"view shape {c.shape}"
    loss = lt.Ops.sum(c * c)
    loss.backward()
    if a.grad is None:
        return False, "a.grad None"
    gv = vec(a.grad)
    if not all_finite(gv):
        return False, "grad NaN/Inf"
    return True, ""


def t_transpose():
    a = lt.Tensor.from_vector(randn(6), [2, 3], DEV, True)
    b = a.transpose(0, 1)
    if list(b.shape) != [3, 2]:
        return False, f"transpose shape {b.shape}"
    loss = lt.Ops.sum(b * b)
    loss.backward()
    if a.grad is None:
        return False, "a.grad None"
    ga = vec(a.grad)
    av = vec(a.to(CPU))
    for x, g in zip(av, ga):
        if abs(g - 2.0 * x) > 1e-3:
            return False, f"transpose grad wrong: {g} vs {2.0 * x}"
    return True, ""


def t_cat():
    a = lt.Tensor.from_vector(randn(6), [2, 3], DEV, True)
    b = lt.Tensor.from_vector(randn(6), [2, 3], DEV, True)
    c = lt.Ops.cat([a, b], 0)
    if list(c.shape) != [4, 3]:
        return False, f"cat shape {c.shape}"
    loss = lt.Ops.sum(c)
    loss.backward()
    ok, msg = grad_ok([a, b])
    if not ok:
        return False, msg
    ga = vec(a.grad)
    if any(abs(g - 1.0) > 1e-4 for g in ga):
        return False, "cat grad_a wrong"
    return True, ""


def t_squeeze_unsqueeze():
    a = lt.Tensor.from_vector(randn(6), [1, 2, 3], DEV, True)
    b = lt.Ops.squeeze(a, 0)
    if list(b.shape) != [2, 3]:
        return False, f"squeeze shape {b.shape}"
    c = lt.Ops.unsqueeze(b, 0)
    if list(c.shape) != [1, 2, 3]:
        return False, f"unsqueeze shape {c.shape}"
    loss = lt.Ops.sum(c)
    loss.backward()
    if a.grad is None:
        return False, "a.grad None"
    return True, ""


def t_broadcast():
    a = lt.Tensor.from_vector(randn(6), [2, 3], DEV, True)
    b = lt.Tensor.from_vector(randn(3), [3], DEV, True)
    c = lt.Ops.add(a, b)
    loss = lt.Ops.sum(c)
    loss.backward()
    if a.grad is None or b.grad is None:
        return False, "grad None"
    gb = vec(b.grad)
    if any(abs(g - 2.0) > 1e-4 for g in gb):
        return False, f"broadcast grad_b wrong: {gb}"
    d = lt.Tensor.from_vector(randn(6), [2, 3], DEV, True)
    e = lt.Tensor.from_vector(randn(3), [3], DEV, True)
    f = lt.Ops.mul(d, e)
    lt.Ops.sum(f).backward()
    if e.grad is None:
        return False, "mul broadcast grad None"
    return True, ""


def t_noncontig():
    a = lt.Tensor.from_vector(randn(12), [3, 4], DEV, True)
    t = a.transpose(0, 1)
    if t.is_contiguous():
        return False, "expected non-contiguous"
    c = t.contiguous()
    if not c.is_contiguous():
        return False, "contiguous() failed"
    r = lt.Ops.relu(t)
    loss = lt.Ops.sum(r)
    loss.backward()
    if a.grad is None:
        return False, "a.grad None after non-contig relu"
    gv = vec(a.grad)
    if not all_finite(gv):
        return False, "grad NaN/Inf"
    return True, ""


def ag_checkpoint():
    def fn(x):
        return lt.Ops.relu(lt.Ops.add(x, x))
    x = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0], [4], DEV, True)
    y = lt.checkpoint(fn, x)
    loss = lt.Ops.sum(y)
    loss.backward()
    if x.grad is None:
        return False, "x.grad None"
    gx = vec(x.grad)
    if gx != [2.0, 0.0, 2.0, 0.0]:
        return False, f"checkpoint grad wrong: {gx}"
    return True, ""


def ag_create_graph():
    x = lt.Tensor.from_vector([2.0, 3.0], [2], DEV, True)
    y = lt.Ops.sum(x * x)
    y.backward(None, True)
    if x.grad is None:
        return False, "first-order grad None"
    g1 = vec(x.grad)
    if any(abs(g - 2.0 * v) > 1e-3 for g, v in zip(g1, [2.0, 3.0])):
        return False, f"first-order wrong: {g1}"
    z = lt.Ops.sum(x.grad * x.grad)
    z.backward()
    return True, ""


def ag_no_grad():
    x = lt.Tensor.from_vector([1.0, 2.0], [2], DEV, True)
    with lt.no_grad():
        if lt.is_grad_enabled():
            return False, "grad still enabled inside no_grad"
        y = lt.Ops.relu(x)
        if y.requires_grad:
            return False, "y requires_grad inside no_grad"
    if not lt.is_grad_enabled():
        return False, "grad not restored after no_grad"
    z = lt.Ops.relu(x)
    if not z.requires_grad:
        return False, "z should require grad outside no_grad"
    return True, ""


def ag_set_grad_enabled():
    try:
        lt.set_grad_enabled(False)
        if lt.is_grad_enabled():
            return False, "set_grad_enabled(False) failed"
        x = lt.Tensor.from_vector([1.0], [1], DEV, True)
        y = lt.Ops.relu(x)
        if y.requires_grad:
            return False, "op created graph while disabled"
    finally:
        lt.set_grad_enabled(True)
    if not lt.is_grad_enabled():
        return False, "not restored"
    return True, ""


def data_loader():
    n = 20
    xs = lt.Tensor.from_vector(randn(n * 4), [n, 4], CPU)
    ys = lt.Tensor.from_vector([float(i % 2) for i in range(n)], [n], CPU)
    ds = lt.data.TensorDataset(xs, ys)
    if ds.size() != n:
        return False, f"dataset size {ds.size()}"
    dl = lt.data.DataLoader(ds, 5, True, CPU)
    batches = []
    while True:
        b = dl.next()
        if b is None:
            break
        bx, by = b
        batches.append((list(bx.shape), list(by.shape)))
    if len(batches) != 4:
        return False, f"expected 4 batches, got {len(batches)}"
    for bs in batches:
        if bs[0] != [5, 4] or bs[1] != [5]:
            return False, f"bad batch shape {bs}"
    dl.reset()
    b = dl.next()
    if b is None:
        return False, "reset failed"
    return True, ""


def ser_params():
    m = lt.nn.Linear(8, 4)
    m.to(DEV)
    before = [vec(p) for p in m.parameters()]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "params.bin")
        lt.save_parameters(m.parameters(), path)
        m2 = lt.nn.Linear(8, 4)
        m2.to(DEV)
        lt.load_parameters(m2.parameters(), path)
        after = [vec(p) for p in m2.parameters()]
    for b, a in zip(before, after):
        if len(a) != len(b) or any(abs(x - y) > 1e-6 for x, y in zip(a, b)):
            return False, "params mismatch after load"
    return True, ""


def ser_optimizer():
    m = lt.nn.Linear(4, 2)
    m.to(DEV)
    opt = lt.optim.Adam(m.parameters(), lr=1e-3)
    x = lt.Tensor.from_vector(randn(16), [4, 4], DEV)
    y = lt.Tensor.from_vector(randn(8), [4, 2], DEV)
    out = m.forward(x)
    loss = lt.Ops.mse_loss(out, y)
    loss.backward()
    opt.step()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "opt.bin")
        lt.save_optimizer_state(opt, path)
        m2 = lt.nn.Linear(4, 2)
        m2.to(DEV)
        opt2 = lt.optim.Adam(m2.parameters(), lr=1e-3)
        lt.load_optimizer_state(opt2, path)
    return True, ""


def main():
    print(f"device: {DEV.to_string()}", flush=True)
    tests = [
        ("tensor_reshape_view", t_reshape_view),
        ("tensor_transpose", t_transpose),
        ("tensor_cat", t_cat),
        ("tensor_squeeze_unsqueeze", t_squeeze_unsqueeze),
        ("tensor_broadcast", t_broadcast),
        ("tensor_noncontig", t_noncontig),
        ("autograd_checkpoint", ag_checkpoint),
        ("autograd_create_graph", ag_create_graph),
        ("autograd_no_grad", ag_no_grad),
        ("autograd_set_grad_enabled", ag_set_grad_enabled),
        ("data_loader", data_loader),
        ("ser_params", ser_params),
        ("ser_optimizer", ser_optimizer),
    ]
    passed = 0
    for name, fn in tests:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"{type(e).__name__}: {e}"
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {msg}" if msg else ""), flush=True)
        if ok:
            passed += 1
    print(f"CORE: {passed}/{len(tests)} passed", flush=True)
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
