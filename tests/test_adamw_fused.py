import math
import random

import litetorch as lt

CPU = lt.Device("cpu")

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"[FAIL] {name}")

def vec(t):
    return t.to(CPU).to_vector()

def make_param(shape, seed):
    random.seed(seed)
    n = 1
    for d in shape:
        n *= d
    vals = [random.gauss(0.0, 0.5) for _ in range(n)]
    t = lt.Tensor.from_vector(vals, list(shape), CPU)
    t.requires_grad = True
    return t

def set_grad(p, seed):
    random.seed(seed)
    n = p.numel()
    gvals = [random.gauss(0.0, 0.1) for _ in range(n)]
    g = lt.Tensor.from_vector(gvals, list(p.shape), CPU)
    p.grad = g
    return gvals

def ref_adamw_step(params_vals, grads_vals, m_vals, v_vals, step, lr, beta1, beta2, eps, wd):
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    new_p, new_m, new_v = [], [], []
    for pv, gv, mv, vv in zip(params_vals, grads_vals, m_vals, v_vals):
        np_, nm_, nv_ = [], [], []
        for p, g, m, v in zip(pv, gv, mv, vv):
            m2 = beta1 * m + (1.0 - beta1) * g
            v2 = beta2 * v + (1.0 - beta2) * g * g
            m_hat = m2 / bc1
            v_hat = v2 / bc2
            update = m_hat / (math.sqrt(v_hat) + eps)
            if wd != 0.0:
                p2 = p - lr * (wd * p + update)
            else:
                p2 = p - lr * update
            np_.append(p2)
            nm_.append(m2)
            nv_.append(v2)
        new_p.append(np_)
        new_m.append(nm_)
        new_v.append(nv_)
    return new_p, new_m, new_v

def run_case(name, shapes, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01, steps=3):
    params = [make_param(s, 100 + i) for i, s in enumerate(shapes)]
    opt = lt.optim.AdamW(params, lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=wd)

    pv = [vec(p)[:] for p in params]
    mv = [[0.0] * len(v) for v in pv]
    vv = [[0.0] * len(v) for v in pv]

    for s in range(1, steps + 1):
        gv = [set_grad(p, 1000 + s * 10 + i)[:] for i, p in enumerate(params)]
        opt.step()
        pv, mv, vv = ref_adamw_step(pv, gv, mv, vv, s, lr, beta1, beta2, eps, wd)
        got = [vec(p) for p in params]
        ok = True
        for g, e in zip(got, pv):
            for a, b in zip(g, e):
                if abs(a - b) > 1e-5:
                    ok = False
                    break
        check(f"{name} step {s} bit-match", ok)
        opt.zero_grad()

def test_correctness_basic():
    run_case("basic", [[4, 4], [8], [3, 5]])

def test_correctness_no_wd():
    run_case("no_wd", [[4, 4], [8]], wd=0.0)

def test_correctness_many_tensors():
    run_case("many", [[2, 2]] * 10 + [[7], [3, 3, 3]], steps=2)

def test_correctness_single():
    run_case("single", [[5, 5]], steps=2)

def test_correctness_custom_hypers():
    run_case("hypers", [[4, 4], [6]], lr=0.01, beta1=0.8, beta2=0.99, eps=1e-6, wd=0.1, steps=2)

def test_launch_counts():
    shapes = [[4, 4], [8], [3, 5], [2, 2], [7]]
    params = [make_param(s, 200 + i) for i, s in enumerate(shapes)]
    opt = lt.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    for i, p in enumerate(params):
        set_grad(p, 300 + i)
    lt.optim.AdamW.reset_launch_counts()
    opt.step()
    f = lt.optim.AdamW.fused_launch_count()
    s = lt.optim.AdamW.single_launch_count()
    check("fused launch count == 1", f == 1)
    check("single launch count == 0", s == 0)
    print(f"  [info] 5 tensors: fused={f} single={s} (before: 5 single launches)")

def test_launch_counts_single_tensor():
    params = [make_param([4, 4], 400)]
    opt = lt.optim.AdamW(params, lr=1e-3)
    set_grad(params[0], 401)
    lt.optim.AdamW.reset_launch_counts()
    opt.step()
    f = lt.optim.AdamW.fused_launch_count()
    check("single tensor fused == 1", f == 1)

def test_no_grad_skipped():
    p1 = make_param([4, 4], 500)
    p2 = make_param([4, 4], 501)
    set_grad(p1, 502)
    before = vec(p2)[:]
    opt = lt.optim.AdamW([p1, p2], lr=1e-3)
    lt.optim.AdamW.reset_launch_counts()
    opt.step()
    after = vec(p2)
    check("no-grad param untouched", before == after)
    check("no-grad fused == 1", lt.optim.AdamW.fused_launch_count() == 1)

def test_all_no_grad():
    p1 = make_param([4, 4], 600)
    opt = lt.optim.AdamW([p1], lr=1e-3)
    lt.optim.AdamW.reset_launch_counts()
    opt.step()
    check("all-no-grad no crash", True)
    check("all-no-grad fused == 0", lt.optim.AdamW.fused_launch_count() == 0)

if __name__ == "__main__":
    test_correctness_basic()
    test_correctness_no_wd()
    test_correctness_many_tensors()
    test_correctness_single()
    test_correctness_custom_hypers()
    test_launch_counts()
    test_launch_counts_single_tensor()
    test_no_grad_skipped()
    test_all_no_grad()
    print(f"\nTOTAL: {PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
