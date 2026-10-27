import math

import litetorch as lt

CPU = lt.Device("cpu")

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name} {detail}")


def close(a, b, tol=1e-5):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def vec(t):
    return t.to(CPU).to_vector()


def mkp(vals, shape):
    return lt.Tensor.from_vector(vals, shape, CPU, requires_grad=False)


def setgrad(p, gvals):
    p.grad = lt.Tensor.from_vector(gvals, p.shape, CPU)


def ref_adagrad_step(p, g, s, step, lr, lr_decay, wd, eps):
    clr = lr / (1.0 + (step - 1) * lr_decay)
    out_p, out_s = [], []
    for pj, gj, sj in zip(p, g, s):
        gv = gj + wd * pj
        sj2 = sj + gv * gv
        out_s.append(sj2)
        out_p.append(pj - clr * gv / (math.sqrt(sj2) + eps))
    return out_p, out_s


def ref_nadam_step(p, g, m, v, mu_prod, step, lr, b1, b2, eps, wd, md):
    bc2 = 1.0 - b2 ** step
    mu = b1 * (1.0 - 0.5 * 0.96 ** (step * md))
    mu_next = b1 * (1.0 - 0.5 * 0.96 ** ((step + 1) * md))
    mu_prod = mu_prod * mu
    op, om, ov = [], [], []
    for pj, gj, mj, vj in zip(p, g, m, v):
        gv = gj + wd * pj
        mj2 = b1 * mj + (1.0 - b1) * gv
        vj2 = b2 * vj + (1.0 - b2) * gv * gv
        denom = math.sqrt(vj2 / bc2) + eps
        op.append(pj - lr * (1.0 - mu) / (1.0 - mu_prod) * gv / denom
                  - lr * mu_next / (1.0 - mu_prod * mu_next) * mj2 / denom)
        om.append(mj2)
        ov.append(vj2)
    return op, om, ov, mu_prod


def test_adagrad_formula():
    p0 = [0.5, -1.0, 2.0, 0.25]
    g0 = [0.1, 0.2, -0.3, 0.4]
    p = mkp(p0, [4])
    opt = lt.optim.Adagrad([p], lr=0.1, lr_decay=0.0, weight_decay=0.0, eps=1e-10)
    s = [0.0] * 4
    pp, gg = p0[:], g0[:]
    ok = True
    for step in (1, 2):
        setgrad(p, gg)
        opt.step()
        pp, s = ref_adagrad_step(pp, gg, s, step, 0.1, 0.0, 0.0, 1e-10)
        got = vec(p)
        ok = ok and all(close(a, b) for a, b in zip(got, pp))
    check("adagrad_matches_formula", ok)


def test_adagrad_wd_lrdecay():
    p0 = [1.0, -0.5]
    g0 = [0.3, -0.7]
    p = mkp(p0, [2])
    opt = lt.optim.Adagrad([p], lr=0.2, lr_decay=0.1, weight_decay=0.01, eps=1e-10)
    s = [0.0] * 2
    pp = p0[:]
    ok = True
    for step in (1, 2, 3):
        setgrad(p, g0)
        opt.step()
        pp, s = ref_adagrad_step(pp, g0, s, step, 0.2, 0.1, 0.01, 1e-10)
        got = vec(p)
        ok = ok and all(close(a, b) for a, b in zip(got, pp))
    check("adagrad_weight_decay_lr_decay", ok)


def test_adagrad_state_roundtrip():
    p0 = [0.8, -0.3, 0.1]
    mk = lambda: mkp(p0, [3])
    pa, pb = mk(), mk()
    oa = lt.optim.Adagrad([pa], lr=0.05)
    for _ in range(2):
        setgrad(pa, [0.2, -0.1, 0.4])
        oa.step()
    sd = oa.state_dict()
    pb.copy_(pa)
    ob = lt.optim.Adagrad([pb], lr=0.05)
    ob.load_state_dict(sd)
    ok = True
    for _ in range(3):
        setgrad(pa, [0.2, -0.1, 0.4])
        setgrad(pb, [0.2, -0.1, 0.4])
        oa.step()
        ob.step()
        ok = ok and all(close(a, b) for a, b in zip(vec(pa), vec(pb)))
    check("adagrad_state_dict_roundtrip", ok)


def test_adagrad_param_groups():
    p1 = mkp([1.0, 1.0], [2])
    p2 = mkp([1.0, 1.0], [2])
    opt = lt.optim.Adagrad([p1], lr=0.1)
    opt.add_param_group([p2], 0.01, 0.0)
    groups = opt.param_groups()
    ok = len(groups) == 2 and close(groups[0]["lr"], 0.1) and close(groups[1]["lr"], 0.01)
    setgrad(p1, [0.5, 0.5])
    setgrad(p2, [0.5, 0.5])
    opt.step()
    v1, v2 = vec(p1), vec(p2)
    ok = ok and abs(v1[0] - v2[0]) > 1e-6
    s = [0.0, 0.0]
    e1, _ = ref_adagrad_step([1.0, 1.0], [0.5, 0.5], s, 1, 0.1, 0.0, 0.0, 1e-10)
    s2 = [0.0, 0.0]
    e2, _ = ref_adagrad_step([1.0, 1.0], [0.5, 0.5], s2, 1, 0.01, 0.0, 0.0, 1e-10)
    ok = ok and all(close(a, b) for a, b in zip(v1, e1)) and all(close(a, b) for a, b in zip(v2, e2))
    check("adagrad_param_groups", ok)


def test_nadam_formula():
    p0 = [0.5, -1.0, 2.0]
    g0 = [0.1, 0.2, -0.3]
    p = mkp(p0, [3])
    opt = lt.optim.NAdam([p], lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, momentum_decay=0.004)
    m = [0.0] * 3
    v = [0.0] * 3
    mu_prod = 1.0
    pp = p0[:]
    ok = True
    for step in (1, 2, 3):
        setgrad(p, g0)
        opt.step()
        pp, m, v, mu_prod = ref_nadam_step(pp, g0, m, v, mu_prod, step, 0.002, 0.9, 0.999, 1e-8, 0.0, 0.004)
        got = vec(p)
        ok = ok and all(close(a, b, tol=1e-4) for a, b in zip(got, pp))
    check("nadam_matches_formula", ok)


def test_nadam_wd_state_roundtrip():
    p0 = [0.7, -0.2]
    mk = lambda: mkp(p0, [2])
    pa, pb = mk(), mk()
    oa = lt.optim.NAdam([pa], lr=0.001, weight_decay=0.01)
    for _ in range(2):
        setgrad(pa, [0.3, -0.2])
        oa.step()
    sd = oa.state_dict()
    check("nadam_state_dict_keys", "exp_avg" in sd["state"][0] and "exp_avg_sq" in sd["state"][0] and sd["betas"] == (0.9, 0.999))
    pb.copy_(pa)
    ob = lt.optim.NAdam([pb], lr=0.001, weight_decay=0.01)
    ob.load_state_dict(sd)
    ok = True
    for _ in range(2):
        setgrad(pa, [0.3, -0.2])
        setgrad(pb, [0.3, -0.2])
        oa.step()
        ob.step()
        ok = ok and all(close(a, b, tol=1e-4) for a, b in zip(vec(pa), vec(pb)))
    check("nadam_state_dict_roundtrip", ok)


def test_multisteplr():
    p = mkp([1.0], [1])
    opt = lt.optim.SGD([p], lr=0.1)
    sch = lt.optim.MultiStepLR(opt, milestones=[2, 4], gamma=0.1)
    lrs = []
    for _ in range(5):
        sch.step()
        lrs.append(sch.get_last_lr())
    exp = [0.1, 0.01, 0.01, 0.001, 0.001]
    check("multisteplr_schedule", all(close(a, b) for a, b in zip(lrs, exp)), f"got {lrs}")
    sd = sch.state_dict()
    sch2 = lt.optim.MultiStepLR(opt, milestones=[2, 4], gamma=0.1)
    opt.set_lr(0.1)
    sch2.load_state_dict(sd)
    check("multisteplr_state_dict", sch2.last_epoch == 5 and sd["milestones"] == [2, 4] and close(sd["gamma"], 0.1))


def test_exponentiallr():
    p = mkp([1.0], [1])
    opt = lt.optim.SGD([p], lr=0.1)
    sch = lt.optim.ExponentialLR(opt, gamma=0.9)
    lrs = []
    for _ in range(4):
        sch.step()
        lrs.append(sch.get_last_lr())
    exp = [0.1 * 0.9 ** i for i in (1, 2, 3, 4)]
    check("exponentiallr_schedule", all(close(a, b) for a, b in zip(lrs, exp)), f"got {lrs}")
    sd = sch.state_dict()
    sch2 = lt.optim.ExponentialLR(opt, gamma=0.9)
    sch2.load_state_dict(sd)
    check("exponentiallr_state_dict", sch2.last_epoch == 4 and close(sd["gamma"], 0.9))


def test_plateau_reduce():
    p = mkp([1.0], [1])
    opt = lt.optim.SGD([p], lr=0.1)
    sch = lt.optim.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, threshold=1e-4)
    lrs = []
    for _ in range(5):
        sch.step(1.0)
        lrs.append(sch.get_last_lr())
    check("plateau_reduce_on_flat", all(close(a, b) for a, b in zip(lrs, [0.1, 0.1, 0.1, 0.05, 0.05])), f"got {lrs}")


def test_plateau_improvement_resets():
    p = mkp([1.0], [1])
    opt = lt.optim.SGD([p], lr=0.1)
    sch = lt.optim.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    lrs = []
    for m in (1.0, 0.5, 0.5, 0.5, 0.5):
        sch.step(m)
        lrs.append(sch.get_last_lr())
    check("plateau_improvement_resets", all(close(a, b) for a, b in zip(lrs, [0.1, 0.1, 0.1, 0.1, 0.05])), f"got {lrs}")


def test_plateau_max_mode():
    p = mkp([1.0], [1])
    opt = lt.optim.SGD([p], lr=0.2)
    sch = lt.optim.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=1)
    lrs = []
    for m in (0.1, 0.1, 0.1):
        sch.step(m)
        lrs.append(sch.get_last_lr())
    check("plateau_max_mode", all(close(a, b) for a, b in zip(lrs, [0.2, 0.2, 0.02])), f"got {lrs}")
    sd = sch.state_dict()
    check("plateau_state_dict", sd["mode"] == "max" and sd["num_bad_epochs"] == 0 and close(sd["best"], 0.1))
    sch2 = lt.optim.ReduceLROnPlateau(opt, mode="min")
    sch2.load_state_dict(sd)
    check("plateau_load_state_dict", sch2.mode == "max" and sch2.patience == 1 and close(sch2.best, 0.1))


def test_scheduler_factories():
    p = mkp([1.0], [1])
    opt = lt.optim.SGD([p], lr=0.1)
    s1 = lt.optim.lr_scheduler.MultiStepLR(opt, [3], 0.5)
    s2 = lt.optim.lr_scheduler.ExponentialLR(opt, 0.9)
    s3 = lt.optim.lr_scheduler.ReduceLROnPlateau(opt)
    ok = s1.get_last_lr() == 0.1 and s2.get_last_lr() == 0.1 and s3.get_last_lr() == 0.1
    s3.step(2.0)
    ok = ok and s3.last_epoch == 1
    check("lr_scheduler_factories", ok)


def test_adagrad_trains():
    import random
    random.seed(0)
    w = mkp([random.gauss(0, 1) for _ in range(4)], [4])
    w.requires_grad = True
    x = lt.Tensor.from_vector([0.5, -0.5, 1.0, 0.0], [1, 4], CPU)
    y = lt.Tensor.from_vector([1.0], [1, 1], CPU)
    opt = lt.optim.Adagrad([w], lr=0.1)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        pred = lt.Ops.matmul(x, w.reshape([4, 1]))
        loss = lt.Ops.mse_loss(pred, y)
        losses.append(loss.to(CPU).to_vector()[0])
        loss.backward()
        opt.step()
    check("adagrad_trains_loss_down", losses[-1] < losses[0] * 0.5, f"{losses[0]:.4f}->{losses[-1]:.4f}")


def test_nadam_trains():
    import random
    random.seed(1)
    w = mkp([random.gauss(0, 1) for _ in range(4)], [4])
    w.requires_grad = True
    x = lt.Tensor.from_vector([0.5, -0.5, 1.0, 0.0], [1, 4], CPU)
    y = lt.Tensor.from_vector([1.0], [1, 1], CPU)
    opt = lt.optim.NAdam([w], lr=0.01)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        pred = lt.Ops.matmul(x, w.reshape([4, 1]))
        loss = lt.Ops.mse_loss(pred, y)
        losses.append(loss.to(CPU).to_vector()[0])
        loss.backward()
        opt.step()
    check("nadam_trains_loss_down", losses[-1] < losses[0] * 0.5, f"{losses[0]:.4f}->{losses[-1]:.4f}")


if __name__ == "__main__":
    test_adagrad_formula()
    test_adagrad_wd_lrdecay()
    test_adagrad_state_roundtrip()
    test_adagrad_param_groups()
    test_nadam_formula()
    test_nadam_wd_state_roundtrip()
    test_multisteplr()
    test_exponentiallr()
    test_plateau_reduce()
    test_plateau_improvement_resets()
    test_plateau_max_mode()
    test_scheduler_factories()
    test_adagrad_trains()
    test_nadam_trains()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
