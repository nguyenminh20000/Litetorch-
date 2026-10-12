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


def close(a, b, tol=1e-4):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def vec(t):
    return t.to(CPU).to_vector()


def mkp(vals, shape):
    return lt.Tensor.from_vector(vals, shape, CPU, requires_grad=False)


def ref_kl_div(log_probs, probs, reduction):
    n = len(log_probs)
    elems = [p * (math.log(max(p, 1e-10)) - lp) for lp, p in zip(log_probs, probs)]
    if reduction == "none":
        return elems
    if reduction == "sum":
        return sum(elems)
    if reduction == "batchmean":
        return sum(elems) / 1
    return sum(elems) / n


def test_kldiv():
    lp = [-0.5, -1.2, -2.0, -0.8]
    pr = [0.4, 0.3, 0.1, 0.2]
    x = mkp(lp, [1, 4])
    y = mkp(pr, [1, 4])
    for red in ("mean", "sum", "none", "batchmean"):
        loss = lt.nn.KLDivLoss(reduction=red)(x, y)
        got = vec(loss)
        exp = ref_kl_div(lp, pr, red)
        if red == "none":
            ok = all(close(g, e) for g, e in zip(got, exp))
        else:
            ok = close(got[0], exp)
        check(f"kldiv_{red}", ok, f"got={got} exp={exp}")


def ref_smooth_l1(diff, beta):
    out = []
    for d in diff:
        ad = abs(d)
        out.append(0.5 * d * d / beta if ad < beta else ad - 0.5 * beta)
    return out


def test_smooth_l1():
    inp = [0.5, -1.5, 2.0, -0.2, 3.0]
    tgt = [0.0, 0.0, 0.0, 0.0, 0.0]
    x = mkp(inp, [5])
    y = mkp(tgt, [5])
    for beta in (1.0, 0.5):
        for red in ("mean", "sum", "none"):
            loss = lt.nn.SmoothL1Loss(beta=beta, reduction=red)(x, y)
            got = vec(loss)
            elems = ref_smooth_l1([a - b for a, b in zip(inp, tgt)], beta)
            if red == "none":
                exp = elems
                ok = all(close(g, e) for g, e in zip(got, exp))
            elif red == "sum":
                ok = close(got[0], sum(elems))
            else:
                ok = close(got[0], sum(elems) / len(elems))
            check(f"smoothl1_beta{beta}_{red}", ok, f"got={got}")


def test_huber_matches_smooth():
    inp = [0.5, -1.5, 2.0, -0.2, 3.0]
    tgt = [0.1, 0.2, -0.1, 0.0, 0.5]
    x = mkp(inp, [5])
    y = mkp(tgt, [5])
    for delta in (1.0, 2.0):
        a = vec(lt.nn.HuberLoss(delta=delta)(x, y))[0]
        b = vec(lt.nn.SmoothL1Loss(beta=delta)(x, y))[0]
        check(f"huber_delta{delta}_eq_smooth", close(a, b), f"{a} vs {b}")


def ctc_collapse(path, blank=0):
    out = []
    prev = None
    for c in path:
        if c != prev:
            if c != blank:
                out.append(c)
            prev = c
    return tuple(out)


def ref_ctc_bruteforce(log_probs, T, C, target, blank=0):
    total = 0.0
    for idx in range(C ** T):
        path = []
        k = idx
        for _ in range(T):
            path.append(k % C)
            k //= C
        if ctc_collapse(path, blank) == tuple(target):
            lp = sum(log_probs[t * C + path[t]] for t in range(T))
            total += math.exp(lp)
    return -math.log(total)


def test_ctc_bruteforce():
    T, N, C = 5, 1, 3
    target = [1, 2]
    lp_vals = [
        -0.5, -1.0, -2.0,
        -1.5, -0.3, -1.2,
        -2.0, -0.8, -0.4,
        -1.0, -1.0, -1.0,
        -0.2, -2.5, -1.8,
    ]
    lp = mkp(lp_vals, [T, N, C])
    tgt = mkp([float(v) for v in target], [N, len(target)])
    il = mkp([float(T)], [N])
    tl = mkp([float(len(target))], [N])
    for red in ("mean", "sum", "none"):
        loss = lt.nn.CTCLoss(blank=0, reduction=red)(lp, tgt, il, tl)
        got = vec(loss)
        exp = ref_ctc_bruteforce(lp_vals, T, C, target)
        if red == "none":
            ok = close(got[0], exp, tol=1e-3)
        else:
            ok = close(got[0], exp, tol=1e-3)
        check(f"ctc_bruteforce_{red}", ok, f"got={got[0]:.6f} exp={exp:.6f}")


def test_ctc_grad_finite_diff():
    T, N, C = 4, 1, 3
    target = [1]
    lp_vals = [-0.5, -1.0, -2.0, -1.5, -0.3, -1.2, -2.0, -0.8, -0.4, -1.0, -1.0, -1.0]
    tgt = mkp([1.0], [N, 1])
    il = mkp([float(T)], [N])
    tl = mkp([1.0], [N])
    loss_fn = lt.nn.CTCLoss(blank=0, reduction="sum")

    def fwd(vals):
        lp = mkp(vals, [T, N, C])
        return vec(loss_fn(lp, tgt, il, tl))[0]

    eps = 1e-3
    num_grad = []
    base = fwd(lp_vals)
    for i in range(len(lp_vals)):
        vp = lp_vals[:]
        vp[i] += eps
        num_grad.append((fwd(vp) - base) / eps)

    lp = lt.Tensor.from_vector(lp_vals, [T, N, C], CPU, requires_grad=True)
    loss = loss_fn(lp, tgt, il, tl)
    loss.backward()
    ana_grad = vec(lp.grad)
    ok = all(close(a, b, tol=5e-2) for a, b in zip(ana_grad, num_grad))
    max_err = max(abs(a - b) for a, b in zip(ana_grad, num_grad))
    check("ctc_grad_finitediff", ok, f"max_err={max_err:.2e}")


def test_ctc_zero_infinity():
    T, N, C = 2, 1, 3
    lp = mkp([-0.5] * (T * N * C), [T, N, C])
    tgt = mkp([1.0, 1.0], [N, 2])
    il = mkp([float(T)], [N])
    tl = mkp([2.0], [N])
    loss = lt.nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)(lp, tgt, il, tl)
    check("ctc_zero_infinity", vec(loss)[0] == 0.0, f"got={vec(loss)[0]}")
    loss2 = lt.nn.CTCLoss(blank=0, reduction="sum", zero_infinity=False)(lp, tgt, il, tl)
    check("ctc_inf_without_flag", math.isinf(vec(loss2)[0]), f"got={vec(loss2)[0]}")


def test_ctc_1d_targets():
    T, N, C = 5, 2, 3
    lp_vals = [-0.5, -1.0, -2.0] * (T * N)
    lp = mkp(lp_vals, [T, N, C])
    tgt = mkp([1.0, 2.0, 1.0], [3])
    il = mkp([float(T), float(T)], [N])
    tl = mkp([2.0, 1.0], [N])
    loss = lt.nn.CTCLoss(blank=0, reduction="none")(lp, tgt, il, tl)
    got = vec(loss)
    check("ctc_1d_targets", len(got) == 2 and all(math.isfinite(v) for v in got), f"got={got}")


def test_trunc_normal():
    t = lt.Tensor.from_vector([0.0] * 2000, [2000], CPU)
    lt.nn.init.trunc_normal_(t, mean=0.0, std=1.0, a=-2.0, b=2.0)
    v = vec(t)
    in_range = all(-2.0 <= x <= 2.0 for x in v)
    mean = sum(v) / len(v)
    var = sum((x - mean) ** 2 for x in v) / len(v)
    check("trunc_normal_range", in_range)
    check("trunc_normal_mean", abs(mean) < 0.15, f"mean={mean:.4f}")
    check("trunc_normal_std", 0.7 < math.sqrt(var) < 1.0, f"std={math.sqrt(var):.4f}")


def test_orthogonal():
    for shape in ([8, 8], [4, 8], [8, 4]):
        rows, cols = shape
        t = lt.Tensor.from_vector([0.0] * (rows * cols), shape, CPU)
        lt.nn.init.orthogonal_(t, gain=1.0)
        v = vec(t)
        m = rows if rows >= cols else cols
        n = cols if rows >= cols else rows
        ok = True
        for i in range(n):
            for j in range(n):
                dot = 0.0
                for k in range(m):
                    a = v[k * cols + i] if rows >= cols else v[i * cols + k]
                    b = v[k * cols + j] if rows >= cols else v[j * cols + k]
                    dot += a * b
                if not close(dot, 1.0 if i == j else 0.0, tol=1e-4):
                    ok = False
        check(f"orthogonal_{rows}x{cols}", ok)


def test_clip_grad_value():
    p1 = lt.Tensor.from_vector([1.0, 2.0], [2], CPU, requires_grad=True)
    p2 = lt.Tensor.from_vector([3.0], [1], CPU, requires_grad=True)
    p1.grad = mkp([5.0, -5.0], [2])
    p2.grad = mkp([-0.5], [1])
    lt.nn.utils.clip_grad_value_([p1, p2], 1.0)
    g1, g2 = vec(p1.grad), vec(p2.grad)
    check("clip_grad_value", g1 == [1.0, -1.0] and g2 == [-0.5], f"{g1} {g2}")


def test_params_vector_roundtrip():
    p1 = mkp([1.0, 2.0, 3.0], [3])
    p2 = mkp([4.0, 5.0], [2, 1])
    v = lt.nn.utils.parameters_to_vector([p1, p2])
    check("params_to_vector", vec(v) == [1.0, 2.0, 3.0, 4.0, 5.0], f"{vec(v)}")
    v2 = mkp([10.0, 20.0, 30.0, 40.0, 50.0], [5])
    lt.nn.utils.vector_to_parameters(v2, [p1, p2])
    check("vector_to_params", vec(p1) == [10.0, 20.0, 30.0] and vec(p2) == [40.0, 50.0])


def test_weight_norm():
    v = mkp([3.0, 4.0, 0.0, 0.0, 6.0, 8.0], [2, 3])
    g = mkp([2.0, 5.0], [2])
    w = lt.nn.utils.weight_norm(v, g, dim=0)
    wv = vec(w)
    n0 = math.sqrt(wv[0] ** 2 + wv[1] ** 2 + wv[2] ** 2)
    n1 = math.sqrt(wv[3] ** 2 + wv[4] ** 2 + wv[5] ** 2)
    check("weight_norm_row0", close(n0, 2.0, tol=1e-5), f"n0={n0}")
    check("weight_norm_row1", close(n1, 5.0, tol=1e-5), f"n1={n1}")
    check("weight_norm_dir", close(wv[0] / wv[1], 3.0 / 4.0, tol=1e-5))


def test_spectral_norm():
    w = mkp([1.0, 2.0, 3.0, 4.0], [2, 2])
    wn = lt.nn.utils.spectral_norm(w, n_power_iterations=20)
    wv = vec(wn)
    a, b, c, d = wv
    e11 = a * a + c * c
    e12 = a * b + c * d
    e22 = b * b + d * d
    tr = e11 + e22
    det = e11 * e22 - e12 * e12
    disc = math.sqrt(max(tr * tr - 4 * det, 0.0))
    sigma_max = math.sqrt(max((tr + disc) / 2, 0.0))
    check("spectral_norm_sigma1", close(sigma_max, 1.0, tol=1e-3), f"sigma={sigma_max:.6f}")


test_kldiv()
test_smooth_l1()
test_huber_matches_smooth()
test_ctc_bruteforce()
test_ctc_grad_finite_diff()
test_ctc_zero_infinity()
test_ctc_1d_targets()
test_trunc_normal()
test_orthogonal()
test_clip_grad_value()
test_params_vector_roundtrip()
test_weight_norm()
test_spectral_norm()

print(f"\nTOTAL: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
