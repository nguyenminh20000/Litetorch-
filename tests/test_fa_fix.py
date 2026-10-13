import math
import litetorch as lt
from litetorch import Ops

def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return cond

def test_lse_math():
    import random
    random.seed(0)
    ok = True
    for causal in (False, True):
        for T in (7, 16, 33):
            S = [random.uniform(-3, 3) for _ in range(T)]
            mx = max(S)
            lse = mx + math.log(sum(math.exp(s - mx) for s in S))
            for j in range(T):
                p_softmax = math.exp(S[j] - mx) / sum(math.exp(s - mx) for s in S)
                p_lse = math.exp(S[j] - lse)
                if abs(p_softmax - p_lse) > 1e-12:
                    ok = False
            if causal:
                for j in range(T):
                    masked = 0.0
                    smax = max(S[:j + 1])
                    slse = smax + math.log(sum(math.exp(s - smax) for s in S[:j + 1]))
                    p_lse = math.exp(S[j] - slse)
                    p_ref = math.exp(S[j] - smax) / sum(math.exp(s - smax) for s in S[:j + 1])
                    if abs(p_lse - p_ref) > 1e-12:
                        ok = False
    check("lse_math: exp(S-LSE)==softmax, causal masking", ok)

def test_lse_bwd_math():
    import random
    random.seed(1)
    B, T, D = 1, 8, 4
    scale = 1.0 / math.sqrt(D)
    q = [random.uniform(-1, 1) for _ in range(T * D)]
    k = [random.uniform(-1, 1) for _ in range(T * D)]
    v = [random.uniform(-1, 1) for _ in range(T * D)]
    def fwd(qv, kv, vv):
        S = [[sum(qv[i * D + d] * kv[j * D + d] for d in range(D)) * scale for j in range(T)] for i in range(T)]
        LSE = []
        P = []
        for i in range(T):
            mx = max(S[i])
            lse = mx + math.log(sum(math.exp(s - mx) for s in S[i]))
            LSE.append(lse)
            P.append([math.exp(s - lse) for s in S[i]])
        O = [[sum(P[i][j] * vv[j * D + d] for j in range(T)) for d in range(D)] for i in range(T)]
        return O, P, LSE, S
    def bwd_with_lse(qv, kv, vv, dO):
        O, _, LSE, S = fwd(qv, kv, vv)
        P = [[math.exp(S[i][j] - LSE[i]) for j in range(T)] for i in range(T)]
        dV = [[sum(P[i][j] * dO[i * D + d] for i in range(T)) for d in range(D)] for j in range(T)]
        dP = [[sum(dO[i * D + d] * vv[j * D + d] for d in range(D)) for j in range(T)] for i in range(T)]
        dS = [[P[i][j] * (dP[i][j] - sum(dP[i][t] * P[i][t] for t in range(T))) * scale for j in range(T)] for i in range(T)]
        dQ = [[sum(dS[i][j] * kv[j * D + d] for j in range(T)) for d in range(D)] for i in range(T)]
        dK = [[sum(dS[i][j] * qv[i * D + d] for i in range(T)) for d in range(D)] for j in range(T)]
        return dQ, dK, dV
    dO = [random.uniform(-1, 1) for _ in range(T * D)]
    dQ, dK, dV = bwd_with_lse(q, k, v, dO)
    eps = 1e-4
    maxerr = 0.0
    for arr, name in ((q, "q"), (k, "k"), (v, "v")):
        got = {"q": dQ, "k": dK, "v": dV}[name]
        for idx in range(len(arr)):
            orig = arr[idx]
            arr[idx] = orig + eps
            Op, _, _, _ = fwd(q, k, v)
            lp = sum(Op[i][d] * dO[i * D + d] for i in range(T) for d in range(D))
            arr[idx] = orig - eps
            Om, _, _, _ = fwd(q, k, v)
            lm = sum(Om[i][d] * dO[i * D + d] for i in range(T) for d in range(D))
            arr[idx] = orig
            fd = (lp - lm) / (2 * eps)
            gotv = got[idx // D][idx % D] if name == "q" else got[idx // D][idx % D]
            maxerr = max(maxerr, abs(fd - gotv))
    check("lse_bwd_math: LSE-recompute grads vs finite diff", maxerr < 1e-3, f"maxerr={maxerr:.2e}")

def test_cpu_regression():
    d = lt.Device("cpu")
    B, H, T, Dd = 1, 2, 8, 16
    q = lt.Tensor.randn([B, H, T, Dd], d, True)
    k = lt.Tensor.randn([B, H, T, Dd], d, True)
    v = lt.Tensor.randn([B, H, T, Dd], d, True)
    out = Ops.flash_attention(q, k, v, causal=False)
    gv = out.to_vector()
    ok_fwd = all(math.isfinite(x) for x in gv) and len(gv) == B * H * T * Dd
    check("cpu_regression: forward finite", ok_fwd)
    loss = Ops.sum(out)
    loss.backward()
    gq = q.grad.to_vector() if q.grad is not None else []
    ok_bwd = len(gq) == B * H * T * Dd and all(math.isfinite(x) for x in gq)
    check("cpu_regression: backward grads finite", ok_bwd)
    q2 = lt.Tensor.randn([B, H, T, Dd], d, True)
    k2 = lt.Tensor.randn([B, H, T, Dd], d, True)
    v2 = lt.Tensor.randn([B, H, T, Dd], d, True)
    out2 = Ops.flash_attention(q2, k2, v2, causal=True)
    gv2 = out2.to_vector()
    check("cpu_regression: causal forward finite", all(math.isfinite(x) for x in gv2))

def test_no_p_allocation():
    with open("src/ops/attention.cpp") as f:
        src = f.read()
    assert "attn_p" not in src, "attn_p still present"
    assert "need_p" not in src, "need_p still present"
    assert "p_saved" not in src, "p_saved still present"
    assert "Tensor::create({B, H, Tq, Tk}" not in src, "O(N^2) P allocation still present"
    assert "Tensor::create({B, H, T, T}" not in src, "O(N^2) P allocation still present"
    assert "attn_lse = Tensor::create({B, H, Tq}" in src, "LSE allocation missing"
    print("[PASS] no_p_allocation")

if __name__ == "__main__":
    test_lse_math()
    test_lse_bwd_math()
    test_cpu_regression()
    test_no_p_allocation()
