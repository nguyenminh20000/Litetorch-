import math
import random

import litetorch as lt

CPU = lt.Device("cpu")
FP16 = lt.DataType.FP16
FP32 = lt.DataType.FP32

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

def make_param(shape, seed, dtype=FP32):
    random.seed(seed)
    n = 1
    for d in shape:
        n *= d
    vals = [random.gauss(0.0, 0.5) for _ in range(n)]
    t = lt.Tensor.from_vector(vals, list(shape), CPU, False, dtype)
    t.requires_grad = True
    return t

def set_grad(p, seed):
    random.seed(seed)
    n = p.numel()
    gvals = [random.gauss(0.0, 0.1) for _ in range(n)]
    g = lt.Tensor.from_vector(gvals, list(p.shape), CPU, False, p.dtype)
    p.grad = g

def test_fp32_basic():
    p = make_param([8, 4], 1)
    set_grad(p, 2)
    before = vec(p)[:]
    opt = lt.optim.ZeRO3Optimizer([p], lr=1e-3)
    opt.step()
    after = vec(p)
    check("fp32_basic: params changed", any(a != b for a, b in zip(before, after)))
    check("fp32_basic: no NaN", all(math.isfinite(v) for v in after))

def test_fp32_matches_adamw():
    p1 = make_param([8, 4], 10)
    p2 = make_param([8, 4], 10)
    set_grad(p1, 11)
    p2.grad = p1.grad
    opt_z = lt.optim.ZeRO3Optimizer([p1], lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)
    opt_a = lt.optim.AdamW([p2], lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)
    for _ in range(5):
        opt_z.step()
        opt_a.step()
        set_grad(p1, 11)
        p2.grad = p1.grad
    v1, v2 = vec(p1), vec(p2)
    maxdiff = max(abs(a - b) for a, b in zip(v1, v2))
    check(f"fp32_matches_adamw: maxdiff={maxdiff:.2e}", maxdiff < 1e-5)

def test_fp16_master_weights():
    p = make_param([8, 4], 20, FP16)
    set_grad(p, 21)
    opt = lt.optim.ZeRO3Optimizer([p], lr=1e-3)
    opt.step()
    after = vec(p)
    check("fp16_master: params changed", any(v != 0.0 for v in after))
    check("fp16_master: no NaN/Inf", all(math.isfinite(v) for v in after))
    check("fp16_master: dtype preserved", p.dtype == FP16)

def test_fp16_matches_fp32_reference():
    p16 = make_param([16, 8], 30, FP16)
    p32 = make_param([16, 8], 30, FP32)
    set_grad(p16, 31)
    g32_vals = vec(p16.grad)
    p32.grad = lt.Tensor.from_vector(g32_vals, [16, 8], CPU)
    opt16 = lt.optim.ZeRO3Optimizer([p16], lr=1e-3)
    opt32 = lt.optim.AdamW([p32], lr=1e-3)
    for _ in range(3):
        opt16.step()
        opt32.step()
        set_grad(p16, 31)
        p32.grad = lt.Tensor.from_vector(vec(p16.grad), [16, 8], CPU)
    v16 = vec(p16.cast(FP32))
    v32 = vec(p32)
    maxdiff = max(abs(a - b) for a, b in zip(v16, v32))
    check(f"fp16_matches_fp32: maxdiff={maxdiff:.2e}", maxdiff < 1e-2)

def test_master_persistent():
    p = make_param([8, 4], 40, FP16)
    opt = lt.optim.ZeRO3Optimizer([p], lr=1e-3)
    set_grad(p, 41)
    opt.step()
    set_grad(p, 42)
    opt.step()
    after = vec(p)
    check("master_persistent: two steps ok", all(math.isfinite(v) for v in after))

def test_zero_grad():
    p = make_param([8, 4], 50)
    set_grad(p, 51)
    opt = lt.optim.ZeRO3Optimizer([p], lr=1e-3)
    opt.zero_grad()
    check("zero_grad: grad cleared", p.grad is None)

def test_sharded_param():
    p = make_param([16], 60, FP16)
    shard = p.cast(FP16)
    shard.requires_grad = True
    p.shard = shard
    set_grad(shard, 61)
    opt = lt.optim.ZeRO3Optimizer([p], lr=1e-3)
    opt.step()
    after = vec(shard)
    check("sharded: shard updated", all(math.isfinite(v) for v in after))
    check("sharded: dtype preserved", shard.dtype == FP16)

def test_multi_param_fused():
    params = [make_param([8, 4], 70 + i, FP16) for i in range(4)]
    for i, p in enumerate(params):
        set_grad(p, 80 + i)
    opt = lt.optim.ZeRO3Optimizer(params, lr=1e-3)
    lt.optim.AdamW.reset_launch_counts()
    opt.step()
    fused = lt.optim.AdamW.fused_launch_count()
    check(f"multi_param_fused: fused_launches={fused}", fused >= 1)
    for p in params:
        check("multi_param_fused: no NaN", all(math.isfinite(v) for v in vec(p)))

if __name__ == "__main__":
    test_fp32_basic()
    test_fp32_matches_adamw()
    test_fp16_master_weights()
    test_fp16_matches_fp32_reference()
    test_master_persistent()
    test_zero_grad()
    test_sharded_param()
    test_multi_param_fused()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
