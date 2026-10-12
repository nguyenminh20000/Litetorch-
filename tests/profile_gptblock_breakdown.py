import litetorch as lt
from litetorch import nn
import time
import sys

use_fp16 = "--fp16" in sys.argv
dtype_name = "FP16" if use_fp16 else "FP32"

B, T = 1, 512
D_MODEL, NQ, NKV, D_FF = 2560, 20, 5, 6912
HEAD_DIM = D_MODEL // NQ
WARMUP = 3
ITERS = 20

device = lt.Device("cuda:0")


def sync():
    lt.cuda_synchronize()


def prep_module(mod):
    mod.to(device)
    if use_fp16:
        mod.half()
        lt.cuda.set_fp16_math_enabled(True)
    return mod


def prep_input(t):
    if use_fp16:
        t = t.cast(lt.DataType.FP16)
    return t


def zero_all(params):
    for p in params:
        p.zero_grad()


def bench(name, fwd_fn, params):
    for _ in range(WARMUP):
        loss = fwd_fn()
        sync()
        loss.backward()
        sync()
        zero_all(params)
    sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        loss = fwd_fn()
    sync()
    t1 = time.perf_counter()
    fwd_ms = (t1 - t0) / ITERS * 1000.0
    bwd_ms = 0.0
    for _ in range(ITERS):
        loss = fwd_fn()
        sync()
        tb0 = time.perf_counter()
        loss.backward()
        sync()
        tb1 = time.perf_counter()
        bwd_ms += (tb1 - tb0) * 1000.0
        zero_all(params)
    bwd_ms /= ITERS
    return fwd_ms, bwd_ms


results = []


def run_stage(name, fwd_fn, params):
    fwd_ms, bwd_ms = bench(name, fwd_fn, params)
    results.append((name, fwd_ms, bwd_ms))
    print(f"[{dtype_name}] stage {name}: fwd {fwd_ms:.2f} ms | bwd {bwd_ms:.2f} ms | total {(fwd_ms + bwd_ms):.2f} ms", flush=True)


print(f"=== GPTBlock per-stage breakdown [{dtype_name}] B={B} T={T} d_model={D_MODEL} ===", flush=True)

block = prep_module(nn.GPTBlock(D_MODEL, NQ, NKV, D_FF))
bx = prep_input(lt.randn([B, T, D_MODEL], device, True))
bparams = block.parameters() + [bx]


def block_fwd():
    out = block.forward(bx)
    return lt.Ops.sum(out)


full_fwd, full_bwd = bench("full_block", block_fwd, bparams)
results.append(("full_block", full_fwd, full_bwd))
print(f"[{dtype_name}] stage full_block: fwd {full_fwd:.2f} ms | bwd {full_bwd:.2f} ms | total {(full_fwd + full_bwd):.2f} ms", flush=True)

norm = prep_module(nn.RMSNorm([D_MODEL]))
nx = prep_input(lt.randn([B, T, D_MODEL], device, True))
run_stage("attn_norm_rmsnorm", lambda: lt.Ops.sum(norm.forward(nx)), norm.parameters() + [nx])

for lname, li, lo in [("proj_q", D_MODEL, D_MODEL), ("proj_k", D_MODEL, D_MODEL // NQ * NKV),
                      ("proj_v", D_MODEL, D_MODEL // NQ * NKV), ("proj_o", D_MODEL, D_MODEL)]:
    lin = prep_module(nn.Linear(li, lo, False))
    lx = prep_input(lt.randn([B, T, li], device, True))
    run_stage(lname, lambda lin=lin, lx=lx: lt.Ops.sum(lin.forward(lx)), lin.parameters() + [lx])

fq = prep_input(lt.randn([B, NQ, T, HEAD_DIM], device, True))
fk = prep_input(lt.randn([B, NKV, T, HEAD_DIM], device, True))
fv = prep_input(lt.randn([B, NKV, T, HEAD_DIM], device, True))
run_stage("flash_attention", lambda: lt.Ops.sum(lt.Ops.flash_attention(fq, fk, fv, True)), [fq, fk, fv])

rx = prep_input(lt.randn([B, NQ, T, HEAD_DIM], device, True))
rcos = lt.randn([T, HEAD_DIM // 2], device, False)
rsin = lt.randn([T, HEAD_DIM // 2], device, False)
run_stage("rope", lambda: lt.Ops.sum(lt.Ops.rope(rx, rcos, rsin)), [rx])

for lname, li, lo in [("ffn_gate", D_MODEL, D_FF), ("ffn_up", D_MODEL, D_FF), ("ffn_down", D_FF, D_MODEL)]:
    lin = prep_module(nn.Linear(li, lo, False))
    lx = prep_input(lt.randn([B, T, li], device, True))
    run_stage(lname, lambda lin=lin, lx=lx: lt.Ops.sum(lin.forward(lx)), lin.parameters() + [lx])

gx = prep_input(lt.randn([B, T, D_FF], device, True))
ux = prep_input(lt.randn([B, T, D_FF], device, True))
run_stage("swiglu_elemwise", lambda: lt.Ops.sum(lt.Ops.mul(lt.Ops.mul(gx, lt.Ops.sigmoid(gx)), ux)), [gx, ux])

ax = prep_input(lt.randn([B, T, D_MODEL], device, True))
ay = prep_input(lt.randn([B, T, D_MODEL], device, True))
run_stage("residual_add", lambda: lt.Ops.sum(ax + ay), [ax, ay])

total = full_fwd + full_bwd
print(f"\n=== per-stage table [{dtype_name}] (full_block total {total:.2f} ms = 100%) ===")
print(f"{'stage':<18} {'fwd_ms':>9} {'bwd_ms':>9} {'total_ms':>9} {'pct_of_full':>11}")
for name, fwd_ms, bwd_ms in results:
    tot = fwd_ms + bwd_ms
    pct = tot / total * 100.0 if total > 0 else 0.0
    print(f"{name:<18} {fwd_ms:>9.2f} {bwd_ms:>9.2f} {tot:>9.2f} {pct:>10.1f}%")
micro_sum = sum(f + b for n, f, b in results if n != "full_block")
print(f"{'micro_sum':<18} {'':>9} {'':>9} {micro_sum:>9.2f} {(micro_sum / total * 100.0 if total > 0 else 0):>10.1f}%")
print("done")
