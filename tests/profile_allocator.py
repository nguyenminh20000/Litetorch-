import litetorch as lt
from litetorch import nn
import sys
import traceback

B, T = 1, 512
D_MODEL, NQ, NKV, D_FF = 2560, 20, 5, 6912
HEAD_DIM = D_MODEL // NQ
KV_DIM = HEAD_DIM * NKV

device = lt.Device("cuda:0")


def mem_bytes():
    try:
        return lt.cuda.memory_allocated()
    except Exception:
        pass
    try:
        return lt.get_gpu_memory_used()
    except Exception:
        return None


def mem_mb():
    b = mem_bytes()
    return None if b is None else b / (1024.0 * 1024.0)


def show(label, prev):
    cur = mem_mb()
    if cur is None:
        print(f"[mem] {label}: memory API unavailable", flush=True)
        return None
    d = "" if prev is None else f" (delta {cur - prev:+.2f} MB)"
    print(f"[mem] {label}: {cur:.2f} MB{d}", flush=True)
    return cur


def build_replica(fp16):
    def mk(mod):
        mod.to(device)
        if fp16:
            mod.half()
        return mod
    mods = {
        "attn_norm": mk(nn.RMSNorm([D_MODEL])),
        "q_norm": mk(nn.RMSNorm([HEAD_DIM])),
        "k_norm": mk(nn.RMSNorm([HEAD_DIM])),
        "ffn_norm": mk(nn.RMSNorm([D_MODEL])),
        "q_proj": mk(nn.Linear(D_MODEL, D_MODEL, False)),
        "k_proj": mk(nn.Linear(D_MODEL, KV_DIM, False)),
        "v_proj": mk(nn.Linear(D_MODEL, KV_DIM, False)),
        "o_proj": mk(nn.Linear(D_MODEL, D_MODEL, False)),
        "gate": mk(nn.Linear(D_MODEL, D_FF, False)),
        "up": mk(nn.Linear(D_MODEL, D_FF, False)),
        "down": mk(nn.Linear(D_FF, D_MODEL, False)),
    }
    return mods


def replica_forward(mods, x, cos, sin):
    stages = []
    h = mods["attn_norm"].forward(x)
    stages.append(("attn_norm", h))
    q = mods["q_proj"].forward(h)
    stages.append(("q_proj", q))
    k = mods["k_proj"].forward(h)
    stages.append(("k_proj", k))
    v = mods["v_proj"].forward(h)
    stages.append(("v_proj", v))
    q4 = q.view(B, T, NQ, HEAD_DIM).transpose(1, 2).contiguous()
    k4 = k.view(B, T, NKV, HEAD_DIM).transpose(1, 2).contiguous()
    v4 = v.view(B, T, NKV, HEAD_DIM).transpose(1, 2).contiguous()
    stages.append(("reshape_transpose", v4))
    q4 = mods["q_norm"].forward(q4)
    stages.append(("q_norm", q4))
    k4 = mods["k_norm"].forward(k4)
    stages.append(("k_norm", k4))
    q4 = lt.Ops.rope(q4, cos, sin)
    stages.append(("rope_q", q4))
    k4 = lt.Ops.rope(k4, cos, sin)
    stages.append(("rope_k", k4))
    a4 = lt.Ops.flash_attention(q4, k4, v4, True)
    stages.append(("flash_attention", a4))
    aflat = a4.transpose(1, 2).contiguous().view(B, T, D_MODEL)
    ao = mods["o_proj"].forward(aflat)
    stages.append(("o_proj", ao))
    x1 = x + ao
    stages.append(("residual1", x1))
    h2 = mods["ffn_norm"].forward(x1)
    stages.append(("ffn_norm", h2))
    g = mods["gate"].forward(h2)
    stages.append(("gate_proj", g))
    u = mods["up"].forward(h2)
    stages.append(("up_proj", u))
    sg = lt.Ops.mul(g, lt.Ops.sigmoid(g))
    sw = lt.Ops.mul(sg, u)
    stages.append(("swiglu_elemwise", sw))
    fo = mods["down"].forward(sw)
    stages.append(("down_proj", fo))
    out = x1 + fo
    stages.append(("residual2", out))
    loss = lt.Ops.sum(out)
    stages.append(("sum_loss", loss))
    return loss, stages


def trace_phase(fp16):
    tag = "FP16" if fp16 else "FP32"
    print(f"\n===== allocator trace [{tag}] =====", flush=True)
    if fp16:
        lt.cuda.set_fp16_math_enabled(True)
    mods = build_replica(fp16)
    x = lt.randn([B, T, D_MODEL], device, True)
    if fp16:
        x = x.cast(lt.DataType.FP16)
    cos = lt.randn([T, HEAD_DIM // 2], device, False)
    sin = lt.randn([T, HEAD_DIM // 2], device, False)
    params = []
    for m in mods.values():
        params.extend(m.parameters())
    params.append(x)
    prev = show(f"{tag} after replica build", None)
    loss, stages = replica_forward(mods, x, cos, sin)
    lt.cuda_synchronize()
    for sname, _ in stages:
        prev = show(f"{tag} fwd stage {sname}", prev)
    try:
        lt.cuda_synchronize()
        m0 = mem_mb()
        loss.backward()
        lt.cuda_synchronize()
        m1 = mem_mb()
        d = "" if (m0 is None or m1 is None) else f" (delta {m1 - m0:+.2f} MB)"
        print(f"[mem] {tag} after loss.backward(): {m1:.2f} MB{d}", flush=True)
        prev = m1
    except Exception as e:
        cur = mem_mb()
        print(f"[mem] {tag} BACKWARD FAILED at {cur} MB: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False
    for p in params:
        p.zero_grad()
    lt.cuda_synchronize()
    show(f"{tag} after zero_grad", prev)
    return True


def trace_real_block(fp16):
    tag = "FP16" if fp16 else "FP32"
    print(f"\n===== real GPTBlock iters [{tag}] =====", flush=True)
    block = nn.GPTBlock(D_MODEL, NQ, NKV, D_FF)
    block.to(device)
    if fp16:
        block.half()
        lt.cuda.set_fp16_math_enabled(True)
    x = lt.randn([B, T, D_MODEL], device, True)
    if fp16:
        x = x.cast(lt.DataType.FP16)
    params = block.parameters()
    prev = show(f"{tag} block built", None)
    base = prev
    ok = True
    for it in range(3):
        try:
            out = block.forward(x)
            lt.cuda_synchronize()
            prev = show(f"{tag} iter{it} forward", prev)
            loss = lt.Ops.sum(out)
            loss.backward()
            lt.cuda_synchronize()
            prev = show(f"{tag} iter{it} backward", prev)
        except Exception as e:
            cur = mem_mb()
            print(f"[mem] {tag} iter{it} FAILED at {cur} MB: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            ok = False
            break
        for p in params:
            p.zero_grad()
        lt.cuda_synchronize()
        cur = show(f"{tag} iter{it} zero_grad", prev)
        prev = cur
        if base is not None and cur is not None:
            print(f"[mem] {tag} iter{it} growth vs build baseline: {cur - base:+.2f} MB", flush=True)
    return ok


print("=== LiteTorch allocator trace: GPTBlock fwd+bwd, FP32 then FP16 ===", flush=True)
if mem_bytes() is None:
    print("memory API unavailable (lt.cuda.memory_allocated / lt.get_gpu_memory_used missing); exiting", flush=True)
    sys.exit(0)

ok32 = trace_phase(False)
ok32b = trace_real_block(False)
ok16 = trace_phase(True)
ok16b = trace_real_block(True)
print(f"\nsummary: fp32_replica_ok={ok32} fp32_block_ok={ok32b} fp16_replica_ok={ok16} fp16_block_ok={ok16b}")
print("done")
