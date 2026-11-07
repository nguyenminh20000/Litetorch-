import os
import sys
import math
import random
import struct

TREE = os.environ.get("LITETORCH_TREE") or (sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/litetorch")
sys.path.insert(0, os.path.join(TREE, "python"))

def _hook(t, v, tb):
    print("REPRO softmax-warp-gpu-gate FAIL: %s: %s" % (t.__name__, v))
    sys.exit(1)
sys.excepthook = _hook

import litetorch as lt

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("TORCH-ABSENT")
    sys.exit(1)

print("FA3 lib present:")
fa3_present = False
for p in ("libflash_attn.so", "/usr/local/cuda/lib64/libflash_attn.so", "./libflash_attn.so"):
    hit = os.path.exists(p)
    fa3_present = fa3_present or hit
    print(" ", p, hit)

print("FWD-COVERAGE: softmax_fwd_warp_kernel and softmax_fwd_warp_half_kernel have no Python-reachable path on native CUDA without FA3: Ops.flash_attention forward with requires_grad saves LSE (forward_save_lse returns 1), so backward takes the with_lse branch, which rebuilds P with p_from_lse and never calls launch_softmax_fwd. The non-LSE backward (the only caller of launch_softmax_fwd) runs only when FA3 is present or the save_lse symbol is missing. This script therefore covers the BWD warp kernels exactly and reports FWD as NOT_COVERED, never as PASS.")
if fa3_present:
    print("FWD-COVERAGE: FA3 lib present, so the native backward itself is bypassed (g_fa3_bwd_fn): neither warp kernel is exercised by this run. All cases NOT_COVERED.")

gpu = lt.Device("cuda")
B = 1
H = 2
D = 8
SCALE = 1.0 / math.sqrt(D)
FP32_TOL = 1e-4
FP16_TOL16 = 1e-4
FP16_TOL32 = 1e-3
TKS = (1, 4, 16, 32, 33, 64)

def half_round(vals):
    return [struct.unpack("e", struct.pack("e", v))[0] for v in vals]

def rand_vals(rng, n):
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]

def max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))

def lt_build(q_vals, k_vals, v_vals, do_vals, Tq, Tk, dt):
    q = lt.Tensor.from_vector(q_vals, [B, H, Tq, D], gpu, True, dt)
    k = lt.Tensor.from_vector(k_vals, [B, H, Tk, D], gpu, True, dt)
    v = lt.Tensor.from_vector(v_vals, [B, H, Tk, D], gpu, True, dt)
    do = lt.Tensor.from_vector(do_vals, [B, H, Tq, D], gpu, False, dt)
    return q, k, v, do

def lt_grads(q, k, v):
    grads = {}
    for name, t in (("dq", q), ("dk", k), ("dv", v)):
        g = t.grad
        if g is None:
            raise RuntimeError("grad None for %s" % name)
        grads[name] = g.to_vector()
    return grads

def lt_case(Tk, causal, half):
    Tq = Tk
    n_q = B * H * Tq * D
    n_kv = B * H * Tk * D
    rng = random.Random(1000 + Tk)
    q_vals = rand_vals(rng, n_q)
    k_vals = rand_vals(rng, n_kv)
    v_vals = rand_vals(rng, n_kv)
    do_vals = rand_vals(random.Random(2000 + Tk), n_q)
    if half:
        q_vals = half_round(q_vals)
        k_vals = half_round(k_vals)
        v_vals = half_round(v_vals)
        do_vals = half_round(do_vals)
    dt = lt.DataType.FP16 if half else lt.DataType.FP32
    q, k, v, do = lt_build(q_vals, k_vals, v_vals, do_vals, Tq, Tk, dt)
    out = lt.Ops.flash_attention(q, k, v, causal)
    assert list(out.shape) == [B, H, Tq, D], (list(out.shape), [B, H, Tq, D])
    out_vals = out.to_vector()
    try:
        out.backward(do)
    except Exception as e:
        print("  backward(grad) raised %s: %s; rebuilding fresh tensors and retrying via scalar sum path" % (type(e).__name__, e))
        q, k, v, do = lt_build(q_vals, k_vals, v_vals, do_vals, Tq, Tk, dt)
        out = lt.Ops.flash_attention(q, k, v, causal)
        loss = lt.Ops.sum(lt.Ops.mul(out, do))
        loss.backward()
    grads = lt_grads(q, k, v)
    return out_vals, grads, q_vals, k_vals, v_vals, do_vals

def torch_ref(Tk, causal, half, q_vals, k_vals, v_vals, do_vals):
    Tq = Tk
    q32 = torch.tensor(q_vals, dtype=torch.float32).reshape(B, H, Tq, D).requires_grad_(True)
    k32 = torch.tensor(k_vals, dtype=torch.float32).reshape(B, H, Tk, D).requires_grad_(True)
    v32 = torch.tensor(v_vals, dtype=torch.float32).reshape(B, H, Tk, D).requires_grad_(True)
    do32 = torch.tensor(do_vals, dtype=torch.float32).reshape(B, H, Tq, D)
    out32 = F.scaled_dot_product_attention(q32, k32, v32, is_causal=causal)
    (out32 * do32).sum().backward()
    ref32 = {
        "out": out32.detach().reshape(-1).tolist(),
        "dq": q32.grad.reshape(-1).tolist(),
        "dk": k32.grad.reshape(-1).tolist(),
        "dv": v32.grad.reshape(-1).tolist(),
    }
    ref16 = None
    if half:
        try:
            q16 = torch.tensor(q_vals, dtype=torch.float16).reshape(B, H, Tq, D).requires_grad_(True)
            k16 = torch.tensor(k_vals, dtype=torch.float16).reshape(B, H, Tk, D).requires_grad_(True)
            v16 = torch.tensor(v_vals, dtype=torch.float16).reshape(B, H, Tk, D).requires_grad_(True)
            do16 = torch.tensor(do_vals, dtype=torch.float16).reshape(B, H, Tq, D)
            out16 = F.scaled_dot_product_attention(q16, k16, v16, is_causal=causal)
            (out16 * do16).sum().backward()
            ref16 = {
                "dq": [float(x) for x in q16.grad.reshape(-1).tolist()],
                "dk": [float(x) for x in k16.grad.reshape(-1).tolist()],
                "dv": [float(x) for x in v16.grad.reshape(-1).tolist()],
            }
        except Exception as e:
            print("  torch FP16 SDPA reference unavailable: %s: %s" % (type(e).__name__, e))
    return ref32, ref16

results = []
for Tk in TKS:
    for causal in (False, True):
        for half in (False, True):
            tag = "Tk=%d causal=%s half=%s" % (Tk, causal, half)
            kind = "warp" if Tk <= 32 else "fused"
            try:
                out_vals, grads, q_vals, k_vals, v_vals, do_vals = lt_case(Tk, causal, half)
                ref32, ref16 = torch_ref(Tk, causal, half, q_vals, k_vals, v_vals, do_vals)
                fwd_info = max_abs_diff(out_vals, ref32["out"])
                print("[%s %s] FWD-OUT-INFO max_abs_diff %.10f (fa_fwd_kernel, r5 track, NON-GATING; expect mismatch on a softmax-only tree)" % (kind, tag, fwd_info))
                d32 = {n: max_abs_diff(grads[n], ref32[n]) for n in ("dq", "dk", "dv")}
                if half:
                    if ref16 is not None:
                        d16 = {n: max_abs_diff(grads[n], ref16[n]) for n in ("dq", "dk", "dv")}
                        ok16 = all(d16[n] <= FP16_TOL16 for n in ("dq", "dk", "dv"))
                        ok32 = all(d32[n] <= FP16_TOL32 for n in ("dq", "dk", "dv"))
                        passed = ok16 or ok32
                        ref_used = "FP16" if ok16 else ("FP32" if ok32 else "none")
                        print("[%s %s] bwd dq/dk/dv vs FP16 %.10f/%.10f/%.10f vs FP32 %.10f/%.10f/%.10f ref=%s %s" % (
                            kind, tag, d16["dq"], d16["dk"], d16["dv"], d32["dq"], d32["dk"], d32["dv"], ref_used, "PASS" if passed else "FAIL"))
                    else:
                        ok32 = all(d32[n] <= FP16_TOL32 for n in ("dq", "dk", "dv"))
                        passed = ok32
                        print("[%s %s] bwd dq/dk/dv vs FP32 %.10f/%.10f/%.10f ref=FP32(ref16-unavailable) %s" % (
                            kind, tag, d32["dq"], d32["dk"], d32["dv"], "PASS" if passed else "FAIL"))
                else:
                    passed = all(d32[n] <= FP32_TOL for n in ("dq", "dk", "dv"))
                    print("[%s %s] bwd dq/dk/dv vs FP32 %.10f/%.10f/%.10f %s" % (
                        kind, tag, d32["dq"], d32["dk"], d32["dv"], "PASS" if passed else "FAIL"))
                results.append((tag, passed))
            except Exception as e:
                print("[%s %s] bwd ERROR %s: %s FAIL" % (kind, tag, type(e).__name__, e))
                results.append((tag, False))

n_pass = sum(1 for _, p in results if p)
print("SOFTMAX-WARP-GATE SUMMARY bwd %d/%d PASS" % (n_pass, len(results)))
for tag, p in results:
    print("  %s %s" % (tag, "PASS" if p else "FAIL"))
print("SOFTMAX-WARP-GATE RESULT: BWD %d/%d PASS; FWD warp kernels NOT_COVERED (no Python-reachable path on native CUDA without FA3); VERDICT INCOMPLETE" % (n_pass, len(results)))
fwd_covered = False
sys.exit(0 if (n_pass == len(results) and fwd_covered and not fa3_present) else 1)
