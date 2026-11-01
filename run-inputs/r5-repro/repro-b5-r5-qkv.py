import os
import sys
import math
import random

TREE = os.environ.get("LITETORCH_TREE") or (sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/litetorch")
sys.path.insert(0, os.path.join(TREE, "python"))

def _hook(t, v, tb):
    print("REPRO r5-flash-attention-qkv FAIL: %s: %s" % (t.__name__, v))
    sys.exit(1)
sys.excepthook = _hook

import litetorch as lt

print("FA3 lib present:")
for p in ("libflash_attn.so", "/usr/local/cuda/lib64/libflash_attn.so", "./libflash_attn.so"):
    print(" ", p, os.path.exists(p))

gpu = lt.Device("cuda")

def rand_qkv(rng, B, T, C3):
    vals = [rng.uniform(-1.0, 1.0) for _ in range(B * T * C3)]
    return vals

def sdpa_ref(qkv_vals, B, T, H, D):
    import torch
    import torch.nn.functional as F
    C = H * D
    qkv_t = torch.tensor(qkv_vals, dtype=torch.float32).reshape(B, T, 3 * C)
    q, k, v = qkv_t.split(C, dim=-1)
    q = q.reshape(B, T, H, D).transpose(1, 2)
    k = k.reshape(B, T, H, D).transpose(1, 2)
    v = v.reshape(B, T, H, D).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)

def ratio_stats(out_vals, ref_vals):
    ratios = []
    for o, r in zip(out_vals, ref_vals):
        if abs(r) > 1e-6:
            ratios.append(o / r)
    ratios.sort()
    n = len(ratios)
    return ratios[0], ratios[n // 2], ratios[-1]

def run_case(name, B, T, H, D, vals, expect_factor=None):
    C = H * D
    qkv = lt.Tensor.from_vector(vals, [B, T, 3 * C], gpu, False)
    out = lt.Ops.flash_attention_qkv(qkv, H)
    assert list(out.shape) == [B, H, T, D], (name, list(out.shape))
    out_vals = out.to_vector()
    print("[%s] lt out shape %s out_sum %.6f" % (name, list(out.shape), sum(out_vals)))
    try:
        ref = sdpa_ref(vals, B, T, H, D)
    except ImportError:
        print("[%s] TORCH-ABSENT" % name)
        return None
    ref_vals = ref.reshape(-1).tolist()
    diff = max(abs(a - b) for a, b in zip(out_vals, ref_vals))
    lo, med, hi = ratio_stats(out_vals, ref_vals)
    print("[%s] max_abs_diff %.10f ref_sum %.6f ratio min/med/max %.6f/%.6f/%.6f" % (
        name, diff, sum(ref_vals), lo, med, hi))
    if expect_factor is not None:
        print("[%s] expected factor 1/%.6f, median ratio %.6f" % (name, expect_factor, med))
    return diff

# Case 0: exact B5 config — reproduces the 0.525 diff.
vals0 = [0.1 * (i % 6 + 1) for i in range(192)]
d0 = run_case("b5-exact", 1, 4, 2, 8, vals0, expect_factor=8)

# Case 1: non-degenerate random data, same shape — discriminates factor vs layout.
rng = random.Random(0)
vals1 = rand_qkv(rng, 1, 4, 48)
d1 = run_case("random-T4-D8", 1, 4, 2, 8, vals1, expect_factor=8)

# Case 2: larger head dim — checks whether the factor tracks D (1/16) or stays 1/8.
vals2 = rand_qkv(rng, 1, 7, 192)
d2 = run_case("random-T7-D16", 1, 7, 4, 16, vals2, expect_factor=16)

# Case 3: plain flash_attention(q, k, v) on the case-1 data — isolates the
# shared fa_fwd_kernel from the QKV split path.
C = 16
qkv_t = None
try:
    import torch
    qkv_t = torch.tensor(vals1, dtype=torch.float32).reshape(1, 4, 48)
    q, k, v = qkv_t.split(16, dim=-1)
    q = q.reshape(1, 4, 2, 8).transpose(1, 2)
    k = k.reshape(1, 4, 2, 8).transpose(1, 2)
    v = v.reshape(1, 4, 2, 8).transpose(1, 2)
    qlt = lt.Tensor.from_vector(q.reshape(-1).tolist(), [1, 2, 4, 8], gpu, False)
    klt = lt.Tensor.from_vector(k.reshape(-1).tolist(), [1, 2, 4, 8], gpu, False)
    vlt = lt.Tensor.from_vector(v.reshape(-1).tolist(), [1, 2, 4, 8], gpu, False)
    out3 = lt.Ops.flash_attention(qlt, klt, vlt)
    out3_vals = out3.to_vector()
    ref3 = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    ref3_vals = ref3.reshape(-1).tolist()
    diff3 = max(abs(a - b) for a, b in zip(out3_vals, ref3_vals))
    lo, med, hi = ratio_stats(out3_vals, ref3_vals)
    print("[plain-fa] max_abs_diff %.10f ratio min/med/max %.6f/%.6f/%.6f" % (diff3, lo, med, hi))
except ImportError:
    print("[plain-fa] TORCH-ABSENT")

print("REPRO r5-flash-attention-qkv DONE b5=%.10f random=%.10f random16=%.10f" % (
    d0 if d0 is not None else -1.0, d1 if d1 is not None else -1.0, d2 if d2 is not None else -1.0))
