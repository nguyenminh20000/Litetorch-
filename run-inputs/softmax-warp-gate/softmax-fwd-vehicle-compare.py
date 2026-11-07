import os
import struct
import sys
import math

FP32_TOL = 1e-4
FP16_TOL16 = 1e-4
FP16_TOL32 = 1e-3
TKS = (1, 4, 16, 32, 33, 64)

def read_meta(casedir):
    meta = {}
    with open(os.path.join(casedir, "meta.txt")) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
    return meta

def read_f32(casedir, name, n):
    with open(os.path.join(casedir, name), "rb") as f:
        raw = f.read()
    if len(raw) != 4 * n:
        raise RuntimeError("%s size %d, expected %d bytes" % (name, len(raw), 4 * n))
    return list(struct.unpack("<%df" % n, raw))

def read_f16(casedir, name, n):
    with open(os.path.join(casedir, name), "rb") as f:
        raw = f.read()
    if len(raw) != 2 * n:
        raise RuntimeError("%s size %d, expected %d bytes" % (name, len(raw), 2 * n))
    bits = struct.unpack("<%dH" % n, raw)
    return [struct.unpack("<e", struct.pack("<H", b))[0] for b in bits]

def max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))

def torch_ref(meta, q_vals, k_vals, v_vals, do_vals):
    import torch
    import torch.nn.functional as F
    B = int(meta["B"])
    H = int(meta["H"])
    Tq = int(meta["Tq"])
    Tk = int(meta["Tk"])
    D = int(meta["D"])
    causal = meta["causal"] == "1"
    half = meta["half"] == "1"
    q32 = torch.tensor(q_vals, dtype=torch.float32).reshape(B, H, Tq, D).requires_grad_(True)
    k32 = torch.tensor(k_vals, dtype=torch.float32).reshape(B, H, Tk, D).requires_grad_(True)
    v32 = torch.tensor(v_vals, dtype=torch.float32).reshape(B, H, Tk, D).requires_grad_(True)
    do32 = torch.tensor(do_vals, dtype=torch.float32).reshape(B, H, Tq, D)
    out32 = F.scaled_dot_product_attention(q32, k32, v32, is_causal=causal)
    (out32 * do32).sum().backward()
    ref32 = {
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

def run_case(casedir):
    meta = read_meta(casedir)
    Tk = int(meta["Tk"])
    causal = meta["causal"] == "1"
    half = meta["half"] == "1"
    n_q = int(meta["n_q"])
    n_kv = int(meta["n_kv"])
    if int(meta["B"]) != 1 or int(meta["H"]) != 2 or int(meta["D"]) != 8 or int(meta["Tq"]) != Tk:
        raise RuntimeError("unexpected meta shape B/H/D/Tq: %s" % meta)
    if half:
        q_vals = read_f16(casedir, "q.bin", n_q)
        k_vals = read_f16(casedir, "k.bin", n_kv)
        v_vals = read_f16(casedir, "v.bin", n_kv)
        do_vals = read_f16(casedir, "do.bin", n_q)
        grads = {
            "dq": read_f16(casedir, "dq.bin", n_q),
            "dk": read_f16(casedir, "dk.bin", n_kv),
            "dv": read_f16(casedir, "dv.bin", n_kv),
        }
    else:
        q_vals = read_f32(casedir, "q.bin", n_q)
        k_vals = read_f32(casedir, "k.bin", n_kv)
        v_vals = read_f32(casedir, "v.bin", n_kv)
        do_vals = read_f32(casedir, "do.bin", n_q)
        grads = {
            "dq": read_f32(casedir, "dq.bin", n_q),
            "dk": read_f32(casedir, "dk.bin", n_kv),
            "dv": read_f32(casedir, "dv.bin", n_kv),
        }
    ref32, ref16 = torch_ref(meta, q_vals, k_vals, v_vals, do_vals)
    kind = "warp" if Tk <= 32 else "fused"
    tag = "Tk=%d causal=%s half=%s" % (Tk, causal, half)
    d32 = {n: max_abs_diff(grads[n], ref32[n]) for n in ("dq", "dk", "dv")}
    if half:
        if ref16 is not None:
            d16 = {n: max_abs_diff(grads[n], ref16[n]) for n in ("dq", "dk", "dv")}
            ok16 = all(d16[n] <= FP16_TOL16 for n in ("dq", "dk", "dv"))
            ok32 = all(d32[n] <= FP16_TOL32 for n in ("dq", "dk", "dv"))
            passed = ok16 or ok32
            ref_used = "FP16" if ok16 else ("FP32" if ok32 else "none")
            print("[%s %s] fwd-path=non-LSE dq/dk/dv vs FP16 %.10f/%.10f/%.10f vs FP32 %.10f/%.10f/%.10f ref=%s %s" % (
                kind, tag, d16["dq"], d16["dk"], d16["dv"], d32["dq"], d32["dk"], d32["dv"], ref_used, "PASS" if passed else "FAIL"))
        else:
            ok32 = all(d32[n] <= FP16_TOL32 for n in ("dq", "dk", "dv"))
            passed = ok32
            print("[%s %s] fwd-path=non-LSE dq/dk/dv vs FP32 %.10f/%.10f/%.10f ref=FP32(ref16-unavailable) %s" % (
                kind, tag, d32["dq"], d32["dk"], d32["dv"], "PASS" if passed else "FAIL"))
    else:
        passed = all(d32[n] <= FP32_TOL for n in ("dq", "dk", "dv"))
        print("[%s %s] fwd-path=non-LSE dq/dk/dv vs FP32 %.10f/%.10f/%.10f %s" % (
            kind, tag, d32["dq"], d32["dk"], d32["dv"], "PASS" if passed else "FAIL"))
    return tag, passed

def main():
    try:
        import torch
    except ImportError:
        print("TORCH-ABSENT")
        return 1
    cases = []
    if len(sys.argv) == 3 and sys.argv[1] == "--all":
        root = sys.argv[2]
        for Tk in TKS:
            for causal in (0, 1):
                for half in (0, 1):
                    casedir = os.path.join(root, "Tk%d-causal%d-half%d" % (Tk, causal, half))
                    cases.append(casedir)
    elif len(sys.argv) == 2:
        cases.append(sys.argv[1])
    else:
        print("usage: compare.py <casedir> | compare.py --all <rootdir>")
        return 1
    print("FWD-PATH: non-LSE backward (gpu_flash_attention_backward / _half); P recomputed by launch_softmax_fwd / _half, not by p_from_lse")
    print("FWD-PATH: O input is unused on the native non-LSE path (flash_attention.cu:1225-1299, 1370-1445); the vehicle passes a valid zero-filled O buffer")
    results = []
    for casedir in cases:
        tag = os.path.basename(casedir.rstrip("/"))
        try:
            _, passed = run_case(casedir)
            results.append((tag, passed))
        except Exception as e:
            print("[%s] ERROR %s: %s FAIL" % (tag, type(e).__name__, e))
            results.append((tag, False))
    n_pass = sum(1 for _, p in results if p)
    print("SOFTMAX-FWD-VEHICLE SUMMARY %d/%d PASS" % (n_pass, len(results)))
    for tag, p in results:
        print("  %s %s" % (tag, "PASS" if p else "FAIL"))
    print("SOFTMAX-FWD-VEHICLE RESULT: FWD %d/%d PASS (non-LSE backward path; P recomputed by launch_softmax_fwd)" % (n_pass, len(results)))
    return 0 if n_pass == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
