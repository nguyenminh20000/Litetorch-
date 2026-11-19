import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

CONFIGS = [
    ("f32_512", 512, 512, 512, "float32", 11, 3),
    ("f32_4096", 4096, 4096, 4096, "float32", 7, 2),
    ("bf16_4096", 4096, 4096, 4096, "bfloat16", 7, 2),
    ("bf16_rect_1024x4096x4096", 1024, 4096, 4096, "bfloat16", 7, 2),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ns = ap.parse_args()
    workdir = ns.workdir
    out = {"role": "reference_standard_pytorch_xla", "runs": {}}
    try:
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm

        out["torch_version"] = torch.__version__
        out["torch_xla_version"] = getattr(torch_xla, "__version__", "unknown")
        device = torch_xla.device()
        out["device"] = str(device)
        rng = np.random.default_rng(20261119)
        for name, m, k, n, dt, iters, warmup in CONFIGS:
            dtype = torch.float32 if dt == "float32" else torch.bfloat16
            a_np = (rng.standard_normal((m, k)) * 0.5).astype(np.float32)
            b_np = (rng.standard_normal((k, n)) * 0.5).astype(np.float32)
            a = torch.from_numpy(a_np).to(dtype).to(device)
            b = torch.from_numpy(b_np).to(dtype).to(device)
            for _ in range(warmup):
                c = a @ b
                torch_xla.sync()
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                c = a @ b
                torch_xla.sync()
                times.append((time.perf_counter() - t0) * 1000.0)
            times.sort()
            median = times[len(times) // 2]
            entry = {"M": m, "K": k, "N": n, "dtype": dt,
                     "execute_median_ms": median, "execute_min_ms": times[0],
                     "approx_gflops": 2.0 * m * n * k / (median * 1e6), "iters": iters}
            if name == "f32_512":
                ref = torch.from_numpy(a_np) @ torch.from_numpy(b_np)
                got = c.float().cpu()
                entry["vs_torch_cpu_max_abs_diff"] = float((got - ref).abs().max())
            out["runs"][name] = entry
            del a, b, c
        a_np = np.fromfile(os.path.join(workdir, "gate_a_512.bin"), dtype=np.float32).reshape(512, 512)
        b_np = np.fromfile(os.path.join(workdir, "gate_b_512.bin"), dtype=np.float32).reshape(512, 512)
        a = torch.from_numpy(a_np).to(device)
        b = torch.from_numpy(b_np).to(device)
        c = a @ b
        torch_xla.sync()
        c_np = c.cpu().numpy().astype(np.float32)
        c_np.tofile(os.path.join(workdir, "out_torch_xla_512.bin"))
        ref = torch.from_numpy(a_np) @ torch.from_numpy(b_np)
        out["gate_512"] = {"vs_torch_cpu_max_abs_diff": float((c.cpu() - ref).abs().max()),
                           "output_file": "out_torch_xla_512.bin"}
    except Exception:
        out["error"] = traceback.format_exc()
    with open(os.path.join(workdir, "partial_torch_xla.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:4000])
    print("partial_torch_xla.json written")


if __name__ == "__main__":
    sys.exit(main())
