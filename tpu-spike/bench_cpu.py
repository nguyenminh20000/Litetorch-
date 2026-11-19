import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

CONFIGS = [
    ("f32_512", 512, 512, 512, 5, 1),
    ("f32_4096", 4096, 4096, 4096, 3, 1),
    ("bf16_4096_as_f32", 4096, 4096, 4096, 3, 1),
    ("bf16_rect_1024x4096x4096_as_f32", 1024, 4096, 4096, 5, 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ns = ap.parse_args()
    workdir = ns.workdir
    out = {"role": "host_cpu_signature_numpy_f32", "runs": {}}
    try:
        out["numpy_version"] = np.__version__
        out["threads_note"] = "numpy default BLAS threading on the Kaggle host"
        rng = np.random.default_rng(20261119)
        for name, m, k, n, iters, warmup in CONFIGS:
            a = (rng.standard_normal((m, k)) * 0.5).astype(np.float32)
            b = (rng.standard_normal((k, n)) * 0.5).astype(np.float32)
            for _ in range(warmup):
                c = a @ b
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                c = a @ b
                times.append((time.perf_counter() - t0) * 1000.0)
            times.sort()
            median = times[len(times) // 2]
            out["runs"][name] = {"M": m, "K": k, "N": n, "dtype": "float32",
                                 "execute_median_ms": median, "execute_min_ms": times[0],
                                 "approx_gflops": 2.0 * m * n * k / (median * 1e6), "iters": iters}
            del a, b, c
    except Exception:
        out["error"] = traceback.format_exc()
    with open(os.path.join(workdir, "partial_cpu.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:4000])
    print("partial_cpu.json written")


if __name__ == "__main__":
    sys.exit(main())
