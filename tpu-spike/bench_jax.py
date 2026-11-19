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
    out = {"role": "perf_ceiling_reference_and_secondary_numerics_cross_check_only", "runs": {}}
    try:
        import jax
        import jax.numpy as jnp

        out["jax_version"] = jax.__version__
        try:
            import jaxlib
            out["jaxlib_version"] = jaxlib.__version__
        except Exception:
            out["jaxlib_version"] = "unknown"
        devices = jax.devices()
        out["devices"] = [str(d) for d in devices]
        out["default_backend"] = jax.default_backend()
        rng = np.random.default_rng(20261119)
        for name, m, k, n, dt, iters, warmup in CONFIGS:
            dtype = jnp.float32 if dt == "float32" else jnp.bfloat16
            a_np = (rng.standard_normal((m, k)) * 0.5).astype(np.float32)
            b_np = (rng.standard_normal((k, n)) * 0.5).astype(np.float32)
            a = jnp.asarray(a_np, dtype=dtype)
            b = jnp.asarray(b_np, dtype=dtype)
            for _ in range(warmup):
                c = a @ b
                c.block_until_ready()
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                c = a @ b
                c.block_until_ready()
                times.append((time.perf_counter() - t0) * 1000.0)
            times.sort()
            median = times[len(times) // 2]
            entry = {"M": m, "K": k, "N": n, "dtype": dt,
                     "execute_median_ms": median, "execute_min_ms": times[0],
                     "approx_gflops": 2.0 * m * n * k / (median * 1e6), "iters": iters}
            if name == "f32_512":
                ref = a_np @ b_np
                got = np.asarray(c, dtype=np.float32)
                entry["vs_numpy_max_abs_diff"] = float(np.abs(got - ref).max())
            out["runs"][name] = entry
            del a, b, c
    except Exception:
        out["error"] = traceback.format_exc()
    with open(os.path.join(workdir, "partial_jax.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:4000])
    print("partial_jax.json written")


if __name__ == "__main__":
    sys.exit(main())
