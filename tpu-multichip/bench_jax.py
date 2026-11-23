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

ALLREDUCE_SIZES = [512, 2048, 4096]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ns = ap.parse_args()
    workdir = ns.workdir
    out = {"role": "perf_ceiling_reference_and_secondary_numerics_cross_check_only", "runs": {}}
    try:
        import jax
        import jax.numpy as jnp
        from jax import lax

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
        if len(devices) < 8:
            raise RuntimeError("JAX multi-device baselines need 8 devices, got %d" % len(devices))

        @jax.pmap(axis_name="i")
        def allreduce_pmap(x):
            return lax.psum(x, axis_name="i")

        ar_entry = {}
        for size in ALLREDUCE_SIZES:
            ar_rng = np.random.default_rng(20261124)
            stacked = (ar_rng.standard_normal((8, size, size)) * 0.5).astype(np.float32)
            x = jnp.asarray(stacked)
            for _ in range(2):
                y = allreduce_pmap(x)
                y.block_until_ready()
            times = []
            for _ in range(7):
                t0 = time.perf_counter()
                y = allreduce_pmap(x)
                y.block_until_ready()
                times.append((time.perf_counter() - t0) * 1000.0)
            times.sort()
            median = times[len(times) // 2]
            ar_entry["bw_%d" % size] = {"size": size, "bytes_per_device": size * size * 4,
                                        "execute_median_ms": median, "execute_min_ms": times[0],
                                        "gbps": (size * size * 4) / median / 1e6, "iters": 7}
            del x, y
        out["allreduce_pmap"] = ar_entry

        @jax.pmap(axis_name="i")
        def rowshard_pmap(a_shard, b):
            return a_shard @ b

        shard_entry = {}
        for name, m, k, n, dt, iters, warmup in [("f32_4096", 4096, 4096, 4096, "float32", 7, 2),
                                                 ("bf16_4096", 4096, 4096, 4096, "bfloat16", 7, 2)]:
            dtype = jnp.float32 if dt == "float32" else jnp.bfloat16
            s_rng = np.random.default_rng(20261119)
            a_np = (s_rng.standard_normal((m, k)) * 0.5).astype(np.float32)
            b_np = (s_rng.standard_normal((k, n)) * 0.5).astype(np.float32)
            a_stack = np.stack(np.split(a_np, 8, axis=0), axis=0).astype(np.float32)
            a = jnp.asarray(a_stack, dtype=dtype)
            b = jnp.asarray(b_np, dtype=dtype)
            for _ in range(warmup):
                c = rowshard_pmap(a, b)
                c.block_until_ready()
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                c = rowshard_pmap(a, b)
                c.block_until_ready()
                times.append((time.perf_counter() - t0) * 1000.0)
            times.sort()
            median = times[len(times) // 2]
            shard_entry[name] = {"M": m, "K": k, "N": n, "dtype": dt,
                                 "execute_median_ms": median, "execute_min_ms": times[0],
                                 "approx_gflops": 2.0 * m * n * k / (median * 1e6), "iters": iters}
            del a, b, c
        out["rowshard_pmap"] = shard_entry
    except Exception:
        out["error"] = traceback.format_exc()
    with open(os.path.join(workdir, "partial_jax.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:4000])
    print("partial_jax.json written")


if __name__ == "__main__":
    sys.exit(main())
