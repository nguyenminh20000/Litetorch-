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


def _spmd_collective_sum(stacked, device, device_ids, mesh):
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs

    x = torch.from_numpy(stacked).to(device)
    full_shape = torch.Size(stacked.shape)
    x = xs.enable_manual_sharding(x, (0, None, None), mesh=mesh).global_tensor
    for _ in range(2):
        y = torch_xla._XLAC._xla_spmd_all_reduce(xm.REDUCE_SUM, x, 1.0, [device_ids])
        xm.mark_step()
        xm.wait_device_ops()
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        y = torch_xla._XLAC._xla_spmd_all_reduce(xm.REDUCE_SUM, x, 1.0, [device_ids])
        xm.mark_step()
        xm.wait_device_ops()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    y = torch_xla._XLAC._xla_spmd_all_reduce(xm.REDUCE_SUM, x, 1.0, [device_ids])
    y = xs.disable_manual_sharding(y, (0, None, None), full_shape, mesh=mesh).global_tensor
    xm.mark_step()
    xm.wait_device_ops()
    return y.cpu().numpy().astype(np.float32), times[len(times) // 2], times[0]


def _spmd_allreduce_child(workdir):
    import torch_xla
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

    xr.use_spmd()
    n_devices = xr.global_runtime_device_count()
    if n_devices != 8:
        raise RuntimeError("SPMD all-reduce reference needs 8 devices, got %d" % n_devices)
    device_ids = list(range(n_devices))
    mesh = xs.Mesh(device_ids, (n_devices,), ("x",))
    xs.set_global_mesh(mesh)
    device = torch_xla.device()
    inputs = np.fromfile(os.path.join(workdir, "allreduce_inputs_1024.bin"), dtype=np.float32)
    inputs = inputs.reshape(8, 1024, 1024)
    out_np, med, mn = _spmd_collective_sum(inputs, device, device_ids, mesh)
    cpu_sum = inputs.sum(axis=0).astype(np.float32)
    expected = np.tile(cpu_sum, (8, 1, 1))
    diff = float(np.abs(out_np - expected).max())
    if diff >= 1e-4:
        raise RuntimeError("SPMD all-reduce reference disagrees with exact CPU sum by %g" % diff)
    out_np.tofile(os.path.join(workdir, "out_torch_xla_allreduce_1024.bin"))
    runs = {"gate_1024": {"size": 1024, "execute_median_ms": med, "execute_min_ms": mn, "iters": 7}}
    for size in ALLREDUCE_SIZES:
        rng = np.random.default_rng(20261124)
        stacked = (rng.standard_normal((8, size, size)) * 0.5).astype(np.float32)
        _, smed, smin = _spmd_collective_sum(stacked, device, device_ids, mesh)
        runs["bw_%d" % size] = {"size": size, "execute_median_ms": smed, "execute_min_ms": smin, "iters": 7}
    per_rank_runs = {}
    for rank in range(8):
        per_rank_runs["rank%d" % rank] = {"runs": dict(runs)}
    ar_entry = {"output_file": "out_torch_xla_allreduce_1024.bin",
                "mode": "single_process_spmd",
                "note": "one SPMD program across 8 devices; per-rank entries are the same single collective timing, not 8 independent measurements; bw inputs use one seed-20261124 RNG for the (8,size,size) stack",
                "per_rank_runs": per_rank_runs}
    for key in ("gate_1024", "bw_512", "bw_2048", "bw_4096"):
        medians = [per_rank_runs["rank%d" % r]["runs"][key]["execute_median_ms"] for r in range(8)]
        ar_entry[key] = {"execute_median_ms_max_rank": max(medians),
                         "execute_median_ms_rank0": medians[0],
                         "per_rank_median_ms": medians}
    ar_entry["gate_1024"]["vs_cpu_sum_max_abs_diff"] = diff
    return ar_entry


def _spmd_allreduce_entrypoint(workdir):
    entry = {"role": "reference_standard_pytorch_xla_spmd_allreduce"}
    try:
        entry["allreduce"] = _spmd_allreduce_child(workdir)
    except Exception:
        entry["error"] = traceback.format_exc()
        with open(os.path.join(workdir, "partial_torch_xla_allreduce.json"), "w") as f:
            json.dump(entry, f, indent=2)
        print(traceback.format_exc())
        return 1
    with open(os.path.join(workdir, "partial_torch_xla_allreduce.json"), "w") as f:
        json.dump(entry, f, indent=2)
    print(json.dumps(entry, indent=2)[:4000])
    print("partial_torch_xla_allreduce.json written")
    return 0


def _load_spmd_allreduce_reference(workdir):
    path = os.path.join(workdir, "partial_torch_xla_allreduce.json")
    with open(path) as f:
        child = json.load(f)
    if "allreduce" not in child:
        raise RuntimeError("SPMD all-reduce reference produced no allreduce entry: %s" % path)
    return child["allreduce"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--spmd-allreduce-only", action="store_true")
    ns = ap.parse_args()
    workdir = ns.workdir
    if ns.spmd_allreduce_only:
        return _spmd_allreduce_entrypoint(workdir)
    out = {"role": "reference_standard_pytorch_xla", "runs": {}}
    try:
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm

        out["torch_version"] = torch.__version__
        out["torch_xla_version"] = getattr(torch_xla, "__version__", "unknown")
        device = torch_xla.device()
        out["device"] = str(device)
        if not hasattr(xm, "wait_device_ops"):
            raise RuntimeError("xm.wait_device_ops unavailable; refusing unsynchronized timing")

        def sync_device():
            xm.mark_step()
            xm.wait_device_ops()

        rng = np.random.default_rng(20261119)
        for name, m, k, n, dt, iters, warmup in CONFIGS:
            dtype = torch.float32 if dt == "float32" else torch.bfloat16
            a_np = (rng.standard_normal((m, k)) * 0.5).astype(np.float32)
            b_np = (rng.standard_normal((k, n)) * 0.5).astype(np.float32)
            a = torch.from_numpy(a_np).to(dtype).to(device)
            b = torch.from_numpy(b_np).to(dtype).to(device)
            for _ in range(warmup):
                c = a @ b
                sync_device()
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                c = a @ b
                sync_device()
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
        sync_device()
        c_np = c.cpu().numpy().astype(np.float32)
        c_np.tofile(os.path.join(workdir, "out_torch_xla_512.bin"))
        ref = torch.from_numpy(a_np) @ torch.from_numpy(b_np)
        out["gate_512"] = {"vs_torch_cpu_max_abs_diff": float((c.cpu() - ref).abs().max()),
                           "output_file": "out_torch_xla_512.bin"}
        ba_np = np.fromfile(os.path.join(workdir, "gate_bf16_4096_a.bin"), dtype=np.float32).reshape(4096, 4096)
        bb_np = np.fromfile(os.path.join(workdir, "gate_bf16_4096_b.bin"), dtype=np.float32).reshape(4096, 4096)
        ba = torch.from_numpy(ba_np).to(torch.bfloat16).to(device)
        bb = torch.from_numpy(bb_np).to(torch.bfloat16).to(device)
        cb = ba @ bb
        sync_device()
        cb_np = cb.float().cpu().numpy().astype(np.float32)
        cb_np.tofile(os.path.join(workdir, "out_torch_xla_bf16_4096.bin"))
        out["gate_bf16_4096"] = {"output_file": "out_torch_xla_bf16_4096.bin",
                                 "note": "unsharded BF16 matmul on the shared BF16-rounded inputs, saved widened to FP32"}
        out["allreduce"] = _load_spmd_allreduce_reference(workdir)
    except Exception:
        out["error"] = traceback.format_exc()
    with open(os.path.join(workdir, "partial_torch_xla.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:4000])
    print("partial_torch_xla.json written")


if __name__ == "__main__":
    sys.exit(main())
