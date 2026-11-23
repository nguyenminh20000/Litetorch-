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


def _allreduce_worker(rank, workdir):
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    entry = {"rank": rank, "device": str(device), "runs": {}}
    inputs = np.fromfile(os.path.join(workdir, "allreduce_inputs_1024.bin"), dtype=np.float32)
    inputs = inputs.reshape(8, 1024, 1024)
    x = torch.from_numpy(inputs[rank].copy()).to(device)
    for _ in range(2):
        y = xm.all_reduce(xm.REDUCE_SUM, x)
        xm.mark_step()
        xm.wait_device_ops()
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        y = xm.all_reduce(xm.REDUCE_SUM, x)
        xm.mark_step()
        xm.wait_device_ops()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    entry["runs"]["gate_1024"] = {"size": 1024, "execute_median_ms": times[len(times) // 2],
                                  "execute_min_ms": times[0], "iters": 7}
    y_np = y.cpu().numpy().astype(np.float32)
    y_np.tofile(os.path.join(workdir, "out_torch_xla_allreduce_1024_rank%d.bin" % rank))
    for size in ALLREDUCE_SIZES:
        rng = np.random.default_rng(20261124 + rank)
        a_np = (rng.standard_normal((size, size)) * 0.5).astype(np.float32)
        a = torch.from_numpy(a_np).to(device)
        for _ in range(2):
            z = xm.all_reduce(xm.REDUCE_SUM, a)
            xm.mark_step()
            xm.wait_device_ops()
        stimes = []
        for _ in range(7):
            t0 = time.perf_counter()
            z = xm.all_reduce(xm.REDUCE_SUM, a)
            xm.mark_step()
            xm.wait_device_ops()
            stimes.append((time.perf_counter() - t0) * 1000.0)
        stimes.sort()
        entry["runs"]["bw_%d" % size] = {"size": size, "execute_median_ms": stimes[len(stimes) // 2],
                                         "execute_min_ms": stimes[0], "iters": 7}
        del a, z
    with open(os.path.join(workdir, "partial_torch_xla_allreduce_rank%d.json" % rank), "w") as f:
        json.dump(entry, f, indent=2)


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
        import torch_xla.distributed.xla_multiprocessing as xmp

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
        xmp.spawn(_allreduce_worker, args=(workdir,), nprocs=8, start_method="fork")
        ar_runs = {}
        slices = []
        for rank in range(8):
            with open(os.path.join(workdir, "partial_torch_xla_allreduce_rank%d.json" % rank)) as f:
                rentry = json.load(f)
            ar_runs["rank%d" % rank] = rentry["runs"]
            slices.append(np.fromfile(os.path.join(workdir, "out_torch_xla_allreduce_1024_rank%d.bin" % rank), dtype=np.float32))
        np.concatenate(slices).tofile(os.path.join(workdir, "out_torch_xla_allreduce_1024.bin"))
        ar_entry = {"output_file": "out_torch_xla_allreduce_1024.bin", "per_rank_runs": ar_runs}
        for key in ("gate_1024", "bw_512", "bw_2048", "bw_4096"):
            medians = [ar_runs["rank%d" % r][key]["execute_median_ms"] for r in range(8)]
            ar_entry[key] = {"execute_median_ms_max_rank": max(medians),
                             "execute_median_ms_rank0": medians[0],
                             "per_rank_median_ms": medians}
        out["allreduce"] = ar_entry
    except Exception:
        out["error"] = traceback.format_exc()
    with open(os.path.join(workdir, "partial_torch_xla.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:4000])
    print("partial_torch_xla.json written")


if __name__ == "__main__":
    sys.exit(main())
