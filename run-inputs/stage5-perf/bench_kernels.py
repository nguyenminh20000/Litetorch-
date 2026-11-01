import argparse
import json
import os
import statistics
import sys
import time

try:
    import torch
    HAVE_TORCH = True
except Exception as e:
    HAVE_TORCH = False
    TORCH_ERR = str(e)

try:
    import litetorch as LT
    HAVE_LT = True
except Exception as e:
    HAVE_LT = False
    LT_ERR = str(e)


def lt_fn(*names):
    if not HAVE_LT:
        return None
    for n in names:
        f = getattr(LT, n, None)
        if callable(f):
            return f
    return None


def sync(device):
    if HAVE_TORCH and device == "cuda":
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    if HAVE_LT:
        for name in ("cuda_synchronize", "synchronize", "sync"):
            f = getattr(LT, name, None)
            if callable(f):
                try:
                    f()
                except Exception:
                    pass
                break


def time_fn(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    sync(device)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    n = len(ts)
    return {
        "median_ms": ts[n // 2],
        "mean_ms": sum(ts) / n,
        "p95_ms": ts[int(n * 0.95)],
        "iters": n,
    }


def skip(reason):
    return {"skipped": True, "reason": reason}


def torch_tensor(shape, dtype, device):
    tdtype = {"fp32": torch.float32, "uint8": torch.uint8, "int64": torch.int64}[dtype]
    kw = {"device": device} if device in ("cuda", "cpu") else {}
    return torch.randint(0, 255, shape, dtype=tdtype, **kw).to(tdtype)


def bench_matmul(args, device):
    out = {"name": "matmul_fp32_1024"}
    if HAVE_TORCH:
        td = "cuda" if device == "cuda" else "cpu"
        a = torch.randn(1024, 1024, device=td)
        b = torch.randn(1024, 1024, device=td)
        out["torch"] = time_fn(lambda: torch.matmul(a, b), device, args.warmup, args.iters)
    else:
        out["torch"] = skip("torch not importable: " + TORCH_ERR)
    mm = lt_fn("matmul", "mm", "mat_mul")
    mk = lt_fn("tensor", "from_torch", "from_numpy", "Tensor")
    if mm is None or mk is None:
        out["litetorch"] = skip("litetorch matmul/tensor constructor not found in module")
    else:
        try:
            la = mk(a) if HAVE_TORCH else None
            lb = mk(b) if HAVE_TORCH else None
            out["litetorch"] = time_fn(lambda: mm(la, lb), device, args.warmup, args.iters)
        except Exception as e:
            out["litetorch"] = skip("litetorch matmul failed: " + str(e))
    return out


def bench_ew_add(args, device, dtype):
    out = {"name": "ew_add_" + dtype + "_16M"}
    n = 16 * 1024 * 1024
    if HAVE_TORCH:
        td = "cuda" if device == "cuda" else "cpu"
        a = torch_tensor((n,), dtype, td)
        b = torch_tensor((n,), dtype, td)
        out["torch"] = time_fn(lambda: torch.add(a, b), device, args.warmup, args.iters)
    else:
        out["torch"] = skip("torch not importable")
    add = lt_fn("add", "ew_add", "elementwise_add")
    mk = lt_fn("tensor", "from_torch", "from_numpy", "Tensor")
    if add is None or mk is None or not HAVE_TORCH:
        out["litetorch"] = skip("litetorch add/tensor constructor not found")
    else:
        try:
            la, lb = mk(a), mk(b)
            out["litetorch"] = time_fn(lambda: add(la, lb), device, args.warmup, args.iters)
        except Exception as e:
            out["litetorch"] = skip("litetorch add failed: " + str(e))
    return out


def bench_conv2d(args, device):
    out = {"name": "conv2d_3x64x32x32_k3"}
    if HAVE_TORCH:
        import torch.nn.functional as F
        td = "cuda" if device == "cuda" else "cpu"
        x = torch.randn(3, 64, 32, 32, device=td)
        w = torch.randn(64, 64, 3, 3, device=td)
        out["torch"] = time_fn(lambda: F.conv2d(x, w, padding=1), device, args.warmup, args.iters)
    else:
        out["torch"] = skip("torch not importable")
    conv = lt_fn("conv2d", "conv_2d")
    mk = lt_fn("tensor", "from_torch", "from_numpy", "Tensor")
    if conv is None or mk is None or not HAVE_TORCH:
        out["litetorch"] = skip("litetorch conv2d/tensor constructor not found")
    else:
        try:
            lx, lw = mk(x), mk(w)
            out["litetorch"] = time_fn(lambda: conv(lx, lw, 1), device, args.warmup, args.iters)
        except Exception as e:
            out["litetorch"] = skip("litetorch conv2d failed: " + str(e))
    return out


def bench_softmax(args, device):
    out = {"name": "softmax_1M"}
    if HAVE_TORCH:
        td = "cuda" if device == "cuda" else "cpu"
        x = torch.randn(1000000, device=td)
        out["torch"] = time_fn(lambda: torch.softmax(x, dim=0), device, args.warmup, args.iters)
    else:
        out["torch"] = skip("torch not importable")
    sm = lt_fn("softmax")
    mk = lt_fn("tensor", "from_torch", "from_numpy", "Tensor")
    if sm is None or mk is None or not HAVE_TORCH:
        out["litetorch"] = skip("litetorch softmax/tensor constructor not found")
    else:
        try:
            lx = mk(x)
            out["litetorch"] = time_fn(lambda: sm(lx, 0), device, args.warmup, args.iters)
        except Exception as e:
            out["litetorch"] = skip("litetorch softmax failed: " + str(e))
    return out


def bench_reduce(args, device):
    out = {"name": "reduce_sum_16M"}
    n = 16 * 1024 * 1024
    if HAVE_TORCH:
        td = "cuda" if device == "cuda" else "cpu"
        x = torch.randn(n, device=td)
        out["torch"] = time_fn(lambda: torch.sum(x), device, args.warmup, args.iters)
    else:
        out["torch"] = skip("torch not importable")
    sm = lt_fn("sum", "reduce_sum")
    mk = lt_fn("tensor", "from_torch", "from_numpy", "Tensor")
    if sm is None or mk is None or not HAVE_TORCH:
        out["litetorch"] = skip("litetorch sum/tensor constructor not found")
    else:
        try:
            lx = mk(x)
            out["litetorch"] = time_fn(lambda: sm(lx), device, args.warmup, args.iters)
        except Exception as e:
            out["litetorch"] = skip("litetorch sum failed: " + str(e))
    return out


def _allreduce_worker(rank, world, port, iters, warmup, q):
    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    t = torch.randn(1024 * 1024)
    for _ in range(warmup):
        dist.all_reduce(t)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        dist.all_reduce(t)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    if rank == 0:
        q.put({"median_ms": ts[len(ts) // 2], "mean_ms": sum(ts) / len(ts), "iters": len(ts)})
    dist.destroy_process_group()


def bench_allreduce(args, device):
    out = {"name": "allreduce_2rank_4MB"}
    if not HAVE_TORCH:
        out["torch"] = skip("torch not importable")
        out["litetorch"] = skip("torch missing; litetorch distributed API unprobed")
        return out
    try:
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_allreduce_worker, args=(0, 2, 29577, 20, 5, q))
        c = ctx.Process(target=_allreduce_worker, args=(1, 2, 29577, 20, 5, q))
        p.start()
        c.start()
        p.join(300)
        c.join(300)
        if p.exitcode == 0 and c.exitcode == 0 and not q.empty():
            out["torch"] = q.get()
        else:
            out["torch"] = skip("gloo 2-rank run failed")
    except Exception as e:
        out["torch"] = skip("gloo 2-rank run failed: " + str(e))
    init_pg = lt_fn("init_process_group", "dist_init")
    ar = lt_fn("all_reduce", "dist_all_reduce")
    if init_pg is None or ar is None:
        out["litetorch"] = skip("litetorch distributed API not found; run on Kaggle with built bindings")
    else:
        out["litetorch"] = skip("litetorch 2-rank all_reduce needs manual rank launch; see perf-gate.md")
    return out


def bench_onnx_export(args, device):
    out = {"name": "onnx_export_mlp"}
    if HAVE_TORCH:
        import torch.nn as nn
        m = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
        m.eval()
        x = torch.randn(1, 256)
        def do_export():
            torch.onnx.export(m, x, "/tmp/mlp_bench.onnx", input_names=["x"], output_names=["y"],
                              dynamo=False)
        try:
            out["torch"] = time_fn(do_export, "cpu", 2, 5)
        except Exception as e:
            out["torch"] = skip("torch.onnx.export failed: " + str(e))
    else:
        out["torch"] = skip("torch not importable")
    exp = lt_fn("export_onnx", "export_onnx_binary", "onnx_export")
    if exp is None:
        out["litetorch"] = skip("litetorch onnx export API not found")
    else:
        out["litetorch"] = skip("litetorch onnx export needs a traced module handle; wire to Phase 5 export API on Kaggle")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "tpu"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default="bench_results.json")
    ap.add_argument("--record-baseline", action="store_true")
    args = ap.parse_args()

    if not HAVE_LT:
        sys.stdout.write("litetorch Python module not importable: " + LT_ERR + "\n")
        sys.stdout.write("Build it on this machine, then retry:\n")
        sys.stdout.write("  cd <integration> && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n")
        sys.stdout.write("  cmake --build build --target litetorch_py\n")
        sys.stdout.write("  PYTHONPATH=<integration>/build:$PYTHONPATH python3 bench_kernels.py --device " + args.device + "\n")
        sys.stdout.write("torch-side benchmarks still run below.\n")
    if not HAVE_TORCH:
        sys.stdout.write("torch not importable: " + TORCH_ERR + "\n")

    results = []
    results.append(bench_matmul(args, args.device))
    for dt in ("fp32", "uint8", "int64"):
        results.append(bench_ew_add(args, args.device, dt))
    results.append(bench_conv2d(args, args.device))
    results.append(bench_softmax(args, args.device))
    results.append(bench_reduce(args, args.device))
    results.append(bench_allreduce(args, args.device))
    results.append(bench_onnx_export(args, args.device))

    payload = {"device": args.device, "warmup": args.warmup, "iters": args.iters, "benchmarks": results}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    for r in results:
        line = r["name"] + ": "
        for side in ("torch", "litetorch"):
            s = r.get(side, {})
            if s.get("skipped"):
                line += side + "=SKIP(" + s["reason"][:60] + ") "
            else:
                line += side + " median=%.3fms mean=%.3fms p95=%.3fms " % (s["median_ms"], s["mean_ms"], s["p95_ms"])
        t, l = r.get("torch", {}), r.get("litetorch", {})
        if not t.get("skipped") and not l.get("skipped"):
            line += "lt/torch ratio=%.3f" % (l["median_ms"] / t["median_ms"])
        sys.stdout.write(line + "\n")
    sys.stdout.write("wrote " + args.out + "\n")
    if args.record_baseline:
        base = "baseline_" + args.device + ".json"
        with open(base, "w") as f:
            json.dump(payload, f, indent=2)
        sys.stdout.write("recorded baseline " + base + "\n")


if __name__ == "__main__":
    main()
