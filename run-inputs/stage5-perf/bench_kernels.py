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


def lt_ops_attr(name):
    if not HAVE_LT:
        return None
    ops = getattr(LT, "Ops", None)
    if ops is None:
        return None
    f = getattr(ops, name, None)
    return f if callable(f) else None


def lt_device(device):
    return LT.Device(device)


def lt_randn(shape, device):
    return LT.Tensor.randn(shape, lt_device(device))


def lt_zeros(shape, device, dtype):
    return LT.Tensor.zeros(shape, lt_device(device), False, dtype)


def lt_dtype(name):
    return getattr(LT.DataType, name)


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
    mm = lt_ops_attr("matmul")
    if mm is None:
        out["litetorch"] = skip("LT.Ops.matmul not found in module")
    else:
        try:
            la = lt_randn([1024, 1024], device)
            lb = lt_randn([1024, 1024], device)
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
    add = lt_ops_attr("add")
    if add is None:
        out["litetorch"] = skip("LT.Ops.add not found in module")
    elif dtype != "fp32" and device != "cpu":
        out["litetorch"] = skip("no GPU integer add kernel: Ops.add requires CPU tensors for integer dtypes and otherwise falls back through FP32 cast to CPU, so a CUDA timing would measure CPU work under a CUDA label")
    else:
        try:
            if dtype == "fp32":
                la = lt_randn([n], device)
                lb = lt_randn([n], device)
            else:
                la = lt_zeros([n], device, lt_dtype(dtype.upper()))
                lb = lt_zeros([n], device, lt_dtype(dtype.upper()))
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
    conv = lt_ops_attr("conv2d")
    if conv is None:
        out["litetorch"] = skip("LT.Ops.conv2d not found in module")
    else:
        try:
            lx = lt_randn([3, 64, 32, 32], device)
            lw = lt_randn([64, 64, 3, 3], device)
            out["litetorch"] = time_fn(lambda: conv(lx, lw, None, 1, 1), device, args.warmup, args.iters)
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
    sm = lt_ops_attr("softmax")
    if sm is None:
        out["litetorch"] = skip("LT.Ops.softmax not found in module")
    else:
        try:
            lx = lt_randn([1000000], device)
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
    sm = lt_ops_attr("sum")
    if sm is None:
        out["litetorch"] = skip("LT.Ops.sum not found in module")
    else:
        try:
            lx = lt_randn([n], device)
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
        q.put({"median_ms": ts[len(ts) // 2], "mean_ms": sum(ts) / len(ts), "p95_ms": ts[int(len(ts) * 0.95)], "iters": len(ts)})
    dist.destroy_process_group()


def _lt_allreduce_worker(rank, world, port, device, iters, warmup, q):
    import litetorch as LT
    pg = None
    try:
        pg = LT.distributed.ProcessGroup.get()
        pg.init(rank, world, "127.0.0.1", port)
        t = LT.Tensor.randn([1024 * 1024], LT.Device(device))
        for _ in range(warmup):
            pg.all_reduce(t)
        if device == "cuda":
            LT.cuda_synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            pg.all_reduce(t)
            if device == "cuda":
                LT.cuda_synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts.sort()
        if rank == 0:
            q.put({"median_ms": ts[len(ts) // 2], "mean_ms": sum(ts) / len(ts), "p95_ms": ts[int(len(ts) * 0.95)], "iters": len(ts)})
    finally:
        if pg is not None:
            try:
                pg.shutdown()
            except Exception:
                pass


def bench_allreduce(args, device):
    out = {"name": "allreduce_2rank_4MB"}
    if not HAVE_TORCH:
        out["torch"] = skip("torch not importable")
    else:
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
    if not HAVE_LT:
        out["litetorch"] = skip("litetorch not importable: " + LT_ERR)
        return out
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_lt_allreduce_worker, args=(0, 2, 29578, device, 20, 5, q))
        c = ctx.Process(target=_lt_allreduce_worker, args=(1, 2, 29578, device, 20, 5, q))
        p.start()
        c.start()
        p.join(300)
        c.join(300)
        if p.exitcode == 0 and c.exitcode == 0 and not q.empty():
            out["litetorch"] = q.get()
        else:
            out["litetorch"] = skip("litetorch 2-rank all_reduce run failed")
    except Exception as e:
        out["litetorch"] = skip("litetorch 2-rank all_reduce run failed: " + str(e))
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
    if not HAVE_LT:
        out["litetorch"] = skip("litetorch not importable: " + LT_ERR)
        return out
    try:
        nn_mod = getattr(LT, "nn", None)
        onnx_mod = getattr(LT, "onnx", None)
        if nn_mod is None or onnx_mod is None:
            out["litetorch"] = skip("LT.nn or LT.onnx submodule not found in module")
            return out
        model = nn_mod.Sequential([nn_mod.Linear(256, 256), nn_mod.ReLU(), nn_mod.Linear(256, 256), nn_mod.ReLU(), nn_mod.Linear(256, 10)])
        x = LT.Tensor.randn([1, 256], LT.Device("cpu"))
        def do_export_lt():
            onnx_mod.export(model, x, "/tmp/mlp_bench_lt.onnx")
        out["litetorch"] = time_fn(do_export_lt, "cpu", 2, 5)
    except Exception as e:
        out["litetorch"] = skip("litetorch onnx export failed: " + str(e))
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
                med = s.get("median_ms")
                mean = s.get("mean_ms")
                p95 = s.get("p95_ms")
                p95_note = ""
                if p95 is None:
                    p95 = med
                    p95_note = "p95=median-fallback "
                line += side + " median=%.3fms mean=%.3fms p95=%.3fms %s" % (med, mean, p95, p95_note)
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
