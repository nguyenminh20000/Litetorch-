import multiprocessing as mp

import litetorch as lt

PORT = 29684
results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def allclose(a, b, tol=1e-5):
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tol + tol * abs(y) for x, y in zip(a, b))


def make_linear(in_f, out_f, seed):
    import random
    random.seed(seed)
    w = [random.gauss(0.0, 0.5) for _ in range(in_f * out_f)]
    b = [random.gauss(0.0, 0.5) for _ in range(out_f)]
    lin = lt.nn.Linear(in_f, out_f, True)
    lin.weight = lt.Tensor.from_vector(w, [out_f, in_f], lt.Device("cpu"))
    lin.weight.requires_grad = True
    lin.bias = lt.Tensor.from_vector(b, [out_f], lt.Device("cpu"))
    lin.bias.requires_grad = True
    return lin


def worker(rank, world_size, port, out):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    r = {}
    try:
        lin = make_linear(4, 8, 100 + rank * 0)
        w_before = lin.weight.to_vector()[:]
        b_before = lin.bias.to_vector()[:]

        lt.distributed.fsdp_fully_shard(lin)
        r["shard_api_exists"] = True

        units = lt.distributed.FSDPUnit.__doc__ is not None
        r["unit_bound"] = True

        x = lt.Tensor.from_vector([0.1 * (i + 1) for i in range(8)], [2, 4], cpu)
        y = lin(x)
        r["fwd_shape"] = list(y.shape) == [2, 8]
        r["fwd_finite"] = all(v == v and abs(v) < 1e10 for v in y.to_vector())

        ref = make_linear(4, 8, 100)
        ref_w = lt.Tensor.from_vector(w_before, [8, 4], cpu)
        ref_w.requires_grad = True
        ref_b = lt.Tensor.from_vector(b_before, [8], cpu)
        ref_b.requires_grad = True
        ref.weight = ref_w
        ref.bias = ref_b
        y_ref = ref(x)
        r["fwd_matches_replicated"] = allclose(y.to_vector(), y_ref.to_vector(), 1e-4)

        loss = y.sum()
        loss.backward()
        r["bwd_runs"] = True
        r["weight_grad_exists"] = lin.weight.grad is not None
        r["bias_grad_exists"] = lin.bias.grad is not None

        ref_loss = y_ref.sum()
        ref_loss.backward()
        r["grad_w_matches"] = allclose(lin.weight.grad.to_vector(), ref_w.grad.to_vector(), 1e-4)
        r["grad_b_matches"] = allclose(lin.bias.grad.to_vector(), ref_b.grad.to_vector(), 1e-4)

        lt.distributed.fsdp_set_prefetch_limit(1)
        r["prefetch_limit_set"] = True
        lt.distributed.fsdp_set_prefetch_limit(2)

        lin2 = make_linear(8, 4, 200)
        seq_params_before = [p.to_vector()[:] for p in [lin2.weight, lin2.bias]]
        lt.distributed.fsdp_fully_shard(lin2)
        x2 = lt.Tensor.from_vector([0.05 * (i + 1) for i in range(16)], [2, 8], cpu)
        y2 = lin2(x2)
        r["second_unit_fwd"] = list(y2.shape) == [2, 4]
        loss2 = y2.sum()
        loss2.backward()
        r["second_unit_bwd"] = lin2.weight.grad is not None

        out["rank%d" % rank] = r
    except Exception as e:
        import traceback
        out["rank%d" % rank] = {"error": "%s: %s" % (type(e).__name__, e),
                                "tb": traceback.format_exc()[-2000:]}
    finally:
        try:
            lt.distributed.destroy_process_group()
        except Exception:
            pass


def main():
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    out = manager.dict()
    procs = []
    for rank in range(2):
        p = ctx.Process(target=worker, args=(rank, 2, PORT, out))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(180)
        if p.is_alive():
            p.terminate()
    ok = True
    for rank in range(2):
        key = "rank%d" % rank
        if key not in out:
            print(f"[FAIL] rank {rank}: no result", flush=True)
            ok = False
            continue
        r = dict(out[key])
        if "error" in r:
            print(f"[FAIL] rank {rank}: {r['error']}", flush=True)
            print(r.get("tb", "")[:1500], flush=True)
            ok = False
            continue
        for k, v in sorted(r.items()):
            status = "PASS" if v else "FAIL"
            print(f"[{status}] rank{rank}/{k}", flush=True)
            if not v:
                ok = False
    print("ALL PASS" if ok else "SOME FAILED", flush=True)


if __name__ == "__main__":
    main()
