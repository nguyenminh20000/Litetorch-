import multiprocessing as mp

import litetorch as lt

PORT = 29683
results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def allclose(a, b, tol=1e-5):
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tol + tol * abs(y) for x, y in zip(a, b))


def worker(rank, world_size, port, out):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    r = {}
    try:
        p1 = lt.Tensor.from_vector([float(i + 1) for i in range(6)], [2, 3], cpu)
        p2 = lt.Tensor.from_vector([0.5 * (i + 1) for i in range(4)], [4], cpu)
        p3 = lt.Tensor.from_vector([9.0], [1], cpu)

        fp = lt.distributed.FlatParameter([p1, p2, p3])
        r["flat_meta_count"] = len(fp.metadata()) == 3
        m = fp.metadata()
        r["flat_meta_offsets"] = (m[0]["offset"], m[0]["numel"], m[1]["offset"], m[1]["numel"], m[2]["offset"], m[2]["numel"]) == (0, 6, 6, 4, 10, 1)
        r["flat_meta_shapes"] = m[0]["shape"] == [2, 3] and m[1]["shape"] == [4] and m[2]["shape"] == [1]
        r["flat_numel"] = fp.flat_numel() == 12 and fp.shard_numel() == 6 and fp.unpadded_numel() == 11

        shard = fp.local_shard()
        full_cat = [float(i + 1) for i in range(6)] + [0.5 * (i + 1) for i in range(4)] + [9.0]
        full_padded = full_cat + [0.0]
        expect_shard = full_padded[rank * 6:(rank + 1) * 6]
        r["flat_shard_init"] = allclose(shard.to_vector(), expect_shard)

        fp.all_gather_flat()
        r["flat_all_gather"] = allclose(fp.full_buffer().to_vector(), full_padded)

        views = fp.param_views()
        r["flat_views"] = (allclose(views[0].to_vector(), full_cat[0:6]) and
                           allclose(views[1].to_vector(), full_cat[6:10]) and
                           views[2].to_vector() == [9.0] and
                           list(views[0].shape) == [2, 3])

        grad_full = lt.Tensor.from_vector([1.0] * 12, [12], cpu)
        grad_shard = fp.reduce_scatter_flat(grad_full)
        r["flat_reduce_scatter"] = allclose(grad_shard.to_vector(), [2.0] * 6)

        q1 = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu)
        q2 = lt.Tensor.from_vector([4.0, 5.0, 6.0, 7.0], [4], cpu)
        fq = lt.distributed.FlatParameter([q1, q2])
        r["flat_pad_numel"] = fq.flat_numel() == 8 and fq.shard_numel() == 4 and fq.unpadded_numel() == 7
        fq.all_gather_flat()
        r["flat_pad_gather"] = allclose(fq.full_buffer().to_vector(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0])
        qv = fq.param_views()
        r["flat_pad_views"] = allclose(qv[0].to_vector(), [1.0, 2.0, 3.0]) and allclose(qv[1].to_vector(), [4.0, 5.0, 6.0, 7.0])

        try:
            lt.distributed.FlatParameter([])
            r["flat_err_empty"] = False
        except Exception:
            r["flat_err_empty"] = True
        try:
            bad = lt.Tensor.from_vector([1.0], [1], cpu, False, lt.DataType.FP16)
            lt.distributed.FlatParameter([p1, bad])
            r["flat_err_dtype"] = False
        except Exception:
            r["flat_err_dtype"] = True
    except Exception as e:
        r["exception"] = f"{type(e).__name__}: {e}"
    out.put(r)


def worker_single(out):
    cpu = lt.Device("cpu")
    r = {}
    try:
        p1 = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu)
        fp = lt.distributed.FlatParameter([p1])
        r["flat_single_world"] = fp.world_size() == 1
        fp.all_gather_flat()
        r["flat_single_gather"] = fp.full_buffer().to_vector() == [1.0, 2.0, 3.0]
        g = fp.reduce_scatter_flat(fp.full_buffer())
        r["flat_single_rs"] = g.to_vector() == [1.0, 2.0, 3.0]
    except Exception as e:
        r["exception"] = f"{type(e).__name__}: {e}"
    out.put(r)


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(rk, 2, PORT, q)) for rk in range(2)]
    for p in procs:
        p.start()
    rank_results = [q.get() for _ in procs]
    for p in procs:
        p.join()

    names = sorted({k for rr in rank_results for k in rr})
    for n in names:
        vals = [rr.get(n) for rr in rank_results]
        if n == "exception":
            check(n, all(v is None for v in vals))
            for v in vals:
                if v:
                    print("  ", v, flush=True)
        else:
            check(n, all(v is True for v in vals))

    q2 = ctx.Queue()
    p = ctx.Process(target=worker_single, args=(q2,))
    p.start()
    sr = q2.get()
    p.join()
    for n, v in sorted(sr.items()):
        if n == "exception":
            check(n, False)
            print("  ", v, flush=True)
        else:
            check(n, v is True)

    fails = [n for n, c in results if not c]
    print(f"\n{len(results) - len(fails)}/{len(results)} passed", flush=True)
    raise SystemExit(1 if fails else 0)
