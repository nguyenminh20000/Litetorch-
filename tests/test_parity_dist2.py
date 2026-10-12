import os
import sys
import time

import multiprocessing as mp

import litetorch as lt

PORT_BASE = 29611

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def worker(rank, world_size, port, out):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    pg = lt.distributed.ProcessGroup.get()
    r = {}
    r["rank_ok"] = pg.get_rank() == rank
    r["ws_ok"] = pg.get_world_size() == world_size
    r["init_ok"] = pg.is_initialized()

    nb = lt.distributed.NCCLBridge.get()
    r["nccl_no_crash"] = True
    r["nccl_avail"] = nb.is_available()

    lt.distributed.barrier()
    r["barrier_ok"] = True

    t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [4], cpu)
    lt.distributed.all_reduce(t)
    r["ar_sum"] = t.to_vector() == [2.0, 4.0, 6.0, 8.0]

    t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [4], cpu)
    lt.distributed.all_reduce(t, lt.distributed.ReduceOp.AVG)
    r["ar_avg"] = t.to_vector() == [1.0, 2.0, 3.0, 4.0]

    t = lt.Tensor.from_vector([float(rank + 1), float(rank + 2)], [2], cpu)
    lt.distributed.all_reduce(t, lt.distributed.ReduceOp.PRODUCT)
    r["ar_prod"] = t.to_vector() == [2.0, 6.0]

    t = lt.Tensor.from_vector([float(rank + 1), float(10 - rank)], [2], cpu)
    lt.distributed.all_reduce(t, lt.distributed.ReduceOp.MIN)
    r["ar_min"] = t.to_vector() == [1.0, 9.0]

    t = lt.Tensor.from_vector([float(rank + 1), float(10 - rank)], [2], cpu)
    lt.distributed.all_reduce(t, lt.distributed.ReduceOp.MAX)
    r["ar_max"] = t.to_vector() == [2.0, 10.0]

    t = lt.Tensor.from_vector([float(rank)], [1], cpu)
    lt.distributed.broadcast(t, 0)
    r["bcast"] = t.to_vector() == [0.0]

    shard = lt.Tensor.from_vector([float(rank + 1), float(rank + 2)], [2], cpu)
    full = lt.Tensor.from_vector([0.0] * 4, [4], cpu)
    lt.distributed.all_gather(shard, full)
    r["ag"] = full.to_vector() == [1.0, 2.0, 2.0, 3.0]

    full = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [4], cpu)
    shard = lt.Tensor.from_vector([0.0] * 2, [2], cpu)
    lt.distributed.reduce_scatter(shard, full)
    r["rs_sum"] = shard.to_vector() == ([2.0, 4.0] if rank == 0 else [6.0, 8.0])

    full = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [4], cpu)
    shard = lt.Tensor.from_vector([0.0] * 2, [2], cpu)
    lt.distributed.reduce_scatter(shard, full, lt.distributed.ReduceOp.AVG)
    r["rs_avg"] = shard.to_vector() == ([1.0, 2.0] if rank == 0 else [3.0, 4.0])

    full = lt.Tensor.from_vector([float(rank + 1)] * 4, [4], cpu)
    shard = lt.Tensor.from_vector([0.0] * 2, [2], cpu)
    lt.distributed.reduce_scatter(shard, full, lt.distributed.ReduceOp.MAX)
    r["rs_max"] = shard.to_vector() == [2.0, 2.0]

    base = rank * 10
    inp = lt.Tensor.from_vector([float(base + 1), float(base + 2), float(base + 3), float(base + 4)], [4], cpu)
    a2a_out = lt.Tensor.from_vector([0.0] * 4, [4], cpu)
    lt.distributed.all_to_all(inp, a2a_out)
    r["a2a"] = a2a_out.to_vector() == ([1.0, 2.0, 11.0, 12.0] if rank == 0 else [3.0, 4.0, 13.0, 14.0])

    sout = lt.Tensor.from_vector([0.0] * 2, [2], cpu)
    sinp = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0] if rank == 0 else [0.0] * 4, [4], cpu)
    lt.distributed.scatter(sout, sinp, 0)
    r["scatter"] = sout.to_vector() == ([1.0, 2.0] if rank == 0 else [3.0, 4.0])

    ginp = lt.Tensor.from_vector([float(rank + 1), float(rank + 2)], [2], cpu)
    gout = lt.Tensor.from_vector([0.0] * 4, [4], cpu)
    lt.distributed.gather(ginp, gout, 0)
    r["gather"] = gout.to_vector() == [1.0, 2.0, 2.0, 3.0] if rank == 0 else True

    t = lt.Tensor.from_vector([3.0, 1.0], [2], cpu)
    pg.all_reduce(t, lt.distributed.ReduceOp.AVG)
    r["pg_ar_avg"] = t.to_vector() == [3.0, 1.0]
    pg.barrier()
    r["pg_barrier_ok"] = True

    model = lt.nn.Linear(1, 1, False)
    model.weight = lt.Tensor.from_vector([1.0], [1, 1], cpu, True)
    ddp = lt.distributed.DistributedDataParallel(model)
    x = lt.Tensor.from_vector([float(rank + 1)], [1, 1], cpu)
    out_t = ddp.forward(x)
    loss = lt.Ops.sum(lt.Ops.mul(out_t, out_t))
    ddp.backward(loss)
    g = model.weight.grad.to_vector()[0]
    r["ddp_grad"] = abs(g - 5.0) < 1e-4

    lt.distributed.shutdown()
    out[rank] = r
    print(f"rank {rank} done", flush=True)


def main():
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Manager() as mgr:
        out = mgr.dict()
        procs = [ctx.Process(target=worker, args=(rank, 2, PORT_BASE, out)) for rank in (0, 1)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
        failed_proc = any(p.exitcode != 0 for p in procs)
        res = dict(out)

    names = ["rank_ok", "ws_ok", "init_ok", "nccl_no_crash", "barrier_ok",
             "ar_sum", "ar_avg", "ar_prod", "ar_min", "ar_max",
             "bcast", "ag", "rs_sum", "rs_avg", "rs_max",
             "a2a", "scatter", "gather", "pg_ar_avg", "pg_barrier_ok",
             "ddp_grad"]
    for rank in (0, 1):
        rr = res[rank]
        for n in names:
            check(f"rank{rank} {n}", rr.get(n, False))
    check("nccl unavailable on CPU-only (expected)", res[0]["nccl_avail"] == False)

    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    if failed_proc:
        print("WARNING: a worker process exited non-zero", flush=True)
        failed += 1
    print(f"== SUMMARY: {passed} PASS, {failed} FAIL in {time.perf_counter()-t0:.1f}s ==", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
