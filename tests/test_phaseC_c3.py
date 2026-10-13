import multiprocessing as mp
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import litetorch as lt

PORT = 29641

HAS_INTERLEAVED = hasattr(lt.distributed, "InterleavedPipelineModule")


def check(name, cond, out):
    out.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def allclose(a, b, tol=1e-4):
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tol + tol * abs(y) for x, y in zip(a, b))


W = [
    [float(i + 1) for i in range(16)],
    [0.1 * float(i + 1) for i in range(16)],
    [0.01 * float(i + 1) for i in range(16)],
    [0.02 * float(i + 1) for i in range(16)],
]
MB0 = [float(i) * 0.5 for i in range(8)]
MB1 = [1.0 + float(i) * 0.25 for i in range(8)]
MB2 = [2.0 - float(i) * 0.125 for i in range(8)]


def make_chunk(vs, cpu):
    lin = lt.nn.Linear(4, 4, bias=False)
    lin.parameters()[0].copy_(lt.Tensor.from_vector(W[vs], [4, 4], cpu))
    return lin


def ref_forward(mb_data, cpu):
    x = lt.Tensor.from_vector(mb_data, [2, 4], cpu)
    for vs in range(4):
        x = make_chunk(vs, cpu).forward(x)
    return x.to_vector()


def ref_grads(mb_datas, cpu):
    chunks = [make_chunk(vs, cpu) for vs in range(4)]
    for mb_data in mb_datas:
        x = lt.Tensor.from_vector(mb_data, [2, 4], cpu, True)
        for c in chunks:
            x = c.forward(x)
        loss = lt.Ops.sum(x)
        scale = lt.Tensor.from_vector([1.0 / len(mb_datas)], [1], cpu, False)
        loss.backward(scale)
    return [c.parameters()[0].grad.to_vector() for c in chunks]


def worker(rank, world_size, port, out):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    r = []

    p, v = 2, 2
    my_vs = [rank + j * p for j in range(v)]
    chunks = [make_chunk(vs, cpu) for vs in my_vs]
    pipe = lt.distributed.InterleavedPipelineModule(chunks)
    check("num_chunks", pipe.num_chunks() == v, r)
    check("param_count", len(pipe.parameters()) == v, r)

    mbs = [MB0, MB1, MB2]
    mts = [lt.Tensor.from_vector(d, [2, 4], cpu, True) for d in mbs]
    outs = pipe.schedule_interleaved_1f1b(mts)
    check("fwd_count", len(outs) == 3, r)
    if rank == 1:
        for i, d in enumerate(mbs):
            check(f"fwd_mb{i}", allclose(outs[i].to_vector(), ref_forward(d, cpu)), r)

    lt.distributed.barrier()

    chunks2 = [make_chunk(vs, cpu) for vs in my_vs]
    pipe2 = lt.distributed.InterleavedPipelineModule(chunks2)

    def loss_fn(out, idx):
        return lt.Ops.sum(out)

    nts = [lt.Tensor.from_vector(d, [2, 4], cpu, True) for d in mbs]
    losses = pipe2.train_step_interleaved_1f1b(nts, loss_fn)
    if rank == 1:
        check("bwd_loss_count", len(losses) == 3, r)
    egs = ref_grads(mbs, cpu)
    for j, vs in enumerate(my_vs):
        g = chunks2[j].parameters()[0].grad
        check(f"bwd_has_grad_vs{vs}", g is not None, r)
        if g is not None:
            check(f"bwd_grad_vs{vs}", allclose(g.to_vector(), egs[vs]), r)

    lt.distributed.barrier()

    try:
        lt.distributed.InterleavedPipelineModule([])
        check("empty_chunks_throws", False, r)
    except Exception:
        check("empty_chunks_throws", True, r)

    try:
        lt.distributed.InterleavedPipelineModule([None])
        check("null_chunk_throws", False, r)
    except Exception:
        check("null_chunk_throws", True, r)

    c3 = [make_chunk(vs, cpu) for vs in my_vs]
    pipe3 = lt.distributed.InterleavedPipelineModule(c3)
    check("empty_mbs", pipe3.schedule_interleaved_1f1b([]) == [], r)
    check("empty_mbs_train", pipe3.train_step_interleaved_1f1b([]) == [], r)

    try:
        pipe3.schedule_interleaved_1f1b([None])
        check("null_mb_throws", False, r)
    except Exception:
        check("null_mb_throws", True, r)

    lt.distributed.barrier()
    lt.distributed.shutdown()
    out.extend(r)


def main():
    if not HAS_INTERLEAVED:
        print("SKIP: InterleavedPipelineModule not bound")
        sys.exit(0)
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        out = manager.list()
        procs = []
        for rank in range(2):
            p = ctx.Process(target=worker, args=(rank, 2, PORT, out))
            p.start()
            procs.append(p)
        for p in procs:
            p.join(180)
            if p.is_alive():
                p.terminate()
        results = list(out)
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
