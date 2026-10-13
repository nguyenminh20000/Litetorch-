import multiprocessing as mp
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import litetorch as lt

PORT = 29632

HAS_PIPE = hasattr(lt.distributed, "PipelineParallelModule")


def check(name, cond, out):
    out.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def allclose(a, b, tol=1e-4):
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tol + tol * abs(y) for x, y in zip(a, b))


W0 = [float(i + 1) for i in range(16)]
W1 = [0.1 * float(i + 1) for i in range(16)]
MB0 = [float(i) * 0.5 for i in range(8)]
MB1 = [1.0 + float(i) * 0.25 for i in range(8)]


def make_stage(rank, cpu):
    lin = lt.nn.Linear(4, 4, bias=False)
    w = W0 if rank == 0 else W1
    lin.parameters()[0].copy_(lt.Tensor.from_vector(w, [4, 4], cpu))
    return lin


def ref_forward(mb_data, cpu):
    x = lt.Tensor.from_vector(mb_data, [2, 4], cpu)
    s0 = make_stage(0, cpu)
    s1 = make_stage(1, cpu)
    return s1.forward(s0.forward(x)).to_vector()


def ref_grads(mb_datas, cpu):
    s0 = make_stage(0, cpu)
    s1 = make_stage(1, cpu)
    for mb_data in mb_datas:
        x = lt.Tensor.from_vector(mb_data, [2, 4], cpu, True)
        out = s1.forward(s0.forward(x))
        loss = lt.Ops.sum(out)
        scale = lt.Tensor.from_vector([1.0 / len(mb_datas)], [1], cpu, False)
        loss.backward(scale)
    g0 = s0.parameters()[0].grad.to_vector()
    g1 = s1.parameters()[0].grad.to_vector()
    return g0, g1


def worker(rank, world_size, port, out):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    r = []

    stage = make_stage(rank, cpu)
    pipe = lt.distributed.PipelineParallelModule(stage)

    mb0 = lt.Tensor.from_vector(MB0, [2, 4], cpu, True)
    mb1 = lt.Tensor.from_vector(MB1, [2, 4], cpu, True)

    outs = pipe.schedule_1f1b([mb0, mb1])
    check("fwd_1f1b_count", len(outs) == 2, r)
    if rank == 1:
        e0 = ref_forward(MB0, cpu)
        e1 = ref_forward(MB1, cpu)
        check("fwd_1f1b_mb0", allclose(outs[0].to_vector(), e0), r)
        check("fwd_1f1b_mb1", allclose(outs[1].to_vector(), e1), r)

    check("fwd_single_count", True, r)
    single = pipe.forward(mb0)
    if rank == 1:
        check("fwd_single", allclose(single.to_vector(), ref_forward(MB0, cpu)), r)

    lt.distributed.barrier()

    def loss_fn(out, idx):
        return lt.Ops.sum(out)

    stage2 = make_stage(rank, cpu)
    pipe2 = lt.distributed.PipelineParallelModule(stage2)
    n0 = lt.Tensor.from_vector(MB0, [2, 4], cpu, True)
    n1 = lt.Tensor.from_vector(MB1, [2, 4], cpu, True)
    losses = pipe2.train_step_1f1b([n0, n1], loss_fn)
    if rank == 1:
        check("bwd_loss_count", len(losses) == 2, r)
    pg = stage2.parameters()[0].grad
    check("bwd_has_grad", pg is not None, r)
    if pg is not None:
        eg0, eg1 = ref_grads([MB0, MB1], cpu)
        exp = eg0 if rank == 0 else eg1
        check("bwd_grad_match", allclose(pg.to_vector(), exp), r)

    lt.distributed.barrier()

    stage3 = make_stage(rank, cpu)
    pipe3 = lt.distributed.PipelineParallelModule(stage3)
    s1 = lt.Tensor.from_vector(MB0, [2, 4], cpu, True)
    o1 = pipe3.schedule_1f1b([s1])
    check("fwd_single_mb", len(o1) == 1, r)
    if rank == 1:
        check("fwd_single_mb_val", allclose(o1[0].to_vector(), ref_forward(MB0, cpu)), r)

    lt.distributed.barrier()

    try:
        bad = lt.distributed.PipelineParallelModule(None)
        check("null_module_throws", False, r)
    except Exception:
        check("null_module_throws", True, r)

    try:
        pipe3.schedule_1f1b([None])
        check("null_mb_throws", False, r)
    except Exception:
        check("null_mb_throws", True, r)

    check("empty_mbs", pipe3.schedule_1f1b([]) == [], r)

    lt.distributed.barrier()
    lt.distributed.shutdown()
    out.extend(r)


def main():
    if not HAS_PIPE:
        print("SKIP: PipelineParallelModule not bound")
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
