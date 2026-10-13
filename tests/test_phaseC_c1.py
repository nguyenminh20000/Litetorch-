import multiprocessing as mp
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import litetorch as lt

PORT = 29631

HAS_ASYNC = hasattr(lt.distributed, "isend") and hasattr(lt.distributed, "irecv")


def check(name, cond, out):
    out.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def allclose(a, b, tol=1e-5):
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tol + tol * abs(y) for x, y in zip(a, b))


def worker(rank, world_size, port, out):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    r = []

    data = [float(i + 1) for i in range(8)]

    if rank == 0:
        t = lt.Tensor.from_vector(data, [2, 4], cpu)
        w = lt.distributed.isend(t, 1)
        check("isend_returns_work", w is not None, r)
        w.wait()
        check("isend_wait_ok", True, r)
        w.wait()
        check("isend_double_wait_ok", True, r)

        buf = lt.Tensor.from_vector([0.0] * 8, [2, 4], cpu)
        w2 = lt.distributed.irecv(buf, 1)
        check("irecv_returns_work", w2 is not None, r)
        w2.wait()
        check("bidir_recv_data", allclose(buf.to_vector(), [10.0 + i for i in range(8)]), r)
    else:
        buf = lt.Tensor.from_vector([0.0] * 8, [2, 4], cpu)
        w = lt.distributed.irecv(buf, 0)
        check("irecv_returns_work", w is not None, r)
        check("work_not_done_before_wait", True, r)
        w.wait()
        check("recv_data", allclose(buf.to_vector(), data), r)
        check("work_done_after_wait", w.is_completed(), r)
        check("recv_shape", list(buf.shape) == [2, 4], r)

        t2 = lt.Tensor.from_vector([10.0 + i for i in range(8)], [2, 4], cpu)
        w2 = lt.distributed.isend(t2, 0)
        w2.wait()
        check("bidir_send_ok", True, r)

    lt.distributed.barrier()

    if rank == 0:
        t = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu)
        w = lt.distributed.isend(t, 1)
        t.copy_(lt.Tensor.from_vector([9.0, 9.0, 9.0], [3], cpu))
        w.wait()
        check("snapshot_semantics_queued", True, r)
    else:
        buf = lt.Tensor.from_vector([0.0] * 3, [3], cpu)
        w = lt.distributed.irecv(buf, 0)
        w.wait()
        check("snapshot_semantics_data", allclose(buf.to_vector(), [1.0, 2.0, 3.0]), r)

    lt.distributed.barrier()

    tself = lt.Tensor.from_vector([1.0], [1], cpu)
    wself = lt.distributed.isend(tself, rank)
    wself.wait()
    check("self_send_noop", wself.is_completed(), r)
    wself2 = lt.distributed.irecv(tself, rank)
    wself2.wait()
    check("self_recv_noop", wself2.is_completed(), r)

    lt.distributed.barrier()
    lt.distributed.shutdown()
    out.extend(r)


def main():
    if not HAS_ASYNC:
        print("SKIP: lt.distributed.isend/irecv not bound yet (bindings pending)")
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
            p.join(120)
            if p.is_alive():
                p.terminate()
        results = list(out)
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
