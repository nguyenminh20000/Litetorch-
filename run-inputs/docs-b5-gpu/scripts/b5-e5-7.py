import os
import sys
TREE = os.environ.get("LITETORCH_TREE") or (sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/litetorch")
sys.path.insert(0, os.path.join(TREE, "python"))
def _hook(t, v, tb):
    print("B5 E5-7 FAIL: %s: %s" % (t.__name__, v))
    sys.exit(1)
sys.excepthook = _hook
import sys
import multiprocessing as mp
import litetorch as lt


def worker(rank, world_size, port, q):
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        device = lt.Device("cuda:" + str(rank))
        lt.cuda.set_device(rank)
        model = lt.nn.Sequential([lt.nn.Linear(4096, 4096, True)])
        model.to(device)
        lt.distributed.fsdp_fully_shard(model)
        x = lt.Tensor.from_vector([0.5] * 4096, [1, 4096], device)
        y = model.forward(x)
        loss = y.sum()
        loss.backward()
        q.put((rank, y.to_vector()[:4]))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, 29645, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(300)
        assert p.exitcode == 0, "a rank failed"
    results = [q.get(timeout=30) for _ in range(2)]
    print("ranks done:", sorted(r for r, _ in results))


if __name__ == "__main__":
    main()
print("B5 E5-7 PASS")
