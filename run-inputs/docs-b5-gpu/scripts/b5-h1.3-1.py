import os
import sys
TREE = os.environ.get("LITETORCH_TREE") or (sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/litetorch")
sys.path.insert(0, os.path.join(TREE, "python"))
def _hook(t, v, tb):
    print("B5 H1.3-1 FAIL: %s: %s" % (t.__name__, v))
    sys.exit(1)
sys.excepthook = _hook
import multiprocessing as mp

import litetorch as lt

PORT = 29713


def worker(rank, world_size, port):
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        gpu = lt.Device("cuda:%d" % rank)
        model = lt.nn.Sequential([
            lt.nn.Linear(1024, 1024, True),
            lt.nn.GELU(),
            lt.nn.Linear(1024, 1024, True),
        ])
        model.to(gpu)
        params = model.parameters()
        assert params[0].device.type == lt.DeviceType.GPU, "model is not on GPU"
        x = lt.Tensor.from_vector([0.01] * 2048, [2, 1024], gpu)
        y = model(x)
        loss = y.sum()
        loss.backward()
        print("Rank %d GPU step done, loss: %s" % (rank, loss.to_vector()))
    finally:
        lt.distributed.shutdown()


def main():
    assert lt.cuda.device_count() == 2, "expected 2 GPUs"
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(2):
        p = ctx.Process(target=worker, args=(rank, 2, PORT))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(300)
        if p.is_alive():
            p.terminate()
    print("exit codes:", [p.exitcode for p in procs])
    assert [p.exitcode for p in procs] == [0, 0], "a rank failed"


if __name__ == "__main__":
    main()
print("B5 H1.3-1 PASS")
