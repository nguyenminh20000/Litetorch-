import os
import sys

TREE = os.environ.get("LITETORCH_TREE") or (sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/litetorch")
sys.path.insert(0, os.path.join(TREE, "python"))

import faulthandler
import multiprocessing as mp
import litetorch as lt


def _libnccl_lines():
    try:
        with open("/proc/self/maps") as f:
            return [ln.strip() for ln in f if "nccl" in ln.lower()]
    except Exception as e:
        return ["maps-unreadable: %r" % (e,)]


def _phase(rank, msg):
    print("PHASE rank=%d %s" % (rank, msg), flush=True)


def _hook(t, v, tb):
    print("B5 E5-7 FAIL: %s: %s" % (t.__name__, v), flush=True)
    sys.exit(1)


sys.excepthook = _hook


def worker(rank, world_size, port, q, logdir):
    faulthandler.enable()
    logpath = os.path.join(logdir, "b5-e5-7-rank%d.log" % rank)
    logf = open(logpath, "w", buffering=1)
    sys.stdout = logf
    sys.stderr = logf
    _phase(rank, "start pid=%d" % os.getpid())
    _phase(rank, "LITETORCH_NCCL_LIB=%r" % os.environ.get("LITETORCH_NCCL_LIB"))
    _phase(rank, "CUDA_VISIBLE_DEVICES=%r" % os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        import torch
        _phase(rank, "torch.cuda.device_count=%d current_device=%d" % (
            torch.cuda.device_count(), torch.cuda.current_device()))
    except Exception as e:
        _phase(rank, "torch ground truth unavailable: %r" % (e,))
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    _phase(rank, "init_process_group done")
    for ln in _libnccl_lines():
        _phase(rank, "libnccl map: %s" % ln)
    try:
        device = lt.Device("cuda:" + str(rank))
        lt.cuda.set_device(rank)
        _phase(rank, "set_device(%d) done current_device=%d" % (rank, lt.cuda.current_device()))
        model = lt.nn.Sequential([lt.nn.Linear(4096, 4096, True)])
        model.to(device)
        _phase(rank, "model built")
        lt.distributed.fsdp_fully_shard(model)
        _phase(rank, "fsdp_fully_shard done")
        x = lt.Tensor.from_vector([0.5] * 4096, [1, 4096], device)
        y = model.forward(x)
        _phase(rank, "forward done")
        loss = y.sum()
        loss.backward()
        _phase(rank, "backward done")
        q.put((rank, y.to_vector()[:4]))
        _phase(rank, "q.put done")
    finally:
        _phase(rank, "shutdown begin")
        lt.distributed.shutdown()
        _phase(rank, "shutdown done")
    _phase(rank, "worker return")
    logf.close()


def main():
    logdir = os.environ.get("B5_E5_7_LOGDIR") or "."
    os.makedirs(logdir, exist_ok=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, 29645, q, logdir)) for r in range(2)]
    for p in procs:
        p.start()
    codes = {}
    for r, p in enumerate(procs):
        p.join(300)
        codes[r] = p.exitcode
        print("EXIT rank=%d code=%s" % (r, p.exitcode), flush=True)
    for r in (0, 1):
        assert codes[r] == 0, "a rank failed rank=%d code=%s" % (r, codes[r])
    results = [q.get(timeout=30) for _ in range(2)]
    print("ranks done:", sorted(r for r, _ in results), flush=True)


if __name__ == "__main__":
    main()
print("B5 E5-7 PASS", flush=True)
