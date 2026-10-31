import os
import sys

TREE = os.environ.get("LITETORCH_TREE") or (sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/litetorch")
sys.path.insert(0, os.path.join(TREE, "python"))

VARIANT = os.environ.get("B5_E5_7_VARIANT", "seq-forward")
if VARIANT not in ("seq-forward", "seq-call", "leaf-call"):
    raise SystemExit("unknown B5_E5_7_VARIANT=%r" % VARIANT)

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


def _avail(rank):
    try:
        return "nccl_available=%s" % (lt.distributed.NCCLBridge.get().is_available(),)
    except Exception as e:
        return "nccl_available_probe_failed: %r" % (e,)


def _hook(t, v, tb):
    print("B5 E5-7 FAIL: %s: %s" % (t.__name__, v), flush=True)
    sys.exit(1)


sys.excepthook = _hook


def worker(rank, world_size, port, q, logdir):
    faulthandler.enable()
    logpath = os.path.join(logdir, "b5-e5-7-v2-%s-rank%d.log" % (VARIANT, rank))
    logf = open(logpath, "w", buffering=1)
    sys.stdout = logf
    sys.stderr = logf
    _phase(rank, "start variant=%s pid=%d" % (VARIANT, os.getpid()))
    _phase(rank, "LITETORCH_NCCL_LIB=%r" % os.environ.get("LITETORCH_NCCL_LIB"))
    _phase(rank, "CUDA_VISIBLE_DEVICES=%r" % os.environ.get("CUDA_VISIBLE_DEVICES"))
    for ln in _libnccl_lines():
        _phase(rank, "libnccl map pre-torch: %s" % ln)
    try:
        import torch
        _phase(rank, "torch.cuda.device_count=%d current_device=%d" % (
            torch.cuda.device_count(), torch.cuda.current_device()))
    except Exception as e:
        _phase(rank, "torch ground truth unavailable: %r" % (e,))
    for ln in _libnccl_lines():
        _phase(rank, "libnccl map post-torch: %s" % ln)
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    _phase(rank, "init_process_group done %s" % _avail(rank))
    for ln in _libnccl_lines():
        _phase(rank, "libnccl map post-init: %s" % ln)
    try:
        device = lt.Device("cuda:" + str(rank))
        lt.cuda.set_device(rank)
        _phase(rank, "set_device(%d) done lt_device_count=%d current_device=%d" % (
            rank, lt.cuda.device_count(), lt.cuda.current_device()))
        if VARIANT == "leaf-call":
            model = lt.nn.Linear(4096, 4096, True)
        else:
            model = lt.nn.Sequential([lt.nn.Linear(4096, 4096, True)])
        model.to(device)
        _phase(rank, "model built")
        lt.distributed.fsdp_fully_shard(model)
        _phase(rank, "fsdp_fully_shard done %s" % _avail(rank))
        x = lt.Tensor.from_vector([0.5] * 4096, [1, 4096], device)
        if VARIANT == "seq-forward":
            y = model.forward(x)
        else:
            y = model(x)
        _phase(rank, "forward done %s" % _avail(rank))
        loss = y.sum()
        try:
            loss.backward()
        except Exception:
            _phase(rank, "backward raised %s" % _avail(rank))
            raise
        _phase(rank, "backward done %s" % _avail(rank))
        q.put((rank, y.to_vector()[:4]))
        _phase(rank, "q.put done")
    finally:
        _phase(rank, "shutdown begin")
        lt.distributed.shutdown()
        _phase(rank, "shutdown done %s" % _avail(rank))
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
    if codes[0] != 0 or codes[1] != 0:
        print("B5 E5-7 VARIANT %s FAIL: a rank failed rank=0 code=%s rank=1 code=%s" % (
            VARIANT, codes[0], codes[1]), flush=True)
        sys.exit(1)
    results = [q.get(timeout=30) for _ in range(2)]
    print("B5 E5-7 VARIANT %s PASS ranks done: %s" % (VARIANT, sorted(r for r, _ in results)), flush=True)


if __name__ == "__main__":
    main()
