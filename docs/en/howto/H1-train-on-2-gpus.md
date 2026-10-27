# How-To H1 — Train on 2 GPUs

Two processes, one per GPU, joined by a `ProcessGroup`. The same launch
pattern works on CPU (this page, runnable here) and on GPU (H1.3, Kaggle).
LiteTorch has no `torchrun` equivalent: you spawn the processes yourself
with `multiprocessing` and each rank calls `init_process_group` first.

Prerequisites: LiteTorch installed and importable. CPU examples below run
against LiteTorch 88a4ba1. GPU steps are marked `REQUIRES-GPU` and verified
on Kaggle in Batch B5.

> PyTorch differences touched here: D-02 (`.to(device)` silently falls back
> to CPU when the accelerator is unavailable — always check `t.device`
> afterwards), D-03 (`.to()` takes a `Device`, no string form).

## H1.1 — Launch two ranks and run collectives

Spawn two processes with `mp.get_context("spawn")`. Each rank initializes
its process group, then `all_reduce` sums a tensor across ranks and
`broadcast` copies rank 0's tensor to every rank. Always `shutdown()` in a
`finally` block — a rank that exits without shutting down leaves its peer
waiting on the socket.

<!-- ex: id=H1.1-1 device=cpu backend=cpu runtime=~2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp

import litetorch as lt

PORT = 29711


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        pg = lt.distributed.ProcessGroup.get()
        t = lt.Tensor.from_vector([float(rank + 1)] * 4, [4], cpu, False)
        pg.all_reduce(t)
        print("Rank %d all_reduce sum: %s" % (rank, t.to_vector()))
        b = lt.Tensor.from_vector([0.0] * 4, [4], cpu, False)
        if rank == 0:
            b = lt.Tensor.from_vector([7.0] * 4, [4], cpu, False)
        pg.broadcast(b, 0)
        print("Rank %d broadcast: %s" % (rank, b.to_vector()))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(2):
        p = ctx.Process(target=worker, args=(rank, 2, PORT))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(120)
        if p.is_alive():
            p.terminate()
    print("exit codes:", [p.exitcode for p in procs])


if __name__ == "__main__":
    main()
```

Verified on CPU, LiteTorch 88a4ba1. Expected output: both ranks print
`all_reduce sum: [3.0, 3.0, 3.0, 3.0]` (1 + 2) and
`broadcast: [7.0, 7.0, 7.0, 7.0]`, exit codes `[0, 0]`.

The other collectives follow the same shape: `all_gather(shard, full)`,
`reduce_scatter(shard, full)`, `send_tensor(tensor, dst)` /
`recv_tensor(tensor, src)`, `barrier()`. See H2.4 for `all_gather` /
`reduce_scatter` in use.

## H1.2 — Mismatched collectives throw on every rank

Before reducing, ranks exchange dtype and shape headers. If they disagree,
every rank raises `RuntimeError: collective tensor dtype/shape mismatch
across ranks` — no rank hangs, no silent wrong result. The fix is always
the same: make the tensor dtype and shape identical on all ranks before
the collective call.

<!-- ex: id=H1.2-1 device=cpu backend=cpu runtime=~2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp

import litetorch as lt

PORT = 29712


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        pg = lt.distributed.ProcessGroup.get()
        n = 4 if rank == 0 else 3
        t = lt.Tensor.from_vector([1.0] * n, [n], cpu, False)
        try:
            pg.all_reduce(t)
            print("Rank %d: NO ERROR (unexpected)" % rank)
        except RuntimeError as e:
            print("Rank %d: threw as expected: %s" % (rank, str(e)))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(2):
        p = ctx.Process(target=worker, args=(rank, 2, PORT))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(60)
        if p.is_alive():
            p.terminate()
    print("exit codes:", [p.exitcode for p in procs])


if __name__ == "__main__":
    main()
```

Verified on CPU, LiteTorch 88a4ba1. Expected output: both ranks print
`threw as expected: collective tensor dtype/shape mismatch across ranks`,
exit codes `[0, 0]`. This is the same check the 2-GPU NCCL path applies
before a collective runs — a shape bug fails loud instead of corrupting
the reduction.

## H1.3 — The same launch on 2 GPUs (REQUIRES-GPU)

On GPU the pattern is identical, with one change per rank: move the model
to that rank's GPU device and check the device stuck. Never assume the
move worked — on a box without CUDA, `.to()` silently returns a CPU
tensor (D-02).

<!-- ex: id=H1.3-1 device=gpu backend=cuda runtime=~60s torch-parity=n/a status=unverified -->
```python
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


if __name__ == "__main__":
    main()
```

REQUIRES-GPU. Verify on Kaggle (Batch B5) with the pre-flight gate first:
the `Native GPU (CUDA/HIP)` init line is present, `liblitetorch_gpu.so`
sits next to the extension under `<tree>/python/`, and
`lt.cuda.device_count()` returns 2. For a real 2-GPU training step with
gradient synchronization, shard the model with FSDP (H2.1) instead of
replicating it — replication alone does not synchronize gradients.
