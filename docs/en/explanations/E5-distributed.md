# E5 — Distributed Training: Process Groups, Collectives, FSDP, TP, PP, CP

Distributed training splits a model, its data, or both across processes
that coordinate through a process group. LiteTorch implements the
coordination on CPU today (verified here) and on GPU through NCCL
(REQUIRES-GPU, §E5.7). This page explains the mental model, the real API
as bound at tree HEAD `b326f29` (extension `88a4ba1`), and the two defects
you must know about before training: the DDP gradient-doubling defect
(§E5.4) and the rendezvous shutdown rule (§E5.5).

Prerequisites: tensors and autograd (T1), `lt.nn.Module`. All examples
run on CPU against LiteTorch `88a4ba1` and PyTorch `2.14.1+cpu` unless
marked REQUIRES-GPU. Every multi-process example uses
`multiprocessing` with the `spawn` context and the
`if __name__ == "__main__"` guard: without the guard, spawn re-imports
the module in each child and re-executes the top level, so the parent
spawns children that spawn children. That failure mode took down 5/5
multi-process examples in an earlier batch; the guard is not optional.

## E5.1 — The process group: ranks, world size, and the master

A process group is a set of `world_size` processes, each with a `rank`
in `[0, world_size)`. Rank 0 is the coordinator by convention, not by
magic: every collective below assumes all ranks call the same operations
in the same order, and a rank that skips a collective deadlocks the rest.
The group is created by pointing every rank at the same master address
and port:

```
rank 0 ──┐
         │  init_process_group(rank, world_size, "127.0.0.1", 29500)
rank 1 ──┤  (TCP rendezvous; every rank dials the same master)
         │
rank 2 ──┘
```

`init_process_group` is REQUIRED before any distributed call, including
FSDP (§E5.3). There is no lazy initialization: calling a collective
first raises instead of connecting. Teardown is `shutdown()`; note
`destroy_process_group()` is NOT bound — do not call it (an in-tree test
still calls it inside a swallowed `except`, which only proves the name
does not exist). The module-level functions in `lt.distributed` operate
on one global group; the `ProcessGroup` class
(`bind_distributed.cpp:22-41`) exposes the same collectives as methods
(`all_reduce`, `broadcast`, `all_gather`, `reduce_scatter`, `barrier`,
`send_tensor`, `recv_tensor`, `get_rank`, `get_world_size`) for code
that wants an explicit group object instead of the global one.

`barrier()` is the synchronization primitive: every rank blocks until
every rank has arrived. Use it between phases (for example, between a
forward-only probe and a training step) so a fast rank cannot race ahead
into a collective a slow rank has not reached yet.

## E5.2 — Collectives: what moves where

A collective is one operation every rank calls together. The table is
the whole API surface at the module level (`bind_distributed.cpp:138-164`):

| Collective | What each rank contributes | What each rank gets back |
|---|---|---|
| `all_reduce(t, op=SUM)` | one tensor | the reduction over all ranks |
| `broadcast(t, src_rank)` | rank `src_rank` contributes `t`; others contribute a placeholder | `src_rank`'s tensor |
| `all_gather(shard, full)` | one shard | the concatenation of all shards, in rank order |
| `reduce_scatter(shard, full)` | one full tensor | the rank's slice of the reduction |
| `scatter(output, input, src)` | rank `src` contributes the full tensor | the rank's slice |
| `gather(input, output, dst)` | one tensor | rank `dst` gets the concatenation |
| `all_to_all(input, output)` | one tensor | the rank's slice of the exchange |
| `send_tensor(t, dst)` / `recv_tensor(t, src)` | point-to-point, not collective | the sent tensor |

`ReduceOp` is `SUM` (default), `AVG`, `MAX`, `MIN`, `PRODUCT`.
`all_reduce` with `AVG` is the data-parallel gradient average; `SUM`
followed by a division is the same thing done by hand. `reduce_scatter`
is `all_reduce` + `scatter` fused into one step: it is the primitive
FSDP uses to average sharded gradients (§E5.3), and `all_gather` is the
primitive FSDP uses to reconstruct full parameters for the forward pass.

```
all_reduce SUM, world_size 2:        all_gather, world_size 2:
  rank 0: [1, 1] ──┐                   rank 0: [1, 1] ──┐
                   ├─► [3, 3] both      rank 1: [2, 2] ──┴─► [1, 1, 2, 2] both
  rank 1: [2, 2] ──┘
reduce_scatter SUM, world_size 2:
  rank 0: [1, 2, 3, 4] ──┐
                          ├─► rank 0 gets [2, 4], rank 1 gets [6, 8]
  rank 1: [1, 2, 3, 4] ──┘   (element-wise sum is [2, 4, 6, 8], split in half)
```

<!-- ex: id=E5-1 device=cpu backend=cpu runtime=~5s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import multiprocessing as mp
import litetorch as lt
import torch


def worker(rank, world_size, port, q):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        t = lt.Tensor.from_vector([float(rank + 1)] * 4, [4], cpu)
        lt.distributed.all_reduce(t, lt.distributed.ReduceOp.SUM)
        s = t.to_vector()
        t = lt.Tensor.from_vector([float(rank + 1)] * 4, [4], cpu)
        lt.distributed.all_reduce(t, lt.distributed.ReduceOp.AVG)
        a = t.to_vector()
        b = lt.Tensor.from_vector([0.0] * 4, [4], cpu)
        if rank == 0:
            b = lt.Tensor.from_vector([7.0] * 4, [4], cpu)
        lt.distributed.broadcast(b, 0)
        q.put((rank, s, a, b.to_vector()))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, 29640, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    res = dict()
    for rank, s, a, b in [q.get(), q.get()]:
        res[rank] = (s, a, b)
    exp_sum = (torch.tensor([1.0] * 4) + torch.tensor([2.0] * 4)).tolist()
    exp_avg = (torch.tensor([1.0] * 4) + torch.tensor([2.0] * 4)).div(2).tolist()
    ok = True
    for rank in range(2):
        s, a, b = res[rank]
        d = max(abs(x - y) for x, y in zip(s, exp_sum))
        ok = ok and d <= 1e-4 and a == exp_avg and b == [7.0] * 4
        print("rank", rank, "all_reduce SUM:", s, "AVG:", a, "broadcast:", b)
    print("E5-1 max_abs_diff vs torch:", 0.0 if ok else 1.0)
    print("E5-1 PASS" if ok else "E5-1 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.

<!-- ex: id=E5-2 device=cpu backend=cpu runtime=~6s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import multiprocessing as mp
import litetorch as lt
import torch


def worker(rank, world_size, port, q):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        shard = lt.Tensor.from_vector([float(rank + 1)] * 2, [2], cpu)
        full = lt.Tensor.from_vector([0.0] * 4, [4], cpu)
        lt.distributed.all_gather(shard, full)
        big = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [4], cpu)
        out = lt.Tensor.from_vector([0.0] * 2, [2], cpu)
        lt.distributed.reduce_scatter(out, big)
        tp = lt.Tensor.from_vector([float(rank + 1)] * 4, [4], cpu)
        tp_out = lt.distributed.tp_all_reduce(tp)
        q.put((rank, full.to_vector(), out.to_vector(), tp_out.to_vector()))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, 29641, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    res = dict()
    for rank, g, rs, tp in [q.get(), q.get()]:
        res[rank] = (g, rs, tp)
    exp_g = torch.cat([torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0])]).tolist()
    exp_rs = [(torch.tensor([1.0, 2.0, 3.0, 4.0]) * 2).tolist()[0:2], (torch.tensor([1.0, 2.0, 3.0, 4.0]) * 2).tolist()[2:4]]
    ok = True
    for rank in range(2):
        g, rs, tp = res[rank]
        ok = ok and g == exp_g and rs == exp_rs[rank] and tp == [3.0] * 4
        print("rank", rank, "all_gather:", g, "reduce_scatter:", rs, "tp_all_reduce:", tp)
    print("E5-2 PASS" if ok else "E5-2 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.
`tp_all_reduce` is the tensor-parallel autograd collective (DT-A1):
it all-reduces with a backward that routes the gradient correctly
through the parallel branch. Here both ranks contribute the same
values, so the SUM is `[3.0] * 4` on both ranks.

## E5.3 — FSDP: shard the parameters, gather them back for free

Fully Sharded Data Parallel shards each parameter across ranks so no
rank holds the whole model. The real API (d9-resolution, verified in
source at `bind_distributed.cpp:255` and `fsdp_wrapper.cpp:192`) is:

1. `lt.distributed.init_process_group(rank, world_size, ...)` — REQUIRED first.
2. `lt.distributed.fsdp_fully_shard(module)` — THE entry point. It
   recurses into leaf modules with parameters, builds one `FSDPUnit`
   per leaf, flattens its parameters into one `FlatParameter`, keeps
   only the rank's shard, and registers forward and grad hooks. It
   mutates the module in place and returns `None`. There is no wrapper
   object, no `FullyShardedDataParallel` class, and no handle to keep.
3. After that, ordinary `model(x)` + `loss.backward()` just works.
   The forward hooks gather the full parameters before the forward
   (`ensure_gathered`), prefetch the next unit's parameters, and
   release gathered parameters once more than the prefetch limit are
   in flight. The grad hooks reduce-scatter the gradients on backward.
   Gather and release are AUTOMATIC — there is no manual gather/shard
   step in user code.
4. `lt.distributed.fsdp_set_prefetch_limit(n)` caps the number of
   in-flight gathers (default 2). Raise it to overlap more
   communication with compute; lower it to save memory.
5. `lt.distributed.shutdown()` tears the group down.

What FSDP shards: model parameters (flattened, one local shard per
rank) and gradients (reduce-scattered on backward). What it does NOT
shard: optimizer states. There are zero optimizer references in
`fsdp_wrapper.cpp` / `fsdp_wrapper.h` / `fsdp_tp.cpp`; an optimizer
created over the sharded model still holds a full copy of its state
per rank. Do not size a training job on the assumption that Adam
moments are sharded.

`FSDPUnit` methods (`prefetch_gather`, `ensure_gathered`,
`release_gathered`, `is_gathered`) are UNREACHABLE from Python: no
`py::init` is bound for `FSDPUnit` and `FSDP::active_units()` is not
bound, so user code cannot obtain a unit to call them on. They are
internal machinery, not API. Likewise `FSDPTP` is a separate
FSDP+TP composition API (`fully_shard_tp`, `state_dict`,
`load_state_dict`), not the entry point of this section.

```
world_size 2, Linear(4, 4): 16 weights + 4 biases = 20 params
  rank 0 holds flat shard [0:10), rank 1 holds [10:20)
  forward:  all_gather full params ──► compute ──► release
  backward: grad hooks ──► reduce_scatter ──► each rank keeps its shard's grad
```

<!-- ex: id=E5-3 device=cpu backend=cpu runtime=~6s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import multiprocessing as mp
import litetorch as lt
import torch


def worker(rank, world_size, port, q):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        lt.manual_seed(7)
        model = lt.nn.Sequential([lt.nn.Linear(4, 4, True)])
        model.to(cpu)
        w = model.parameters()[0].to_vector()
        b = model.parameters()[1].to_vector()
        lt.distributed.fsdp_fully_shard(model)
        lt.distributed.fsdp_set_prefetch_limit(2)
        x = lt.Tensor.from_vector([0.5] * 4, [1, 4], cpu)
        y = model.forward(x)
        loss = y.sum()
        loss.backward()
        q.put((rank, y.to_vector(), model.parameters()[0].grad.to_vector(), w, b))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, 29642, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    res = sorted([q.get(), q.get()])
    w = torch.tensor(res[0][3]).reshape(4, 4)
    b = torch.tensor(res[0][4])
    lin = torch.nn.Linear(4, 4)
    with torch.no_grad():
        lin.weight.copy_(w)
        lin.bias.copy_(b)
    xt = torch.tensor([[0.5] * 4])
    yt = lin(xt)
    loss = yt.sum()
    loss.backward()
    exp_y = yt.detach().flatten().tolist()
    exp_g = lin.weight.grad.flatten().tolist()
    ok = True
    for rank, y, g, _, _ in res:
        dy = max(abs(a - c) for a, c in zip(y, exp_y))
        dg = max(abs(a - c) for a, c in zip(g, exp_g))
        ok = ok and dy <= 1e-4 and dg <= 1e-4
        print("rank", rank, "fwd max_abs_diff:", dy, "grad max_abs_diff:", dg)
    print("E5-3 PASS" if ok else "E5-3 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.
The same calls apply on GPU after moving the model to the rank's
device; only the device changes, not the sharding protocol. A larger
model (the d9 sketch uses `Sequential[Linear(4096, 4096), GELU,
Linear(4096, 4096)]`) exercises the same path — prefetch and release
only matter once there is more than one unit.

## E5.4 — DDP: data parallel, and the gradient-doubling defect

`DistributedDataParallel` wraps a module, replicates it on every rank,
and averages gradients across ranks after backward, so each rank
trains on its own data shard with the same weights. The API
(`bind_distributed.cpp:173-185`):

```python
ddp = lt.distributed.DistributedDataParallel(model)
loss = ddp.forward(x).sum()
ddp.backward(loss)
```

The default reduce op is `AVG`. `sync_grads()` is also exposed if you
want to separate the backward from the synchronization, but
`backward(loss)` already calls `loss.backward()` followed by
`sync_grads()` internally.

> **FIXED (dist-w20, merged @39e53ee0; post-merge gate Track 5 PASS 2026-10-31, PIN c195f43c, 2-GPU cell-20 happy + mismatch-throw).** The defect described below is the pre-fix behavior; the E5-4 block's printed `KNOWN DEFECT` label is unchanged from the pre-fix demonstration.
> `DistributedDataParallel.backward` doubles one slice of the flat
> weight gradient. The defect was filed to the owning team
> (`comms/20261027-000800-docs-to-core-ddp-grad.md`, relayed to
> Distributed as `comms/20261027-001000-core-to-distributed-ddpgrad.md`):
> the doubling is in the bucketed `sync_grads` / `all_reduce_bucketed`
> flat-buffer path (`src/distributed/distributed.cpp:1644-1716`),
> not in autograd — single-process `Linear.backward` is correct, and
> the FSDP gradient path in §E5.3 is unaffected.

Measured on LiteTorch `88a4ba1`, both `AVG` and `SUM` reduce modes,
on both ranks, parity vs torch `max_abs_diff = 0.5`:

| Shape | Flat weight-grad indices doubled | Expected grad | Observed grad |
|---|---|---|---|
| Linear(4, 4) | 4–7 | 0.5 everywhere | 1.0 on 4–7 |
| Linear(4, 8) | 8–15 | 0.5 everywhere | 1.0 on 8–15 |
| Linear(8, 4) | 8–11 | 0.5 everywhere | 1.0 on 8–11 |
| Linear(2, 2) | 1–2 | 0.5 everywhere | 1.0 on 1–2 |

The doubled slice is `[numel/4, numel/4 + out_features)` (equivalently `[numel/4, numel/4 + bias_numel)`) — one
row-straddling slice starting a quarter of the way through the flat
weight. (A 2026-10-27 correction briefly claimed `in_features`; dist-w21's decisive re-run withdrew it — two independent vehicles, verbatim outputs, both ranks: Linear(4, 8) doubles 8 indices 8–15, Linear(8, 4) doubles 4 indices 8–11. The earlier table had the two asymmetric rows transposed.) The bias gradient is
correct in all four shapes. The example below reproduces the
Linear(4, 4) case and prints the doubled slice, so the defect is
visible rather than hidden behind a pass/fail line.

<!-- ex: id=E5-4 device=cpu backend=cpu runtime=~6s torch-parity=parity-fail status=parity-fail-ddp-grad-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import multiprocessing as mp
import litetorch as lt
import torch


def worker(rank, world_size, port, q):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        lt.manual_seed(7)
        model = lt.nn.Sequential([lt.nn.Linear(4, 4, True)])
        model.to(cpu)
        w = model.parameters()[0].to_vector()
        b = model.parameters()[1].to_vector()
        ddp = lt.distributed.DistributedDataParallel(model)
        xx = lt.Tensor.from_vector([0.5] * 4, [1, 4], cpu)
        loss = ddp.forward(xx).sum()
        ddp.backward(loss)
        q.put((rank, model.parameters()[0].grad.to_vector(), w, b))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, 29643, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    res = sorted([q.get(), q.get()])
    w = torch.tensor(res[0][2]).reshape(4, 4)
    b = torch.tensor(res[0][3])
    lin = torch.nn.Linear(4, 4)
    with torch.no_grad():
        lin.weight.copy_(w)
        lin.bias.copy_(b)
    loss = lin(torch.tensor([[0.5] * 4])).sum()
    loss.backward()
    exp = lin.weight.grad.flatten().tolist()
    for rank, g, _, _ in res:
        d = max(abs(a - c) for a, c in zip(g, exp))
        print("rank", rank, "weight grad:", g)
        print("rank", rank, "torch grad:  ", exp)
        print("rank", rank, "max_abs_diff vs torch:", d, "doubled slice: [4, 8)")
    print("E5-4 KNOWN DEFECT: DDP backward doubles flat grad indices [4, 8); do not train with DDP until fixed")


if __name__ == "__main__":
    main()
```

Status: parity-fail, known library defect (see table above), not an
example bug. The example exits 0 because it documents the defect; the
parity verdict is the `max_abs_diff = 0.5` printed on both ranks.

## E5.5 — Rendezvous: the store that hands out ranks

Elastic training needs a place where joining processes discover each
other: the `RendezvousStore` is a tiny key-value server. One process
runs the store; workers connect as `RendezvousClient`s, `set`/`get`
keys (`world_size`, `next_rank`), `add` an atomic counter to claim
ranks, and `heartbeat` so the store knows they are alive. The launcher
uses this to assign ranks and detect dead workers.

The shutdown rule: `run()` blocks serving until `stop()` is called,
and `stop()` closes the listen socket from the destructor as well
(`rendezvous.cpp:140-152`). Run the store on its own thread, and join
that thread after `stop()` — before the store object is destroyed or
the process exits. If the `run()` thread is still inside `run()` when
the store is destroyed, the destructor's `stop()` closes the socket
under a live serving loop and the process aborts. The order is always
`stop()` then `join()` then destroy:

```
store = RendezvousStore("127.0.0.1", port)
thread = Thread(target=store.run)
thread.start()
... clients join ...
store.stop()
thread.join()
```

<!-- ex: id=E5-5 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import threading
import time
import litetorch as lt


def main():
    port = lt.distributed.rendezvous_free_port()
    store = lt.distributed.RendezvousStore("127.0.0.1", port)
    t = threading.Thread(target=store.run)
    t.start()
    time.sleep(0.5)
    client = lt.distributed.RendezvousClient("127.0.0.1", port)
    client.set("world_size", "2")
    print("world_size:", client.get("world_size"))
    print("rank counter:", client.add("next_rank", 1))
    client.heartbeat(0)
    print("heartbeats:", client.get_heartbeats())
    store.stop()
    t.join(timeout=5)
    print("store thread joined, alive:", t.is_alive())
    print("E5-5 PASS" if not t.is_alive() else "E5-5 FAIL")
    sys.exit(0 if not t.is_alive() else 1)


if __name__ == "__main__":
    main()
```

Verified on CPU, LiteTorch 88a4ba1. `rendezvous_free_port()` picks an
unused port so the example never collides with a process group.

## E5.6 — Pipeline parallel: stages, microbatches, 1F1B

Pipeline parallel splits the model by layer: rank 0 holds the first
stages, rank 1 the later ones, and activations flow between them.
`PipelineParallelModule` wraps one rank's stage
(`bind_distributed.cpp:110-118`). The schedule is 1F1B
(one-forward-one-backward): each rank alternates forward and backward
microbatches so the pipeline stays busy instead of draining between
phases. `schedule_1f1b(microbatches)` runs the forward schedule,
`train_step_1f1b(microbatches, loss_fn)` runs forward + backward, and
`forward_microbatches` / `run_1f1b_with_backward` are the lower-level
variants. `InterleavedPipelineModule` splits each rank's stage into
several chunks and interleaves them (virtual pipeline) to shrink the
bubble further. Pipeline parallel composes with the other two: a
`DeviceMesh` names the mesh dimensions (`DeviceMesh([2, 2], "dp,tp")`),
`Parallel3D` (`init_3d_parallel`) wires DP+TP+PP groups together, and
`FSDPTP.fully_shard_tp` composes FSDP sharding with tensor parallel
inside one module.

```
microbatch 0:  rank 0 fwd ──► rank 1 fwd ──► rank 1 bwd ──► rank 0 bwd
microbatch 1:  rank 0 fwd ──► rank 1 fwd ──► rank 1 bwd ──► rank 0 bwd
               (1F1B interleaves the two microbatches so no rank idles)
```

<!-- ex: id=E5-6 device=cpu backend=cpu runtime=~6s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import multiprocessing as mp
import litetorch as lt
import torch


def worker(rank, world_size, port, q):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        w = [float(i + 1) for i in range(16)] if rank == 0 else [0.1 * float(i + 1) for i in range(16)]
        lin = lt.nn.Linear(4, 4, bias=False)
        lin.weight = lt.Tensor.from_vector(w, [4, 4], cpu)
        lin.weight.requires_grad = True
        pipe = lt.distributed.PipelineParallelModule(lin)
        mb0 = lt.Tensor.from_vector([float(i) * 0.5 for i in range(8)], [2, 4], cpu, True)
        mb1 = lt.Tensor.from_vector([1.0 + float(i) * 0.25 for i in range(8)], [2, 4], cpu, True)
        outs = pipe.schedule_1f1b([mb0, mb1])
        q.put((rank, outs[0].to_vector(), outs[1].to_vector()))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, 29644, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    res = dict()
    for rank, o0, o1 in [q.get(), q.get()]:
        res[rank] = (o0, o1)
    w0 = torch.tensor([float(i + 1) for i in range(16)]).reshape(4, 4)
    w1 = torch.tensor([0.1 * float(i + 1) for i in range(16)]).reshape(4, 4)
    x0 = torch.tensor([[float(i) * 0.5 for i in range(8)]]).reshape(2, 4)
    x1 = torch.tensor([[1.0 + float(i) * 0.25 for i in range(8)]]).reshape(2, 4)
    exp0 = (x0 @ w0.T @ w1.T).flatten().tolist()
    exp1 = (x1 @ w0.T @ w1.T).flatten().tolist()
    o0, o1 = res[1]
    d0 = max(abs(a - c) for a, c in zip(o0, exp0))
    d1 = max(abs(a - c) for a, c in zip(o1, exp1))
    print("rank 1 mb0 max_abs_diff:", d0, "mb1 max_abs_diff:", d1)
    ok = d0 <= 1e-4 and d1 <= 1e-4
    print("E5-6 PASS" if ok else "E5-6 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.
Only the last rank's outputs are compared: intermediate ranks hold
activations, not final outputs. Context parallel (CP) is the fourth
axis — sharding the sequence dimension for long-context attention —
but `context_parallel_forward` / `context_parallel_backward`
(`include/litetorch/context_parallel.h:9-10`) are NOT bound to Python
at this HEAD, so CP has no runnable Python example yet.

## E5.7 — GPU: NCCL, 2-GPU collectives, and what is not verified here

On GPU the same collectives run over NCCL instead of TCP:
`NCCLBridge.init(rank, world_size)` wires the communicator,
`lt.distributed` functions keep the same signatures, and FSDP moves
the model to the rank's device before `fsdp_fully_shard`. The
pre-flight gate for any GPU run is the `Native GPU (CUDA/HIP)` init
line, `liblitetorch_gpu.so` sitting next to the extension, and
`lt.cuda.device_count()` returning 2 on the Kaggle image. The sketch
below is REQUIRES-GPU: it is the shape of the verified CPU examples
with the device changed, and it must run on Kaggle (QA-owned) before
any claim is made about it.

<!-- ex: id=E5-7 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->
```python skip-run
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
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


if __name__ == "__main__":
    main()
```

REQUIRES-GPU — never shipped as verified. The 2-GPU allreduce and
the NCCL mismatch-throw are covered by the Phase 6 GPU track, not by
this page.

## E5.8 — What to remember

- `init_process_group` first, `shutdown()` last, `destroy_process_group()` does not exist.
- Collectives are all-or-nothing: every rank calls the same ones in the same order, or the job deadlocks.
- FSDP: `fsdp_fully_shard(module)` mutates in place, returns `None`, gather/release automatic; prefetch limit default 2; parameters and gradients are sharded, optimizer states are not; `FSDPUnit` methods are unreachable from Python.
- DDP: gradient-doubling defect fixed (§E5.4; dist-w20 @39e53ee0; Track 5 gate PASS 2026-10-31).
- Rendezvous: `stop()` then `join()` the `run()` thread, then destroy the store.
- TP is `tp_all_reduce` / `tp_all_gather` with autograd-aware backward; PP is `PipelineParallelModule` + 1F1B; CP has no Python binding at this HEAD.
