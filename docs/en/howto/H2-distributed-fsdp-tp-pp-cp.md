# How-To H2 — Distributed: FSDP / TP / PP / CP Setup

The four parallelism dimensions, each with a runnable recipe. FSDP shards
parameters and gradients; tensor parallelism splits linear layers; pipeline
parallelism splits stages across devices; context parallelism splits the
sequence. All CPU examples below run against LiteTorch 88a4ba1.

Prerequisites: LiteTorch installed and importable. The multi-process
recipes use the same `mp` spawn launch as H1.1 — read that first.

> PyTorch differences touched here: D-06 (`.long()` — FIXED agent-v1
> @7c471db2, qa-r62: now returns INT64; the 88a4ba1 ext used for this
> batch predates the fix), D-11 (indexing is
> int/slice on dim 0 only — use `Ops.gather` / `index_select` instead).

## H2.1 — FSDP: shard a model across 2 ranks

Call `lt.distributed.fsdp_fully_shard(model)` after `init_process_group`
and after moving the model to the rank's device. It mutates the model in
place and returns `None`: one `FSDPUnit` per leaf module, grad hooks that
reduce-scatter on backward, forward hooks that gather and release
automatically. After this call, ordinary `model(x)` +
`loss.backward()` just works — there is no `fsdp_model` handle and no
manual gather/shard step.

FSDP shards model parameters (flattened, one local shard per rank) and
gradients. It does **not** shard optimizer states.

<!-- ex: id=H2.1-1 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp

import litetorch as lt

PORT = 29721


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        model = lt.nn.Sequential([
            lt.nn.Linear(1024, 1024, True),
            lt.nn.GELU(),
            lt.nn.Linear(1024, 1024, True),
        ])
        model.to(cpu)
        lt.distributed.fsdp_fully_shard(model)
        lt.distributed.fsdp_set_prefetch_limit(2)
        x = lt.Tensor.from_vector([0.01] * 2048, [2, 1024], cpu)
        y = model(x)
        print("Rank %d forward shape: %s" % (rank, list(y.shape)))
        loss = y.sum()
        loss.backward()
        print("Rank %d backward done" % rank)
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
        p.join(180)
        if p.is_alive():
            p.terminate()
    print("exit codes:", [p.exitcode for p in procs])


if __name__ == "__main__":
    main()
```

Verified on CPU, LiteTorch 88a4ba1. Expected output: both ranks print
their forward shape `[2, 1024]` and `backward done`, exit codes `[0, 0]`.

Two things this example deliberately does **not** do. First, it does not
claim identical losses across ranks: each rank randomly initializes its
own copy of the model, so per-rank losses differ. In real training, load
the same checkpoint on all ranks (or broadcast parameters from rank 0)
before calling `fsdp_fully_shard`. Second, it does not call
`FSDPUnit` methods (`prefetch_gather`, `ensure_gathered`,
`release_gathered`): they are unreachable from Python — no `py::init` is
bound and `FSDP::active_units()` is not bound, so no Python code can
obtain an `FSDPUnit` instance. Do not document them as callable. Tear
down with `lt.distributed.shutdown()`; `destroy_process_group()` does
not exist.

There is no `FullyShardedDataParallel` class in LiteTorch — an older
example called it, but it never existed in this tree. If you see it in
older material, it is stale.

## H2.2 — Tensor parallelism: split a linear layer

`ColumnParallelLinear` splits the weight matrix column-wise (each rank
computes a slice of the output features); `RowParallelLinear` splits it
row-wise (each rank computes a partial sum over a slice of the input
features). A Megatron-style MLP is one column-parallel projection, a
nonlinearity, and one row-parallel projection. This runs in a single
process on CPU; under a real TP group each rank holds its own shard and
the row-parallel layer all-reduces the partial sums.

<!-- ex: id=H2.2-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()


class TensorParallelMLP(lt.nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.col_proj = lt.nn.ColumnParallelLinear(hidden_dim, ffn_dim, True)
        self.gelu = lt.nn.GELU()
        self.row_proj = lt.nn.RowParallelLinear(ffn_dim, hidden_dim, True)

    def forward(self, x):
        h = self.col_proj.forward(x)
        h = self.gelu.forward(h)
        out = self.row_proj.forward(h)
        return out


lt.manual_seed(0)
mlp = TensorParallelMLP(16, 32)
mlp.to(device)
x = lt.Tensor.from_vector([0.1] * 32, [2, 16], device, False)
y = mlp.forward(x)
print("TP MLP output shape:", list(y.shape))
print("TP MLP output values:", y.to_vector()[:4])
```

Verified on CPU, LiteTorch 88a4ba1. Expected output: shape `[2, 16]`.
For the multi-rank form, wrap the same module construction in the H1.1
launch and initialize the TP group first
(`lt.distributed.init_tp_groups` after `init_process_group`); the
`FSDPTP` class composes FSDP sharding with tensor parallelism.

## H2.3 — Pipeline parallelism: 1F1B schedule over microbatches

`PipelineParallelModule` wraps one stage. `schedule_1f1b(microbatches)`
runs the one-forward-one-backward schedule over the microbatches to
minimize pipeline bubbles. This runs in a single process on CPU.

<!-- ex: id=H2.3-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

stage_module = lt.nn.Sequential([
    lt.nn.Linear(8, 8, True),
    lt.nn.GELU(),
])
stage_module.to(device)

pp_module = lt.distributed.PipelineParallelModule(stage_module)

mb1 = lt.Tensor.from_vector([1.0] * 8, [1, 8], device, False)
mb2 = lt.Tensor.from_vector([2.0] * 8, [1, 8], device, False)

outputs = pp_module.schedule_1f1b([mb1, mb2])
print("1F1B outputs:", len(outputs))
for i, out in enumerate(outputs):
    print("microbatch %d output shape: %s" % (i, list(out.shape)))
```

Verified on CPU, LiteTorch 88a4ba1. Expected output: 2 outputs, each
shape `[1, 8]`. For training, use `train_step_1f1b(microbatches,
loss_fn)` instead — it runs forward, backward, and the optimizer step
per microbatch. `InterleavedPipelineModule` is the interleaved
(chimera-style) variant of the same wrapper.

## H2.4 — Context parallelism: shard the sequence, gather it back

Context parallelism splits a long sequence across ranks. The collectives
that move the shards are `all_gather` (every rank gets the full tensor)
and `reduce_scatter` (every rank gets its reduced shard). This recipe
demonstrates the shard movement that a CP implementation is built on.

> Known gap: `lt.Ops.ring_attention(q, k, v, pg, causal)` is bound but
> not callable from Python — passing the `ProcessGroup` singleton raises
> a pybind holder error (`Unable to load a custom holder type from a
> default-holder instance`). Do not use it until the binding is fixed.
> Shard movement below is the working CP primitive today.

<!-- ex: id=H2.4-1 device=cpu backend=cpu runtime=~2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp

import litetorch as lt

PORT = 29724


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        pg = lt.distributed.ProcessGroup.get()
        shard = lt.Tensor.from_vector([float(rank * 4 + i) for i in range(4)], [4], cpu, False)
        full = lt.Tensor.from_vector([0.0] * 8, [8], cpu, False)
        pg.all_gather(shard, full)
        print("Rank %d all_gather: %s" % (rank, full.to_vector()))
        full2 = lt.Tensor.from_vector([float(i + 1) for i in range(8)], [8], cpu, False)
        shard2 = lt.Tensor.from_vector([0.0] * 4, [4], cpu, False)
        pg.reduce_scatter(shard2, full2)
        print("Rank %d reduce_scatter: %s" % (rank, shard2.to_vector()))
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
`all_gather: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]` (the two
4-element shards concatenated in rank order) and `reduce_scatter`
`[2.0, 4.0, 6.0, 8.0]` on rank 0 / `[10.0, 12.0, 14.0, 16.0]` on rank
1 (the element-wise sum of `[1..8]` and `[1..8]` is `[2..16]`; each rank
receives its 4-element shard). Exit codes `[0, 0]`.

## H2.5 — DeviceMesh and DTensor: name the sharding

A `DeviceMesh` names the process grid; a `DTensor` pairs a local tensor
with the mesh and a placement per dimension (`SHARD`, `REPLICATE`, or
`PARTIAL`). This is metadata plus local storage — the collectives in
H2.4 are what actually move data.

<!-- ex: id=H2.5-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

mesh = lt.distributed.DeviceMesh([2, 2], "tp,dp")

placement_shard = lt.distributed.Placement()
placement_shard.type = lt.distributed.PlacementType.SHARD
placement_shard.dim = 0

local_data = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], lt.auto_device(), False)
dtensor = lt.distributed.DTensor(local_data, mesh, [placement_shard])
print("DTensor local shape:", list(dtensor.local_tensor.shape))
print("DTensor mesh shape:", mesh.mesh_shape)
print("DTensor placement:", dtensor.placements[0].type, "dim", dtensor.placements[0].dim)
```

Verified on CPU, LiteTorch 88a4ba1. Expected output: local shape
`[2, 2]`, mesh shape `[2, 2]`, placement `SHARD` dim `0`.
