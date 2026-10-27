# Distributed — API Reference

`litetorch.distributed` is the distributed-training surface: process groups and collectives, 3D parallelism (DP/TP/PP), DTensor/DeviceMesh, FSDP, pipeline parallelism, sharded checkpoints, and elastic restart (rendezvous + heartbeats). All compute is C++14; Python is the pybind11 interface layer.

- **Source of truth:** `src/bindings/bind_distributed.cpp` (321 lines, Python bindings), `include/litetorch/distributed.h`, `fsdp_wrapper.h`, `fsdp_tp.h`, `device_mesh.h`, `dtensor.h`, `flat_param.h`, `dist_checkpoint.h`, `elastic.h`, `rendezvous.h`, `parallel_3d.h`, `pipeline.h`.
- **Tree:** agent-v1@02cc271a / local HEAD 89afd67 (docs baseline is the 88a4ba1 extension; B5 re-verifies at head).
- **Conventions:** `cpu = lt.Device("cpu")` in every example. Multi-process examples use `mp.get_context("spawn")` with 2 processes and a distinct port per example. Every example is self-contained and runnable on CPU. Torch parity is FP32 1e-4 unless the header says otherwise.
- **Related deviations:** D-02 (`to(device)` silent CPU fallback — a rank that falls back to CPU still joins the group, silently slow). See `plan/differences.md`.
- **NCCL:** collectives use NCCL when a native GPU backend is present. **Status, settled 2026-10-27 by dist-w22: the `LITETORCH_NCCL_LIB` selection mechanism is a Kaggle-verified patch, not yet merged** (dist-w15 + the whole 2-GPU round w10–w13 were approved and Kaggle-verified 2026-10-19 on a patched session tree, but no merge step ever ran; the patches sit unmerged in `patches/distributed/`). Current behavior at HEAD b326f29 is fail-soft dlopen: `dlopen("libnccl.so")` → `dlopen("librccl.so")`, and a missing library sets `available_=false` instead of failing closed. Do not write code that sets `LITETORCH_NCCL_LIB` expecting it to be honored until the w10→w15 set merges. On a CPU-only box the process group uses the CPU transport; NCCL is not required.
- **Failure modes:** calling any collective before `init_process_group` raises; `init_tp_groups` raises if the process group is not initialized; mismatched tensor shapes/dtypes across ranks throw (mismatch-throw) instead of hanging; a rank that never joins makes the others block at the first collective — always `shutdown()` in a `finally` block. There is no `destroy_process_group()` — use `shutdown()`.

## Process groups and collectives

### `distributed.init_process_group`

- **Signature:** `init_process_group(rank, world_size, master_addr="127.0.0.1", master_port=29500) -> None`
- **Source:** `src/bindings/bind_distributed.cpp:138`
- **Parameters:** `rank` — int, this process's rank in `[0, world_size)`. `world_size` — int, total ranks. `master_addr` — rendezvous address, default `"127.0.0.1"`. `master_port` — rendezvous port, default 29500. Every rank must pass the same `world_size`, `master_addr`, and `master_port`; ranks must be distinct.
- **Returns:** `None`. Initializes the singleton process group. Calling it twice without `shutdown()` in between is an error.
- **Edge cases:** a second process that passes a different `world_size` hangs or throws at the first collective. The rendezvous port must be free; use `rendezvous_free_port()` to pick one.
- **PyTorch:** `torch.distributed.init_process_group(backend, rank, world_size)` — equivalent in shape; LiteTorch has no `backend` argument (the transport is chosen by the available backend) and no `init_method` string.

### `distributed.shutdown`

- **Signature:** `shutdown() -> None`
- **Source:** `src/bindings/bind_distributed.cpp:141`
- **Returns:** `None`. Tears the process group down. Call it in a `finally` block.
- **Edge cases:** `destroy_process_group()` does NOT exist in LiteTorch — do not call it (it raises `AttributeError`).
- **PyTorch:** `torch.distributed.destroy_process_group()` — same role, different name.

### `distributed.barrier`

- **Signature:** `barrier() -> None`
- **Source:** `src/bindings/bind_distributed.cpp:142`
- **Returns:** `None`. Blocks until every rank reaches the barrier.
- **Edge cases:** a rank that never calls `barrier()` deadlocks the others. There is no timeout argument.
- **PyTorch:** `torch.distributed.barrier()` — equivalent.

### `distributed.all_reduce`

- **Signature:** `all_reduce(tensor) -> None`; `all_reduce(tensor, op=ReduceOp.SUM) -> None`
- **Source:** `src/bindings/bind_distributed.cpp:143-148`
- **Parameters:** `tensor` — the tensor to reduce in place (every rank contributes its own values; every rank receives the reduced result). `op` — a `ReduceOp`, default `SUM`.
- **Returns:** `None`. The reduction is in place: `tensor` holds the result on return.
- **Edge cases:** all ranks must pass tensors of the same shape and dtype or the call throws (mismatch-throw). The no-`op` form always sums.
- **PyTorch:** `torch.distributed.all_reduce(tensor, op=...)` — equivalent, except torch's default op is also `SUM` and torch returns the tensor instead of `None`.

### `distributed.all_gather`

- **Signature:** `all_gather(shard, full) -> None`
- **Source:** `src/bindings/bind_distributed.cpp:149`
- **Parameters:** `shard` — this rank's local shard. `full` — output tensor sized to hold every rank's shard, in rank order.
- **Returns:** `None`. `full` is filled in place.
- **Edge cases:** `full` must have room for `world_size` shards; a too-small `full` throws.
- **PyTorch:** `torch.distributed.all_gather(list, tensor)` — torch takes a list of output tensors; LiteTorch takes one pre-sized `full` tensor.

### `distributed.reduce_scatter`

- **Signature:** `reduce_scatter(shard, full) -> None`; `reduce_scatter(shard, full, op) -> None`
- **Source:** `src/bindings/bind_distributed.cpp:150-155`
- **Parameters:** `shard` — output, this rank's slice of the reduced result. `full` — input, the full tensor to reduce and scatter. `op` — a `ReduceOp` (no default on the 3-arg form).
- **Returns:** `None`. `shard` is filled in place.
- **Edge cases:** `full` must split evenly across ranks.
- **PyTorch:** `torch.distributed.reduce_scatter(output, input_list, op=...)` — torch takes a list of inputs; LiteTorch takes one `full` tensor.

### `distributed.broadcast`

- **Signature:** `broadcast(tensor, src_rank) -> None`
- **Source:** `src/bindings/bind_distributed.cpp:156`
- **Parameters:** `tensor` — the tensor to broadcast in place (the `src_rank` rank's values win). `src_rank` — int, the source rank.
- **Returns:** `None`.
- **Edge cases:** every rank must call `broadcast` with the same `src_rank`.
- **PyTorch:** `torch.distributed.broadcast(tensor, src=...)` — equivalent.

### `distributed.all_to_all`

- **Signature:** `all_to_all(input, output) -> None`
- **Source:** `src/bindings/bind_distributed.cpp:157`
- **Parameters:** `input` — tensor holding this rank's chunk for every rank. `output` — tensor receiving every rank's chunk for this rank.
- **Returns:** `None`.
- **PyTorch:** `torch.distributed.all_to_all(output, input)` — same operation, argument order swapped (LiteTorch is `input, output`).

### `distributed.scatter` / `distributed.gather`

- **Signature:** `scatter(output, input, src) -> None`; `gather(input, output, dst) -> None`
- **Source:** `src/bindings/bind_distributed.cpp:158-159`
- **Parameters:** `scatter`: `output` — this rank's received shard; `input` — the full tensor (only meaningful on `src`); `src` — source rank. `gather`: `input` — this rank's shard; `output` — the gathered tensor (only meaningful on `dst`); `dst` — destination rank.
- **Returns:** `None`.
- **PyTorch:** `torch.distributed.scatter(output, src, list)` / `gather(output_list, input, dst)` — same operations, different argument shapes.

### `distributed.send_tensor` / `distributed.recv_tensor`

- **Signature:** `send_tensor(tensor, dst) -> None`; `recv_tensor(tensor, src) -> None`
- **Source:** `src/bindings/bind_distributed.cpp:160-161`
- **Parameters:** `tensor` — the tensor to send / the buffer to receive into. `dst` / `src` — the peer rank.
- **Returns:** `None`. Point-to-point, blocking.
- **Edge cases:** an unmatched `send_tensor` (no peer `recv_tensor`) blocks forever. Shapes must match on both sides.
- **PyTorch:** `torch.distributed.send(tensor, dst)` / `recv(tensor, src)` — equivalent.

<!-- ex: id=r12-collectives device=cpu backend=cpu runtime=~15s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp
import litetorch as lt


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        t = lt.Tensor.from_vector([float(rank + 1)] * 4, [4], cpu)
        lt.distributed.all_reduce(t)
        print("rank", rank, "all_reduce SUM:", t.to_vector())
        b = lt.Tensor.from_vector([0.0] * 4, [4], cpu)
        if rank == 0:
            b = lt.Tensor.from_vector([7.0] * 4, [4], cpu)
        lt.distributed.broadcast(b, 0)
        print("rank", rank, "broadcast:", b.to_vector())
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, 2, 29610)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)



if __name__ == "__main__":
    main()
```

### `distributed.ProcessGroup`

- **Signature:** class, singleton via `ProcessGroup.get()`
- **Source:** `src/bindings/bind_distributed.cpp:22-41`
- **Methods:** `get()` (static, returns the singleton) · `init(rank, world_size, master_addr, master_port)` · `shutdown()` · `all_reduce(tensor)` / `all_reduce(tensor, op)` · `barrier()` · `all_to_all(input, output)` · `scatter(output, input, src)` · `gather(input, output, dst)` · `broadcast(tensor, src)` · `all_gather(shard, full)` · `reduce_scatter(shard, full)` / `reduce_scatter(shard, full, op)` · `send_tensor(tensor, dst)` · `recv_tensor(tensor, src)` · `sync_comm()` · `get_rank()` · `get_world_size()` · `is_initialized()`.
- **Returns:** the methods mirror the module-level functions above, but bound to the group object. `get_rank()` / `get_world_size()` return ints; `is_initialized()` returns bool.
- **Edge cases:** `ProcessGroup.get()` before `init()` returns an uninitialized group; calling a collective on it throws. Prefer the module-level functions unless you need the object.
- **PyTorch:** `torch.distributed.ProcessGroup` — same concept; LiteTorch exposes it as a concrete class, not a backend-selected subclass.

### `distributed.ReduceOp`

- **Signature:** enum with `SUM`, `AVG`, `PRODUCT`, `MIN`, `MAX` (exported as top-level names too)
- **Source:** `src/bindings/bind_distributed.cpp:130-137`
- **Returns:** each value is a reduction operator for `all_reduce` / `reduce_scatter`.
- **Edge cases:** `AVG` divides by `world_size` after summing.
- **PyTorch:** `torch.distributed.ReduceOp` — same values (`SUM`, `AVG`, `PRODUCT`, `MIN`, `MAX`); torch also has `BAND`, `BOR`, `BXOR`, which LiteTorch does not.

### `distributed.init_tp_groups`

- **Signature:** `init_tp_groups(tp_size) -> None`
- **Source:** `src/bindings/bind_distributed.cpp:14-21`
- **Parameters:** `tp_size` — int, the tensor-parallel group size.
- **Returns:** `None`. Splits the initialized process group into TP groups via `NCCLBridge`.
- **Edge cases:** raises `[litetorch Error] init_tp_groups: init_process_group first` if the process group is not initialized. `tp_size` must divide `world_size`.
- **PyTorch:** no direct equivalent; torch builds TP groups with `torch.distributed.new_group()`.

### `distributed.tp_all_reduce` / `distributed.tp_all_gather`

- **Signature:** `tp_all_reduce(tensor) -> Tensor`; `tp_all_gather(shard) -> Tensor`
- **Source:** `src/bindings/bind_distributed.cpp:193-196`
- **Parameters:** `tensor` / `shard` — the tensor to reduce / gather.
- **Returns:** the reduced tensor / the gathered tensor. Unlike the plain collectives, these are autograd-aware: gradients flow through them.
- **Edge cases:** require initialized TP groups (`init_tp_groups` first).
- **PyTorch:** `torch.distributed._functional_collectives` — same idea (differentiable collectives); LiteTorch exposes them as plain functions.

## 3D parallelism

### `distributed.Parallel3D`

- **Signature:** class, singleton via `Parallel3D.get()`
- **Source:** `src/bindings/bind_distributed.cpp:56-74`
- **Methods:** `get()` (static) · `init(rank, world_size, tp_size, pp_size)` · `reset()` · `is_initialized()` · `tp_size()` · `pp_size()` · `dp_size()` · `global_rank()` · `world_size()` · `tp_rank()` · `pp_rank()` · `dp_rank()` · `coords()` · `tp_group()` · `pp_group()` · `dp_group()` · `wire_nccl()`.
- **Returns:** `init` sets up the 3D grid (`dp_size = world_size // (tp_size * pp_size)`); the `*_rank()` methods return this rank's coordinate on each axis; `tp_group()` / `pp_group()` / `dp_group()` return the sub-group objects.
- **Edge cases:** `tp_size * pp_size` must divide `world_size`. `wire_nccl()` wires the NCCL groups; on a CPU-only box it is a no-op.
- **PyTorch:** no single equivalent; torch composes 3D parallelism from `device_mesh` + `new_group()` calls.

### `distributed.init_3d_parallel` / `distributed.shutdown_3d_parallel`

- **Signature:** `init_3d_parallel(tp_size, pp_size=1) -> None`; `shutdown_3d_parallel() -> None`
- **Source:** `src/bindings/bind_distributed.cpp:76-78`
- **Parameters:** `tp_size` — tensor-parallel size; `pp_size` — pipeline-parallel size, default 1.
- **Returns:** `None`. Convenience wrappers around `Parallel3D.get().init(...)` / `reset()`. Require an initialized process group.
- **PyTorch:** no direct equivalent.

<!-- ex: id=r12-3d device=cpu backend=cpu runtime=~15s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp
import litetorch as lt


def worker(rank, world_size, port):
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        lt.distributed.init_3d_parallel(2, 1)
        p = lt.distributed.Parallel3D.get()
        print("rank", rank, "tp", p.tp_rank(), "dp", p.dp_rank(),
              "tp_size", p.tp_size(), "dp_size", p.dp_size())
        lt.distributed.shutdown_3d_parallel()
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, 2, 29612)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)



if __name__ == "__main__":
    main()
```

## Device mesh and DTensor

### `distributed.DeviceMesh`

- **Signature:** `DeviceMesh(shape, dim_names)`
- **Source:** `src/bindings/bind_distributed.cpp:43-53`
- **Parameters:** `shape` — list of ints, e.g. `[2, 2]` for a 2x2 mesh. `dim_names` — string naming the mesh dimensions (e.g. `"dp_tp"`).
- **Attributes (read/write):** `mesh_shape`, `mesh_dim_names`, `mesh_topology`.
- **Methods:** `get_rank(coords)` · `get_coords(rank)` · `get_dim_group(dim, coords)` · `dim_size(dim)` · `num_dims()` · `numel()`.
- **Returns:** `get_rank` maps coordinates to a global rank; `get_coords` is the inverse; `numel()` is the total number of devices in the mesh.
- **Edge cases:** `get_dim_group` returns the sub-group for one mesh dimension; it raises for an out-of-range `dim`.
- **PyTorch:** `torch.distributed.device_mesh.DeviceMesh` — same concept; torch's constructor takes a device type plus shape, LiteTorch takes shape plus a name string.

### `distributed.PlacementType` / `distributed.Placement`

- **Signature:** enum `PlacementType` with `SHARD`, `REPLICATE`, `PARTIAL`; class `Placement()` with read/write `type` and `dim`
- **Source:** `src/bindings/bind_distributed.cpp:81-90`
- **Parameters:** `Placement.type` — a `PlacementType`; `Placement.dim` — the sharded dimension (for `SHARD`).
- **Returns:** a placement describes how a `DTensor` is laid out on a mesh: `SHARD` splits one dimension across the mesh, `REPLICATE` copies the whole tensor to every device, `PARTIAL` holds a partial (unreduced) result.
- **PyTorch:** `torch.distributed._tensor.Placement` (`Shard`, `Replicate`, `Partial`) — same three kinds.

### `distributed.DTensor`

- **Signature:** `DTensor(local_tensor, mesh, placements)`
- **Source:** `src/bindings/bind_distributed.cpp:92-98`
- **Parameters:** `local_tensor` — this rank's local shard. `mesh` — a `DeviceMesh`. `placements` — list of `Placement`, one per mesh dimension.
- **Attributes (read/write):** `device_mesh`, `placements`, `local_tensor`.
- **Methods:** `redistribute(new_mesh, new_placements)` — returns a new `DTensor` with a different layout (may communicate).
- **Returns:** a distributed tensor; it is a `Tensor` subclass (`include/litetorch/dtensor.h: DTensor : public Tensor`).
- **Edge cases:** the number of placements must match the mesh dimensions. `redistribute` to a `REPLICATE` placement triggers an all-gather.
- **PyTorch:** `torch.distributed._tensor.DTensor` — same concept; torch's `redistribute()` takes placements only (same mesh), LiteTorch also takes a new mesh.

<!-- ex: id=r12-mesh device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
mesh = lt.distributed.DeviceMesh([2], "dp")
print("mesh numel:", mesh.numel(), "dims:", mesh.num_dims())
local = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
p = lt.distributed.Placement()
p.type = lt.distributed.PlacementType.SHARD
p.dim = 0
dt = lt.distributed.DTensor(local, mesh, [p])
print("dtensor local shape:", dt.local_tensor.shape)
print("placement:", dt.placements[0].type, "dim", dt.placements[0].dim)
```

## Overlapped collectives

### `distributed.OverlappedAllReducer`

- **Signature:** class, singleton via `OverlappedAllReducer.get()`
- **Source:** `src/bindings/bind_distributed.cpp:100-104`
- **Methods:** `get()` (static) · `push_and_all_reduce(tensor)` · `push_and_reduce_scatter(shard, full)` · `sync()`.
- **Returns:** `push_and_*` enqueue a collective that runs in the background; `sync()` blocks until all enqueued collectives complete.
- **Edge cases:** tensors pushed must stay alive until `sync()`. Do not mutate a pushed tensor before `sync()`.
- **PyTorch:** no direct equivalent; closest is `torch.distributed` async ops (`async_op=True`).

### `distributed.overlapped_all_reduce` / `distributed.overlapped_reduce_scatter` / `distributed.sync_overlapped_all_reduce`

- **Signature:** `overlapped_all_reduce(tensor) -> None`; `overlapped_reduce_scatter(shard, full) -> None`; `sync_overlapped_all_reduce() -> None`
- **Source:** `src/bindings/bind_distributed.cpp:106-108`
- **Parameters:** same tensors as the plain collectives.
- **Returns:** `None`. Module-level wrappers around the singleton `OverlappedAllReducer`.
- **PyTorch:** no direct equivalent.

<!-- ex: id=r12-overlapped device=cpu backend=cpu runtime=~15s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp
import litetorch as lt


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        t = lt.Tensor.from_vector([float(rank + 1)] * 2, [2], cpu)
        lt.distributed.overlapped_all_reduce(t)
        lt.distributed.sync_overlapped_all_reduce()
        print("rank", rank, "overlapped all_reduce:", t.to_vector())
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, 2, 29614)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)



if __name__ == "__main__":
    main()
```

## Pipeline parallelism

### `distributed.PipelineParallelModule`

- **Signature:** `PipelineParallelModule(sub_module)`
- **Source:** `src/bindings/bind_distributed.cpp:110-118`
- **Parameters:** `sub_module` — an `nn.Module` (one pipeline stage).
- **Methods:** `forward(input)` · `forward_microbatches(microbatches)` · `schedule_1f1b(microbatches)` · `train_step_1f1b(microbatches, loss_fn=None)` · `run_1f1b_with_backward(microbatches, loss_fn)` · `parameters()` · `to(device)`.
- **Returns:** `forward` runs the stage; `schedule_1f1b` returns the 1F1B (one-forward-one-backward) schedule; `train_step_1f1b` runs a full training step over micro-batches.
- **Edge cases:** the module must already be placed on the rank's device before wrapping. `loss_fn` defaults to `None` (no loss is computed).
- **PyTorch:** `torch.distributed.pipelining` — same idea (GPipe-style stages); LiteTorch's 1F1B schedule is built in.

### `distributed.InterleavedPipelineModule`

- **Signature:** `InterleavedPipelineModule(chunks)`
- **Source:** `src/bindings/bind_distributed.cpp:120-128`
- **Parameters:** `chunks` — list of `nn.Module`, the interleaved stage chunks on this rank.
- **Methods:** `forward(input)` · `schedule_interleaved_1f1b(microbatches)` · `train_step_interleaved_1f1b(microbatches, loss_fn=None)` · `parameters()` · `to(device)` · `sync()` · `num_chunks()`.
- **Returns:** like `PipelineParallelModule`, but the stage is split into interleaved chunks (virtual pipeline parallelism); `num_chunks()` returns the chunk count.
- **PyTorch:** `torch.distributed.pipelining` with `num_chunks` — same concept.

<!-- ex: id=r12-pipeline device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
stage = lt.nn.Sequential([lt.nn.Linear(4, 4, True), lt.nn.ReLU()])
pp = lt.distributed.PipelineParallelModule(stage)
x = lt.Tensor.from_vector([0.5] * 4, [1, 4], cpu)
print("pipeline forward shape:", pp.forward(x).shape)
chunks = [lt.nn.Linear(4, 4, True), lt.nn.Linear(4, 4, True)]
ip = lt.distributed.InterleavedPipelineModule(chunks)
print("interleaved chunks:", ip.num_chunks())
```

## Data-parallel wrapper

### `distributed.DistributedDataParallel`

- **Signature:** `DistributedDataParallel(module, op=ReduceOp.AVG)`
- **Source:** `src/bindings/bind_distributed.cpp:175-187`
- **Parameters:** `module` — an `nn.Module` to replicate. `op` — a `ReduceOp`, default `AVG` (gradients are averaged across ranks).
- **Attributes (read/write):** `reduce_op`.
- **Methods:** `forward(input)` · `backward(loss)` / `backward(loss, grad_output)` · `sync_grads()` · `parameters()` · `to(device)`.
- **Returns:** `forward` runs the replica; `backward` runs the backward pass and reduces gradients across ranks; `sync_grads()` forces a gradient sync outside `backward`.
- **Edge cases:** the process group must be initialized before construction. Parameters must be identical on every rank before the first `forward` — broadcast them yourself or load the same checkpoint everywhere.
- **Known defect (verified 2026-10-26, extension 88a4ba1, torch 2.14.1+cpu; formula settled 2026-10-27 by dist-w21):** `DDP.backward` doubles one slice of the weight gradient. For a `Linear(in, out)` the flat weight grad is doubled on indices `[numel/4, numel/4 + out)` (equivalently `[numel/4, numel/4 + bias_numel)`) — e.g. `Linear(4, 4)` gives `[0.5 x4, 1.0 x4, 0.5 x4, 0.5 x4]` instead of all `0.5` (torch: all `0.5`). Affects both `AVG` (row becomes 1.0) and `SUM` (row becomes 2.0); the bias grad is correct; single-process `Linear.backward` is correct (all 0.5). The example above prints the full grad so the defect is visible. Do not train with DDP until this is fixed.
- **PyTorch:** `torch.nn.parallel.DistributedDataParallel` — same role; torch averages gradients by default too, but torch's DDP hooks the autograd engine automatically while LiteTorch's `backward()` does the reduction explicitly.

<!-- ex: id=r12-ddp device=cpu backend=cpu runtime=~15s torch-parity=parity-fail status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp
import litetorch as lt


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        model = lt.nn.Sequential([lt.nn.Linear(4, 4, True)])
        model.to(cpu)
        ddp = lt.distributed.DistributedDataParallel(model)
        x = lt.Tensor.from_vector([0.5] * 4, [1, 4], cpu)
        loss = ddp.forward(x).sum()
        ddp.backward(loss)
        print("rank", rank, "ddp grad:", model.parameters()[0].grad.to_vector())
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, 2, 29613)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)



if __name__ == "__main__":
    main()
```

## FSDP

### `distributed.fsdp_fully_shard`

- **Signature:** `fsdp_fully_shard(module) -> None`
- **Source:** `src/bindings/bind_distributed.cpp:255` (`FSDP::fully_shard`, `fsdp_wrapper.cpp:192`)
- **Parameters:** `module` — an `nn.Module` to shard.
- **Returns:** `None`. Mutates the module in place: recurses into leaf modules with parameters, creates one `FSDPUnit` per leaf, and registers forward hooks (gather the full parameters before forward, release after) and gradient hooks (reduce-scatter gradients on backward). After this call, ordinary `model(x)` + `loss.backward()` just works — gather/release is automatic. `fsdp_set_prefetch_limit(n)` caps in-flight gathers (default 2).
- **Edge cases:** the process group must be initialized first. Parameters are flattened via `FlatParameter` (one local shard per rank); gradients are reduce-scattered. FSDP does NOT shard optimizer states — there are zero optimizer references in `fsdp_wrapper.cpp` / `fsdp_tp.cpp`.
- **PyTorch:** `torch.distributed.fsdp.fully_shard` — same entry point and the same in-place contract.

### `distributed.FSDPUnit`

- **Signature:** class with methods `prefetch_gather()` · `ensure_gathered()` · `release_gathered()` · `is_gathered()` — **not callable from Python**
- **Source:** `src/bindings/bind_distributed.cpp:248-252`
- **Returns:** the methods exist on the C++ class, but no `py::init` is bound and `FSDP::active_units()` is not bound, so no Python code can obtain an `FSDPUnit` instance. Constructing one raises `TypeError: litetorch.distributed.FSDPUnit: No constructor defined!`. This is a latent API gap, not a docs problem — use `fsdp_fully_shard()` instead.
- **PyTorch:** no equivalent class is exposed; torch's FSDP units are likewise internal.

### `distributed.fsdp_set_prefetch_limit`

- **Signature:** `fsdp_set_prefetch_limit(n) -> None`
- **Source:** `src/bindings/bind_distributed.cpp:256`
- **Parameters:** `n` — int, the maximum number of in-flight parameter gathers (default 2).
- **Returns:** `None`.
- **PyTorch:** `torch.distributed.fsdp` has no direct equivalent; prefetch depth is controlled by `prefetch_policy`.

### `distributed.FSDPTP`

- **Signature:** class of static methods: `init(tp_size, dp_size)` · `is_initialized()` · `tp_size()` · `dp_size()` · `tp_rank()` · `dp_rank()` · `fully_shard_tp(module)` · `state_dict(module)` · `load_state_dict(module, sd)`
- **Source:** `src/bindings/bind_distributed.cpp:259-268`
- **Returns:** FSDP+TP composition: `init` splits the world into TP groups of `tp_size` and DP groups of `dp_size`; `fully_shard_tp` shards a module across both axes; `state_dict` / `load_state_dict` gather / scatter the full state dict.
- **Edge cases:** `init` must be called before `fully_shard_tp`. Unlike `FSDPUnit`, `FSDPTP` is fully reachable from Python (all statics).
- **PyTorch:** no single equivalent; torch composes FSDP2 with TP via `device_mesh` + `parallelize_module()`.

<!-- ex: id=r12-fsdp device=cpu backend=cpu runtime=~15s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp
import litetorch as lt


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        model = lt.nn.Sequential([
            lt.nn.Linear(8, 8, True),
            lt.nn.GELU(),
            lt.nn.Linear(8, 8, True),
        ])
        model.to(cpu)
        lt.distributed.fsdp_fully_shard(model)
        x = lt.Tensor.from_vector([0.1] * 16, [2, 8], cpu)
        loss = model(x).sum()
        loss.backward()
        print("rank", rank, "fsdp loss:", loss.item())
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, 2, 29611)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)



if __name__ == "__main__":
    main()
```

## NCCL bridge

### `distributed.NCCLBridge`

- **Signature:** class, singleton via `NCCLBridge.get()`
- **Source:** `src/bindings/bind_distributed.cpp:163-173`
- **Methods:** `get()` (static) · `is_available()` · `init(rank, world_size)` · `shutdown()` · `sync_comm()`.
- **Returns:** `is_available()` reports whether a native NCCL backend is usable; `init` wires the NCCL communicator for the group; `sync_comm()` synchronizes the communicator.
- **Edge cases:** on a CPU-only box `is_available()` is False and `init` is a no-op. The NCCL library is found by plain dlopen (see the header note — the `LITETORCH_NCCL_LIB` selection is an unmerged patch); a missing library reports unavailable instead of failing closed at `init`.
- **PyTorch:** `torch.distributed` NCCL backend — same role, but torch selects NCCL via the `backend="nccl"` argument while LiteTorch finds it by plain dlopen (the `LITETORCH_NCCL_LIB` override is an unmerged patch — see the header note).

## Sharded checkpoints

### `distributed.TensorShardInfo` / `distributed.CheckpointMetadata`

- **Signature:** classes with read/write attributes. `TensorShardInfo`: `global_shape`, `dtype_str`, `shard_dim`, `shard_ranges`. `CheckpointMetadata`: `version`, `world_size`, `step`, `timestamp`, `tensors` (a dict of name to `TensorShardInfo`).
- **Source:** `src/bindings/bind_distributed.cpp:200-214`
- **Returns:** plain data holders describing one shard of a distributed checkpoint.
- **PyTorch:** `torch.distributed.checkpoint` metadata — same role, different shape.

### `distributed.make_sharded_spec` / `distributed.make_replicated_spec`

- **Signature:** `make_sharded_spec(global_shape, dtype_str, shard_dim, world_size) -> TensorShardInfo`; `make_replicated_spec(global_shape, dtype_str) -> TensorShardInfo`
- **Source:** `src/bindings/bind_distributed.cpp:215-218`
- **Parameters:** `global_shape` — list of ints, the full tensor shape. `dtype_str` — e.g. `"float32"`. `shard_dim` — the dimension sharded across ranks. `world_size` — rank count.
- **Returns:** a `TensorShardInfo` describing the shard layout.
- **PyTorch:** no direct equivalent; torch infers sharding from the `DTensor` placements.

### `distributed.save_distributed` / `distributed.load_distributed` / `distributed.read_dist_metadata`

- **Signature:** `save_distributed(local_shards, shard_specs, ckpt_dir, step) -> None`; `load_distributed(local_shards, ckpt_dir) -> None`; `read_dist_metadata(ckpt_dir) -> CheckpointMetadata`
- **Source:** `src/bindings/bind_distributed.cpp:219-236`
- **Parameters:** `local_shards` — dict of name to this rank's shard tensor. `shard_specs` — dict of name to `TensorShardInfo`. `ckpt_dir` — directory path. `step` — int checkpoint step.
- **Returns:** `save_distributed` writes one file per rank plus metadata; `load_distributed` fills the given local shards from disk; `read_dist_metadata` returns the stored metadata.
- **Edge cases:** `load_distributed` requires the local shards to already have the right shapes. `ckpt_dir` is created if it does not exist.
- **PyTorch:** `torch.distributed.checkpoint.save` / `load_state_dict` — same role.

<!-- ex: id=r12-checkpoint device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import tempfile
import litetorch as lt

cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [4], cpu)
spec = lt.distributed.make_sharded_spec([4], "float32", 0, 1)
with tempfile.TemporaryDirectory() as d:
    lt.distributed.save_distributed({"w": t}, {"w": spec}, d, 1)
    meta = lt.distributed.read_dist_metadata(d)
    print("meta step:", meta.step, "world:", meta.world_size)
    t2 = lt.Tensor.zeros([4], cpu)
    lt.distributed.load_distributed({"w": t2}, d)
    print("loaded:", t2.to_vector())
```

### `distributed.FlatParameter` / `distributed.make_flat_param`

- **Signature:** `FlatParameter(params)`; `make_flat_param(params) -> FlatParameter`
- **Source:** `src/bindings/bind_distributed.cpp:238-250`
- **Parameters:** `params` — list of tensors to flatten into one buffer.
- **Methods:** `flat_numel()` · `shard_numel()` · `unpadded_numel()` · `world_size()` · `rank()` · `local_shard()` · `full_buffer()` · `param_views()` · `all_gather_flat()` · `reduce_scatter_flat(full_grad)` · `metadata()` (list of `{offset, numel, shape}` dicts).
- **Returns:** a single flat buffer holding all parameters contiguously, with per-parameter views. `local_shard()` is this rank's slice; `full_buffer()` gathers all slices. `param_views()` requires `all_gather_flat()` first — it raises `[litetorch Error] FlatParameter::param_views: full buffer not allocated; call all_gather_flat() first` otherwise. Used internally by FSDP; usable directly for custom sharding.
- **Edge cases:** `world_size()` / `rank()` read from the process group; without an initialized group they report the single-process defaults.
- **PyTorch:** `torch.distributed.fsdp.FlatParameter` — same concept.

<!-- ex: id=r12-flatparam device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
b = lt.Tensor.from_vector([3.0, 4.0, 5.0], [3], cpu)
fp = lt.distributed.make_flat_param([a, b])
fp.all_gather_flat()
print("flat numel:", fp.flat_numel(), "views:", len(fp.param_views()))
print("metadata:", fp.metadata())
```

## Elastic restart

### `distributed.RendezvousStore`

- **Signature:** `RendezvousStore(bind_addr, port)`
- **Source:** `src/bindings/bind_distributed.cpp:272-280`
- **Parameters:** `bind_addr` — address to bind, e.g. `"127.0.0.1"`. `port` — port to listen on.
- **Methods:** `port()` · `run()` (blocks; releases the GIL — run it in a thread) · `stop()`.
- **Returns:** a key-value store that ranks use to discover each other and to publish heartbeats. `port()` returns the bound port (useful when `port=0` asks the OS to pick).
- **Edge cases:** `run()` blocks until `stop()` is called from another thread. Only one store per address/port. Join the thread running `run()` after `stop()` before the interpreter exits — otherwise the process can abort at shutdown with `terminate called without an active exception` (verified: the example aborted without the join, and passed 3/3 with it).
- **PyTorch:** `torch.distributed.elastic.rendezvous` (`etcd` / `c10d` stores) — same role; LiteTorch's store is a single lightweight TCP store.

### `distributed.RendezvousClient`

- **Signature:** `RendezvousClient(addr, port, timeout_sec=30)`
- **Source:** `src/bindings/bind_distributed.cpp:281-290`
- **Parameters:** `addr` — store address. `port` — store port. `timeout_sec` — default 30.
- **Methods:** `set(key, value)` · `get(key, timeout_sec=-1)` · `add(key, delta)` (atomic increment, returns the new value) · `heartbeat(rank)` · `get_heartbeats()`.
- **Returns:** `get` blocks up to `timeout_sec` (-1 waits forever); `add` is atomic across ranks — use it to assign ranks.
- **PyTorch:** `torch.distributed.elastic.rendezvous` store client — same operations.

### `distributed.rendezvous_free_port`

- **Signature:** `rendezvous_free_port() -> int`
- **Source:** `src/bindings/bind_distributed.cpp:291`
- **Returns:** a currently-free local port number, for `init_process_group` or a `RendezvousStore`.
- **Edge cases:** the port can be taken between the call and the bind (TOCTOU) — retry on bind failure.
- **PyTorch:** no direct equivalent; torch users bind port 0 or use `torch.distributed.elastic.utils`.

<!-- ex: id=r12-rendezvous device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import threading
import time
import litetorch as lt

port = lt.distributed.rendezvous_free_port()
store = lt.distributed.RendezvousStore("127.0.0.1", port)
t = threading.Thread(target=store.run, daemon=True)
t.start()
time.sleep(0.5)
client = lt.distributed.RendezvousClient("127.0.0.1", port)
client.set("world_size", "2")
print("world_size:", client.get("world_size"))
print("rank counter:", client.add("next_rank", 1))
client.heartbeat(0)
print("heartbeats:", client.get_heartbeats())
store.stop()
t.join(timeout=2.0)
print("store thread joined:", not t.is_alive())
```

### `distributed.HeartbeatAgent`

- **Signature:** `HeartbeatAgent(store_addr, store_port, rank, interval_ms=1000)`
- **Source:** `src/bindings/bind_distributed.cpp:293-300`
- **Parameters:** `store_addr` / `store_port` — the rendezvous store. `rank` — this rank. `interval_ms` — heartbeat interval, default 1000.
- **Methods:** `start()` · `stop()` · `running()`.
- **Returns:** a background agent that heartbeats the store every `interval_ms` until `stop()`. `running()` reports whether the agent thread is alive.
- **PyTorch:** `torch.distributed.elastic.agent` heartbeat — same role.

### `distributed.detect_dead_ranks`

- **Signature:** `detect_dead_ranks(store_addr, store_port, timeout_ms) -> list[int]`
- **Source:** `src/bindings/bind_distributed.cpp:301-302`
- **Parameters:** `store_addr` / `store_port` — the rendezvous store. `timeout_ms` — a rank is dead if its last heartbeat is older than this.
- **Returns:** the list of dead ranks.
- **PyTorch:** `torch.distributed.elastic` failure detection — same role.

### `distributed.save_consistent` / `distributed.latest_consistent_step` / `distributed.load_latest`

- **Signature:** `save_consistent(local_shards, shard_specs, ckpt_dir, step) -> None`; `latest_consistent_step(ckpt_dir) -> int`; `load_latest(local_shards, ckpt_dir) -> None`
- **Source:** `src/bindings/bind_distributed.cpp:303-316`
- **Parameters:** same shard arguments as `save_distributed`; `step` — int checkpoint step.
- **Returns:** `save_consistent` writes a checkpoint only when every rank has reported in (so a restart never loads a half-written checkpoint); `latest_consistent_step` returns the newest consistent step, or -1 if none; `load_latest` loads it.
- **Edge cases:** use these (not `save_distributed`) for checkpoints that must survive a mid-write crash.
- **PyTorch:** `torch.distributed.checkpoint` with atomic commit — same guarantee, different API.
