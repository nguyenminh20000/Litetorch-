# E3 — Memory Eviction

Memory eviction (giải phóng bộ nhớ) is LiteTorch's differentiator: when GPU memory runs short, the framework moves idle GPU tensors back to CPU memory instead of failing, and moves them back on demand. This page describes the mechanism exactly as implemented in the tree at HEAD `b326f29` (extension build `88a4ba1`). Every behavioral claim cites its source file. Nothing here is inferred from PyTorch: PyTorch has no equivalent automatic GPU-to-CPU eviction; `torch.cuda` caching-allocator behavior is not the same mechanism.

## 1. The pieces

Three components cooperate:

```text
Tensor (StorageImpl)
    │  holds gpu_data (device buffer) and cpu_data (host copy)
    ▼
MemoryManager (singleton, LRU list of GPU storages)
    │  decides what to evict, tracks gpu_used_ against gpu_limit_
    ▼
StorageImpl::evict_impl() / swap_in_impl()
    │  the actual move: GPU -> CPU (evict) and CPU -> GPU (swap-in)
    ▼
rt_gpu_read / rt_gpu_write / rt_gpu_allocate   (include/litetorch/backend_route.h)
    │  route to the native GPU backend when present, else OpenCL
```

Source files: `include/litetorch/memory_manager.h`, `src/memory_manager/memory_manager.cpp`, `src/tensor/storage.cpp`, `include/litetorch/backend_route.h`.

## 2. The limit

`MemoryManager` holds a single budget, `gpu_limit_`, defaulting to **512 MiB** (`include/litetorch/memory_manager.h:53`: `gpu_limit_{512 * 1024 * 1024}`). `gpu_used_` counts the bytes of all currently registered GPU storages.

Two facts matter and both are verified against the tree and the built extension:

- The limit is set from C++ via `MemoryManager::get().set_gpu_limit(bytes)` (`src/memory_manager/memory_manager.cpp:21-25`). Lowering the limit evicts immediately: `set_gpu_limit` calls `evict_to_free_impl(0)`, which frees until `gpu_used_` fits the new limit.
- **There is no Python binding for `set_gpu_limit` at this commit.** Verified live: `hasattr(lt, 'set_gpu_limit')` and `hasattr(lt.cuda, 'set_gpu_limit')` are both `False` on build `88a4ba1`. From Python the 512 MiB default cannot be changed; eviction is automatic and only observable through the counters in §6.

## 3. When eviction triggers

Eviction is lazy and demand-driven. It runs in exactly three situations, all inside `MemoryManager`:

1. **A new GPU storage registers.** `register_gpu_impl` calls `evict_to_free_impl(size_bytes)` *before* inserting the new storage (`src/memory_manager/memory_manager.cpp:64-70`). A large allocation evicts enough older tensors to fit under the limit first.
2. **The limit is lowered.** `set_gpu_limit` evicts to the new limit immediately (see §2).
3. **A swapped-out tensor swaps back in.** `swap_in_impl` re-registers the storage (`src/tensor/storage.cpp:256-262`), and registration evicts as in (1). Bringing one tensor back can push another one out.

Eviction never runs on a timer, never runs in the background, and never runs for CPU tensors. CPU allocations go through `CachingAllocator::allocate_cpu` with no eviction step (`src/tensor/storage.cpp:55-57`).

## 4. The eviction policy: LRU, skipping tensors in use

`evict_to_free_impl` (`src/memory_manager/memory_manager.cpp:96-117`) walks the LRU list from the **oldest** end:

```text
while gpu_used_ + required_bytes > gpu_limit_ and list not empty:
    take oldest storage
    if storage is NOT in use AND NOT already swapped out:
        evict it (GPU -> CPU copy, free GPU buffer)
        evicted += its bytes
    else:
        skip it, try the next one
```

- **LRU order.** Every use refreshes a storage's position: `touch_impl` moves it to the back of the list (`memory_manager.cpp:88-95`), and `ensure_gpu` touches a storage that is already resident (`memory_manager.cpp:44-52`). The least-recently-used tensor is evicted first.
- **In-use tensors are never evicted.** A storage with `in_use` set is skipped. This is the safety rule that keeps a kernel from losing its input mid-launch.
- **Already-swapped tensors are skipped.** A storage with `is_swapped` set holds no GPU buffer, so there is nothing to free.
- **Eviction is best-effort, not a guarantee.** If every storage is in use (or already swapped), the loop frees nothing and returns `0`. The caller then attempts its allocation anyway. For a fresh GPU allocation that failure is a hard error: the `StorageImpl` constructor throws `GPU out of memory` with a diagnostics string (`src/tensor/storage.cpp:30-36`). Eviction reduces OOMs; it does not abolish them.

The outcome of the last eviction pass is recorded in `last_evicted_bytes_` and `last_eviction_required_` (`memory_manager.cpp:114-115`), which the OOM diagnostics string reports as `last_eviction_freed=` / `last_eviction_required=` (`memory_manager.cpp:147-148`).

## 5. What eviction does to a tensor

`StorageImpl::evict_impl` (`src/tensor/storage.cpp:222-234`):

1. Allocates a CPU buffer if the tensor has none.
2. Copies the device buffer to CPU (`rt_gpu_read`).
3. Frees the GPU buffer immediately (`rt_gpu_free_now`).
4. Sets `gpu_data = nullptr`, `is_swapped = true`.
5. Unregisters the storage from the `MemoryManager`, which subtracts its bytes from `gpu_used_`.

The tensor object stays alive and usable. Its data now lives only in CPU memory; the `Tensor` still reports its original device. Eviction is transparent to user code except for speed: the next access pays a swap-in.

**Swap-in (recompute by reload, not by re-execution).** When a swapped-out tensor is needed on the device again, `get_gpu_ptr` calls `swap_in_impl` (`src/tensor/storage.cpp:114-133, 244-271`):

1. Re-registers with the `MemoryManager` (which may evict other tensors first).
2. Allocates a fresh GPU buffer; allocation failure throws `GPU out of memory: failed to swap in ...` — swap-in can itself OOM.
3. Copies CPU → GPU (`rt_gpu_write`), frees the CPU copy, clears `is_swapped`.

Note the asymmetry: eviction keeps the CPU copy, swap-in frees it. A tensor that is evicted and swapped back ends in the same state as before eviction, with no CPU copy retained.

One edge case is worth knowing: if swap-in finds no GPU at all (the backend disappeared), it does not throw — it silently converts the storage to CPU (`storage.cpp:248-253`). This is the same silent-fallback family as D-02 in the Behavioral Differences page.

## 6. Observability from Python

On build `88a4ba1` the Python-visible surface is exactly two counters:

| Call | Returns | Source |
|---|---|---|
| `lt.cuda.memory_allocated()` | `MemoryManager::get().get_gpu_used()` | `src/bindings/bind_cuda.cpp:39` |
| `lt.get_gpu_memory_used()` | same value | `src/bindings/bind_cuda.cpp:57` |

Because eviction unregisters the storage, `memory_allocated()` **drops** when eviction frees tensors. On a CPU-only machine both return `0`: no GPU storage ever registers, so eviction never runs (Example E3-2). The C++ diagnostics string (`gpu_oom_diagnostics`, `memory_manager.cpp:129-150`) reports `gpu_used`, `gpu_limit`, `allocator_cached`, `lru_storages`, `active_storages`, and the last eviction figures, but it is not bound to Python at this commit.

## 7. Interaction with checkpointing

Eviction and activation checkpointing attack the same problem — activations do not fit in GPU memory — from opposite directions, and they are independent mechanisms:

| | Eviction | Checkpointing (`litetorch.checkpoint`) |
|---|---|---|
| What it saves | GPU bytes, by moving tensors to CPU | GPU bytes, by not storing intermediates at all |
| What it costs | PCIe/NVLink copy on swap-in | Re-running the forward pass in backward |
| When it acts | At allocation time, automatically | Only where the user wraps a function |
| Needs the user to do anything | No | Yes: wrap the segment in `checkpoint` |

Checkpointing is recompute in the strict sense. `litetorch.checkpoint(function, input)` (`src/autograd/checkpoint.cpp:60-77`) runs `function(input)` under `NoGradGuard` in the forward pass, saves only a copy of the input plus the RNG state, and re-runs the function inside the backward pass (`CheckpointNode::backward`, `checkpoint.cpp:26-58`), restoring the RNG state first so stochastic layers reproduce the same forward values. If grad is disabled or the input does not require grad, `checkpoint` simply calls the function directly — no node is created (`checkpoint.cpp:61-63`).

`Ops.checkpoint(module, input)` (`src/ops/checkpoint.cpp`, bound at `src/bindings/bind_ops.cpp:161`) is the module form: it saves the module's inputs and parameters and re-runs `module->forward` in backward.

The two compose without coordination: a checkpointed segment's saved input can itself be evicted to CPU while waiting for backward, and its recomputed intermediates are ordinary tensors subject to the same LRU policy. Neither mechanism knows about the other, and neither is required for the other to work. The practical rule is: checkpoint the segments whose recompute is cheap, and let eviction absorb the tensors you did not checkpoint.

## 8. Limits: what eviction is not

- **Not a CPU mechanism.** CPU tensors are never evicted; host memory pressure is not managed by `MemoryManager`.
- **Not a TPU mechanism.** TPU storages are allocated through the TPU backend directly (`src/tensor/storage.cpp:37-50`) and are never registered with the `MemoryManager`. A TPU allocation failure silently falls back to CPU instead of evicting (`storage.cpp:43-48`).
- **Not the OpenCL cache.** The OpenCL backend has its own, separate cache eviction: `CLBackend::evict_cache_to_limit` releases whole free super-blocks while `total_device_allocated_` exceeds the same `gpu_limit_` (`src/cl_backend/cl_backend.cpp:489-501`). It frees cached allocator blocks, not live tensors. Do not confuse the two: one evicts tensors, the other evicts free cache blocks.
- **Not free.** Evicted data crosses the bus twice (out on evict, back on swap-in). A workload that thrashes — evicting tensors it immediately needs again — can run slower than one that simply OOMs early and forces the user to shrink the model. If `memory_allocated()` oscillates and throughput collapses, the fix is a smaller activation footprint (checkpointing, smaller batch), not more eviction.
- **Not a substitute for the limit.** The 512 MiB default is small for modern models and cannot be raised from Python at this commit. Any workload that needs a larger GPU budget must set it from C++ or accept eviction churn.

## Examples

<!-- ex: id=e3-1 device=cpu backend=cpu runtime=~90s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import torch
import litetorch as lt
lt.manual_seed(42)
torch.manual_seed(42)
x = lt.randn([2, 4])
x.requires_grad = True
w = lt.randn([4, 3])
w.requires_grad = True
def fn(t):
    return lt.Ops.relu(lt.Ops.matmul(t, w))
y = lt.checkpoint(fn, x)
y.sum().backward()
xt = torch.tensor(x.to_vector()).reshape(2, 4).requires_grad_(True)
wt = torch.tensor(w.to_vector()).reshape(4, 3).requires_grad_(True)
yt = torch.utils.checkpoint.checkpoint(lambda t: torch.relu(t @ wt), xt, use_reentrant=False)
yt.sum().backward()
print("fwd parity", torch.allclose(torch.tensor(y.to_vector()), yt.detach().flatten(), atol=1e-4, rtol=1e-4))
print("grad x parity", torch.allclose(torch.tensor(x.grad.to_vector()), xt.grad.flatten(), atol=1e-4, rtol=1e-4))
print("grad w parity", torch.allclose(torch.tensor(w.grad.to_vector()), wt.grad.flatten(), atol=1e-4, rtol=1e-4))
print("fwd values", y.to_vector()[:3])
print("grad x values", x.grad.to_vector()[:3])
```
Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4. The checkpointed segment's forward values and both gradients match `torch.utils.checkpoint` exactly at 1e-4: recompute reproduces the forward pass.

<!-- ex: id=e3-2 device=cpu backend=cpu runtime=~55s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import torch
import litetorch as lt
print("cuda available", lt.cuda.is_available())
print("device count", lt.cuda.device_count())
print("auto device", lt.auto_device())
print("backend", lt.get_backend_name())
print("memory allocated", lt.cuda.memory_allocated())
print("gpu memory used", lt.get_gpu_memory_used())
t = lt.randn([4, 4])
moved = t.to(lt.Device("cuda"))
print("requested cuda, got", moved.device)
print("torch cuda available", torch.cuda.is_available())
```
Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. On a CPU-only machine both memory counters read `0` because no GPU storage ever registers, so eviction never runs. The `.to(Device("cuda"))` line also demonstrates D-02: the tensor silently stays on CPU.

<!-- ex: id=e3-3 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
lt.manual_seed(42)
print("cuda available", lt.cuda.is_available())
print("device count", lt.cuda.device_count())
print("memory before", lt.cuda.memory_allocated())
held = []
for i in range(6):
    t = lt.randn([1024, 1024]).to(lt.Device("cuda"))
    held.append(t)
    print("after alloc", i, "memory allocated", lt.cuda.memory_allocated())
print("survived allocations", len(held))
```
**REQUIRES-GPU — not verified.** This example needs a real GPU: six 4 MiB FP32 tensors against the 512 MiB default limit exercise registration, and larger allocations would force the LRU pass of §4. On this CPU-only machine it cannot run — `.to(Device("cuda"))` silently falls back to CPU (D-02) and `memory_allocated()` stays `0`. There is no `set_gpu_limit` in Python, so the example observes the default limit only; lowering the limit to force eviction on demand requires C++.

<!-- ex: id=e3-4 device=cpu backend=cpu runtime=~100s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import torch
import litetorch as lt
lt.manual_seed(42)
torch.manual_seed(42)
x = lt.randn([2, 4])
x.requires_grad = True
w1 = lt.randn([4, 8])
w1.requires_grad = True
w2 = lt.randn([8, 3])
w2.requires_grad = True
def segment(t):
    return lt.Ops.matmul(lt.Ops.relu(lt.Ops.matmul(t, w1)), w2)
y = lt.checkpoint(segment, x)
y.sum().backward()
xt = torch.tensor(x.to_vector()).reshape(2, 4).requires_grad_(True)
w1t = torch.tensor(w1.to_vector()).reshape(4, 8).requires_grad_(True)
w2t = torch.tensor(w2.to_vector()).reshape(8, 3).requires_grad_(True)
yt = torch.utils.checkpoint.checkpoint(lambda t: (torch.relu(t @ w1t) @ w2t), xt, use_reentrant=False)
yt.sum().backward()
print("fwd parity", torch.allclose(torch.tensor(y.to_vector()), yt.detach().flatten(), atol=1e-4, rtol=1e-4))
print("grad x parity", torch.allclose(torch.tensor(x.grad.to_vector()), xt.grad.flatten(), atol=1e-4, rtol=1e-4))
print("grad w1 parity", torch.allclose(torch.tensor(w1.grad.to_vector()), w1t.grad.flatten(), atol=1e-4, rtol=1e-4))
print("grad w2 parity", torch.allclose(torch.tensor(w2.grad.to_vector()), w2t.grad.flatten(), atol=1e-4, rtol=1e-4))
```
Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4. A two-layer checkpointed segment recomputes both layers in backward and still matches torch gradients at 1e-4.
