# Tensor — API Reference

`litetorch.Tensor` is the central tensor class: an n-dimensional array backed by a `StorageImpl`, with autograd support. All compute is C++14; Python is the pybind11 interface layer.

- **Source of truth:** `src/bindings/bind_tpu.cpp:31-236` (Python bindings), `include/litetorch/tensor.h` (C++ class).
- **Tree:** agent-v1@02cc271a / local HEAD 88a4ba1 (docs baseline; B5 re-verifies at head).
- **Conventions:** `cpu = lt.Device("cpu")` in every example. Every example is self-contained and runnable on CPU. Torch parity is FP32 1e-4 unless the header says otherwise.
- **Related deviations:** D-01 (`data_ptr()`), D-02 (`to(device)` silent CPU fallback), D-03 (no string form for `.to()`), D-05 (no `.pow()` method), D-06 (`.long()` — FIXED, now INT64; see `plan/differences.md`), D-11 (indexing), D-12 (single-int dim), D-13 (missing cast methods), D-14 (`.numpy()` dtype limits), D-15 (`arange` integer-only), D-18 (`.item()`/`.tolist()` precision). See `plan/differences.md`.

## Static factories

### `Tensor.create`

- **Signature:** `Tensor.create(shape, device=Device(CPU, 0), requires_grad=False, dtype=DataType.FP32) -> Tensor`
- **Source:** `src/bindings/bind_tpu.cpp:32`
- **Parameters:** `shape` — list/tuple of ints, e.g. `[2, 3]`. `device` — a `litetorch.Device`. `requires_grad` — track for autograd. `dtype` — a `litetorch.DataType`, default FP32.
- **Returns:** uninitialized tensor of the given shape (contents undefined; do not read before writing).
- **Edge cases:** a shape containing 0 creates an empty tensor with `numel() == 0`.
- **PyTorch:** `torch.empty(shape, device=..., dtype=...)` — equivalent, except LiteTorch takes a single shape sequence, never varargs.

<!-- ex: id=r1-create device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.create([2, 3], cpu)
print(t.shape, t.dtype)
```

### `Tensor.from_vector`

- **Signature:** `Tensor.from_vector(data, shape, device=Device(CPU, 0), requires_grad=False, dtype=DataType.FP32) -> Tensor`
- **Source:** `src/bindings/bind_tpu.cpp:33`
- **Parameters:** `data` — flat Python list of numbers, length must equal `prod(shape)`. `shape` — list/tuple of ints. `dtype` — default FP32; values pass through `float` first (see D-18: int64 values above 2^24 lose precision through this path).
- **Returns:** tensor filled with `data` in row-major order.
- **Edge cases:** length mismatch raises. Nested lists are not accepted here — use `lt.tensor()` for nested sequences.
- **PyTorch:** `torch.tensor(data, dtype=...).reshape(shape)` — equivalent for FP32 data.

<!-- ex: id=r1-from-vector device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
print(t.to_vector())
```

### `Tensor.zeros`

- **Signature:** `Tensor.zeros(shape, device=Device(CPU, 0), requires_grad=False) -> Tensor`; overload `Tensor.zeros(shape, device, requires_grad=False, dtype=DataType.FP32) -> Tensor`
- **Source:** `src/bindings/bind_tpu.cpp:34-35`
- **Parameters:** `shape` — list/tuple of ints. The first overload always returns FP32; the second takes an explicit `dtype` (note: `device` has no default in the second overload — pass it explicitly).
- **Returns:** all-zero tensor.
- **PyTorch:** `torch.zeros(shape, dtype=...)` — equivalent.

<!-- ex: id=r1-zeros device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.zeros([2, 2], cpu)
b = lt.Tensor.zeros([2, 2], cpu, False, lt.DataType.INT32)
print(a.to_vector(), b.dtype)
```

### `Tensor.ones`

- **Signature:** `Tensor.ones(shape, device=Device(CPU, 0), requires_grad=False, dtype=DataType.FP32) -> Tensor`
- **Source:** `src/bindings/bind_tpu.cpp:36`
- **Returns:** all-one tensor.
- **PyTorch:** `torch.ones(shape, dtype=...)` — equivalent.

<!-- ex: id=r1-ones device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.ones([3], cpu)
print(t.to_vector())
```

### `Tensor.randn`

- **Signature:** `Tensor.randn(shape, device=Device(CPU, 0), requires_grad=False, dtype=DataType.FP32) -> Tensor`
- **Source:** `src/bindings/bind_tpu.cpp:37`
- **Parameters:** `shape` — single shape sequence only; `randn(2, 3)` raises (D-04).
- **Returns:** standard-normal samples. Seeding: `lt.manual_seed(seed)`; the generator is process-global.
- **PyTorch:** `torch.randn(shape)` — same distribution; same seed does not imply the same stream.

<!-- ex: id=r1-randn device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(42)
t = lt.Tensor.randn([2, 2], cpu)
print(t.shape, t.dtype)
```

### `Tensor.arange`

- **Signature:** `Tensor.arange(start, end, step=1, device=Device(CPU, 0), dtype=DataType.FP32) -> Tensor`
- **Source:** `src/bindings/bind_tpu.cpp:38`
- **Parameters:** `start`, `end`, `step` — integers only (D-15); values are materialized as FP32 then cast to `dtype`.
- **Returns:** 1-D tensor `[start, start+step, ...]` below `end`.
- **Edge cases:** `step == 0` is invalid; a float step raises.
- **PyTorch:** `torch.arange(start, end, step)` — differs: torch accepts float start/end/step.

<!-- ex: id=r1-arange device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.arange(0, 5, 2, cpu)
print(t.to_vector())
```

## Attributes

All attributes are read/write (`def_readwrite`, `src/bindings/bind_tpu.cpp:58-64`). Assigning `t.shape` directly does not re-layout storage — prefer `view`/`reshape` for shape changes.

### `Tensor.shape`

- **Type:** list of ints. **Source:** `bind_tpu.cpp:58`.
- **PyTorch:** `t.shape` / `t.size()` — equivalent (torch returns a `torch.Size`, LiteTorch a plain list).

### `Tensor.strides`

- **Type:** list of ints (elements per step along each dim). **Source:** `bind_tpu.cpp:59`.
- **PyTorch:** `t.stride()` — equivalent.

### `Tensor.dtype`

- **Type:** `litetorch.DataType`. **Source:** `bind_tpu.cpp:60`.
- **PyTorch:** `t.dtype` — equivalent; note LiteTorch has extra low-precision values (FP8_E4M3, FP8_E5M2, NF4, FP4_E2M1, INT4) with no torch counterpart.

### `Tensor.device`

- **Type:** `litetorch.Device`. **Source:** `bind_tpu.cpp:61`.
- **PyTorch:** `t.device` — equivalent.

### `Tensor.requires_grad`

- **Type:** bool. **Source:** `bind_tpu.cpp:62`.
- **PyTorch:** `t.requires_grad` — equivalent.

### `Tensor.grad`

- **Type:** `Tensor` or `None`; the accumulated gradient after `.backward()`. **Source:** `bind_tpu.cpp:63`.
- **PyTorch:** `t.grad` — equivalent. Non-leaf tensors have `grad is None` unless `retain_grad()` was called.

### `Tensor.shard`

- **Type:** shard metadata for distributed tensors. **Source:** `bind_tpu.cpp:64`.
- **PyTorch:** no direct equivalent (see `torch.distributed.tensor.DTensor`).

<!-- ex: id=r1-attrs device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu, True)
print(t.shape, t.strides, t.dtype, t.device, t.requires_grad, t.grad)
print(t.numel(), t.dim(), t.ndim)
```

## Inspection

### `Tensor.numel`

- **Signature:** `numel() -> int`
- **Source:** `bind_tpu.cpp:65`
- **Returns:** total element count (`prod(shape)`; 0 for empty tensors).
- **PyTorch:** `t.numel()` — equivalent.

### `Tensor.dim` / `Tensor.ndim`

- **Signature:** `dim() -> int`; `ndim` is a read-only property returning the same value.
- **Source:** `bind_tpu.cpp:150-151`
- **PyTorch:** `t.dim()` / `t.ndim` — equivalent.

### `Tensor.item`

- **Signature:** `item() -> float`
- **Source:** `bind_tpu.cpp:66`
- **Returns:** the single element as a Python `float`; non-FP32 tensors are cast to FP32 first (D-18: int64 values above 2^53 lose precision).
- **Edge cases:** raises on tensors with `numel() != 1`.
- **PyTorch:** `t.item()` — differs: torch returns a Python `int` with full precision for int64 tensors.

<!-- ex: id=r1-item device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([3.5], [1], cpu)
print(t.item(), t.numel(), t.dim())
```

### `Tensor.to_vector`

- **Signature:** `to_vector() -> list[float]`
- **Source:** `bind_tpu.cpp:67`
- **Returns:** flat row-major list of floats; non-FP32 tensors are cast to FP32 first (D-18).
- **PyTorch:** `t.flatten().tolist()` — equivalent for FP32; differs on int64 precision (see D-18).

### `Tensor.tolist`

- **Signature:** `tolist() -> list`
- **Source:** `bind_tpu.cpp:202`
- **Returns:** nested Python lists mirroring the shape. INT32/INT16/INT8 elements come back as Python `int`; all other dtypes as `float` (D-18: int64 values above 2^53 lose precision). Non-CPU tensors are moved to CPU first.
- **PyTorch:** `t.tolist()` — equivalent for FP32/INT32; differs on int64 precision.

<!-- ex: id=r1-tolist device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
print(t.to_vector(), t.tolist())
```

### `Tensor.numpy`

- **Signature:** `numpy() -> numpy.ndarray`
- **Source:** `bind_tpu.cpp:158`
- **Returns:** a numpy array sharing (FP32) or copying (other dtypes) the data.
- **Edge cases:** non-CPU tensors throw — call `.cpu()` first. Supported dtypes: FP64, FP32, FP16, INT32, INT16, INT8. BF16 is cast to FP32. UINT8, INT64, BOOL throw `unsupported dtype for copy conversion` (D-14).
- **PyTorch:** `t.numpy()` — differs: torch supports uint8 and int64 directly.

<!-- ex: id=r1-numpy device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
a = t.numpy()
print(a.dtype, a.tolist())
```

### `Tensor.__repr__`

- **Signature:** `repr(t) -> str`
- **Source:** `bind_tpu.cpp:236`
- **Returns:** `"<litetorch.Tensor shape=<numel> device=<device>>"` — note it prints the element count, not the shape tuple.
- **PyTorch:** `repr(t)` prints values; LiteTorch prints only a summary.

## Indexing and selection

### `Tensor.__getitem__`

- **Signature:** `t[int] -> Tensor`; `t[slice] -> Tensor`
- **Source:** `bind_tpu.cpp:42`
- **Behavior:** int indexes dim 0 and squeezes it; slice selects a range along dim 0.
- **Edge cases:** boolean masks, integer-tensor indices, and multi-dim tuples raise `TypeError("unsupported index type")` (D-11). Use `Ops.gather`, `index_select`, or `slice` instead.
- **PyTorch:** `t[i]`, `t[a:b]`, `t[mask]`, `t[:, 1]` — LiteTorch supports only the first two forms.

<!-- ex: id=r1-getitem device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([10.0, 20.0, 30.0], [3], cpu)
print(t[1].to_vector(), t[0:2].to_vector())
```

### `Tensor.slice`

- **Signature:** `slice(dim, start, end, step=1) -> Tensor`
- **Source:** `bind_tpu.cpp:41`; C++ `include/litetorch/tensor.h:139`
- **Returns:** a view-like slice along `dim` (shares storage).
- **PyTorch:** `t.narrow(dim, start, end - start)` / slicing — equivalent.

### `Tensor.index_select`

- **Signature:** `index_select(dim, indices) -> Tensor`
- **Source:** `bind_tpu.cpp:40`; C++ `tensor.h:138`
- **Parameters:** `indices` — an integer tensor of positions along `dim`.
- **Returns:** gathered tensor; shape is `shape` with `shape[dim]` replaced by `len(indices)`.
- **PyTorch:** `torch.index_select(t, dim, indices)` — equivalent.

<!-- ex: id=r1-index-select device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([10.0, 20.0, 30.0, 40.0], [2, 2], cpu)
idx = lt.Tensor.from_vector([1.0, 0.0], [2], cpu, False, lt.DataType.INT32)
print(t.index_select(0, idx).to_vector())
print(t.slice(0, 0, 1).to_vector())
```

### `Tensor.masked_fill`

- **Signature:** `masked_fill(mask, value) -> Tensor`
- **Source:** `bind_tpu.cpp:39`; C++ `tensor.h:137`
- **Parameters:** `mask` — a boolean tensor broadcastable to `self`; `value` — a float.
- **Returns:** a new tensor with `value` wherever `mask` is true.
- **PyTorch:** `t.masked_fill(mask, value)` — equivalent.

<!-- ex: id=r1-masked-fill device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
m = lt.Tensor.from_vector([1.0, 0.0, 0.0, 1.0], [2, 2], cpu, False, lt.DataType.BOOL)
print(t.masked_fill(m, -1.0).to_vector())
```

### `Tensor.gather`

- **Signature:** `gather(dim, index) -> Tensor`
- **Source:** `bind_tpu.cpp:93`; C++ `tensor.h:169`
- **Parameters:** `index` — integer tensor; output shape equals `index` shape.
- **PyTorch:** `torch.gather(t, dim, index)` — equivalent.

<!-- ex: id=r1-gather device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([10.0, 20.0, 30.0, 40.0], [2, 2], cpu)
idx = lt.Tensor.from_vector([1.0, 0.0], [1, 2], cpu, False, lt.DataType.INT32)
print(t.gather(1, idx).to_vector())
```

## Shape manipulation

### `Tensor.view`

- **Signature:** `view(shape) -> Tensor` or `view(*dims) -> Tensor`
- **Source:** `bind_tpu.cpp:76`
- **Parameters:** a single list/tuple, or variadic ints. `-1` infers one dimension.
- **Returns:** a view sharing storage; raises if the tensor is non-contiguous or the size mismatches.
- **PyTorch:** `t.view(...)` — equivalent.

### `Tensor.reshape`

- **Signature:** `reshape(shape) -> Tensor`
- **Source:** `bind_tpu.cpp:88`
- **Returns:** a view when possible, otherwise a copy.
- **PyTorch:** `t.reshape(shape)` — equivalent.

<!-- ex: id=r1-view-reshape device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], cpu)
print(t.view([3, 2]).shape, t.view(6).shape, t.reshape([3, 2]).to_vector())
```

### `Tensor.as_strided`

- **Signature:** `as_strided(shape, strides) -> Tensor`
- **Source:** `bind_tpu.cpp:89`; C++ `tensor.h:163`
- **Returns:** a view with the given shape and element strides (no bounds check — out-of-range strides read out-of-bounds memory).
- **PyTorch:** `t.as_strided(shape, strides)` — equivalent.

### `Tensor.permute`

- **Signature:** `permute(dims) -> Tensor`
- **Source:** `bind_tpu.cpp:90`; C++ `tensor.h:166`
- **Parameters:** `dims` — a permutation of `range(ndim)`.
- **Returns:** a view with reordered dimensions.
- **PyTorch:** `t.permute(dims)` — equivalent.

### `Tensor.transpose`

- **Signature:** `transpose(dim0, dim1) -> Tensor`
- **Source:** `bind_tpu.cpp:135`; C++ `tensor.h:165`
- **Returns:** a view with two dimensions swapped.
- **PyTorch:** `t.transpose(dim0, dim1)` — equivalent.

<!-- ex: id=r1-permute-transpose device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], cpu)
print(t.permute([1, 0]).shape, t.transpose(0, 1).to_vector())
print(t.as_strided([3, 2], [1, 3]).to_vector())
```

### `Tensor.expand`

- **Signature:** `expand(shape) -> Tensor`
- **Source:** `bind_tpu.cpp:92`; C++ `tensor.h:168`
- **Returns:** a view broadcasting size-1 dimensions to `shape`; expanded dimensions have stride 0.
- **PyTorch:** `t.expand(shape)` — equivalent.

### `Tensor.detach`

- **Signature:** `detach() -> Tensor`
- **Source:** `bind_tpu.cpp:91`
- **Returns:** a tensor sharing storage but detached from the autograd graph.
- **PyTorch:** `t.detach()` — equivalent.

### `Tensor.clone`

- **Signature:** `clone() -> Tensor`
- **Source:** `bind_tpu.cpp:138`; C++ `tensor.h:159`
- **Returns:** a deep copy (new storage); autograd history is preserved.
- **PyTorch:** `t.clone()` — equivalent.

### `Tensor.contiguous` / `Tensor.is_contiguous`

- **Signature:** `contiguous() -> Tensor`; `is_contiguous() -> bool`
- **Source:** `bind_tpu.cpp:136-137`
- **Behavior:** `contiguous()` returns `self` if already contiguous, else a contiguous copy.
- **PyTorch:** `t.contiguous()` / `t.is_contiguous()` — equivalent.

<!-- ex: id=r1-expand-clone device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0], [2, 1], cpu)
e = t.expand([2, 3])
print(e.shape, e.to_vector(), e.is_contiguous())
print(t.clone().to_vector(), t.detach().shape)
```

## Device and dtype movement

### `Tensor.to`

- **Signature:** `to(device) -> Tensor`; `to(dtype) -> Tensor`; `to(device, dtype) -> Tensor`
- **Source:** `bind_tpu.cpp:73-75`
- **Behavior:** moves and/or casts. No string form: `t.to("cuda")` raises a pybind TypeError — pass a `litetorch.Device` (D-03). If the target accelerator backend is absent, the tensor silently lands on CPU with no error (D-02) — always check `t.device` after `.to()`.
- **PyTorch:** `t.to("cuda")`, `t.to(torch.float32)` — differs: torch accepts strings and raises when CUDA is unavailable.

### `Tensor.to_device_async`

- **Signature:** `to_device_async(target_device) -> Tensor`
- **Source:** `bind_tpu.cpp:139`; C++ `tensor.h:147`
- **Returns:** a tensor on `target_device`, transferred asynchronously (the caller must synchronize before reading on the target).
- **PyTorch:** `t.to(device, non_blocking=True)` — similar semantics.

### `Tensor.cuda` / `Tensor.cpu`

- **Signature:** `cuda(device=None) -> Tensor`; `cpu() -> Tensor`
- **Source:** `bind_tpu.cpp:142-149`
- **Behavior:** `.cuda()` moves to GPU index (`-1` means current); on a GPU-less box it silently returns a CPU tensor (D-02). `.cpu()` always moves to CPU.
- **PyTorch:** `t.cuda()` raises without CUDA; `t.cpu()` — equivalent.

<!-- ex: id=r1-to-cpu device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
u = t.to(cpu)
print(u.device, u.cpu().shape)
print(t.to(lt.DataType.FP64).dtype)
```

### `Tensor.cast` / `Tensor.type_as`

- **Signature:** `cast(target_dtype) -> Tensor`; `type_as(other) -> Tensor`
- **Source:** `bind_tpu.cpp:140, 157`
- **Behavior:** `cast` converts to the given `DataType`; `type_as` converts to `other.dtype`.
- **PyTorch:** `t.to(dtype)` / `t.type_as(other)` — equivalent.

### `Tensor.half` / `Tensor.float` / `Tensor.double`

- **Signature:** `half() -> Tensor`; `float() -> Tensor`; `double() -> Tensor`
- **Source:** `bind_tpu.cpp:152-154`
- **Behavior:** cast to FP16 / FP32 / FP64.
- **PyTorch:** `t.half()` / `t.float()` / `t.double()` — equivalent.

### `Tensor.long` / `Tensor.int`

- **Signature:** `long() -> Tensor`; `int() -> Tensor`
- **Source:** `bind_tpu.cpp:155-156`
- **Behavior:** `.long()` maps to `Tensor::to_long` (INT64) since the D-06 fix (qa-r62); `.int()` maps to `Tensor::to_int` (INT32). Note the interplay with D-14 (still open): `.numpy()` rejects INT64, so `.long().numpy()` throws — use `.to_vector()` or cast first.
- **PyTorch:** `t.long()` is INT64; `t.int()` is INT32 — both match now.

<!-- ex: id=r1-cast-long device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=unverified -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
print(t.half().dtype, t.float().dtype, t.double().dtype)
print(t.long().dtype, t.int().dtype, t.cast(lt.DataType.INT64).dtype)
print(t.type_as(t.cast(lt.DataType.INT64)).dtype)
```

### `Tensor.copy_` / `Tensor.add_`

- **Signature:** `copy_(src) -> None`; `add_(other) -> None`
- **Source:** `bind_tpu.cpp:233-234`; C++ `tensor.h:160-161`
- **Behavior:** in-place copy / add; `src`/`other` must be broadcast-compatible and on the same device.
- **PyTorch:** `t.copy_(src)` / `t.add_(other)` — equivalent.

<!-- ex: id=r1-inplace device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.zeros([2], cpu)
t.copy_(lt.Tensor.from_vector([5.0, 6.0], [2], cpu))
t.add_(lt.Tensor.from_vector([1.0, 1.0], [2], cpu))
print(t.to_vector())
```

## Autograd

### `Tensor.backward`

- **Signature:** `backward(gradient=None, create_graph=False) -> None`
- **Source:** `bind_tpu.cpp:68`
- **Behavior:** runs reverse-mode autograd from this tensor. Non-scalar tensors require an explicit `gradient` of the same shape. `create_graph=True` builds the graph for higher-order gradients.
- **PyTorch:** `t.backward(gradient=..., create_graph=...)` — equivalent.

### `Tensor.zero_grad`

- **Signature:** `zero_grad() -> None`
- **Source:** `bind_tpu.cpp:71`
- **Behavior:** zeroes `.grad` in place (no-op when `grad is None`).
- **PyTorch:** `t.grad.zero_()` / `optimizer.zero_grad()` — equivalent in effect.

### `Tensor.retain_grad`

- **Signature:** `retain_grad() -> None`
- **Source:** `bind_tpu.cpp:72`
- **Behavior:** marks a non-leaf tensor to keep its `.grad` after backward.
- **PyTorch:** `t.retain_grad()` — equivalent.

### `Tensor.register_hook`

- **Signature:** `register_hook(hook) -> None`
- **Source:** `bind_tpu.cpp:235`; C++ `tensor.h:200`
- **Parameters:** `hook` — a callable taking the gradient tensor and returning a (possibly modified) gradient tensor.
- **PyTorch:** `t.register_hook(hook)` — equivalent.

<!-- ex: id=r1-autograd device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([2.0, 3.0], [2], cpu, True)
y = (x * x).sum()
y.backward()
print(x.grad.to_vector())
x.zero_grad()
x.retain_grad()
```

## Reduction and comparison methods

### `Tensor.sum` / `Tensor.mean`

- **Signature:** `sum(dim=None, keepdim=False) -> Tensor`; `mean(dim=None, keepdim=False) -> Tensor`
- **Source:** `bind_tpu.cpp:105-119`
- **Behavior:** `dim=None` reduces over all elements. A single int `dim` reduces along that axis. A list/tuple of dims is NOT accepted — the binding casts `dim` to a single `int64_t` (D-12); use `litetorch.Ops.sum(t, [0, 1])` for multi-dim reduction.
- **PyTorch:** `t.sum(dim=(0, 1))` works in torch — differs for tuple dims.

### `Tensor.max`

- **Signature:** `max(dim=None, keepdim=False) -> Tensor | (Tensor, Tensor)`
- **Source:** `bind_tpu.cpp:105-119`
- **Behavior:** `dim=None` returns the scalar max; with `dim` returns a `(values, indices)` tuple.
- **PyTorch:** `t.max()` / `t.max(dim)` returns `torch.return_types.max` — equivalent values.

### `Tensor.argmax` / `Tensor.argmin`

- **Signature:** `argmax(dim=-1, keepdim=False) -> Tensor`; `argmin(dim=-1, keepdim=False) -> Tensor`
- **Source:** `bind_tpu.cpp:94-95`
- **Returns:** index tensor (INT64) of the max/min position along `dim`.
- **PyTorch:** `t.argmax(dim, keepdim)` — equivalent.

### `Tensor.topk`

- **Signature:** `topk(k, dim=-1, largest=True, sorted=True) -> (Tensor, Tensor)`
- **Source:** `bind_tpu.cpp:96`
- **Returns:** `(values, indices)` tuple.
- **PyTorch:** `t.topk(k, dim, largest, sorted)` — equivalent.

<!-- ex: id=r1-reduce-methods device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 3.0, 2.0, 4.0], [2, 2], cpu)
print(t.sum().to_vector(), t.sum(0).to_vector(), t.mean(1, True).to_vector())
print(t.max().to_vector(), t.max(1)[0].to_vector(), t.argmax(1).to_vector())
print(t.topk(1, dim=1)[0].to_vector())
```

### `Tensor.eq` / `ne` / `lt` / `gt` / `le` / `ge`

- **Signature:** `eq(other) -> Tensor` (and `ne`, `lt`, `gt`, `le`, `ge`)
- **Source:** `bind_tpu.cpp:97-102`
- **Parameters:** `other` — a tensor (broadcastable).
- **Returns:** boolean tensor.
- **PyTorch:** `t.eq(other)` etc. — equivalent.

### `Tensor.log_softmax`

- **Signature:** `log_softmax(dim=-1) -> Tensor`
- **Source:** `bind_tpu.cpp:103`
- **PyTorch:** `torch.log_softmax(t, dim)` — equivalent.

### `Tensor.silu`

- **Signature:** `silu() -> Tensor`
- **Source:** `bind_tpu.cpp:104`
- **Behavior:** `x * sigmoid(x)`.
- **PyTorch:** `torch.nn.functional.silu(t)` — equivalent.

<!-- ex: id=r1-compare-methods device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu)
print(t.gt(lt.Tensor.from_vector([1.5, 1.5, 1.5], [3], cpu)).to_vector())
print(t.log_softmax().to_vector(), t.silu().to_vector())
```

## Arithmetic dunders

Elementwise operators delegate to `Ops.*`. Scalar overloads materialize the Python number as a 1-element FP32 tensor, so an int64 tensor plus a Python int promotes to FP32 (D-08). Cross-device operands are silently transferred instead of raising (D-10); on GPU, integer math runs through FP32 (D-21).

### `Tensor.__add__` / `__radd__`, `__sub__` / `__rsub__`, `__mul__` / `__rmul__`, `__truediv__` / `__rtruediv__`, `__neg__`

- **Signatures:** `t + u`, `t + scalar`, `scalar + t` (and `-`, `*`, `/`); `-t`.
- **Source:** `bind_tpu.cpp:120-133`
- **PyTorch:** same operators — equivalent except D-08/D-10/D-21.

### `Tensor.__matmul__`

- **Signature:** `t @ u -> Tensor`
- **Source:** `bind_tpu.cpp:133`
- **PyTorch:** `t @ u` — equivalent.

### `Tensor.__pow__`

- **Signature:** `t ** scalar -> Tensor` (scalar exponent only, a `double`)
- **Source:** `bind_tpu.cpp:134`
- **Behavior:** there is NO `.pow()` method (D-05) and no tensor-exponent form: `t ** u` raises. Use `litetorch.Ops.pow(t, float_exp)`.
- **PyTorch:** `t ** u` (tensor exponent) and `t.pow(exp)` both work — differs.

<!-- ex: id=r1-dunders device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
b = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print((a + b).to_vector(), (a * 2.0).to_vector(), (-a).to_vector())
print((a ** 2.0).to_vector())
m1 = lt.Tensor.from_vector([1.0, 2.0], [1, 2], cpu)
m2 = lt.Tensor.from_vector([3.0, 4.0], [2, 1], cpu)
print((m1 @ m2).to_vector())
```

## C++-only members (not bound to Python)

### `Tensor::data_ptr` / `Tensor::raw_data_ptr`

- **C++:** `float* data_ptr()` (`include/litetorch/tensor.h:185`); `void* raw_data_ptr()` (`tensor.h:186`).
- **Behavior:** `data_ptr()` throws `runtime_error` on any non-FP32 tensor (D-01) and returns a CPU pointer even for GPU tensors. Use `raw_data_ptr()` for byte-level access on any dtype.
- **PyTorch:** `t.data_ptr()` works for every dtype — differs.

### `Ops::reduce_broadcast` / `Ops::from_data`

- **C++ only:** declared in `include/litetorch/ops.h`, not bound. `from_data` is used internally by `lt.tensor()`.
