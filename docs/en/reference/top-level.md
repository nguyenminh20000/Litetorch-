# Top-Level Functions — API Reference

Functions on the `litetorch` module itself: creation, RNG, serialization, grad-mode, allocator cache, and device/backend queries.

- **Source of truth:** `src/bindings/bind_core_ops.cpp` (creation, grad-mode, cache), `src/bindings/bind_data.cpp:175-187` (`randn`/`ones`/`zeros`/`arange`), `src/bindings/bind_core.cpp` (device/backend), `src/bindings/bind_cuda.cpp:54-58` (`equal`, `get_gpu_memory_used`).
- **Tree:** agent-v1@02cc271a / local HEAD 88a4ba1 (docs baseline; B5 re-verifies at head).
- **Conventions:** `cpu = lt.Device("cpu")` in every example. Shape arguments take a single sequence (`[2, 3]`), never varargs (D-04). Torch parity is FP32 1e-4 unless the header says otherwise.

## Creation

### `litetorch.tensor`

- **Signature:** `tensor(data, device=None, requires_grad=False, dtype=None) -> Tensor`
- **Source:** `src/bindings/bind_core_ops.cpp:20`
- **Parameters:** `data` — nested Python sequences (lists/tuples) of numbers; ragged nesting raises `inconsistent nested sequence sizes`. Values pass through `float` first (D-18: int64 values above 2^24 lose precision). `dtype` — default FP32; use `lt.DataType.INT64` (not `lt.int64`, which does not exist — see defect-fixes.md D3).
- **Returns:** tensor with shape inferred from nesting; a scalar becomes shape `[1]`.
- **PyTorch:** `torch.tensor(data, dtype=...)` — equivalent for FP32 data; differs on int64 precision.

<!-- ex: id=r2-tensor device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
t = lt.tensor([[1.0, 2.0], [3.0, 4.0]])
print(t.shape, t.to_vector())
print(lt.tensor([1, 2, 3], dtype=lt.DataType.INT64).dtype)
```

### `litetorch.empty`

- **Signature:** `empty(shape, device=None, requires_grad=False, dtype=None) -> Tensor`
- **Source:** `src/bindings/bind_core_ops.cpp:40`
- **Parameters:** `shape` — a single int or a sequence (a bare int means 1-D). `dtype` — default FP32.
- **Returns:** uninitialized tensor; contents undefined.
- **PyTorch:** `torch.empty(shape)` — equivalent (torch also accepts varargs; LiteTorch does not, D-04).

### `litetorch.full`

- **Signature:** `full(shape, fill_value, device=None, requires_grad=False, dtype=None) -> Tensor`
- **Source:** `src/bindings/bind_core_ops.cpp:43`
- **Parameters:** `fill_value` — a Python number, cast to `float` first.
- **Returns:** tensor filled with `fill_value`.
- **PyTorch:** `torch.full(shape, fill_value)` — equivalent.

<!-- ex: id=r2-empty-full device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
print(lt.empty([2, 2], cpu).shape)
print(lt.full([2, 2], 7.0, cpu).to_vector())
```

### `litetorch.linspace` / `litetorch.logspace`

- **Signatures:** `linspace(start, end, steps, device=None, dtype=None) -> Tensor`; `logspace(start, end, steps, base=10.0, device=None, dtype=None) -> Tensor`
- **Source:** `src/bindings/bind_core_ops.cpp:46-52`
- **Parameters:** `start`, `end` — floats (cast to FP32); `steps` — int, the number of points including both ends.
- **Returns:** 1-D tensor of `steps` evenly spaced values (`linspace`) or `base ** linspace` values (`logspace`).
- **PyTorch:** `torch.linspace(start, end, steps)` / `torch.logspace(start, end, steps, base)` — equivalent.

<!-- ex: id=r2-linspace device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
print(lt.linspace(0.0, 1.0, 3, cpu).to_vector())
print(lt.logspace(0.0, 2.0, 3, 10.0, cpu).to_vector())
```

### `litetorch.eye`

- **Signature:** `eye(n, m=None, device=None, dtype=None) -> Tensor`
- **Source:** `src/bindings/bind_core_ops.cpp:52`
- **Parameters:** `n` — rows; `m` — columns, defaults to `n`.
- **Returns:** identity matrix (ones on the diagonal, zeros elsewhere).
- **PyTorch:** `torch.eye(n, m)` — equivalent.

<!-- ex: id=r2-eye device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
print(lt.eye(2, 3, cpu).to_vector())
```

### `litetorch.rand`

- **Signature:** `rand(shape, device=None, requires_grad=False, dtype=None) -> Tensor`
- **Source:** `src/bindings/bind_core_ops.cpp:56`
- **Parameters:** `shape` — a single int or a sequence.
- **Returns:** uniform samples in [0, 1). Seeding: `lt.manual_seed(seed)`.
- **PyTorch:** `torch.rand(shape)` — same distribution; same seed does not imply the same stream.

### `litetorch.randint`

- **Signatures:** `randint(low, high, shape, device=None, dtype=None) -> Tensor`; `randint(high, shape, device=None, dtype=None) -> Tensor`
- **Source:** `src/bindings/bind_core_ops.cpp:59-64`
- **Parameters:** integers in `[low, high)` (second form: `[0, high)`). Default dtype INT32.
- **PyTorch:** `torch.randint(low, high, shape)` — equivalent.

### `litetorch.randperm`

- **Signature:** `randperm(n, device=None, dtype=None) -> Tensor`
- **Source:** `src/bindings/bind_core_ops.cpp:65`
- **Returns:** a random permutation of `0..n-1`. Default dtype INT32.
- **PyTorch:** `torch.randperm(n)` — equivalent.

<!-- ex: id=r2-rand device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
print(lt.rand([2], cpu).shape)
print(lt.randint(0, 10, [3], cpu).dtype, lt.randperm(4, cpu).shape)
```

### `litetorch.randn` / `litetorch.ones` / `litetorch.zeros` / `litetorch.arange`

- **Signatures:** `randn(shape, device=Device(CPU, 0), requires_grad=False, dtype=DataType.FP32)`; `ones(shape, ...)`; `zeros(shape, ...)`; `arange(start, end, step=1, device=..., dtype=...)`
- **Source:** `src/bindings/bind_data.cpp:175-187`
- **Behavior:** top-level aliases of the `Tensor` static factories (same signatures, same semantics, same deviations: D-04 single shape sequence, D-15 integer-only `arange`).
- **PyTorch:** `torch.randn(shape)` etc. — equivalent except D-04/D-15.

<!-- ex: id=r2-factories device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(1)
print(lt.randn([2], cpu).shape, lt.ones([2], cpu).to_vector())
print(lt.zeros([2], cpu).to_vector(), lt.arange(0, 4, 2, cpu).to_vector())
```

### `litetorch.zeros_like` / `ones_like` / `empty_like` / `full_like` / `rand_like` / `randn_like`

- **Signatures:** `zeros_like(a, device=None, dtype=None)`; `ones_like(a, ...)`; `empty_like(a, ...)`; `full_like(a, fill_value, device=None, dtype=None)`; `rand_like(a, ...)`; `randn_like(a, ...)`
- **Source:** `src/bindings/bind_core_ops.cpp:68-90`
- **Parameters:** `a` — the reference tensor. `device`/`dtype` default to `a`'s own device/dtype when omitted.
- **Returns:** a tensor of the same shape as `a`, filled accordingly.
- **PyTorch:** `torch.zeros_like(a)` etc. — equivalent.

<!-- ex: id=r2-like device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
print(lt.zeros_like(a).to_vector(), lt.ones_like(a).to_vector())
print(lt.full_like(a, 9.0).to_vector(), lt.empty_like(a).shape)
```

## RNG

### `litetorch.manual_seed`

- **Signature:** `manual_seed(seed) -> None`
- **Source:** `src/bindings/bind_data.cpp:172`
- **Behavior:** seeds the process-global RNG used by `rand`/`randn`/`randint`/`randperm` and the `rand*_like` family. Does NOT seed `DataLoader` shuffle (B1 finding: shuffle uses its own RNG; use `shuffle=False` for reproducible examples).
- **PyTorch:** `torch.manual_seed(seed)` — equivalent in scope, different stream.

### `litetorch.get_rng_state` / `litetorch.set_rng_state`

- **Signatures:** `get_rng_state() -> state`; `set_rng_state(state) -> None`
- **Source:** `src/bindings/bind_data.cpp:173-174`
- **Behavior:** snapshot/restore the global RNG state for exact reproducibility.
- **PyTorch:** `torch.get_rng_state()` / `torch.set_rng_state(state)` — equivalent.

<!-- ex: id=r2-rng device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(7)
a = lt.randn([2], cpu).to_vector()
st = lt.get_rng_state()
b = lt.randn([2], cpu).to_vector()
lt.set_rng_state(st)
c = lt.randn([2], cpu).to_vector()
print(a, b == c)
```

## Serialization

Files are written in the `.lt` format (see `11.data_and_io.md` / R11). `save`/`load` dispatch on the object type; the `save_*`/`load_*` functions take explicit parameter lists.

### `litetorch.save` / `litetorch.load`

- **Signatures:** `save(obj, filepath) -> None`; `load(obj, filepath) -> None`
- **Source:** `src/bindings/bind_data.cpp:150-170`
- **Parameters:** `obj` — an `nn.Module`, a list/tuple of parameter tensors, or an `Optimizer`. Anything else raises.
- **Behavior:** `save` writes parameters (module or list) or optimizer state; `load` reads them back into the same object in place.
- **PyTorch:** `torch.save(obj, path)` / `torch.load(path)` — differs: torch returns the loaded object; LiteTorch loads into the given object.

### `litetorch.save_parameters` / `litetorch.load_parameters`

- **Signatures:** `save_parameters(params, filepath)`; `load_parameters(params, filepath)`
- **Source:** `src/bindings/bind_data.cpp:118-119`
- **Parameters:** `params` — list of tensors.
- **Edge cases:** shape mismatch on load raises; see R11 for the `.lt` format.

### `litetorch.save_safetensors` / `litetorch.load_safetensors`

- **Signatures:** `save_safetensors(tensors, filepath)`; `load_safetensors(filepath) -> dict`
- **Source:** `src/bindings/bind_data.cpp:120-129`
- **Parameters:** `tensors` — dict of `str -> Tensor`. `load_safetensors` returns a dict of `str -> Tensor`.
- **PyTorch:** `safetensors.torch.save_file` / `load_file` — equivalent format.

### `litetorch.save_optimizer_state` / `litetorch.load_optimizer_state`

- **Signatures:** `save_optimizer_state(optimizer, filepath)`; `load_optimizer_state(optimizer, filepath)`
- **Source:** `src/bindings/bind_data.cpp:130-131`
- **Parameters:** `optimizer` — an `optim.Optimizer` instance.

<!-- ex: id=r2-serialization device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import os
cpu = lt.Device("cpu")
m = lt.nn.Linear(2, 2, True)
path = "/tmp/b2-r2-params.lt"
lt.save(m, path)
n = lt.nn.Linear(2, 2, True)
lt.load(n, path)
print([p.to_vector() for p in m.parameters()] == [p.to_vector() for p in n.parameters()])
os.remove(path)
```

## Grad mode

### `litetorch.no_grad` / `litetorch.enable_grad` / `litetorch.inference_mode`

- **Signatures:** context managers, used as `with lt.no_grad(): ...`
- **Source:** `src/bindings/bind_core_ops.cpp:106-124`
- **Behavior:** `no_grad` disables gradient tracking inside the block; `enable_grad` re-enables it inside a `no_grad` block; `inference_mode` disables tracking and version-counter bumps (fastest inference path).
- **PyTorch:** `torch.no_grad()` / `torch.enable_grad()` / `torch.inference_mode()` — equivalent.

### `litetorch.is_grad_enabled` / `litetorch.set_grad_enabled`

- **Signatures:** `is_grad_enabled() -> bool`; `set_grad_enabled(enabled) -> None`
- **Source:** `src/bindings/bind_core_ops.cpp:126-132`
- **PyTorch:** `torch.is_grad_enabled()` / `torch.set_grad_enabled(enabled)` — equivalent.

### `litetorch.is_inference_mode`

- **Signature:** `is_inference_mode() -> bool`
- **Source:** `src/bindings/bind_core_ops.cpp:124`
- **PyTorch:** `torch.is_inference_mode_enabled()` — equivalent.

<!-- ex: id=r2-grad-mode device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
print(lt.is_grad_enabled())
with lt.no_grad():
    x = lt.Tensor.from_vector([2.0], [1], cpu, True)
    print(lt.is_grad_enabled(), x.requires_grad)
print(lt.is_grad_enabled(), lt.is_inference_mode())
```

## Allocator cache

### `litetorch.empty_cache`

- **Signature:** `empty_cache() -> None`
- **Source:** `src/bindings/bind_core_ops.cpp:92`
- **Behavior:** releases cached CPU and GPU blocks held by the caching allocator (see the memory-eviction explanation, E2).
- **PyTorch:** `torch.cuda.empty_cache()` — similar; LiteTorch also empties the CPU cache.

### `litetorch.set_max_cpu_cache_size` / `get_cached_cpu_bytes` / `get_cached_gpu_bytes`

- **Signatures:** `set_max_cpu_cache_size(bytes) -> None`; `get_cached_cpu_bytes() -> int`; `get_cached_gpu_bytes() -> int`
- **Source:** `src/bindings/bind_core_ops.cpp:97-105`
- **PyTorch:** no direct equivalent for the CPU cache knobs.

<!-- ex: id=r2-cache device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
lt.empty_cache()
lt.set_max_cpu_cache_size(1 << 30)
print(lt.get_cached_cpu_bytes(), lt.get_cached_gpu_bytes())
```

## Device and backend queries

### `litetorch.Device`

- **Signatures:** `Device(type=DeviceType.CPU, index=0)`; `Device(device_str)`; attributes `type`, `index` (read/write); `to_string()` / `__repr__`.
- **Source:** `src/bindings/bind_core.cpp:23-43`
- **Behavior:** the string form parses `"cuda"`/`"gpu"` (optional `:index`), `"tpu"` (optional `:index`), `"meta"` (case-insensitive). ANY other string — including typos such as `"cude"` — silently becomes CPU with no error (D-03).
- **PyTorch:** `torch.device("cuda:0")` — differs: torch raises on an invalid device string.

### `litetorch.DeviceType` / `litetorch.DataType`

- **Values:** `DeviceType`: CPU, GPU, META, TPU (`bind_core.cpp:14`). `DataType`: FP64, FP32, FP16, BF16, INT16, INT8, INT4, FP8_E4M3, FP8_E5M2, NF4, FP4_E2M1 (alias FP4), INT32, INT64, UINT8, BOOL (`bind_tpu.cpp:12-29`, exported top-level, so `lt.INT64` works too).
- **PyTorch:** `torch.dtype` covers a subset; FP8/NF4/FP4/INT4 have no torch counterpart.

<!-- ex: id=r2-device device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
print(lt.Device("cuda:1").to_string(), lt.Device("cude").to_string())
print(lt.DeviceType.CPU, lt.DataType.FP32, lt.INT64)
```

### `litetorch.auto_device`

- **Signature:** `auto_device() -> Device`
- **Source:** `src/bindings/bind_core.cpp:66`
- **Behavior:** returns GPU 0 if a native backend is available, else OpenCL GPU 0, else TPU 0, else CPU 0. Honors `LITETORCH_NO_NATIVE_GPU`.
- **PyTorch:** no equivalent single call; `torch.cuda.is_available()` + manual selection.

### `litetorch.is_gpu_available` / `litetorch.cuda_synchronize` / `litetorch.is_tpu_available` / `litetorch.get_backend_name`

- **Signatures:** `is_gpu_available() -> bool`; `cuda_synchronize() -> None`; `is_tpu_available() -> bool`; `get_backend_name() -> str`
- **Source:** `src/bindings/bind_core.cpp:45-64, 80`
- **Behavior:** `get_backend_name()` returns `"CUDA"`, `"ROCm"`, `"OpenCL"`, `"TPU (Google PJRT / libtpu)"`, or `"CPU"`.
- **PyTorch:** `torch.cuda.is_available()` / `torch.cuda.synchronize()` — equivalent for the CUDA pair.

### `litetorch.build_commit`

- **Signature:** `build_commit() -> str`
- **Source:** `src/bindings/bind_core.cpp:6`
- **Returns:** the git commit the extension was built from, or `"unknown"`.
- **PyTorch:** `torch.__version__` — similar purpose.

### `litetorch.equal`

- **Signature:** `equal(a, b) -> bool`
- **Source:** `src/bindings/bind_cuda.cpp:54`
- **Behavior:** exact element-wise equality (same shape, same values).
- **PyTorch:** `torch.equal(a, b)` — equivalent.

### `litetorch.get_gpu_memory_used`

- **Signature:** `get_gpu_memory_used() -> int`
- **Source:** `src/bindings/bind_cuda.cpp:56`
- **Returns:** GPU bytes currently held by the memory manager (same value as `lt.cuda.memory_allocated()`).
- **PyTorch:** `torch.cuda.memory_allocated()` — equivalent.

### `litetorch.checkpoint`

- **Signature:** `checkpoint(fn, *args) -> Tensor` (top-level activation-checkpointing helper)
- **Source:** `src/bindings/bind_ops.cpp:181`
- **Behavior:** runs `fn` without storing intermediate activations; recomputes them on backward. See also `Ops.checkpoint(module, input)` in R5/R13.
- **PyTorch:** `torch.utils.checkpoint.checkpoint(fn, *args)` — equivalent in purpose.

<!-- ex: id=r2-backend device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
print(lt.auto_device(), lt.get_backend_name(), lt.build_commit())
print(lt.is_gpu_available(), lt.is_tpu_available(), lt.get_gpu_memory_used())
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
print(lt.equal(a, a), lt.equal(a, lt.Tensor.zeros([2], cpu)))
```
