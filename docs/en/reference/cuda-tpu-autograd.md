# CUDA, TPU, Autograd, Enums — API Reference

This page covers device management (`litetorch.cuda`, `litetorch.tpu`), the autograd engine (`litetorch.autograd`), and the two core enums (`DataType`, `DeviceType`).

- **Source of truth:** `src/bindings/bind_cuda.cpp`, `bind_tpu.cpp`, `bind_autograd.cpp`, `bind_core.cpp` (Python bindings); `include/litetorch/device.h`, `autograd.h`, `tensor.h`, `cuda.h`, `tpu.h`.
- **Tree:** agent-v1@02cc271a / local HEAD 89afd67 (docs baseline is the 88a4ba1 extension; B5 re-verifies at head).
- **Conventions:** `cpu = lt.Device("cpu")` in every example. Every example is self-contained and runnable on CPU. GPU/TPU examples are marked `REQUIRES-GPU` / `REQUIRES-TPU` and verified on Kaggle in B5. Torch parity is FP32 1e-4 unless the header says otherwise.
- **Related deviations:** D-02 (`to(device)` silent CPU fallback — `lt.cuda.is_available()` is False on this box, so any `.to("cuda")` falls back to CPU silently; see `plan/differences.md`). D-20 (`flash_attention_half` throws on TPU). D-01 (`Tensor.data_ptr()` throws on non-FP32 — C++-only; see `cpp-api.md`).

## CUDA

`litetorch.cuda` mirrors `torch.cuda` for device management. Note the binding quirk: `is_available` is registered **twice** in `src/bindings/bind_cuda.cpp` — once at line 5 (a lambda that checks the native CUDA backend) and once at line 48 (bound directly to `litetorch::cuda::is_available`). Both return the same value; the duplicate is harmless.

### `cuda.is_available`

- **Signature:** `is_available() -> bool`
- **Source:** `src/bindings/bind_cuda.cpp:5,48`
- **Returns:** `True` when a native CUDA (or HIP) backend is loaded; `False` on a CPU-only box (verified: `False` on the 88a4ba1 CPU extension).
- **PyTorch:** `torch.cuda.is_available()` — same role.

### `cuda.device_count`

- **Signature:** `device_count() -> int`
- **Source:** `src/bindings/bind_cuda.cpp:49`
- **Returns:** the number of CUDA devices (0 on a CPU-only box).
- **PyTorch:** `torch.cuda.device_count()` — same role.

### `cuda.current_device`

- **Signature:** `current_device() -> int`
- **Source:** `src/bindings/bind_cuda.cpp:50`
- **Returns:** the index of the current CUDA device (0 when no device exists).
- **PyTorch:** `torch.cuda.current_device()` — same role.

### `cuda.set_device`

- **Signature:** `set_device(device=0) -> None`
- **Source:** `src/bindings/bind_cuda.cpp:51`
- **Parameters:** `device` — int device index, default 0.
- **Returns:** `None`. On a CPU-only box this is a no-op.
- **PyTorch:** `torch.cuda.set_device()` — same role.

### `cuda.synchronize`

- **Signature:** `synchronize() -> None`
- **Source:** `src/bindings/bind_cuda.cpp:41`
- **Returns:** `None`. Waits for all CUDA work on the current device to finish. No-op without CUDA.
- **PyTorch:** `torch.cuda.synchronize()` — same role.

### `cuda.memory_allocated` / `cuda.memory_reserved`

- **Signature:** `memory_allocated() -> int`; `memory_reserved() -> int`
- **Source:** `src/bindings/bind_cuda.cpp:38,52`
- **Returns:** bytes currently allocated / reserved by the caching allocator on the current device (0 without CUDA).
- **PyTorch:** `torch.cuda.memory_allocated()` / `memory_reserved()` — same role.

### `cuda.set_tf32_enabled` / `cuda.is_tf32_enabled`

- **Signature:** `set_tf32_enabled(enabled) -> None`; `is_tf32_enabled() -> bool`
- **Source:** `src/bindings/bind_cuda.cpp:12,18`
- **Parameters:** `enabled` — bool.
- **Returns:** `is_tf32_enabled` reports whether TF32 matmul is allowed (default False on this build — verify by printing).
- **PyTorch:** `torch.backends.cuda.matmul.allow_tf32` — same setting, different shape (torch uses a module attribute, LiteTorch uses functions).

### `cuda.set_fp16_math_enabled` / `cuda.is_fp16_math_enabled`

- **Signature:** `set_fp16_math_enabled(enabled) -> None`; `is_fp16_math_enabled() -> bool`
- **Source:** `src/bindings/bind_cuda.cpp:25,31`
- **Parameters:** `enabled` — bool.
- **Returns:** `is_fp16_math_enabled` reports whether FP16 math is enabled.
- **PyTorch:** no direct equivalent; closest is `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction`.

### `cuda.manual_seed_all`

- **Signature:** `manual_seed_all(seed) -> None`
- **Source:** `src/bindings/bind_cuda.cpp:53`
- **Parameters:** `seed` — int.
- **Returns:** `None`. Seeds the RNG on all CUDA devices. No-op without CUDA (the CPU RNG is seeded by `lt.manual_seed`).
- **PyTorch:** `torch.cuda.manual_seed_all(seed)` — same role.

### `cuda.get_gpu_memory_used`

- **Signature:** `get_gpu_memory_used() -> int` (bound at module level, not under `cuda`)
- **Source:** `src/bindings/bind_cuda.cpp:56`
- **Returns:** bytes of GPU memory currently in use (0 without CUDA).
- **PyTorch:** no direct equivalent; closest is `torch.cuda.memory_allocated()`.

<!-- ex: id=r14-cuda device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

print("cuda available:", lt.cuda.is_available())
print("device count:", lt.cuda.device_count())
print("current device:", lt.cuda.current_device())
print("memory allocated:", lt.cuda.memory_allocated())
print("memory reserved:", lt.cuda.memory_reserved())
print("tf32 enabled:", lt.cuda.is_tf32_enabled())
lt.cuda.set_tf32_enabled(True)
print("tf32 after set:", lt.cuda.is_tf32_enabled())
lt.cuda.set_tf32_enabled(False)
print("fp16 math enabled:", lt.cuda.is_fp16_math_enabled())
lt.cuda.manual_seed_all(42)
lt.cuda.synchronize()
print("gpu memory used:", lt.get_gpu_memory_used())
```

## TPU

`litetorch.tpu` mirrors the `cuda` surface for TPU devices (PJRT backend). On a CPU-only box every query returns the empty-device answer.

### `tpu.is_available`

- **Signature:** `is_available() -> bool`
- **Source:** `src/bindings/bind_tpu.cpp:5`
- **Returns:** `True` when a TPU (PJRT) backend is loaded; `False` on this box (verified).
- **PyTorch:** `torch_xla.runtime` / `torch.tpu` — no `torch.tpu.is_available`; the closest is `torch_xla.core.xla_model.get_xla_supported_devices()`.

### `tpu.device_count`

- **Signature:** `device_count() -> int`
- **Source:** `src/bindings/bind_tpu.cpp:6`
- **Returns:** the number of TPU devices (0 on this box).
- **PyTorch:** `torch_xla.core.xla_model.xrt_world_size()` — same role, different shape.

### `tpu.current_device`

- **Signature:** `current_device() -> int`
- **Source:** `src/bindings/bind_tpu.cpp:7`
- **Returns:** the index of the current TPU device (0 when none exists).
- **PyTorch:** `torch_xla.core.xla_model.get_local_ordinal()` — same role.

### `tpu.set_device`

- **Signature:** `set_device(device_id=0) -> None`
- **Source:** `src/bindings/bind_tpu.cpp:8`
- **Parameters:** `device_id` — int, default 0.
- **Returns:** `None`. No-op without a TPU.
- **PyTorch:** no direct equivalent.

### `tpu.synchronize`

- **Signature:** `synchronize() -> None`
- **Source:** `src/bindings/bind_tpu.cpp:9`
- **Returns:** `None`. Blocks until all TPU work completes. No-op without a TPU.
- **PyTorch:** `torch_xla.core.xla_model.mark_step()` — same role (flush the lazy graph).

### `tpu.get_device_name`

- **Signature:** `get_device_name(device_id) -> str`
- **Source:** `src/bindings/bind_tpu.cpp:10`
- **Parameters:** `device_id` — int.
- **Returns:** the device name string, or `"N/A"` when the device does not exist (verified: `get_device_name(0)` returns `"N/A"` on this box).
- **PyTorch:** `torch.cuda.get_device_name` — same shape, for CUDA.

<!-- ex: id=r14-tpu device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

print("tpu available:", lt.tpu.is_available())
print("device count:", lt.tpu.device_count())
print("current device:", lt.tpu.current_device())
print("device 0 name:", lt.tpu.get_device_name(0))
lt.tpu.synchronize()
```

## Autograd

### `autograd.grad`

- **Signature:** `grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=False, allow_unused=False) -> tuple[Tensor | None, ...]`
- **Source:** `src/bindings/bind_autograd.cpp:52-67`
- **Parameters:** `outputs` — tensor or list of tensors to differentiate. `inputs` — tensor or list of tensors to differentiate with respect to. `grad_outputs` — upstream gradients (default `None` = ones). `retain_graph` — keep the graph for a second backward. `create_graph` — build a graph of the gradient itself (for higher-order derivatives). `allow_unused` — return `None` for inputs that do not affect the outputs instead of throwing.
- **Returns:** a tuple of gradients, one per input, in input order. Unused inputs give `None` when `allow_unused=True`.
- **Edge cases:** without `retain_graph`, calling `grad` twice on the same graph throws. `create_graph=True` is required for `gradgradcheck`.
- **PyTorch:** `torch.autograd.grad` — same signature, same defaults, same semantics.

### `autograd.gradcheck`

- **Signature:** `gradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True) -> bool`
- **Source:** `src/bindings/bind_autograd.cpp:68-80`
- **Parameters:** `func` — a callable taking the inputs and returning a tensor (or list). `inputs` — tensor or list of tensors (must be FP64 or FP32; double precision is recommended). `eps` — finite-difference step, default 1e-6. `atol` / `rtol` — tolerances, defaults 1e-5 / 1e-3. `raise_exception` — throw on failure instead of returning False.
- **Returns:** `True` when the analytic gradient matches the numerical gradient within tolerance.
- **Edge cases:** FP32 inputs need `eps=1e-3` — the default `eps=1e-6` is below FP32 resolution at O(1) magnitudes, so the numerical gradient comes out wrong (verified: for x^2 at x=3 the numerical gradient is 5.72 instead of 6). Torch has the same constraint, which is why torch's gradcheck requires double-precision inputs; LiteTorch accepts FP32 but you must raise `eps`.
- **PyTorch:** `torch.autograd.gradcheck` — same signature and defaults.

### `autograd.gradgradcheck`

- **Signature:** `gradgradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True) -> bool`
- **Source:** `src/bindings/bind_autograd.cpp:81-93`
- **Parameters:** same as `gradcheck`.
- **Returns:** `True` when the second-order gradient matches numerically. Requires `create_graph` support in the backward. Same FP32 `eps=1e-3` rule as `gradcheck`.
- **PyTorch:** `torch.autograd.gradgradcheck` — same signature and defaults.

### `autograd.retain_grad`

- **Signature:** `retain_grad(tensor) -> None`
- **Source:** `src/bindings/bind_autograd.cpp:94-95`
- **Parameters:** `tensor` — a non-leaf tensor.
- **Returns:** `None`. Marks the tensor so `.grad` is populated after backward (by default only leaf tensors keep `.grad`).
- **PyTorch:** `Tensor.retain_grad()` — same role; LiteTorch exposes it as a module function as well as `Tensor.retain_grad()`.

### `autograd.Function`

- **Signature:** class `Function()` with a classmethod `apply(*args) -> Tensor | tuple[Tensor, ...]`; subclass it and define `forward(ctx, *args)` and `backward(ctx, *grad_outputs)`
- **Source:** `src/bindings/bind_autograd.cpp:24-51`
- **Parameters:** `apply` takes the function inputs (tensors and non-tensor arguments). Inside `forward`, `ctx.save_for_backward(*tensors)` saves tensors for the backward; `ctx.saved_tensors` and `ctx.needs_input_grad` are readable in `backward`.
- **Returns:** `forward` returns a tensor (or tuple); `backward` returns the gradient for each forward input (or `None` per input).
- **Edge cases:** non-tensor arguments to `apply` are passed through to `forward` as-is; tensor arguments arrive as tensors. `forward` must be a `@staticmethod`. A subclass that does not define `backward` raises when a gradient is requested.
- **PyTorch:** `torch.autograd.Function` — same shape (`forward` / `backward` / `ctx.save_for_backward` / `apply`).

<!-- ex: id=r14-autograd device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")


class Square(lt.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * x * 2.0


x = lt.Tensor.from_vector([3.0], [1], cpu)
x.requires_grad = True
y = Square.apply(x)
print("forward:", y.to_vector())
g = lt.autograd.grad(y, x)
print("grad:", g[0].to_vector())
print("gradcheck:", lt.autograd.gradcheck(lambda t: Square.apply(t), x, 1e-3))
print("gradgradcheck:", lt.autograd.gradgradcheck(lambda t: Square.apply(t), x, 1e-3))
```

### `autograd.FunctionContext`

- **Signature:** class `FunctionContext()`; methods `save_for_backward(*tensors)`; read-only properties `saved_tensors`, `needs_input_grad`
- **Source:** `src/bindings/bind_autograd.cpp:6-23`
- **Returns:** the `ctx` object passed to `Function.forward` / `Function.backward`. `save_for_backward` stores tensors for the backward pass; `saved_tensors` returns them as a tuple; `needs_input_grad` is a tuple of bools, one per forward input.
- **PyTorch:** `torch.autograd.function.FunctionCtx` — same role.

## Enums

### `DataType`

- **Signature:** enum with 16 names (15 distinct values — `FP4` is an alias of `FP4_E2M1`): `FP64`, `FP32`, `FP16`, `BF16`, `INT16`, `INT8`, `INT4`, `FP8_E4M3`, `FP8_E5M2`, `NF4`, `FP4_E2M1`, `FP4`, `INT32`, `INT64`, `UINT8`, `BOOL`
- **Source:** `src/bindings/bind_tpu.cpp:12-29`
- **Returns:** the dtype of a tensor (`tensor.dtype`). `FP64` is in the enum but most kernels only accept FP32/FP16/BF16 and the integer types — creating an FP64 tensor is allowed, computing with it may throw.
- **Edge cases:** `FP4` and `FP4_E2M1` compare equal (`lt.DataType.FP4 == lt.DataType.FP4_E2M1` is True). There is no `lt.int64` / `lt.float32` shorthand — use `lt.DataType.INT64` (see D3 in `plan/defect-fixes.md`).
- **PyTorch:** `torch.dtype` (`torch.float32`, `torch.int64`, ...) — same set of types; torch has no FP4/NF4/FP8 dtypes at the `torch.dtype` level (those live in `torchao` / `transformers`).

### `DeviceType`

- **Signature:** enum with 4 values: `CPU`, `GPU`, `META`, `TPU`
- **Source:** `src/bindings/bind_core.cpp:14-19`
- **Returns:** the device type of a `Device` (`device.type`). `META` is a placeholder device with no storage (used for shape-only tracing).
- **PyTorch:** `torch.device.type` (`"cpu"`, `"cuda"`, `"meta"`) — same four ideas; torch has no single `DeviceType` enum (the type is a string).

### `Device`

- **Signature:** `Device(type=DeviceType.CPU, index=0)`; `Device("cpu")` / `Device("cuda:0")` / `Device("tpu:0")` (string form, case-insensitive)
- **Source:** `src/bindings/bind_core.cpp:20-30`; `include/litetorch/device.h`
- **Parameters:** `type` — a `DeviceType`; `index` — int device index. The string form parses `"cpu"`, `"gpu"`, `"cuda"`, `"cuda:N"`, `"tpu"`, `"tpu:N"`, `"meta"`.
- **Returns:** a device object. `Device("cuda")` on a CPU-only box constructs fine — the silent fallback (D-02) happens at `.to(device)` time, not at construction.
- **PyTorch:** `torch.device` — same two forms (`torch.device("cuda:0")` / `torch.device("cpu")`).

<!-- ex: id=r14-enums device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

print("dtypes:", [lt.DataType.FP64, lt.DataType.FP32, lt.DataType.FP16,
      lt.DataType.BF16, lt.DataType.INT64, lt.DataType.UINT8, lt.DataType.BOOL])
print("fp4 alias:", lt.DataType.FP4 == lt.DataType.FP4_E2M1)
print("device types:", [lt.DeviceType.CPU, lt.DeviceType.GPU,
      lt.DeviceType.META, lt.DeviceType.TPU])
cpu = lt.Device("cpu")
print("cpu device:", cpu.type, cpu.index)
print("cuda device:", lt.Device("cuda:0").type, lt.Device("cuda:0").index)
x = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
print("tensor dtype:", x.dtype, "device:", x.device.type)
```
