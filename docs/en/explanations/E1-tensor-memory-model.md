# E1 — Tensor and Memory Model

A `Tensor` is a view over a storage block, not the block itself. This page explains what a tensor holds, how strides describe layout, how devices and the allocator work, and where LiteTorch deliberately differs from PyTorch. Every example is self-contained and runnable on CPU.

- **Tree:** integration HEAD `b326f29`; runnable extension built from `88a4ba1` (`lt.build_commit()`, asserted in every example).
- **Reference:** PyTorch 2.14.1+cpu. Torch parity is FP32 1e-4 unless the header says otherwise.
- **Related deviations:** D-01 (`data_ptr()`), D-02 (`to(device)` silent CPU fallback), D-03 (no string form for `.to()`), D-06 (`.long()` — FIXED on agent-v1, see §3), D-14 (`.numpy()` dtype limits), D-18 (`.item()`/`.tolist()` precision). See `plan/differences.md`.

## 1. Tensor anatomy

A tensor is four things: a shape, a stride per dimension, a dtype, and a pointer into a storage block plus an offset.

```
Tensor
├── shape    [2, 3]        logical size per dimension
├── strides  [3, 1]        elements to skip per step, per dimension
├── dtype    FP32          element type (DataType)
├── device   cpu:0         where the storage lives
├── offset   0             elements into the storage block
└── storage  StorageImpl   the allocated block (shared by views)
```

The address of element `(i, j)` is `base + offset + i * stride[0] + j * stride[1]`, in elements. Strides are in elements, not bytes — the same convention as PyTorch. `numel()` is the product of the shape; `ndim` is the number of dimensions; `is_contiguous()` reports whether the elements lie in row-major order with no gaps.

<!-- ex: id=E1-1 device=cpu backend=cpu runtime=~60s torch-parity=exact status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

a = lt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("shape:", a.shape)
print("strides:", a.strides)
print("dtype:", a.dtype)
print("device:", a.device)
print("numel:", a.numel())
print("ndim:", a.ndim)
print("is_contiguous:", a.is_contiguous())
b = a.transpose(0, 1)
print("transposed shape:", b.shape)
print("transposed strides:", b.strides)
print("transposed is_contiguous:", b.is_contiguous())
at = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("torch strides:", list(at.stride()))
print("torch transposed strides:", list(at.t().stride()))
assert list(a.shape) == list(at.shape)
assert list(a.strides) == list(at.stride())
assert list(b.strides) == list(at.t().stride())
print("PARITY OK")
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity exact (shapes and strides are integers).

Expected output:

```
shape: [2, 3]
strides: [3, 1]
dtype: DataType.FP32
device: cpu:0
numel: 6
ndim: 2
is_contiguous: True
transposed shape: [3, 2]
transposed strides: [1, 3]
transposed is_contiguous: False
torch strides: [3, 1]
torch transposed strides: [1, 3]
PARITY OK
```

`transpose` swaps shape and strides but does not move any data: the transposed tensor is non-contiguous, with strides `[1, 3]`. Any op that needs dense row-major data calls `.contiguous()` first, which copies. The default dtype is FP32 and the default device is `cpu:0`.

## 2. Views vs copies

`view`, `transpose`, `permute`, and `slice` create new tensors over the same storage: writing through the view writes the original. `clone` allocates new storage: writing through the clone leaves the original untouched. `copy_` writes into the destination tensor's storage, so it is the way to observe sharing.

<!-- ex: id=E1-2 device=cpu backend=cpu runtime=~60s torch-parity=exact status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

a = lt.tensor([[1.0, 2.0], [3.0, 4.0]])
v = a.view([4])
v.copy_(lt.tensor([9.0, 9.0, 9.0, 9.0]))
print("after copy_ through view, a:", a.to_vector())
c = a.clone()
c.copy_(lt.tensor([[7.0, 7.0], [7.0, 7.0]]))
print("after copy_ through clone, a:", a.to_vector())
print("clone:", c.to_vector())
t = lt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tt = t.transpose(0, 1)
print("transposed is_contiguous:", tt.is_contiguous())
print("transposed values:", tt.to_vector())
print("contiguous values:", tt.contiguous().to_vector())
at = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
av = at.view(4)
av.copy_(torch.tensor([9.0, 9.0, 9.0, 9.0]))
assert at.flatten().tolist() == a.to_vector()
assert tt.to_vector() == torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).t().flatten().tolist()
print("PARITY OK")
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity exact.

Expected output:

```
after copy_ through view, a: [9.0, 9.0, 9.0, 9.0]
after copy_ through clone, a: [9.0, 9.0, 9.0, 9.0]
clone: [7.0, 7.0, 7.0, 7.0]
transposed is_contiguous: False
transposed values: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
contiguous values: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
PARITY OK
```

Two facts to remember. First, `to_vector()` always returns values in logical order, so a transposed tensor and its contiguous copy print identically — the difference is only in strides and in whether a later op must copy. Second, indexing is limited (D-11): only `int` and `slice` on dimension 0 are supported, so elementwise inspection in this book goes through `to_vector()`, `slice()`, or `.numpy()`.

## 3. The dtype system

LiteTorch dtypes are the `DataType` enum: `FP32`, `FP64`, `FP16`, `BF16`, `INT64`, `INT32`, `INT16`, `INT8`, `UINT8`, `BOOL`, plus quantized types (`FP8_E4M3`, `FP8_E5M2`, `FP4`, `FP4_E2M1`, `NF4`, `INT4`). Casts are `.float()`, `.double()`, `.half()`, `.int()`, `.long()`, and the general `.cast(DataType)`. Casts truncate toward zero for float-to-int, matching torch.

<!-- ex: id=E1-3 device=cpu backend=cpu runtime=~60s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

x = lt.tensor([1.5, -2.5, 3.0])
print("float dtype:", x.float().dtype, x.float().to_vector())
print("double dtype:", x.double().dtype)
print("half dtype:", x.half().dtype)
print("int dtype:", x.int().dtype, x.int().to_vector())
print("long dtype:", x.long().dtype, x.long().to_vector())
print("cast INT64 dtype:", x.cast(lt.DataType.INT64).dtype, x.cast(lt.DataType.INT64).to_vector())
xt = torch.tensor([1.5, -2.5, 3.0])
print("torch long dtype:", xt.long().dtype, xt.long().tolist())
print("torch int dtype:", xt.int().dtype, xt.int().tolist())
assert x.int().to_vector() == [float(v) for v in xt.int().tolist()]
assert x.long().to_vector() == [float(v) for v in xt.long().tolist()]
assert x.cast(lt.DataType.INT64).to_vector() == [float(v) for v in xt.long().tolist()]
print("VALUES PARITY OK; long dtype differs on this build (see text)")
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. Values match torch exactly; the `.long()` dtype does not (see below).

Expected output:

```
float dtype: DataType.FP32 [1.5, -2.5, 3.0]
double dtype: DataType.FP64
half dtype: DataType.FP16
int dtype: DataType.INT32 [1.0, -2.0, 3.0]
long dtype: DataType.INT32 [1.0, -2.0, 3.0]
cast INT64 dtype: DataType.INT64 [1.0, -2.0, 3.0]
torch long dtype: torch.int64 [1, -2, 3]
torch int dtype: torch.int32 [1, -2, 3]
VALUES PARITY OK; long dtype differs on this build (see text)
```

**D-06 drift, read carefully.** D-06 (`.long()` returned INT32) is FIXED on agent-v1 — the binding now maps `long` to `Tensor::to_long` (INT64), merged as `89afd67` and present in integration HEAD `b326f29` (`src/bindings/bind_tpu.cpp:155`). But the runnable extension used for this batch is built from `88a4ba1`, which predates the fix, so `.long()` still returns INT32 here. The values are correct in both cases; only the dtype differs. Until the extension is rebuilt from a post-`89afd67` tree, use `.cast(lt.DataType.INT64)` for INT64 — it is correct on this build and on the fixed tree. There is no `.bool()`, `.byte()`, `.char()`, `.short()`, or `.to_long()` method (D-13); `.cast()` is the general path.

## 4. Moving data to the host: `.numpy()`, `.item()`, `.tolist()`

### 4.1 `.numpy()` dtype limits (D-14)

`.numpy()` supports FP64, FP32, FP16, INT32, INT16, INT8. UINT8, INT64, and BOOL throw `unsupported dtype for copy conversion`. BF16 is silently cast to FP32 first. Non-CPU tensors throw and must be moved with `.cpu()` first.

<!-- ex: id=E1-4 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"

f = lt.tensor([1.0, 2.0, 3.0])
print("FP32 numpy:", f.numpy(), f.numpy().dtype)
i64 = lt.tensor([1, 2, 3]).cast(lt.DataType.INT64)
try:
    i64.numpy()
    print("INT64 numpy: OK")
except RuntimeError as e:
    print("INT64 numpy throws:", e)
u8 = lt.tensor([1, 2, 3]).cast(lt.DataType.UINT8)
try:
    u8.numpy()
    print("UINT8 numpy: OK")
except RuntimeError as e:
    print("UINT8 numpy throws:", e)
bl = lt.tensor([1, 2, 3]).cast(lt.DataType.BOOL)
try:
    bl.numpy()
    print("BOOL numpy: OK")
except RuntimeError as e:
    print("BOOL numpy throws:", e)
bf = lt.tensor([1.0, 2.0]).cast(lt.DataType.BF16)
print("BF16 numpy:", bf.numpy(), bf.numpy().dtype)
i32 = lt.tensor([1, 2, 3]).cast(lt.DataType.INT32)
print("INT32 numpy:", i32.numpy(), i32.numpy().dtype)
```

Verified on CPU, LiteTorch 88a4ba1. No torch parity: this is a behavioral difference, documented here rather than hidden.

Expected output:

```
FP32 numpy: [1. 2. 3.] float32
INT64 numpy throws: [litetorch Error] numpy(): unsupported dtype for copy conversion
UINT8 numpy throws: [litetorch Error] numpy(): unsupported dtype for copy conversion
BOOL numpy throws: [litetorch Error] numpy(): unsupported dtype for copy conversion
BF16 numpy: [1. 2.] float32
INT32 numpy: [1 2 3] int32
```

Migration: cast to FP32 first (precision caveat for int64 above 2^24 — see §4.2), or read values with `.to_vector()` / `.tolist()`. Note the interplay with D-06: `.long().numpy()` throws on this build and on the fixed tree alike, because the throw is about INT64, not about `.long()`.

### 4.2 `.item()` and `.tolist()` precision (D-18)

`.item()` returns a Python `float`, casting non-FP32 tensors to FP32 first. `.to_vector()` and `.tolist()` do the same. An int64 value above 2^53 therefore comes back rounded, silently. PyTorch `.item()` on an int64 tensor returns an exact Python `int`.

<!-- ex: id=E1-5 device=cpu backend=cpu runtime=~60s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

cpu = lt.Device("cpu")
big = lt.Tensor.from_vector([2**60 + 1], [1], cpu, False, lt.DataType.INT64)
print("lt item of 2**60+1:", big.item())
print("lt tolist of 2**60+1:", big.tolist())
print("lt to_vector of 2**60+1:", big.to_vector())
print("lt lost precision:", big.item() == float(2**60))
tb = torch.tensor(2**60 + 1)
print("torch item of 2**60+1:", tb.item())
print("exact value:", 2**60 + 1)
small = lt.tensor([7]).cast(lt.DataType.INT64)
print("lt item of 7:", small.item())
assert big.item() == float(2**60)
assert tb.item() == 2**60 + 1
print("DIVERGENCE CONFIRMED")
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. The divergence is the point: LiteTorch rounds, torch does not.

Expected output:

```
lt item of 2**60+1: 1.152921504606847e+18
lt tolist of 2**60+1: [1.152921504606847e+18]
lt to_vector of 2**60+1: [1.152921504606847e+18]
lt lost precision: True
torch item of 2**60+1: 1152921504606846977
exact value: 1152921504606846977
lt item of 7: 7.0
DIVERGENCE CONFIRMED
```

The rounding happens twice over: `from_vector` passes values through `float` at construction, and `.item()` returns `float`. Small integers are exact, so the bug only bites above 2^53. Never use `.item()` for large int64 values — indices, token ids near the boundary, hash values. Keep them in tensor form.

## 5. The device model

A `Device` names a backend and an index: `cpu:0`, `gpu:0`, `tpu:0`, `meta`. LiteTorch has no string form for `.to()` (D-03): construct `lt.Device("cuda")` explicitly. Two silent behaviors follow from the device model, and both are traps.

**D-02 — silent CPU fallback.** `.to(Device("cuda"))` on a machine without a GPU backend silently returns a CPU tensor. No error, no warning, no log. `.cuda()` goes through the same path. PyTorch raises.

**D-03 — malformed device strings become CPU.** `Device("cude")` — a typo — parses as CPU, silently. PyTorch raises on an invalid device string.

<!-- ex: id=E1-6 device=cpu backend=cpu runtime=~90s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

print("GPU available:", lt.is_gpu_available())
print("TPU available:", lt.is_tpu_available())
t = lt.tensor([1.0, 2.0])
g = t.to(lt.Device("cuda"))
print("after to(cuda):", g.device)
typo = t.to(lt.Device("cude"))
print("after to(Device('cude')):", typo.device)
c = t.cpu()
print("after .cpu():", c.device)
assert str(g.device) == "cpu:0"
assert str(typo.device) == "cpu:0"
print("SILENT FALLBACK CONFIRMED")
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. No torch parity: torch raises where LiteTorch silently falls back.

Expected output (stdout; the `[litetorch Warning] GPU acceleration (OpenCL) is not available` line goes to stderr):

```
GPU available: False
TPU available: False
after to(cuda): cpu:0
after to(Device('cude')): cpu:0
after .cpu(): cpu:0
SILENT FALLBACK CONFIRMED
```

The migration is a one-line discipline: always check `t.device` after `.to()` / `.cuda()` and fail loudly if it is not the requested device. Never assume a tensor is on the GPU because the code asked for the GPU. The same fallback applies to TPU, and `Module.to()` inherits it per parameter. Binary ops across devices also silently transfer instead of raising (D-10) — move tensors explicitly before mixing devices.

<!-- ex: id=E1-9 device=gpu backend=cuda runtime=~? torch-parity=n/a status=verified-2026-10-31@cbba5af3 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"

assert lt.is_gpu_available()
t = lt.tensor([1.0, 2.0]).to(lt.Device("cuda"))
assert str(t.device) == "gpu:0"
print("on GPU:", t.device)
```

**REQUIRES-GPU — not run on CPU, never shipped as verified.** The block above is the GPU counterpart of E1-6: on a machine with a real CUDA backend the same `.to(Device("cuda"))` must land on `gpu:0`, and the device check must pass. It runs on Kaggle (QA owns the session) after the pre-flight gate asserts the `Native GPU (CUDA/HIP)` init line and `liblitetorch_gpu.so` placement.

## 6. Raw pointers: `data_ptr()` (D-01)

In C++, `Tensor::data_ptr()` returns a `float*` and throws on any dtype that is not FP32: `[litetorch Error] data_ptr(): tensor dtype is not FP32; use raw_data_ptr() for byte-level access` (`src/tensor/tensor_core.cpp:1292-1297`). `raw_data_ptr()` returns `void*` for any dtype (`tensor_core.cpp:1299-1303`). It also returns a CPU pointer (`storage->get_cpu_ptr() + offset`) even for GPU tensors — never a device pointer. PyTorch `Tensor.data_ptr()` works for every dtype.

Neither function is bound in Python on this build:

<!-- ex: id=E1-7 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"

t = lt.tensor([1.0, 2.0, 3.0])
print("has data_ptr:", hasattr(t, "data_ptr"))
print("has raw_data_ptr:", hasattr(t, "raw_data_ptr"))
print("host values via to_vector:", t.to_vector())
print("host values via numpy:", t.numpy().tolist())
assert not hasattr(t, "data_ptr")
assert not hasattr(t, "raw_data_ptr")
print("PYTHON HAS NEITHER BINDING; C++ data_ptr() throws on non-FP32 (see text)")
```

Verified on CPU, LiteTorch 88a4ba1. No torch parity: the binding absence is itself the finding.

Expected output:

```
has data_ptr: False
has raw_data_ptr: False
host values via to_vector: [1.0, 2.0, 3.0]
host values via numpy: [1.0, 2.0, 3.0]
PYTHON HAS NEITHER BINDING; C++ data_ptr() throws on non-FP32 (see text)
```

From Python, host access goes through `.numpy()` (subject to the D-14 dtype limits) or `.to_vector()` / `.tolist()` (subject to the D-18 float rounding). Ported code that calls `t.data_ptr()` on an int, bool, FP16, or FP64 tensor — in C++ or through a future binding — must migrate to `raw_data_ptr()` and cast to the element type itself.

## 7. The allocator and the caching allocator

LiteTorch allocates through a caching allocator: freed blocks are kept in a cache keyed by size and reused by the next allocation of the same size, instead of returning to the OS. `lt.get_cached_cpu_bytes()` reports how many bytes are currently sitting in the CPU cache; `lt.empty_cache()` releases them; `lt.set_max_cpu_cache_size(n)` caps the cache. The same pattern exists on GPU (`get_cached_gpu_bytes()`).

<!-- ex: id=E1-8 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"

lt.manual_seed(0)
print("cached bytes before:", lt.get_cached_cpu_bytes())
a = lt.randn([1000, 1000])
print("cached bytes while alive:", lt.get_cached_cpu_bytes())
del a
print("cached bytes after del:", lt.get_cached_cpu_bytes())
lt.empty_cache()
print("cached bytes after empty_cache:", lt.get_cached_cpu_bytes())
lt.set_max_cpu_cache_size(1024 * 1024)
print("set_max_cpu_cache_size ok")
assert lt.get_cached_cpu_bytes() == 0
print("CACHING ALLOCATOR OK")
```

Verified on CPU, LiteTorch 88a4ba1. No torch parity: this observes LiteTorch's own allocator, not a torch operation.

Expected output:

```
cached bytes before: 0
cached bytes while alive: 0
cached bytes after del: 4000000
cached bytes after empty_cache: 0
set_max_cpu_cache_size ok
CACHING ALLOCATOR OK
```

Read the numbers carefully: the cache is 0 while the 4,000,000-byte tensor is alive — the bytes are in use, not cached. `del` frees the tensor, and the block moves into the cache instead of returning to the OS: 4,000,000 cached bytes. `empty_cache()` releases them. That is the intended behavior, and it is why "memory still allocated after `del`" is usually the cache, not a leak. Two consequences follow. First, peak memory is driven by the largest live set, not by the total ever allocated — reuse keeps the footprint flat across training steps. Second, memory eviction (LiteTorch's differentiator: offloading tensors under pressure) interacts with the cache — evicted tensors free their blocks into the cache, where a later allocation of the same size reuses them without touching the OS.

## 8. What this page does not cover

- Autograd mechanics — see [E2 Autograd](E2-autograd.md).
- The full dtype promotion table, including the FP32 promotion of quantized dtypes (D-19) and the GPU integer-through-FP32 path (D-21) — see `plan/differences.md` and the API reference.
- The op surface itself — see `reference/tensor.md` and `reference/ops-elementwise.md`.
