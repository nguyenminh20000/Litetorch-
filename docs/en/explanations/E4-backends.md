# E4 — Backends

LiteTorch runs on five backends — CPU, CUDA, ROCm, OpenCL, TPU — behind one `Tensor` API. This page describes how the framework picks a backend, how kernels are dispatched, and where it silently does something different from PyTorch. All claims are grounded in the tree at HEAD `b326f29` (extension build `88a4ba1`); anything not verified on hardware is marked as such. Cross-references: D-02 / D-03 / D-10 in the Behavioral Differences page.

## 1. Capability matrix

| Backend | How it is reached | Status on this machine (CPU-only, build 88a4ba1) | Verified |
|---|---|---|---|
| CPU | Always available; the fallback for everything | `auto_device()` → `cpu:0`, `get_backend_name()` → `CPU` | Yes, CPU harness (Examples E4-1–E4-3) |
| CUDA | Native GPU library `liblitetorch_gpu.so`, loaded at startup | Absent: `lt.cuda.is_available()` → `False`, `device_count()` → `0` | No — needs Kaggle GPU (Example E4-4 is REQUIRES-GPU) |
| ROCm | Same native library, HIP build (`__HIP_PLATFORM_AMD__` / `__HIPCC__`) | Absent; no ROCm hardware anywhere in this pipeline | **UNVERIFIED-ROCm** — Example E4-5 ships as a callout, not a verified example |
| OpenCL | `CLBackend`, used only when no native GPU exists | Absent: `[litetorch Warning] GPU acceleration (OpenCL) is not available. Falling back to CPU.` | No — needs an OpenCL device |
| TPU | PJRT / libtpu via `TPUBackend` | Absent: `lt.is_tpu_available()` → `False` | No — Kaggle TPU, only after GPU green (standing order) |

The matrix is the point: on a machine with no accelerator, LiteTorch still runs everything on CPU. That is convenient and dangerous in equal measure — see §3.

## 2. Dispatch: how a backend is chosen

**Startup.** `BackendDispatcher`'s constructor (`src/backend/backend.cpp:521-541`) tries, in order:

1. A `NativeGPUBackend`: it `dlopen`s `liblitetorch_gpu.so` from a fixed search list — the extension's own directory first, then `./build/`, `build/`, the current directory, `/tmp/`, `/usr/local/lib/`, `/usr/lib/`, and `/opt/rocm/lib/` + `/opt/rocm/lib64/` (`backend.cpp:165-205`). If the library loads and its symbols resolve, the dispatcher prints the init line of §4 and keeps it. `LITETORCH_NO_GPU` / `LITETORCH_NO_NATIVE_GPU` in the environment disable this step entirely (`backend.cpp:160-162`).
2. A `TPUBackend`: if available it prints `[litetorch Info] Google TPU (Systolic Array) successfully initialized.` and is kept.

There is no configuration file and no priority flag: native GPU wins over TPU by construction order, and OpenCL is not a `BackendDispatcher` backend at all — it is consulted only when the native GPU is absent.

**Per-device routing.** `get_backend_for_device` (`backend.cpp:549-557`) maps TPU → TPU backend, GPU → native GPU backend, anything else → CPU backend. The low-level memory ops in `include/litetorch/backend_route.h` implement the same rule inline: `rt_gpu_allocate` / `rt_gpu_free` / `rt_gpu_read` / `rt_gpu_write` / `rt_gpu_copy` call the native backend when `has_native_gpu()` is true and fall through to `CLBackend` otherwise. In other words, OpenCL is the fallback *implementation* of the GPU path, not a peer backend: a machine with a native GPU never touches OpenCL for tensor memory.

**`auto_device` and `get_backend_name`.** The Python-visible selection (`src/bindings/bind_core.cpp:69-110`) is a fixed priority chain:

```text
auto_device():
    native GPU available (and LITETORCH_NO_NATIVE_GPU unset)  -> gpu:0
    else OpenCL available                                      -> gpu:0
    else TPU available                                         -> TPU:0
    else                                                       -> CPU:0

get_backend_name():
    native GPU  -> "CUDA"  (or "ROCm" on a HIP build)
    OpenCL      -> "OpenCL"
    TPU         -> "TPU (Google PJRT / libtpu)"
    else        -> "CPU"
```

Note that `auto_device()` returns `gpu:0` for OpenCL as well as for native GPU: the `Device` does not record *which* GPU implementation backs it. Code that needs to know must call `get_backend_name()`.

**Kernel dispatch.** A kernel call first asks the native backend for a precompiled kernel by name (`gpu_get_kernel_fn`, `backend.cpp:486-497`); only if none exists does it compile from source. The 2026-10-26 cached-kernel collision fix (merged `agent-v1 @02cc271a`) is part of this path: kernels are now looked up by name, because two byte-identical FP32/FP16 backward typedefs previously shared one cache entry. Kernel launch requires the `gpu_launch` symbol and throws if it is missing (`backend.cpp:503-504`).

## 3. Fallback rules: the silent ones

Three fallbacks are silent — no error, no warning, no log — and all three are live deviations from PyTorch, documented as D-02, D-03, and D-10.

**D-02 — `.to(device)` silently falls back to CPU.** `StorageImpl`'s GPU constructor (`src/tensor/storage.cpp:15-34`): if no GPU backend exists, the storage is built on CPU and `this->device` is rewritten to CPU. `Tensor::to(Device)` inherits this (`src/tensor/tensor_core.cpp:997-1006`), as does `Tensor::cuda()` (`tensor_core.cpp:2125-2128`) and `Module.to()` per parameter (`src/nn/container.cpp:363`). A TPU allocation failure falls back the same way (`storage.cpp:43-48`), while a GPU out-of-memory *throws*. PyTorch raises on `.to("cuda")` without CUDA. Example E4-2 shows the guard every ported program needs: check `t.device` after `.to()` and fail loudly.

**D-03 — malformed device strings silently become CPU.** `Device("...")` parses `cuda` / `gpu` / `tpu` / `meta` (case-insensitive, optional `:index`) and returns CPU for *anything* else, including typos such as `Device("cude")` (`src/bindings/bind_core.cpp:23-43`). And `.to()` takes no string form at all: `Tensor.to` is bound only for `Device`, `DataType`, and `(Device, DataType)` (`src/bindings/bind_tpu.cpp:73-75`), so `t.to("cuda")` raises a pybind `TypeError` while `t.to(Device("cude"))` silently runs on CPU. Example E4-3 demonstrates both.

**D-10 — binary ops across devices silently transfer.** `add` / `sub` / `mul` / `div` move the mismatched operand to the other operand's device (`src/ops/elementwise.cpp:583, 723, 874, 1015`) instead of raising as PyTorch does. Combined with D-02, a stray CPU tensor can silently pin a whole computation to CPU.

The remaining fallback is loud, and it is the one to rely on: **GPU out-of-memory throws.** A failed GPU allocation raises `GPU out of memory` with the `MemoryManager` diagnostics string (E3 §4, §6). Only the *absence* of a device is silent; the *exhaustion* of a present device is not.

## 4. The pre-flight gate: the `Native GPU (CUDA/HIP)` init line

When the native backend initializes, the dispatcher prints exactly:

```text
[litetorch Info] Native GPU (CUDA/HIP) successfully initialized.
```

(`src/backend/backend.cpp:525`.) The 2026-10-18 maximum-scrutiny mandate makes this line a **pre-flight gate** for every GPU run: before any GPU example runs on Kaggle, the session must assert (a) this line is present and says CUDA/HIP, *not* OpenCL; (b) `liblitetorch_gpu.so` sits next to the extension under `<tree>/python/` (the `/tmp` copy is wiped on session restart); (c) `lt.cuda.device_count()` returns the expected count (2× T4 on the Kaggle GPU image). If any assertion fails, the run stops and the failure is reported as a harness/blocker issue, not as example failures. Absence of the line with a silent CPU fallback is the failure mode the gate exists to catch: without it, a GPU run would execute on CPU, pass its numeric checks slowly, and report success. Example E4-4 is the gate as code; it is REQUIRES-GPU and unverified on this machine.

## 5. NCCL: multi-GPU collectives

Multi-GPU collectives go through `NCCLBridge` (`src/distributed/distributed.cpp:1083-1296`): `init`, `all_reduce`, `broadcast`, `all_gather`, `reduce_scatter`, with `nccl_check` throwing `nccl <op> failed` on a nonzero return code. The bridge resolves 10 symbols by `dlsym` and requires 7 of them (`ncclGetUniqueId`, `ncclCommInitRank`, `ncclAllReduce`, `ncclBroadcast`, `ncclAllGather`, `ncclReduceScatter`, `ncclCommDestroy`); if any is missing it closes the library and reports unavailable (`distributed.cpp:1115-1132`).

> **DRIFT — read this before using the NCCL section.** The task brief for this page described an `LITETORCH_NCCL_LIB` environment override with fail-closed semantics, probing `pip nvidia/nccl`, then `/usr/local/cuda/lib64`, then system paths, preferring the highest NCCL 2.x and rejecting non-2.x. **That selection logic does not exist in the tree at HEAD `b326f29`.** Verified by search: `LITETORCH_NCCL_LIB` occurs nowhere in `integration/`; `distributed.cpp:1104-1108` simply does `dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL)`, falling back to `dlopen("librccl.so", ...)` — the system loader's search path only, no version check, no environment override. Failure to load is also **fail-soft**, not fail-closed: `available_` is set to `false` and collectives report unavailable (`distributed.cpp:1109-1113`). **Status, settled 2026-10-27 by dist-w22: Kaggle-verified patch, not yet merged.** The selection mechanism (dist-w15) plus the whole 2-GPU round (w10–w13) were leader/commander-approved and Kaggle-verified 2026-10-19 on a patched session tree, but no merge step ever ran — `git log --all -S "LITETORCH_NCCL_LIB"` is empty, and the five patch files sit unmerged in `patches/distributed/`. The commander has authorized resubmission of w10→w15 as one ordered set for QA merge review; until that set actually merges, the current behavior is fail-soft dlopen and this page documents the code as it is. The Kaggle-verified behavior recorded in memory (2-GPU collectives green, pip NCCL 2.31.2 auto-selected, 2026-10-19) is consistent with the system loader finding pip's `libnccl.so` on the library path, not with an explicit probe order. Do not write code that sets `LITETORCH_NCCL_LIB` expecting it to be honored.

What the tree *does* guarantee, and what a GPU run can rely on: symbol verification before use (a half-present NCCL is treated as absent, not as a crash later), and loud failure at collective time via `nccl_check`. What it does *not* guarantee: which `libnccl.so` the loader finds, or that a missing NCCL fails loudly at startup.

## 6. ROCm

> **UNVERIFIED-ROCm (commander decision).** ROCm examples ship as callouts in this page; the page is covered, not deferred. No ROCm hardware is available locally or on the Kaggle GPU image, so no ROCm claim below is verified by execution.

What the tree establishes about ROCm:

- The native library search list includes `/opt/rocm/lib/liblitetorch_gpu.so` and `/opt/rocm/lib64/liblitetorch_gpu.so` (`src/backend/backend.cpp:172-173`).
- The same `liblitetorch_gpu.so` serves CUDA and ROCm: the init line is `Native GPU (CUDA/HIP)` in both cases, and `get_backend_name()` returns `"ROCm"` only when the extension itself was built with `__HIP_PLATFORM_AMD__` or `__HIPCC__` defined (`src/bindings/bind_core.cpp:96-100`). A CUDA-built extension on a ROCm machine reports `"CUDA"`, not `"ROCm"`.
- The NCCL bridge falls back to `librccl.so` (ROCm's collective library) when `libnccl.so` is absent (`distributed.cpp:1106`).
- The FP32/FP16 kernel-cache collision fix and the `gpu_launch` requirement apply identically on the HIP path, since both go through `NativeGPUBackend`.

What is **not** established: that any kernel actually runs correctly on ROCm, that the ROCm native library builds, or that `device_count()` reports AMD devices. Example E4-5 is the ROCm callout: it states the expected observable (`get_backend_name() == "ROCm"`) and the conditions under which it could hold, and it is marked `backend=rocm`, `status=unverified`. Treat every ROCm sentence in this page as tree-reading, not as a tested claim.

## 7. TPU

The TPU backend is initialized second (`backend.cpp:532-538`) and selected only when no GPU — native or OpenCL — is present (`bind_core.cpp:69-83`). Its Python surface is `lt.is_tpu_available()` and the `Device("tpu")` constructor. Two tree facts constrain TPU use: a TPU allocation failure silently falls back to CPU (E3 §8), and `flash_attention_half` is not implemented on TPU and throws (`src/backend/tpu/tpu_backend.cpp:343`, D-20). TPU examples are REQUIRES-TPU and run on Kaggle only after GPU green, per standing order; none are shipped in this batch.

## Examples

<!-- ex: id=e4-1 device=cpu backend=cpu runtime=~70s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import torch
import litetorch as lt
print("auto device", lt.auto_device())
print("backend name", lt.get_backend_name())
print("gpu available", lt.is_gpu_available())
print("tpu available", lt.is_tpu_available())
print("cuda available", lt.cuda.is_available())
print("cuda device count", lt.cuda.device_count())
print("torch cuda available", torch.cuda.is_available())
```
Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. On this CPU-only machine the whole discovery chain lands on CPU: no native GPU, no OpenCL, no TPU.

<!-- ex: id=e4-2 device=cpu backend=cpu runtime=~80s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import torch
import litetorch as lt
t = lt.randn([2, 2])
moved = t.to(lt.Device("cuda"))
print("requested cuda, got", moved.device)
print("silent fallback", str(moved.device) == "cpu:0")
def require_device(tensor, device):
    if str(tensor.device) != str(device):
        raise RuntimeError("device mismatch")
try:
    require_device(moved, lt.Device("cuda"))
    print("guard passed")
except RuntimeError:
    print("guard raised")
try:
    torch.randn([2, 2]).to("cuda")
    print("torch silent")
except Exception as e:
    print("torch raised", type(e).__name__)
```
Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. LiteTorch silently keeps the tensor on CPU (D-02); the `require_device` guard is the mandatory pattern for ported code. PyTorch raises instead.

<!-- ex: id=e4-3 device=cpu backend=cpu runtime=~10s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
print("cuda", lt.Device("cuda"))
print("CUDA:0", lt.Device("CUDA:0"))
print("cude typo", lt.Device("cude"))
print("tpu", lt.Device("tpu"))
print("meta", lt.Device("meta"))
try:
    lt.randn([2, 2]).to("cuda")
    print("string to ok")
except TypeError:
    print("string to raised TypeError")
```
Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. The typo `Device("cude")` parses to CPU with no error (D-03), while `.to("cuda")` with a string raises `TypeError` because no string overload is bound.

<!-- ex: id=e4-4 device=gpu backend=cuda runtime=~? torch-parity=n/a status=verified-2026-10-31@cbba5af3 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import os
import litetorch as lt
print("cuda available", lt.cuda.is_available())
print("device count", lt.cuda.device_count())
print("backend name", lt.get_backend_name())
print("native lib next to extension", os.path.exists(os.path.join(os.path.dirname(lt.__file__), "liblitetorch_gpu.so")))
lt.cuda.synchronize()
print("gate passed")
```
**REQUIRES-GPU — not verified.** The Kaggle pre-flight gate as code: native backend present, device count as expected, `liblitetorch_gpu.so` beside the extension. The `Native GPU (CUDA/HIP)` init line itself is printed by the C++ dispatcher at import time and is checked in the session log, not from Python. On this CPU-only machine `is_available()` is `False` and the gate fails, as it must.

<!-- ex: id=e4-5 device=gpu backend=rocm runtime=~? torch-parity=n/a status=unverified -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
print("backend name", lt.get_backend_name())
print("cuda available", lt.cuda.is_available())
print("device count", lt.cuda.device_count())
```
**UNVERIFIED-ROCm — not verified, REQUIRES-GPU.** Expected observable on a HIP-built extension with a ROCm device: `get_backend_name()` returns `"ROCm"`, `cuda.is_available()` is `True`, and the dispatcher prints `Native GPU (CUDA/HIP) successfully initialized.` No ROCm hardware exists in this pipeline, so this callout is tree-reading only (§6): it is shipped so the ROCm path is documented, not deferred — and it must not be read as a tested claim.
