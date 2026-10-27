# How-To H8 — Debug: Gradcheck, Parity vs Torch, OOM Diagnosis

Three debugging skills, in the order you reach for them: prove the
gradients are right (`gradcheck`), prove the numbers match PyTorch
(parity), and prove the memory failure is real (OOM diagnosis). All
examples below run on CPU against LiteTorch 88a4ba1.

Prerequisites: LiteTorch installed and importable. The parity examples
import torch — run them with `~/torch-venv` (torch 2.14.1+cpu); the
system pip has no torch.

> PyTorch differences touched here: D-02 (`.to(device)` silently falls
> back to CPU when the accelerator is unavailable), D-09 (GELU is
> tanh-approximation only), D-12 (autograd retains grads on leaf tensors
> — see H8.3).

## H8.1 — Gradcheck: prove a gradient is correct

`lt.autograd.gradcheck(func, inputs, eps, atol, rtol)` compares the
analytic gradient against a finite-difference gradient. On FP32, use
`eps=1e-3` (or `1e-4`): the default `eps=1e-6` is too small for FP32
rounding and fails a correct gradient. Pass `raise_exception=True` so
a failure raises instead of returning `False`.

<!-- ex: id=H8.1-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

x = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], device, True)


def quad(t):
    return lt.Ops.sum(t * t)


ok = lt.autograd.gradcheck(quad, [x], eps=1e-3)
print("gradcheck passed:", ok)
loss = quad(x)
loss.backward()
print("analytic grad:", x.grad.to_vector())
```

Verified on CPU, LiteTorch 88a4ba1, torch parity 1e-4
(`torch.autograd.gradcheck` on the same function returns `True`).
Expected output: `gradcheck passed: True`, analytic grad
`[2.0, 4.0, 6.0]`. If gradcheck fails, the usual suspects are: a
custom op whose backward does not match its forward, a reduction that
drops a broadcast term, or an in-place op that overwrote a saved
tensor — check them in that order.

## H8.2 — Parity vs torch: the five-line comparison pattern

The reference pattern for every numeric claim in this documentation:
same seed, same input, same op, max absolute difference, and the
gradients compared too — not just the forward values.

<!-- ex: id=H8.2-1 device=cpu backend=cpu runtime=~5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import torch

import litetorch as lt

torch.manual_seed(0)
lt.manual_seed(0)

x_lt = lt.Tensor.from_vector([0.5, -1.0, 2.0, -0.25], [2, 2], lt.auto_device(), True)
x_pt = torch.tensor([[0.5, -1.0], [2.0, -0.25]], dtype=torch.float32, requires_grad=True)

lin_lt = lt.nn.Linear(2, 2, True)
lin_lt.to(lt.auto_device())
w_lt = lin_lt.weight
b_lt = lin_lt.bias
lin_pt = torch.nn.Linear(2, 2, dtype=torch.float32)
with torch.no_grad():
    lin_pt.weight.copy_(torch.tensor(w_lt.to_vector()).reshape(2, 2))
    lin_pt.bias.copy_(torch.tensor(b_lt.to_vector()))

y_lt = lin_lt(x_lt)
y_pt = lin_pt(x_pt)
loss_lt = lt.Ops.sum(y_lt)
loss_pt = y_pt.sum()
loss_lt.backward()
loss_pt.backward()

fwd_diff = max(abs(a - b) for a, b in zip(y_lt.to_vector(), y_pt.detach().flatten().tolist()))
grad_diff = max(abs(a - b) for a, b in zip(x_lt.grad.to_vector(), x_pt.grad.flatten().tolist()))
print("forward max abs diff:", fwd_diff)
print("grad max abs diff:", grad_diff)
```

Verified on CPU, LiteTorch 88a4ba1, torch 2.14.1+cpu. Expected output:
both diffs at or below 1e-4 — in practice exactly `0.0` for a linear
layer. The tolerance is 1e-4 for FP32, 1e-2 for FP16/BF16. A diff above
tolerance is a real divergence until proven otherwise: do not widen
the tolerance to make it pass.

## H8.3 — OOM diagnosis and the silent CPU fallback check

Out-of-memory and wrong-device failures look alike from the loss
curve, so check the device first, then the memory. This example runs
the three checks that settle it: is CUDA actually available, did the
tensor land where you asked, and how much CPU cache is held right now.

<!-- ex: id=H8.3-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

print("cuda available:", lt.cuda.is_available())
print("cuda devices:", lt.cuda.device_count())

device = lt.Device("cuda")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], device, False)
print("requested cuda, got:", t.device)
print("fallback to CPU:", t.device.type == lt.DeviceType.CPU)

cpu_bytes = lt.get_cached_cpu_bytes()
print("cached CPU bytes at startup:", cpu_bytes)
scratch = lt.Tensor.from_vector([1.0] * 1000000, [1000, 1000], lt.auto_device(), False)
del scratch
print("cached CPU bytes before empty_cache:", lt.get_cached_cpu_bytes())
lt.empty_cache()
print("cached CPU bytes after empty_cache:", lt.get_cached_cpu_bytes())
```

Verified on CPU, LiteTorch 88a4ba1. Expected output on a CPU-only box:
`cuda available: False`, `cuda devices: 0`, the tensor's device prints
`cpu:0` even though `cuda` was requested, and the cached-bytes count
drops from 4000000 (the freed 4 MB scratch tensor, still held by the
allocator) to 0 after `empty_cache()`. That silent fallback is D-02: if
your training run is slow but not crashing, this is the first thing
to check — you may be on CPU without knowing it.

When a real OOM hits, the knobs are: `lt.empty_cache()` (release the
caching allocator's held blocks), `lt.set_max_cpu_cache_size(bytes)`
(cap the CPU cache so it cannot grow past a budget), and
`lt.get_gpu_memory_used()` (bytes currently held on GPU). Reduce the
batch size or enable gradient checkpointing before raising the cache
cap — a bigger cache only moves the failure to a bigger tensor.
