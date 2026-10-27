# Ops — Elementwise and Math — API Reference

`litetorch.Ops.*` functions: elementwise arithmetic, math, trigonometry, activations, comparisons, matmul, and casting. Tensor methods exist only for a few of these (`log_softmax`, `silu`); the rest are `Ops.*` only (D-17).

- **Source of truth:** `src/bindings/bind_ops.cpp:5-178` (bindings), `include/litetorch/ops.h` (C++ declarations).
- **Tree:** agent-v1@02cc271a / local HEAD 88a4ba1 (docs baseline; B5 re-verifies at head).
- **Conventions:** `cpu = lt.Device("cpu")` in every example. Torch parity is FP32 1e-4 unless the header says otherwise.
- **Related deviations:** D-08 (int scalar promotes to FP32), D-09 (GELU tanh-only), D-10 (cross-device silent transfer), D-17 (Ops-only, no Tensor methods), D-21 (GPU integer math via FP32).

## Arithmetic

All four binary ops broadcast like torch, promote dtypes via `promote_types` (elementwise promotion is torch-correct — see differences.md, D-07 correction), and silently transfer a mismatched operand to the other operand's device (D-10) instead of raising.

### `Ops.add` / `Ops.sub` / `Ops.mul` / `Ops.div`

- **Signatures:** `add(a, b) -> Tensor`; `sub(a, b)`; `mul(a, b)`; `div(a, b)`
- **Source:** `src/bindings/bind_ops.cpp:5-8`
- **Parameters:** `a`, `b` — tensors (broadcastable) or broadcastable shapes.
- **Returns:** elementwise result; integer division is true division (FP result) when the promoted dtype is float.
- **Edge cases:** division by zero yields `inf`/`nan` following IEEE rules for floats; integer `div` on GPU runs through FP32 (D-21).
- **PyTorch:** `torch.add(a, b)` / `a + b` etc. — equivalent except D-10/D-21.

<!-- ex: id=r3-arith device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
b = lt.Tensor.from_vector([10.0, 20.0, 30.0, 40.0], [2, 2], cpu)
print(lt.Ops.add(a, b).to_vector())
print(lt.Ops.sub(b, a).to_vector())
print(lt.Ops.mul(a, b).to_vector())
print(lt.Ops.div(b, a).to_vector())
```

### `Ops.pow`

- **Signature:** `pow(a, exponent) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:19`; C++ `include/litetorch/ops.h:35` (`pow(Tensor, float)`)
- **Parameters:** `exponent` — a float scalar only. A two-tensor `pow(a, b)` is forward-declared but never defined or bound (D-05).
- **PyTorch:** `torch.pow(a, exp)` / `a ** exp` / `a ** u` — differs: torch accepts a tensor exponent.

<!-- ex: id=r3-pow device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 3.0], [2], cpu)
print(lt.Ops.pow(a, 2.0).to_vector(), (a ** 2.0).to_vector())
```

### `Ops.neg` / `Ops.abs`

- **Signatures:** `neg(a) -> Tensor`; `abs(a) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:23-24`
- **PyTorch:** `torch.neg(a)` / `-a`; `torch.abs(a)` — equivalent.

### `Ops.sqrt` / `Ops.rsqrt` / `Ops.reciprocal`

- **Signatures:** `sqrt(a) -> Tensor`; `rsqrt(a) -> Tensor` (`1/sqrt(a)`); `reciprocal(a) -> Tensor` (`1/a`)
- **Source:** `src/bindings/bind_ops.cpp:20, 96, 98`
- **Edge cases:** negative inputs give `nan` (sqrt); zero gives `inf` (rsqrt, reciprocal).
- **PyTorch:** `torch.sqrt` / `torch.rsqrt` / `torch.reciprocal` — equivalent.

<!-- ex: id=r3-sqrt device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([4.0, 9.0, 16.0], [3], cpu)
print(lt.Ops.sqrt(a).to_vector(), lt.Ops.rsqrt(a).to_vector())
print(lt.Ops.reciprocal(a).to_vector(), lt.Ops.neg(a).to_vector())
print(lt.Ops.abs(lt.Tensor.from_vector([-1.0, 2.0], [2], cpu)).to_vector())
```

### `Ops.exp` / `Ops.log` / `Ops.log2` / `Ops.log10`

- **Signatures:** `exp(a)`; `log(a)` (natural); `log2(a)`; `log10(a)`
- **Source:** `src/bindings/bind_ops.cpp:21-22, 94-95`
- **Edge cases:** `log` of 0 gives `-inf`; of a negative gives `nan`.
- **PyTorch:** `torch.exp` / `torch.log` / `torch.log2` / `torch.log10` — equivalent.

<!-- ex: id=r3-exp-log device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 10.0, 100.0], [3], cpu)
print(lt.Ops.exp(lt.Tensor.from_vector([0.0, 1.0], [2], cpu)).to_vector())
print(lt.Ops.log(a).to_vector(), lt.Ops.log2(a).to_vector(), lt.Ops.log10(a).to_vector())
```

### `Ops.sin` / `Ops.cos` / `Ops.tan`

- **Signatures:** `sin(a)`; `cos(a)`; `tan(a)` (radians)
- **Source:** `src/bindings/bind_ops.cpp:26-27, 97`
- **PyTorch:** `torch.sin` / `torch.cos` / `torch.tan` — equivalent.

### `Ops.sign` / `Ops.floor` / `Ops.ceil` / `Ops.round` / `Ops.trunc`

- **Signatures:** `sign(a)` (-1/0/1); `floor(a)`; `ceil(a)`; `round(a)` (half to even); `trunc(a)` (toward zero)
- **Source:** `src/bindings/bind_ops.cpp:99-103`
- **PyTorch:** `torch.sign` / `torch.floor` / `torch.ceil` / `torch.round` / `torch.trunc` — equivalent.

<!-- ex: id=r3-trig-round device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([0.0, 1.5707963, 3.1415926], [3], cpu)
print(lt.Ops.sin(a).to_vector(), lt.Ops.cos(a).to_vector(), lt.Ops.tan(a).to_vector())
b = lt.Tensor.from_vector([-1.5, 2.5, 0.0], [3], cpu)
print(lt.Ops.sign(b).to_vector(), lt.Ops.floor(b).to_vector())
print(lt.Ops.ceil(b).to_vector(), lt.Ops.round(b).to_vector(), lt.Ops.trunc(b).to_vector())
```

### `Ops.clamp` / `Ops.clamp_min` / `Ops.clamp_max`

- **Signatures:** `clamp(a, min=None, max=None) -> Tensor`; `clamp_min(a, min) -> Tensor`; `clamp_max(a, max) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:25, 104-105` (second `clamp` overload at :106: both-None raises)
- **Parameters:** scalar float bounds only (D-16: no tensor min/max, unlike torch).
- **Edge cases:** `clamp(a)` with neither bound raises `clamp: at least one of min or max must be given`.
- **PyTorch:** `torch.clamp(a, min, max)` — differs: torch accepts tensor bounds.

<!-- ex: id=r3-clamp device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([-2.0, 0.5, 3.0], [3], cpu)
print(lt.Ops.clamp(a, 0.0, 2.0).to_vector())
print(lt.Ops.clamp_min(a, 0.0).to_vector(), lt.Ops.clamp_max(a, 1.0).to_vector())
```

## Activations

### `Ops.relu`

- **Signature:** `relu(a) -> Tensor` (`max(0, x)`)
- **Source:** `src/bindings/bind_ops.cpp:28`
- **PyTorch:** `torch.relu(a)` — equivalent.

### `Ops.leaky_relu`

- **Signature:** `leaky_relu(a, negative_slope=0.01) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:29`
- **PyTorch:** `torch.nn.functional.leaky_relu(a, negative_slope)` — equivalent.

### `Ops.sigmoid` / `Ops.tanh`

- **Signatures:** `sigmoid(a) -> Tensor`; `tanh(a) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:30-31`
- **PyTorch:** `torch.sigmoid(a)` / `torch.tanh(a)` — equivalent.

### `Ops.gelu`

- **Signature:** `gelu(a) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:33`
- **Behavior:** tanh approximation ONLY, forward and backward: `0.5x(1+tanh(0.79788456(x+0.044715x^3)))` (D-09). No `approximate` argument. Max difference vs torch's default (exact erf) is ~4.4e-4 — parity checks must use tolerance >= 5e-4 or compare against `F.gelu(x, approximate='tanh')`.
- **PyTorch:** `torch.nn.functional.gelu(a)` (exact erf by default) — differs by up to ~4.4e-4.

### `Ops.silu`

- **Signature:** `silu(a) -> Tensor` (`x * sigmoid(x)`); also a `Tensor.silu()` method.
- **Source:** `src/bindings/bind_ops.cpp:66`
- **PyTorch:** `torch.nn.functional.silu(a)` — equivalent.

<!-- ex: id=r3-activations device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([-2.0, -0.5, 0.0, 0.5, 2.0], [5], cpu)
print(lt.Ops.relu(a).to_vector())
print(lt.Ops.leaky_relu(a, 0.1).to_vector())
print(lt.Ops.sigmoid(a).to_vector(), lt.Ops.tanh(a).to_vector())
print(lt.Ops.gelu(a).to_vector(), lt.Ops.silu(a).to_vector())
```

### `Ops.softmax` / `Ops.log_softmax`

- **Signatures:** `softmax(a, dim=-1) -> Tensor`; `log_softmax(a, dim=-1) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:32, 65`
- **PyTorch:** `torch.softmax(a, dim)` / `torch.log_softmax(a, dim)` — equivalent.

<!-- ex: id=r3-softmax device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu)
print(lt.Ops.softmax(a).to_vector(), lt.Ops.log_softmax(a).to_vector())
```

## Comparisons and selection

### `Ops.eq` / `ne` / `lt` / `gt` / `le` / `ge`

- **Signatures:** `eq(a, b) -> Tensor` (and `ne`, `lt`, `gt`, `le`, `ge`)
- **Source:** `src/bindings/bind_ops.cpp:58-63`
- **Returns:** boolean tensor, broadcast like the arithmetic ops (D-10 cross-device transfer applies).
- **PyTorch:** `torch.eq(a, b)` etc. — equivalent.

### `Ops.where`

- **Signature:** `where(cond, a, b) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:64`
- **Parameters:** `cond` — boolean tensor; `a`, `b` — tensors broadcastable with `cond`.
- **Returns:** `a` where `cond` is true, else `b`.
- **PyTorch:** `torch.where(cond, a, b)` — equivalent.

<!-- ex: id=r3-compare device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu)
b = lt.Tensor.from_vector([1.5, 1.5, 1.5], [3], cpu)
print(lt.Ops.gt(a, b).to_vector(), lt.Ops.eq(a, b).to_vector())
print(lt.Ops.where(lt.Ops.gt(a, b), a, b).to_vector())
```

## Matmul

### `Ops.matmul`

- **Signature:** `matmul(a, b) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:9`; C++ `include/litetorch/ops.h:18`
- **Behavior:** 2-D matrix multiply; batched inputs follow torch broadcasting rules for the batch dims.
- **PyTorch:** `torch.matmul(a, b)` — equivalent.

### `Ops.matmul_bias`

- **Signature:** `matmul_bias(a, b, bias) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:10`
- **Behavior:** `a @ b + bias` fused in one call.
- **PyTorch:** `torch.addmm(bias, a, b)` for 2-D — equivalent in value.

### `Ops.bmm`

- **Signature:** `bmm(a, b) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:11`
- **Behavior:** batch matrix multiply on 3-D tensors `(B, N, M) @ (B, M, P) -> (B, N, P)`.
- **PyTorch:** `torch.bmm(a, b)` — equivalent.

<!-- ex: id=r3-matmul device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
b = lt.Tensor.from_vector([5.0, 6.0, 7.0, 8.0], [2, 2], cpu)
print(lt.Ops.matmul(a, b).to_vector())
print(lt.Ops.matmul_bias(a, b, lt.Tensor.from_vector([1.0, 1.0], [2], cpu)).to_vector())
print(lt.Ops.bmm(a.reshape([1, 2, 2]), b.reshape([1, 2, 2])).to_vector())
```

## Casting and comparison helpers

### `Ops.cast`

- **Signature:** `cast(a, target_dtype) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:67`
- **PyTorch:** `a.to(dtype)` — equivalent.

### `Ops.allclose` / `Ops.isclose`

- **Signatures:** `allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False) -> bool`; `isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:127-130`
- **Behavior:** elementwise `|a - b| <= atol + rtol * |b|`; `allclose` reduces to a single bool, `isclose` returns the boolean tensor.
- **PyTorch:** `torch.allclose(a, b, rtol, atol, equal_nan)` / `torch.isclose(...)` — equivalent (note torch defaults are rtol=1e-5, atol=1e-8 — same).

### `Ops.clip_grad_norm_`

- **Signature:** `clip_grad_norm_(params, max_norm, norm_type=2.0) -> float`
- **Source:** `src/bindings/bind_ops.cpp:79`
- **Parameters:** `params` — list of tensors with `.grad`; `max_norm` — clipping threshold.
- **Returns:** the total norm before clipping. Gradients are scaled in place when the norm exceeds `max_norm`.
- **PyTorch:** `torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type)` — equivalent.

<!-- ex: id=r3-cast-close device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
print(lt.Ops.cast(a, lt.DataType.FP64).dtype)
print(lt.Ops.allclose(a, a), lt.Ops.isclose(a, a).to_vector())
p = lt.Tensor.from_vector([3.0, 4.0], [2], cpu, True)
p.grad = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print(lt.Ops.clip_grad_norm_([p], 1.0))
print(p.grad.to_vector())
```
