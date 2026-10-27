# Ops — Losses, Quantization, FP8, Fused

`litetorch.Ops` loss functions, fake quantization, and the FP8 / INT4 / W8A8 quantized matmul helpers. The quantized helpers are documented in full in `ops-conv-attention.md` and cross-referenced here.

- **Source of truth:** `src/bindings/bind_ops.cpp:76-79,143-154,165-166,176-178` (Python bindings), `include/litetorch/ops.h` (C++ declarations).
- **Tree:** agent-v1@02cc271a / local HEAD 89afd67 (docs baseline; B5 re-verifies at head). No binding drift in `bind_ops.cpp` / `ops.h` between 88a4ba1 and 89afd67.
- **Conventions:** `cpu = lt.Device("cpu")` in every example. Every example is self-contained and runnable on CPU. Torch parity is FP32 1e-4 unless the header says otherwise.
- **Known defect (filed to Core 2026-10-26, thread `docs-fix-requests-20261026`):** `cross_entropy_loss`, `nll_loss`, and `fused_linear_cross_entropy` reject INT64 targets — see the entries below. Workaround: cast targets to INT32.

## Losses

### `Ops.mse_loss`

- **Signature:** `mse_loss(input, target) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:176`; C++ `include/litetorch/ops.h:152`
- **Parameters:** `input`, `target` — same shape.
- **Returns:** scalar `[1]`, `mean((input - target)^2)`. There is no `reduction` argument — the loss is always the mean.
- **PyTorch:** `F.mse_loss(input, target)` (default `reduction="mean"`) — equivalent.

<!-- ex: id=r6-mse-loss device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [4], cpu, False)
y = lt.Tensor.from_vector([1.0, 2.0, 3.0, 5.0], [4], cpu, False)
out = lt.Ops.mse_loss(x, y)
print(list(out.shape), out.to_vector())
```

### `Ops.l1_loss`

- **Signature:** `l1_loss(input, target) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:177`; C++ `include/litetorch/ops.h:153`
- **Parameters:** `input`, `target` — same shape.
- **Returns:** scalar `[1]`, `mean(|input - target|)`. There is no `reduction` argument — the loss is always the mean.
- **PyTorch:** `F.l1_loss(input, target)` (default `reduction="mean"`) — equivalent.

<!-- ex: id=r6-l1-loss device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [4], cpu, False)
y = lt.Tensor.from_vector([1.0, 2.0, 3.0, 5.0], [4], cpu, False)
out = lt.Ops.l1_loss(x, y)
print(list(out.shape), out.to_vector())
```

### `Ops.bce_loss`

- **Signature:** `bce_loss(input, target) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:178`; C++ `include/litetorch/ops.h:154`
- **Parameters:** `input` — probabilities in `[0, 1]`, `target` — same shape, 0/1.
- **Returns:** scalar `[1]`, `mean(-(target * log(input) + (1 - target) * log(1 - input)))`. There is no `reduction` or `weight` argument.
- **Edge cases:** `input` is treated as a probability, not a logit — pass logits to `bce_with_logits_loss` instead. `input` values of exactly 0 or 1 make the loss infinite (`log(0)`).
- **PyTorch:** `F.binary_cross_entropy(input, target)` (default `reduction="mean"`) — equivalent.

<!-- ex: id=r6-bce-loss device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([0.9, 0.2, 0.8, 0.4], [4], cpu, False)
y = lt.Tensor.from_vector([1.0, 0.0, 1.0, 0.0], [4], cpu, False)
out = lt.Ops.bce_loss(x, y)
print(list(out.shape), out.to_vector())
```

### `Ops.cross_entropy_loss`

- **Signature:** `cross_entropy_loss(input, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:76`; C++ `include/litetorch/ops.h:32`
- **Parameters:** `input` — `[N, C]` logits. `target` — `[N]`, class indices. `weight` — `[C]` class weights or `None`. `ignore_index` — `int`, default `-100`. `reduction` — `"mean"` / `"sum"` / `"none"`, default `"mean"`. `label_smoothing` — `float`, default `0.0`.
- **Returns:** scalar `[1]` for `"mean"`/`"sum"`, `[N]` for `"none"`.
- **Edge cases:** **targets must be INT32 (or FP32/INT16/INT8) — INT64 targets throw** `cross_entropy_loss: target must have integer or float dtype`. This is a genuine defect, not a design choice: the target-index switch (`src/ops/loss.cpp:24-40`) has no INT64 case. Torch uses INT64 class indices by default, so ported code must cast first (`target` built with `dtype=lt.DataType.INT32`). Filed to Core 2026-10-26 (thread `docs-fix-requests-20261026`). The same restriction applies to `nll_loss` and `fused_linear_cross_entropy`.
- **PyTorch:** `F.cross_entropy(input, target, weight, ignore_index, reduction, label_smoothing)` — equivalent once the target is INT32.

<!-- ex: id=r6-cross-entropy-loss device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
logits = lt.Tensor.from_vector([2.0, 1.0, 0.0, 1.0, 0.0, 2.0], [2, 3], cpu, False)
target = lt.Tensor.from_vector([0.0, 2.0], [2], cpu, False, lt.DataType.INT32)
out = lt.Ops.cross_entropy_loss(logits, target)
print(list(out.shape), out.to_vector())
```

### `Ops.nll_loss`

- **Signature:** `nll_loss(input, target, weight=None, ignore_index=-100, reduction="mean") -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:77`; C++ `include/litetorch/ops.h:33`
- **Parameters:** `input` — `[N, C]` log-probabilities (e.g. the output of `log_softmax`). `target` — `[N]`, class indices. `weight`, `ignore_index`, `reduction` — same as `cross_entropy_loss`.
- **Returns:** scalar `[1]` for `"mean"`/`"sum"`, `[N]` for `"none"`.
- **Edge cases:** same INT64 restriction as `cross_entropy_loss` — INT64 targets throw; use INT32. The input must already be log-probabilities; raw logits give wrong values (use `cross_entropy_loss` for logits).
- **PyTorch:** `F.nll_loss(input, target, weight, ignore_index, reduction)` — equivalent once the target is INT32.

<!-- ex: id=r6-nll-loss device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import math
cpu = lt.Device("cpu")
logp = lt.Tensor.from_vector([math.log(0.7), math.log(0.3), math.log(0.2), math.log(0.8)], [2, 2], cpu, False)
target = lt.Tensor.from_vector([0.0, 1.0], [2], cpu, False, lt.DataType.INT32)
out = lt.Ops.nll_loss(logp, target)
print(list(out.shape), out.to_vector())
```

### `Ops.bce_with_logits_loss`

- **Signature:** `bce_with_logits_loss(input, target, weight=None, reduction="mean", pos_weight=None) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:78`; C++ `include/litetorch/ops.h:34`
- **Parameters:** `input` — logits (any shape). `target` — same shape, 0/1. `weight` — same shape or `None`. `reduction` — `"mean"` / `"sum"` / `"none"`, default `"mean"`. `pos_weight` — per-element weight for the positive class, or `None`.
- **Returns:** scalar `[1]` for `"mean"`/`"sum"`, same shape as `input` for `"none"`.
- **Edge cases:** unlike `bce_loss`, this takes raw logits — do not apply a sigmoid first.
- **PyTorch:** `F.binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction)` — equivalent.

<!-- ex: id=r6-bce-with-logits-loss device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, -1.0, 0.5, -0.5], [4], cpu, False)
y = lt.Tensor.from_vector([1.0, 0.0, 1.0, 0.0], [4], cpu, False)
out = lt.Ops.bce_with_logits_loss(x, y)
print(list(out.shape), out.to_vector())
```

### `Ops.fused_linear_cross_entropy`

- **Signature:** `fused_linear_cross_entropy(x, weight, target, bias=None) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:79`; C++ `include/litetorch/ops.h:35`
- **Parameters:** `x` — `[N, in]` hidden states. `weight` — `[C, in]` (Linear-style classifier weight). `target` — `[N]`, class indices. `bias` — `[C]` or `None`.
- **Returns:** scalar `[1]`, `cross_entropy(x @ weight.T + bias, target)` — the logits are never materialized, so this avoids the `[N, C]` intermediate (the memory saving matters for large vocabularies).
- **Edge cases:** same INT64 restriction as `cross_entropy_loss` — INT64 targets throw; use INT32. No `reduction`, `weight`, `ignore_index`, or `label_smoothing` arguments — this is always mean-reduced CE over the full target.
- **PyTorch:** no single equivalent; `F.cross_entropy(x @ weight.T + bias, target)` computes the same values.

<!-- ex: id=r6-fused-linear-cross-entropy device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], cpu, False)
w = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], [3, 4], cpu, False)
target = lt.Tensor.from_vector([0.0, 2.0], [2], cpu, False, lt.DataType.INT32)
out = lt.Ops.fused_linear_cross_entropy(x, w, target)
print(list(out.shape), out.to_vector())
```

## Quantization

### `Ops.fake_quantize`

- **Signature:** `fake_quantize(input, scale, zero_point=0.0, bits=8) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:165-166`; C++ `include/litetorch/ops.h:231-232`
- **Parameters:** `input` — any shape, FP32. `scale` — `float`. `zero_point` — `float`, default `0.0`. `bits` — `int`, default `8`.
- **Returns:** `round((input - zero_point) / scale) * scale + zero_point` (per the `bits` range), same shape and dtype as `input` — the values are quantized but stay in FP32 (fake quantization for quantization-aware training).
- **PyTorch:** no direct equivalent; `torch.fake_quantize_per_tensor_affine(input, scale, zero_point, -128, 127)` computes the same values for `bits=8`.

<!-- ex: id=r6-fake-quantize device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.03, 2.07, -0.96, 0.04], [4], cpu, False)
out = lt.Ops.fake_quantize(x, 0.1)
print(list(out.shape), out.to_vector())
```

## Quantized matmul helpers (documented in `ops-conv-attention.md`)

These ops live with the LLM kernels in R5; the entries below are pointers — the full signatures, parameters, edge cases, and runnable examples are in `ops-conv-attention.md`.

- `Ops.w8a8_matmul(x, w, x_scale, w_scale)` — FP32 inputs, quantizes both to INT8, computes `(round(x / x_scale) @ round(w / w_scale).T) * x_scale * w_scale`. See R5 "Quantized ops".
- `Ops.fp8_cast_scaled(x, scale, dtype=FP8_E4M3)` — FP32 to FP8 with a scale. See R5 "Quantized ops".
- `Ops.fp8_uncast_scaled(x, scale)` — FP8 back to FP32. See R5 "Quantized ops".
- `Ops.fp8_matmul(a, b, a_scale, b_scale, out_scale, out_dtype=FP8_E4M3)` — FP8 inputs only; cast first. See R5 "Quantized ops".
- `Ops.int4_pack(x)` / `Ops.int4_unpack(packed, numel)` — INT8 round-trip, two 4-bit values per byte. See R5 "Quantized ops".
- `Ops.int4_weight_only_linear(x, packed_w, w_scale, bias=None)` — 2D `x`, 2D `packed_w [N, (K+1)/2]`, per-channel `w_scale [N]`. See R5 "Quantized ops".
