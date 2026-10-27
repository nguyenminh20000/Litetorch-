# How-To H3 — Quantize a Model

Three ways to shrink a model: quantization-aware training (fake
quantization during training), post-training calibration (measure scales
from real activations), and direct low-precision matmuls (W8A8, FP8,
INT4/NF4). All examples below run on CPU against LiteTorch 88a4ba1.

Prerequisites: LiteTorch installed and importable. Torch parity is
checked with `~/torch-venv` (torch 2.14.1+cpu).

> PyTorch differences touched here: D-19 (quantized dtypes promote to
> FP32 — live check still open, see H3.1 note), D-09 (GELU is
> tanh-approximation only; compare against `F.gelu(approximate='tanh')`).

## H3.1 — Quantization-aware training with QATLinear

`QATLinear` wraps a `Linear` and applies fake quantization to weights
and activations on every forward pass, so the model learns weights that
survive real INT8/INT4 quantization. Set the scales and zero points from
your calibration run (H3.2); `b` is the bit-width.

<!-- ex: id=H3.1-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

lt.manual_seed(0)
raw_linear = lt.nn.Linear(512, 256, True)
raw_linear.to(device)

qat_layer = lt.quantization.QATLinear(
    raw_linear,
    w_scale=0.005,
    w_zp=0.0,
    a_scale=0.01,
    a_zp=0.0,
    b=8
)

x = lt.Tensor.from_vector([0.5] * (4 * 512), [4, 512], device, True)
out = qat_layer.forward(x)
loss = lt.Ops.sum(out)
loss.backward()

print("QAT output shape:", list(out.shape))
print("Weight grad first 5:", qat_layer.weight.grad.to_vector()[:5])
```

Verified on CPU, LiteTorch 88a4ba1. Expected output: shape `[4, 256]`
and a non-zero weight gradient. The fake-quantization error is bounded
by half the weight scale (`w_scale / 2` = 0.0025 here) per weight; that
bound is what makes QAT-trained weights robust when you later quantize
for real.

## H3.2 — Calibrate scales from real activations

Run representative inputs through the model, collect activation
statistics with `Calibrator`, and read off the scale and asymmetric
parameters for your bit-width. Use these numbers as the `QATLinear`
scales in H3.1 or for a PTQ conversion.

<!-- ex: id=H3.2-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()
calibrator = lt.quantization.Calibrator()

sample_act = lt.Tensor.from_vector([0.1, -0.5, 2.3, -1.8], [1, 4], device, False)
calibrator.collect("layer1_act", sample_act)

scale = calibrator.get_scale("layer1_act", bits=8)
params = calibrator.get_asymmetric_params("layer1_act", bits=8)
print("Scale (8-bit):", scale)
print("Asymmetric params:", params)
```

Verified on CPU, LiteTorch 88a4ba1. Expected output: a positive scale
and an asymmetric parameter set for `layer1_act`. Collect from real
data, not from synthetic constants — the scale is only as good as the
activations you measured. For a full PTQ pass, use
`lt.quantization.quantize_linear(layer)` after calibration, or
`quantize_dynamic_linear` / `quantize_dynamic_lstm` /
`quantize_dynamic_gru` for dynamic quantization without a calibration
set.

## H3.3 — W8A8 matmul: FP32 in, INT8 math, FP32 out

`w8a8_matmul(x, w, x_scale, w_scale)` quantizes both FP32 inputs to
INT8, multiplies them in integer arithmetic, and dequantizes with
`x_scale * w_scale`. The weight is `[N, K]` — output channels first,
like an `nn.Linear` weight — and the op computes `x @ w.T`, **not**
`x @ w`. Both inputs must be FP32; INT8 inputs raise
`w8a8_matmul requires FP32 inputs`.

<!-- ex: id=H3.3-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

x = lt.Tensor.from_vector([10.0, 20.0, -15.0, 30.0], [2, 2], device, False)
w = lt.Tensor.from_vector([5.0, -2.0, 1.0, 4.0], [2, 2], device, False)

w8a8_out = lt.Ops.w8a8_matmul(x, w, 0.01, 0.02)
print("W8A8 result:", w8a8_out.to_vector())
print("W8A8 dtype:", w8a8_out.dtype)
```

Verified on CPU, LiteTorch 88a4ba1, torch parity 1e-4 against the
integer formula `round(x / x_scale) @ round(w / w_scale).T * x_scale *
w_scale` with clamping to [-128, 127]. Expected output:
`[0.6858, 4.4958, -5.7912, 1.9458]`. Note the transpose: with
`x = [[10, 20], [-15, 30]]` and `w = [[5, -2], [1, 4]]`, the result is
`x @ w.T` in quantized arithmetic — passing a `[K, N]` weight by
mistake silently computes the wrong product, so check the weight shape
first.

## H3.4 — 4-bit weights: NF4 and FP4 casts

Cast FP32 weights to NF4 (NormalFloat4, the QLoRA format) or FP4_E2M1
(NVFP4). The cast is a storage conversion; compute still runs in FP32
after an implicit upcast, so measure the round-trip error before
trusting a quantized model.

<!-- ex: id=H3.4-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

w_fp32 = lt.Tensor.from_vector([0.1, -0.5, 1.2, -2.4], [2, 2], device, False, lt.DataType.FP32)
w_fp4 = w_fp32.cast(lt.DataType.FP4_E2M1)
w_nf4 = w_fp32.cast(lt.DataType.NF4)

print("FP4 dtype:", w_fp4.dtype)
print("NF4 dtype:", w_nf4.dtype)
print("FP4 round-trip:", w_fp4.cast(lt.DataType.FP32).to_vector())
print("NF4 round-trip:", w_nf4.cast(lt.DataType.FP32).to_vector())
```

Verified on CPU, LiteTorch 88a4ba1. Expected output: dtypes
`FP4_E2M1` and `NF4`, with round-trip values
FP4 `[0.0, -0.5, 1.0, -2.0]` and NF4 `[0.0596, -0.5251, 1.0, -1.0]`.
Read them honestly: FP4_E2M1 rounds each value to its grid (0.1 has no
nearby grid point and rounds to 0.0), and NF4's values are normalized
to [-1, 1], so out-of-range inputs clamp (1.2 → 1.0, -2.4 → -1.0). A
bare `.cast()` with no per-tensor scale is not a usable quantization —
that is exactly what calibration (H3.2) and QAT (H3.1) are for. If the
round-trip error is larger than your tolerance, the layer needs QAT or
a wider format, not a blind cast.
