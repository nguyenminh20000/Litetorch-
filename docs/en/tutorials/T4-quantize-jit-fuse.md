# Tutorial T4 — Quantize and JIT-Fuse a Model, End to End

Take a plain FP32 linear layer, train it quantization-aware, calibrate
scales from real activations, run an INT8 matmul, fuse an elementwise
graph into one JIT kernel, and squeeze weights into 4 bits. Every step
runs on CPU against LiteTorch 88a4ba1, with the quantization error
measured and stated — never hidden.

What you will learn:

- How fake quantization works in `QATLinear`, and how big its error is.
- How the `Calibrator` turns observed activations into scales.
- How `w8a8_matmul` quantizes, multiplies, and dequantizes.
- How to trace and fuse a JIT expression, and how fused ops carry
  gradients.
- What 4-bit casts (NF4, FP4) cost in round-trip error.

Prerequisites: Tutorials T1–T2. FP8 numerics caveats are called out
where they apply; GPU-only quantization paths are marked REQUIRES-GPU
and verified on Kaggle in B5, not claimed here.

> PyTorch differences touched in this tutorial: D-09 (GELU is
> tanh-approximation only — compare against `F.gelu(x,
> approximate='tanh')`, never against the exact erf default), D-19
> (quantized dtypes promote to FP32 — tree-only, not yet probed against
> torch; do not assert torch parity for quantized promotion), D-21 (on
> GPU, integer elementwise math runs through FP32; the CPU integer path
> used here is exact).

## T4.1 — Quantization-aware training with QATLinear

`QATLinear` wraps a plain `Linear` and fake-quantizes weights and
activations to 8 bits on every forward pass, so training sees the error
inference will see. The error is real and measurable: with
`w_scale=0.005` and `a_scale=0.01` on the input below, the QAT output
differs from the FP32 output by at most 0.0500 (mean 0.0133). That is
the expected fake-quantization error, not a bug — it scales with the
scales you choose.

<!-- ex: id=T4.1-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

lt.manual_seed(2)
device = lt.auto_device()

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

print("QAT output shape:", out.shape)
print("weight grad first values:", qat_layer.weight.grad.to_vector()[:5])

plain = raw_linear.forward(x)
diffs = [abs(a - b) for a, b in zip(out.to_vector(), plain.to_vector())]
print("max abs diff vs FP32:", max(diffs))
print("mean abs diff vs FP32:", sum(diffs) / len(diffs))
```

Verified on CPU, LiteTorch 88a4ba1: gradients flow through the fake
quantizer (weight grad is non-zero), max abs diff vs FP32 is 0.0500,
mean abs diff is 0.0133. No torch parity claim: torch has no
`QATLinear` equivalent with these scales; the comparison is against
the wrapped FP32 layer itself.

## T4.2 — Calibrate scales from real activations

Guessing scales is wrong; measure them. The `Calibrator` collects
activation statistics, then `get_scale` / `get_asymmetric_params`
return the scale and zero point for a chosen bit-width.

<!-- ex: id=T4.2-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

device = lt.auto_device()
calibrator = lt.quantization.Calibrator()

sample_act = lt.Tensor.from_vector([0.1, -0.5, 2.3, -1.8], [1, 4], device, False)
calibrator.collect("layer1_act", sample_act)

scale = calibrator.get_scale("layer1_act", bits=8)
params = calibrator.get_asymmetric_params("layer1_act", bits=8)
print("scale:", scale)
print("asymmetric params (scale, zero point):", params)
```

Verified on CPU, LiteTorch 88a4ba1: scale is 0.01811 and the
asymmetric params are (0.01614, -15.0) for this sample. In a PTQ flow
these scales feed `QATLinear` (or `quantize_linear`) instead of
hand-picked constants.

## T4.3 — W8A8 matmul

`w8a8_matmul` takes FP32 inputs — passing INT8 tensors throws
(`w8a8_matmul requires FP32 inputs`; the old skeleton §10.3 did
exactly that). It quantizes both inputs to INT8, multiplies in INT8,
and dequantizes: out = (round(x / x_scale) @ round(w / w_scale)^T) *
x_scale * w_scale, where `w` is `[N, K]`. The result matches that
formula exactly.

<!-- ex: id=T4.3-1 device=cpu backend=cpu runtime=~4.1s torch-parity=exact status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt
import torch

device = lt.auto_device()

x = lt.Tensor.from_vector([10.0, 20.0, -15.0, 30.0], [2, 2], device, False, lt.DataType.FP32)
w = lt.Tensor.from_vector([5.0, -2.0, 1.0, 4.0], [2, 2], device, False, lt.DataType.FP32)

out = lt.Ops.w8a8_matmul(x, w, x_scale=0.01, w_scale=0.02)
print("W8A8 result:", out.to_vector())

xt = torch.tensor([[10.0, 20.0], [-15.0, 30.0]])
wt = torch.tensor([[5.0, -2.0], [1.0, 4.0]])
qx = torch.clamp(torch.round(xt / 0.01), -128, 127).to(torch.int32)
qw = torch.clamp(torch.round(wt / 0.02), -128, 127).to(torch.int32)
ref = (qx @ qw.T).float() * 0.01 * 0.02
print("torch formula result:", ref.flatten().tolist())
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity exact:
both give [0.6858, 4.4958, -5.7912, 1.9458]. The INT8 multiply itself
is exact; the only error vs a pure FP32 matmul is the quantization of
the inputs, which saturates here (values / 0.01 exceed the INT8 range).

## T4.4 — JIT: trace an expression and fuse it

Build a symbolic expression with `JITVar`, compile it into a
`JITFunction`, run it, save it, and load it back. `Tracer.trace` does
the same from an ordinary Python function.

<!-- ex: id=T4.4-1 device=cpu backend=cpu runtime=~4.1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt
import torch

device = lt.auto_device()

v1 = lt.jit.JITVar(lt.jit.OpType.INPUT, "x")
v2 = lt.jit.JITVar(lt.jit.OpType.INPUT, "w")
expr = lt.jit.relu(v1 * 2.0 + v2)

jit_fn = lt.jit.JITFunction("fused_relu_add", expr, [v1, v2])

arg1 = lt.Tensor.from_vector([1.0, 2.0], [2], device, False)
arg2 = lt.Tensor.from_vector([-3.0, 1.0], [2], device, False)

res = jit_fn([arg1, arg2])
print("JIT output:", res.to_vector())

jit_fn.save("/tmp/fused_relu.jit")
loaded_fn = lt.jit.JITFunction.load("/tmp/fused_relu.jit")
print("reloaded output:", loaded_fn([arg1, arg2]).to_vector())

def custom_activation(args):
    return lt.Ops.gelu(lt.Ops.add(args[0], args[1]))

traced_fn = lt.jit.Tracer.trace([arg1, arg2], custom_activation, "traced_gelu")
traced_res = traced_fn([arg1, arg2])
print("traced GELU output:", traced_res.to_vector())

ref = torch.relu(torch.tensor([1.0, 2.0]) * 2.0 + torch.tensor([-3.0, 1.0]))
print("torch reference:", ref.tolist())
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4
(JIT output [0.0, 5.0] matches torch exactly; the traced GELU output
matches `F.gelu(x, approximate='tanh')` — see T4.5 for why the
`approximate` matters).

## T4.5 — Fused ops carry gradients

`JITFunction` integrates with autograd: with `requires_grad=True`
inputs, backward propagates through the fused kernel. The fused GELU
here is the tanh approximation (D-09) — compare it against
`F.gelu(approximate='tanh')`, never against torch's exact-erf default,
which differs by up to ~4.4e-4.

<!-- ex: id=T4.5-1 device=cpu backend=cpu runtime=~4.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt
import torch

device = lt.auto_device()

x = lt.Tensor.from_vector([1.0, -2.0, 3.0], [3], device, True)
bias = lt.Tensor.from_vector([0.5, 0.5, 0.5], [3], device, True)

out = lt.jit.fused_bias_gelu(x, bias)
loss = lt.Ops.sum(out)
loss.backward()

print("fused output:", out.to_vector())
print("x.grad:", x.grad.to_vector())
print("bias.grad:", bias.grad.to_vector())

xt = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
bt = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
ref = torch.nn.functional.gelu(xt + bt, approximate="tanh")
ref.sum().backward()
print("torch tanh-GELU:", ref.detach().tolist())
print("torch x.grad:", xt.grad.tolist())
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4:
forward and gradient max diffs are both 0.0 against
`F.gelu(approximate='tanh')`. Against torch's default exact-erf GELU
the same outputs differ by up to ~4.4e-4 — that is D-09, a documented
difference, not a defect.

## T4.6 — 4-bit weights: NF4 and FP4

Cast FP32 weights down to 4 bits and back. The round trip is lossy by
design — that is the price of 4-bit storage. On the values below the
FP4 round trip errs by at most 0.40 and the NF4 round trip by at most
1.40 (the -2.4 value is outside NF4's well-covered range).

<!-- ex: id=T4.6-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

device = lt.auto_device()

w_fp32 = lt.Tensor.from_vector([0.1, -0.5, 1.2, -2.4], [2, 2], device, False, lt.DataType.FP32)
w_fp4 = w_fp32.cast(lt.DataType.FP4_E2M1)
w_nf4 = w_fp32.cast(lt.DataType.NF4)

print("FP4 dtype:", w_fp4.dtype)
print("NF4 dtype:", w_nf4.dtype)

back_fp4 = w_fp4.cast(lt.DataType.FP32)
back_nf4 = w_nf4.cast(lt.DataType.FP32)
orig = w_fp32.to_vector()
print("FP4 round trip:", back_fp4.to_vector())
print("NF4 round trip:", back_nf4.to_vector())
print("FP4 max round-trip error:", max(abs(a - b) for a, b in zip(orig, back_fp4.to_vector())))
print("NF4 max round-trip error:", max(abs(a - b) for a, b in zip(orig, back_nf4.to_vector())))
```

Verified on CPU, LiteTorch 88a4ba1: FP4 max round-trip error 0.40,
NF4 max round-trip error 1.40. Do not assert torch parity for
quantized promotion or 4-bit casts (D-19, tree-only): measure the
error on your own value range before trusting a 4-bit weight.
