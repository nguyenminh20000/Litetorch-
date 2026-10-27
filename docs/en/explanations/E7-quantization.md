# Explanation E7 — Quantization

Quantization stores weights and activations in fewer bits than FP32:
8 bits (INT8, FP8), 4 bits (INT4, NF4, FP4), sometimes fewer. The
payoff is not arithmetic speed alone — it is memory. A 7B-parameter
model in FP32 needs 28 GB of weights; in INT8 it needs 7 GB, in INT4
3.5 GB. On bandwidth-bound hardware (a T4 serving an LLM, a phone
running a classifier) the weights must be read from memory for every
token, so halving the bytes roughly halves the time, and the smaller
type unlocks integer and FP8 tensor-core paths that FP32 cannot use.

The price is error. Quantization is lossy by design: every value is
rounded to the nearest representable grid point, and values outside
the grid range saturate. This page states the error budgets as
measured numbers, never as vibes — and it states the exact formula
LiteTorch implements, because the naive formula is wrong (see E7.2).

Prerequisites: Tutorials T1–T2 (tensors, autograd) and T4
(quantization end to end). All examples run on CPU against LiteTorch
88a4ba1, PyTorch 2.14.1+cpu. GPU throughput claims are marked
REQUIRES-GPU and are not verified here.

> PyTorch differences touched in this page: D-19 (quantized dtypes
> promote to FP32 — tree-only, not yet probed against torch; do not
> assert torch parity for quantized promotion), D-21 (on GPU, integer
> elementwise math runs through FP32; the CPU integer path used here
> is exact), D-08 (an integer tensor plus a Python int scalar promotes
> to FP32 — build scale tensors explicitly when the dtype matters).

## E7.1 — The formats: FP8, INT8, INT4, NF4

Four formats cover nearly all practical quantization. They differ in
what they are good at, not just in bit count:

| Format | Bits | Values | Grid | Best for |
|---|---|---|---|---|
| FP8 E4M3 | 8 | float, 4 exponent + 3 mantissa bits | logarithmic, ~±448 max | activations, matmul inputs |
| FP8 E5M2 | 8 | float, 5 exponent + 2 mantissa bits | logarithmic, ~±57344 max | gradients, wide-dynamic-range tensors |
| INT8 | 8 | integer, -128…127 (see E7.3) | uniform, step = scale | weights + activations (w8a8) |
| INT4 | 4 | integer, -8…7 | uniform, step = scale | weight-only compression |
| NF4 | 4 | float, normal-distribution levels | non-uniform, dense near 0 | LLM weights (QLoRA-style) |
| FP4 E2M1 | 4 | float, 2 exponent + 1 mantissa bit | logarithmic, coarse | weight-only, extreme compression |

The split that matters most is **uniform vs non-uniform grid**.
INT8 and INT4 use a uniform grid: values are `round(x / s)`, spaced
`s` apart, and the scale `s` is the only parameter (plus an optional
zero point for asymmetric ranges). FP8 and NF4 use non-uniform grids:
FP8 spacing grows with magnitude (like all floats), and NF4 places
its 16 levels at the quantiles of a standard normal distribution, so
levels crowd near zero where LLM weights actually live. That is why
NF4 beats INT4 on normally-distributed weights at the same 4 bits —
and why NF4 is a poor choice for uniformly-distributed values.

```
FP32 value line:    |----*----*----*----*----*----*----*----*----|
INT8 grid (uniform): +    +    +    +    +    +    +    +    +
NF4 grid:            +  + +  +   +      +        +          +
                     ^ dense near zero, sparse in the tails
```

A second split is **what gets quantized**. Weight-only quantization
(INT4, NF4) shrinks storage and memory traffic; activations stay
FP16/FP32, so the matmul itself is still floating-point. Full
quantization (w8a8, FP8) quantizes both sides, so the matmul runs in
INT8/FP8 — faster, but activation quantization error now enters the
result too, and activations are harder to quantize than weights
(outliers in a few channels dominate the scale).

## E7.2 — The quant-dequant formula, and the w8a8 correction

Quantization is two steps: quantize, then (for the math) dequantize.

```
quantize:   q = clamp(round(x / s), qmin, qmax)     (integer codes)
dequantize: x_hat = q * s                            (back to float)
error:      |x - x_hat| <= s / 2   (inside the range; larger outside it)
```

The scale `s` sets the error: half a step at most, plus saturation
error for anything with `|x| > qmax * s`. Choosing `s` is therefore
choosing the error budget — which is why calibration (E7.5) measures
`s` from real data instead of guessing it.

For a matmul, the quantized form is **not** the naive
`x @ w.T * s`. Both operands must be quantized *before* the multiply,
and each is dequantized by its own scale afterwards:

```
out = (round(x / x_scale) @ round(w / w_scale).T) * x_scale * w_scale
```

with INT8 clamping to [-128, 127]. The naive form `(x @ w.T)` first
multiplies in FP32 and then scales — it never quantizes anything, and
it returns different numbers. The B3 reference pass caught exactly
this error in an earlier draft: the claimed formula
`(x @ w.T) * x_scale * w_scale` was refuted by a live run, and the
formula above is what `src/ops/w8a8_ops.cpp` implements. E7-1 below
verifies it against a torch mirror.

<!-- ex: id=E7-1 device=cpu backend=cpu runtime=~2s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([3.0, 7.0, -2.0, 9.0], [2, 2], cpu, False)
w = lt.Tensor.from_vector([4.0, -1.0, 2.0, 5.0], [2, 2], cpu, False)
out = lt.Ops.w8a8_matmul(x, w, x_scale=0.05, w_scale=0.04)
print("W8A8 result:", out.to_vector())
xt = torch.tensor([[3.0, 7.0], [-2.0, 9.0]])
wt = torch.tensor([[4.0, -1.0], [2.0, 5.0]])
qx = torch.clamp(torch.round(xt / 0.05), -128, 127).to(torch.int32)
qw = torch.clamp(torch.round(wt / 0.04), -128, 127).to(torch.int32)
ref = (qx @ qw.T).float() * 0.05 * 0.04
print("torch formula result:", ref.flatten().tolist())
diff = max(abs(a - b) for a, b in zip(out.to_vector(), ref.flatten().tolist()))
print("max diff vs torch formula:", diff)
assert diff <= 1e-4
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4
(max diff 1.9e-06, float32 rounding in the final scale multiply):
both give [5.65, 37.75, -14.35, 27.75]. The inputs saturate here
(7.0 / 0.05 = 140 clamps to 127), so the result differs from the pure
FP32 matmul — that is quantization error, not a bug. Note
`w8a8_matmul` takes FP32 inputs; passing INT8 tensors throws
(`w8a8_matmul requires FP32 inputs`).

## E7.3 — `fake_quantize`: rounding, clamping, and two measured quirks

Fake quantization simulates INT8 rounding in FP32: the value is
rounded to the grid and put back in FP32, so the network trains
against the error it will meet at inference. `Ops.fake_quantize(x,
scale, zero_point=0.0, bits=8)` is the primitive; `QATLinear` (E7.5)
calls it on weights and activations every forward pass.

Two behaviors of this implementation were measured on this build
(2026-10-27) and differ from the torch `round` + clamp idiom, so do
not write a torch mirror that assumes torch semantics:

- **Clamp range is [-127, 127], not [-128, 127].** `130.0` at
  `scale=0.1` fake-quantizes to `12.7`, and `-200.0` to `-12.7`.
- **Ties round half away from zero.** `-0.05` at `scale=0.1` becomes
  `-0.1`; `torch.round(-0.5)` gives `-0.0` (half to even).

Avoid exact `.5` ties and the `-128` code in torch mirrors of this op;
the mirrored part of E7-2 below uses tie-free values.

<!-- ex: id=E7-2 device=cpu backend=cpu runtime=~2s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([0.3, -1.7, 2.9, 0.04, 12.0, -12.0], [6], cpu, False)
fq = lt.Ops.fake_quantize(x, 0.1, 0.0, 8)
print("fake_quantize s=0.1:", fq.to_vector())
xt = torch.tensor([0.3, -1.7, 2.9, 0.04, 12.0, -12.0])
ref = torch.clamp(torch.round(xt / 0.1), -127, 127) * 0.1
print("torch mirror:", ref.tolist())
diff = max(abs(a - b) for a, b in zip(fq.to_vector(), ref.tolist()))
print("max diff vs torch mirror:", diff)
assert diff <= 1e-4
sat = lt.Tensor.from_vector([130.0, -200.0, -0.05], [3], cpu, False)
print("saturation probe:", lt.Ops.fake_quantize(sat, 0.1, 0.0, 8).to_vector())
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4 on
the mirrored values. The saturation probe prints
[12.7, -12.7, -0.1]: the clamp is [-127, 127] and the tie rounds away
from zero (see the quirks above).

## E7.4 — FP8: cast, matmul, uncast

FP8 keeps a floating-point grid, so it handles wide dynamic range
better than INT8 at the same 8 bits — at the cost of fewer mantissa
bits (E4M3 has 3, so adjacent values differ by 12.5%). LiteTorch
exposes the three steps separately:

- `fp8_cast_scaled(x, scale)` stores `round(x / scale)` in FP8. The
  scale divides: casting `[0.5, -1.25, 3.0, 0.125]` with `scale=0.05`
  stores `[10.0, -24.0, 60.0, 2.5]` (measured 2026-10-27; `-25`
  is not representable in E4M3, so it lands on `-24`).
- `fp8_uncast_scaled(x, scale)` multiplies the stored values back by
  the scale. Cast + uncast with the same scale round-trips values
  that are representable in FP8.
- `fp8_matmul(a, b, a_scale, b_scale, out_scale, out_dtype=FP8_E4M3)`
  takes **already-cast FP8 tensors** — passing FP32 inputs raises
  `fp8_uncast_scaled: input must be FP8`. With `out_dtype=FP32` the
  result is the plain FP32 product of the dequantized inputs.

<!-- ex: id=E7-3 device=cpu backend=cpu runtime=~2s torch-parity=1e-2 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu, False)
b = lt.Tensor.from_vector([1.0, 0.0, 0.0, 1.0], [2, 2], cpu, False)
ac = lt.Ops.fp8_cast_scaled(a, 1.0)
bc = lt.Ops.fp8_cast_scaled(b, 1.0)
out = lt.Ops.fp8_matmul(ac, bc, 1.0, 1.0, 1.0, lt.DataType.FP32)
print("fp8_matmul FP32 out:", out.to_vector())
ref = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) @ torch.tensor([[1.0, 0.0], [0.0, 1.0]])
diff = max(abs(x - y) for x, y in zip(out.to_vector(), ref.flatten().tolist()))
print("max diff vs FP32 matmul:", diff)
assert diff <= 1e-2
x = lt.Tensor.from_vector([0.5, -1.25, 3.0, 0.125], [4], cpu, False)
back = lt.Ops.fp8_uncast_scaled(lt.Ops.fp8_cast_scaled(x, 0.05), 0.05)
print("FP8 round trip:", back.to_vector())
err = max(abs(p - q) for p, q in zip(x.to_vector(), back.to_vector()))
print("FP8 max round-trip error:", err)
assert err <= 0.06
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1: the FP8 matmul
of exactly-representable inputs matches the FP32 matmul exactly
([1.0, 2.0, 3.0, 4.0]), and the round trip errs by at most 0.05 on
the probe values — one FP8 grid step at scale 0.05. The 1e-2
tolerance class applies to the matmul comparison (diff 0.0); the
round-trip error is stated as a measured absolute, not hidden.
E4M3 vs E5M2 is a range/precision trade: E4M3 for activations (more
mantissa), E5M2 for gradients (more range).

## E7.5 — QAT vs PTQ: train with the error, or calibrate after

**QAT (quantization-aware training)** trains with fake quantization
in the loop, so the weights adapt to the rounding. `QATLinear` wraps
a plain `Linear` and fake-quantizes weights and activations on every
forward pass; gradients flow through the fake quantizer (straight-
through estimator), so training sees the error inference will see.
Cost: training time and the fake-quantized forward on every step.
Use it when you retrain and accuracy matters most.

**PTQ (post-training quantization)** quantizes a finished model
without retraining: measure activation/weight ranges on calibration
data (`Calibrator`, `prepare` + `calibrate`), then swap each `Linear`
for a quantized one (`convert`). Cost: almost none. Accuracy: worse
than QAT, sometimes much worse on outliers — the calibration data
must resemble deployment data, or the scales are wrong for the real
inputs.

```
QAT:  train --fake-quant--> deploy (same graph, real INT8)
       ^ error is in the loss, weights adapt
PTQ:  train (FP32) --> calibrate scales --> convert --> deploy
       ^ error appears only after training, weights never adapt
```

<!-- ex: id=E7-4 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
lt.manual_seed(4)
raw_linear = lt.nn.Linear(64, 32, True)
raw_linear.to(cpu)
qat_layer = lt.quantization.QATLinear(raw_linear, w_scale=0.01, w_zp=0.0, a_scale=0.02, a_zp=0.0, b=8)
x = lt.Tensor.from_vector([0.5] * (2 * 64), [2, 64], cpu, True)
out = qat_layer.forward(x)
loss = lt.Ops.sum(out)
loss.backward()
plain = raw_linear.forward(x)
diffs = [abs(a - b) for a, b in zip(out.to_vector(), plain.to_vector())]
print("QAT max abs diff vs FP32:", max(diffs))
print("QAT mean abs diff vs FP32:", sum(diffs) / len(diffs))
print("weight grad nonzero:", any(v != 0.0 for v in qat_layer.weight.grad.to_vector()))
assert max(diffs) <= 0.05
```

Verified on CPU, LiteTorch 88a4ba1: max abs diff vs FP32 is
0.0203, mean 0.0065, and the weight gradient is non-zero — gradients
flow through the fake quantizer. No torch parity claim: torch has no
`QATLinear` equivalent with these scales; the comparison is against
the wrapped FP32 layer itself. (T4.1 measured 0.0500 max / 0.0133
mean at `w_scale=0.005`, `a_scale=0.01` — the error scales with the
scales you choose.)

<!-- ex: id=E7-5 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
lt.manual_seed(3)
model = lt.nn.Sequential([lt.nn.Linear(8, 8, True)])
model.to(cpu)
xin = lt.Tensor.from_vector([0.1, -0.4, 0.9, 0.2, -0.7, 1.3, 0.05, -1.1], [1, 8], cpu, False)
fp32_out = model.forward(xin).to_vector()
cfg = lt.quantization.PTQConfig()
print("PTQConfig defaults bits/per_channel/symmetric:", cfg.bits, cfg.per_channel, cfg.symmetric)
prepared = lt.quantization.prepare(model, cfg)
prepared.calibrate(lambda: lt.Tensor.from_vector([0.1, -0.4, 0.9, 0.2, -0.7, 1.3, 0.05, -1.1], [1, 8], cpu, False), 4)
converted = lt.quantization.convert(prepared)
ptq_out = converted.forward(xin).to_vector()
diffs = [abs(a - b) for a, b in zip(fp32_out, ptq_out)]
print("PTQ max abs diff vs FP32:", max(diffs))
print("PTQ mean abs diff vs FP32:", sum(diffs) / len(diffs))
assert max(diffs) <= 1e-2
```

Verified on CPU, LiteTorch 88a4ba1: PTQ defaults are 8 bits,
per-channel, symmetric; on this tiny model the converted output
differs from FP32 by at most 0.0020 (mean 0.0010) — inside the 1e-2
after-dequant tolerance. Measure the FP32 output **before**
`convert`: `convert` replaces the `Linear` in place inside the model,
so reading the "FP32" output afterwards compares the model with
itself. `convert` quantizes weights per output channel
(`src/ops/ptq.cpp`: scale = row max / 127, INT8 codes, dequantized in
the forward pass).

## E7.6 — 4-bit weights: INT4, NF4, FP4

Weight-only 4-bit quantization is the cheapest way to halve model
size again after INT8. `int4_pack` / `int4_unpack` pack two 4-bit
values (range -8…7) per INT8 byte; `int4_weight_only_linear(x,
packed_w, w_scale, bias=None)` computes `x @ dequant(packed_w).T
(+ bias)` with `packed_w` 2D `[N, (K+1)/2]` and one scale per output
channel. A 1D packed tensor must be reshaped first — passing it
raises `packed_w shape mismatch`.

NF4 and FP4 are casts, not a matmul path: `w.cast(DataType.NF4)`
stores each weight as the nearest NF4 level. The round trip is lossy
by design, and the loss depends on the value range: on
`[0.1, -0.5, 1.2, -2.4]` the FP4 round trip errs by at most 0.40 and
the NF4 round trip by at most 1.40 — the -2.4 value sits outside
NF4's well-covered range (NF4 levels crowd near zero). Measure the
error on your own weight distribution before trusting a 4-bit weight.

<!-- ex: id=E7-6 device=cpu backend=cpu runtime=~2s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], cpu, False)
w = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0], [16], cpu, False, lt.DataType.INT8)
packed = lt.Ops.int4_pack(w).reshape([4, 2])
w_scale = lt.Tensor.from_vector([1.0, 1.0, 1.0, 1.0], [4], cpu, False)
out = lt.Ops.int4_weight_only_linear(x, packed, w_scale)
print("INT4 weight-only out:", out.to_vector())
wt = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 1.0], [2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 1.0, 2.0]])
xt = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
ref = xt @ wt.T
diff = max(abs(a - b) for a, b in zip(out.to_vector(), ref.flatten().tolist()))
print("max diff vs torch dequant mirror:", diff)
assert diff <= 1e-4
w_fp32 = lt.Tensor.from_vector([0.1, -0.5, 1.2, -2.4], [2, 2], cpu, False, lt.DataType.FP32)
for dtype, name in [(lt.DataType.FP4_E2M1, "FP4"), (lt.DataType.NF4, "NF4")]:
    back = w_fp32.cast(dtype).cast(lt.DataType.FP32)
    err = max(abs(a - b) for a, b in zip(w_fp32.to_vector(), back.to_vector()))
    print(name, "max round-trip error:", err)
    assert err <= 1.5
```

Verified on CPU, LiteTorch 88a4ba1, PyTorch 2.14.1: the INT4
weight-only linear matches the torch dequant mirror exactly
([30.0, 42.0, 40.0, 31.0, 70.0, 118.0, 96.0, 95.0]); FP4 max
round-trip error 0.40, NF4 1.40 — re-measured 2026-10-27, confirming
the B1-tut numbers. Do not assert torch parity for 4-bit casts
(D-19, tree-only).

## E7.7 — Error budgets: what to expect, what to assert

| Path | Measured error (this build, CPU) | Tolerance to assert |
|---|---|---|
| w8a8 matmul vs quant-dequant formula | 1.9e-06 (FP32 scale-multiply rounding) | 1e-4 |
| `fake_quantize` vs torch mirror (tie-free) | 0.0 | 1e-4 |
| FP8 matmul vs FP32 matmul (representable inputs) | 0.0 | 1e-2 after dequant |
| FP8 cast/uncast round trip (probe values) | 0.05 | absolute budget; 1 grid step at s=0.05 |
| QAT vs FP32 (`w_scale=0.01`, `a_scale=0.02`) | max 0.0203, mean 0.0065 | budget, not parity — scales with `s` |
| PTQ 8-bit vs FP32 (tiny Linear) | max 0.0020, mean 0.0010 | 1e-2 after dequant |
| INT4 weight-only vs dequant mirror | exact (0.0) | 1e-4 |
| FP4 round trip | 0.40 | budget — range-dependent |
| NF4 round trip | 1.40 | budget — range-dependent |

Rules of thumb: quantization error scales with the scale `s`, and `s`
scales with the largest magnitude in the tensor — one outlier channel
can wreck the whole tensor's precision. That is why per-channel
scales (one `s` per output row) beat per-tensor scales, and why
activation outliers are the usual reason PTQ accuracy collapses.
Quantized parity tolerance is 1e-2 after dequant (verification.md
§3.2); FP32 parity stays 1e-4.

## E7.8 — When not to quantize, and what needs a GPU

Do not quantize the parts where error compounds fastest: the final
logits layer of a classifier (small logit differences flip argmax),
embeddings with rare tokens (few calibration samples), and anything
whose outputs feed a long chain of further computation without
renormalization. Quantize the big matmuls first — they hold the
bytes — and measure accuracy after each step, not after all of them.

> REQUIRES-GPU: serving throughput with quantized weights (tokens/s,
> time-to-first-token), FP8 matmul at scale, and INT8 kernel
> benchmarks are GPU measurements. They belong on Kaggle (B5), not in
> this page — no throughput number is claimed here.
