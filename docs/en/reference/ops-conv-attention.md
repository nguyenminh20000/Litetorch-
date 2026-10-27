# Ops — Convolution, Pooling, Normalization, Attention, LLM

`litetorch.Ops` functions for convolution, pooling, normalization, attention, and LLM kernels (fused, quantized, and serving ops). All compute is C++14; Python is the pybind11 interface layer.

- **Source of truth:** `src/bindings/bind_ops.cpp:37-42,69-83,143-164,174-178` (Python bindings), `include/litetorch/ops.h` (C++ declarations).
- **Tree:** agent-v1@02cc271a / local HEAD 89afd67 (docs baseline; B5 re-verifies at head). No binding drift in `bind_ops.cpp` / `ops.h` between 88a4ba1 and 89afd67.
- **Conventions:** `cpu = lt.Device("cpu")` in every example. Every example is self-contained and runnable on CPU unless marked REQUIRES-GPU. Torch parity is FP32 1e-4 unless the header says otherwise.
- **Related deviations:** D-20 (no `F.scaled_dot_product_attention` — `Ops.flash_attention` only; `flash_attention_qkv` is GPU-only), D-09 (GELU is tanh-only — see `ops-elementwise.md`). See `plan/differences.md`.

## Convolution

### `Ops.conv1d`

- **Signature:** `conv1d(input, weight, bias=None, stride=1, padding=0) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:174`; C++ `include/litetorch/ops.h:150`
- **Parameters:** `input` — `[N, C_in, L]`. `weight` — `[C_out, C_in, K]`. `bias` — `[C_out]` or `None`. `stride`, `padding` — scalar `int` only, no tuple form.
- **Returns:** `[N, C_out, L_out]`, `L_out = floor((L + 2 * padding - K) / stride) + 1`.
- **Edge cases:** no dilation or groups arguments (unlike torch); a scalar `stride`/`padding` applies to the single spatial dim. Mismatched `C_in` raises.
- **PyTorch:** `F.conv1d(input, weight, bias, stride, padding)` — equivalent for scalar stride/padding; torch also accepts tuple stride/padding, dilation, groups, which LiteTorch does not.

<!-- ex: id=r5-conv1d device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 2, 4], cpu, False)
w = lt.Tensor.from_vector([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [3, 2, 3], cpu, False)
out = lt.Ops.conv1d(x, w)
print(list(out.shape), out.to_vector())
```

### `Ops.conv2d`

- **Signature:** `conv2d(input, weight, bias=None, stride=1, padding=0) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:37`; C++ `include/litetorch/ops.h:26`
- **Parameters:** `input` — `[N, C_in, H, W]`. `weight` — `[C_out, C_in, K_H, K_W]`. `bias` — `[C_out]` or `None`. `stride`, `padding` — scalar `int` only, applied to both spatial dims.
- **Returns:** `[N, C_out, H_out, W_out]`.
- **Edge cases:** no dilation or groups arguments; no tuple stride/padding. Square kernels only in the sense that one scalar `stride`/`padding` covers both dims.
- **PyTorch:** `F.conv2d(input, weight, bias, stride, padding)` — equivalent for scalar stride/padding; torch also accepts tuple stride/padding, dilation, groups.

<!-- ex: id=r5-conv2d device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i) for i in range(32)], [1, 2, 4, 4], cpu, False)
w = lt.Tensor.from_vector([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0] * 3, [3, 2, 3, 3], cpu, False)
out = lt.Ops.conv2d(x, w)
print(list(out.shape), out.to_vector()[:4])
```

### `Ops.conv2d_relu`

- **Signature:** `conv2d_relu(input, weight, bias=None, stride=1, padding=0) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:38`; C++ `include/litetorch/ops.h:27`
- **Parameters:** same as `conv2d`.
- **Returns:** `relu(conv2d(input, weight, bias, stride, padding))`, fused into one kernel launch.
- **Edge cases:** same restrictions as `conv2d` (scalar stride/padding, no dilation/groups). Negative outputs are zeroed by the fused ReLU — do not expect negative values even when the raw convolution is negative.
- **PyTorch:** no single equivalent; `F.relu(F.conv2d(...))` computes the same values.

<!-- ex: id=r5-conv2d-relu device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i) for i in range(32)], [1, 2, 4, 4], cpu, False)
w = lt.Tensor.from_vector([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0] * 3, [3, 2, 3, 3], cpu, False)
out = lt.Ops.conv2d_relu(x, w)
print(list(out.shape), out.to_vector()[:4])
```

### `Ops.conv3d`

- **Signature:** `conv3d(input, weight, bias=None, stride=1, padding=0) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:39`; C++ `include/litetorch/ops.h:28`
- **Parameters:** `input` — `[N, C_in, D, H, W]`. `weight` — `[C_out, C_in, K_D, K_H, K_W]`. `bias` — `[C_out]` or `None`. `stride`, `padding` — scalar `int` only.
- **Returns:** `[N, C_out, D_out, H_out, W_out]`.
- **Edge cases:** same restrictions as `conv2d` (scalar stride/padding, no dilation/groups).
- **PyTorch:** `F.conv3d(input, weight, bias, stride, padding)` — equivalent for scalar stride/padding.

<!-- ex: id=r5-conv3d device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i % 5) for i in range(54)], [1, 2, 3, 3, 3], cpu, False)
w = lt.Tensor.from_vector([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0] * 2, [2, 2, 2, 2, 2], cpu, False)
out = lt.Ops.conv3d(x, w)
print(list(out.shape), out.to_vector()[:4])
```

## Pooling

### `Ops.max_pool2d`

- **Signature:** `max_pool2d(input, kernel_size, stride=-1, padding=0) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:40`; C++ `include/litetorch/ops.h:69`
- **Parameters:** `input` — `[N, C, H, W]`. `kernel_size` — `int`. `stride` — `int`, default `-1` means stride equals `kernel_size`. `padding` — `int`.
- **Returns:** `[N, C, H_out, W_out]`, per-window maximum. No indices are returned (unlike torch, which can return them).
- **Edge cases:** `kernel_size` larger than the input raises; `stride=-1` is the only way to spell "stride = kernel size".
- **PyTorch:** `F.max_pool2d(input, kernel_size, stride, padding)` — equivalent when indices are not needed; torch returns indices only on request.

<!-- ex: id=r5-max-pool2d device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i) for i in range(32)], [1, 2, 4, 4], cpu, False)
out = lt.Ops.max_pool2d(x, 2)
print(list(out.shape), out.to_vector()[:4])
```

### `Ops.max_pool3d`

- **Signature:** `max_pool3d(input, kernel_size, stride=-1, padding=0) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:41`; C++ `include/litetorch/ops.h:70`
- **Parameters:** `input` — `[N, C, D, H, W]`. `kernel_size`, `stride`, `padding` — same conventions as `max_pool2d`.
- **Returns:** `[N, C, D_out, H_out, W_out]`, per-window maximum, no indices.
- **PyTorch:** `F.max_pool3d(input, kernel_size, stride, padding)` — equivalent when indices are not needed.

<!-- ex: id=r5-max-pool3d device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i % 7) for i in range(128)], [1, 2, 4, 4, 4], cpu, False)
out = lt.Ops.max_pool3d(x, 2)
print(list(out.shape), out.to_vector()[:4])
```

### `Ops.avg_pool2d`

- **Signature:** `avg_pool2d(input, kernel_size, stride=-1, padding=0) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:175`; C++ `include/litetorch/ops.h:151`
- **Parameters:** `input` — `[N, C, H, W]`. `kernel_size`, `stride` (default `-1` = kernel size), `padding` — same conventions as `max_pool2d`.
- **Returns:** `[N, C, H_out, W_out]`, per-window mean.
- **PyTorch:** `F.avg_pool2d(input, kernel_size, stride, padding)` — equivalent.

<!-- ex: id=r5-avg-pool2d device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i) for i in range(32)], [1, 2, 4, 4], cpu, False)
out = lt.Ops.avg_pool2d(x, 2)
print(list(out.shape), out.to_vector()[:4])
```

### `Ops.adaptive_avg_pool2d`

- **Signature:** `adaptive_avg_pool2d(input, output_height, output_width) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:42`; C++ `include/litetorch/ops.h:83`
- **Parameters:** `input` — `[N, C, H, W]`. `output_height`, `output_width` — `int` target sizes.
- **Returns:** `[N, C, output_height, output_width]`, per-window mean over adaptive bins.
- **PyTorch:** `F.adaptive_avg_pool2d(input, (output_height, output_width))` — equivalent.

<!-- ex: id=r5-adaptive-avg-pool2d device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i) for i in range(32)], [1, 2, 4, 4], cpu, False)
out = lt.Ops.adaptive_avg_pool2d(x, 2, 2)
print(list(out.shape), out.to_vector()[:4])
```

## Normalization

### `Ops.layer_norm`

- **Signature:** `layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:34`; C++ `include/litetorch/ops.h:80`
- **Parameters:** `input` — any rank. `normalized_shape` — list of ints (the trailing dims to normalize). `weight` / `bias` — tensors matching `normalized_shape`, or `None`. `eps` — `float`, default `1e-5`.
- **Returns:** same shape as `input`, normalized to zero mean and unit variance over `normalized_shape`, then scaled by `weight` and shifted by `bias` when given.
- **Edge cases:** `normalized_shape` must be a list — torch also accepts a bare int, LiteTorch does not.
- **PyTorch:** `F.layer_norm(input, normalized_shape, weight, bias, eps)` — equivalent.

<!-- ex: id=r5-layer-norm device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], cpu, False)
out = lt.Ops.layer_norm(x, [3])
print(list(out.shape), out.to_vector()[:3])
```

### `Ops.batch_norm2d`

- **Signature:** `batch_norm2d(input, running_mean, running_var, weight, bias, training, momentum=0.1, eps=1e-5) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:36`; C++ `include/litetorch/ops.h:71`
- **Parameters:** `input` — `[N, C, H, W]`. `running_mean` / `running_var` / `weight` / `bias` — `[C]` each, all required (no `None` defaults). `training` — `bool`: `True` uses batch statistics and updates the running stats; `False` uses the running stats. `momentum` — `float`, default `0.1`. `eps` — `float`, default `1e-5`.
- **Returns:** same shape as `input`.
- **Edge cases:** all four `[C]` tensors are required — there is no affine-free form at the `Ops` level (use the `nn.BatchNorm2d` module for optional affine). `training=True` on a single-sample batch normalizes with that batch's own statistics.
- **PyTorch:** `F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)` — equivalent.

<!-- ex: id=r5-batch-norm2d device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i % 4) for i in range(96)], [2, 3, 4, 4], cpu, False)
mean = lt.Tensor.from_vector([0.0, 0.0, 0.0], [3], cpu, False)
var = lt.Tensor.from_vector([1.0, 1.0, 1.0], [3], cpu, False)
w = lt.Tensor.from_vector([1.0, 1.0, 1.0], [3], cpu, False)
b = lt.Tensor.from_vector([0.0, 0.0, 0.0], [3], cpu, False)
out = lt.Ops.batch_norm2d(x, mean, var, w, b, True)
print(list(out.shape), out.to_vector()[:4])
```

### `Ops.rmsnorm`

- **Signature:** `rmsnorm(input, weight, eps=1e-5) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:80`; C++ `include/litetorch/ops.h:81`
- **Parameters:** `input` — any rank. `weight` — same shape as the normalized trailing dim, required. `eps` — `float`, default `1e-5`.
- **Returns:** `input / sqrt(mean(input^2) + eps) * weight`, same shape as `input`. No mean subtraction, no bias term.
- **PyTorch:** no direct equivalent; `F.rms_norm(input, normalized_shape, weight, eps)` (torch 2.4+) computes the same values.

<!-- ex: id=r5-rmsnorm device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], cpu, False)
w = lt.Tensor.from_vector([1.0, 1.0, 1.0], [3], cpu, False)
out = lt.Ops.rmsnorm(x, w)
print(list(out.shape), out.to_vector()[:3])
```

### `Ops.fused_add_layernorm`

- **Signature:** `fused_add_layernorm(input, residual, normalized_shape, weight=None, bias=None, eps=1e-5) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:35`; C++ `include/litetorch/ops.h:82`
- **Parameters:** `input`, `residual` — same shape. `normalized_shape`, `weight`, `bias`, `eps` — same as `layer_norm`.
- **Returns:** `layer_norm(input + residual)`, computed in one fused kernel.
- **PyTorch:** no single equivalent; `F.layer_norm(input + residual, normalized_shape, weight, bias, eps)` computes the same values.

<!-- ex: id=r5-fused-add-layernorm device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], cpu, False)
r = lt.Tensor.from_vector([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [2, 3], cpu, False)
out = lt.Ops.fused_add_layernorm(x, r, [3])
print(list(out.shape), out.to_vector()[:3])
```

## Shape transforms (vision)

### `Ops.fold`

- **Signature:** `fold(input, output_size, kernel_size, dilation, padding, stride) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:168-169`; C++ `include/litetorch/ops.h:235-237`
- **Parameters:** `input` — `[N, C * prod(kernel_size), L]`. All six size arguments are lists of ints (no defaults — pass all six). `output_size` — `[H, W]`.
- **Returns:** `[N, C, H, W]`, combining sliding blocks back into an image (overlapping regions are summed).
- **PyTorch:** `F.fold(input, output_size, kernel_size, dilation, padding, stride)` — equivalent.

<!-- ex: id=r5-fold device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i) for i in range(16)], [1, 4, 4], cpu, False)
out = lt.Ops.fold(x, [4, 4], [2, 2], [1, 1], [0, 0], [2, 2])
print(list(out.shape), out.to_vector()[:6])
```

### `Ops.unfold`

- **Signature:** `unfold(input, kernel_size, dilation, padding, stride) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:170-171`; C++ `include/litetorch/ops.h:238-240`
- **Parameters:** `input` — `[N, C, H, W]`. `kernel_size`, `dilation`, `padding`, `stride` — lists of ints (no defaults — pass all four).
- **Returns:** `[N, C * prod(kernel_size), L]`, the sliding local blocks.
- **PyTorch:** `F.unfold(input, kernel_size, dilation, padding, stride)` — equivalent.

<!-- ex: id=r5-unfold device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i) for i in range(16)], [1, 1, 4, 4], cpu, False)
out = lt.Ops.unfold(x, [2, 2], [1, 1], [0, 0], [2, 2])
print(list(out.shape), out.to_vector()[:6])
```

### `Ops.bilinear`

- **Signature:** `bilinear(input1, input2, weight, bias=None) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:167`; C++ `include/litetorch/ops.h:233-234`
- **Parameters:** `input1` — `[N, in1]`. `input2` — `[N, in2]`. `weight` — `[out, in1, in2]`. `bias` — `[out]` or `None`.
- **Returns:** `[N, out]`, `out[n, o] = input1[n] @ weight[o] @ input2[n] + bias[o]`.
- **PyTorch:** `F.bilinear(input1, input2, weight, bias)` — equivalent.

<!-- ex: id=r5-bilinear device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], cpu, False)
b = lt.Tensor.from_vector([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0], [2, 4], cpu, False)
w = lt.Tensor.from_vector([float(i % 3) for i in range(60)], [5, 3, 4], cpu, False)
out = lt.Ops.bilinear(a, b, w)
print(list(out.shape), out.to_vector()[:6])
```

## Attention

### `Ops.qkv_extract`

- **Signature:** `qkv_extract(qkv, num_heads, index) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:81`; C++ `include/litetorch/ops.h:88`
- **Parameters:** `qkv` — `[B, T, 3 * H * D]` (packed query/key/value). `num_heads` — `H`. `index` — `0` for query, `1` for key, `2` for value.
- **Returns:** `[B, H, T, D]`, the selected component split out per head.
- **PyTorch:** no direct equivalent; `qkv.reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)[index]` computes the same values.

<!-- ex: id=r5-qkv-extract device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
qkv = lt.Tensor.from_vector([float(i) for i in range(12)], [1, 1, 12], cpu, False)
q = lt.Ops.qkv_extract(qkv, 2, 0)
k = lt.Ops.qkv_extract(qkv, 2, 1)
print(list(q.shape), q.to_vector(), k.to_vector()[:4])
```

### `Ops.rope`

- **Signature:** `rope(x, cos, sin) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:82`; C++ `include/litetorch/ops.h:90`
- **Parameters:** `x` — 4D `[B, H, T, D]`. `cos`, `sin` — `[T, D/2]` each (one frequency pair per position, half the head dim).
- **Returns:** same shape as `x`, rotary position embedding applied with interleaved pairs: for each consecutive pair `(x0, x1)` with frequency `(c, s)`, the output pair is `(x0 * c - x1 * s, x0 * s + x1 * c)`.
- **Edge cases:** `x` must be 4D (`RoPE input must be a 4D tensor [B, H, T, D]`); `cos`/`sin` must be exactly `[T, D/2]` or the call raises.
- **PyTorch:** no direct equivalent; the standard rotary embedding with interleaved pairs computes the same values (note: `torch.nn.functional` rotary helpers and HF implementations often use the half-split layout instead — do not mix the two).

<!-- ex: id=r5-rope device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i) for i in range(8)], [1, 1, 2, 4], cpu, False)
cos = lt.Tensor.from_vector([1.0, 0.9, 0.8, 0.7], [2, 2], cpu, False)
sin = lt.Tensor.from_vector([0.0, 0.1, 0.2, 0.3], [2, 2], cpu, False)
out = lt.Ops.rope(x, cos, sin)
print(list(out.shape), out.to_vector())
```

### `Ops.flash_attention`

- **Signature:** `flash_attention(q, k, v, causal=False) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:143`; C++ `include/litetorch/ops.h:86`
- **Parameters:** `q`, `k`, `v` — `[B, H, T, D]` each. `causal` — `bool`, default `False`.
- **Returns:** `[B, H, T, D]`, `softmax(q @ k.T / sqrt(D)) @ v` (with the causal mask when `causal=True`).
- **Edge cases:** head dim must satisfy `1 <= D <= 256` and `H % H_kv == 0` (D-20). There are no `scale` or `dropout_p` arguments — apply a scale to `q` manually before the call if you need one (`Ops.flash_attention(q * scale, k, v)`).
- **PyTorch:** `F.scaled_dot_product_attention(q, k, v, is_causal=causal)` — equivalent for the default scale; note LiteTorch has no `F.scaled_dot_product_attention` at all (D-20), this op is the replacement.

<!-- ex: id=r5-flash-attention device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
q = lt.Tensor.from_vector([0.1 * (i % 7 + 1) for i in range(64)], [1, 2, 4, 8], cpu, False)
k = lt.Tensor.from_vector([0.05 * (i % 5 + 1) for i in range(64)], [1, 2, 4, 8], cpu, False)
v = lt.Tensor.from_vector([0.3 * (i % 6 + 1) for i in range(64)], [1, 2, 4, 8], cpu, False)
out = lt.Ops.flash_attention(q, k, v)
print(list(out.shape), out.to_vector()[:4])
```

### `Ops.flash_attention_qkv`

- **Signature:** `flash_attention_qkv(qkv, num_heads, causal=False) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:144`; C++ `include/litetorch/ops.h:87`; implementation `src/ops/attention.cpp:785-849`
- **Parameters:** `qkv` — 3D `[B, T, 3*C]`, where `3*C` must be divisible by `3 * num_heads`; head dim `D = 3*C / (3 * num_heads)`. `num_heads` — `H`. `causal` — `bool`, default `False`.
- **Returns:** `[B, H, T, D]`, attention over the packed qkv (split internally, same math as `flash_attention`).
- **Edge cases:** FP32 and FP16 only. **GPU native backend required** — on CPU (or without the native lib) the call raises `flash_attention_qkv: GPU native backend required`, unlike `flash_attention`, which runs on CPU.
- **PyTorch:** no direct equivalent; split the packed qkv and call `F.scaled_dot_product_attention`.

```python skip-run
import litetorch as lt
gpu = lt.Device("cuda")
qkv = lt.Tensor.from_vector([0.1 * (i % 6 + 1) for i in range(192)], [1, 4, 48], gpu, False)
out = lt.Ops.flash_attention_qkv(qkv, 2)
print(list(out.shape))
```

> REQUIRES-GPU (B5): this example is not run on CPU — `flash_attention_qkv` throws without a GPU native backend. The call above is the exact form to run on Kaggle GPU.

### `Ops.paged_attention`

- **Signature:** `paged_attention(q, k_cache, v_cache, block_tables, context_lens, block_size) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:145`; C++ `include/litetorch/ops.h:91-98`
- **Parameters:** six positional arguments, no kwargs. `q` — rank 3 `[num_seqs, num_heads, head_dim]`. `k_cache` / `v_cache` — rank 4 `[num_blocks, num_kv_heads, block_size, head_dim]`. `block_tables` — rank 2 `[num_seqs, max_num_blocks_per_seq]`, INT32. `context_lens` — rank 1 `[num_seqs]`, INT32. `block_size` — `int`.
- **Returns:** `[num_seqs, num_heads, head_dim]`, attention over the cached positions gathered through the block table.
- **Edge cases:** `head_dim` must be at most 128. Rank-3 `k_cache`/`v_cache` raises `paged_attention: bad input ranks`; FP32 tables raise `block_tables and context_lens must be INT32`.
- **PyTorch:** no equivalent; this is the vLLM-style paged KV-cache attention.

<!-- ex: id=r5-paged-attention device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
q = lt.Tensor.from_vector([0.1] * 32, [1, 1, 32], cpu, False)
k_cache = lt.Tensor.from_vector([0.2] * (2 * 1 * 16 * 32), [2, 1, 16, 32], cpu, False)
v_cache = lt.Tensor.from_vector([0.3] * (2 * 1 * 16 * 32), [2, 1, 16, 32], cpu, False)
block_tables = lt.Tensor.from_vector([0.0, 1.0], [1, 2], cpu, False, lt.DataType.INT32)
context_lens = lt.Tensor.from_vector([32.0], [1], cpu, False, lt.DataType.INT32)
out = lt.Ops.paged_attention(q, k_cache, v_cache, block_tables, context_lens, 16)
print(list(out.shape), out.to_vector()[:4])
```

### `Ops.flash_decoding`

- **Signature:** `flash_decoding(q, k, v, num_splits) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:159`; C++ `include/litetorch/ops.h:114-119`
- **Parameters:** `q`, `k`, `v` — `[B, H, T, D]` each. `num_splits` — `int`, the number of KV splits the decode is parallelized over.
- **Returns:** `[B, H, T, D]`, same attention math as `flash_attention`.
- **PyTorch:** no direct equivalent; `F.scaled_dot_product_attention(q, k, v)` computes the same values (the split count is a performance knob, not a math change).

<!-- ex: id=r5-flash-decoding device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
q = lt.Tensor.from_vector([0.1 * (i % 7 + 1) for i in range(64)], [1, 2, 4, 8], cpu, False)
k = lt.Tensor.from_vector([0.05 * (i % 5 + 1) for i in range(64)], [1, 2, 4, 8], cpu, False)
v = lt.Tensor.from_vector([0.3 * (i % 6 + 1) for i in range(64)], [1, 2, 4, 8], cpu, False)
out = lt.Ops.flash_decoding(q, k, v, 2)
print(list(out.shape), out.to_vector()[:4])
```

### `Ops.ring_attention`

- **Signature:** `ring_attention(q, k, v, pg, causal=False) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:160`; C++ `include/litetorch/ops.h:107-113`
- **Parameters:** `q`, `k`, `v` — `[B, H, T, D]` each. `pg` — a `distributed.ProcessGroup` (the only way to obtain one is `ProcessGroup.get()` after `init_process_group`). `causal` — `bool`, default `False`.
- **Returns:** `[B, H, T, D]`, attention computed in a ring over ranks (context-parallel attention).
- **Edge cases:** **Uncallable from Python today (documented gap, not a docs problem).** `ProcessGroup.get()` is bound with a reference holder (`bind_distributed.cpp:23`), but the `pg` parameter is `std::shared_ptr<distributed::ProcessGroup>` — passing the returned group raises `RuntimeError: Unable to load a custom holder type from a default-holder instance`. There is no Python path to call this op until the binding is fixed (filed to Distributed, thread `docs-d9-fsdp-20261026`). Same class of latent gap as the `FSDPUnit` methods.
- **PyTorch:** no direct equivalent; ring attention is the context-parallel attention pattern.

```python skip-run
import litetorch as lt
lt.distributed.init_process_group(0, 2, "127.0.0.1", 29500)
pg = lt.distributed.ProcessGroup.get()
out = lt.Ops.ring_attention(q, k, v, pg, False)
```

> Documented gap: the call above raises the holder error — do not run it as written. Use manual sharding (see `howto/H2-distributed-fsdp-tp-pp-cp.md`) until the binding is fixed.

## LLM / fused kernels

### `Ops.scaled_matmul`

- **Signature:** `scaled_matmul(a, b, a_scale, b_scale, bias=None, out_dtype=DataType.FP32) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:146`; C++ `include/litetorch/ops.h:99-106`
- **Parameters:** `a` — `[M, K]`. `b` — `[K, N]`. `a_scale`, `b_scale` — `float` scales. `bias` — `[N]` or `None`. `out_dtype` — a `DataType`, default FP32.
- **Returns:** `(a @ b) * a_scale * b_scale (+ bias)`, in `out_dtype`.
- **PyTorch:** no direct equivalent; `(a @ b) * a_scale * b_scale` computes the same values.

<!-- ex: id=r5-scaled-matmul device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], cpu, False)
b = lt.Tensor.from_vector([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [3, 4], cpu, False)
out = lt.Ops.scaled_matmul(a, b, 0.5, 0.25)
print(list(out.shape), out.to_vector()[:4])
```

### `Ops.fused_swiglu`

- **Signature:** `fused_swiglu(gate, up) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:163`; C++ `include/litetorch/ops.h:226`
- **Parameters:** `gate`, `up` — same shape, broadcast-compatible.
- **Returns:** `silu(gate) * up`, elementwise, in one fused kernel. (Note: this is SwiGLU with the SiLU applied to `gate`; the `up` branch is not activated.)
- **PyTorch:** no single equivalent; `F.silu(gate) * up` computes the same values.

<!-- ex: id=r5-fused-swiglu device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
gate = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu, False)
up = lt.Tensor.from_vector([0.5, 1.5, -0.5, 2.0], [2, 2], cpu, False)
out = lt.Ops.fused_swiglu(gate, up)
print(list(out.shape), out.to_vector())
```

### `Ops.fused_qkv_proj`

- **Signature:** `fused_qkv_proj(x, wq, wk, wv) -> (Tensor, Tensor, Tensor)`
- **Source:** `src/bindings/bind_ops.cpp:164`; C++ `include/litetorch/ops.h:227-229`
- **Parameters:** `x` — `[..., in]`. `wq`, `wk`, `wv` — `[out, in]` each (Linear-style weights).
- **Returns:** a tuple `(q, k, v)`, each `x @ w.T` for the corresponding weight, computed in one fused kernel.
- **PyTorch:** no single equivalent; `(x @ wq.T, x @ wk.T, x @ wv.T)` computes the same values.

<!-- ex: id=r5-fused-qkv-proj device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i) for i in range(16)], [2, 8], cpu, False)
wq = lt.Tensor.from_vector([0.1 * (i % 5) for i in range(64)], [8, 8], cpu, False)
wk = lt.Tensor.from_vector([0.1 * (i % 3) for i in range(64)], [8, 8], cpu, False)
wv = lt.Tensor.from_vector([0.1 * (i % 7) for i in range(64)], [8, 8], cpu, False)
q, k, v = lt.Ops.fused_qkv_proj(x, wq, wk, wv)
print(list(q.shape), list(k.shape), list(v.shape), q.to_vector()[:4])
```

### `Ops.checkpoint`

- **Signature:** `checkpoint(module, input) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:161`; C++ `include/litetorch/ops.h:161`
- **Parameters:** `module` — an `nn.Module`. `input` — the module's input tensor.
- **Returns:** `module(input)`, computed with activation checkpointing: activations are freed after the forward pass and recomputed during the backward pass, trading compute for memory.
- **Edge cases:** the output values equal a plain `module(input)` call exactly — checkpointing changes memory use, not math. The module must be a LiteTorch `nn.Module`, not a bare function.
- **PyTorch:** `torch.utils.checkpoint.checkpoint(module, input)` — equivalent.

<!-- ex: id=r5-checkpoint device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(42)
mod = lt.nn.Sequential([lt.nn.Linear(4, 4, True)])
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], cpu, False)
direct = mod(x)
saved = lt.Ops.checkpoint(mod, x)
diff = (direct - saved).to_vector()
print(list(saved.shape), max(abs(d) for d in diff))
```

### `Ops.fused_linear_cross_entropy`

- See `ops-losses-quant.md` (`Ops.fused_linear_cross_entropy`) — the fused linear + cross-entropy op is documented with the loss functions. It computes `cross_entropy(x @ weight.T + bias, target)` in one kernel.

## Quantized ops

### `Ops.w8a8_matmul`

- **Signature:** `w8a8_matmul(x, w, x_scale, w_scale) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:147`; C++ `include/litetorch/ops.h:198-201`
- **Parameters:** `x` — `[M, K]`, FP32 only. `w` — `[N, K]`, FP32 only (note: `w` is transposed internally — this computes `x @ w.T`, not `x @ w`). `x_scale`, `w_scale` — `float` scales.
- **Returns:** `(round(x / x_scale) @ round(w / w_scale).T) * x_scale * w_scale`, FP32 `[M, N]` — both inputs are quantized to INT8 (clamped to `[-128, 127]`), multiplied in integer arithmetic, then dequantized. The quantization error is real: the result is only as close to `(x @ w.T) * x_scale * w_scale` as the INT8 rounding allows.
- **Edge cases:** both inputs must be FP32 — quantized tensors are not accepted as inputs here. The layout is `x @ w.T`, matching a Linear weight layout; passing a `[K, N]` weight silently computes the wrong product.
- **PyTorch:** no direct equivalent; `(torch.round(x / x_scale) @ torch.round(w / w_scale).T) * x_scale * w_scale` (with clamping to `[-128, 127]`) computes the same values.

<!-- ex: id=r5-w8a8-matmul device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], cpu, False)
w = lt.Tensor.from_vector([0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.9, 0.8, 0.7, 0.2, 0.1, 0.4], [4, 3], cpu, False)
out = lt.Ops.w8a8_matmul(x, w, 0.5, 0.25)
print(list(out.shape), out.to_vector()[:4])
```

### `Ops.fp8_cast_scaled`

- **Signature:** `fp8_cast_scaled(x, scale, dtype=DataType.FP8_E4M3) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:148`; C++ `include/litetorch/ops.h:202`
- **Parameters:** `x` — FP32. `scale` — `float`. `dtype` — a `DataType`, default `FP8_E4M3`.
- **Returns:** `round(x * scale)` stored in the FP8 format, same shape as `x`.
- **PyTorch:** no direct equivalent; `torch.float8_e4m3fn` casts exist but with a different (scale-free) API.

<!-- ex: id=r5-fp8-cast-scaled device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, -3.0, 0.5], [2, 2], cpu, False)
out = lt.Ops.fp8_cast_scaled(x, 2.0)
print(list(out.shape), out.dtype, out.to_vector())
```

### `Ops.fp8_uncast_scaled`

- **Signature:** `fp8_uncast_scaled(x, scale) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:149`; C++ `include/litetorch/ops.h:203`
- **Parameters:** `x` — an FP8 tensor (e.g. the output of `fp8_cast_scaled`). `scale` — `float`, the same scale used for the cast.
- **Returns:** the FP8 values divided by `scale`, FP32, same shape as `x`. Cast + uncast with the same scale round-trips exact values that are representable in FP8.
- **Edge cases:** a non-FP8 input raises `fp8_uncast_scaled: input must be FP8`.
- **PyTorch:** no direct equivalent.

<!-- ex: id=r5-fp8-uncast-scaled device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, -3.0, 0.5], [2, 2], cpu, False)
cast = lt.Ops.fp8_cast_scaled(x, 2.0)
back = lt.Ops.fp8_uncast_scaled(cast, 2.0)
print(list(back.shape), back.dtype, back.to_vector())
```

### `Ops.fp8_matmul`

- **Signature:** `fp8_matmul(a, b, a_scale, b_scale, out_scale, out_dtype=DataType.FP8_E4M3) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:150`; C++ `include/litetorch/ops.h:204`
- **Parameters:** `a`, `b` — FP8 tensors (already cast — pass FP32 here and the call raises). `a_scale`, `b_scale`, `out_scale` — `float`. `out_dtype` — a `DataType`, default `FP8_E4M3`.
- **Returns:** the dequantized product scaled by `out_scale`, in `out_dtype`. With `out_dtype=FP32` the result is the plain FP32 product of the dequantized inputs.
- **Edge cases:** inputs must already be FP8 — cast them with `fp8_cast_scaled` first. FP32 inputs raise `fp8_uncast_scaled: input must be FP8`.
- **PyTorch:** no direct equivalent; torch's `torch._scaled_mm` is the closest analogue (different API).

<!-- ex: id=r5-fp8-matmul device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu, False)
b = lt.Tensor.from_vector([1.0, 0.0, 0.0, 1.0], [2, 2], cpu, False)
ac = lt.Ops.fp8_cast_scaled(a, 1.0)
bc = lt.Ops.fp8_cast_scaled(b, 1.0)
out = lt.Ops.fp8_matmul(ac, bc, 1.0, 1.0, 1.0, lt.DataType.FP32)
print(list(out.shape), out.dtype, out.to_vector())
```

### `Ops.int4_pack`

- **Signature:** `int4_pack(x) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:151`; C++ `include/litetorch/ops.h:205`
- **Parameters:** `x` — INT8, values in `[-8, 7]`, any shape.
- **Returns:** INT8, `numel` halved, two 4-bit values packed per byte. The packed layout is internal — do not construct packed tensors by hand; use `int4_pack` / `int4_unpack`.
- **PyTorch:** no equivalent.

<!-- ex: id=r5-int4-pack device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, -2.0, -1.0, 0.0, 5.0], [8], cpu, False, lt.DataType.INT8)
packed = lt.Ops.int4_pack(x)
back = lt.Ops.int4_unpack(packed, 8)
print(list(packed.shape), packed.dtype, back.to_vector())
```

### `Ops.int4_unpack`

- **Signature:** `int4_unpack(packed, numel) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:152`; C++ `include/litetorch/ops.h:206`
- **Parameters:** `packed` — INT8, the output of `int4_pack`. `numel` — `int`, the number of values to unpack (must equal the original `numel`).
- **Returns:** INT8, `[numel]`, the original values.
- **Edge cases:** `numel` must match the packed buffer — unpacking with the wrong `numel` returns the wrong count (no error is raised for a too-small `numel`).
- **PyTorch:** no equivalent.

<!-- ex: id=r5-int4-unpack device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, -2.0, -1.0, 0.0, 5.0], [8], cpu, False, lt.DataType.INT8)
packed = lt.Ops.int4_pack(x)
back = lt.Ops.int4_unpack(packed, 8)
print(list(back.shape), back.dtype, back.to_vector())
```

### `Ops.int4_weight_only_linear`

- **Signature:** `int4_weight_only_linear(x, packed_w, w_scale, bias=None) -> Tensor`
- **Source:** `src/bindings/bind_ops.cpp:153`; C++ `include/litetorch/ops.h:207`
- **Parameters:** `x` — 2D `[M, K]`, FP32. `packed_w` — 2D INT8 `[N, (K+1)/2]`, the `int4_pack` of a `[N, K]` INT8 weight, reshaped to 2D. `w_scale` — `[N]`, one scale per output channel (`N = w_scale.shape[0]`). `bias` — `[N]` or `None`.
- **Returns:** `[M, N]`, `x @ dequant(packed_w).T (+ bias)`.
- **Edge cases:** `x` must be 2D; `packed_w` must be exactly 2D `[N, (K+1)/2]` or the call raises `packed_w shape mismatch` (a 1D packed tensor must be reshaped first).
- **PyTorch:** no direct equivalent; `(x @ (unpack(packed_w) * w_scale).T)` computes the same values.

<!-- ex: id=r5-int4-weight-only-linear device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], cpu, False)
w = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0], [16], cpu, False, lt.DataType.INT8)
packed = lt.Ops.int4_pack(w).reshape([4, 2])
w_scale = lt.Tensor.from_vector([1.0, 1.0, 1.0, 1.0], [4], cpu, False)
out = lt.Ops.int4_weight_only_linear(x, packed, w_scale)
print(list(out.shape), out.to_vector()[:6])
```

### `Ops.moe_gate`

- **Signature:** `moe_gate(input, gate_weight, top_k) -> (Tensor, Tensor)`
- **Source:** `src/bindings/bind_ops.cpp:154`; C++ `include/litetorch/ops.h:208-209`
- **Parameters:** `input` — `[T, H]` (tokens by hidden). `gate_weight` — `[num_experts, H]`. `top_k` — `int`.
- **Returns:** a tuple `(probs, indices)`: `probs` `[T, top_k]` FP32 — the softmax probabilities of the selected experts, **renormalized to sum to 1 over the selected top-k** (`probs[k] = softmax(logits)[k] / sum(top-k)`); `indices` `[T, top_k]` INT64 — the expert indices. Logits are `input @ gate_weight.T` (gate weight layout is `[num_experts, H]`).
- **Edge cases:** `gate_weight` must be `[num_experts, H]` — passing `[H, num_experts]` raises a matrix-shape error. `top_k` must satisfy `1 <= top_k <= num_experts`, else the call throws.
- **PyTorch:** no direct equivalent; `probs, indices = torch.topk(torch.softmax(input @ gate_weight.T, dim=-1), top_k, dim=-1)` followed by `probs = probs / probs.sum(dim=-1, keepdim=True)` computes the same values.

<!-- ex: id=r5-moe-gate device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], cpu, False)
gw = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], [3, 4], cpu, False)
probs, indices = lt.Ops.moe_gate(x, gw, 2)
print(list(probs.shape), list(indices.shape), indices.dtype, probs.to_vector(), indices.to_vector())
```
