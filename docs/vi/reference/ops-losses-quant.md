# Ops — Hàm loss, Lượng tử hóa, FP8, Fused

Các hàm loss của `litetorch.Ops`, fake quantization và các helper matmul lượng tử hóa FP8 / INT4 / W8A8. Các helper lượng tử hóa được ghi đầy đủ trong `ops-conv-attention.md` và được tham chiếu chéo ở đây.

- **Nguồn chuẩn:** `src/bindings/bind_ops.cpp:76-79,143-154,165-166,176-178` (binding Python), `include/litetorch/ops.h` (khai báo C++).
- **Cây:** agent-v1@02cc271a / HEAD cục bộ 89afd67 (baseline tài liệu; B5 kiểm tra lại ở head). Không có drift binding trong `bind_ops.cpp` / `ops.h` giữa 88a4ba1 và 89afd67.
- **Quy ước:** `cpu = lt.Device("cpu")` trong mọi ví dụ. Mọi ví dụ đều tự chứa và chạy được trên CPU. Parity với torch là FP32 1e-4 trừ khi header ghi khác.
- **Khuyết điểm đã biết (đã gửi cho Core 2026-10-26, thread `docs-fix-requests-20261026`):** `cross_entropy_loss`, `nll_loss` và `fused_linear_cross_entropy` từ chối target INT64 — xem các mục bên dưới. Cách làm việc: cast target sang INT32.

## Hàm loss

### `Ops.mse_loss`

- **Chữ ký:** `mse_loss(input, target) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:176`; C++ `include/litetorch/ops.h:152`
- **Tham số:** `input`, `target` — cùng shape.
- **Trả về:** vô hướng `[1]`, `mean((input - target)^2)`. Không có đối số `reduction` — loss luôn là trung bình.
- **PyTorch:** `F.mse_loss(input, target)` (mặc định `reduction="mean"`) — tương đương.

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

- **Chữ ký:** `l1_loss(input, target) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:177`; C++ `include/litetorch/ops.h:153`
- **Tham số:** `input`, `target` — cùng shape.
- **Trả về:** vô hướng `[1]`, `mean(|input - target|)`. Không có đối số `reduction` — loss luôn là trung bình.
- **PyTorch:** `F.l1_loss(input, target)` (mặc định `reduction="mean"`) — tương đương.

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

- **Chữ ký:** `bce_loss(input, target) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:178`; C++ `include/litetorch/ops.h:154`
- **Tham số:** `input` — xác suất trong `[0, 1]`, `target` — cùng shape, 0/1.
- **Trả về:** vô hướng `[1]`, `mean(-(target * log(input) + (1 - target) * log(1 - input)))`. Không có đối số `reduction` hay `weight`.
- **Trường hợp biên:** `input` được xem là xác suất, không phải logit — truyền logit cho `bce_with_logits_loss` thay thế. `input` bằng đúng 0 hoặc 1 làm loss vô hạn (`log(0)`).
- **PyTorch:** `F.binary_cross_entropy(input, target)` (mặc định `reduction="mean"`) — tương đương.

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

- **Chữ ký:** `cross_entropy_loss(input, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:76`; C++ `include/litetorch/ops.h:32`
- **Tham số:** `input` — `[N, C]` logit. `target` — `[N]`, chỉ số lớp. `weight` — `[C]` trọng số lớp hoặc `None`. `ignore_index` — `int`, mặc định `-100`. `reduction` — `"mean"` / `"sum"` / `"none"`, mặc định `"mean"`. `label_smoothing` — `float`, mặc định `0.0`.
- **Trả về:** vô hướng `[1]` với `"mean"`/`"sum"`, `[N]` với `"none"`.
- **Trường hợp biên:** **target phải là INT32 (hoặc FP32/INT16/INT8) — target INT64 throw** `cross_entropy_loss: target must have integer or float dtype`. Đây là khuyết điểm thật, không phải thiết kế: switch chỉ số target (`src/ops/loss.cpp:24-40`) không có case INT64. Torch dùng chỉ số lớp INT64 theo mặc định, nên code port phải cast trước (target tạo với `dtype=lt.DataType.INT32`). Đã gửi cho Core 2026-10-26 (thread `docs-fix-requests-20261026`). Cùng giới hạn áp dụng cho `nll_loss` và `fused_linear_cross_entropy`.
- **PyTorch:** `F.cross_entropy(input, target, weight, ignore_index, reduction, label_smoothing)` — tương đương khi target là INT32.

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

- **Chữ ký:** `nll_loss(input, target, weight=None, ignore_index=-100, reduction="mean") -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:77`; C++ `include/litetorch/ops.h:33`
- **Tham số:** `input` — `[N, C]` log-xác suất (ví dụ đầu ra của `log_softmax`). `target` — `[N]`, chỉ số lớp. `weight`, `ignore_index`, `reduction` — giống `cross_entropy_loss`.
- **Trả về:** vô hướng `[1]` với `"mean"`/`"sum"`, `[N]` với `"none"`.
- **Trường hợp biên:** cùng giới hạn INT64 như `cross_entropy_loss` — target INT64 throw; dùng INT32. Đầu vào phải đã là log-xác suất; logit thô cho giá trị sai (dùng `cross_entropy_loss` cho logit).
- **PyTorch:** `F.nll_loss(input, target, weight, ignore_index, reduction)` — tương đương khi target là INT32.

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

- **Chữ ký:** `bce_with_logits_loss(input, target, weight=None, reduction="mean", pos_weight=None) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:78`; C++ `include/litetorch/ops.h:34`
- **Tham số:** `input` — logit (shape bất kỳ). `target` — cùng shape, 0/1. `weight` — cùng shape hoặc `None`. `reduction` — `"mean"` / `"sum"` / `"none"`, mặc định `"mean"`. `pos_weight` — trọng số cho mỗi phần tử của lớp dương, hoặc `None`.
- **Trả về:** vô hướng `[1]` với `"mean"`/`"sum"`, cùng shape với `input` với `"none"`.
- **Trường hợp biên:** khác `bce_loss`, hàm này nhận logit thô — đừng áp sigmoid trước.
- **PyTorch:** `F.binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction)` — tương đương.

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

- **Chữ ký:** `fused_linear_cross_entropy(x, weight, target, bias=None) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:79`; C++ `include/litetorch/ops.h:35`
- **Tham số:** `x` — `[N, in]` hidden state. `weight` — `[C, in]` (trọng số classifier kiểu Linear). `target` — `[N]`, chỉ số lớp. `bias` — `[C]` hoặc `None`.
- **Trả về:** vô hướng `[1]`, `cross_entropy(x @ weight.T + bias, target)` — logit không bao giờ được materialize, nên tránh tensor trung gian `[N, C]` (tiết kiệm bộ nhớ này quan trọng với vocab lớn).
- **Trường hợp biên:** cùng giới hạn INT64 như `cross_entropy_loss` — target INT64 throw; dùng INT32. Không có đối số `reduction`, `weight`, `ignore_index` hay `label_smoothing` — luôn là CE mean trên toàn bộ target.
- **PyTorch:** không có hàm tương đương duy nhất; `F.cross_entropy(x @ weight.T + bias, target)` cho cùng giá trị.

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

## Lượng tử hóa

### `Ops.fake_quantize`

- **Chữ ký:** `fake_quantize(input, scale, zero_point=0.0, bits=8) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:165-166`; C++ `include/litetorch/ops.h:231-232`
- **Tham số:** `input` — shape bất kỳ, FP32. `scale` — `float`. `zero_point` — `float`, mặc định `0.0`. `bits` — `int`, mặc định `8`.
- **Trả về:** `round((input - zero_point) / scale) * scale + zero_point` (theo khoảng của `bits`), cùng shape và kiểu dữ liệu với `input` — các giá trị được lượng tử hóa nhưng vẫn ở FP32 (fake quantization cho quantization-aware training).
- **PyTorch:** không có hàm tương đương trực tiếp; `torch.fake_quantize_per_tensor_affine(input, scale, zero_point, -128, 127)` cho cùng giá trị với `bits=8`.

<!-- ex: id=r6-fake-quantize device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.03, 2.07, -0.96, 0.04], [4], cpu, False)
out = lt.Ops.fake_quantize(x, 0.1)
print(list(out.shape), out.to_vector())
```

## Helper matmul lượng tử hóa (ghi trong `ops-conv-attention.md`)

Các op này nằm cùng các kernel LLM ở R5; các mục dưới đây là tham chiếu — chữ ký đầy đủ, tham số, trường hợp biên và ví dụ chạy được nằm trong `ops-conv-attention.md`.

- `Ops.w8a8_matmul(x, w, x_scale, w_scale)` — đầu vào FP32, lượng tử hóa cả hai sang INT8 rồi tính `(round(x / x_scale) @ round(w / w_scale).T) * x_scale * w_scale`. Xem R5 "Ops lượng tử hóa".
- `Ops.fp8_cast_scaled(x, scale, dtype=FP8_E4M3)` — FP32 sang FP8 với một scale. Xem R5 "Ops lượng tử hóa".
- `Ops.fp8_uncast_scaled(x, scale)` — FP8 trở lại FP32. Xem R5 "Ops lượng tử hóa".
- `Ops.fp8_matmul(a, b, a_scale, b_scale, out_scale, out_dtype=FP8_E4M3)` — chỉ nhận đầu vào FP8; cast trước. Xem R5 "Ops lượng tử hóa".
- `Ops.int4_pack(x)` / `Ops.int4_unpack(packed, numel)` — round-trip INT8, hai giá trị 4-bit trong một byte. Xem R5 "Ops lượng tử hóa".
- `Ops.int4_weight_only_linear(x, packed_w, w_scale, bias=None)` — `x` 2D, `packed_w` 2D `[N, (K+1)/2]`, `w_scale` `[N]` cho mỗi kênh. Xem R5 "Ops lượng tử hóa".
