# Ops — Tích chập, Pooling, Chuẩn hóa, Attention, LLM

Các hàm `litetorch.Ops` cho tích chập, pooling, chuẩn hóa, attention và các kernel LLM (fused, lượng tử hóa và serving). Toàn bộ tính toán là C++14; Python chỉ là lớp giao diện pybind11.

- **Nguồn chuẩn:** `src/bindings/bind_ops.cpp:37-42,69-83,143-164,174-178` (binding Python), `include/litetorch/ops.h` (khai báo C++).
- **Cây:** agent-v1@02cc271a / HEAD cục bộ 89afd67 (baseline tài liệu; B5 kiểm tra lại ở head). Không có drift binding trong `bind_ops.cpp` / `ops.h` giữa 88a4ba1 và 89afd67.
- **Quy ước:** `cpu = lt.Device("cpu")` trong mọi ví dụ. Mọi ví dụ đều tự chứa và chạy được trên CPU trừ khi ghi REQUIRES-GPU. Parity với torch là FP32 1e-4 trừ khi header ghi khác.
- **Sai lệch liên quan:** D-20 (không có `F.scaled_dot_product_attention` — chỉ có `Ops.flash_attention`; `flash_attention_qkv` chỉ chạy trên GPU), D-09 (GELU chỉ dùng tanh — xem `ops-elementwise.md`). Xem `plan/differences.md`.

## Tích chập

### `Ops.conv1d`

- **Chữ ký:** `conv1d(input, weight, bias=None, stride=1, padding=0) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:174`; C++ `include/litetorch/ops.h:150`
- **Tham số:** `input` — `[N, C_in, L]`. `weight` — `[C_out, C_in, K]`. `bias` — `[C_out]` hoặc `None`. `stride`, `padding` — chỉ nhận `int` vô hướng, không nhận tuple.
- **Trả về:** `[N, C_out, L_out]`, `L_out = floor((L + 2 * padding - K) / stride) + 1`.
- **Trường hợp biên:** không có đối số dilation hay groups (khác torch); `stride`/`padding` vô hướng áp dụng cho chiều không gian duy nhất. `C_in` không khớp sẽ raise.
- **PyTorch:** `F.conv1d(input, weight, bias, stride, padding)` — tương đương với stride/padding vô hướng; torch còn nhận tuple stride/padding, dilation, groups, mà LiteTorch không có.

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

- **Chữ ký:** `conv2d(input, weight, bias=None, stride=1, padding=0) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:37`; C++ `include/litetorch/ops.h:26`
- **Tham số:** `input` — `[N, C_in, H, W]`. `weight` — `[C_out, C_in, K_H, K_W]`. `bias` — `[C_out]` hoặc `None`. `stride`, `padding` — chỉ nhận `int` vô hướng, áp dụng cho cả hai chiều không gian.
- **Trả về:** `[N, C_out, H_out, W_out]`.
- **Trường hợp biên:** không có đối số dilation hay groups; không nhận tuple stride/padding. Một `stride`/`padding` vô hướng bao cả hai chiều.
- **PyTorch:** `F.conv2d(input, weight, bias, stride, padding)` — tương đương với stride/padding vô hướng; torch còn nhận tuple stride/padding, dilation, groups.

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

- **Chữ ký:** `conv2d_relu(input, weight, bias=None, stride=1, padding=0) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:38`; C++ `include/litetorch/ops.h:27`
- **Tham số:** giống `conv2d`.
- **Trả về:** `relu(conv2d(input, weight, bias, stride, padding))`, fused trong một lần gọi kernel.
- **Trường hợp biên:** cùng giới hạn như `conv2d` (stride/padding vô hướng, không dilation/groups). Các đầu ra âm bị ReLU fused đưa về 0 — đừng kỳ vọng giá trị âm dù tích chập thô có âm.
- **PyTorch:** không có hàm tương đương duy nhất; `F.relu(F.conv2d(...))` cho cùng giá trị.

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

- **Chữ ký:** `conv3d(input, weight, bias=None, stride=1, padding=0) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:39`; C++ `include/litetorch/ops.h:28`
- **Tham số:** `input` — `[N, C_in, D, H, W]`. `weight` — `[C_out, C_in, K_D, K_H, K_W]`. `bias` — `[C_out]` hoặc `None`. `stride`, `padding` — chỉ nhận `int` vô hướng.
- **Trả về:** `[N, C_out, D_out, H_out, W_out]`.
- **Trường hợp biên:** cùng giới hạn như `conv2d` (stride/padding vô hướng, không dilation/groups).
- **PyTorch:** `F.conv3d(input, weight, bias, stride, padding)` — tương đương với stride/padding vô hướng.

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

- **Chữ ký:** `max_pool2d(input, kernel_size, stride=-1, padding=0) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:40`; C++ `include/litetorch/ops.h:69`
- **Tham số:** `input` — `[N, C, H, W]`. `kernel_size` — `int`. `stride` — `int`, mặc định `-1` nghĩa là stride bằng `kernel_size`. `padding` — `int`.
- **Trả về:** `[N, C, H_out, W_out]`, giá trị lớn nhất mỗi cửa sổ. Không trả về indices (khác torch, vốn có thể trả về).
- **Trường hợp biên:** `kernel_size` lớn hơn đầu vào sẽ raise; `stride=-1` là cách duy nhất để viết "stride = kernel size".
- **PyTorch:** `F.max_pool2d(input, kernel_size, stride, padding)` — tương đương khi không cần indices; torch chỉ trả indices khi được yêu cầu.

<!-- ex: id=r5-max-pool2d device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i) for i in range(32)], [1, 2, 4, 4], cpu, False)
out = lt.Ops.max_pool2d(x, 2)
print(list(out.shape), out.to_vector()[:4])
```

### `Ops.max_pool3d`

- **Chữ ký:** `max_pool3d(input, kernel_size, stride=-1, padding=0) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:41`; C++ `include/litetorch/ops.h:70`
- **Tham số:** `input` — `[N, C, D, H, W]`. `kernel_size`, `stride`, `padding` — cùng quy ước như `max_pool2d`.
- **Trả về:** `[N, C, D_out, H_out, W_out]`, giá trị lớn nhất mỗi cửa sổ, không có indices.
- **PyTorch:** `F.max_pool3d(input, kernel_size, stride, padding)` — tương đương khi không cần indices.

<!-- ex: id=r5-max-pool3d device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i % 7) for i in range(128)], [1, 2, 4, 4, 4], cpu, False)
out = lt.Ops.max_pool3d(x, 2)
print(list(out.shape), out.to_vector()[:4])
```

### `Ops.avg_pool2d`

- **Chữ ký:** `avg_pool2d(input, kernel_size, stride=-1, padding=0) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:175`; C++ `include/litetorch/ops.h:151`
- **Tham số:** `input` — `[N, C, H, W]`. `kernel_size`, `stride` (mặc định `-1` = kernel size), `padding` — cùng quy ước như `max_pool2d`.
- **Trả về:** `[N, C, H_out, W_out]`, trung bình mỗi cửa sổ.
- **PyTorch:** `F.avg_pool2d(input, kernel_size, stride, padding)` — tương đương.

<!-- ex: id=r5-avg-pool2d device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i) for i in range(32)], [1, 2, 4, 4], cpu, False)
out = lt.Ops.avg_pool2d(x, 2)
print(list(out.shape), out.to_vector()[:4])
```

### `Ops.adaptive_avg_pool2d`

- **Chữ ký:** `adaptive_avg_pool2d(input, output_height, output_width) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:42`; C++ `include/litetorch/ops.h:83`
- **Tham số:** `input` — `[N, C, H, W]`. `output_height`, `output_width` — kích thước đích dạng `int`.
- **Trả về:** `[N, C, output_height, output_width]`, trung bình mỗi cửa sổ trên các bin thích ứng.
- **PyTorch:** `F.adaptive_avg_pool2d(input, (output_height, output_width))` — tương đương.

<!-- ex: id=r5-adaptive-avg-pool2d device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i) for i in range(32)], [1, 2, 4, 4], cpu, False)
out = lt.Ops.adaptive_avg_pool2d(x, 2, 2)
print(list(out.shape), out.to_vector()[:4])
```

## Chuẩn hóa

### `Ops.layer_norm`

- **Chữ ký:** `layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:34`; C++ `include/litetorch/ops.h:80`
- **Tham số:** `input` — hạng bất kỳ. `normalized_shape` — danh sách int (các chiều cuối cần chuẩn hóa). `weight` / `bias` — tensor khớp `normalized_shape`, hoặc `None`. `eps` — `float`, mặc định `1e-5`.
- **Trả về:** cùng shape với `input`, chuẩn hóa về trung bình 0 và phương sai đơn vị trên `normalized_shape`, rồi nhân `weight` và cộng `bias` khi có.
- **Trường hợp biên:** `normalized_shape` phải là danh sách — torch còn nhận một int đơn, LiteTorch không.
- **PyTorch:** `F.layer_norm(input, normalized_shape, weight, bias, eps)` — tương đương.

<!-- ex: id=r5-layer-norm device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], cpu, False)
out = lt.Ops.layer_norm(x, [3])
print(list(out.shape), out.to_vector()[:3])
```

### `Ops.batch_norm2d`

- **Chữ ký:** `batch_norm2d(input, running_mean, running_var, weight, bias, training, momentum=0.1, eps=1e-5) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:36`; C++ `include/litetorch/ops.h:71`
- **Tham số:** `input` — `[N, C, H, W]`. `running_mean` / `running_var` / `weight` / `bias` — mỗi cái `[C]`, đều bắt buộc (không có mặc định `None`). `training` — `bool`: `True` dùng thống kê batch và cập nhật running stats; `False` dùng running stats. `momentum` — `float`, mặc định `0.1`. `eps` — `float`, mặc định `1e-5`.
- **Trả về:** cùng shape với `input`.
- **Trường hợp biên:** cả bốn tensor `[C]` đều bắt buộc — không có dạng không-affine ở tầng `Ops` (dùng module `nn.BatchNorm2d` để có affine tùy chọn). `training=True` trên batch một mẫu sẽ chuẩn hóa bằng thống kê của chính batch đó.
- **PyTorch:** `F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)` — tương đương.

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

- **Chữ ký:** `rmsnorm(input, weight, eps=1e-5) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:80`; C++ `include/litetorch/ops.h:81`
- **Tham số:** `input` — hạng bất kỳ. `weight` — cùng shape với chiều cuối được chuẩn hóa, bắt buộc. `eps` — `float`, mặc định `1e-5`.
- **Trả về:** `input / sqrt(mean(input^2) + eps) * weight`, cùng shape với `input`. Không trừ trung bình, không có bias.
- **PyTorch:** không có hàm tương đương trực tiếp; `F.rms_norm(input, normalized_shape, weight, eps)` (torch 2.4+) cho cùng giá trị.

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

- **Chữ ký:** `fused_add_layernorm(input, residual, normalized_shape, weight=None, bias=None, eps=1e-5) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:35`; C++ `include/litetorch/ops.h:82`
- **Tham số:** `input`, `residual` — cùng shape. `normalized_shape`, `weight`, `bias`, `eps` — giống `layer_norm`.
- **Trả về:** `layer_norm(input + residual)`, tính trong một kernel fused.
- **PyTorch:** không có hàm tương đương duy nhất; `F.layer_norm(input + residual, normalized_shape, weight, bias, eps)` cho cùng giá trị.

<!-- ex: id=r5-fused-add-layernorm device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], cpu, False)
r = lt.Tensor.from_vector([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [2, 3], cpu, False)
out = lt.Ops.fused_add_layernorm(x, r, [3])
print(list(out.shape), out.to_vector()[:3])
```

## Biến đổi shape (vision)

### `Ops.fold`

- **Chữ ký:** `fold(input, output_size, kernel_size, dilation, padding, stride) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:168-169`; C++ `include/litetorch/ops.h:235-237`
- **Tham số:** `input` — `[N, C * prod(kernel_size), L]`. Cả sáu đối số kích thước đều là danh sách int (không có mặc định — truyền đủ sáu). `output_size` — `[H, W]`.
- **Trả về:** `[N, C, H, W]`, ghép các khối trượt trở lại thành ảnh (vùng chồng lấp được cộng dồn).
- **PyTorch:** `F.fold(input, output_size, kernel_size, dilation, padding, stride)` — tương đương.

<!-- ex: id=r5-fold device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i) for i in range(16)], [1, 4, 4], cpu, False)
out = lt.Ops.fold(x, [4, 4], [2, 2], [1, 1], [0, 0], [2, 2])
print(list(out.shape), out.to_vector()[:6])
```

### `Ops.unfold`

- **Chữ ký:** `unfold(input, kernel_size, dilation, padding, stride) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:170-171`; C++ `include/litetorch/ops.h:238-240`
- **Tham số:** `input` — `[N, C, H, W]`. `kernel_size`, `dilation`, `padding`, `stride` — danh sách int (không có mặc định — truyền đủ bốn).
- **Trả về:** `[N, C * prod(kernel_size), L]`, các khối cục bộ trượt.
- **PyTorch:** `F.unfold(input, kernel_size, dilation, padding, stride)` — tương đương.

<!-- ex: id=r5-unfold device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([float(i) for i in range(16)], [1, 1, 4, 4], cpu, False)
out = lt.Ops.unfold(x, [2, 2], [1, 1], [0, 0], [2, 2])
print(list(out.shape), out.to_vector()[:6])
```

### `Ops.bilinear`

- **Chữ ký:** `bilinear(input1, input2, weight, bias=None) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:167`; C++ `include/litetorch/ops.h:233-234`
- **Tham số:** `input1` — `[N, in1]`. `input2` — `[N, in2]`. `weight` — `[out, in1, in2]`. `bias` — `[out]` hoặc `None`.
- **Trả về:** `[N, out]`, `out[n, o] = input1[n] @ weight[o] @ input2[n] + bias[o]`.
- **PyTorch:** `F.bilinear(input1, input2, weight, bias)` — tương đương.

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

- **Chữ ký:** `qkv_extract(qkv, num_heads, index) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:81`; C++ `include/litetorch/ops.h:88`
- **Tham số:** `qkv` — `[B, T, 3 * H * D]` (query/key/value đóng gói). `num_heads` — `H`. `index` — `0` cho query, `1` cho key, `2` cho value.
- **Trả về:** `[B, H, T, D]`, thành phần được chọn, tách theo từng head.
- **PyTorch:** không có hàm tương đương trực tiếp; `qkv.reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)[index]` cho cùng giá trị.

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

- **Chữ ký:** `rope(x, cos, sin) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:82`; C++ `include/litetorch/ops.h:90`
- **Tham số:** `x` — 4D `[B, H, T, D]`. `cos`, `sin` — mỗi cái `[T, D/2]` (một cặp tần số cho mỗi vị trí, bằng một nửa chiều head).
- **Trả về:** cùng shape với `x`, rotary position embedding áp dụng theo cặp xen kẽ: với mỗi cặp liên tiếp `(x0, x1)` và tần số `(c, s)`, cặp đầu ra là `(x0 * c - x1 * s, x0 * s + x1 * c)`.
- **Trường hợp biên:** `x` phải là 4D (`RoPE input must be a 4D tensor [B, H, T, D]`); `cos`/`sin` phải đúng `[T, D/2]` nếu không sẽ raise.
- **PyTorch:** không có hàm tương đương trực tiếp; rotary embedding chuẩn với các cặp xen kẽ cho cùng giá trị (lưu ý: các helper rotary của `torch.nn.functional` và HF thường dùng layout chia nửa thay vì xen kẽ — đừng trộn hai layout).

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

- **Chữ ký:** `flash_attention(q, k, v, causal=False) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:143`; C++ `include/litetorch/ops.h:86`
- **Tham số:** `q`, `k`, `v` — mỗi cái `[B, H, T, D]`. `causal` — `bool`, mặc định `False`.
- **Trả về:** `[B, H, T, D]`, `softmax(q @ k.T / sqrt(D)) @ v` (có mặt nạ causal khi `causal=True`).
- **Trường hợp biên:** chiều head phải thỏa `1 <= D <= 256` và `H % H_kv == 0` (D-20). Không có đối số `scale` hay `dropout_p` — cần scale thì nhân vào `q` trước khi gọi (`Ops.flash_attention(q * scale, k, v)`).
- **PyTorch:** `F.scaled_dot_product_attention(q, k, v, is_causal=causal)` — tương đương với scale mặc định; lưu ý LiteTorch hoàn toàn không có `F.scaled_dot_product_attention` (D-20), op này là bản thay thế.

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

- **Chữ ký:** `flash_attention_qkv(qkv, num_heads, causal=False) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:144`; C++ `include/litetorch/ops.h:87`; triển khai `src/ops/attention.cpp:785-849`
- **Tham số:** `qkv` — 3D `[B, T, 3*C]`, trong đó `3*C` phải chia hết cho `3 * num_heads`; chiều head `D = 3*C / (3 * num_heads)`. `num_heads` — `H`. `causal` — `bool`, mặc định `False`.
- **Trả về:** `[B, H, T, D]`, attention trên qkv đóng gói (tách nội bộ, cùng công thức như `flash_attention`).
- **Trường hợp biên:** chỉ FP32 và FP16. **Cần GPU native backend** — trên CPU (hoặc thiếu lib native) lệnh gọi raise `flash_attention_qkv: GPU native backend required`, khác `flash_attention` vốn chạy được trên CPU.
- **PyTorch:** không có hàm tương đương trực tiếp; tách qkv đóng gói rồi gọi `F.scaled_dot_product_attention`.

```python skip-run
import litetorch as lt
gpu = lt.Device("cuda")
qkv = lt.Tensor.from_vector([0.1 * (i % 6 + 1) for i in range(192)], [1, 4, 48], gpu, False)
out = lt.Ops.flash_attention_qkv(qkv, 2)
print(list(out.shape))
```

> REQUIRES-GPU (B5): ví dụ này không chạy trên CPU — `flash_attention_qkv` throw khi không có GPU native backend. Lệnh gọi trên là dạng chính xác để chạy trên Kaggle GPU.

### `Ops.paged_attention`

- **Chữ ký:** `paged_attention(q, k_cache, v_cache, block_tables, context_lens, block_size) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:145`; C++ `include/litetorch/ops.h:91-98`
- **Tham số:** sáu đối số vị trí, không có kwargs. `q` — hạng 3 `[num_seqs, num_heads, head_dim]`. `k_cache` / `v_cache` — hạng 4 `[num_blocks, num_kv_heads, block_size, head_dim]`. `block_tables` — hạng 2 `[num_seqs, max_num_blocks_per_seq]`, INT32. `context_lens` — hạng 1 `[num_seqs]`, INT32. `block_size` — `int`.
- **Trả về:** `[num_seqs, num_heads, head_dim]`, attention trên các vị trí đã cache, thu thập qua bảng block.
- **Trường hợp biên:** `head_dim` tối đa 128. `k_cache`/`v_cache` hạng 3 sẽ raise `paged_attention: bad input ranks`; bảng dạng FP32 raise `block_tables and context_lens must be INT32`.
- **PyTorch:** không có hàm tương đương; đây là attention KV-cache phân trang kiểu vLLM.

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

- **Chữ ký:** `flash_decoding(q, k, v, num_splits) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:159`; C++ `include/litetorch/ops.h:114-119`
- **Tham số:** `q`, `k`, `v` — mỗi cái `[B, H, T, D]`. `num_splits` — `int`, số phần KV mà phép decode được song song hóa.
- **Trả về:** `[B, H, T, D]`, cùng công thức attention như `flash_attention`.
- **PyTorch:** không có hàm tương đương trực tiếp; `F.scaled_dot_product_attention(q, k, v)` cho cùng giá trị (số split là knob hiệu năng, không đổi công thức).

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

- **Chữ ký:** `ring_attention(q, k, v, pg, causal=False) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:160`; C++ `include/litetorch/ops.h:107-113`
- **Tham số:** `q`, `k`, `v` — mỗi cái `[B, H, T, D]`. `pg` — một `distributed.ProcessGroup` (cách duy nhất để lấy là `ProcessGroup.get()` sau `init_process_group`). `causal` — `bool`, mặc định `False`.
- **Trả về:** `[B, H, T, D]`, attention tính theo vòng qua các rank (attention song song theo ngữ cảnh).
- **Trường hợp biên:** **Hiện không gọi được từ Python (khoảng trống đã ghi nhận, không phải lỗi tài liệu).** `ProcessGroup.get()` được bind với holder tham chiếu (`bind_distributed.cpp:23`), nhưng tham số `pg` là `std::shared_ptr<distributed::ProcessGroup>` — truyền group trả về sẽ raise `RuntimeError: Unable to load a custom holder type from a default-holder instance`. Không có đường Python nào gọi được op này cho đến khi binding được sửa (đã gửi cho Distributed, thread `docs-d9-fsdp-20261026`). Cùng lớp khoảng trống tiềm ẩn như các phương thức `FSDPUnit`.
- **PyTorch:** không có hàm tương đương trực tiếp; ring attention là mẫu attention song song theo ngữ cảnh.

```python skip-run
import litetorch as lt
lt.distributed.init_process_group(0, 2, "127.0.0.1", 29500)
pg = lt.distributed.ProcessGroup.get()
out = lt.Ops.ring_attention(q, k, v, pg, False)
```

> Khoảng trống đã ghi nhận: lệnh gọi trên raise lỗi holder — đừng chạy như đã viết. Dùng sharding thủ công (xem `howto/H2-distributed-fsdp-tp-pp-cp.md`) cho đến khi binding được sửa.

## LLM / kernel fused

### `Ops.scaled_matmul`

- **Chữ ký:** `scaled_matmul(a, b, a_scale, b_scale, bias=None, out_dtype=DataType.FP32) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:146`; C++ `include/litetorch/ops.h:99-106`
- **Tham số:** `a` — `[M, K]`. `b` — `[K, N]`. `a_scale`, `b_scale` — các scale dạng `float`. `bias` — `[N]` hoặc `None`. `out_dtype` — một `DataType`, mặc định FP32.
- **Trả về:** `(a @ b) * a_scale * b_scale (+ bias)`, theo `out_dtype`.
- **PyTorch:** không có hàm tương đương trực tiếp; `(a @ b) * a_scale * b_scale` cho cùng giá trị.

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

- **Chữ ký:** `fused_swiglu(gate, up) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:163`; C++ `include/litetorch/ops.h:226`
- **Tham số:** `gate`, `up` — cùng shape, tương thích broadcast.
- **Trả về:** `silu(gate) * up`, theo từng phần tử, trong một kernel fused. (Lưu ý: đây là SwiGLU với SiLU áp dụng cho `gate`; nhánh `up` không được kích hoạt.)
- **PyTorch:** không có hàm tương đương duy nhất; `F.silu(gate) * up` cho cùng giá trị.

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

- **Chữ ký:** `fused_qkv_proj(x, wq, wk, wv) -> (Tensor, Tensor, Tensor)`
- **Nguồn:** `src/bindings/bind_ops.cpp:164`; C++ `include/litetorch/ops.h:227-229`
- **Tham số:** `x` — `[..., in]`. `wq`, `wk`, `wv` — mỗi cái `[out, in]` (trọng số kiểu Linear).
- **Trả về:** một tuple `(q, k, v)`, mỗi cái là `x @ w.T` với trọng số tương ứng, tính trong một kernel fused.
- **PyTorch:** không có hàm tương đương duy nhất; `(x @ wq.T, x @ wk.T, x @ wv.T)` cho cùng giá trị.

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

- **Chữ ký:** `checkpoint(module, input) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:161`; C++ `include/litetorch/ops.h:161`
- **Tham số:** `module` — một `nn.Module`. `input` — tensor đầu vào của module.
- **Trả về:** `module(input)`, tính với activation checkpointing: các activation được giải phóng sau forward pass và tính lại trong backward pass, đổi compute lấy bộ nhớ.
- **Trường hợp biên:** giá trị đầu ra bằng chính xác một lệnh gọi `module(input)` thông thường — checkpointing thay đổi mức dùng bộ nhớ, không đổi công thức. Module phải là `nn.Module` của LiteTorch, không phải hàm trần.
- **PyTorch:** `torch.utils.checkpoint.checkpoint(module, input)` — tương đương.

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

- Xem `ops-losses-quant.md` (`Ops.fused_linear_cross_entropy`) — op linear + cross-entropy fused được ghi cùng các hàm loss. Nó tính `cross_entropy(x @ weight.T + bias, target)` trong một kernel.

## Ops lượng tử hóa

### `Ops.w8a8_matmul`

- **Chữ ký:** `w8a8_matmul(x, w, x_scale, w_scale) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:147`; C++ `include/litetorch/ops.h:198-201`
- **Tham số:** `x` — `[M, K]`, chỉ FP32. `w` — `[N, K]`, chỉ FP32 (lưu ý: `w` được chuyển vị nội bộ — phép này tính `x @ w.T`, không phải `x @ w`). `x_scale`, `w_scale` — các scale dạng `float`.
- **Trả về:** `(round(x / x_scale) @ round(w / w_scale).T) * x_scale * w_scale`, FP32 `[M, N]` — cả hai đầu vào được lượng tử hóa sang INT8 (kẹp trong `[-128, 127]`), nhân bằng số học nguyên, rồi giải lượng tử. Sai số lượng tử hóa là thật: kết quả chỉ gần `(x @ w.T) * x_scale * w_scale` trong giới hạn làm tròn INT8.
- **Trường hợp biên:** cả hai đầu vào phải là FP32 — tensor lượng tử hóa không được nhận ở đây. Layout là `x @ w.T`, khớp layout trọng số Linear; truyền nhầm trọng số `[K, N]` sẽ âm thầm tính sai tích.
- **PyTorch:** không có hàm tương đương trực tiếp; `(torch.round(x / x_scale) @ torch.round(w / w_scale).T) * x_scale * w_scale` (với kẹp trong `[-128, 127]`) cho cùng giá trị.

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

- **Chữ ký:** `fp8_cast_scaled(x, scale, dtype=DataType.FP8_E4M3) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:148`; C++ `include/litetorch/ops.h:202`
- **Tham số:** `x` — FP32. `scale` — `float`. `dtype` — một `DataType`, mặc định `FP8_E4M3`.
- **Trả về:** `round(x * scale)` lưu ở định dạng FP8, cùng shape với `x`.
- **PyTorch:** không có hàm tương đương trực tiếp; các phép cast `torch.float8_e4m3fn` tồn tại nhưng với API khác (không có scale).

<!-- ex: id=r5-fp8-cast-scaled device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, -3.0, 0.5], [2, 2], cpu, False)
out = lt.Ops.fp8_cast_scaled(x, 2.0)
print(list(out.shape), out.dtype, out.to_vector())
```

### `Ops.fp8_uncast_scaled`

- **Chữ ký:** `fp8_uncast_scaled(x, scale) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:149`; C++ `include/litetorch/ops.h:203`
- **Tham số:** `x` — một tensor FP8 (ví dụ đầu ra của `fp8_cast_scaled`). `scale` — `float`, cùng scale dùng khi cast.
- **Trả về:** các giá trị FP8 chia cho `scale`, FP32, cùng shape với `x`. Cast rồi uncast với cùng scale cho round-trip chính xác với các giá trị biểu diễn được trong FP8.
- **Trường hợp biên:** đầu vào không phải FP8 sẽ raise `fp8_uncast_scaled: input must be FP8`.
- **PyTorch:** không có hàm tương đương trực tiếp.

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

- **Chữ ký:** `fp8_matmul(a, b, a_scale, b_scale, out_scale, out_dtype=DataType.FP8_E4M3) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:150`; C++ `include/litetorch/ops.h:204`
- **Tham số:** `a`, `b` — các tensor FP8 (đã cast — truyền FP32 vào đây sẽ raise). `a_scale`, `b_scale`, `out_scale` — `float`. `out_dtype` — một `DataType`, mặc định `FP8_E4M3`.
- **Trả về:** tích đã giải lượng tử nhân với `out_scale`, theo `out_dtype`. Với `out_dtype=FP32` kết quả là tích FP32 thuần của các đầu vào đã giải lượng tử.
- **Trường hợp biên:** đầu vào phải đã là FP8 — cast trước bằng `fp8_cast_scaled`. Đầu vào FP32 raise `fp8_uncast_scaled: input must be FP8`.
- **PyTorch:** không có hàm tương đương trực tiếp; `torch._scaled_mm` của torch là tương tự gần nhất (API khác).

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

- **Chữ ký:** `int4_pack(x) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:151`; C++ `include/litetorch/ops.h:205`
- **Tham số:** `x` — INT8, giá trị trong `[-8, 7]`, shape bất kỳ.
- **Trả về:** INT8, `numel` giảm một nửa, hai giá trị 4-bit đóng gói trong một byte. Layout đóng gói là nội bộ — đừng tự tạo tensor đóng gói bằng tay; dùng `int4_pack` / `int4_unpack`.
- **PyTorch:** không có hàm tương đương.

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

- **Chữ ký:** `int4_unpack(packed, numel) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:152`; C++ `include/litetorch/ops.h:206`
- **Tham số:** `packed` — INT8, đầu ra của `int4_pack`. `numel` — `int`, số giá trị cần giải nén (phải bằng `numel` gốc).
- **Trả về:** INT8, `[numel]`, các giá trị gốc.
- **Trường hợp biên:** `numel` phải khớp buffer đóng gói — giải nén với `numel` sai sẽ trả về số lượng sai (không raise lỗi với `numel` quá nhỏ).
- **PyTorch:** không có hàm tương đương.

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

- **Chữ ký:** `int4_weight_only_linear(x, packed_w, w_scale, bias=None) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:153`; C++ `include/litetorch/ops.h:207`
- **Tham số:** `x` — 2D `[M, K]`, FP32. `packed_w` — INT8 2D `[N, (K+1)/2]`, là `int4_pack` của trọng số INT8 `[N, K]` đã reshape thành 2D. `w_scale` — `[N]`, một scale cho mỗi kênh đầu ra (`N = w_scale.shape[0]`). `bias` — `[N]` hoặc `None`.
- **Trả về:** `[M, N]`, `x @ dequant(packed_w).T (+ bias)`.
- **Trường hợp biên:** `x` phải là 2D; `packed_w` phải đúng 2D `[N, (K+1)/2]` nếu không lệnh gọi raise `packed_w shape mismatch` (tensor đóng gói 1D phải reshape trước).
- **PyTorch:** không có hàm tương đương trực tiếp; `(x @ (unpack(packed_w) * w_scale).T)` cho cùng giá trị.

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

- **Chữ ký:** `moe_gate(input, gate_weight, top_k) -> (Tensor, Tensor)`
- **Nguồn:** `src/bindings/bind_ops.cpp:154`; C++ `include/litetorch/ops.h:208-209`
- **Tham số:** `input` — `[T, H]` (token theo chiều ẩn). `gate_weight` — `[num_experts, H]`. `top_k` — `int`.
- **Trả về:** một tuple `(probs, indices)`: `probs` `[T, top_k]` FP32 — xác suất softmax của các expert được chọn, **được chuẩn hóa lại để tổng bằng 1 trên top-k đã chọn** (`probs[k] = softmax(logits)[k] / sum(top-k)`); `indices` `[T, top_k]` INT64 — chỉ số expert. Logit là `input @ gate_weight.T` (layout trọng số gate là `[num_experts, H]`).
- **Trường hợp biên:** `gate_weight` phải là `[num_experts, H]` — truyền `[H, num_experts]` sẽ raise lỗi shape ma trận. `top_k` phải thỏa `1 <= top_k <= num_experts`, nếu không lệnh gọi throw.
- **PyTorch:** không có hàm tương đương trực tiếp; `probs, indices = torch.topk(torch.softmax(input @ gate_weight.T, dim=-1), top_k, dim=-1)` rồi `probs = probs / probs.sum(dim=-1, keepdim=True)` cho cùng giá trị.

<!-- ex: id=r5-moe-gate device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], cpu, False)
gw = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], [3, 4], cpu, False)
probs, indices = lt.Ops.moe_gate(x, gw, 2)
print(list(probs.shape), list(indices.shape), indices.dtype, probs.to_vector(), indices.to_vector())
```
