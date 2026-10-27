# Ops — Phép Toán Phần Tử và Toán Học — Tài Liệu Tham Khảo API

Các hàm `litetorch.Ops.*`: phép toán số học phần tử, toán học, lượng giác, activation, so sánh, matmul và ép kiểu. Phương thức Tensor chỉ tồn tại cho vài hàm trong số này (`log_softmax`, `silu`); còn lại chỉ có dạng `Ops.*` (D-17).

- **Nguồn chân lý:** `src/bindings/bind_ops.cpp:5-178` (binding), `include/litetorch/ops.h` (khai báo C++).
- **Cây mã nguồn:** agent-v1@02cc271a / HEAD cục bộ 88a4ba1 (mốc tài liệu; B5 kiểm tra lại ở head).
- **Quy ước:** `cpu = lt.Device("cpu")` trong mọi ví dụ. Tương đương PyTorch là FP32 1e-4 trừ khi header ghi khác.
- **Khác biệt liên quan:** D-08 (vô hướng int thăng cấp lên FP32), D-09 (GELU chỉ dạng tanh), D-10 (chuyển thiết bị lặng lẽ), D-17 (chỉ Ops, không có phương thức Tensor), D-21 (phép toán số nguyên trên GPU qua FP32).

## Phép toán số học

Cả bốn phép toán hai ngôi đều broadcast như torch, thăng cấp kiểu qua `promote_types` (thăng cấp phần tử đúng như torch — xem differences.md, phần sửa D-07), và lặng lẽ chuyển toán hạng lệch thiết bị sang thiết bị của toán hạng còn lại (D-10) thay vì báo lỗi.

### `Ops.add` / `Ops.sub` / `Ops.mul` / `Ops.div`

- **Chữ ký:** `add(a, b) -> Tensor`; `sub(a, b)`; `mul(a, b)`; `div(a, b)`
- **Nguồn:** `src/bindings/bind_ops.cpp:5-8`
- **Tham số:** `a`, `b` — tensor (broadcast được) hoặc shape broadcast được.
- **Trả về:** kết quả phần tử; phép chia số nguyên cho kết quả thực (FP) khi kiểu thăng cấp là số thực.
- **Trường hợp biên:** chia cho 0 cho `inf`/`nan` theo quy tắc IEEE với số thực; phép `div` số nguyên trên GPU chạy qua FP32 (D-21).
- **PyTorch:** `torch.add(a, b)` / `a + b` v.v. — tương đương trừ D-10/D-21.

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

- **Chữ ký:** `pow(a, exponent) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:19`; C++ `include/litetorch/ops.h:35` (`pow(Tensor, float)`)
- **Tham số:** `exponent` — chỉ số mũ vô hướng kiểu float. Dạng hai tensor `pow(a, b)` được khai báo trước nhưng không bao giờ định nghĩa hay bind (D-05).
- **PyTorch:** `torch.pow(a, exp)` / `a ** exp` / `a ** u` — khác: torch chấp nhận số mũ dạng tensor.

<!-- ex: id=r3-pow device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([2.0, 3.0], [2], cpu)
print(lt.Ops.pow(a, 2.0).to_vector(), (a ** 2.0).to_vector())
```

### `Ops.neg` / `Ops.abs`

- **Chữ ký:** `neg(a) -> Tensor`; `abs(a) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:23-24`
- **PyTorch:** `torch.neg(a)` / `-a`; `torch.abs(a)` — tương đương.

### `Ops.sqrt` / `Ops.rsqrt` / `Ops.reciprocal`

- **Chữ ký:** `sqrt(a) -> Tensor`; `rsqrt(a) -> Tensor` (`1/sqrt(a)`); `reciprocal(a) -> Tensor` (`1/a`)
- **Nguồn:** `src/bindings/bind_ops.cpp:20, 96, 98`
- **Trường hợp biên:** đầu vào âm cho `nan` (sqrt); số 0 cho `inf` (rsqrt, reciprocal).
- **PyTorch:** `torch.sqrt` / `torch.rsqrt` / `torch.reciprocal` — tương đương.

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

- **Chữ ký:** `exp(a)`; `log(a)` (logarit tự nhiên); `log2(a)`; `log10(a)`
- **Nguồn:** `src/bindings/bind_ops.cpp:21-22, 94-95`
- **Trường hợp biên:** `log` của 0 cho `-inf`; của số âm cho `nan`.
- **PyTorch:** `torch.exp` / `torch.log` / `torch.log2` / `torch.log10` — tương đương.

<!-- ex: id=r3-exp-log device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 10.0, 100.0], [3], cpu)
print(lt.Ops.exp(lt.Tensor.from_vector([0.0, 1.0], [2], cpu)).to_vector())
print(lt.Ops.log(a).to_vector(), lt.Ops.log2(a).to_vector(), lt.Ops.log10(a).to_vector())
```

### `Ops.sin` / `Ops.cos` / `Ops.tan`

- **Chữ ký:** `sin(a)`; `cos(a)`; `tan(a)` (radian)
- **Nguồn:** `src/bindings/bind_ops.cpp:26-27, 97`
- **PyTorch:** `torch.sin` / `torch.cos` / `torch.tan` — tương đương.

### `Ops.sign` / `Ops.floor` / `Ops.ceil` / `Ops.round` / `Ops.trunc`

- **Chữ ký:** `sign(a)` (-1/0/1); `floor(a)`; `ceil(a)`; `round(a)` (làm tròn nửa chẵn); `trunc(a)` (hướng về 0)
- **Nguồn:** `src/bindings/bind_ops.cpp:99-103`
- **PyTorch:** `torch.sign` / `torch.floor` / `torch.ceil` / `torch.round` / `torch.trunc` — tương đương.

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

- **Chữ ký:** `clamp(a, min=None, max=None) -> Tensor`; `clamp_min(a, min) -> Tensor`; `clamp_max(a, max) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:25, 104-105` (dạng nạp chồng `clamp` thứ hai ở :106: cả hai None đều báo lỗi)
- **Tham số:** chỉ biên vô hướng kiểu float (D-16: không có min/max dạng tensor như torch).
- **Trường hợp biên:** `clamp(a)` không biên nào báo lỗi `clamp: at least one of min or max must be given`.
- **PyTorch:** `torch.clamp(a, min, max)` — khác: torch chấp nhận biên dạng tensor.

<!-- ex: id=r3-clamp device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([-2.0, 0.5, 3.0], [3], cpu)
print(lt.Ops.clamp(a, 0.0, 2.0).to_vector())
print(lt.Ops.clamp_min(a, 0.0).to_vector(), lt.Ops.clamp_max(a, 1.0).to_vector())
```

## Activation

### `Ops.relu`

- **Chữ ký:** `relu(a) -> Tensor` (`max(0, x)`)
- **Nguồn:** `src/bindings/bind_ops.cpp:28`
- **PyTorch:** `torch.relu(a)` — tương đương.

### `Ops.leaky_relu`

- **Chữ ký:** `leaky_relu(a, negative_slope=0.01) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:29`
- **PyTorch:** `torch.nn.functional.leaky_relu(a, negative_slope)` — tương đương.

### `Ops.sigmoid` / `Ops.tanh`

- **Chữ ký:** `sigmoid(a) -> Tensor`; `tanh(a) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:30-31`
- **PyTorch:** `torch.sigmoid(a)` / `torch.tanh(a)` — tương đương.

### `Ops.gelu`

- **Chữ ký:** `gelu(a) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:33`
- **Hành vi:** CHỈ xấp xỉ tanh, cả chiều xuôi và chiều ngược: `0.5x(1+tanh(0.79788456(x+0.044715x^3)))` (D-09). Không có đối số `approximate`. Sai khác tối đa so với dạng erf chính xác mặc định của torch là ~4.4e-4 — kiểm tra tương đương phải dùng dung sai >= 5e-4 hoặc so với `F.gelu(x, approximate='tanh')`.
- **PyTorch:** `torch.nn.functional.gelu(a)` (mặc định dạng erf chính xác) — khác tới ~4.4e-4.

### `Ops.silu`

- **Chữ ký:** `silu(a) -> Tensor` (`x * sigmoid(x)`); cũng có phương thức `Tensor.silu()`.
- **Nguồn:** `src/bindings/bind_ops.cpp:66`
- **PyTorch:** `torch.nn.functional.silu(a)` — tương đương.

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

- **Chữ ký:** `softmax(a, dim=-1) -> Tensor`; `log_softmax(a, dim=-1) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:32, 65`
- **PyTorch:** `torch.softmax(a, dim)` / `torch.log_softmax(a, dim)` — tương đương.

<!-- ex: id=r3-softmax device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu)
print(lt.Ops.softmax(a).to_vector(), lt.Ops.log_softmax(a).to_vector())
```

## So sánh và lựa chọn

### `Ops.eq` / `ne` / `lt` / `gt` / `le` / `ge`

- **Chữ ký:** `eq(a, b) -> Tensor` (và `ne`, `lt`, `gt`, `le`, `ge`)
- **Nguồn:** `src/bindings/bind_ops.cpp:58-63`
- **Trả về:** tensor boolean, broadcast như phép toán số học (chuyển thiết bị lặng lẽ D-10 cũng áp dụng).
- **PyTorch:** `torch.eq(a, b)` v.v. — tương đương.

### `Ops.where`

- **Chữ ký:** `where(cond, a, b) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:64`
- **Tham số:** `cond` — tensor boolean; `a`, `b` — tensor broadcast được với `cond`.
- **Trả về:** `a` ở nơi `cond` đúng, ngược lại `b`.
- **PyTorch:** `torch.where(cond, a, b)` — tương đương.

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

- **Chữ ký:** `matmul(a, b) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:9`; C++ `include/litetorch/ops.h:18`
- **Hành vi:** nhân ma trận 2 chiều; đầu vào batch tuân theo quy tắc broadcast của torch cho các chiều batch.
- **PyTorch:** `torch.matmul(a, b)` — tương đương.

### `Ops.matmul_bias`

- **Chữ ký:** `matmul_bias(a, b, bias) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:10`
- **Hành vi:** `a @ b + bias` hợp nhất trong một lệnh gọi.
- **PyTorch:** `torch.addmm(bias, a, b)` với 2 chiều — tương đương về giá trị.

### `Ops.bmm`

- **Chữ ký:** `bmm(a, b) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:11`
- **Hành vi:** nhân ma trận theo batch trên tensor 3 chiều `(B, N, M) @ (B, M, P) -> (B, N, P)`.
- **PyTorch:** `torch.bmm(a, b)` — tương đương.

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

## Ép kiểu và trình trợ giúp so sánh

### `Ops.cast`

- **Chữ ký:** `cast(a, target_dtype) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:67`
- **PyTorch:** `a.to(dtype)` — tương đương.

### `Ops.allclose` / `Ops.isclose`

- **Chữ ký:** `allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False) -> bool`; `isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False) -> Tensor`
- **Nguồn:** `src/bindings/bind_ops.cpp:127-130`
- **Hành vi:** từng phần tử `|a - b| <= atol + rtol * |b|`; `allclose` rút gọn thành một bool, `isclose` trả về tensor boolean.
- **PyTorch:** `torch.allclose(a, b, rtol, atol, equal_nan)` / `torch.isclose(...)` — tương đương (lưu ý mặc định của torch là rtol=1e-5, atol=1e-8 — giống nhau).

### `Ops.clip_grad_norm_`

- **Chữ ký:** `clip_grad_norm_(params, max_norm, norm_type=2.0) -> float`
- **Nguồn:** `src/bindings/bind_ops.cpp:79`
- **Tham số:** `params` — danh sách các tensor có `.grad`; `max_norm` — ngưỡng cắt.
- **Trả về:** tổng norm trước khi cắt. Gradient được co lại tại chỗ khi norm vượt `max_norm`.
- **PyTorch:** `torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type)` — tương đương.

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
