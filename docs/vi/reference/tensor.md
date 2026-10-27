# Tensor — Tài Liệu Tham Khảo API

`litetorch.Tensor` là lớp tensor trung tâm: một mảng n chiều được hỗ trợ bởi `StorageImpl`, với khả năng vi phân tự động (autograd). Mọi phép tính đều là C++14; Python chỉ là lớp giao diện pybind11.

- **Nguồn chân lý:** `src/bindings/bind_tpu.cpp:31-236` (binding Python), `include/litetorch/tensor.h` (lớp C++).
- **Cây mã nguồn:** agent-v1@02cc271a / HEAD cục bộ 88a4ba1 (mốc tài liệu; B5 kiểm tra lại ở head).
- **Quy ước:** `cpu = lt.Device("cpu")` trong mọi ví dụ. Mọi ví dụ đều tự chứa và chạy được trên CPU. Tương đương PyTorch là FP32 1e-4 trừ khi header ghi khác.
- **Khác biệt liên quan:** D-01 (`data_ptr()`), D-02 (`to(device)` lặng lẽ rơi về CPU), D-03 (không có dạng chuỗi cho `.to()`), D-05 (không có phương thức `.pow()`), D-06 (`.long()` — ĐÃ SỬA, nay là INT64; xem `plan/differences.md`), D-11 (chỉ mục), D-12 (dim đơn kiểu int), D-13 (thiếu phương thức ép kiểu), D-14 (giới hạn dtype của `.numpy()`), D-15 (`arange` chỉ số nguyên), D-18 (mất chính xác của `.item()`/`.tolist()`). Xem `plan/differences.md`.

## Factory tĩnh

### `Tensor.create`

- **Chữ ký:** `Tensor.create(shape, device=Device(CPU, 0), requires_grad=False, dtype=DataType.FP32) -> Tensor`
- **Nguồn:** `src/bindings/bind_tpu.cpp:32`
- **Tham số:** `shape` — danh sách/tuple các số nguyên, ví dụ `[2, 3]`. `device` — một `litetorch.Device`. `requires_grad` — theo dõi để tính gradient. `dtype` — một `litetorch.DataType`, mặc định FP32.
- **Trả về:** tensor chưa khởi tạo với shape đã cho (nội dung không xác định; đừng đọc trước khi ghi).
- **Trường hợp biên:** shape chứa 0 tạo tensor rỗng với `numel() == 0`.
- **PyTorch:** `torch.empty(shape, device=..., dtype=...)` — tương đương, trừ việc LiteTorch chỉ nhận một shape dạng dãy, không bao giờ nhận varargs.

<!-- ex: id=r1-create device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.create([2, 3], cpu)
print(t.shape, t.dtype)
```

### `Tensor.from_vector`

- **Chữ ký:** `Tensor.from_vector(data, shape, device=Device(CPU, 0), requires_grad=False, dtype=DataType.FP32) -> Tensor`
- **Nguồn:** `src/bindings/bind_tpu.cpp:33`
- **Tham số:** `data` — danh sách phẳng các số, độ dài phải bằng `prod(shape)`. `shape` — danh sách/tuple các số nguyên. `dtype` — mặc định FP32; giá trị đi qua `float` trước (xem D-18: giá trị int64 trên 2^24 mất chính xác qua bước này).
- **Trả về:** tensor chứa `data` theo thứ tự hàng chính (row-major).
- **Trường hợp biên:** sai độ dài sẽ báo lỗi. Không nhận danh sách lồng nhau ở đây — dùng `lt.tensor()` cho dãy lồng nhau.
- **PyTorch:** `torch.tensor(data, dtype=...).reshape(shape)` — tương đương với dữ liệu FP32.

<!-- ex: id=r1-from-vector device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
print(t.to_vector())
```

### `Tensor.zeros`

- **Chữ ký:** `Tensor.zeros(shape, device=Device(CPU, 0), requires_grad=False) -> Tensor`; dạng nạp chồng `Tensor.zeros(shape, device, requires_grad=False, dtype=DataType.FP32) -> Tensor`
- **Nguồn:** `src/bindings/bind_tpu.cpp:34-35`
- **Tham số:** `shape` — danh sách/tuple các số nguyên. Dạng đầu luôn trả về FP32; dạng thứ hai nhận `dtype` tường minh (lưu ý: `device` không có giá trị mặc định trong dạng thứ hai — phải truyền tường minh).
- **Trả về:** tensor toàn số 0.
- **PyTorch:** `torch.zeros(shape, dtype=...)` — tương đương.

<!-- ex: id=r1-zeros device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.zeros([2, 2], cpu)
b = lt.Tensor.zeros([2, 2], cpu, False, lt.DataType.INT32)
print(a.to_vector(), b.dtype)
```

### `Tensor.ones`

- **Chữ ký:** `Tensor.ones(shape, device=Device(CPU, 0), requires_grad=False, dtype=DataType.FP32) -> Tensor`
- **Nguồn:** `src/bindings/bind_tpu.cpp:36`
- **Trả về:** tensor toàn số 1.
- **PyTorch:** `torch.ones(shape, dtype=...)` — tương đương.

<!-- ex: id=r1-ones device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.ones([3], cpu)
print(t.to_vector())
```

### `Tensor.randn`

- **Chữ ký:** `Tensor.randn(shape, device=Device(CPU, 0), requires_grad=False, dtype=DataType.FP32) -> Tensor`
- **Nguồn:** `src/bindings/bind_tpu.cpp:37`
- **Tham số:** `shape` — chỉ một shape dạng dãy; `randn(2, 3)` sẽ báo lỗi (D-04).
- **Trả về:** mẫu phân phối chuẩn tắc. Đặt hạt giống: `lt.manual_seed(seed)`; bộ sinh là toàn cục theo tiến trình.
- **PyTorch:** `torch.randn(shape)` — cùng phân phối; cùng hạt giống không đồng nghĩa cùng luồng số.

<!-- ex: id=r1-randn device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(42)
t = lt.Tensor.randn([2, 2], cpu)
print(t.shape, t.dtype)
```

### `Tensor.arange`

- **Chữ ký:** `Tensor.arange(start, end, step=1, device=Device(CPU, 0), dtype=DataType.FP32) -> Tensor`
- **Nguồn:** `src/bindings/bind_tpu.cpp:38`
- **Tham số:** `start`, `end`, `step` — chỉ số nguyên (D-15); giá trị được tạo dưới dạng FP32 rồi ép sang `dtype`.
- **Trả về:** tensor 1 chiều `[start, start+step, ...]` nhỏ hơn `end`.
- **Trường hợp biên:** `step == 0` không hợp lệ; bước số thực sẽ báo lỗi.
- **PyTorch:** `torch.arange(start, end, step)` — khác: torch chấp nhận start/end/step số thực.

<!-- ex: id=r1-arange device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.arange(0, 5, 2, cpu)
print(t.to_vector())
```

## Thuộc tính

Mọi thuộc tính đều đọc/ghi được (`def_readwrite`, `src/bindings/bind_tpu.cpp:58-64`). Gán trực tiếp `t.shape` không bố trí lại bộ nhớ — dùng `view`/`reshape` để đổi shape.

### `Tensor.shape`

- **Kiểu:** danh sách các số nguyên. **Nguồn:** `bind_tpu.cpp:58`.
- **PyTorch:** `t.shape` / `t.size()` — tương đương (torch trả về `torch.Size`, LiteTorch trả về danh sách thường).

### `Tensor.strides`

- **Kiểu:** danh sách các số nguyên (số phần tử mỗi bước dọc theo từng chiều). **Nguồn:** `bind_tpu.cpp:59`.
- **PyTorch:** `t.stride()` — tương đương.

### `Tensor.dtype`

- **Kiểu:** `litetorch.DataType`. **Nguồn:** `bind_tpu.cpp:60`.
- **PyTorch:** `t.dtype` — tương đương; lưu ý LiteTorch có thêm các kiểu chính xác thấp (FP8_E4M3, FP8_E5M2, NF4, FP4_E2M1, INT4) mà torch không có.

### `Tensor.device`

- **Kiểu:** `litetorch.Device`. **Nguồn:** `bind_tpu.cpp:61`.
- **PyTorch:** `t.device` — tương đương.

### `Tensor.requires_grad`

- **Kiểu:** bool. **Nguồn:** `bind_tpu.cpp:62`.
- **PyTorch:** `t.requires_grad` — tương đương.

### `Tensor.grad`

- **Kiểu:** `Tensor` hoặc `None`; gradient tích lũy sau `.backward()`. **Nguồn:** `bind_tpu.cpp:63`.
- **PyTorch:** `t.grad` — tương đương. Tensor không phải lá có `grad is None` trừ khi đã gọi `retain_grad()`.

### `Tensor.shard`

- **Kiểu:** siêu dữ liệu shard cho tensor phân tán. **Nguồn:** `bind_tpu.cpp:64`.
- **PyTorch:** không có tương đương trực tiếp (xem `torch.distributed.tensor.DTensor`).

<!-- ex: id=r1-attrs device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu, True)
print(t.shape, t.strides, t.dtype, t.device, t.requires_grad, t.grad)
print(t.numel(), t.dim(), t.ndim)
```

## Kiểm tra

### `Tensor.numel`

- **Chữ ký:** `numel() -> int`
- **Nguồn:** `bind_tpu.cpp:65`
- **Trả về:** tổng số phần tử (`prod(shape)`; 0 với tensor rỗng).
- **PyTorch:** `t.numel()` — tương đương.

### `Tensor.dim` / `Tensor.ndim`

- **Chữ ký:** `dim() -> int`; `ndim` là thuộc tính chỉ đọc trả về cùng giá trị.
- **Nguồn:** `bind_tpu.cpp:150-151`
- **PyTorch:** `t.dim()` / `t.ndim` — tương đương.

### `Tensor.item`

- **Chữ ký:** `item() -> float`
- **Nguồn:** `bind_tpu.cpp:66`
- **Trả về:** phần tử duy nhất dưới dạng `float` của Python; tensor không phải FP32 được ép sang FP32 trước (D-18: giá trị int64 trên 2^53 mất chính xác).
- **Trường hợp biên:** báo lỗi với tensor có `numel() != 1`.
- **PyTorch:** `t.item()` — khác: torch trả về `int` của Python với đầy đủ chính xác cho tensor int64.

<!-- ex: id=r1-item device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([3.5], [1], cpu)
print(t.item(), t.numel(), t.dim())
```

### `Tensor.to_vector`

- **Chữ ký:** `to_vector() -> list[float]`
- **Nguồn:** `bind_tpu.cpp:67`
- **Trả về:** danh sách phẳng các số thực theo thứ tự hàng chính; tensor không phải FP32 được ép sang FP32 trước (D-18).
- **PyTorch:** `t.flatten().tolist()` — tương đương với FP32; khác về chính xác int64 (xem D-18).

### `Tensor.tolist`

- **Chữ ký:** `tolist() -> list`
- **Nguồn:** `bind_tpu.cpp:202`
- **Trả về:** danh sách lồng nhau theo đúng shape. Phần tử INT32/INT16/INT8 trả về dưới dạng `int` của Python; các kiểu khác dưới dạng `float` (D-18: giá trị int64 trên 2^53 mất chính xác). Tensor không phải CPU được chuyển sang CPU trước.
- **PyTorch:** `t.tolist()` — tương đương với FP32/INT32; khác về chính xác int64.

<!-- ex: id=r1-tolist device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
print(t.to_vector(), t.tolist())
```

### `Tensor.numpy`

- **Chữ ký:** `numpy() -> numpy.ndarray`
- **Nguồn:** `bind_tpu.cpp:158`
- **Trả về:** mảng numpy chia sẻ (FP32) hoặc sao chép (kiểu khác) dữ liệu.
- **Trường hợp biên:** tensor không phải CPU sẽ báo lỗi — gọi `.cpu()` trước. Các kiểu được hỗ trợ: FP64, FP32, FP16, INT32, INT16, INT8. BF16 được ép sang FP32. UINT8, INT64, BOOL báo lỗi `unsupported dtype for copy conversion` (D-14).
- **PyTorch:** `t.numpy()` — khác: torch hỗ trợ trực tiếp uint8 và int64.

<!-- ex: id=r1-numpy device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
a = t.numpy()
print(a.dtype, a.tolist())
```

### `Tensor.__repr__`

- **Chữ ký:** `repr(t) -> str`
- **Nguồn:** `bind_tpu.cpp:236`
- **Trả về:** `"<litetorch.Tensor shape=<numel> device=<device>>"` — lưu ý nó in số phần tử, không phải tuple shape.
- **PyTorch:** `repr(t)` in các giá trị; LiteTorch chỉ in tóm tắt.

## Chỉ mục và lựa chọn

### `Tensor.__getitem__`

- **Chữ ký:** `t[int] -> Tensor`; `t[slice] -> Tensor`
- **Nguồn:** `bind_tpu.cpp:42`
- **Hành vi:** số nguyên chỉ mục chiều 0 và loại bỏ chiều đó; slice chọn một khoảng dọc chiều 0.
- **Trường hợp biên:** mặt nạ boolean, chỉ mục dạng tensor số nguyên và tuple nhiều chiều báo lỗi `TypeError("unsupported index type")` (D-11). Dùng `Ops.gather`, `index_select` hoặc `slice` thay thế.
- **PyTorch:** `t[i]`, `t[a:b]`, `t[mask]`, `t[:, 1]` — LiteTorch chỉ hỗ trợ hai dạng đầu.

<!-- ex: id=r1-getitem device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([10.0, 20.0, 30.0], [3], cpu)
print(t[1].to_vector(), t[0:2].to_vector())
```

### `Tensor.slice`

- **Chữ ký:** `slice(dim, start, end, step=1) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:41`; C++ `include/litetorch/tensor.h:139`
- **Trả về:** lát cắt dạng view dọc theo `dim` (chia sẻ bộ nhớ).
- **PyTorch:** `t.narrow(dim, start, end - start)` / slicing — tương đương.

### `Tensor.index_select`

- **Chữ ký:** `index_select(dim, indices) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:40`; C++ `tensor.h:138`
- **Tham số:** `indices` — tensor số nguyên chứa các vị trí dọc theo `dim`.
- **Trả về:** tensor đã gather; shape là `shape` với `shape[dim]` thay bằng `len(indices)`.
- **PyTorch:** `torch.index_select(t, dim, indices)` — tương đương.

<!-- ex: id=r1-index-select device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([10.0, 20.0, 30.0, 40.0], [2, 2], cpu)
idx = lt.Tensor.from_vector([1.0, 0.0], [2], cpu, False, lt.DataType.INT32)
print(t.index_select(0, idx).to_vector())
print(t.slice(0, 0, 1).to_vector())
```

### `Tensor.masked_fill`

- **Chữ ký:** `masked_fill(mask, value) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:39`; C++ `tensor.h:137`
- **Tham số:** `mask` — tensor boolean có thể broadcast tới `self`; `value` — một số thực.
- **Trả về:** tensor mới với `value` ở mọi nơi `mask` đúng.
- **PyTorch:** `t.masked_fill(mask, value)` — tương đương.

<!-- ex: id=r1-masked-fill device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
m = lt.Tensor.from_vector([1.0, 0.0, 0.0, 1.0], [2, 2], cpu, False, lt.DataType.BOOL)
print(t.masked_fill(m, -1.0).to_vector())
```

### `Tensor.gather`

- **Chữ ký:** `gather(dim, index) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:93`; C++ `tensor.h:169`
- **Tham số:** `index` — tensor số nguyên; shape đầu ra bằng shape của `index`.
- **PyTorch:** `torch.gather(t, dim, index)` — tương đương.

<!-- ex: id=r1-gather device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([10.0, 20.0, 30.0, 40.0], [2, 2], cpu)
idx = lt.Tensor.from_vector([1.0, 0.0], [1, 2], cpu, False, lt.DataType.INT32)
print(t.gather(1, idx).to_vector())
```

## Thao tác shape

### `Tensor.view`

- **Chữ ký:** `view(shape) -> Tensor` hoặc `view(*dims) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:76`
- **Tham số:** một danh sách/tuple duy nhất, hoặc các số nguyên rời rạc. `-1` suy ra một chiều.
- **Trả về:** view chia sẻ bộ nhớ; báo lỗi nếu tensor không liên tục hoặc sai kích thước.
- **PyTorch:** `t.view(...)` — tương đương.

### `Tensor.reshape`

- **Chữ ký:** `reshape(shape) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:88`
- **Trả về:** view khi có thể, ngược lại sao chép.
- **PyTorch:** `t.reshape(shape)` — tương đương.

<!-- ex: id=r1-view-reshape device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], cpu)
print(t.view([3, 2]).shape, t.view(6).shape, t.reshape([3, 2]).to_vector())
```

### `Tensor.as_strided`

- **Chữ ký:** `as_strided(shape, strides) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:89`; C++ `tensor.h:163`
- **Trả về:** view với shape và stride phần tử đã cho (không kiểm tra biên — stride vượt giới hạn sẽ đọc ngoài vùng nhớ).
- **PyTorch:** `t.as_strided(shape, strides)` — tương đương.

### `Tensor.permute`

- **Chữ ký:** `permute(dims) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:90`; C++ `tensor.h:166`
- **Tham số:** `dims` — một hoán vị của `range(ndim)`.
- **Trả về:** view với các chiều đã sắp xếp lại.
- **PyTorch:** `t.permute(dims)` — tương đương.

### `Tensor.transpose`

- **Chữ ký:** `transpose(dim0, dim1) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:135`; C++ `tensor.h:165`
- **Trả về:** view hoán đổi hai chiều.
- **PyTorch:** `t.transpose(dim0, dim1)` — tương đương.

<!-- ex: id=r1-permute-transpose device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], cpu)
print(t.permute([1, 0]).shape, t.transpose(0, 1).to_vector())
print(t.as_strided([3, 2], [1, 3]).to_vector())
```

### `Tensor.expand`

- **Chữ ký:** `expand(shape) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:92`; C++ `tensor.h:168`
- **Trả về:** view broadcast các chiều kích thước 1 tới `shape`; các chiều mở rộng có stride 0.
- **PyTorch:** `t.expand(shape)` — tương đương.

### `Tensor.detach`

- **Chữ ký:** `detach() -> Tensor`
- **Nguồn:** `bind_tpu.cpp:91`
- **Trả về:** tensor chia sẻ bộ nhớ nhưng tách khỏi đồ thị autograd.
- **PyTorch:** `t.detach()` — tương đương.

### `Tensor.clone`

- **Chữ ký:** `clone() -> Tensor`
- **Nguồn:** `bind_tpu.cpp:138`; C++ `tensor.h:159`
- **Trả về:** bản sao sâu (bộ nhớ mới); lịch sử autograd được giữ nguyên.
- **PyTorch:** `t.clone()` — tương đương.

### `Tensor.contiguous` / `Tensor.is_contiguous`

- **Chữ ký:** `contiguous() -> Tensor`; `is_contiguous() -> bool`
- **Nguồn:** `bind_tpu.cpp:136-137`
- **Hành vi:** `contiguous()` trả về `self` nếu đã liên tục, ngược lại trả về bản sao liên tục.
- **PyTorch:** `t.contiguous()` / `t.is_contiguous()` — tương đương.

<!-- ex: id=r1-expand-clone device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0], [2, 1], cpu)
e = t.expand([2, 3])
print(e.shape, e.to_vector(), e.is_contiguous())
print(t.clone().to_vector(), t.detach().shape)
```

## Di chuyển thiết bị và kiểu dữ liệu

### `Tensor.to`

- **Chữ ký:** `to(device) -> Tensor`; `to(dtype) -> Tensor`; `to(device, dtype) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:73-75`
- **Hành vi:** di chuyển và/hoặc ép kiểu. Không có dạng chuỗi: `t.to("cuda")` báo lỗi pybind TypeError — phải truyền `litetorch.Device` (D-03). Nếu backend tăng tốc đích không tồn tại, tensor lặng lẽ nằm trên CPU mà không báo lỗi (D-02) — luôn kiểm tra `t.device` sau `.to()`.
- **PyTorch:** `t.to("cuda")`, `t.to(torch.float32)` — khác: torch chấp nhận chuỗi và báo lỗi khi CUDA không khả dụng.

### `Tensor.to_device_async`

- **Chữ ký:** `to_device_async(target_device) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:139`; C++ `tensor.h:147`
- **Trả về:** tensor trên `target_device`, truyền bất đồng bộ (người gọi phải đồng bộ trước khi đọc trên thiết bị đích).
- **PyTorch:** `t.to(device, non_blocking=True)` — ngữ nghĩa tương tự.

### `Tensor.cuda` / `Tensor.cpu`

- **Chữ ký:** `cuda(device=None) -> Tensor`; `cpu() -> Tensor`
- **Nguồn:** `bind_tpu.cpp:142-149`
- **Hành vi:** `.cuda()` chuyển tới GPU theo chỉ số (`-1` nghĩa là thiết bị hiện tại); trên máy không có GPU nó lặng lẽ trả về tensor CPU (D-02). `.cpu()` luôn chuyển sang CPU.
- **PyTorch:** `t.cuda()` báo lỗi khi không có CUDA; `t.cpu()` — tương đương.

<!-- ex: id=r1-to-cpu device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
u = t.to(cpu)
print(u.device, u.cpu().shape)
print(t.to(lt.DataType.FP64).dtype)
```

### `Tensor.cast` / `Tensor.type_as`

- **Chữ ký:** `cast(target_dtype) -> Tensor`; `type_as(other) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:140, 157`
- **Hành vi:** `cast` chuyển sang `DataType` đã cho; `type_as` chuyển sang `other.dtype`.
- **PyTorch:** `t.to(dtype)` / `t.type_as(other)` — tương đương.

### `Tensor.half` / `Tensor.float` / `Tensor.double`

- **Chữ ký:** `half() -> Tensor`; `float() -> Tensor`; `double() -> Tensor`
- **Nguồn:** `bind_tpu.cpp:152-154`
- **Hành vi:** ép kiểu sang FP16 / FP32 / FP64.
- **PyTorch:** `t.half()` / `t.float()` / `t.double()` — tương đương.

### `Tensor.long` / `Tensor.int`

- **Chữ ký:** `long() -> Tensor`; `int() -> Tensor`
- **Nguồn:** `bind_tpu.cpp:155-156`
- **Hành vi:** `.long()` ánh xạ tới `Tensor::to_long` (INT64) kể từ bản sửa D-06 (qa-r62); `.int()` ánh xạ tới `Tensor::to_int` (INT32). Lưu ý tương tác với D-14 (vẫn mở): `.numpy()` từ chối INT64, nên `.long().numpy()` sẽ báo lỗi — dùng `.to_vector()` hoặc ép kiểu trước.
- **PyTorch:** `t.long()` là INT64; `t.int()` là INT32 — cả hai nay đều giống.

<!-- ex: id=r1-cast-long device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=unverified -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
print(t.half().dtype, t.float().dtype, t.double().dtype)
print(t.long().dtype, t.int().dtype, t.cast(lt.DataType.INT64).dtype)
print(t.type_as(t.cast(lt.DataType.INT64)).dtype)
```

### `Tensor.copy_` / `Tensor.add_`

- **Chữ ký:** `copy_(src) -> None`; `add_(other) -> None`
- **Nguồn:** `bind_tpu.cpp:233-234`; C++ `tensor.h:160-161`
- **Hành vi:** sao chép / cộng tại chỗ; `src`/`other` phải broadcast tương thích và cùng thiết bị.
- **PyTorch:** `t.copy_(src)` / `t.add_(other)` — tương đương.

<!-- ex: id=r1-inplace device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.zeros([2], cpu)
t.copy_(lt.Tensor.from_vector([5.0, 6.0], [2], cpu))
t.add_(lt.Tensor.from_vector([1.0, 1.0], [2], cpu))
print(t.to_vector())
```

## Autograd

### `Tensor.backward`

- **Chữ ký:** `backward(gradient=None, create_graph=False) -> None`
- **Nguồn:** `bind_tpu.cpp:68`
- **Hành vi:** chạy vi phân tự động ngược từ tensor này. Tensor không vô hướng yêu cầu `gradient` tường minh cùng shape. `create_graph=True` xây đồ thị cho gradient bậc cao.
- **PyTorch:** `t.backward(gradient=..., create_graph=...)` — tương đương.

### `Tensor.zero_grad`

- **Chữ ký:** `zero_grad() -> None`
- **Nguồn:** `bind_tpu.cpp:71`
- **Hành vi:** đưa `.grad` về 0 tại chỗ (không làm gì khi `grad is None`).
- **PyTorch:** `t.grad.zero_()` / `optimizer.zero_grad()` — tương đương về hiệu quả.

### `Tensor.retain_grad`

- **Chữ ký:** `retain_grad() -> None`
- **Nguồn:** `bind_tpu.cpp:72`
- **Hành vi:** đánh dấu tensor không phải lá để giữ `.grad` sau backward.
- **PyTorch:** `t.retain_grad()` — tương đương.

### `Tensor.register_hook`

- **Chữ ký:** `register_hook(hook) -> None`
- **Nguồn:** `bind_tpu.cpp:235`; C++ `tensor.h:200`
- **Tham số:** `hook` — hàm nhận gradient tensor và trả về gradient tensor (có thể đã sửa).
- **PyTorch:** `t.register_hook(hook)` — tương đương.

<!-- ex: id=r1-autograd device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([2.0, 3.0], [2], cpu, True)
y = (x * x).sum()
y.backward()
print(x.grad.to_vector())
x.zero_grad()
x.retain_grad()
```

## Phương thức rút gọn và so sánh

### `Tensor.sum` / `Tensor.mean`

- **Chữ ký:** `sum(dim=None, keepdim=False) -> Tensor`; `mean(dim=None, keepdim=False) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:105-119`
- **Hành vi:** `dim=None` rút gọn trên mọi phần tử. Một `dim` kiểu int rút gọn dọc trục đó. Danh sách/tuple các dim KHÔNG được chấp nhận — binding ép `dim` sang một `int64_t` duy nhất (D-12); dùng `litetorch.Ops.sum(t, [0, 1])` để rút gọn nhiều chiều.
- **PyTorch:** `t.sum(dim=(0, 1))` chạy được trong torch — khác với dim dạng tuple.

### `Tensor.max`

- **Chữ ký:** `max(dim=None, keepdim=False) -> Tensor | (Tensor, Tensor)`
- **Nguồn:** `bind_tpu.cpp:105-119`
- **Hành vi:** `dim=None` trả về max vô hướng; với `dim` trả về tuple `(values, indices)`.
- **PyTorch:** `t.max()` / `t.max(dim)` trả về `torch.return_types.max` — giá trị tương đương.

### `Tensor.argmax` / `Tensor.argmin`

- **Chữ ký:** `argmax(dim=-1, keepdim=False) -> Tensor`; `argmin(dim=-1, keepdim=False) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:94-95`
- **Trả về:** tensor chỉ mục (INT64) của vị trí max/min dọc theo `dim`.
- **PyTorch:** `t.argmax(dim, keepdim)` — tương đương.

### `Tensor.topk`

- **Chữ ký:** `topk(k, dim=-1, largest=True, sorted=True) -> (Tensor, Tensor)`
- **Nguồn:** `bind_tpu.cpp:96`
- **Trả về:** tuple `(values, indices)`.
- **PyTorch:** `t.topk(k, dim, largest, sorted)` — tương đương.

<!-- ex: id=r1-reduce-methods device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 3.0, 2.0, 4.0], [2, 2], cpu)
print(t.sum().to_vector(), t.sum(0).to_vector(), t.mean(1, True).to_vector())
print(t.max().to_vector(), t.max(1)[0].to_vector(), t.argmax(1).to_vector())
print(t.topk(1, dim=1)[0].to_vector())
```

### `Tensor.eq` / `ne` / `lt` / `gt` / `le` / `ge`

- **Chữ ký:** `eq(other) -> Tensor` (và `ne`, `lt`, `gt`, `le`, `ge`)
- **Nguồn:** `bind_tpu.cpp:97-102`
- **Tham số:** `other` — một tensor (broadcast được).
- **Trả về:** tensor boolean.
- **PyTorch:** `t.eq(other)` v.v. — tương đương.

### `Tensor.log_softmax`

- **Chữ ký:** `log_softmax(dim=-1) -> Tensor`
- **Nguồn:** `bind_tpu.cpp:103`
- **PyTorch:** `torch.log_softmax(t, dim)` — tương đương.

### `Tensor.silu`

- **Chữ ký:** `silu() -> Tensor`
- **Nguồn:** `bind_tpu.cpp:104`
- **Hành vi:** `x * sigmoid(x)`.
- **PyTorch:** `torch.nn.functional.silu(t)` — tương đương.

<!-- ex: id=r1-compare-methods device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu)
print(t.gt(lt.Tensor.from_vector([1.5, 1.5, 1.5], [3], cpu)).to_vector())
print(t.log_softmax().to_vector(), t.silu().to_vector())
```

## Toán tử dunder số học

Các toán tử phần tử ủy thác cho `Ops.*`. Dạng nạp chồng vô hướng biến số Python thành tensor FP32 một phần tử, nên tensor int64 cộng một số nguyên Python sẽ thăng cấp lên FP32 (D-08). Toán hạng khác thiết bị được chuyển lặng lẽ thay vì báo lỗi (D-10); trên GPU, phép toán số nguyên chạy qua FP32 (D-21).

### `Tensor.__add__` / `__radd__`, `__sub__` / `__rsub__`, `__mul__` / `__rmul__`, `__truediv__` / `__rtruediv__`, `__neg__`

- **Chữ ký:** `t + u`, `t + scalar`, `scalar + t` (và `-`, `*`, `/`); `-t`.
- **Nguồn:** `bind_tpu.cpp:120-133`
- **PyTorch:** cùng toán tử — tương đương trừ D-08/D-10/D-21.

### `Tensor.__matmul__`

- **Chữ ký:** `t @ u -> Tensor`
- **Nguồn:** `bind_tpu.cpp:133`
- **PyTorch:** `t @ u` — tương đương.

### `Tensor.__pow__`

- **Chữ ký:** `t ** scalar -> Tensor` (chỉ số mũ vô hướng, kiểu `double`)
- **Nguồn:** `bind_tpu.cpp:134`
- **Hành vi:** KHÔNG có phương thức `.pow()` (D-05) và không có dạng số mũ tensor: `t ** u` báo lỗi. Dùng `litetorch.Ops.pow(t, float_exp)`.
- **PyTorch:** `t ** u` (số mũ tensor) và `t.pow(exp)` đều chạy — khác.

<!-- ex: id=r1-dunders device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
b = lt.Tensor.from_vector([3.0, 4.0], [2], cpu)
print((a + b).to_vector(), (a * 2.0).to_vector(), (-a).to_vector())
print((a ** 2.0).to_vector())
m1 = lt.Tensor.from_vector([1.0, 2.0], [1, 2], cpu)
m2 = lt.Tensor.from_vector([3.0, 4.0], [2, 1], cpu)
print((m1 @ m2).to_vector())
```

## Thành viên chỉ có ở C++ (không bind sang Python)

### `Tensor::data_ptr` / `Tensor::raw_data_ptr`

- **C++:** `float* data_ptr()` (`include/litetorch/tensor.h:185`); `void* raw_data_ptr()` (`tensor.h:186`).
- **Hành vi:** `data_ptr()` báo lỗi `runtime_error` với mọi tensor không phải FP32 (D-01) và trả về con trỏ CPU ngay cả với tensor GPU. Dùng `raw_data_ptr()` để truy cập mức byte với mọi kiểu.
- **PyTorch:** `t.data_ptr()` chạy với mọi kiểu — khác.

### `Ops::reduce_broadcast` / `Ops::from_data`

- **Chỉ C++:** khai báo trong `include/litetorch/ops.h`, không được bind. `from_data` được `lt.tensor()` dùng nội bộ.
