# Hàm Cấp Cao Nhất — Tài Liệu Tham Khảo API

Các hàm trên module `litetorch`: tạo tensor, RNG, tuần tự hóa, chế độ gradient, cache của bộ cấp phát, và truy vấn thiết bị/backend.

- **Nguồn chân lý:** `src/bindings/bind_core_ops.cpp` (tạo tensor, chế độ gradient, cache), `src/bindings/bind_data.cpp:175-187` (`randn`/`ones`/`zeros`/`arange`), `src/bindings/bind_core.cpp` (thiết bị/backend), `src/bindings/bind_cuda.cpp:54-58` (`equal`, `get_gpu_memory_used`).
- **Cây mã nguồn:** agent-v1@02cc271a / HEAD cục bộ 88a4ba1 (mốc tài liệu; B5 kiểm tra lại ở head).
- **Quy ước:** `cpu = lt.Device("cpu")` trong mọi ví dụ. Đối số shape nhận một dãy duy nhất (`[2, 3]`), không bao giờ nhận varargs (D-04). Tương đương PyTorch là FP32 1e-4 trừ khi header ghi khác.

## Tạo tensor

### `litetorch.tensor`

- **Chữ ký:** `tensor(data, device=None, requires_grad=False, dtype=None) -> Tensor`
- **Nguồn:** `src/bindings/bind_core_ops.cpp:20`
- **Tham số:** `data` — dãy Python lồng nhau (danh sách/tuple) các số; lồng nhau không đều báo lỗi `inconsistent nested sequence sizes`. Giá trị đi qua `float` trước (D-18: giá trị int64 trên 2^24 mất chính xác). `dtype` — mặc định FP32; dùng `lt.DataType.INT64` (không phải `lt.int64`, vì tên đó không tồn tại — xem defect-fixes.md D3).
- **Trả về:** tensor với shape suy ra từ mức lồng nhau; số vô hướng thành shape `[1]`.
- **PyTorch:** `torch.tensor(data, dtype=...)` — tương đương với dữ liệu FP32; khác về chính xác int64.

<!-- ex: id=r2-tensor device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
t = lt.tensor([[1.0, 2.0], [3.0, 4.0]])
print(t.shape, t.to_vector())
print(lt.tensor([1, 2, 3], dtype=lt.DataType.INT64).dtype)
```

### `litetorch.empty`

- **Chữ ký:** `empty(shape, device=None, requires_grad=False, dtype=None) -> Tensor`
- **Nguồn:** `src/bindings/bind_core_ops.cpp:40`
- **Tham số:** `shape` — một số nguyên duy nhất hoặc một dãy (số nguyên đơn nghĩa là 1 chiều). `dtype` — mặc định FP32.
- **Trả về:** tensor chưa khởi tạo; nội dung không xác định.
- **PyTorch:** `torch.empty(shape)` — tương đương (torch cũng chấp nhận varargs; LiteTorch không chấp nhận, D-04).

### `litetorch.full`

- **Chữ ký:** `full(shape, fill_value, device=None, requires_grad=False, dtype=None) -> Tensor`
- **Nguồn:** `src/bindings/bind_core_ops.cpp:43`
- **Tham số:** `fill_value` — một số Python, ép sang `float` trước.
- **Trả về:** tensor chứa đầy `fill_value`.
- **PyTorch:** `torch.full(shape, fill_value)` — tương đương.

<!-- ex: id=r2-empty-full device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
print(lt.empty([2, 2], cpu).shape)
print(lt.full([2, 2], 7.0, cpu).to_vector())
```

### `litetorch.linspace` / `litetorch.logspace`

- **Chữ ký:** `linspace(start, end, steps, device=None, dtype=None) -> Tensor`; `logspace(start, end, steps, base=10.0, device=None, dtype=None) -> Tensor`
- **Nguồn:** `src/bindings/bind_core_ops.cpp:46-52`
- **Tham số:** `start`, `end` — số thực (ép sang FP32); `steps` — số nguyên, số điểm bao gồm cả hai đầu.
- **Trả về:** tensor 1 chiều gồm `steps` giá trị cách đều (`linspace`) hoặc `base ** linspace` (`logspace`).
- **PyTorch:** `torch.linspace(start, end, steps)` / `torch.logspace(start, end, steps, base)` — tương đương.

<!-- ex: id=r2-linspace device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
print(lt.linspace(0.0, 1.0, 3, cpu).to_vector())
print(lt.logspace(0.0, 2.0, 3, 10.0, cpu).to_vector())
```

### `litetorch.eye`

- **Chữ ký:** `eye(n, m=None, device=None, dtype=None) -> Tensor`
- **Nguồn:** `src/bindings/bind_core_ops.cpp:52`
- **Tham số:** `n` — số hàng; `m` — số cột, mặc định bằng `n`.
- **Trả về:** ma trận đơn vị (1 trên đường chéo, 0 ở nơi khác).
- **PyTorch:** `torch.eye(n, m)` — tương đương.

<!-- ex: id=r2-eye device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
print(lt.eye(2, 3, cpu).to_vector())
```

### `litetorch.rand`

- **Chữ ký:** `rand(shape, device=None, requires_grad=False, dtype=None) -> Tensor`
- **Nguồn:** `src/bindings/bind_core_ops.cpp:56`
- **Tham số:** `shape` — một số nguyên duy nhất hoặc một dãy.
- **Trả về:** mẫu đều trong [0, 1). Đặt hạt giống: `lt.manual_seed(seed)`.
- **PyTorch:** `torch.rand(shape)` — cùng phân phối; cùng hạt giống không đồng nghĩa cùng luồng số.

### `litetorch.randint`

- **Chữ ký:** `randint(low, high, shape, device=None, dtype=None) -> Tensor`; `randint(high, shape, device=None, dtype=None) -> Tensor`
- **Nguồn:** `src/bindings/bind_core_ops.cpp:59-64`
- **Tham số:** số nguyên trong `[low, high)` (dạng thứ hai: `[0, high)`). Kiểu mặc định INT32.
- **PyTorch:** `torch.randint(low, high, shape)` — tương đương.

### `litetorch.randperm`

- **Chữ ký:** `randperm(n, device=None, dtype=None) -> Tensor`
- **Nguồn:** `src/bindings/bind_core_ops.cpp:65`
- **Trả về:** một hoán vị ngẫu nhiên của `0..n-1`. Kiểu mặc định INT32.
- **PyTorch:** `torch.randperm(n)` — tương đương.

<!-- ex: id=r2-rand device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(0)
print(lt.rand([2], cpu).shape)
print(lt.randint(0, 10, [3], cpu).dtype, lt.randperm(4, cpu).shape)
```

### `litetorch.randn` / `litetorch.ones` / `litetorch.zeros` / `litetorch.arange`

- **Chữ ký:** `randn(shape, device=Device(CPU, 0), requires_grad=False, dtype=DataType.FP32)`; `ones(shape, ...)`; `zeros(shape, ...)`; `arange(start, end, step=1, device=..., dtype=...)`
- **Nguồn:** `src/bindings/bind_data.cpp:175-187`
- **Hành vi:** bí danh cấp cao nhất của các factory tĩnh `Tensor` (cùng chữ ký, cùng ngữ nghĩa, cùng khác biệt: D-04 shape dạng dãy duy nhất, D-15 `arange` chỉ số nguyên).
- **PyTorch:** `torch.randn(shape)` v.v. — tương đương trừ D-04/D-15.

<!-- ex: id=r2-factories device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(1)
print(lt.randn([2], cpu).shape, lt.ones([2], cpu).to_vector())
print(lt.zeros([2], cpu).to_vector(), lt.arange(0, 4, 2, cpu).to_vector())
```

### `litetorch.zeros_like` / `ones_like` / `empty_like` / `full_like` / `rand_like` / `randn_like`

- **Chữ ký:** `zeros_like(a, device=None, dtype=None)`; `ones_like(a, ...)`; `empty_like(a, ...)`; `full_like(a, fill_value, device=None, dtype=None)`; `rand_like(a, ...)`; `randn_like(a, ...)`
- **Nguồn:** `src/bindings/bind_core_ops.cpp:68-90`
- **Tham số:** `a` — tensor tham chiếu. `device`/`dtype` mặc định theo thiết bị/kiểu của `a` khi bỏ qua.
- **Trả về:** tensor cùng shape với `a`, chứa giá trị tương ứng.
- **PyTorch:** `torch.zeros_like(a)` v.v. — tương đương.

<!-- ex: id=r2-like device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu)
print(lt.zeros_like(a).to_vector(), lt.ones_like(a).to_vector())
print(lt.full_like(a, 9.0).to_vector(), lt.empty_like(a).shape)
```

## RNG

### `litetorch.manual_seed`

- **Chữ ký:** `manual_seed(seed) -> None`
- **Nguồn:** `src/bindings/bind_data.cpp:172`
- **Hành vi:** đặt hạt giống cho RNG toàn cục theo tiến trình mà `rand`/`randn`/`randint`/`randperm` và họ `rand*_like` dùng. KHÔNG đặt hạt giống cho shuffle của `DataLoader` (phát hiện ở B1: shuffle dùng RNG riêng; dùng `shuffle=False` cho ví dụ tái lập được).
- **PyTorch:** `torch.manual_seed(seed)` — tương đương về phạm vi, khác luồng số.

### `litetorch.get_rng_state` / `litetorch.set_rng_state`

- **Chữ ký:** `get_rng_state() -> state`; `set_rng_state(state) -> None`
- **Nguồn:** `src/bindings/bind_data.cpp:173-174`
- **Hành vi:** chụp/khôi phục trạng thái RNG toàn cục để tái lập chính xác.
- **PyTorch:** `torch.get_rng_state()` / `torch.set_rng_state(state)` — tương đương.

<!-- ex: id=r2-rng device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(7)
a = lt.randn([2], cpu).to_vector()
st = lt.get_rng_state()
b = lt.randn([2], cpu).to_vector()
lt.set_rng_state(st)
c = lt.randn([2], cpu).to_vector()
print(a, b == c)
```

## Tuần tự hóa

Tệp được ghi ở định dạng `.lt` (xem `11.data_and_io.md` / R11). `save`/`load` điều phối theo kiểu đối tượng; các hàm `save_*`/`load_*` nhận danh sách tham số tường minh.

### `litetorch.save` / `litetorch.load`

- **Chữ ký:** `save(obj, filepath) -> None`; `load(obj, filepath) -> None`
- **Nguồn:** `src/bindings/bind_data.cpp:150-170`
- **Tham số:** `obj` — một `nn.Module`, danh sách/tuple các tensor tham số, hoặc một `Optimizer`. Kiểu khác sẽ báo lỗi.
- **Hành vi:** `save` ghi tham số (module hoặc danh sách) hoặc trạng thái optimizer; `load` đọc lại vào cùng đối tượng tại chỗ.
- **PyTorch:** `torch.save(obj, path)` / `torch.load(path)` — khác: torch trả về đối tượng đã tải; LiteTorch tải vào đối tượng đã cho.

### `litetorch.save_parameters` / `litetorch.load_parameters`

- **Chữ ký:** `save_parameters(params, filepath)`; `load_parameters(params, filepath)`
- **Nguồn:** `src/bindings/bind_data.cpp:118-119`
- **Tham số:** `params` — danh sách các tensor.
- **Trường hợp biên:** sai shape khi tải sẽ báo lỗi; xem R11 về định dạng `.lt`.

### `litetorch.save_safetensors` / `litetorch.load_safetensors`

- **Chữ ký:** `save_safetensors(tensors, filepath)`; `load_safetensors(filepath) -> dict`
- **Nguồn:** `src/bindings/bind_data.cpp:120-129`
- **Tham số:** `tensors` — dict `str -> Tensor`. `load_safetensors` trả về dict `str -> Tensor`.
- **PyTorch:** `safetensors.torch.save_file` / `load_file` — cùng định dạng.

### `litetorch.save_optimizer_state` / `litetorch.load_optimizer_state`

- **Chữ ký:** `save_optimizer_state(optimizer, filepath)`; `load_optimizer_state(optimizer, filepath)`
- **Nguồn:** `src/bindings/bind_data.cpp:130-131`
- **Tham số:** `optimizer` — một thực thể `optim.Optimizer`.

<!-- ex: id=r2-serialization device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import os
cpu = lt.Device("cpu")
m = lt.nn.Linear(2, 2, True)
path = "/tmp/b2-r2-params.lt"
lt.save(m, path)
n = lt.nn.Linear(2, 2, True)
lt.load(n, path)
print([p.to_vector() for p in m.parameters()] == [p.to_vector() for p in n.parameters()])
os.remove(path)
```

## Chế độ gradient

### `litetorch.no_grad` / `litetorch.enable_grad` / `litetorch.inference_mode`

- **Chữ ký:** context manager, dùng dạng `with lt.no_grad(): ...`
- **Nguồn:** `src/bindings/bind_core_ops.cpp:106-124`
- **Hành vi:** `no_grad` tắt theo dõi gradient trong khối; `enable_grad` bật lại bên trong khối `no_grad`; `inference_mode` tắt theo dõi và tắt tăng bộ đếm phiên bản (đường suy luận nhanh nhất).
- **PyTorch:** `torch.no_grad()` / `torch.enable_grad()` / `torch.inference_mode()` — tương đương.

### `litetorch.is_grad_enabled` / `litetorch.set_grad_enabled`

- **Chữ ký:** `is_grad_enabled() -> bool`; `set_grad_enabled(enabled) -> None`
- **Nguồn:** `src/bindings/bind_core_ops.cpp:126-132`
- **PyTorch:** `torch.is_grad_enabled()` / `torch.set_grad_enabled(enabled)` — tương đương.

### `litetorch.is_inference_mode`

- **Chữ ký:** `is_inference_mode() -> bool`
- **Nguồn:** `src/bindings/bind_core_ops.cpp:124`
- **PyTorch:** `torch.is_inference_mode_enabled()` — tương đương.

<!-- ex: id=r2-grad-mode device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
print(lt.is_grad_enabled())
with lt.no_grad():
    x = lt.Tensor.from_vector([2.0], [1], cpu, True)
    print(lt.is_grad_enabled(), x.requires_grad)
print(lt.is_grad_enabled(), lt.is_inference_mode())
```

## Cache của bộ cấp phát

### `litetorch.empty_cache`

- **Chữ ký:** `empty_cache() -> None`
- **Nguồn:** `src/bindings/bind_core_ops.cpp:92`
- **Hành vi:** giải phóng các khối cache CPU và GPU mà bộ cấp phát đệm đang giữ (xem phần giải thích về giải phóng bộ nhớ, E2).
- **PyTorch:** `torch.cuda.empty_cache()` — tương tự; LiteTorch còn dọn cả cache CPU.

### `litetorch.set_max_cpu_cache_size` / `get_cached_cpu_bytes` / `get_cached_gpu_bytes`

- **Chữ ký:** `set_max_cpu_cache_size(bytes) -> None`; `get_cached_cpu_bytes() -> int`; `get_cached_gpu_bytes() -> int`
- **Nguồn:** `src/bindings/bind_core_ops.cpp:97-105`
- **PyTorch:** không có tương đương trực tiếp cho các nút chỉnh cache CPU.

<!-- ex: id=r2-cache device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
lt.empty_cache()
lt.set_max_cpu_cache_size(1 << 30)
print(lt.get_cached_cpu_bytes(), lt.get_cached_gpu_bytes())
```

## Truy vấn thiết bị và backend

### `litetorch.Device`

- **Chữ ký:** `Device(type=DeviceType.CPU, index=0)`; `Device(device_str)`; thuộc tính `type`, `index` (đọc/ghi); `to_string()` / `__repr__`.
- **Nguồn:** `src/bindings/bind_core.cpp:23-43`
- **Hành vi:** dạng chuỗi phân tích `"cuda"`/`"gpu"` (tùy chọn `:index`), `"tpu"` (tùy chọn `:index`), `"meta"` (không phân biệt hoa thường). MỌI chuỗi khác — kể cả lỗi gõ như `"cude"` — lặng lẽ thành CPU mà không báo lỗi (D-03).
- **PyTorch:** `torch.device("cuda:0")` — khác: torch báo lỗi với chuỗi thiết bị không hợp lệ.

### `litetorch.DeviceType` / `litetorch.DataType`

- **Giá trị:** `DeviceType`: CPU, GPU, META, TPU (`bind_core.cpp:14`). `DataType`: FP64, FP32, FP16, BF16, INT16, INT8, INT4, FP8_E4M3, FP8_E5M2, NF4, FP4_E2M1 (bí danh FP4), INT32, INT64, UINT8, BOOL (`bind_tpu.cpp:12-29`, export cấp cao nhất, nên `lt.INT64` cũng chạy).
- **PyTorch:** `torch.dtype` bao phủ một tập con; FP8/NF4/FP4/INT4 không có đối tác trong torch.

<!-- ex: id=r2-device device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
print(lt.Device("cuda:1").to_string(), lt.Device("cude").to_string())
print(lt.DeviceType.CPU, lt.DataType.FP32, lt.INT64)
```

### `litetorch.auto_device`

- **Chữ ký:** `auto_device() -> Device`
- **Nguồn:** `src/bindings/bind_core.cpp:66`
- **Hành vi:** trả về GPU 0 nếu backend native khả dụng, ngược lại GPU 0 OpenCL, rồi TPU 0, rồi CPU 0. Tôn trọng `LITETORCH_NO_NATIVE_GPU`.
- **PyTorch:** không có lệnh gọi đơn tương đương; `torch.cuda.is_available()` rồi chọn thủ công.

### `litetorch.is_gpu_available` / `litetorch.cuda_synchronize` / `litetorch.is_tpu_available` / `litetorch.get_backend_name`

- **Chữ ký:** `is_gpu_available() -> bool`; `cuda_synchronize() -> None`; `is_tpu_available() -> bool`; `get_backend_name() -> str`
- **Nguồn:** `src/bindings/bind_core.cpp:45-64, 80`
- **Hành vi:** `get_backend_name()` trả về `"CUDA"`, `"ROCm"`, `"OpenCL"`, `"TPU (Google PJRT / libtpu)"` hoặc `"CPU"`.
- **PyTorch:** `torch.cuda.is_available()` / `torch.cuda.synchronize()` — tương đương với cặp CUDA.

### `litetorch.build_commit`

- **Chữ ký:** `build_commit() -> str`
- **Nguồn:** `src/bindings/bind_core.cpp:6`
- **Trả về:** commit git mà extension được build từ đó, hoặc `"unknown"`.
- **PyTorch:** `torch.__version__` — cùng mục đích.

### `litetorch.equal`

- **Chữ ký:** `equal(a, b) -> bool`
- **Nguồn:** `src/bindings/bind_cuda.cpp:54`
- **Hành vi:** bằng nhau chính xác từng phần tử (cùng shape, cùng giá trị).
- **PyTorch:** `torch.equal(a, b)` — tương đương.

### `litetorch.get_gpu_memory_used`

- **Chữ ký:** `get_gpu_memory_used() -> int`
- **Nguồn:** `src/bindings/bind_cuda.cpp:56`
- **Trả về:** số byte GPU mà trình quản lý bộ nhớ đang giữ (cùng giá trị với `lt.cuda.memory_allocated()`).
- **PyTorch:** `torch.cuda.memory_allocated()` — tương đương.

### `litetorch.checkpoint`

- **Chữ ký:** `checkpoint(fn, *args) -> Tensor` (trình trợ giúp checkpoint activation cấp cao nhất)
- **Nguồn:** `src/bindings/bind_ops.cpp:181`
- **Hành vi:** chạy `fn` mà không lưu activation trung gian; tính lại chúng trong backward. Xem thêm `Ops.checkpoint(module, input)` trong R5/R13.
- **PyTorch:** `torch.utils.checkpoint.checkpoint(fn, *args)` — tương đương về mục đích.

<!-- ex: id=r2-backend device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
print(lt.auto_device(), lt.get_backend_name(), lt.build_commit())
print(lt.is_gpu_available(), lt.is_tpu_available(), lt.get_gpu_memory_used())
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
print(lt.equal(a, a), lt.equal(a, lt.Tensor.zeros([2], cpu)))
```
