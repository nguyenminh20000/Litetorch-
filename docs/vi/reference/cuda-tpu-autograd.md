# CUDA, TPU, Autograd, Enum — Tài liệu tham khảo API

Trang này bao phủ quản lý thiết bị (`litetorch.cuda`, `litetorch.tpu`), engine autograd (`litetorch.autograd`), và hai enum cốt lõi (`DataType`, `DeviceType`).

- **Nguồn chân lý:** `src/bindings/bind_cuda.cpp`, `bind_tpu.cpp`, `bind_autograd.cpp`, `bind_core.cpp` (binding Python); `include/litetorch/device.h`, `autograd.h`, `tensor.h`, `cuda.h`, `tpu.h`.
- **Cây mã nguồn:** agent-v1@02cc271a / HEAD cục bộ 89afd67 (baseline tài liệu là extension 88a4ba1; B5 xác minh lại tại head).
- **Quy ước:** `cpu = lt.Device("cpu")` trong mọi ví dụ. Mọi ví dụ đều tự chứa và chạy được trên CPU. Ví dụ GPU/TPU được đánh dấu `REQUIRES-GPU` / `REQUIRES-TPU` và xác minh trên Kaggle trong B5. Ngang bằng torch là FP32 1e-4 trừ khi tiêu đề ghi khác.
- **Sai khác liên quan:** D-02 (`to(device)` tự động rơi về CPU trong im lặng — `lt.cuda.is_available()` là False trên máy này, nên mọi `.to("cuda")` đều rơi về CPU trong im lặng; xem `plan/differences.md`). D-20 (`flash_attention_half` throw trên TPU). D-01 (`Tensor.data_ptr()` throw trên tensor không phải FP32 — chỉ C++; xem `cpp-api.md`).

## CUDA

`litetorch.cuda` phản chiếu `torch.cuda` về quản lý thiết bị. Lưu ý quirk của binding: `is_available` được đăng ký **hai lần** trong `src/bindings/bind_cuda.cpp` — một lần tại dòng 5 (một lambda kiểm tra backend CUDA native) và một lần tại dòng 48 (bind trực tiếp tới `litetorch::cuda::is_available`). Cả hai trả về cùng giá trị; bản trùng lặp là vô hại.

### `cuda.is_available`

- **Chữ ký:** `is_available() -> bool`
- **Nguồn:** `src/bindings/bind_cuda.cpp:5,48`
- **Trả về:** `True` khi backend CUDA (hoặc HIP) native được load; `False` trên máy chỉ có CPU (đã xác minh: `False` trên extension CPU 88a4ba1).
- **PyTorch:** `torch.cuda.is_available()` — cùng vai trò.

### `cuda.device_count`

- **Chữ ký:** `device_count() -> int`
- **Nguồn:** `src/bindings/bind_cuda.cpp:49`
- **Trả về:** số thiết bị CUDA (0 trên máy chỉ có CPU).
- **PyTorch:** `torch.cuda.device_count()` — cùng vai trò.

### `cuda.current_device`

- **Chữ ký:** `current_device() -> int`
- **Nguồn:** `src/bindings/bind_cuda.cpp:50`
- **Trả về:** chỉ số của thiết bị CUDA hiện tại (0 khi không có thiết bị).
- **PyTorch:** `torch.cuda.current_device()` — cùng vai trò.

### `cuda.set_device`

- **Chữ ký:** `set_device(device=0) -> None`
- **Nguồn:** `src/bindings/bind_cuda.cpp:51`
- **Tham số:** `device` — chỉ số thiết bị int, mặc định 0.
- **Trả về:** `None`. Trên máy chỉ có CPU đây là no-op.
- **PyTorch:** `torch.cuda.set_device()` — cùng vai trò.

### `cuda.synchronize`

- **Chữ ký:** `synchronize() -> None`
- **Nguồn:** `src/bindings/bind_cuda.cpp:41`
- **Trả về:** `None`. Chờ mọi công việc CUDA trên thiết bị hiện tại hoàn tất. No-op khi không có CUDA.
- **PyTorch:** `torch.cuda.synchronize()` — cùng vai trò.

### `cuda.memory_allocated` / `cuda.memory_reserved`

- **Chữ ký:** `memory_allocated() -> int`; `memory_reserved() -> int`
- **Nguồn:** `src/bindings/bind_cuda.cpp:38,52`
- **Trả về:** số byte hiện đang được cấp phát / giữ bởi caching allocator trên thiết bị hiện tại (0 khi không có CUDA).
- **PyTorch:** `torch.cuda.memory_allocated()` / `memory_reserved()` — cùng vai trò.

### `cuda.set_tf32_enabled` / `cuda.is_tf32_enabled`

- **Chữ ký:** `set_tf32_enabled(enabled) -> None`; `is_tf32_enabled() -> bool`
- **Nguồn:** `src/bindings/bind_cuda.cpp:12,18`
- **Tham số:** `enabled` — bool.
- **Trả về:** `is_tf32_enabled` báo cáo matmul TF32 có được phép không (mặc định False trên build này — xác minh bằng cách in ra).
- **PyTorch:** `torch.backends.cuda.matmul.allow_tf32` — cùng cài đặt, hình thức khác (torch dùng thuộc tính module, LiteTorch dùng hàm).

### `cuda.set_fp16_math_enabled` / `cuda.is_fp16_math_enabled`

- **Chữ ký:** `set_fp16_math_enabled(enabled) -> None`; `is_fp16_math_enabled() -> bool`
- **Nguồn:** `src/bindings/bind_cuda.cpp:25,31`
- **Tham số:** `enabled` — bool.
- **Trả về:** `is_fp16_math_enabled` báo cáo phép toán FP16 có được bật không.
- **PyTorch:** không có tương đương trực tiếp; gần nhất là `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction`.

### `cuda.manual_seed_all`

- **Chữ ký:** `manual_seed_all(seed) -> None`
- **Nguồn:** `src/bindings/bind_cuda.cpp:53`
- **Tham số:** `seed` — int.
- **Trả về:** `None`. Seed RNG trên mọi thiết bị CUDA. No-op khi không có CUDA (RNG CPU được seed bằng `lt.manual_seed`).
- **PyTorch:** `torch.cuda.manual_seed_all(seed)` — cùng vai trò.

### `cuda.get_gpu_memory_used`

- **Chữ ký:** `get_gpu_memory_used() -> int` (bind ở mức module, không nằm dưới `cuda`)
- **Nguồn:** `src/bindings/bind_cuda.cpp:56`
- **Trả về:** số byte bộ nhớ GPU hiện đang dùng (0 khi không có CUDA).
- **PyTorch:** không có tương đương trực tiếp; gần nhất là `torch.cuda.memory_allocated()`.

<!-- ex: id=r14-cuda device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

print("cuda available:", lt.cuda.is_available())
print("device count:", lt.cuda.device_count())
print("current device:", lt.cuda.current_device())
print("memory allocated:", lt.cuda.memory_allocated())
print("memory reserved:", lt.cuda.memory_reserved())
print("tf32 enabled:", lt.cuda.is_tf32_enabled())
lt.cuda.set_tf32_enabled(True)
print("tf32 after set:", lt.cuda.is_tf32_enabled())
lt.cuda.set_tf32_enabled(False)
print("fp16 math enabled:", lt.cuda.is_fp16_math_enabled())
lt.cuda.manual_seed_all(42)
lt.cuda.synchronize()
print("gpu memory used:", lt.get_gpu_memory_used())
```

## TPU

`litetorch.tpu` phản chiếu bề mặt `cuda` cho thiết bị TPU (backend PJRT). Trên máy chỉ có CPU, mọi truy vấn đều trả về đáp án thiết bị trống.

### `tpu.is_available`

- **Chữ ký:** `is_available() -> bool`
- **Nguồn:** `src/bindings/bind_tpu.cpp:5`
- **Trả về:** `True` khi backend TPU (PJRT) được load; `False` trên máy này (đã xác minh).
- **PyTorch:** `torch_xla.runtime` / `torch.tpu` — không có `torch.tpu.is_available`; gần nhất là `torch_xla.core.xla_model.get_xla_supported_devices()`.

### `tpu.device_count`

- **Chữ ký:** `device_count() -> int`
- **Nguồn:** `src/bindings/bind_tpu.cpp:6`
- **Trả về:** số thiết bị TPU (0 trên máy này).
- **PyTorch:** `torch_xla.core.xla_model.xrt_world_size()` — cùng vai trò, hình thức khác.

### `tpu.current_device`

- **Chữ ký:** `current_device() -> int`
- **Nguồn:** `src/bindings/bind_tpu.cpp:7`
- **Trả về:** chỉ số của thiết bị TPU hiện tại (0 khi không có).
- **PyTorch:** `torch_xla.core.xla_model.get_local_ordinal()` — cùng vai trò.

### `tpu.set_device`

- **Chữ ký:** `set_device(device_id=0) -> None`
- **Nguồn:** `src/bindings/bind_tpu.cpp:8`
- **Tham số:** `device_id` — int, mặc định 0.
- **Trả về:** `None`. No-op khi không có TPU.
- **PyTorch:** không có tương đương trực tiếp.

### `tpu.synchronize`

- **Chữ ký:** `synchronize() -> None`
- **Nguồn:** `src/bindings/bind_tpu.cpp:9`
- **Trả về:** `None`. Chặn cho đến khi mọi công việc TPU hoàn tất. No-op khi không có TPU.
- **PyTorch:** `torch_xla.core.xla_model.mark_step()` — cùng vai trò (flush đồ thị lazy).

### `tpu.get_device_name`

- **Chữ ký:** `get_device_name(device_id) -> str`
- **Nguồn:** `src/bindings/bind_tpu.cpp:10`
- **Tham số:** `device_id` — int.
- **Trả về:** chuỗi tên thiết bị, hoặc `"N/A"` khi thiết bị không tồn tại (đã xác minh: `get_device_name(0)` trả về `"N/A"` trên máy này).
- **PyTorch:** `torch.cuda.get_device_name` — cùng hình thức, cho CUDA.

<!-- ex: id=r14-tpu device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

print("tpu available:", lt.tpu.is_available())
print("device count:", lt.tpu.device_count())
print("current device:", lt.tpu.current_device())
print("device 0 name:", lt.tpu.get_device_name(0))
lt.tpu.synchronize()
```

## Autograd

### `autograd.grad`

- **Chữ ký:** `grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=False, allow_unused=False) -> tuple[Tensor | None, ...]`
- **Nguồn:** `src/bindings/bind_autograd.cpp:52-67`
- **Tham số:** `outputs` — tensor hoặc danh sách tensor cần lấy đạo hàm. `inputs` — tensor hoặc danh sách tensor cần lấy đạo hàm theo. `grad_outputs` — gradient thượng nguồn (mặc định `None` = toàn 1). `retain_graph` — giữ đồ thị cho lần backward thứ hai. `create_graph` — xây đồ thị của chính gradient (cho đạo hàm bậc cao). `allow_unused` — trả về `None` cho đầu vào không ảnh hưởng đầu ra thay vì throw.
- **Trả về:** một tuple gradient, một cho mỗi đầu vào, theo thứ tự đầu vào. Đầu vào không dùng cho `None` khi `allow_unused=True`.
- **Trường hợp biên:** không có `retain_graph`, gọi `grad` hai lần trên cùng đồ thị sẽ throw. `create_graph=True` là bắt buộc cho `gradgradcheck`.
- **PyTorch:** `torch.autograd.grad` — cùng chữ ký, cùng mặc định, cùng ngữ nghĩa.

### `autograd.gradcheck`

- **Chữ ký:** `gradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True) -> bool`
- **Nguồn:** `src/bindings/bind_autograd.cpp:68-80`
- **Tham số:** `func` — callable nhận đầu vào và trả về một tensor (hoặc danh sách). `inputs` — tensor hoặc danh sách tensor (phải là FP64 hoặc FP32; nên dùng độ chính xác kép). `eps` — bước sai phân hữu hạn, mặc định 1e-6. `atol` / `rtol` — dung sai, mặc định 1e-5 / 1e-3. `raise_exception` — throw khi thất bại thay vì trả về False.
- **Trả về:** `True` khi gradient giải tích khớp gradient số trong dung sai.
- **Trường hợp biên:** đầu vào FP32 cần `eps=1e-3` — `eps=1e-6` mặc định nằm dưới độ phân giải FP32 ở độ lớn O(1), nên gradient số ra sai (đã xác minh: với x^2 tại x=3, gradient số là 5.72 thay vì 6). Torch có cùng ràng buộc, đó là lý do gradcheck của torch yêu cầu đầu vào độ chính xác kép; LiteTorch chấp nhận FP32 nhưng bạn phải tăng `eps`.
- **PyTorch:** `torch.autograd.gradcheck` — cùng chữ ký và mặc định.

### `autograd.gradgradcheck`

- **Chữ ký:** `gradgradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True) -> bool`
- **Nguồn:** `src/bindings/bind_autograd.cpp:81-93`
- **Tham số:** giống `gradcheck`.
- **Trả về:** `True` khi gradient bậc hai khớp về mặt số. Cần hỗ trợ `create_graph` trong backward. Cùng quy tắc FP32 `eps=1e-3` như `gradcheck`.
- **PyTorch:** `torch.autograd.gradgradcheck` — cùng chữ ký và mặc định.

### `autograd.retain_grad`

- **Chữ ký:** `retain_grad(tensor) -> None`
- **Nguồn:** `src/bindings/bind_autograd.cpp:94-95`
- **Tham số:** `tensor` — một tensor không phải leaf.
- **Trả về:** `None`. Đánh dấu tensor để `.grad` được điền sau backward (mặc định chỉ tensor leaf giữ `.grad`).
- **PyTorch:** `Tensor.retain_grad()` — cùng vai trò; LiteTorch expose nó vừa dưới dạng hàm module vừa dưới dạng `Tensor.retain_grad()`.

### `autograd.Function`

- **Chữ ký:** class `Function()` với classmethod `apply(*args) -> Tensor | tuple[Tensor, ...]`; subclass nó và định nghĩa `forward(ctx, *args)` và `backward(ctx, *grad_outputs)`
- **Nguồn:** `src/bindings/bind_autograd.cpp:24-51`
- **Tham số:** `apply` nhận đầu vào của hàm (tensor và đối số không phải tensor). Bên trong `forward`, `ctx.save_for_backward(*tensors)` lưu tensor cho backward; `ctx.saved_tensors` và `ctx.needs_input_grad` đọc được trong `backward`.
- **Trả về:** `forward` trả về một tensor (hoặc tuple); `backward` trả về gradient cho mỗi đầu vào của forward (hoặc `None` cho từng đầu vào).
- **Trường hợp biên:** đối số không phải tensor của `apply` được truyền thẳng tới `forward` nguyên vẹn; đối số tensor tới dưới dạng tensor. `forward` phải là `@staticmethod`. Một subclass không định nghĩa `backward` sẽ raise khi gradient được yêu cầu.
- **PyTorch:** `torch.autograd.Function` — cùng hình thức (`forward` / `backward` / `ctx.save_for_backward` / `apply`).

<!-- ex: id=r14-autograd device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")


class Square(lt.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * x * 2.0


x = lt.Tensor.from_vector([3.0], [1], cpu)
x.requires_grad = True
y = Square.apply(x)
print("forward:", y.to_vector())
g = lt.autograd.grad(y, x)
print("grad:", g[0].to_vector())
print("gradcheck:", lt.autograd.gradcheck(lambda t: Square.apply(t), x, 1e-3))
print("gradgradcheck:", lt.autograd.gradgradcheck(lambda t: Square.apply(t), x, 1e-3))
```

### `autograd.FunctionContext`

- **Chữ ký:** class `FunctionContext()`; phương thức `save_for_backward(*tensors)`; thuộc tính chỉ đọc `saved_tensors`, `needs_input_grad`
- **Nguồn:** `src/bindings/bind_autograd.cpp:6-23`
- **Trả về:** đối tượng `ctx` được truyền cho `Function.forward` / `Function.backward`. `save_for_backward` lưu tensor cho backward; `saved_tensors` trả về chúng dưới dạng tuple; `needs_input_grad` là tuple bool, một cho mỗi đầu vào của forward.
- **PyTorch:** `torch.autograd.function.FunctionCtx` — cùng vai trò.

## Enum

### `DataType`

- **Chữ ký:** enum với 16 tên (15 giá trị riêng biệt — `FP4` là alias của `FP4_E2M1`): `FP64`, `FP32`, `FP16`, `BF16`, `INT16`, `INT8`, `INT4`, `FP8_E4M3`, `FP8_E5M2`, `NF4`, `FP4_E2M1`, `FP4`, `INT32`, `INT64`, `UINT8`, `BOOL`
- **Nguồn:** `src/bindings/bind_tpu.cpp:12-29`
- **Trả về:** kiểu dữ liệu của một tensor (`tensor.dtype`). `FP64` nằm trong enum nhưng hầu hết kernel chỉ chấp nhận FP32/FP16/BF16 và các kiểu nguyên — tạo tensor FP64 được phép, tính toán với nó có thể throw.
- **Trường hợp biên:** `FP4` và `FP4_E2M1` so sánh bằng nhau (`lt.DataType.FP4 == lt.DataType.FP4_E2M1` là True). Không có shorthand `lt.int64` / `lt.float32` — dùng `lt.DataType.INT64` (xem D3 trong `plan/defect-fixes.md`).
- **PyTorch:** `torch.dtype` (`torch.float32`, `torch.int64`, ...) — cùng tập kiểu; torch không có dtype FP4/NF4/FP8 ở cấp `torch.dtype` (các kiểu đó nằm trong `torchao` / `transformers`).

### `DeviceType`

- **Chữ ký:** enum với 4 giá trị: `CPU`, `GPU`, `META`, `TPU`
- **Nguồn:** `src/bindings/bind_core.cpp:14-19`
- **Trả về:** loại thiết bị của một `Device` (`device.type`). `META` là thiết bị giữ chỗ không có bộ nhớ (dùng cho tracing chỉ-shape).
- **PyTorch:** `torch.device.type` (`"cpu"`, `"cuda"`, `"meta"`) — cùng bốn ý tưởng; torch không có enum `DeviceType` đơn lẻ (loại là một chuỗi).

### `Device`

- **Chữ ký:** `Device(type=DeviceType.CPU, index=0)`; `Device("cpu")` / `Device("cuda:0")` / `Device("tpu:0")` (dạng chuỗi, không phân biệt hoa thường)
- **Nguồn:** `src/bindings/bind_core.cpp:20-30`; `include/litetorch/device.h`
- **Tham số:** `type` — một `DeviceType`; `index` — chỉ số thiết bị int. Dạng chuỗi parse `"cpu"`, `"gpu"`, `"cuda"`, `"cuda:N"`, `"tpu"`, `"tpu:N"`, `"meta"`.
- **Trả về:** một đối tượng thiết bị. `Device("cuda")` trên máy chỉ có CPU vẫn khởi tạo được — fallback im lặng (D-02) xảy ra tại lúc `.to(device)`, không phải lúc khởi tạo.
- **PyTorch:** `torch.device` — cùng hai dạng (`torch.device("cuda:0")` / `torch.device("cpu")`).

<!-- ex: id=r14-enums device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

print("dtypes:", [lt.DataType.FP64, lt.DataType.FP32, lt.DataType.FP16,
      lt.DataType.BF16, lt.DataType.INT64, lt.DataType.UINT8, lt.DataType.BOOL])
print("fp4 alias:", lt.DataType.FP4 == lt.DataType.FP4_E2M1)
print("device types:", [lt.DeviceType.CPU, lt.DeviceType.GPU,
      lt.DeviceType.META, lt.DeviceType.TPU])
cpu = lt.Device("cpu")
print("cpu device:", cpu.type, cpu.index)
print("cuda device:", lt.Device("cuda:0").type, lt.Device("cuda:0").index)
x = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
print("tensor dtype:", x.dtype, "device:", x.device.type)
```
