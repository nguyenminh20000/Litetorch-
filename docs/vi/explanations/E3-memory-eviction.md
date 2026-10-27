# E3 — Giải Phóng Bộ Nhớ

Giải phóng bộ nhớ (memory eviction) là điểm khác biệt của LiteTorch: khi bộ nhớ GPU sắp hết, framework chuyển các tensor GPU đang nhàn rỗi về bộ nhớ CPU thay vì báo lỗi, rồi chuyển chúng trở lại khi cần. Trang này mô tả cơ chế đúng như được cài đặt trong cây mã tại HEAD `b326f29` (extension build `88a4ba1`). Mọi khẳng định về hành vi đều trích dẫn tệp nguồn của nó. Không có gì ở đây được suy ra từ PyTorch: PyTorch không có cơ chế tự động đẩy tensor từ GPU về CPU tương đương; hành vi của caching allocator trong `torch.cuda` không phải là cùng một cơ chế.

## 1. Các thành phần

Ba thành phần phối hợp với nhau:

```text
Tensor (StorageImpl)
    │  giữ gpu_data (bộ đệm trên thiết bị) và cpu_data (bản sao trên host)
    ▼
MemoryManager (singleton, danh sách LRU các storage GPU)
    │  quyết định đẩy cái gì ra, theo dõi gpu_used_ so với gpu_limit_
    ▼
StorageImpl::evict_impl() / swap_in_impl()
    │  thao tác di chuyển thực sự: GPU -> CPU (evict) và CPU -> GPU (swap-in)
    ▼
rt_gpu_read / rt_gpu_write / rt_gpu_allocate   (include/litetorch/backend_route.h)
    │  chuyển tới backend GPU native khi có, nếu không thì dùng OpenCL
```

Tệp nguồn: `include/litetorch/memory_manager.h`, `src/memory_manager/memory_manager.cpp`, `src/tensor/storage.cpp`, `include/litetorch/backend_route.h`.

## 2. Giới hạn

`MemoryManager` giữ một ngân sách duy nhất, `gpu_limit_`, mặc định là **512 MiB** (`include/litetorch/memory_manager.h:53`: `gpu_limit_{512 * 1024 * 1024}`). `gpu_used_` đếm số byte của mọi storage GPU đang được đăng ký.

Hai điểm quan trọng, cả hai đều đã kiểm chứng trên cây mã và extension đã build:

- Giới hạn được đặt từ C++ qua `MemoryManager::get().set_gpu_limit(bytes)` (`src/memory_manager/memory_manager.cpp:21-25`). Hạ giới hạn sẽ đẩy ra ngay lập tức: `set_gpu_limit` gọi `evict_to_free_impl(0)`, hàm này giải phóng cho tới khi `gpu_used_` vừa với giới hạn mới.
- **Không có Python binding cho `set_gpu_limit` tại commit này.** Đã kiểm chứng trực tiếp: `hasattr(lt, 'set_gpu_limit')` và `hasattr(lt.cuda, 'set_gpu_limit')` đều là `False` trên build `88a4ba1`. Từ Python, giới hạn mặc định 512 MiB không thể thay đổi; việc giải phóng diễn ra tự động và chỉ quan sát được qua các bộ đếm ở §6.

## 3. Khi nào việc giải phóng kích hoạt

Việc giải phóng là lười biếng (lazy) và driven theo nhu cầu. Nó chạy trong đúng ba tình huống, tất cả đều nằm trong `MemoryManager`:

1. **Một storage GPU mới đăng ký.** `register_gpu_impl` gọi `evict_to_free_impl(size_bytes)` *trước* khi chèn storage mới (`src/memory_manager/memory_manager.cpp:64-70`). Một cấp phát lớn sẽ đẩy đủ các tensor cũ ra để vừa dưới giới hạn trước.
2. **Giới hạn bị hạ xuống.** `set_gpu_limit` đẩy ra tới giới hạn mới ngay lập tức (xem §2).
3. **Một tensor đã bị đẩy ra được nạp trở lại.** `swap_in_impl` đăng ký lại storage (`src/tensor/storage.cpp:256-262`), và việc đăng ký sẽ đẩy ra như ở (1). Đưa một tensor trở lại có thể đẩy một tensor khác ra ngoài.

Việc giải phóng không bao giờ chạy theo hẹn giờ, không bao giờ chạy nền, và không bao giờ chạy cho các tensor CPU. Các cấp phát CPU đi qua `CachingAllocator::allocate_cpu` mà không có bước giải phóng nào (`src/tensor/storage.cpp:55-57`).

## 4. Chính sách giải phóng: LRU, bỏ qua các tensor đang dùng

`evict_to_free_impl` (`src/memory_manager/memory_manager.cpp:96-117`) duyệt danh sách LRU từ đầu **cũ nhất**:

```text
while gpu_used_ + required_bytes > gpu_limit_ and list not empty:
    take oldest storage
    if storage is NOT in use AND NOT already swapped out:
        evict it (GPU -> CPU copy, free GPU buffer)
        evicted += its bytes
    else:
        skip it, try the next one
```

- **Thứ tự LRU.** Mỗi lần dùng sẽ làm mới vị trí của một storage: `touch_impl` chuyển nó về cuối danh sách (`memory_manager.cpp:88-95`), và `ensure_gpu` chạm vào một storage đã thường trú (`memory_manager.cpp:44-52`). Tensor ít được dùng gần đây nhất sẽ bị đẩy ra trước.
- **Các tensor đang dùng không bao giờ bị đẩy ra.** Một storage có cờ `in_use` sẽ bị bỏ qua. Đây là quy tắc an toàn giữ cho kernel không mất đầu vào giữa chừng khi đang chạy.
- **Các tensor đã bị đẩy ra sẽ bị bỏ qua.** Một storage có cờ `is_swapped` không còn giữ bộ đệm GPU, nên không có gì để giải phóng.
- **Việc giải phóng là best-effort, không phải đảm bảo.** Nếu mọi storage đều đang dùng (hoặc đã bị đẩy ra), vòng lặp không giải phóng gì và trả về `0`. Bên gọi sau đó vẫn thử cấp phát. Với một cấp phát GPU mới, thất bại đó là lỗi cứng: constructor của `StorageImpl` ném `GPU out of memory` kèm chuỗi chẩn đoán (`src/tensor/storage.cpp:30-36`). Giải phóng làm giảm số lần OOM; nó không xóa bỏ OOM.

Kết quả của lần giải phóng gần nhất được ghi vào `last_evicted_bytes_` và `last_eviction_required_` (`memory_manager.cpp:114-115`), chuỗi chẩn đoán OOM báo cáo chúng dưới dạng `last_eviction_freed=` / `last_eviction_required=` (`memory_manager.cpp:147-148`).

## 5. Việc giải phóng làm gì với một tensor

`StorageImpl::evict_impl` (`src/tensor/storage.cpp:222-234`):

1. Cấp phát một bộ đệm CPU nếu tensor chưa có.
2. Sao chép bộ đệm trên thiết bị về CPU (`rt_gpu_read`).
3. Giải phóng bộ đệm GPU ngay lập tức (`rt_gpu_free_now`).
4. Đặt `gpu_data = nullptr`, `is_swapped = true`.
5. Hủy đăng ký storage khỏi `MemoryManager`, thao tác này trừ số byte của nó khỏi `gpu_used_`.

Đối tượng tensor vẫn tồn tại và vẫn dùng được. Dữ liệu của nó giờ chỉ nằm trong bộ nhớ CPU; `Tensor` vẫn báo cáo thiết bị gốc của nó. Việc giải phóng là trong suốt đối với code người dùng, ngoại trừ tốc độ: lần truy cập tiếp theo phải trả giá swap-in.

**Swap-in (tính toán lại bằng cách nạp lại, không phải bằng cách chạy lại).** Khi một tensor đã bị đẩy ra lại cần dùng trên thiết bị, `get_gpu_ptr` gọi `swap_in_impl` (`src/tensor/storage.cpp:114-133, 244-271`):

1. Đăng ký lại với `MemoryManager` (việc này có thể đẩy các tensor khác ra trước).
2. Cấp phát một bộ đệm GPU mới; cấp phát thất bại sẽ ném `GPU out of memory: failed to swap in ...` — bản thân swap-in cũng có thể OOM.
3. Sao chép CPU → GPU (`rt_gpu_write`), giải phóng bản sao CPU, xóa `is_swapped`.

Lưu ý sự bất đối xứng: giải phóng giữ lại bản sao CPU, swap-in giải phóng nó. Một tensor bị đẩy ra rồi nạp trở lại sẽ kết thúc ở cùng trạng thái như trước khi bị đẩy, không còn bản sao CPU nào được giữ lại.

Một trường hợp biên đáng biết: nếu swap-in không tìm thấy GPU nào cả (backend đã biến mất), nó không ném lỗi — nó lặng lẽ chuyển storage thành CPU (`storage.cpp:248-253`). Đây là cùng họ fallback lặng lẽ như D-02 trong trang Behavioral Differences.

## 6. Quan sát được từ Python

Trên build `88a4ba1`, bề mặt Python nhìn thấy được chỉ đúng hai bộ đếm:

| Lời gọi | Trả về | Nguồn |
|---|---|---|
| `lt.cuda.memory_allocated()` | `MemoryManager::get().get_gpu_used()` | `src/bindings/bind_cuda.cpp:39` |
| `lt.get_gpu_memory_used()` | cùng giá trị | `src/bindings/bind_cuda.cpp:57` |

Vì việc giải phóng hủy đăng ký storage, `memory_allocated()` **giảm** khi việc giải phóng giải phóng các tensor. Trên máy chỉ có CPU, cả hai đều trả về `0`: không storage GPU nào từng đăng ký, nên việc giải phóng không bao giờ chạy (Ví dụ E3-2). Chuỗi chẩn đoán C++ (`gpu_oom_diagnostics`, `memory_manager.cpp:129-150`) báo cáo `gpu_used`, `gpu_limit`, `allocator_cached`, `lru_storages`, `active_storages`, và các số liệu của lần giải phóng gần nhất, nhưng nó không được bound sang Python tại commit này.

## 7. Tương tác với checkpointing

Giải phóng và activation checkpointing tấn công cùng một vấn đề — các activation không vừa trong bộ nhớ GPU — từ hai hướng ngược nhau, và chúng là hai cơ chế độc lập:

| | Giải phóng | Checkpointing (`litetorch.checkpoint`) |
|---|---|---|
| Tiết kiệm cái gì | Số byte GPU, bằng cách chuyển tensor về CPU | Số byte GPU, bằng cách không lưu các giá trị trung gian |
| Trả giá bằng gì | Sao chép qua bus khi swap-in | Chạy lại forward pass trong backward pass |
| Khi nào tác động | Tại thời điểm cấp phát, tự động | Chỉ ở nơi người dùng bọc một hàm |
| Người dùng có cần làm gì không | Không | Có: bọc đoạn đó trong `checkpoint` |

Checkpointing là tính toán lại (recompute) theo nghĩa chặt chẽ. `litetorch.checkpoint(function, input)` (`src/autograd/checkpoint.cpp:60-77`) chạy `function(input)` dưới `NoGradGuard` trong forward pass, chỉ lưu một bản sao của đầu vào cộng trạng thái RNG, rồi chạy lại hàm bên trong backward pass (`CheckpointNode::backward`, `checkpoint.cpp:26-58`), khôi phục trạng thái RNG trước để các lớp ngẫu nhiên tái tạo đúng các giá trị forward. Nếu grad bị tắt hoặc đầu vào không yêu cầu grad, `checkpoint` chỉ gọi hàm trực tiếp — không node nào được tạo (`checkpoint.cpp:61-63`).

`Ops.checkpoint(module, input)` (`src/ops/checkpoint.cpp`, được bound tại `src/bindings/bind_ops.cpp:161`) là dạng module: nó lưu đầu vào và tham số của module rồi chạy lại `module->forward` trong backward pass.

Hai cơ chế kết hợp mà không cần phối hợp: đầu vào đã lưu của một đoạn checkpointed có thể tự nó bị đẩy về CPU trong lúc chờ backward pass, và các giá trị trung gian được tính toán lại của nó là các tensor thông thường chịu cùng chính sách LRU. Không cơ chế nào biết về cơ chế kia, và không cơ chế nào cần cơ chế kia để hoạt động. Quy tắc thực hành là: checkpoint các đoạn mà việc tính toán lại là rẻ, và để việc giải phóng hấp thụ các tensor bạn đã không checkpoint.

## 8. Giới hạn: giải phóng không phải là gì

- **Không phải cơ chế CPU.** Các tensor CPU không bao giờ bị đẩy ra; áp lực bộ nhớ host không do `MemoryManager` quản lý.
- **Không phải cơ chế TPU.** Các storage TPU được cấp phát trực tiếp qua TPU backend (`src/tensor/storage.cpp:37-50`) và không bao giờ đăng ký với `MemoryManager`. Cấp phát TPU thất bại sẽ lặng lẽ fallback về CPU thay vì đẩy ra (`storage.cpp:43-48`).
- **Không phải cache OpenCL.** Backend OpenCL có cơ chế giải phóng cache riêng, tách biệt: `CLBackend::evict_cache_to_limit` giải phóng toàn bộ các super-block trống trong khi `total_device_allocated_` vượt cùng `gpu_limit_` đó (`src/cl_backend/cl_backend.cpp:489-501`). Nó giải phóng các block cache trống, không phải các tensor đang sống. Đừng nhầm hai cái: một cái đẩy tensor ra, cái kia đẩy các block cache trống ra.
- **Không miễn phí.** Dữ liệu bị đẩy ra đi qua bus hai lần (ra khi evict, về khi swap-in). Một workload bị thrashing — đẩy các tensor ra rồi lập tức cần lại chúng — có thể chạy chậm hơn một workload chỉ đơn giản OOM sớm và buộc người dùng thu nhỏ mô hình. Nếu `memory_allocated()` dao động và throughput sụp đổ, cách sửa là footprint activation nhỏ hơn (checkpointing, batch nhỏ hơn), không phải giải phóng nhiều hơn.
- **Không thay thế cho giới hạn.** Giới hạn mặc định 512 MiB là nhỏ đối với các mô hình hiện đại và không thể tăng từ Python tại commit này. Bất kỳ workload nào cần ngân sách GPU lớn hơn đều phải đặt nó từ C++ hoặc chấp nhận việc giải phóng diễn ra liên tục.

## Ví dụ

<!-- ex: id=e3-1 device=cpu backend=cpu runtime=~90s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import torch
import litetorch as lt
lt.manual_seed(42)
torch.manual_seed(42)
x = lt.randn([2, 4])
x.requires_grad = True
w = lt.randn([4, 3])
w.requires_grad = True
def fn(t):
    return lt.Ops.relu(lt.Ops.matmul(t, w))
y = lt.checkpoint(fn, x)
y.sum().backward()
xt = torch.tensor(x.to_vector()).reshape(2, 4).requires_grad_(True)
wt = torch.tensor(w.to_vector()).reshape(4, 3).requires_grad_(True)
yt = torch.utils.checkpoint.checkpoint(lambda t: torch.relu(t @ wt), xt, use_reentrant=False)
yt.sum().backward()
print("fwd parity", torch.allclose(torch.tensor(y.to_vector()), yt.detach().flatten(), atol=1e-4, rtol=1e-4))
print("grad x parity", torch.allclose(torch.tensor(x.grad.to_vector()), xt.grad.flatten(), atol=1e-4, rtol=1e-4))
print("grad w parity", torch.allclose(torch.tensor(w.grad.to_vector()), wt.grad.flatten(), atol=1e-4, rtol=1e-4))
print("gia tri fwd", y.to_vector()[:3])
print("gia tri grad x", x.grad.to_vector()[:3])
```
Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4. Giá trị forward và cả hai gradient của đoạn checkpointed khớp với `torch.utils.checkpoint` ở mức 1e-4: việc tính toán lại tái tạo đúng forward pass.

<!-- ex: id=e3-2 device=cpu backend=cpu runtime=~55s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import torch
import litetorch as lt
print("cuda kha dung", lt.cuda.is_available())
print("so thiet bi", lt.cuda.device_count())
print("thiet bi tu dong", lt.auto_device())
print("backend", lt.get_backend_name())
print("bo nho da cap phat", lt.cuda.memory_allocated())
print("bo nho gpu da dung", lt.get_gpu_memory_used())
t = lt.randn([4, 4])
moved = t.to(lt.Device("cuda"))
print("yeu cau cuda, nhan duoc", moved.device)
print("torch cuda kha dung", torch.cuda.is_available())
```
Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. Trên máy chỉ có CPU, cả hai bộ đếm bộ nhớ đều đọc `0` vì không storage GPU nào từng đăng ký, nên việc giải phóng không bao giờ chạy. Dòng `.to(Device("cuda"))` cũng minh họa D-02: tensor lặng lẽ ở lại CPU.

<!-- ex: id=e3-3 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
lt.manual_seed(42)
print("cuda kha dung", lt.cuda.is_available())
print("so thiet bi", lt.cuda.device_count())
print("bo nho truoc", lt.cuda.memory_allocated())
held = []
for i in range(6):
    t = lt.randn([1024, 1024]).to(lt.Device("cuda"))
    held.append(t)
    print("sau cap phat", i, "bo nho da cap phat", lt.cuda.memory_allocated())
print("so cap phat giu duoc", len(held))
```
**REQUIRES-GPU — chưa kiểm chứng.** Ví dụ này cần GPU thật: sáu tensor FP32 4 MiB so với giới hạn mặc định 512 MiB sẽ tập dượt việc đăng ký, và các cấp phát lớn hơn sẽ buộc lần duyệt LRU ở §4. Trên máy chỉ có CPU này nó không thể chạy — `.to(Device("cuda"))` lặng lẽ fallback về CPU (D-02) và `memory_allocated()` vẫn là `0`. Không có `set_gpu_limit` trong Python, nên ví dụ chỉ quan sát giới hạn mặc định; hạ giới hạn để ép giải phóng theo yêu cầu cần dùng C++.

<!-- ex: id=e3-4 device=cpu backend=cpu runtime=~100s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import torch
import litetorch as lt
lt.manual_seed(42)
torch.manual_seed(42)
x = lt.randn([2, 4])
x.requires_grad = True
w1 = lt.randn([4, 8])
w1.requires_grad = True
w2 = lt.randn([8, 3])
w2.requires_grad = True
def segment(t):
    return lt.Ops.matmul(lt.Ops.relu(lt.Ops.matmul(t, w1)), w2)
y = lt.checkpoint(segment, x)
y.sum().backward()
xt = torch.tensor(x.to_vector()).reshape(2, 4).requires_grad_(True)
w1t = torch.tensor(w1.to_vector()).reshape(4, 8).requires_grad_(True)
w2t = torch.tensor(w2.to_vector()).reshape(8, 3).requires_grad_(True)
yt = torch.utils.checkpoint.checkpoint(lambda t: (torch.relu(t @ w1t) @ w2t), xt, use_reentrant=False)
yt.sum().backward()
print("fwd parity", torch.allclose(torch.tensor(y.to_vector()), yt.detach().flatten(), atol=1e-4, rtol=1e-4))
print("grad x parity", torch.allclose(torch.tensor(x.grad.to_vector()), xt.grad.flatten(), atol=1e-4, rtol=1e-4))
print("grad w1 parity", torch.allclose(torch.tensor(w1.grad.to_vector()), w1t.grad.flatten(), atol=1e-4, rtol=1e-4))
print("grad w2 parity", torch.allclose(torch.tensor(w2.grad.to_vector()), w2t.grad.flatten(), atol=1e-4, rtol=1e-4))
```
Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4. Một đoạn checkpointed hai lớp tính toán lại cả hai lớp trong backward pass và vẫn khớp gradient của torch ở mức 1e-4.
