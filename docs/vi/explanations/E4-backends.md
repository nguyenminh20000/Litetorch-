# E4 — Các Backend

LiteTorch chạy trên năm backend — CPU, CUDA, ROCm, OpenCL, TPU — đằng sau một API `Tensor` duy nhất. Trang này mô tả cách framework chọn một backend, cách các kernel được dispatch, và những chỗ nó lặng lẽ làm khác PyTorch. Mọi khẳng định đều dựa trên cây mã tại HEAD `b326f29` (extension build `88a4ba1`); bất cứ điều gì chưa được kiểm chứng trên phần cứng đều được đánh dấu như vậy. Tham chiếu chéo: D-02 / D-03 / D-10 trong trang Behavioral Differences.

## 1. Ma trận khả năng

| Backend | Cách tiếp cận | Trạng thái trên máy này (chỉ CPU, build 88a4ba1) | Đã kiểm chứng |
|---|---|---|---|
| CPU | Luôn có sẵn; fallback cho mọi thứ | `auto_device()` → `cpu:0`, `get_backend_name()` → `CPU` | Có, CPU harness (Ví dụ E4-1–E4-3) |
| CUDA | Thư viện GPU native `liblitetorch_gpu.so`, nạp lúc khởi động | Vắng mặt: `lt.cuda.is_available()` → `False`, `device_count()` → `0` | Không — cần Kaggle GPU (Ví dụ E4-4 là REQUIRES-GPU) |
| ROCm | Cùng thư viện native, bản build HIP (`__HIP_PLATFORM_AMD__` / `__HIPCC__`) | Vắng mặt; không có phần cứng ROCm ở đâu trong pipeline này | **UNVERIFIED-ROCm** — Ví dụ E4-5 được ship dưới dạng callout, không phải ví dụ đã kiểm chứng |
| OpenCL | `CLBackend`, chỉ dùng khi không có GPU native | Vắng mặt: `[litetorch Warning] GPU acceleration (OpenCL) is not available. Falling back to CPU.` | Không — cần một thiết bị OpenCL |
| TPU | PJRT / libtpu qua `TPUBackend` | Vắng mặt: `lt.is_tpu_available()` → `False` | Không — Kaggle TPU, chỉ sau khi GPU green (lệnh thường trực) |

Ma trận chính là điểm mấu chốt: trên máy không có accelerator, LiteTorch vẫn chạy mọi thứ trên CPU. Điều đó vừa tiện vừa nguy hiểm như nhau — xem §3.

## 2. Dispatch: cách một backend được chọn

**Lúc khởi động.** Constructor của `BackendDispatcher` (`src/backend/backend.cpp:521-541`) thử theo thứ tự:

1. Một `NativeGPUBackend`: nó `dlopen` `liblitetorch_gpu.so` từ một danh sách đường dẫn cố định — thư mục của chính extension trước, rồi `./build/`, `build/`, thư mục hiện tại, `/tmp/`, `/usr/local/lib/`, `/usr/lib/`, và `/opt/rocm/lib/` + `/opt/rocm/lib64/` (`backend.cpp:165-205`). Nếu thư viện nạp được và các symbol resolve được, dispatcher in dòng init ở §4 và giữ nó. Biến môi trường `LITETORCH_NO_GPU` / `LITETORCH_NO_NATIVE_GPU` tắt hoàn toàn bước này (`backend.cpp:160-162`).
2. Một `TPUBackend`: nếu có sẵn, nó in `[litetorch Info] Google TPU (Systolic Array) successfully initialized.` và được giữ lại.

Không có tệp cấu hình và không có cờ ưu tiên: GPU native thắng TPU theo thứ tự khởi tạo, và OpenCL hoàn toàn không phải là một backend của `BackendDispatcher` — nó chỉ được tham khảo khi GPU native vắng mặt.

**Định tuyến theo thiết bị.** `get_backend_for_device` (`backend.cpp:549-557`) ánh xạ TPU → TPU backend, GPU → native GPU backend, còn lại → CPU backend. Các thao tác bộ nhớ cấp thấp trong `include/litetorch/backend_route.h` cài đặt cùng quy tắc đó inline: `rt_gpu_allocate` / `rt_gpu_free` / `rt_gpu_read` / `rt_gpu_write` / `rt_gpu_copy` gọi backend native khi `has_native_gpu()` đúng và rơi xuống `CLBackend` trong các trường hợp còn lại. Nói cách khác, OpenCL là cách cài đặt fallback *của* đường GPU, không phải một backend ngang hàng: một máy có GPU native không bao giờ chạm tới OpenCL cho bộ nhớ tensor.

**`auto_device` và `get_backend_name`.** Việc chọn lựa mà Python nhìn thấy (`src/bindings/bind_core.cpp:69-110`) là một chuỗi ưu tiên cố định:

```text
auto_device():
    GPU native kha dung (va LITETORCH_NO_NATIVE_GPU khong dat)  -> gpu:0
    neu khong thi OpenCL kha dung                               -> gpu:0
    neu khong thi TPU kha dung                                  -> TPU:0
    neu khong                                                 -> CPU:0

get_backend_name():
    GPU native  -> "CUDA"  (hoac "ROCm" tren ban build HIP)
    OpenCL      -> "OpenCL"
    TPU         -> "TPU (Google PJRT / libtpu)"
    neu khong   -> "CPU"
```

Lưu ý `auto_device()` trả về `gpu:0` cho cả OpenCL lẫn GPU native: `Device` không ghi lại *cách cài đặt* GPU nào đứng sau nó. Code nào cần biết phải gọi `get_backend_name()`.

**Dispatch kernel.** Một lời gọi kernel trước tiên hỏi backend native về một kernel đã biên dịch sẵn theo tên (`gpu_get_kernel_fn`, `backend.cpp:486-497`); chỉ khi không có mới biên dịch từ mã nguồn. Bản sửa lỗi va chạm cached-kernel ngày 2026-10-26 (đã merge `agent-v1 @02cc271a`) là một phần của đường này: các kernel giờ được tra cứu theo tên, vì trước đây hai typedef backward FP32/FP16 giống hệt nhau từng chia sẻ một mục cache. Việc launch kernel cần symbol `gpu_launch` và ném lỗi nếu nó thiếu (`backend.cpp:503-504`).

## 3. Các quy tắc fallback: những cái lặng lẽ

Ba fallback là lặng lẽ — không lỗi, không cảnh báo, không log — và cả ba đều là độ lệch đang tồn tại so với PyTorch, được ghi là D-02, D-03, và D-10.

**D-02 — `.to(device)` lặng lẽ fallback về CPU.** Constructor GPU của `StorageImpl` (`src/tensor/storage.cpp:15-34`): nếu không có backend GPU, storage được xây trên CPU và `this->device` bị viết lại thành CPU. `Tensor::to(Device)` thừa hưởng điều này (`src/tensor/tensor_core.cpp:997-1006`), `Tensor::cuda()` cũng vậy (`tensor_core.cpp:2125-2128`), và `Module.to()` theo từng tham số (`src/nn/container.cpp:363`). Cấp phát TPU thất bại cũng fallback theo cùng cách (`storage.cpp:43-48`), trong khi hết bộ nhớ GPU thì *ném lỗi*. PyTorch ném lỗi với `.to("cuda")` khi không có CUDA. Ví dụ E4-2 cho thấy guard mà mọi chương trình ported đều cần: kiểm tra `t.device` sau `.to()` và báo lỗi lớn tiếng.

**D-03 — chuỗi thiết bị sai lặng lẽ thành CPU.** `Device("...")` phân tích `cuda` / `gpu` / `tpu` / `meta` (không phân biệt hoa thường, `:index` tùy chọn) và trả về CPU cho *bất cứ* thứ gì khác, kể cả lỗi gõ như `Device("cude")` (`src/bindings/bind_core.cpp:23-43`). Và `.to()` hoàn toàn không nhận dạng chuỗi: `Tensor.to` chỉ được bound cho `Device`, `DataType`, và `(Device, DataType)` (`src/bindings/bind_tpu.cpp:73-75`), nên `t.to("cuda")` ném `TypeError` của pybind trong khi `t.to(Device("cude"))` lặng lẽ chạy trên CPU. Ví dụ E4-3 minh họa cả hai.

**D-10 — các op nhị phân khác thiết bị lặng lẽ chuyển dữ liệu.** `add` / `sub` / `mul` / `div` di chuyển toán hạng lệch thiết bị sang thiết bị của toán hạng kia (`src/ops/elementwise.cpp:583, 723, 874, 1015`) thay vì ném lỗi như PyTorch. Kết hợp với D-02, một tensor CPU lạc có thể lặng lẽ ghim toàn bộ phép tính về CPU.

Fallback còn lại là ồn ào, và đó là cái nên dựa vào: **hết bộ nhớ GPU sẽ ném lỗi.** Một cấp phát GPU thất bại ném `GPU out of memory` kèm chuỗi chẩn đoán của `MemoryManager` (E3 §4, §6). Chỉ *sự vắng mặt* của một thiết bị là lặng lẽ; *sự cạn kiệt* của một thiết bị đang có thì không.

## 4. Cổng kiểm tra trước khi chạy: dòng init `Native GPU (CUDA/HIP)`

Khi backend native khởi tạo, dispatcher in đúng dòng sau:

```text
[litetorch Info] Native GPU (CUDA/HIP) successfully initialized.
```

(`src/backend/backend.cpp:525`.) Lệnh maximum-scrutiny ngày 2026-10-18 biến dòng này thành một **cổng kiểm tra trước khi chạy (pre-flight gate)** cho mọi lần chạy GPU: trước khi bất kỳ ví dụ GPU nào chạy trên Kaggle, session phải assert (a) dòng này có mặt và ghi CUDA/HIP, *không phải* OpenCL; (b) `liblitetorch_gpu.so` nằm cạnh extension dưới `<tree>/python/` (bản sao `/tmp` bị xóa khi session khởi động lại); (c) `lt.cuda.device_count()` trả về số lượng mong đợi (2× T4 trên image GPU của Kaggle). Nếu bất kỳ assert nào thất bại, lần chạy dừng lại và thất bại được báo cáo như một vấn đề harness/blocker, không phải như các ví dụ thất bại. Việc vắng dòng này kèm fallback CPU lặng lẽ chính là chế độ lỗi mà cổng này tồn tại để bắt: nếu không có nó, một lần chạy GPU sẽ thực thi trên CPU, vượt qua các kiểm tra số học một cách chậm chạp, và báo cáo thành công. Ví dụ E4-4 là cổng đó dưới dạng code; nó là REQUIRES-GPU và chưa được kiểm chứng trên máy này.

## 5. NCCL: các collective đa GPU

Các collective đa GPU đi qua `NCCLBridge` (`src/distributed/distributed.cpp:1083-1296`): `init`, `all_reduce`, `broadcast`, `all_gather`, `reduce_scatter`, với `nccl_check` ném `nccl <op> failed` khi mã trả về khác không. Bridge resolve 10 symbol bằng `dlsym` và yêu cầu 7 trong số đó (`ncclGetUniqueId`, `ncclCommInitRank`, `ncclAllReduce`, `ncclBroadcast`, `ncclAllGather`, `ncclReduceScatter`, `ncclCommDestroy`); nếu thiếu bất kỳ cái nào, nó đóng thư viện và báo cáo không khả dụng (`distributed.cpp:1115-1132`).

> **DRIFT — đọc điều này trước khi dùng phần NCCL.** Brief nhiệm vụ của trang này mô tả một biến môi trường override `LITETORCH_NCCL_LIB` với ngữ nghĩa fail-closed, thăm dò `pip nvidia/nccl`, rồi `/usr/local/cuda/lib64`, rồi các đường dẫn hệ thống, ưu tiên NCCL 2.x cao nhất và từ chối bản không phải 2.x. **Logic lựa chọn đó không tồn tại trong cây mã tại HEAD `b326f29`.** Đã kiểm chứng bằng tìm kiếm: `LITETORCH_NCCL_LIB` không xuất hiện ở đâu trong `integration/`; `distributed.cpp:1104-1108` chỉ đơn giản làm `dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL)`, rồi fallback sang `dlopen("librccl.so", ...)` — chỉ dùng đường dẫn tìm kiếm của system loader, không kiểm tra phiên bản, không có override môi trường. Việc nạp thất bại cũng là **fail-soft**, không phải fail-closed: `available_` được đặt thành `false` và các collective báo cáo không khả dụng (`distributed.cpp:1109-1113`). **Trạng thái, đã chốt 2026-10-27 theo dist-w22: patch đã kiểm chứng trên Kaggle, chưa merge.** Cơ chế lựa chọn (dist-w15) cùng toàn bộ vòng 2-GPU (w10–w13) đã được leader/commander phê duyệt và kiểm chứng trên Kaggle 2026-10-19 trên cây session đã vá, nhưng chưa từng chạy bước merge — `git log --all -S "LITETORCH_NCCL_LIB"` trống, và năm file patch vẫn nằm chưa merge trong `patches/distributed/`. Commander đã cho phép gửi lại w10→w15 thành một bộ có thứ tự để QA xem xét merge; cho đến khi bộ đó thực sự merge, hành vi hiện tại là dlopen fail-soft và trang này ghi lại code đúng như nó là. Hành vi đã kiểm chứng trên Kaggle được ghi trong memory (collective 2-GPU green, pip NCCL 2.31.2 được chọn tự động, 2026-10-19) phù hợp với việc system loader tìm thấy `libnccl.so` của pip trên đường dẫn thư viện, không phải với một thứ tự thăm dò tường minh. Đừng viết code đặt `LITETORCH_NCCL_LIB` với kỳ vọng nó sẽ được tôn trọng.

Những gì cây mã *có* đảm bảo, và một lần chạy GPU có thể dựa vào: kiểm chứng symbol trước khi dùng (một NCCL hiện diện một nửa được coi là vắng mặt, không phải crash về sau), và lỗi ồn ào tại thời điểm collective qua `nccl_check`. Những gì nó *không* đảm bảo: system loader tìm thấy `libnccl.so` nào, hay việc thiếu NCCL sẽ lỗi ồn ào lúc khởi động.

## 6. ROCm

> **UNVERIFIED-ROCm (quyết định của commander).** Các ví dụ ROCm được ship dưới dạng callout trong trang này; trang được bao phủ, không bị hoãn lại. Không có phần cứng ROCm ở máy local hay trên image GPU của Kaggle, nên không khẳng định ROCm nào dưới đây được kiểm chứng bằng thực thi.

Những gì cây mã xác lập về ROCm:

- Danh sách tìm kiếm thư viện native gồm `/opt/rocm/lib/liblitetorch_gpu.so` và `/opt/rocm/lib64/liblitetorch_gpu.so` (`src/backend/backend.cpp:172-173`).
- Cùng một `liblitetorch_gpu.so` phục vụ cả CUDA lẫn ROCm: dòng init là `Native GPU (CUDA/HIP)` trong cả hai trường hợp, và `get_backend_name()` trả về `"ROCm"` chỉ khi chính extension được build với `__HIP_PLATFORM_AMD__` hoặc `__HIPCC__` được định nghĩa (`src/bindings/bind_core.cpp:96-100`). Một extension build CUDA trên máy ROCm báo cáo `"CUDA"`, không phải `"ROCm"`.
- NCCL bridge fallback sang `librccl.so` (thư viện collective của ROCm) khi `libnccl.so` vắng mặt (`distributed.cpp:1106`).
- Bản sửa lỗi va chạm kernel-cache FP32/FP16 và yêu cầu `gpu_launch` áp dụng giống hệt trên đường HIP, vì cả hai đều đi qua `NativeGPUBackend`.

Những gì **chưa** được xác lập: bất kỳ kernel nào thực sự chạy đúng trên ROCm, thư viện native ROCm có build được không, hay `device_count()` có báo cáo các thiết bị AMD không. Ví dụ E4-5 là callout ROCm: nó nêu giá trị quan sát được mong đợi (`get_backend_name() == "ROCm"`) và các điều kiện để nó đúng, và nó được đánh dấu `backend=rocm`, `status=unverified`. Hãy coi mọi câu ROCm trong trang này là đọc từ cây mã, không phải khẳng định đã kiểm thử.

## 7. TPU

Backend TPU được khởi tạo thứ hai (`backend.cpp:532-538`) và chỉ được chọn khi không có GPU nào — native hay OpenCL — hiện diện (`bind_core.cpp:69-83`). Bề mặt Python của nó là `lt.is_tpu_available()` và constructor `Device("tpu")`. Hai điểm từ cây mã giới hạn việc dùng TPU: cấp phát TPU thất bại lặng lẽ fallback về CPU (E3 §8), và `flash_attention_half` chưa được cài đặt trên TPU và sẽ ném lỗi (`src/backend/tpu/tpu_backend.cpp:343`, D-20). Các ví dụ TPU là REQUIRES-TPU và chỉ chạy trên Kaggle sau khi GPU green, theo lệnh thường trực; không ví dụ nào được ship trong batch này.

## Ví dụ

<!-- ex: id=e4-1 device=cpu backend=cpu runtime=~70s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import torch
import litetorch as lt
print("thiet bi tu dong", lt.auto_device())
print("ten backend", lt.get_backend_name())
print("gpu kha dung", lt.is_gpu_available())
print("tpu kha dung", lt.is_tpu_available())
print("cuda kha dung", lt.cuda.is_available())
print("so thiet bi cuda", lt.cuda.device_count())
print("torch cuda kha dung", torch.cuda.is_available())
```
Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. Trên máy chỉ có CPU này, toàn bộ chuỗi phát hiện đều rơi về CPU: không GPU native, không OpenCL, không TPU.

<!-- ex: id=e4-2 device=cpu backend=cpu runtime=~80s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import torch
import litetorch as lt
t = lt.randn([2, 2])
moved = t.to(lt.Device("cuda"))
print("yeu cau cuda, nhan duoc", moved.device)
print("fallback lang le", str(moved.device) == "cpu:0")
def require_device(tensor, device):
    if str(tensor.device) != str(device):
        raise RuntimeError("device mismatch")
try:
    require_device(moved, lt.Device("cuda"))
    print("guard vuot qua")
except RuntimeError:
    print("guard da nem loi")
try:
    torch.randn([2, 2]).to("cuda")
    print("torch lang le")
except Exception as e:
    print("torch da nem loi", type(e).__name__)
```
Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. LiteTorch lặng lẽ giữ tensor trên CPU (D-02); guard `require_device` là mẫu bắt buộc cho code ported. PyTorch ném lỗi thay vào đó.

<!-- ex: id=e4-3 device=cpu backend=cpu runtime=~10s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
print("cuda", lt.Device("cuda"))
print("CUDA:0", lt.Device("CUDA:0"))
print("loi go cude", lt.Device("cude"))
print("tpu", lt.Device("tpu"))
print("meta", lt.Device("meta"))
try:
    lt.randn([2, 2]).to("cuda")
    print("to dang chuoi ok")
except TypeError:
    print("to dang chuoi nem TypeError")
```
Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. Lỗi gõ `Device("cude")` được phân tích thành CPU mà không báo lỗi (D-03), trong khi `.to("cuda")` với chuỗi ném `TypeError` vì không có overload chuỗi nào được bound.

<!-- ex: id=e4-4 device=gpu backend=cuda runtime=~? torch-parity=n/a status=verified-2026-10-31@cbba5af3 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import os
import litetorch as lt
print("cuda kha dung", lt.cuda.is_available())
print("so thiet bi", lt.cuda.device_count())
print("ten backend", lt.get_backend_name())
print("thu vien native canh extension", os.path.exists(os.path.join(os.path.dirname(lt.__file__), "liblitetorch_gpu.so")))
lt.cuda.synchronize()
print("gate vuot qua")
```
**REQUIRES-GPU — chưa kiểm chứng.** Cổng kiểm tra trước khi chạy trên Kaggle dưới dạng code: backend native hiện diện, số thiết bị như mong đợi, `liblitetorch_gpu.so` nằm cạnh extension. Bản thân dòng init `Native GPU (CUDA/HIP)` được dispatcher C++ in ra lúc import và được kiểm tra trong log của session, không phải từ Python. Trên máy chỉ có CPU này, `is_available()` là `False` và cổng thất bại, đúng như nó phải thế.

<!-- ex: id=e4-5 device=gpu backend=rocm runtime=~? torch-parity=n/a status=unverified -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
print("ten backend", lt.get_backend_name())
print("cuda kha dung", lt.cuda.is_available())
print("so thiet bi", lt.cuda.device_count())
```
**UNVERIFIED-ROCm — chưa kiểm chứng, REQUIRES-GPU.** Giá trị quan sát được mong đợi trên extension build HIP với thiết bị ROCm: `get_backend_name()` trả về `"ROCm"`, `cuda.is_available()` là `True`, và dispatcher in `Native GPU (CUDA/HIP) successfully initialized.` Không có phần cứng ROCm trong pipeline này, nên callout này chỉ là đọc từ cây mã (§6): nó được ship để đường ROCm được ghi lại, không bị hoãn — và không được đọc như một khẳng định đã kiểm thử.
