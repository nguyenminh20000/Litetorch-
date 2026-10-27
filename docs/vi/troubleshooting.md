# Khắc Phục Sự Cố và FAQ

Mọi mục dưới đây đều lấy từ một lỗi thực sự quan sát được trong quá trình viết lại tài liệu (các đợt B0–B4) hoặc một khuyết điểm đã xác minh trong cây mã. Triệu chứng, nguyên nhân, cách sửa — không có chế độ lỗi bịa đặt. Mỗi mục liên kết tới trang Khác Biệt Hành Vi (số D) hoặc tham chiếu API, nơi hành vi được ghi kèm số liệu.

## Import và cài đặt

### `ModuleNotFoundError: No module named 'litetorch'`

**Nguyên nhân:** extension đã build nằm dưới `<tree>/python/`, không phải gốc cây. `setup.py` đặt `package_dir={"": "python"}`, nên `build_ext --inplace` đặt `.so` có thể import được dưới `<tree>/python/`. Đây là nguyên nhân gốc của lỗi import Phase 6 (2026-10-17).

**Cách sửa:** thêm `<tree>/python` vào `sys.path` trước khi import — đừng bao giờ import từ gốc cây:

<!-- ex: id=ts-import-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt

print("build_commit:", lt.build_commit())
print("backend:", lt.get_backend_name())
```

Cũng kiểm tra cổng thật, không phải log build: `setup.py` in `BUILD OK` ngay cả khi biên dịch nvcc từng tập tin thất bại. Cổng thật là `.so` tồn tại và `import litetorch` thành công. Trên Kaggle, `liblitetorch_gpu.so` phải nằm cạnh extension dưới `python/` — đường sao chép tới `/tmp` bị xóa khi session khởi động lại.

### `ImportError` / thiếu symbol sau khi build

**Nguyên nhân:** extension cũ từ commit cũ hơn, hoặc bản build lặng lẽ thất bại từng tập tin (xem trên).

**Cách sửa:** build lại bằng `python3 setup.py build_ext --inplace`, kiểm tra `lt.build_commit()` khớp với HEAD của cây, và kiểm tra tập tin extension tồn tại dưới `<tree>/python/`. Nếu bản build bị gián đoạn, việc tiếp tục có thể kích hoạt build lại toàn bộ setuptools — hãy tính trước (build đầy đủ cục bộ không hoàn thành trong một session trên VM 2-CPU; build xác minh chạy trên Kaggle CPU).

## Thiết bị

### "Lần chạy GPU của tôi chạy, nhưng chậm" / "thiết bị là cpu sau `.to('cuda')`"

**Nguyên nhân:** rơi về CPU lặng lẽ (D-02). `.to(Device("cuda"))` trên máy không có backend GPU lặng lẽ trả về tensor CPU — không lỗi, không cảnh báo. Chuỗi thiết bị đánh máy sai cũng vậy (D-03): `Device("cude")` lặng lẽ thành CPU.

**Cách sửa:** kiểm tra `t.device` sau mỗi `.to()` / `.cuda()` và báo lỗi rõ ràng khi không khớp:

```python skip-run
t = t.to(lt.Device("cuda"))
assert str(t.device).startswith("cuda"), f"mong cuda, nhan duoc {t.device} (D-02)"
```

Cũng kiểm tra `lt.cuda.is_available()` trước khi cho rằng GPU tồn tại, và viết đúng chính tả chuỗi thiết bị (`lt.Device("cuda:0")`, không phải `lt.Device("cuda", 0)` — hàm tạo chuỗi hai đối số không tồn tại).

### Op nhị phân lặng lẽ di chuyển tensor giữa các thiết bị

**Nguyên nhân:** `add`/`sub`/`mul`/`div` (và các phép so sánh, matmul, op shape) di chuyển toán hạng lệch thiết bị sang thiết bị của toán hạng còn lại thay vì báo lỗi (D-10).

**Cách sửa:** di chuyển tensor một cách tường minh trước khi trộn thiết bị; coi mọi chuyển ngầm là lỗi cần tìm, không phải tính năng.

## Huấn luyện không học

### Loss giảm trong torch nhưng đứng yên / thành NaN trong LiteTorch

**Nguyên nhân (khả năng cao nhất):** gradient sai trong khi giá trị forward trông đúng. Đây chính là khuyết điểm no-learn ngày 2026-10-25/26 (va chạm `cached_native_kernel` chạy `ConvolutionBackwardData` cho mọi lần gọi "filter") — và cùng dạng lỗi tái diễn với DDP (D-33: backward nhân đôi một lát gradient) và NAdam (D-30: bước đầu tiên lớn hơn ~6.5 lần).

**Cách sửa:** so sánh gradient, không chỉ giá trị forward. Chạy thủ tục gradcheck / parity trong `plan/verification.md` §3: sao chép trọng số từ LiteTorch sang torch, chạy cùng một bước trên cả hai, so sánh loss VÀ gradient của mọi tham số ở FP32 1e-4. Nếu forward khớp mà gradient không khớp, khuyết điểm nằm ở đường backward — hãy gửi báo cáo, đừng chỉnh tốc độ học để né nó.

### Optimizer bước nhưng tham số không bao giờ thay đổi

**Nguyên nhân:** module không có tham số nào được đăng ký. Gán submodule dưới dạng thuộc tính Python (`self.linear = lt.nn.Linear(4, 4)`) KHÔNG đăng ký nó — `parameters()` trả về 0 và `state_dict()` trống (D-24).

**Cách sửa:** đăng ký các module con một cách tường minh bằng `add_module(name, module)`, hoặc ghi đè `parameters()` để thu thập submodule bằng tay (tutorial T1 `MultiLayerPerceptron` làm vậy). Xác minh bằng `len(model.parameters())` trước khi tạo optimizer.

### Loss thành NaN ngay bước đầu tiên

**Nguyên nhân:** kiểm tra các nghi phạm thường gặp trước (tốc độ học, thang dữ liệu), rồi kiểm tra xem bản thân hàm loss có phải vấn đề không: CE/NLL/fused-LCE từ chối target INT64 (D-23) — nhưng lỗi đó báo lỗi, không thành NaN. NaN thật sự ở bước 0 với gradient đúng chỉ tới optimizer: bước đầu tiên của NAdam lớn hơn ~6.5 lần (D-30).

**Cách sửa:** chuyển sang Adam/AdamW và chạy lại; nếu NaN biến mất, đó là D-30.

### `Parameter(tensor)` báo lỗi khi tạo

**Nguyên nhân:** `Parameter(tensor)` bị hỏng trên 88a4ba1 — luôn báo lỗi "copy_: a leaf Tensor that requires grad is being used in an in-place operation", ngay cả với đầu vào requires_grad=False (D-25).

**Cách sửa:** không có — đừng tạo `Parameter` trực tiếp; lấy tham số từ module (ví dụ `Linear.weight`).

### `m(q, k, v)` báo lỗi TypeError

**Nguyên nhân:** `__call__()` chỉ đi qua forward một tensor (D-22). Module đa đối số — MultiHeadAttention, Bilinear, mọi hàm loss 2 đối số — phải gọi qua `.forward(...)`.

**Cách sửa:** `m.forward(q, k, v)`.

## Dữ liệu

### `DataLoader` báo lỗi `RuntimeError: Tried to call pure virtual function "data::Dataset::size"`

**Nguyên nhân:** subclass `Dataset` định nghĩa `__len__`/`__getitem__`, mà trampoline `PyDataset` không bao giờ gọi — nó điều phối tới các phương thức Python tên `size` và `get` (D-37).

**Cách sửa:** định nghĩa `size()` và `get(idx)` đúng như cách viết. Khi chúng tồn tại, `len(dataset)` / `dataset[i]` chạy được (chúng được bind tới cùng phương thức C++).

### Tiến trình abort khi thoát với `FATAL: exception not rethrown` (SIGABRT)

**Nguyên nhân:** `DataLoader` C++ giữ dataset backed bởi Python quá thời điểm tháo dỡ (D-37). `collate_fn` Python kích hoạt cùng lỗi abort. Cùng dạng lỗi xuất hiện với `RendezvousStore` (D-35: join thread `run()` sau `stop()`) và không ổn định với `AsyncCheckpointSaver` (D-36).

**Cách sửa:** `del loader` / `del dataset` tường minh trước khi thoát (đã xác minh thoát sạch exit 0). Với `RendezvousStore`, `join()` thread `run()` sau `stop()`. Với `AsyncCheckpointSaver`, `del saver` trước khi thoát.

### Loss các epoch không tái lập được dù đã `lt.manual_seed(42)`

**Nguyên nhân:** `DataLoader(shuffle=True)` dùng RNG riêng, không phải luồng `manual_seed` (D-46). Loss epoch 1 thay đổi 1.0306 / 0.9762 / 1.0724 giữa các lần chạy trong B1.

**Cách sửa:** dùng `shuffle=False` cho các lần chạy tái lập (mọi ví dụ trong tài liệu đều vậy).

### Tương đương `torch.save(state_dict, path)` thất bại

**Nguyên nhân:** `lt.save`/`lt.load` là in-place và điều phối theo kiểu — tensor trần báo lỗi; `save_parameters` nhận list tham số, không phải module; `save_safetensors`/`save_async` nhận dict, không phải module (D-45).

**Cách sửa:** `save_parameters(model.parameters())` cho trọng số; `save_safetensors(dict)` cho dict; `state_dict()`/`load_state_dict(sd)` cho chuyển mô hình đang chạy. `save_optimizer_state` chỉ hỗ trợ Adam/AdamW/SGD/RMSprop.

## Hàm loss và số học

### `cross_entropy_loss: target must have integer or float dtype`

**Nguyên nhân:** CE/NLL/fused-LCE từ chối target INT64 và yêu cầu INT32 (D-23). Sau bản sửa D-06, `.long()` trả về INT64 — và các hàm loss này từ chối nó.

**Cách sửa:** ép kiểu target sang INT32 một cách tường minh (`.cast(litetorch.DataType.INT32)`); đừng truyền kết quả `.long()` làm target loss.

### Giá trị GELU khác torch ~4e-4

**Nguyên nhân:** GELU của LiteTorch chỉ là xấp xỉ tanh; không có đối số `approximate` và không có đường erf (D-09). Đây là chủ đích, không phải lỗi.

**Cách sửa:** dùng dung sai ít nhất ~5e-4 cho GELU, hoặc so sánh với `F.gelu(x, approximate='tanh')` (khớp chính xác, 0.0).

### `gradcheck` thất bại trên một hàm đơn giản

**Nguyên nhân:** `eps=1e-6` mặc định nằm dưới độ phân giải FP32 (D-34). Ngay cả `t*t` cũng thất bại tại x=3 (đạo hàm giải tích 6 so với đạo hàm số 5.72).

**Cách sửa:** luôn truyền `eps=1e-3` một cách tường minh cho gradcheck FP32:

<!-- ex: id=ts-gradcheck-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt

cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([3.0], [1], cpu, True)

def sq(v):
    return lt.Ops.mul(v, v)

print("gradcheck eps=1e-6:", lt.autograd.gradcheck(sq, [x], 1e-6, 1e-5, 1e-4, False))
print("gradcheck eps=1e-3:", lt.autograd.gradcheck(sq, [x], 1e-3, 1e-5, 1e-4, False))
```

Lưu ý: `gradcheck` xóa `x.grad` — tính gradient giải tích bằng backward tường minh sau gradcheck.

### `.numpy()` báo lỗi "unsupported dtype for copy conversion"

**Nguyên nhân:** `.numpy()` chỉ xử lý FP64/FP32/FP16/INT32/INT16/INT8; UINT8, INT64, BOOL rơi vào default và báo lỗi (D-14). Tensor không phải CPU cũng báo lỗi.

**Cách sửa:** `.cpu()` trước, rồi `.cast(FP32)` (lưu ý mất chính xác với int64 trên 2^24), hoặc dùng `.tolist()` / `.to_vector()`.

### `.item()` trả về giá trị sai với int64 lớn

**Nguyên nhân:** `.item()` trả về float, ép tensor không phải FP32 sang FP32 trước; giá trị int64 trên 2^53 mất chính xác (D-18).

**Cách sửa:** tránh `.item()` với giá trị int64 lớn; đọc qua `.numpy()` (sau khi ép kiểu, xem D-14) hoặc giữ giá trị ở dạng tensor.

### `linalg.matrix_norm(A, ord=1)` báo lỗi TypeError

**Nguyên nhân:** `ord` phải là chuỗi trong binding này (D-42). Cả 8 ord chuỗi đều chạy: fro/nuc/1/2/inf/-inf/-1/-2.

**Cách sửa:** truyền `ord="1"`.

## Huấn luyện phân tán

### Huấn luyện DDP lệch khỏi huấn luyện đơn tiến trình

**Nguyên nhân:** `DistributedDataParallel.backward` nhân đôi một lát của gradient trọng số phẳng (D-33). Gradient bias là đúng; backward đơn tiến trình là đúng.

**Cách sửa:** đừng huấn luyện với DDP cho tới khi sửa xong — dùng huấn luyện đơn tiến trình hoặc FSDP (`fsdp_fully_shard`).

### `ring_attention` báo lỗi pybind holder

**Nguyên nhân:** `ring_attention` không thể gọi từ Python (holder tham chiếu `ProcessGroup.get()` so với tham số `shared_ptr`) (D-38).

**Cách sửa:** dùng cách dự phòng sharding dãy bằng tay trong H2.4 (`all_gather`/`reduce_scatter`).

### `destroy_process_group()` báo lỗi AttributeError

**Nguyên nhân:** nó không được bind — dùng `shutdown()` (xem cách giải quyết D9 trong `plan/d9-resolution.md`).

**Cách sửa:** `lt.distributed.shutdown()`.

### Không thể tạo `FSDPUnit`

**Nguyên nhân:** phương thức `FSDPUnit` không tới được từ Python (không có `py::init`, `active_units()` không được bind). FSDP shard tham số + gradient, KHÔNG phải trạng thái optimizer.

**Cách sửa:** dùng `fsdp_fully_shard(module)` làm điểm vào (xem H2.1).

## Tràn bộ nhớ

### OOM sau huấn luyện / OOM MNIST

**Trạng thái:** OOM sau huấn luyện là vấn đề mở đã biết, chưa giải quyết (9 lần xuất hiện giống hệt). OOM MNIST [BLOCKER] **ĐÃ ĐÓNG 2026-10-27 vì không tái hiện được**: Path A (qa-r58a) chạy đúng cấu hình lỗi (MLP 784→128→10, BATCH=256, EPOCHS=3) trên tip b476a5e và pass — rc 0, loss 0.5112→0.2304→0.1747, accuracy 95.47%, parity với torch 1.192e-07, không OOM. Chẩn đoán qa-r58 trước đó cũng không tái hiện được trên cdbc716c (cả hai lần chạy exit 0, đỉnh dùng GPU của lt ~231 MiB). Đóng với nghĩa "không tái hiện được trên tip hiện tại", KHÔNG phải chứng minh đã sửa: không có nguyên nhân được đặt tên, và rà soát delta của Backend không tìm thấy ứng viên sửa nào trong 536f3b5→cdbc716c. Nếu OOM xuất hiện lại, gửi repro với đúng cấu hình. Đừng coi đây là lỗi tài liệu — tài liệu ghi chúng là vấn đề đã biết.

**Kiểm tra trước tiên:** OOM xảy ra sau khi huấn luyện hoàn tất (OOM sau huấn luyện — bản thân huấn luyện đã thành công) hay trong lúc nạp dữ liệu (OOM MNIST). Giảm batch size / bật gradient checkpointing làm cách khắc phục; gửi repro nếu lỗi là mới.

### `flash_attention_half` báo lỗi trên TPU

**Nguyên nhân:** `flash_attention_half` trên TPU không được triển khai và sẽ báo lỗi (D-20).

**Cách sửa:** dùng attention FP32 trên TPU, hoặc chạy attention trên GPU.

## FAQ

**H: LiteTorch có phải là thay thế trực tiếp cho PyTorch không?**
Đ: Không. Bề mặt API cố tình tương tự, nhưng trang Khác Biệt Hành Vi ghi 46 khác biệt đã xác minh (D-01–D-46), vài cái trong đó là lặng lẽ. Chuyển đổi theo hướng dẫn chuyển đổi, xác minh bằng parity.

**H: Phiên bản PyTorch nào là tham chiếu?**
Đ: torch 2.14.1+cpu, đã thăm dò trực tiếp cho mọi khẳng định phía PyTorch. Dung sai parity FP32 là 1e-4 (tiêu chí QA hiện hành); FP16/BF16 là 1e-2.

**H: Vì sao mô hình chuyển đổi của tôi huấn luyện được trong torch nhưng không trong LiteTorch?**
Đ: Theo thứ tự: (1) kiểm tra `parameters()` không trống (D-24); (2) so sánh gradient, không chỉ giá trị forward (bài học no-learn); (3) kiểm tra DDP (D-33) hoặc NAdam (D-30); (4) kiểm tra target loss là INT32 (D-23).

**H: Tôi chạy ví dụ GPU ở đâu?**
Đ: Trên Kaggle, qua QA — Docs không bao giờ tự điều khiển session Kaggle. Ví dụ GPU được đánh dấu `REQUIRES-GPU` và gom thành một gói `kaggle-run-inputs` cho mỗi đợt. Cổng tiền kiểm tra khẳng định dòng init `Native GPU (CUDA/HIP)`, vị trí `liblitetorch_gpu.so`, và `device_count() == 2`.

**H: Tôi có thể dùng ví dụ ROCm không?**
Đ: Chúng được phát hành dưới dạng chú thích `UNVERIFIED-ROCm` tường minh — chưa có phần cứng ROCm nào xác minh chúng.

**H: Cách nhanh nhất để kiểm tra một khẳng định trong tài liệu này là gì?**
Đ: Mọi ví dụ mã đều mang header metadata (`id/device/backend/runtime/torch-parity/status`) và chạy độc lập. Trích xuất bằng `docs/tools/extract_examples.py` và chạy bằng `docs/tools/run_examples.py` — cùng harness đã xác minh đợt này.

**H: Tôi tìm thấy hành vi mâu thuẫn với tài liệu này. Tôi làm gì?**
Đ: Gửi lên bảng comms (thread `docs-fix-requests-20261026`) kèm ví dụ, commit của extension (`lt.build_commit()`), và bản mirror torch. Một khẳng định trong tài liệu bị chính đầu ra ví dụ của nó bác bỏ là FAIL của tài liệu, không phải của mã.
