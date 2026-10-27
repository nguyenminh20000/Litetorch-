# Khác Biệt Hành Vi so với PyTorch

LiteTorch hướng tới tính tương thích số học với PyTorch, nhưng nó không phải là PyTorch. Trang này là bản ghi chuẩn của mọi khác biệt hành vi đã xác minh giữa LiteTorch và PyTorch: LiteTorch làm gì (kèm tập tin nguồn và dòng), PyTorch làm gì (đã thăm dò trực tiếp với torch 2.14.1+cpu trừ khi ghi chú khác), tác động, và cách chuyển đổi hoặc khắc phục tạm thời. Một khác biệt được ghi là OPEN là có thật và đang tồn tại trong cây mã; một khác biệt được ghi là FIXED đã được sửa và giữ lại ở đây dưới dạng lịch sử để đối chiếu mã cũ và tài liệu cũ.

- **Cơ sở:** cây HEAD `b326f29`; các ví dụ trong trang này chạy trên extension D3 của Core `build_commit 88a4ba1` (`workspace/framework/core/litetorch-d3/python`); PyTorch tham chiếu torch 2.14.1+cpu trong `~/torch-venv`.
- **Khóa trạng thái:** `OPEN` — đang tồn tại trong cây, chưa có bản sửa được hợp nhất. `FILED` — khuyết điểm đã gửi cho team sở hữu qua bảng comms, bản sửa chưa hợp nhất. `FIXED` — đã sửa và hợp nhất; hành vi cũ chỉ còn được ghi lại dưới dạng lịch sử. `DOCUMENTED` — khác biệt có chủ đích, không có kế hoạch sửa; tài liệu là hợp đồng.
- **Đánh số:** D-01–D-21 được xác minh trong đợt lập kế hoạch B0 (`plan/differences.md`). D-22–D-29 được tìm thấy trong đợt tham chiếu nn B3. D-30–D-46 được tìm thấy trong các đợt optim/linalg/data/distributed B3 và được đánh số trong đợt này (B4). D-47–D-49 (autograd) được tìm thấy trong đợt giải thích B4 và được leader bổ sung sau khi worker front2 hoàn thành.
- **Quy tắc cho người chuyển đổi:** đừng bao giờ cho rằng một cách viết PyTorch sẽ chạy trong LiteTorch cho tới khi trang này hoặc trang tham chiếu API nói như vậy. Khi nghi ngờ, hãy chạy kiểm tra parity trong `plan/verification.md` §3.

## Minh họa trực tiếp

Ba hành vi dưới đây được minh họa trực tiếp trên extension xác minh, để người đọc thấy khác biệt thay vì chỉ tin lời.

<!-- ex: id=bd-d01-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt

cpu = lt.Device("cpu")
t_int = lt.Tensor.from_vector([1, 2, 3], [3], cpu, False).cast(lt.DataType.INT32)
print("co data_ptr:", hasattr(t_int, "data_ptr"))
print("co raw_data_ptr:", hasattr(t_int, "raw_data_ptr"))
```

<!-- ex: id=bd-d02-1 device=cpu backend=cpu runtime=~2s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt

cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0], [2], cpu, False)
t2 = t.to(lt.Device("cuda"))
print("yeu cau cuda, nhan duoc thiet bi:", t2.device)
print("cuda kha dung:", lt.cuda.is_available())
```

<!-- ex: id=bd-d09-1 device=cpu backend=cpu runtime=~94s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch

cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([-3.0, -1.0, 0.0, 1.0, 3.0], [5], cpu, False)
g_lt = lt.Ops.gelu(x).to_vector()
xt = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
g_exact = torch.nn.functional.gelu(xt).tolist()
g_tanh = torch.nn.functional.gelu(xt, approximate="tanh").tolist()
print("max |lt - torch chinh xac|:", max(abs(a - b) for a, b in zip(g_lt, g_exact)))
print("max |lt - torch tanh|:", max(abs(a - b) for a, b in zip(g_lt, g_tanh)))
```

## D-01 — `data_ptr()` báo lỗi trên tensor không phải FP32 (C++); không được bind trong Python — OPEN

- LiteTorch (ĐÃ XÁC MINH): `Tensor::data_ptr()` báo lỗi `runtime_error("[litetorch Error] data_ptr(): tensor dtype is not FP32; use raw_data_ptr() for byte-level access")` với mọi kiểu dữ liệu khác FP32 (`src/tensor/tensor_core.cpp:1292-1297`). `raw_data_ptr()` trả về `void*` cho mọi kiểu dữ liệu (`tensor_core.cpp:1300-1303`). Cả hai được khai báo trong `include/litetorch/tensor.h:185-186`.
- **Đính chính (B4, đã xác minh trực tiếp trên extension 88a4ba1): không phương thức nào được bind sang Python.** `hasattr(t, "data_ptr")` và `hasattr(t, "raw_data_ptr")` đều là `False`; không có `.def("data_ptr")` nào trong `src/bindings/`. Hành vi báo lỗi ở trên chỉ áp dụng cho phía gọi C++. Mã Python gọi `t.data_ptr()` nhận `AttributeError`, không phải lỗi runtime đã ghi.
- PyTorch: `Tensor.data_ptr()` chạy với mọi kiểu dữ liệu và trả về địa chỉ thô.
- Tác động: CAO. Mọi mã chuyển đổi dùng `t.data_ptr()` trên tensor int/bool/FP16/FP64 đều sập; ở phía Python phương thức này hoàn toàn không tồn tại. Lưu ý `data_ptr()` trả về con trỏ CPU (`storage->get_cpu_ptr() + offset`) ngay cả với tensor GPU — nó không trả về con trỏ thiết bị.
- Chuyển đổi: trong C++, dùng `raw_data_ptr()` để truy cập mức byte và tự ép kiểu sang kiểu phần tử. Trong Python, không có cách tương đương — dùng `.numpy()` (xem D-14) hoặc `.to_vector()` để đọc dữ liệu ra.
- Xem ví dụ trực tiếp `bd-d01-1` ở trên.

## D-02 — `to(device)` lặng lẽ rơi về CPU khi thiết bị tăng tốc không khả dụng — OPEN

- LiteTorch (ĐÃ XÁC MINH): `Tensor::to(Device)` tạo một `StorageImpl` trên thiết bị đích; nếu backend GPU vắng mặt, bộ nhớ lặng lẽ rơi về CPU và `final_device` bị viết lại thành CPU mà không có lỗi, không cảnh báo, không log (`src/tensor/tensor_core.cpp:997-1006`; nguyên nhân gốc `src/tensor/storage.cpp:31-33`). Tương tự với TPU (nhánh TPU trong `storage.cpp`), và lỗi cấp phát TPU CŨNG lặng lẽ rơi về CPU, trong khi tràn bộ nhớ GPU thì báo lỗi. `Tensor::cuda(idx)` đi qua cùng một đường (`tensor_core.cpp:2125-2128`), nên `.cuda()` trên máy không có GPU lặng lẽ trả về tensor CPU. `Module.to()` kế thừa hành vi này theo từng tham số (`src/nn/container.cpp:363`).
- PyTorch: `.to("cuda")` / `.cuda()` trên máy không có CUDA sẽ báo lỗi (thăm dò: `AssertionError: Torch not compiled with CUDA enabled`).
- Tác động: CAO. Chạy sai thiết bị một cách lặng lẽ: mã tin rằng nó chạy trên GPU nhưng thực ra chạy trên CPU; benchmark và huấn luyện chậm đi một cách lặng lẽ; các assert thiết bị ở phía sau thất bại một cách khó hiểu.
- Chuyển đổi: luôn kiểm tra `t.device` sau `.to()` / `.cuda()` và báo lỗi rõ ràng nếu nó không phải thiết bị đã yêu cầu.
- Xem ví dụ trực tiếp `bd-d02-1` ở trên.

## D-03 — `.to()` / `Module.to()` không chấp nhận dạng chuỗi; chuỗi thiết bị sai lặng lẽ thành CPU — OPEN

- LiteTorch (ĐÃ XÁC MINH): `Tensor.to` chỉ được bind cho `Device`, `DataType`, `(Device, DataType)` (`src/bindings/bind_tpu.cpp:73-75`). `Module.to` chỉ nhận `Device` (`src/bindings/bind_nn_base.cpp:10`; `src/nn/container.cpp:363`). Truyền chuỗi sẽ báo lỗi pybind TypeError. `Device("...")` phân tích "cuda"/"gpu"/"tpu"/"meta" (không phân biệt hoa thường, kèm `:index` tùy chọn) và trả về CPU cho BẤT CỨ thứ gì khác — kể cả lỗi đánh máy như `Device("cude")` — mà không báo lỗi (`src/bindings/bind_core.cpp:23-43`).
- PyTorch: `.to("cuda")`, `.to("cpu")`, `.to(torch.float32)` đều chạy; chuỗi thiết bị không hợp lệ sẽ báo lỗi.
- Tác động: TRUNG BÌNH. Mã chuyển đổi `model.to("cuda")` / `t.to("cuda:0")` thất bại ngay tại chỗ gọi; chuỗi thiết bị đánh máy sai lặng lẽ chạy trên CPU (cộng hưởng với D-02).
- Chuyển đổi: tạo `litetorch.Device("cuda")` một cách tường minh và kiểm tra `t.device` sau đó.

## D-04 — Hàm factory chỉ nhận một dãy shape; không có dạng variadic — OPEN

- LiteTorch (ĐÃ XÁC MINH): `litetorch.randn(shape, ...)`, `zeros`, `ones` chỉ nhận `const std::vector<int64_t>& shape` (`src/bindings/bind_data.cpp:175`; `src/bindings/bind_tpu.cpp:37`). `randn(2, 3)` sẽ báo lỗi.
- PyTorch: `torch.randn(2, 3)` và `torch.randn([2, 3])` đều chạy (thăm dò).
- Tác động: THẤP. Ma sát chuyển đổi cơ học.
- Chuyển đổi: luôn truyền list/tuple: `litetorch.randn([2, 3])`.

## D-05 — Không có phương thức `Tensor.pow()`; `**` và `Ops.pow` chỉ nhận số mũ vô hướng — OPEN

- LiteTorch (ĐÃ XÁC MINH): không có `.def("pow")` trên Tensor. `__pow__` tồn tại nhưng chỉ nhận `double` (`src/bindings/bind_tpu.cpp:134`). `Ops.pow(a, exponent)` nhận số mũ `float` (`src/bindings/bind_ops.cpp:19`; `include/litetorch/ops.h:35`). Dạng hai tensor `pow(a, b)` được khai báo trước (`src/ops/elementwise.cpp:124`) nhưng không bao giờ được định nghĩa hay bind.
- PyTorch: `t.pow(exp)`, `t ** exp`, và `t ** u` (số mũ là tensor) đều chạy (thăm dò: `tensor([2.]) ** tensor([3.]) == [8.]`).
- Tác động: THẤP-TRUNG BÌNH. `t ** u` và `t.pow(u)` thất bại; `t.pow(2.0)` phải viết lại.
- Chuyển đổi: dùng `litetorch.Ops.pow(t, float_exp)`.

## D-06 — `.long()` trả về INT32, không phải INT64 — FIXED

- Hành vi trước bản sửa (ĐÃ XÁC MINH, nay là lịch sử): binding ánh xạ `long` tới `Tensor::to_int` (INT32). Phương thức C++ đúng `Tensor::to_long()` (INT64) đã tồn tại (`src/tensor/tensor_core.cpp:2141`) nhưng không được bind dưới tên nào. `.int()` -> INT32 là đúng.
- Hành vi sau bản sửa (ĐÃ XÁC MINH): `.def("long", &Tensor::to_long)` (`src/bindings/bind_tpu.cpp:155`) — `.long()` nay trả về INT64, khớp với torch. Bản sửa: P-CORE-D06-long-binding REVISED, QA APPROVE qa-r62 2026-10-26 ~23:10, MERGED agent-v1 @89afd67. Bằng chứng kiểm thử: test mới 12/12 PASS + toàn bộ `tests/test_parity_tensor.py` 24 passed trên bản build đã vá; kiểm chứng âm tính trên extension chưa vá: đúng 4 FAIL, 2**31 tràn thành -2147483648.0.
- PyTorch: `.long()` là INT64 (thăm dò: `torch.tensor([1.5, -2.5, 3.0]).long()` -> `torch.int64`).
- Tác động: ĐÃ GIẢI QUYẾT. Dòng chuyển đổi cũ ("dùng `.cast(...)` sang INT64 cho tới khi binding được sửa") đã LỖI THỜI và không được xuất hiện ở bất cứ đâu trong tài liệu này.
- Chuyển đổi (hiện tại): không cần — `.long()` chạy như trong torch. LƯU Ý tương tác với D-14 (vẫn OPEN, bản vá này KHÔNG sửa): `.numpy()` từ chối INT64, nên `.long().numpy()` báo lỗi "unsupported dtype for copy conversion" — dùng `.to_vector()` hoặc ép kiểu trước.
- Kiểm tra hồi quy: L-05 (đầu ra `.long()` nay là INT64 khi dùng làm chỉ mục trong `gather` / `index_select`).

## D-07 — Các op linalg nâng mọi thứ lên FP32 — FIXED

- Hành vi trước bản sửa (ĐÃ XÁC MINH, nay là lịch sử): `promote_dtype(a, b)` trong `src/ops/linalg.cpp:41-44` trả về FP64 nếu một trong hai đầu vào là FP64, còn không thì LUÔN trả về FP32 — kể cả với đầu vào số nguyên. Các op bị ảnh hưởng: `solve`, `lstsq`, `cross`, `vecdot`, `outer`, `inner`, `kron`, `vdot`.
- Hành vi sau bản sửa (ĐÃ XÁC MINH): `promote_dtype` nay ủy quyền cho `promote_types` trung tâm (theo torch). `solve`/`lstsq`/`vecdot` trên đầu vào int/bool/FP16/hỗn hợp nay BÁO LỖI thay vì lặng lẽ trả về FP32; `cross`/`inner`/`vdot` giữ nguyên kiểu dữ liệu; `outer`/`kron` dùng nâng kiểu `result_type` đầy đủ; rank của `lstsq` nay là INT64 (trước đây là FP32); các đường số nguyên là chính xác trong int64/uint64 với tràn đã định nghĩa và không qua vòng double. Bản sửa: P-CORE-D07-linalg-promotion, QA APPROVE 2026-10-26 ~21:48, MERGED agent-v1 @7b8981e. Bằng chứng kiểm thử: 62 CHECK, 38 lỗi nhạy cảm với nâng kiểu trước bản sửa, TẤT CẢ PASS sau bản sửa; quét hồi quy 16 binary exit 0.
- PyTorch: `outer`, `vdot`, `kron` trên đầu vào int64 trả về int64 (thăm dò). Hành vi đã sửa khớp với torch.
- Tác động: ĐÃ GIẢI QUYẾT cho 8 op này. Dòng chuyển đổi cũ ("ép kiểu kết quả về lại kiểu dữ liệu số nguyên mong muốn; đừng dựa vào kiểu dữ liệu đầu ra của linalg") đã LỖI THỜI và không được xuất hiện ở bất cứ đâu trong tài liệu này.
- Chuyển đổi (hiện tại): ép kiểu đầu vào int/bool/FP16/hỗn hợp sang kiểu float trước khi gọi `solve`/`lstsq`/`vecdot`, hoặc chấp nhận lỗi; `outer`/`kron`/`inner`/`vdot`/`cross` nay theo nâng kiểu của torch — không cần ép kiểu.
- Đính chính một báo cáo trước: khẳng định "uint8+int64 nâng lên FP32" là CỤC BỘ, không phải toàn cục. Bảng nâng kiểu elementwise `promote_types` (`src/tensor/tensor_core.cpp:555`) là ĐÚNG: uint8+int64 -> INT64, khớp với torch. Việc nâng lên FP32 chỉ xảy ra trong họ linalg.
- Kiểm tra hồi quy: L-02 (chênh lệch số tối đa so với torch cho 8 op này, kỳ vọng parity).

## D-08 — Tensor số nguyên + vô hướng int trong Python nâng lên FP32 — OPEN

- LiteTorch (ĐÃ XÁC MINH): các overload vô hướng của `__add__` / `__sub__` / `__mul__` hiện thực vô hướng dưới dạng tensor FP32 1 phần tử (`src/bindings/bind_tpu.cpp`: `Tensor::from_vector({(float)b}, {1}, ...)`), nên nâng kiểu cho ra FP32.
- PyTorch: `int64_tensor + 2` vẫn là int64; `int64_tensor + 2.5` thành float32 (thăm dò).
- Tác động: TRUNG BÌNH. Thay đổi kiểu dữ liệu lặng lẽ trên cách viết số học vô hướng phổ biến nhất; giá trị int64 trên 2^24 bị làm tròn qua FP32.
- Chuyển đổi: tạo tensor vô hướng int64 tường minh cho phép số học vô hướng nguyên; hoặc ép kiểu kết quả về lại kiểu dữ liệu đầu vào.

## D-09 — GELU chỉ là xấp xỉ tanh; không có đối số `approximate` — DOCUMENTED

- LiteTorch (ĐÃ XÁC MINH): `Ops::gelu(a)` không có tham số `approximate` (`include/litetorch/ops.h:79`). Forward và backward đều dùng xấp xỉ tanh `0.5x(1+tanh(0.79788456(x+0.044715x^3)))` (`src/ops/activation.cpp:753-790`; backward tại `activation.cpp:419-420`). Không có `erf` trong đường GELU.
- PyTorch: GELU mặc định là dạng erf chính xác; `approximate='tanh'` là tùy chọn (thăm dò: max |exact - tanh| = 4.4e-4 trên linspace(-3, 3, 13); đo lại trực tiếp trong đợt này: 4.13e-04, và |lt - torch tanh| = 0.0 chính xác).
- Tác động: THẤP-TRUNG BÌNH. Số học khác torch mặc định tối đa ~4.4e-4 tuyệt đối; kiểm thử parity phải dùng dung sai ít nhất ~5e-4 cho GELU, hoặc so sánh với `F.gelu(x, approximate='tanh')`.
- Chuyển đổi: không có — ghi rõ dung sai; đừng khẳng định parity erf chính xác.
- Xem ví dụ trực tiếp `bd-d09-1` ở trên.

## D-10 — Op nhị phân giữa các thiết bị lặng lẽ chuyển dữ liệu thay vì báo lỗi — OPEN

- LiteTorch (ĐÃ XÁC MINH): `add`/`sub`/`mul`/`div` di chuyển toán hạng lệch thiết bị sang thiết bị của toán hạng còn lại (`src/ops/elementwise.cpp:583, 723, 874, 1015`); cùng mẫu trong các phép so sánh (`elementwise.cpp:2928, 2953`), matmul/linear (`src/ops/linear.cpp:142, 396, 474`), và các op shape (`src/ops/shape_ops.cpp:75`).
- PyTorch: `cpu_tensor + cuda_tensor` báo lỗi "Expected all tensors to be on the same device".
- Tác động: TRUNG BÌNH. Che giấu lỗi đặt thiết bị; một tensor CPU lạc lõng lặng lẽ kích hoạt chuyển thiết bị (và, kết hợp với D-02, có thể lặng lẽ ghim toàn bộ phép tính về CPU).
- Chuyển đổi: di chuyển tensor một cách tường minh trước khi trộn thiết bị; coi mọi chuyển ngầm là lỗi cần tìm, không phải tính năng.
- Kiểm tra trực tiếp còn cần: L-01 (toán hạng nào di chuyển trong từng họ op, và hành vi khi phía GPU là tensor CPU rơi về — Kaggle GPU).

## D-11 — Chỉ mục: chỉ int / slice trên dim 0; không có boolean, tensor, hay đa chiều — OPEN

- LiteTorch (ĐÃ XÁC MINH): `Tensor.__getitem__` chỉ xử lý `int` và `slice` trên chiều 0 và báo lỗi `TypeError("unsupported index type")` với mọi thứ khác (`src/bindings/bind_tpu.cpp`: lambda `__getitem__`).
- PyTorch: mặt nạ boolean, chỉ mục tensor số nguyên, và tuple đa chiều đều chạy (thăm dò: `t[torch.tensor([True, False, True])] == [1, 3]`).
- Tác động: TRUNG BÌNH. Các cách viết chuyển đổi phổ biến (`x[mask]`, `x[:, 1]`, `x[idx]`) thất bại.
- Chuyển đổi: dùng `Ops.gather`, `Tensor.index_select`, `Tensor.slice` (xem tham chiếu API).

## D-12 — `Tensor.sum/mean/max` chỉ nhận một dim int; không có tuple dim — OPEN

- LiteTorch (ĐÃ XÁC MINH): `Tensor.sum(dim, keepdim)` ép `dim` thành một `int64_t` duy nhất (`src/bindings/bind_tpu.cpp`). `Ops.sum/mean/max` CÓ chấp nhận `vector<int64_t>` dim (`src/bindings/bind_ops.cpp:121-125`).
- PyTorch: `t.sum(dim=(0, 1))` chạy (thăm dò).
- Tác động: THẤP.
- Chuyển đổi: dùng `litetorch.Ops.sum(t, [0, 1])`.

## D-13 — Không có phương thức `.bool()` / `.byte()` / `.char()` / `.short()` / `.to_long()` — OPEN

- LiteTorch (ĐÃ XÁC MINH bằng sự vắng mặt): không phương thức nào trong số này được bind trên Tensor (`src/bindings/bind_tpu.cpp` chỉ có `half`, `float`, `double`, `long`->INT64 (đã sửa, D-06), `int`, `type_as`, `cast`). `.cast(dtype)` tồn tại và là đường chung.
- PyTorch: tất cả đều tồn tại (thăm dò).
- Tác động: THẤP. Cơ học.
- Chuyển đổi: dùng `.cast(litetorch.DataType.<X>)`.

## D-14 — `.numpy()` từ chối UINT8 / INT64 / BOOL; BF16 lặng lẽ ép sang FP32 — OPEN

- LiteTorch (ĐÃ XÁC MINH): switch chuyển đổi `numpy()` chỉ xử lý FP64/FP32/FP16/INT32/INT16/INT8; UINT8, INT64, BOOL rơi vào default và báo lỗi "unsupported dtype for copy conversion" (`src/bindings/bind_tpu.cpp`: lambda `numpy`). BF16 được ép sang FP32 trước khi chuyển đổi. Tensor không phải CPU sẽ báo lỗi (phải `.cpu()` trước).
- PyTorch: `.numpy()` hỗ trợ trực tiếp uint8 và int64 (thăm dò).
- Tác động: THẤP-TRUNG BÌNH. Các đường xuất dữ liệu thất bại trên các kiểu dữ liệu số nguyên phổ biến nhất.
- Chuyển đổi: `.cast(FP32)` trước (lưu ý mất chính xác với int64 trên 2^24); hoặc `.tolist()`.

## D-15 — `arange` chỉ nhận số nguyên; giá trị được hiện thực dưới dạng FP32 — OPEN

- LiteTorch (ĐÃ XÁC MINH): `Tensor::arange(int64_t start, int64_t end, int64_t step, ...)` tạo `vector<float>` rồi ép sang kiểu dữ liệu yêu cầu (`src/tensor/tensor_core.cpp:696-708`). Không có start/end/step dạng float.
- PyTorch: `torch.arange(0, 5, 0.5)` trả về float32 (thăm dò).
- Tác động: THẤP.
- Chuyển đổi: tạo khoảng float qua `from_vector` / các hàm tương đương `linspace` (xem tham chiếu API).

## D-16 — `clamp` chỉ nhận biên vô hướng — OPEN

- LiteTorch (ĐÃ XÁC MINH): `clamp(a, float min_val, float max_val)`, `clamp_min/max(a, float)` (`include/litetorch/ops.h:75, 201-202`).
- PyTorch: `torch.clamp` chấp nhận min/max dạng tensor (thăm dò).
- Tác động: THẤP.
- Chuyển đổi: dùng `Ops.where` / `Ops.minimum` tương đương khi cần biên dạng tensor.

## D-17 — Op một ngôi / op shape chỉ tồn tại dưới dạng hàm `Ops.*`, không phải phương thức Tensor — OPEN

- LiteTorch (ĐÃ XÁC MINH bằng sự vắng mặt): Tensor không có phương thức `.exp()`, `.log()`, `.sqrt()`, `.relu()`, `.sigmoid()`, `.tanh()`, `.abs()`, `.squeeze()`, `.unsqueeze()` (chỉ `log_softmax` và `silu` được bind làm phương thức). Tất cả tồn tại dưới dạng `litetorch.Ops.*` (`src/bindings/bind_ops.cpp:20-32, 46-52`).
- PyTorch: tất cả đều tồn tại dưới dạng phương thức Tensor.
- Tác động: THẤP. Ma sát chuyển đổi cơ học.
- Chuyển đổi: `litetorch.Ops.relu(t)` v.v.

## D-18 — `.item()` và `.tolist()` / `.to_vector()` trả về float; mất chính xác int64 — OPEN

- LiteTorch (ĐÃ XÁC MINH): `Tensor::item()` trả về `float`, ép tensor không phải FP32 sang FP32 trước (`src/tensor/tensor_core.cpp:1598-1608`). `to_vector()` cũng ép sang FP32; `tolist()` tạo từ các double đó và chỉ coi INT32/INT16/INT8 là số nguyên (`src/bindings/bind_tpu.cpp`).
- PyTorch: `.item()` trên tensor int64 trả về Python `int` với đầy đủ chính xác (thăm dò: `2**60` đi về nguyên vẹn).
- Tác động: THẤP-TRUNG BÌNH. Giá trị int64 trên 2^53 lặng lẽ mất chính xác qua `.item()` / `.tolist()`.
- Chuyển đổi: tránh `.item()` với giá trị int64 lớn; đọc qua `.numpy()` (sau khi ép kiểu, xem D-14) hoặc giữ giá trị ở dạng tensor.
- Kiểm tra trực tiếp: L-02 (ranh giới `.item()` int64 tại 2^53).

## D-19 — Kiểu dữ liệu lượng tử nâng lên FP32 — OPEN (TREE-ONLY)

- LiteTorch (TREE-ONLY): `promote_types` trả về FP32 nếu một trong hai toán hạng là kiểu dữ liệu lượng tử (`src/tensor/tensor_core.cpp:555-557`).
- PyTorch: nâng kiểu lượng tử theo quy tắc lượng tử của torch; tương ứng chính xác CHƯA XÁC MINH — cần thăm dò trực tiếp song song.
- Tác động: CHƯA RÕ cho tới khi thăm dò. Được ghi chú để chương lượng tử không lặng lẽ khẳng định parity với torch.
- Chuyển đổi: chưa có.
- Kiểm tra trực tiếp: L-03 (nâng kiểu lượng tử D-19 so với ngữ nghĩa torch).

## D-20 — Không có `F.scaled_dot_product_attention`; chỉ có `Ops.flash_attention` — OPEN

- LiteTorch (ĐÃ XÁC MINH): không có SDPA ở bất cứ đâu trong bindings. Chỉ có `Ops.flash_attention(q, k, v, causal=False)` và `flash_attention_qkv(qkv, num_heads, causal=False)` (`src/bindings/bind_ops.cpp:143-144`). Không có đối số `scale` hay `dropout_p`; chiều head phải thỏa 1 <= D <= 256 và H % H_kv == 0 (`src/backend/gpu_native/nn/flash_attention.cu:167-168`). `flash_attention_half` trên TPU không được triển khai và sẽ báo lỗi (`src/backend/tpu/tpu_backend.cpp:343`).
- PyTorch: `F.scaled_dot_product_attention` tồn tại với `is_causal`, `dropout_p`, `scale` (thăm dò).
- Tác động: TRUNG BÌNH. Mã attention chuyển đổi phải viết lại; scale phải áp lên Q bằng tay; attention FP16 trên TPU không khả dụng.
- Chuyển đổi: `litetorch.Ops.flash_attention(q * scale, k, v)`; kiểm tra ràng buộc chiều head trước khi chuyển đổi.
- Kiểm tra trực tiếp: L-04 (xác nhận lỗi `flash_attention_half` trên TPU).

## D-21 — Toán elementwise số nguyên trên GPU chạy qua FP32 — OPEN (TREE-ONLY, số học)

- LiteTorch (TREE-ONLY): đầu vào elementwise không phải FP32 trên GPU được ép sang FP32, tính toán, rồi ép về lại kiểu dữ liệu đã nâng (`src/ops/elementwise.cpp` đường `add`, ~dòng 571-590; cùng mẫu trong sub/mul/div). Kiểu dữ liệu kết quả là đúng, nhưng số học int64 được thực hiện trong FP32.
- PyTorch: toán elementwise int64 trên CUDA là toán số nguyên chính xác.
- Tác động: TRUNG BÌNH trên GPU. Giá trị int64 trên 2^24 mất chính xác; tràn số nguyên trong torch thì quấn vòng, còn trong đường FP32 thì bão hòa/làm tròn. Đường số nguyên trên CPU (`ew_add_int`) là chính xác.
- Kiểm tra trực tiếp: L-01 (sai số tối đa đo được cho cộng/nhân int64 trên Kaggle GPU so với torch).

## D-22 — `__call__` đa đối số chỉ đi qua forward một tensor — FILED

- LiteTorch (ĐÃ XÁC MINH): `__call__()` chỉ đi qua forward một tensor. `m(q, k, v)`, `m(x1, x2)` và mọi hàm loss 2+ đối số đều báo lỗi TypeError; phải dùng `.forward(...)`. Ảnh hưởng MultiHeadAttention, Bilinear, TripletMarginWithDistanceLoss, GaussianNLLLoss, CTCLoss, DPOLoss, mọi hàm loss 2 đối số.
- PyTorch: module có thể gọi được với đầy đủ chữ ký forward.
- Tác động: TRUNG BÌNH. Mã chuyển đổi gọi trực tiếp module với nhiều đối số sẽ thất bại ngay tại chỗ gọi.
- Chuyển đổi: gọi `.forward(...)` một cách tường minh cho module đa đối số.
- Trạng thái sửa: FILED cho Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-23 — Target của hàm loss phải là INT32; INT64 bị từ chối — FILED

- LiteTorch (ĐÃ XÁC MINH): CrossEntropyLoss, NLLLoss, `F.nll_loss` từ chối target INT64 ("cross_entropy_loss: target must have integer or float dtype") và yêu cầu INT32. Nguyên nhân gốc: `ce_target_indices` trong `src/ops/loss.cpp:24-40` không có case INT64; tương tự trong `fused_loss.cpp:25`. Cũng ảnh hưởng `Ops.cross_entropy_loss` / `Ops.nll_loss` / `Ops.fused_linear_cross_entropy` và `nn::CrossEntropyLoss`.
- PyTorch: target của hàm loss là int64 (thăm dò).
- Tác động: CAO. Sau bản sửa D-06, `.long()` trả về INT64 — và các hàm loss này từ chối nó. Hai bản sửa tương tác: kiểu dữ liệu đúng cho target loss trong LiteTorch hiện nay là INT32, ngược với torch.
- Chuyển đổi: dùng target INT32 một cách tường minh (`.cast(litetorch.DataType.INT32)`); đừng truyền kết quả `.long()` làm target loss.
- Trạng thái sửa: FILED cho Core, comms `20261026-234500-docs-to-core-ceint64.md` (thread `docs-fix-requests-20261026`).

## D-24 — Gán submodule dưới dạng thuộc tính Python KHÔNG đăng ký nó — FILED

- LiteTorch (ĐÃ XÁC MINH): gán submodule dưới dạng thuộc tính Python (`self.linear = lt.nn.Linear(4, 4)`) KHÔNG đăng ký nó — `parameters()` trả về 0 và `state_dict()` trống. `add_module('linear', module)` CÓ đăng ký (`parameters() == 2`).
- PyTorch: gán thuộc tính tự động đăng ký submodule.
- Tác động: CAO. Mọi subclass `nn.Module` chuyển đổi gán lớp dưới dạng thuộc tính lặng lẽ không có tham số nào; optimizer sau đó bước qua khoảng trống và huấn luyện không học được gì.
- Chuyển đổi: đăng ký các module con một cách tường minh bằng `add_module(name, module)`; hoặc ghi đè `parameters()` để thu thập submodule bằng tay (tutorial T1 `MultiLayerPerceptron` làm vậy). Các mục `ModuleDict` đi qua `m[key] = module` / `put()`, KHÔNG phải `add_module()` (nó đăng ký module con nhưng không phải mục dict — `getitem` sau đó báo lỗi KeyError); `ModuleDict` không có phương thức `values()`/`items()` được bind.
- Trạng thái sửa: FILED cho Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-25 — Tạo `Parameter(tensor)` bị HỎNG — FILED

- LiteTorch (ĐÃ XÁC MINH trên 88a4ba1): `Parameter(tensor)` luôn báo lỗi "copy_: a leaf Tensor that requires grad is being used in an in-place operation", ngay cả với đầu vào requires_grad=False. Binding `copy_()` vào chính Parameter.
- PyTorch: `nn.Parameter(tensor)` chạy được.
- Tác động: CAO. Không có cách khắc phục; lấy tham số từ module (ví dụ `Linear.weight`) thay vì tạo `Parameter` trực tiếp.
- Chuyển đổi: không có — đừng tạo `Parameter` trực tiếp; dùng tham số của module.
- Trạng thái sửa: FILED cho Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-26 — Công thức HingeEmbeddingLoss khác — FILED

- LiteTorch (ĐÃ XÁC MINH): tính mean(max(0, margin - target*input)) cho mọi phần tử (1.1750 trên ví dụ tham chiếu).
- PyTorch: tính input với target==1, max(0, margin-input) với target==-1 (0.7000 trên cùng ví dụ).
- Tác động: TRUNG BÌNH. Cùng đầu vào, giá trị loss khác nhau; quỹ đạo huấn luyện lệch nhau.
- Chuyển đổi: không có — đừng so sánh giá trị HingeEmbeddingLoss giữa các framework; nếu cần ngữ nghĩa torch chính xác, hãy triển khai hàm loss bằng các primitive `Ops`.
- Trạng thái sửa: FILED cho Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-27 — Chuẩn hóa MultiMarginLoss khác; `p` là INT — FILED

- LiteTorch (ĐÃ XÁC MINH): trường hợp một mẫu toàn 0: LiteTorch 1.0000 so với torch 0.7500 (torch chia cho #classes, LiteTorch chia cho #classes-1); ví dụ hai mẫu 0.9667 so với 0.6000 (p=1); p=2: 1.1700 so với 0.7525. Thêm nữa: `p` là INT trong binding này (`p=2.0` báo lỗi TypeError).
- PyTorch: `p` là float; chuẩn hóa chia cho #classes.
- Tác động: TRUNG BÌNH. Giá trị loss khác nhau; `p=2.0` (cách viết mặc định của torch) báo lỗi.
- Chuyển đổi: truyền `p` dưới dạng int (`p=2`); đừng so sánh giá trị MultiMarginLoss giữa các framework.
- Trạng thái sửa: FILED cho Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-28 — Giảm trung bình CTCLoss khác — FILED

- LiteTorch (ĐÃ XÁC MINH): trung bình mặc định = trung bình trên batch của loss từng mẫu (4.0802 trên ví dụ tham chiếu).
- PyTorch: 'mean' còn chia mỗi mẫu cho độ dài target của nó (1.3601 trên cùng ví dụ). LiteTorch reduction='sum' khớp chính xác với torch 'sum' (8.1604).
- Tác động: TRUNG BÌNH. Loss CTC giảm trung bình khác nhau theo hệ số độ dài target.
- Chuyển đổi: dùng reduction='sum' để có parity chính xác, và tự chia cho tổng độ dài target nếu cần ngữ nghĩa mean của torch.
- Trạng thái sửa: FILED cho Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-29 — Đầu ra TransformerDecoderLayer khác torch — FILED

- LiteTorch (ĐÃ XÁC MINH): với trọng số giống hệt sao chép từ LiteTorch, forward hai đối số khác torch (chênh lệch tối đa 0.12–0.14, chế độ eval). Nguyên nhân chưa cô lập (có thể là vị trí norm / chi tiết cross-attention). Parity của TransformerEncoderLayer chỉ đúng ở chế độ eval (4.2e-07); ở chế độ train, dropout 0.1 đang bật, nên khẳng định parity cho lớp encoder chỉ đúng ở chế độ eval.
- PyTorch: hành vi tham chiếu.
- Tác động: CAO với bất cứ ai chuyển đổi transformer decoder. Đầu ra decoder không khớp torch ngay cả với trọng số giống hệt.
- Chuyển đổi: không có — đừng kỳ vọng parity decoder; kiểm tra đầu ra decoder với torch trước khi tin tưởng.
- Trạng thái sửa: FILED cho Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-30 — Bước đầu tiên của NAdam lớn hơn ~6.5 lần — FILED

- LiteTorch (ĐÃ XÁC MINH): bước cập nhật đầu tiên lớn hơn ~6.5 lần. Nguyên nhân gốc trong `src/optim/nadam.cpp:49`: `mu_product = mu_next / mu` (một tỉ số) + `step_size = lr / bias_correction1`; torch giữ `mu_product` là tích lũy thừa từ 1 và dùng hai hạng cập nhật với mẫu số `(1 - mu_product)`. Đo 3 bước: lt `[0.9350, 0.8961, 0.8658]` so với torch `[0.9894, 0.9816, 0.9743]`.
- PyTorch: `torch.optim.NAdam` — các giá trị tham chiếu ở trên.
- Tác động: CAO. Huấn luyện NAdam trong LiteTorch lệch khỏi torch ngay lập tức; đừng dùng NAdam cho các lần huấn luyện chuyển đổi.
- Chuyển đổi: dùng Adam hoặc AdamW thay thế; hoặc tự triển khai bước cập nhật NAdam.
- Trạng thái sửa: FILED cho Core, comms `20261027-000400-docs-to-core-nadam-onecycle-negbin.md` (thread `docs-fix-requests-20261026`).

## D-31 — OneCycleLR lệch pha một bước — FILED

- LiteTorch (ĐÃ XÁC MINH): từ lần gọi thứ 3 trở đi, lt == lần gọi trước của torch; đoạn tăng mất 3 lần gọi mới đạt max so với 2 của torch; lần gọi cuối lt rơi về sàn (4e-07) trong khi torch phát lại giá trị trước. Nguồn scheduler: `src/optim/scheduler.cpp` (binding `src/bindings/bind_optim.cpp`).
- PyTorch: `torch.optim.lr_scheduler.OneCycleLR` — lịch tham chiếu ở trên.
- Tác động: TRUNG BÌNH. Lịch tốc độ học lệch một bước; các lần chạy ngắn thấy lịch khác rõ rệt.
- Chuyển đổi: không có — tính đến lệch một bước khi so sánh lịch, hoặc dùng CosineAnnealingLR / StepLR (cả hai đã xác minh khớp chính xác với torch).
- Trạng thái sửa: FILED cho Core, comms `20261027-000400-docs-to-core-nadam-onecycle-negbin.md` (thread `docs-fix-requests-20261026`).

## D-32 — `probs` của NegativeBinomial là xác suất thất bại — FILED

- LiteTorch (ĐÃ XÁC MINH): coi `probs` là xác suất thất bại; `NB(r, p)` trong lt == `NB(r, 1-p)` trong torch một cách chính xác. Trung bình 11.667 so với torch 2.143 (r=5, p=0.3). Binomial/Geometric KHÔNG bị ảnh hưởng.
- PyTorch: `probs` là xác suất thành công (thăm dò).
- Tác động: TRUNG BÌNH. Mẫu và trung bình của phân phối bị hoán đổi so với torch.
- Chuyển đổi: truyền `1 - p` để có ngữ nghĩa torch.
- Trạng thái sửa: FILED cho Core, comms `20261027-000400-docs-to-core-nadam-onecycle-negbin.md` (thread `docs-fix-requests-20261026`).

## D-33 — Backward của DDP nhân đôi một lát của gradient trọng số — FILED

- LiteTorch (ĐÃ XÁC MINH): `DistributedDataParallel.backward` nhân đôi một lát của gradient trọng số phẳng: các chỉ mục `[numel/4, numel/4 + out_features)` (tương đương `[numel/4, numel/4 + bias_numel)`; đã chốt 2026-10-27 theo lần chạy lại quyết định của dist-w21 — một bản sửa cùng ngày khẳng định in_features đã bị rút lại: lần kiểm tra lại B4 chỉ chạy lại Linear(4,4) đối xứng, và các hàng bất đối xứng trong bảng shape B3 đã bị hoán đổi). Đã xác minh trên 4 shape — Linear(4,4): chỉ mục 4–7; Linear(4,8): 8–15; Linear(8,4): 8–11; Linear(2,2): 1–2 — trong cả hai chế độ giảm `AVG` (hàng nhân đôi 1.0) và `SUM` (hàng nhân đôi 2.0). Gradient bias là đúng. Backward `Linear` đơn tiến trình là đúng (tất cả 0.5). Parity so với torch: max_abs_diff 0.5 trên cả hai rank.
- PyTorch: gradient DDP khớp gradient đơn tiến trình.
- Tác động: CAO. Đừng huấn luyện với DDP cho tới khi sửa xong — trọng số nhận một lát gradient nhân đôi mỗi bước.
- Chuyển đổi: không có — dùng huấn luyện đơn tiến trình, hoặc FSDP (`fsdp_fully_shard`) thay cho DDP.
- Trạng thái sửa: FILED cho Core, comms `20261027-000800-docs-to-core-ddp-grad.md` (thread `docs-fix-requests-20261026`).

## D-34 — `gradcheck` mặc định `eps=1e-6` thất bại trên FP32 — DOCUMENTED

- LiteTorch (ĐÃ XÁC MINH): `lt.autograd.gradcheck` với `eps=1e-6` mặc định thất bại ngay cả với `t*t` (đạo hàm giải tích 6 so với đạo hàm số 5.72 tại x=3): 1e-6 nằm dưới độ phân giải FP32. `eps=1e-3` vượt qua gradcheck + gradgradcheck. Binding mặc định: `src/bindings/bind_autograd.cpp:79` (`eps` = 1e-6).
- PyTorch: `torch.autograd.gradcheck` cũng mặc định `eps=1e-6`, nhưng kiểu dữ liệu đầu vào mặc định của gradcheck trong torch là float64 — cùng eps tuyệt đối đó là có ý nghĩa ở đó.
- Tác động: THẤP. Ảnh hưởng mã kiểm tra, không ảnh hưởng huấn luyện.
- Chuyển đổi: luôn truyền `eps=1e-3` một cách tường minh cho gradcheck FP32 trong LiteTorch.
- Xem `troubleshooting.md` ("gradcheck thất bại trên một hàm đơn giản") để có minh họa chạy được.

## D-35 — RendezvousStore abort khi tắt nếu thread `run()` không được join — OPEN

- LiteTorch (ĐÃ XÁC MINH): nếu không join thread `run()` sau `stop()`, tiến trình abort khi thoát trình thông dịch (`terminate called without an active exception`, SIGABRT; không ổn định — lần chạy EN thất bại, lần chạy VI vượt qua). Join thread sẽ sửa được (3/3 sạch). Nguồn: `src/distributed/rendezvous.cpp:113-143` (`RendezvousStore::stop`, destructor).
- PyTorch: không có thành phần tương đương (cơ chế rendezvous của `torch.distributed` khác).
- Tác động: TRUNG BÌNH. SIGABRT không ổn định khi thoát trong mã launcher phân tán.
- Chuyển đổi: luôn `join()` thread `run()` sau `stop()` trước khi thoát.

## D-36 — AsyncCheckpointSaver thỉnh thoảng treo khi tắt — FILED

- LiteTorch (ĐÃ XÁC MINH): 2 lần treo trong ~46 lần chạy (>60 s sau khi `wait()` trả về, phải kill; 0/26 trong các lần chạy lại, không bắt được stack). Nghi ngờ race khi tháo dỡ thread nền. Nguồn: `src/serialization/async_checkpoint.cpp`.
- PyTorch: không có thành phần tương đương.
- Tác động: TRUNG BÌNH. Treo không ổn định khi thoát trong mã checkpoint.
- Chuyển đổi: `del saver` một cách tường minh trước khi thoát; đừng dựa vào việc tháo dỡ trình thông dịch để dừng thread nền.
- Trạng thái sửa: đã báo cho Core dưới dạng ghi chú kèm trong comms `20261027-000400-docs-to-core-nadam-onecycle-negbin.md` (thread `docs-fix-requests-20261026`).

## D-37 — `Dataset` trong Python phải định nghĩa `size`/`get`, không phải `__len__`/`__getitem__`; và loader abort khi tắt — FILED

- LiteTorch (ĐÃ XÁC MINH): subclass `Dataset` định nghĩa `__len__`/`__getitem__` không bao giờ được gọi. Trampoline `PyDataset` (`src/bindings/bindings_common.h:173`) điều phối `size()`/`get()` qua `PYBIND11_OVERRIDE_PURE` tới các phương thức Python tên `size` và `get` — không phải `__len__`/`__getitem__`. `DataLoader` sau đó báo lỗi `RuntimeError: Tried to call pure virtual function "data::Dataset::size"`. Khi `size`/`get` tồn tại, `len(dataset)` / `dataset[i]` chạy được (chúng được bind tới cùng phương thức C++). Sau khi sửa điều đó, ví dụ cho ra kết quả đúng nhưng tiến trình chết với `FATAL: exception not rethrown` (SIGABRT, exit -6) khi tắt — `DataLoader` C++ giữ dataset backed bởi Python quá thời điểm tháo dỡ. Cách khắc phục: `del loader` / `del dataset` tường minh trước khi thoát (đã xác minh thoát sạch exit 0). Một `collate_fn` Python kích hoạt cùng lỗi abort khi tắt.
- PyTorch: subclass `Dataset` định nghĩa `__len__`/`__getitem__`; không abort khi tắt.
- Tác động: CAO. Mọi dataset tùy chỉnh chuyển đổi thất bại hai lần: lần đầu khi tạo, lần sau khi thoát.
- Chuyển đổi: định nghĩa `size()` và `get(idx)` đúng như cách viết; `del loader` / `del dataset` tường minh trước khi thoát.
- Trạng thái sửa: FILED cho Core, comms `20261026-233500-docs-to-core-h7d2.md` (thread `docs-fix-requests-20261026`).

## D-38 — `ring_attention` không thể gọi từ Python — FILED

- LiteTorch (ĐÃ XÁC MINH): `ring_attention` không thể gọi từ Python (holder tham chiếu `ProcessGroup.get()` so với tham số `shared_ptr`; `Unable to load a custom holder type`). Binding: `src/bindings/bind_distributed.cpp`.
- PyTorch: `torch.distributed` không có `ring_attention`; tương đương là `torch.nn.attention.flex_attention` / song song dãy bằng tay.
- Tác động: TRUNG BÌNH. Đường ring-attention đã ghi trong tài liệu thực tế không tồn tại; dùng sharding dãy bằng tay qua `all_gather`/`reduce_scatter` (xem H2.4).
- Chuyển đổi: dùng cách dự phòng sharding bằng tay trong H2.4.
- Trạng thái sửa: FILED cho Distributed, comms `20261026-234600-docs-to-distributed-ring.md` (thread `docs-d9-fsdp-20261026`).

## D-39 — Forward của VocabParallelEmbedding báo lỗi `data_ptr()` không phải FP32 — FILED (hệ quả của D-01)

- LiteTorch (ĐÃ XÁC MINH trên 88a4ba1): tạo `VocabParallelEmbedding` chạy được, nhưng `forward()` báo lỗi `data_ptr()` không phải FP32 với cả tensor chỉ mục INT64 và FP32 — tra cứu embedding chạm vào đường `data_ptr()` chỉ-FP32 của C++ (D-01).
- PyTorch: forward của `nn.Embedding` chạy được với chỉ mục int64.
- Tác động: CAO với việc dùng embedding song song tensor. Module này không dùng được cho forward trên bản build này.
- Chuyển đổi: không có — dùng `nn.Embedding` (đơn thiết bị) thay thế; ví dụ tham chiếu chỉ tạo module và ghi rõ lỗi.
- Trạng thái sửa: đã gửi cùng D-22–D-29 cho Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-40 — `linalg.cholesky_inverse` nhận ma trận, không phải thừa số Cholesky — OPEN

- LiteTorch (ĐÃ XÁC MINH): `cholesky_inverse(A)` nhận chính ma trận — `cholesky_inverse(A) == inv(A)`; truyền thừa số cho kết quả sai. Nguồn: `src/ops/linalg.cpp:642`.
- PyTorch: `torch.cholesky_inverse` nhận thừa số Cholesky.
- Tác động: TRUNG BÌNH. Truyền thừa số (cách viết torch) lặng lẽ trả về kết quả sai.
- Chuyển đổi: truyền ma trận gốc, không phải thừa số của nó.

## D-41 — `linalg.lstsq` trả về tensor 0-D cho residuals và rank — OPEN

- LiteTorch (ĐÃ XÁC MINH): `lstsq` trả về tensor 0-D cho residuals (`0.0`) và rank (`2.0`); torch trả về tensor rỗng và int. Nguồn: `src/ops/linalg.cpp:1025`.
- PyTorch: `torch.linalg.lstsq` trả về `residuals` (tensor rỗng) và `rank` (int).
- Tác động: THẤP. Không khớp shape/kiểu khi giải nén; giá trị là đúng.
- Chuyển đổi: đọc rank qua `.item()` / `.to_vector()[0]`; đừng lập chỉ mục `residuals[0]`.

## D-42 — `linalg.matrix_norm` bắt buộc `ord` là chuỗi — OPEN

- LiteTorch (ĐÃ XÁC MINH): `matrix_norm(A, ord=1)` báo lỗi TypeError; `ord` phải là chuỗi (`ord="1"`). Cả 8 ord chuỗi đã xác minh: fro/nuc/1/2/inf/-inf/-1/-2. Binding: `src/bindings/bind_linalg.cpp:45` (`ord` = "fro").
- PyTorch: `torch.linalg.matrix_norm(A, ord=1)` chấp nhận int.
- Tác động: THẤP. Cơ học.
- Chuyển đổi: truyền `ord` dưới dạng chuỗi.

## D-43 — `fft.ifft` / `ifftn` trả về tensor thực, không phải bố cục phức — OPEN

- LiteTorch (ĐÃ XÁC MINH): `ifft`/`ifftn` trả về tensor thực, không phải bố cục phức `[..., 2]` mà `fft` tạo ra. Binding: `src/bindings/bind_fft.cpp:8, 16`.
- PyTorch: `torch.fft.ifft` trả về tensor phức.
- Tác động: THẤP-TRUNG BÌNH. Mã kỳ vọng đầu ra phức từ `ifft` nhận được đầu ra thực.
- Chuyển đổi: coi đầu ra `ifft` là thực; đừng lập chỉ mục chiều cuối phức.

## D-44 — `Trainer.train_step()` không chạy forward/backward — OPEN

- LiteTorch (ĐÃ XÁC MINH): `Trainer.train_step()` KHÔNG chạy forward/backward (chỉ bước optimizer + zero_grad). Checkpoint của `Trainer` là async theo mặc định (tập tin chỉ xuất hiện sau `wait_pending_saves()`). Nguồn: `src/trainer/trainer.cpp`.
- PyTorch: `Trainer` không phải builtin của torch (đây là API kiểu `transformers`); phép so sánh là với hợp đồng LiteTorch đã ghi, mà phương thức này vi phạm.
- Tác động: CAO với bất cứ ai dùng `Trainer` làm vòng lặp huấn luyện. Gọi `train_step()` mà không forward/backward bằng tay trước đó sẽ bước optimizer trên gradient cũ (hoặc bằng 0).
- Chuyển đổi: chạy forward + backward bằng tay trước `train_step()`; gọi `wait_pending_saves()` trước khi đọc tập tin checkpoint.

## D-45 — `save` / `load` nhận Module/list/Optimizer, không phải state dict — OPEN

- LiteTorch (ĐÃ XÁC MINH): `lt.save`/`lt.load` là in-place và điều phối theo kiểu (tensor trần báo lỗi); `save_parameters` nhận list tham số, không phải module; `save_safetensors`/`save_async` nhận dict, không phải module; `save_optimizer_state` chỉ hỗ trợ Adam/AdamW/SGD/RMSprop và báo lỗi với mọi optimizer khác.
- PyTorch: `torch.save(state_dict, path)` nhận đối tượng tùy ý kể cả state dict.
- Tác động: TRUNG BÌNH. Mã checkpoint chuyển đổi thất bại ngay tại chỗ gọi.
- Chuyển đổi: `save_parameters(model.parameters())` cho trọng số; `save_safetensors(dict)` cho dict; dùng `state_dict()`/`load_state_dict(sd)` riêng cho chuyển mô hình đang chạy.

## D-46 — `DataLoader(shuffle=True)` KHÔNG tái lập được dưới `lt.manual_seed` — DOCUMENTED

- LiteTorch (ĐÃ XÁC MINH trong B1): `DataLoader(shuffle=True)` không tái lập được dù đã `lt.manual_seed(42)` — loss epoch 1 thay đổi 1.0306 / 0.9762 / 1.0724 giữa các lần chạy. Shuffle của DataLoader dùng RNG riêng. Các lớp data nằm dưới `lt.data.*` (không phải cấp cao nhất); sampler không thể lặp (`indices()`/`batches()`); `rank()`/`world_size()`/`is_distributed()` là phương thức.
- PyTorch: `torch.manual_seed` gieo hạt cho shuffle của DataLoader (với generator) một cách xác định.
- Tác động: TRUNG BÌNH với số liệu tài liệu tái lập được và thí nghiệm.
- Chuyển đổi: dùng `shuffle=False` cho các lần chạy tái lập (mọi ví dụ trong tài liệu đều vậy); gieo hạt sampler một cách tường minh nếu cần shuffle.

## D-47 — `.grad` của tensor không phải lá được điền sau `backward()` dù không gọi `retain_grad()` — FILED

- LiteTorch (ĐÃ XÁC MINH trong B4, E2): sau `backward()`, `.grad` của tensor không phải lá giữ gradient gốc ([1.0]). `retain_grad()` tồn tại nhưng không thay đổi gì quan sát được.
- PyTorch: `.grad` của tensor không phải lá giữ `None` (kèm cảnh báo) trừ khi đã gọi `retain_grad()`.
- Tác động: TRUNG BÌNH. Mã đọc grad của tensor không phải lá (ví dụ hook gỡ lỗi, phẫu thuật gradient) lặng lẽ nhận giá trị mà torch không cung cấp; mã kiểm tra `grad is None` để phát hiện tensor không giữ grad sẽ hỏng.
- Chuyển đổi: không có — trong LiteTorch hãy coi `.grad` của tensor không phải lá là đã được điền; đừng dựa vào `None` làm tín hiệu. Đã gửi Core (comms 20261027-012500-docs-to-core-autograd-div.md).

## D-48 — `Tensor.zero_grad()` đặt `.grad` thành `None`, không phải tensor 0 — FILED

- LiteTorch (ĐÃ XÁC MINH trong B4, E2): `zero_grad()` đặt `.grad` thành `None`. Lần `backward()` tiếp theo tạo lại nó, nên vòng huấn luyện vẫn chạy; mã đọc trực tiếp grad vừa được zero sẽ hỏng.
- PyTorch: `zero_grad()` để lại tensor 0 trong `.grad`.
- Tác động: THẤP. Chỉ hỏng mã đọc `.grad` ngay sau khi zero (ví dụ assert, logging).
- Chuyển đổi: kiểm tra `if t.grad is not None` trước khi đọc, hoặc gọi `backward()` trước khi đọc. Đã gửi Core (comms 20261027-012500-docs-to-core-autograd-div.md).

## D-49 — `backward()` không có tham số `retain_graph`, dù thông báo lỗi nhắc tới nó — FILED

- LiteTorch (ĐÃ XÁC MINH trong B4, E2): chữ ký `backward()` là `(gradient=None, create_graph=False)` — không có `retain_graph`. Lần backward thứ hai trên cùng đồ thị báo lỗi với thông báo "pass retain_graph=true to run backward multiple times" — một tham số không tồn tại.
- PyTorch: `backward(retain_graph=True)` cho phép backward lặp lại.
- Tác động: TRUNG BÌNH. Mã chuyển đổi nào chạy backward lại (ví dụ đa loss, tích lũy gradient qua cùng đồ thị) sẽ thất bại; thông báo lỗi hướng người dùng tới một cách sửa không tồn tại.
- Chuyển đổi: xây lại đồ thị (chạy lại forward) cho mỗi backward, hoặc dùng `create_graph=True` khi phù hợp. Đã gửi Core (comms 20261027-012500-docs-to-core-autograd-div.md).

## Các trường hợp biên đã xác minh (không đánh số — sự thật API, không phải khác biệt)

Các mục này đã được xác minh trực tiếp trong đợt tham chiếu B3 và được ghi trong tham chiếu API; chúng được gom ở đây để người chuyển đổi tìm thấy trong một chỗ.

- `fused_bias_relu` / `fused_bias_gelu` yêu cầu shape giống hệt cho x và bias (không broadcasting).
- `Tracer.trace` trả về `JITFunction`, không phải `JITVar`; `Tracer.trace` với trung gian `+` thuần báo lỗi intermediate đã hết hạn (dùng `lt.Ops.*`).
- `PreparedModel.calibrate` nhận callable không đối số `data_fn`, không phải list.
- `ltrec.negative_sampling` nhận tensor chỉ mục cạnh + num_nodes, không phải list item.
- `FlatParameter.param_views()` yêu cầu `all_gather_flat()` trước.
- `fuse_conv_bn` yêu cầu BN ở chế độ eval (giống `fold_conv_bn`).
- `DataType` không thể lặp từ Python; `FP4` là bí danh của `FP4_E2M1`.
- `ServingScheduler.get_block_table_tensor()` trả về `None` khi mọi request đã xong (block đã giải phóng) — chụp bảng trước bước cuối.
- `lt.autograd.gradcheck` xóa `x.grad` — tính gradient giải tích bằng backward tường minh sau gradcheck.
- `SyncBatchNorm` là BatchNorm theo kênh: đầu vào rank-2 (N, C) báo lỗi "channel BatchNorm: bad input rank or channels"; rank-3 (N, C, L) chạy được.
- Hàm tạo `TransformerDecoderLayer` nhận `(embed_dim, num_heads, dim_feedforward=2048)` — KHÔNG phải `(d_model, nhead)` như `TransformerEncoderLayer`.
- `F.dpo_loss` yêu cầu ref logps trừ khi `reference_free=True` (không thì ValueError).
- `lt.Device("cuda", rank)` không phải hàm tạo hợp lệ (hàm tạo chuỗi chỉ nhận một đối số); dùng `lt.Device("cuda:%d" % rank)`.
- Ví dụ đa tiến trình cần guard `if __name__ == "__main__"` khi spawn.
- Công thức `w8a8_matmul` là `(round(x / x_scale) @ round(w / w_scale).T) * x_scale * w_scale` với kẹp INT8 về [-128, 127] — không phải `(x @ w.T) * x_scale * w_scale`.
- `moe_gate` chuẩn hóa lại các xác suất top-k đã chọn để tổng bằng 1 (`/ (sum_top_k + 1e-9f)`).
- Ép kiểu trần NF4/FP4 qua `.cast()` không phải lượng tử hóa dùng được: vòng FP4 của `[0.1, -0.5, 1.2, -2.4]` là `[0.0, -0.5, 1.0, -2.0]`; NF4 là `[0.0596, -0.5251, 1.0, -1.0]` — NF4 kẹp về [-1, 1] mà không có scale theo tensor.
- Phương thức `FSDPUnit` không tới được từ Python (không có `py::init`, `active_units()` không được bind) — đừng ghi tài liệu hay gọi chúng; dùng `fsdp_fully_shard`. `destroy_process_group()` KHÔNG được bind — dùng `shutdown()`. FSDP shard tham số + gradient, KHÔNG phải trạng thái optimizer.
- Các ord chuỗi của `linalg.matrix_norm`: fro/nuc/1/2/inf/-inf/-1/-2 (xem D-42).

## Đã kiểm tra và KHÔNG khác biệt (đừng điều tra lại)

- Context manager `no_grad` và `inference_mode` tồn tại (`src/bindings/bind_core_ops.cpp:114, 124`).
- `cross_entropy_loss` có `weight`, `ignore_index`, `reduction`, `label_smoothing` (`src/bindings/bind_ops.cpp:70-75`).
- `Tensor.__pow__` tồn tại với số mũ vô hướng; bề mặt `Ops` rộng (127 `def` trong bind_ops.cpp).
- `Tensor.cuda()` / `.cpu()` / `.half()` / `.float()` / `.double()` / `.int()` / `.long()` tồn tại và ánh xạ tới đúng kiểu dữ liệu (`.long()` là INT64 kể từ bản sửa D-06).
- Linalg: 27/27 op khớp torch (inv/solve/cholesky/det/slogdet/eigvalsh/trace/matrix_power/cond/diagonal/rank/pinv/norms/lu/cross/vecdot/multi_dot/outer/inner/kron/vdot; qr/svd qua tái tạo do mơ hồ dấu; nghiệm lstsq khớp). FFT: 9/9 khớp (tách thực/ảo của bố cục `[..., 2]`). Phân phối: 19/20 khớp (NegativeBinomial là hoán đổi D-32).
- Các scheduler StepLR, CosineAnnealingLR, ExponentialLR, MultiStepLR, LinearLR, PolynomialLR, LambdaLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts đều khớp chính xác với torch. OneCycleLR là lệch D-31.
- Bỏ qua tràn GradScaler đã xác minh: gradient inf → bước bị bỏ, scale 4.0 → 2.0.
- AdamW8bit/FP8 so với torch AdamW: giống hệt tới 6 chữ số thập phân trên thăm dò 1 bước.
- Nghiệm `linalg.lstsq` khớp torch; `Ops.cross_entropy_loss` / NLL / fused LCE khớp chính xác với torch cho target INT32 (CE 5.96e-08, NLL 0, fused LCE 0).
- `flash_attention` khớp attention thủ công của torch (2.38e-07 causal, 1.19e-07); `all_reduce` chồng chéo khớp chính xác trên cả hai rank.

## Kiểm tra trực tiếp còn cần

- L-01 (Kaggle GPU): D-02 rơi về với GPU thật nhưng bận; hướng chuyển D-10 theo từng họ op; độ lớn sai số làm tròn int64 FP32 của D-21.
- L-02 (harness CPU, nay là kiểm tra hồi quy): chênh lệch số tối đa so với torch cho các op linalg D-07 (kỳ vọng parity); nâng kiểu vô hướng D-08 trên mọi kiểu dữ liệu số nguyên; xác nhận dung sai GELU D-09 (đo lại trong đợt này: 4.13e-04 so với exact, 0.0 so với tanh); ranh giới `.item()` int64 D-18 (2^53).
- L-03 (harness CPU): nâng kiểu lượng tử D-19 so với ngữ nghĩa torch.
- L-04 (Kaggle TPU): D-02 rơi về TPU; xác nhận lỗi `flash_attention_half` D-20 trên TPU.
- L-05 (harness CPU, kiểm tra hồi quy sau bản sửa): D-06 — đầu ra `.long()` nay là INT64 khi dùng làm chỉ mục trong `gather` / `index_select` (chấp nhận INT64).

## Tóm tắt trạng thái sửa

| Trạng thái | Mục |
|---|---|
| FIXED | D-06 (hợp nhất @89afd67), D-07 (hợp nhất @7b8981e) |
| FILED | D-22–D-29 (comms 20261027-001600), D-23 thêm (comms 20261026-234500), D-30–D-32 + ghi chú kèm D-36 (comms 20261027-000400), D-33 (comms 20261027-000800), D-37 (comms 20261026-233500), D-38 (comms 20261026-234600), D-39 (cùng D-22–D-29), D-47–D-49 (comms 20261027-012500) |
| OPEN | D-02, D-03, D-04, D-05, D-08, D-10, D-11, D-12, D-13, D-14, D-15, D-16, D-17, D-18, D-19, D-21, D-35, D-40, D-41, D-42, D-43, D-44, D-45 |
| DOCUMENTED (không có kế hoạch sửa) | D-01 (lỗi C++; Python không bind), D-09 (GELU tanh), D-34 (eps gradcheck), D-46 (RNG shuffle của DataLoader) |
