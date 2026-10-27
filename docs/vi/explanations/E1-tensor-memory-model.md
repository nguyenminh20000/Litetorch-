# E1 — Mô Hình Tensor và Bộ Nhớ

Một `tensor` là một khung nhìn (view) trên một khối lưu trữ (storage), không phải chính khối đó. Trang này giải thích một tensor chứa những gì, stride mô tả bố cục ra sao, thiết bị và allocator hoạt động thế nào, và LiteTorch khác PyTorch có chủ đích ở những điểm nào. Mọi ví dụ đều tự chứa và chạy được trên CPU.

- **Tree:** integration HEAD `b326f29`; extension chạy được build từ `88a4ba1` (`lt.build_commit()`, được assert trong mọi ví dụ).
- **Tham chiếu:** PyTorch 2.14.1+cpu. Parity với torch là FP32 1e-4 trừ khi header nói khác.
- **Các khác biệt liên quan:** D-01 (`data_ptr()`), D-02 (`to(device)` fallback thầm lặng về CPU), D-03 (không có dạng chuỗi cho `.to()`), D-06 (`.long()` — ĐÃ SỬA trên agent-v1, xem §3), D-14 (giới hạn kiểu dữ liệu của `.numpy()`), D-18 (độ chính xác của `.item()`/`.tolist()`). Xem `plan/differences.md`.

## 1. Cấu tạo của tensor

Một tensor gồm bốn thứ: một shape, một stride cho mỗi chiều, một kiểu dữ liệu, và một con trỏ vào khối lưu trữ cộng với một offset.

```
Tensor
├── shape    [2, 3]        kích thước logic của mỗi chiều
├── strides  [3, 1]        số phần tử phải bước qua cho mỗi bước, theo từng chiều
├── dtype    FP32          kiểu phần tử (DataType)
├── device   cpu:0         nơi khối lưu trữ nằm
├── offset   0             số phần tử tính từ đầu khối lưu trữ
└── storage  StorageImpl   khối đã cấp phát (được các view chia sẻ)
```

Địa chỉ của phần tử `(i, j)` là `base + offset + i * stride[0] + j * stride[1]`, tính theo phần tử. Stride tính bằng phần tử, không phải byte — cùng quy ước với PyTorch. `numel()` là tích của shape; `ndim` là số chiều; `is_contiguous()` cho biết các phần tử có nằm liên tục theo thứ tự row-major, không có khoảng trống hay không.

<!-- ex: id=E1-1 device=cpu backend=cpu runtime=~60s torch-parity=exact status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

a = lt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("shape:", a.shape)
print("strides:", a.strides)
print("dtype:", a.dtype)
print("device:", a.device)
print("numel:", a.numel())
print("ndim:", a.ndim)
print("is_contiguous:", a.is_contiguous())
b = a.transpose(0, 1)
print("transposed shape:", b.shape)
print("transposed strides:", b.strides)
print("transposed is_contiguous:", b.is_contiguous())
at = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("torch strides:", list(at.stride()))
print("torch transposed strides:", list(at.t().stride()))
assert list(a.shape) == list(at.shape)
assert list(a.strides) == list(at.stride())
assert list(b.strides) == list(at.t().stride())
print("PARITY OK")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity exact (shape và stride là số nguyên).

Kết quả mong đợi:

```
shape: [2, 3]
strides: [3, 1]
dtype: DataType.FP32
device: cpu:0
numel: 6
ndim: 2
is_contiguous: True
transposed shape: [3, 2]
transposed strides: [1, 3]
transposed is_contiguous: False
torch strides: [3, 1]
torch transposed strides: [1, 3]
PARITY OK
```

`transpose` hoán đổi shape và stride nhưng không di chuyển dữ liệu nào: tensor đã transpose là non-contiguous, với stride `[1, 3]`. Bất kỳ op nào cần dữ liệu row-major đặc đều gọi `.contiguous()` trước, và thao tác đó sao chép. Kiểu dữ liệu mặc định là FP32 và thiết bị mặc định là `cpu:0`.

## 2. View so với bản sao

`view`, `transpose`, `permute` và `slice` tạo ra các tensor mới trên cùng một storage: ghi qua view là ghi vào bản gốc. `clone` cấp phát storage mới: ghi qua clone không chạm tới bản gốc. `copy_` ghi vào storage của tensor đích, nên đó là cách quan sát sự chia sẻ.

<!-- ex: id=E1-2 device=cpu backend=cpu runtime=~60s torch-parity=exact status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

a = lt.tensor([[1.0, 2.0], [3.0, 4.0]])
v = a.view([4])
v.copy_(lt.tensor([9.0, 9.0, 9.0, 9.0]))
print("sau copy_ qua view, a:", a.to_vector())
c = a.clone()
c.copy_(lt.tensor([[7.0, 7.0], [7.0, 7.0]]))
print("sau copy_ qua clone, a:", a.to_vector())
print("clone:", c.to_vector())
t = lt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tt = t.transpose(0, 1)
print("transposed is_contiguous:", tt.is_contiguous())
print("giá trị sau transpose:", tt.to_vector())
print("giá trị sau contiguous:", tt.contiguous().to_vector())
at = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
av = at.view(4)
av.copy_(torch.tensor([9.0, 9.0, 9.0, 9.0]))
assert at.flatten().tolist() == a.to_vector()
assert tt.to_vector() == torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).t().flatten().tolist()
print("PARITY OK")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity exact.

Kết quả mong đợi:

```
sau copy_ qua view, a: [9.0, 9.0, 9.0, 9.0]
sau copy_ qua clone, a: [9.0, 9.0, 9.0, 9.0]
clone: [7.0, 7.0, 7.0, 7.0]
transposed is_contiguous: False
giá trị sau transpose: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
giá trị sau contiguous: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
PARITY OK
```

Hai điểm cần nhớ. Thứ nhất, `to_vector()` luôn trả về các giá trị theo thứ tự logic, nên một tensor đã transpose và bản sao contiguous của nó in ra giống hệt nhau — khác biệt chỉ nằm ở stride và ở việc một op sau đó có phải sao chép hay không. Thứ hai, indexing bị giới hạn (D-11): chỉ `int` và `slice` trên chiều 0 được hỗ trợ, nên việc kiểm tra từng phần tử trong cuốn sách này đi qua `to_vector()`, `slice()` hoặc `.numpy()`.

## 3. Hệ thống kiểu dữ liệu

Các kiểu dữ liệu của LiteTorch nằm trong enum `DataType`: `FP32`, `FP64`, `FP16`, `BF16`, `INT64`, `INT32`, `INT16`, `INT8`, `UINT8`, `BOOL`, cộng thêm các kiểu lượng tử hóa (`FP8_E4M3`, `FP8_E5M2`, `FP4`, `FP4_E2M1`, `NF4`, `INT4`). Các phép cast là `.float()`, `.double()`, `.half()`, `.int()`, `.long()` và dạng tổng quát `.cast(DataType)`. Cast từ float sang int cắt về phía zero, giống torch.

<!-- ex: id=E1-3 device=cpu backend=cpu runtime=~60s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

x = lt.tensor([1.5, -2.5, 3.0])
print("float dtype:", x.float().dtype, x.float().to_vector())
print("double dtype:", x.double().dtype)
print("half dtype:", x.half().dtype)
print("int dtype:", x.int().dtype, x.int().to_vector())
print("long dtype:", x.long().dtype, x.long().to_vector())
print("cast INT64 dtype:", x.cast(lt.DataType.INT64).dtype, x.cast(lt.DataType.INT64).to_vector())
xt = torch.tensor([1.5, -2.5, 3.0])
print("torch long dtype:", xt.long().dtype, xt.long().tolist())
print("torch int dtype:", xt.int().dtype, xt.int().tolist())
assert x.int().to_vector() == [float(v) for v in xt.int().tolist()]
assert x.long().to_vector() == [float(v) for v in xt.long().tolist()]
assert x.cast(lt.DataType.INT64).to_vector() == [float(v) for v in xt.long().tolist()]
print("GIÁ TRỊ PARITY OK; kiểu dữ liệu long khác trên bản build này (xem phần giải thích)")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. Giá trị khớp với torch một cách chính xác; kiểu dữ liệu của `.long()` thì không (xem dưới đây).

Kết quả mong đợi:

```
float dtype: DataType.FP32 [1.5, -2.5, 3.0]
double dtype: DataType.FP64
half dtype: DataType.FP16
int dtype: DataType.INT32 [1.0, -2.0, 3.0]
long dtype: DataType.INT32 [1.0, -2.0, 3.0]
cast INT64 dtype: DataType.INT64 [1.0, -2.0, 3.0]
torch long dtype: torch.int64 [1, -2, 3]
torch int dtype: torch.int32 [1, -2, 3]
GIÁ TRỊ PARITY OK; kiểu dữ liệu long khác trên bản build này (xem phần giải thích)
```

**Độ lệch D-06, đọc kỹ.** D-06 (`.long()` trả về INT32) ĐÃ ĐƯỢC SỬA trên agent-v1 — binding hiện ánh xạ `long` tới `Tensor::to_long` (INT64), đã merge thành `89afd67` và có mặt trong integration HEAD `b326f29` (`src/bindings/bind_tpu.cpp:155`). Nhưng extension chạy được dùng cho batch này được build từ `88a4ba1`, ra đời trước bản sửa, nên `.long()` ở đây vẫn trả về INT32. Giá trị đúng trong cả hai trường hợp; chỉ kiểu dữ liệu khác nhau. Cho tới khi extension được build lại từ một tree sau `89afd67`, hãy dùng `.cast(lt.DataType.INT64)` để lấy INT64 — cách này đúng trên bản build này lẫn trên tree đã sửa. Không có các phương thức `.bool()`, `.byte()`, `.char()`, `.short()` hay `.to_long()` (D-13); `.cast()` là đường đi tổng quát.

## 4. Đưa dữ liệu về host: `.numpy()`, `.item()`, `.tolist()`

### 4.1 Giới hạn kiểu dữ liệu của `.numpy()` (D-14)

`.numpy()` hỗ trợ FP64, FP32, FP16, INT32, INT16, INT8. UINT8, INT64 và BOOL ném lỗi `unsupported dtype for copy conversion`. BF16 được cast thầm lặng sang FP32 trước. Tensor không nằm trên CPU sẽ ném lỗi và phải được chuyển bằng `.cpu()` trước.

<!-- ex: id=E1-4 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"

f = lt.tensor([1.0, 2.0, 3.0])
print("FP32 numpy:", f.numpy(), f.numpy().dtype)
i64 = lt.tensor([1, 2, 3]).cast(lt.DataType.INT64)
try:
    i64.numpy()
    print("INT64 numpy: OK")
except RuntimeError as e:
    print("INT64 numpy báo lỗi:", e)
u8 = lt.tensor([1, 2, 3]).cast(lt.DataType.UINT8)
try:
    u8.numpy()
    print("UINT8 numpy: OK")
except RuntimeError as e:
    print("UINT8 numpy báo lỗi:", e)
bl = lt.tensor([1, 2, 3]).cast(lt.DataType.BOOL)
try:
    bl.numpy()
    print("BOOL numpy: OK")
except RuntimeError as e:
    print("BOOL numpy báo lỗi:", e)
bf = lt.tensor([1.0, 2.0]).cast(lt.DataType.BF16)
print("BF16 numpy:", bf.numpy(), bf.numpy().dtype)
i32 = lt.tensor([1, 2, 3]).cast(lt.DataType.INT32)
print("INT32 numpy:", i32.numpy(), i32.numpy().dtype)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Không có parity với torch: đây là một khác biệt hành vi, được ghi lại ở đây thay vì giấu đi.

Kết quả mong đợi:

```
FP32 numpy: [1. 2. 3.] float32
INT64 numpy báo lỗi: [litetorch Error] numpy(): unsupported dtype for copy conversion
UINT8 numpy báo lỗi: [litetorch Error] numpy(): unsupported dtype for copy conversion
BOOL numpy báo lỗi: [litetorch Error] numpy(): unsupported dtype for copy conversion
BF16 numpy: [1. 2.] float32
INT32 numpy: [1 2 3] int32
```

Cách xử lý: cast sang FP32 trước (lưu ý về độ chính xác với int64 trên 2^24 — xem §4.2), hoặc đọc giá trị bằng `.to_vector()` / `.tolist()`. Lưu ý sự tương tác với D-06: `.long().numpy()` ném lỗi trên bản build này lẫn trên tree đã sửa, vì lỗi nằm ở INT64, không phải ở `.long()`.

### 4.2 Độ chính xác của `.item()` và `.tolist()` (D-18)

`.item()` trả về một `float` của Python, cast các tensor không phải FP32 sang FP32 trước. `.to_vector()` và `.tolist()` cũng vậy. Một giá trị int64 trên 2^53 do đó trở về đã bị làm tròn, một cách thầm lặng. `.item()` của PyTorch trên tensor int64 trả về một `int` chính xác của Python.

<!-- ex: id=E1-5 device=cpu backend=cpu runtime=~60s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

cpu = lt.Device("cpu")
big = lt.Tensor.from_vector([2**60 + 1], [1], cpu, False, lt.DataType.INT64)
print("lt item của 2**60+1:", big.item())
print("lt tolist của 2**60+1:", big.tolist())
print("lt to_vector của 2**60+1:", big.to_vector())
print("lt mất độ chính xác:", big.item() == float(2**60))
tb = torch.tensor(2**60 + 1)
print("torch item của 2**60+1:", tb.item())
print("giá trị chính xác:", 2**60 + 1)
small = lt.tensor([7]).cast(lt.DataType.INT64)
print("lt item của 7:", small.item())
assert big.item() == float(2**60)
assert tb.item() == 2**60 + 1
print("ĐÃ XÁC NHẬN KHÁC BIỆT")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. Sự khác biệt chính là điểm cần chứng minh: LiteTorch làm tròn, torch thì không.

Kết quả mong đợi:

```
lt item của 2**60+1: 1.152921504606847e+18
lt tolist của 2**60+1: [1.152921504606847e+18]
lt to_vector của 2**60+1: [1.152921504606847e+18]
lt mất độ chính xác: True
torch item của 2**60+1: 1152921504606846977
giá trị chính xác: 1152921504606846977
lt item của 7: 7.0
ĐÃ XÁC NHẬN KHÁC BIỆT
```

Việc làm tròn xảy ra hai lần: `from_vector` đưa giá trị qua `float` ngay khi khởi tạo, và `.item()` trả về `float`. Số nguyên nhỏ thì chính xác, nên lỗi chỉ cắn khi vượt quá 2^53. Đừng bao giờ dùng `.item()` cho các giá trị int64 lớn — chỉ số, token id gần ngưỡng, giá trị băm. Hãy giữ chúng ở dạng tensor.

## 5. Mô hình thiết bị

Một `Device` đặt tên cho một backend và một chỉ số: `cpu:0`, `gpu:0`, `tpu:0`, `meta`. LiteTorch không có dạng chuỗi cho `.to()` (D-03): phải khởi tạo `lt.Device("cuda")` một cách tường minh. Hai hành vi thầm lặng nảy sinh từ mô hình thiết bị, và cả hai đều là bẫy.

**D-02 — fallback thầm lặng về CPU.** `.to(Device("cuda"))` trên một máy không có backend GPU thầm lặng trả về một tensor CPU. Không lỗi, không cảnh báo, không log. `.cuda()` đi qua cùng đường đó. PyTorch ném lỗi.

**D-03 — chuỗi thiết bị sai chính tả trở thành CPU.** `Device("cude")` — một lỗi đánh máy — được phân tích thành CPU, một cách thầm lặng. PyTorch ném lỗi với chuỗi thiết bị không hợp lệ.

<!-- ex: id=E1-6 device=cpu backend=cpu runtime=~90s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

print("GPU khả dụng:", lt.is_gpu_available())
print("TPU khả dụng:", lt.is_tpu_available())
t = lt.tensor([1.0, 2.0])
g = t.to(lt.Device("cuda"))
print("sau to(cuda):", g.device)
typo = t.to(lt.Device("cude"))
print("sau to(Device('cude')):", typo.device)
c = t.cpu()
print("sau .cpu():", c.device)
assert str(g.device) == "cpu:0"
assert str(typo.device) == "cpu:0"
print("ĐÃ XÁC NHẬN FALLBACK THẦM LẶNG")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. Không có parity với torch: torch ném lỗi ở đúng chỗ LiteTorch fallback thầm lặng.

Kết quả mong đợi (stdout; dòng `[litetorch Warning] GPU acceleration (OpenCL) is not available` đi ra stderr):

```
GPU khả dụng: False
TPU khả dụng: False
sau to(cuda): cpu:0
sau to(Device('cude')): cpu:0
sau .cpu(): cpu:0
ĐÃ XÁC NHẬN FALLBACK THẦM LẶNG
```

Cách xử lý là một kỷ luật một dòng: luôn kiểm tra `t.device` sau `.to()` / `.cuda()` và báo lỗi lớn tiếng nếu nó không phải thiết bị đã yêu cầu. Đừng bao giờ cho rằng một tensor đang nằm trên GPU chỉ vì code đã yêu cầu GPU. Cùng một fallback áp dụng cho TPU, và `Module.to()` thừa hưởng nó cho từng tham số. Các op nhị phân trên các thiết bị khác nhau cũng chuyển đổi thầm lặng thay vì ném lỗi (D-10) — hãy chuyển tensor một cách tường minh trước khi trộn các thiết bị.

<!-- ex: id=E1-9 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"

assert lt.is_gpu_available()
t = lt.tensor([1.0, 2.0]).to(lt.Device("cuda"))
assert str(t.device) == "gpu:0"
print("on GPU:", t.device)
```

**REQUIRES-GPU — chưa chạy trên CPU, không bao giờ được phát hành như đã kiểm chứng.** Khối trên là bản đối tác GPU của E1-6: trên một máy có backend CUDA thật, cùng một `.to(Device("cuda"))` phải đặt tensor lên `gpu:0`, và phép kiểm tra thiết bị phải đạt. Nó chạy trên Kaggle (QA sở hữu session) sau khi cổng tiền kiểm (pre-flight gate) assert dòng khởi tạo `Native GPU (CUDA/HIP)` và vị trí của `liblitetorch_gpu.so`.

## 6. Con trỏ thô: `data_ptr()` (D-01)

Trong C++, `Tensor::data_ptr()` trả về một `float*` và ném lỗi với bất kỳ kiểu dữ liệu nào không phải FP32: `[litetorch Error] data_ptr(): tensor dtype is not FP32; use raw_data_ptr() for byte-level access` (`src/tensor/tensor_core.cpp:1292-1297`). `raw_data_ptr()` trả về `void*` cho mọi kiểu dữ liệu (`tensor_core.cpp:1299-1303`). Nó cũng trả về một con trỏ CPU (`storage->get_cpu_ptr() + offset`) ngay cả với tensor GPU — không bao giờ là con trỏ thiết bị. `Tensor.data_ptr()` của PyTorch hoạt động với mọi kiểu dữ liệu.

Không hàm nào trong hai hàm được bind trong Python trên bản build này:

<!-- ex: id=E1-7 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"

t = lt.tensor([1.0, 2.0, 3.0])
print("has data_ptr:", hasattr(t, "data_ptr"))
print("has raw_data_ptr:", hasattr(t, "raw_data_ptr"))
print("giá trị host qua to_vector:", t.to_vector())
print("giá trị host qua numpy:", t.numpy().tolist())
assert not hasattr(t, "data_ptr")
assert not hasattr(t, "raw_data_ptr")
print("PYTHON KHÔNG CÓ BINDING NÀO; C++ data_ptr() ném lỗi trên non-FP32 (xem phần giải thích)")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Không có parity với torch: chính sự vắng mặt của binding là phát hiện.

Kết quả mong đợi:

```
has data_ptr: False
has raw_data_ptr: False
giá trị host qua to_vector: [1.0, 2.0, 3.0]
giá trị host qua numpy: [1.0, 2.0, 3.0]
PYTHON KHÔNG CÓ BINDING NÀO; C++ data_ptr() ném lỗi trên non-FP32 (xem phần giải thích)
```

Từ Python, truy cập host đi qua `.numpy()` (chịu các giới hạn kiểu dữ liệu D-14) hoặc `.to_vector()` / `.tolist()` (chịu việc làm tròn float D-18). Code ported gọi `t.data_ptr()` trên tensor int, bool, FP16 hay FP64 — trong C++ hay qua một binding trong tương lai — phải chuyển sang `raw_data_ptr()` và tự cast sang kiểu phần tử.

## 7. Allocator và caching allocator

LiteTorch cấp phát qua một caching allocator: các khối đã giải phóng được giữ trong cache theo kích thước và tái sử dụng cho lần cấp phát tiếp theo cùng kích thước, thay vì trả về hệ điều hành. `lt.get_cached_cpu_bytes()` báo cáo có bao nhiêu byte đang nằm trong cache CPU; `lt.empty_cache()` giải phóng chúng; `lt.set_max_cpu_cache_size(n)` đặt trần cho cache. Cùng mẫu đó tồn tại trên GPU (`get_cached_gpu_bytes()`).

<!-- ex: id=E1-8 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"

lt.manual_seed(0)
print("số byte đã cache trước:", lt.get_cached_cpu_bytes())
a = lt.randn([1000, 1000])
print("số byte đã cache khi tensor còn sống:", lt.get_cached_cpu_bytes())
del a
print("số byte đã cache sau del:", lt.get_cached_cpu_bytes())
lt.empty_cache()
print("số byte đã cache sau empty_cache:", lt.get_cached_cpu_bytes())
lt.set_max_cpu_cache_size(1024 * 1024)
print("set_max_cpu_cache_size ok")
assert lt.get_cached_cpu_bytes() == 0
print("CACHING ALLOCATOR OK")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Không có parity với torch: ví dụ này quan sát chính allocator của LiteTorch, không phải một op của torch.

Kết quả mong đợi:

```
số byte đã cache trước: 0
số byte đã cache khi tensor còn sống: 0
số byte đã cache sau del: 4000000
số byte đã cache sau empty_cache: 0
set_max_cpu_cache_size ok
CACHING ALLOCATOR OK
```

Đọc các con số cẩn thận: cache bằng 0 trong khi tensor 4.000.000 byte còn sống — các byte đang được dùng, không phải đã cache. `del` giải phóng tensor, và khối chuyển vào cache thay vì trả về hệ điều hành: 4.000.000 byte đã cache. `empty_cache()` giải phóng chúng. Đó là hành vi có chủ đích, và đó là lý do "bộ nhớ vẫn được cấp phát sau `del`" thường là cache, không phải rò rỉ. Hai hệ quả theo sau. Thứ nhất, bộ nhớ đỉnh được quyết định bởi tập hợp sống lớn nhất, không phải bởi tổng số từng cấp phát — tái sử dụng giữ footprint phẳng qua các bước huấn luyện. Thứ hai, giải phóng bộ nhớ (memory eviction — điểm khác biệt của LiteTorch: offload các tensor khi chịu áp lực) tương tác với cache — các tensor bị evict giải phóng khối của chúng vào cache, nơi một lần cấp phát sau đó cùng kích thước tái sử dụng chúng mà không chạm tới hệ điều hành.

## 8. Những gì trang này không bao quát

- Cơ chế đạo hàm tự động — xem [E2 Đạo Hàm Tự Động (Autograd)](E2-autograd.md).
- Bảng promotion đầy đủ của kiểu dữ liệu, bao gồm việc promotion lên FP32 của các kiểu lượng tử hóa (D-19) và đường đi int qua FP32 trên GPU (D-21) — xem `plan/differences.md` và tài liệu tham khảo API.
- Chính bề mặt op — xem `reference/tensor.md` và `reference/ops-elementwise.md`.
