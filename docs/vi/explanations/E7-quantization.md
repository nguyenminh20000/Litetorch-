# Giải thích E7 — Lượng Tử Hóa (Quantization)

Lượng tử hóa lưu trọng số và activation bằng ít bit hơn FP32: 8 bit
(INT8, FP8), 4 bit (INT4, NF4, FP4), đôi khi ít hơn nữa. Lợi ích
không chỉ là tốc độ tính toán — mà là bộ nhớ. Một mô hình 7B tham số
ở FP32 cần 28 GB trọng số; ở INT8 cần 7 GB, ở INT4 cần 3,5 GB. Trên
phần cứng bị giới hạn bởi băng thông (một T4 đang phục vụ LLM, một
điện thoại chạy classifier) trọng số phải được đọc từ bộ nhớ cho mỗi
token, nên giảm một nửa số byte thì thời gian giảm gần một nửa, và
kiểu dữ liệu nhỏ hơn mở ra các đường tensor-core số nguyên và FP8 mà
FP32 không dùng được.

Cái giá phải trả là sai số. Lượng tử hóa mất mát theo thiết kế: mỗi
giá trị được làm tròn tới điểm lưới representable gần nhất, và các giá
trị ngoài khoảng lưới sẽ bão hòa. Trang này nêu ngân sách sai số bằng
số đo thực, không bao giờ bằng cảm tính — và nêu đúng công thức mà
LiteTorch cài đặt, vì công thức ngây thơ là sai (xem E7.2).

Điều kiện tiên quyết: Hướng dẫn T1–T2 (tensor, đạo hàm tự động) và T4
(lượng tử hóa trọn vẹn). Mọi ví dụ chạy trên CPU với LiteTorch 88a4ba1,
PyTorch 2.14.1+cpu. Các khẳng định về throughput trên GPU được đánh
dấu REQUIRES-GPU và không được kiểm chứng ở đây.

> Các khác biệt với PyTorch được chạm tới trong trang này: D-19 (kiểu
> dữ liệu đã lượng tử được promote lên FP32 — mới chỉ có bằng chứng
> trên cây, chưa thăm dò với torch; đừng khẳng định torch parity cho
> promotion đã lượng tử), D-21 (trên GPU, phép toán elementwise trên
> số nguyên chạy qua FP32; đường số nguyên trên CPU dùng ở đây là
> chính xác), D-08 (tensor số nguyên cộng một số nguyên Python sẽ
> promote lên FP32 — hãy xây tensor hệ số scale tường minh khi kiểu
> dữ liệu quan trọng).

## E7.1 — Các định dạng: FP8, INT8, INT4, NF4

Bốn định dạng bao phủ gần như mọi trường hợp lượng tử hóa thực tế.
Chúng khác nhau ở điểm mạnh, không chỉ ở số bit:

| Định dạng | Bit | Giá trị | Lưới | Tốt nhất cho |
|---|---|---|---|---|
| FP8 E4M3 | 8 | float, 4 bit exponent + 3 bit mantissa | logarit, tối đa ~±448 | activation, đầu vào phép nhân ma trận |
| FP8 E5M2 | 8 | float, 5 bit exponent + 2 bit mantissa | logarit, tối đa ~±57344 | gradient, tensor có dải động rộng |
| INT8 | 8 | số nguyên, -128…127 (xem E7.3) | đều, bước = hệ số scale | trọng số + activation (w8a8) |
| INT4 | 4 | số nguyên, -8…7 | đều, bước = hệ số scale | nén chỉ trọng số |
| NF4 | 4 | float, các mức phân phối chuẩn | không đều, dày gần 0 | trọng số LLM (kiểu QLoRA) |
| FP4 E2M1 | 4 | float, 2 bit exponent + 1 bit mantissa | logarit, thô | chỉ trọng số, nén cực mạnh |

Điểm phân chia quan trọng nhất là **lưới đều hay không đều**. INT8
và INT4 dùng lưới đều: giá trị là `round(x / s)`, cách nhau `s`, và hệ
số scale `s` là tham số duy nhất (cộng thêm điểm 0 tùy chọn cho
khoảng bất đối xứng). FP8 và NF4 dùng lưới không đều: khoảng cách FP8
tăng theo độ lớn (như mọi số float), và NF4 đặt 16 mức của nó tại các
phân vị của phân phối chuẩn tắc, nên các mức dồn gần 0 — nơi trọng số
LLM thực sự nằm. Đó là lý do NF4 thắng INT4 trên trọng số phân phối
chuẩn ở cùng 4 bit — và là lý do NF4 là lựa chọn kém cho các giá trị
phân phối đều.

```
Đường giá trị FP32:    |----*----*----*----*----*----*----*----*----|
Lưới INT8 (đều):       +    +    +    +    +    +    +    +    +
Lưới NF4:              +  + +  +   +      +        +          +
                       ^ dày gần 0, thưa ở đuôi
```

Điểm phân chia thứ hai là **cái gì được lượng tử hóa**. Lượng tử hóa
chỉ trọng số (INT4, NF4) thu nhỏ lưu trữ và lưu lượng bộ nhớ;
activation vẫn ở FP16/FP32, nên phép nhân ma trận vẫn là số thực.
Lượng tử hóa toàn phần (w8a8, FP8) lượng tử hóa cả hai phía, nên phép
nhân ma trận chạy trong INT8/FP8 — nhanh hơn, nhưng sai số lượng tử
hóa activation giờ cũng đi vào kết quả, và activation khó lượng tử
hóa hơn trọng số (một vài kênh outlier chi phối hệ số scale).

## E7.2 — Công thức lượng tử–giải lượng tử, và hiệu chỉnh w8a8

Lượng tử hóa gồm hai bước: lượng tử hóa, rồi (để tính toán) giải
lượng tử.

```
lượng tử hóa:   q = clamp(round(x / s), qmin, qmax)     (mã số nguyên)
giải lượng tử:  x_hat = q * s                            (trở lại float)
sai số:         |x - x_hat| <= s / 2   (trong khoảng; lớn hơn ngoài khoảng)
```

Hệ số scale `s` quyết định sai số: tối đa nửa bước, cộng thêm sai số
bão hòa cho mọi giá trị có `|x| > qmax * s`. Chọn `s` tức là chọn
ngân sách sai số — đó là lý do hiệu chỉnh (E7.5) đo `s` từ dữ liệu
thật thay vì đoán.

Với phép nhân ma trận, dạng đã lượng tử **không phải** là dạng ngây
thơ `x @ w.T * s`. Cả hai toán hạng phải được lượng tử hóa *trước*
phép nhân, và mỗi phía được giải lượng tử bằng hệ số scale của riêng
nó sau đó:

```
out = (round(x / x_scale) @ round(w / w_scale).T) * x_scale * w_scale
```

với clamp INT8 về [-128, 127]. Dạng ngây thơ `(x @ w.T)` nhân trong
FP32 trước rồi mới scale — nó không lượng tử hóa gì cả, và trả về
các số khác. Lượt tham chiếu B3 đã bắt đúng lỗi này trong một bản
nháp trước: công thức được khẳng định `(x @ w.T) * x_scale * w_scale`
bị bác bỏ bởi một lần chạy thực, và công thức trên mới là thứ
`src/ops/w8a8_ops.cpp` cài đặt. E7-1 dưới đây kiểm chứng nó với một
mirror torch.

<!-- ex: id=E7-1 device=cpu backend=cpu runtime=~2s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([3.0, 7.0, -2.0, 9.0], [2, 2], cpu, False)
w = lt.Tensor.from_vector([4.0, -1.0, 2.0, 5.0], [2, 2], cpu, False)
out = lt.Ops.w8a8_matmul(x, w, x_scale=0.05, w_scale=0.04)
print("Kết quả W8A8:", out.to_vector())
xt = torch.tensor([[3.0, 7.0], [-2.0, 9.0]])
wt = torch.tensor([[4.0, -1.0], [2.0, 5.0]])
qx = torch.clamp(torch.round(xt / 0.05), -128, 127).to(torch.int32)
qw = torch.clamp(torch.round(wt / 0.04), -128, 127).to(torch.int32)
ref = (qx @ qw.T).float() * 0.05 * 0.04
print("Kết quả công thức torch:", ref.flatten().tolist())
diff = max(abs(a - b) for a, b in zip(out.to_vector(), ref.flatten().tolist()))
print("Sai lệch lớn nhất so với công thức torch:", diff)
assert diff <= 1e-4
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity
1e-4 (sai lệch lớn nhất 1.9e-06, làm tròn float32 trong phép nhân
hệ số scale cuối): cả hai đều cho [5.65, 37.75, -14.35, 27.75]. Đầu vào bão hòa ở
đây (7.0 / 0.05 = 140 bị clamp về 127), nên kết quả khác phép nhân
ma trận FP32 thuần — đó là sai số lượng tử hóa, không phải lỗi.
Lưu ý `w8a8_matmul` nhận đầu vào FP32; truyền tensor INT8 sẽ throw
(`w8a8_matmul requires FP32 inputs`).

## E7.3 — `fake_quantize`: làm tròn, clamp, và hai điểm đã đo

Fake quantization mô phỏng làm tròn INT8 trong FP32: giá trị được làm
tròn tới lưới rồi đưa trở lại FP32, để mạng huấn luyện với sai số mà
nó sẽ gặp khi suy luận. `Ops.fake_quantize(x, scale, zero_point=0.0,
bits=8)` là hàm nguyên thủy; `QATLinear` (E7.5) gọi nó trên trọng số
và activation mỗi forward pass.

Hai hành vi của cài đặt này đã được đo trên build này (2026-10-27) và
khác với cách viết `round` + clamp của torch, nên đừng viết mirror
torch mà giả định ngữ nghĩa torch:

- **Khoảng clamp là [-127, 127], không phải [-128, 127].**
  `130.0` ở `scale=0.1` cho fake quantization thành `12.7`, và
  `-200.0` thành `-12.7`.
- **Trường hợp .5 làm tròn ra xa 0.** `-0.05` ở `scale=0.1` thành
  `-0.1`; `torch.round(-0.5)` cho `-0.0` (làm tròn về chẵn).

Hãy tránh các trường hợp .5 chính xác và mã `-128` trong mirror
torch của op này; phần mirror của E7-2 dưới đây dùng giá trị không
có tie.

<!-- ex: id=E7-2 device=cpu backend=cpu runtime=~2s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([0.3, -1.7, 2.9, 0.04, 12.0, -12.0], [6], cpu, False)
fq = lt.Ops.fake_quantize(x, 0.1, 0.0, 8)
print("fake_quantize s=0.1:", fq.to_vector())
xt = torch.tensor([0.3, -1.7, 2.9, 0.04, 12.0, -12.0])
ref = torch.clamp(torch.round(xt / 0.1), -127, 127) * 0.1
print("mirror torch:", ref.tolist())
diff = max(abs(a - b) for a, b in zip(fq.to_vector(), ref.tolist()))
print("Sai lệch lớn nhất so với mirror torch:", diff)
assert diff <= 1e-4
sat = lt.Tensor.from_vector([130.0, -200.0, -0.05], [3], cpu, False)
print("Thăm dò bão hòa:", lt.Ops.fake_quantize(sat, 0.1, 0.0, 8).to_vector())
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity
1e-4 trên các giá trị mirror. Thăm dò bão hòa in ra [12.7, -12.7,
-0.1]: khoảng clamp là [-127, 127] và tie làm tròn ra xa 0 (xem các
điểm đã đo ở trên).

## E7.4 — FP8: cast, nhân ma trận, uncast

FP8 giữ lưới số thực, nên nó xử lý dải động rộng tốt hơn INT8 ở cùng
8 bit — với cái giá là ít bit mantissa hơn (E4M3 có 3 bit, nên các giá
trị kề nhau lệch nhau 12.5%). LiteTorch tách ba bước ra riêng:

- `fp8_cast_scaled(x, scale)` lưu `round(x / scale)` trong FP8. Hệ
  số scale là số chia: cast `[0.5, -1.25, 3.0, 0.125]` với
  `scale=0.05` lưu `[10.0, -24.0, 60.0, 2.5]` (đo 2026-10-27; `-25`
  không representable trong E4M3, nên nó rơi về `-24`).
- `fp8_uncast_scaled(x, scale)` nhân các giá trị đã lưu trở lại với
  hệ số scale. Cast + uncast với cùng hệ số scale cho khứ hồi đúng
  các giá trị representable trong FP8.
- `fp8_matmul(a, b, a_scale, b_scale, out_scale, out_dtype=FP8_E4M3)`
  nhận **tensor FP8 đã cast** — truyền đầu vào FP32 sẽ raise
  `fp8_uncast_scaled: input must be FP8`. Với `out_dtype=FP32` kết
  quả là tích FP32 thuần của các đầu vào đã giải lượng tử.

<!-- ex: id=E7-3 device=cpu backend=cpu runtime=~2s torch-parity=1e-2 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], cpu, False)
b = lt.Tensor.from_vector([1.0, 0.0, 0.0, 1.0], [2, 2], cpu, False)
ac = lt.Ops.fp8_cast_scaled(a, 1.0)
bc = lt.Ops.fp8_cast_scaled(b, 1.0)
out = lt.Ops.fp8_matmul(ac, bc, 1.0, 1.0, 1.0, lt.DataType.FP32)
print("Đầu ra FP32 của fp8_matmul:", out.to_vector())
ref = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) @ torch.tensor([[1.0, 0.0], [0.0, 1.0]])
diff = max(abs(x - y) for x, y in zip(out.to_vector(), ref.flatten().tolist()))
print("Sai lệch lớn nhất so với phép nhân ma trận FP32:", diff)
assert diff <= 1e-2
x = lt.Tensor.from_vector([0.5, -1.25, 3.0, 0.125], [4], cpu, False)
back = lt.Ops.fp8_uncast_scaled(lt.Ops.fp8_cast_scaled(x, 0.05), 0.05)
print("Khứ hồi FP8:", back.to_vector())
err = max(abs(p - q) for p, q in zip(x.to_vector(), back.to_vector()))
print("Sai số khứ hồi lớn nhất FP8:", err)
assert err <= 0.06
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1: phép nhân
ma trận FP8 của các đầu vào representable đúng khớp với phép nhân ma
trận FP32 ([1.0, 2.0, 3.0, 4.0]), và khứ hồi lệch tối đa 0.05 trên
các giá trị thăm dò — một bước lưới FP8 ở hệ số scale 0.05. Lớp dung
sai 1e-2 áp dụng cho phép so sánh nhân ma trận (sai lệch 0.0); sai số
khứ hồi được nêu rõ như một số tuyệt đối đã đo, không giấu.
E4M3 với E5M2 là đánh đổi khoảng/độ chính xác: E4M3 cho activation
(nhiều mantissa hơn), E5M2 cho gradient (khoảng rộng hơn).

## E7.5 — QAT với PTQ: huấn luyện cùng sai số, hay hiệu chỉnh sau

**QAT (huấn luyện nhận biết lượng tử, quantization-aware training)**
huấn luyện với fake quantization trong vòng lặp, để trọng số thích
nghi với việc làm tròn. `QATLinear` bọc một `Linear` thuần và fake
quantization trọng số và activation mỗi forward pass; gradient chảy
qua bộ fake quantizer (straight-through estimator), nên quá trình
huấn luyện thấy sai số mà suy luận sẽ gặp. Chi phí: thời gian huấn
luyện và forward pass fake quantization mỗi bước. Dùng khi bạn
huấn luyện lại và độ chính xác quan trọng nhất.

**PTQ (lượng tử hóa sau huấn luyện, post-training quantization)**
lượng tử hóa một mô hình đã huấn luyện xong mà không huấn luyện lại:
đo khoảng activation/trọng số trên dữ liệu hiệu chỉnh (`Calibrator`,
`prepare` + `calibrate`), rồi thay mỗi `Linear` bằng một bản đã lượng
tử (`convert`). Chi phí: gần như không có. Độ chính xác: kém hơn
QAT, đôi khi kém nhiều trên outlier — dữ liệu hiệu chỉnh phải giống
dữ liệu triển khai, nếu không hệ số scale sẽ sai với đầu vào thật.

```
QAT:  huấn luyện --fake-quant--> triển khai (cùng đồ thị, INT8 thật)
       ^ sai số nằm trong loss, trọng số thích nghi
PTQ:  huấn luyện (FP32) --> hiệu chỉnh hệ số scale --> convert --> triển khai
       ^ sai số chỉ xuất hiện sau huấn luyện, trọng số không bao giờ thích nghi
```

<!-- ex: id=E7-4 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
lt.manual_seed(4)
raw_linear = lt.nn.Linear(64, 32, True)
raw_linear.to(cpu)
qat_layer = lt.quantization.QATLinear(raw_linear, w_scale=0.01, w_zp=0.0, a_scale=0.02, a_zp=0.0, b=8)
x = lt.Tensor.from_vector([0.5] * (2 * 64), [2, 64], cpu, True)
out = qat_layer.forward(x)
loss = lt.Ops.sum(out)
loss.backward()
plain = raw_linear.forward(x)
diffs = [abs(a - b) for a, b in zip(out.to_vector(), plain.to_vector())]
print("Sai lệch tuyệt đối lớn nhất QAT so với FP32:", max(diffs))
print("Sai lệch tuyệt đối trung bình QAT so với FP32:", sum(diffs) / len(diffs))
print("Gradient trọng số khác 0:", any(v != 0.0 for v in qat_layer.weight.grad.to_vector()))
assert max(diffs) <= 0.05
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: sai lệch tuyệt đối lớn
nhất so với FP32 là 0.0203, trung bình 0.0065, và gradient trọng số
khác 0 — gradient chảy qua bộ fake quantizer. Không khẳng định torch
parity: torch không có bản tương đương `QATLinear` với các hệ số
scale này; phép so sánh là với lớp FP32 được bọc. (T4.1 đo 0.0500 lớn
nhất / 0.0133 trung bình ở `w_scale=0.005`, `a_scale=0.01` — sai số
tăng theo hệ số scale bạn chọn.)

<!-- ex: id=E7-5 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
lt.manual_seed(3)
model = lt.nn.Sequential([lt.nn.Linear(8, 8, True)])
model.to(cpu)
xin = lt.Tensor.from_vector([0.1, -0.4, 0.9, 0.2, -0.7, 1.3, 0.05, -1.1], [1, 8], cpu, False)
fp32_out = model.forward(xin).to_vector()
cfg = lt.quantization.PTQConfig()
print("Mặc định PTQConfig bits/per_channel/symmetric:", cfg.bits, cfg.per_channel, cfg.symmetric)
prepared = lt.quantization.prepare(model, cfg)
prepared.calibrate(lambda: lt.Tensor.from_vector([0.1, -0.4, 0.9, 0.2, -0.7, 1.3, 0.05, -1.1], [1, 8], cpu, False), 4)
converted = lt.quantization.convert(prepared)
ptq_out = converted.forward(xin).to_vector()
diffs = [abs(a - b) for a, b in zip(fp32_out, ptq_out)]
print("Sai lệch tuyệt đối lớn nhất PTQ so với FP32:", max(diffs))
print("Sai lệch tuyệt đối trung bình PTQ so với FP32:", sum(diffs) / len(diffs))
assert max(diffs) <= 1e-2
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: mặc định PTQ là 8 bit,
per-channel, symmetric; trên mô hình nhỏ này đầu ra đã convert lệch
so với FP32 tối đa 0.0020 (trung bình 0.0010) — trong dung sai 1e-2
sau giải lượng tử. Hãy đo đầu ra FP32 **trước** `convert`:
`convert` thay `Linear` tại chỗ bên trong mô hình, nên đọc đầu ra
"FP32" sau đó là so sánh mô hình với chính nó. `convert` lượng tử
hóa trọng số theo từng kênh đầu ra (`src/ops/ptq.cpp`: hệ số scale =
max hàng / 127, mã INT8, giải lượng tử trong forward pass).

## E7.6 — Trọng số 4 bit: INT4, NF4, FP4

Lượng tử hóa 4 bit chỉ trọng số là cách rẻ nhất để giảm một nửa kích
thước mô hình lần nữa sau INT8. `int4_pack` / `int4_unpack` đóng hai
giá trị 4 bit (khoảng -8…7) vào mỗi byte INT8;
`int4_weight_only_linear(x, packed_w, w_scale, bias=None)` tính `x @
dequant(packed_w).T (+ bias)` với `packed_w` 2D `[N, (K+1)/2]` và một
hệ số scale cho mỗi kênh đầu ra. Tensor packed 1D phải được reshape
trước — truyền nó vào sẽ raise `packed_w shape mismatch`.

NF4 và FP4 là phép cast, không phải đường nhân ma trận:
`w.cast(DataType.NF4)` lưu mỗi trọng số thành mức NF4 gần nhất. Khứ
hồi mất mát theo thiết kế, và mức mất mát phụ thuộc vào khoảng giá
trị: trên `[0.1, -0.5, 1.2, -2.4]` khứ hồi FP4 lệch tối đa 0.40 và
khứ hồi NF4 lệch tối đa 1.40 — giá trị -2.4 nằm ngoài khoảng NF4 bao
phủ tốt (các mức NF4 dồn gần 0). Hãy đo sai số trên phân phối trọng
số của riêng bạn trước khi tin một trọng số 4 bit.

<!-- ex: id=E7-6 device=cpu backend=cpu runtime=~2s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"
cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], cpu, False)
w = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0], [16], cpu, False, lt.DataType.INT8)
packed = lt.Ops.int4_pack(w).reshape([4, 2])
w_scale = lt.Tensor.from_vector([1.0, 1.0, 1.0, 1.0], [4], cpu, False)
out = lt.Ops.int4_weight_only_linear(x, packed, w_scale)
print("Đầu ra linear chỉ trọng số INT4:", out.to_vector())
wt = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 1.0], [2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 1.0, 2.0]])
xt = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
ref = xt @ wt.T
diff = max(abs(a - b) for a, b in zip(out.to_vector(), ref.flatten().tolist()))
print("Sai lệch lớn nhất so với mirror giải lượng tử torch:", diff)
assert diff <= 1e-4
w_fp32 = lt.Tensor.from_vector([0.1, -0.5, 1.2, -2.4], [2, 2], cpu, False, lt.DataType.FP32)
for dtype, name in [(lt.DataType.FP4_E2M1, "FP4"), (lt.DataType.NF4, "NF4")]:
    back = w_fp32.cast(dtype).cast(lt.DataType.FP32)
    err = max(abs(a - b) for a, b in zip(w_fp32.to_vector(), back.to_vector()))
    print(name, "sai số khứ hồi lớn nhất:", err)
    assert err <= 1.5
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1: linear
chỉ trọng số INT4 khớp đúng mirror giải lượng tử torch ([30.0, 42.0,
40.0, 31.0, 70.0, 118.0, 96.0, 95.0]); sai số khứ hồi lớn nhất FP4
0.40, NF4 1.40 — đo lại 2026-10-27, xác nhận các số B1-tut. Đừng
khẳng định torch parity cho cast 4 bit (D-19, mới chỉ có bằng chứng
trên cây).

## E7.7 — Ngân sách sai số: kỳ vọng gì, assert gì

| Đường | Sai số đã đo (build này, CPU) | Dung sai cần assert |
|---|---|---|
| Nhân ma trận w8a8 so với công thức lượng tử–giải lượng tử | 1.9e-06 (làm tròn float32 khi nhân hệ số scale) | 1e-4 |
| `fake_quantize` so với mirror torch (không tie) | 0.0 | 1e-4 |
| Nhân ma trận FP8 so với nhân ma trận FP32 (đầu vào representable) | 0.0 | 1e-2 sau giải lượng tử |
| Khứ hồi cast/uncast FP8 (giá trị thăm dò) | 0.05 | ngân sách tuyệt đối; 1 bước lưới ở s=0.05 |
| QAT so với FP32 (`w_scale=0.01`, `a_scale=0.02`) | lớn nhất 0.0203, trung bình 0.0065 | ngân sách, không phải parity — tăng theo `s` |
| PTQ 8 bit so với FP32 (Linear nhỏ) | lớn nhất 0.0020, trung bình 0.0010 | 1e-2 sau giải lượng tử |
| INT4 chỉ trọng số so với mirror giải lượng tử | exact (0.0) | 1e-4 |
| Khứ hồi FP4 | 0.40 | ngân sách — phụ thuộc khoảng |
| Khứ hồi NF4 | 1.40 | ngân sách — phụ thuộc khoảng |

Quy tắc chung: sai số lượng tử hóa tăng theo hệ số scale `s`, và `s`
tăng theo độ lớn lớn nhất trong tensor — một kênh outlier có thể phá
độ chính xác của cả tensor. Đó là lý do hệ số scale per-channel (một
`s` cho mỗi hàng đầu ra) thắng hệ số scale per-tensor, và là lý do
outlier activation thường là nguyên nhân khiến độ chính xác PTQ sụp
đổ. Dung sai parity đã lượng tử là 1e-2 sau giải lượng tử
(verification.md §3.2); parity FP32 vẫn là 1e-4.

## E7.8 — Khi nào không lượng tử hóa, và cái gì cần GPU

Đừng lượng tử hóa những phần mà sai số cộng dồn nhanh nhất: lớp
logits cuối của classifier (chênh lệch logit nhỏ lật argmax),
embedding với token hiếm (ít mẫu hiệu chỉnh), và bất cứ thứ gì mà đầu
ra của nó nuôi một chuỗi tính toán dài tiếp theo mà không có chuẩn
hóa lại. Hãy lượng tử hóa các phép nhân ma trận lớn trước — chúng giữ
phần lớn số byte — và đo độ chính xác sau mỗi bước, không phải sau
tất cả các bước.

> REQUIRES-GPU: throughput phục vụ với trọng số đã lượng tử
> (tokens/s, time-to-first-token), nhân ma trận FP8 ở quy mô lớn, và
> benchmark kernel INT8 là các phép đo GPU. Chúng thuộc về Kaggle
> (B5), không thuộc trang này — không khẳng định số throughput nào ở
> đây.
