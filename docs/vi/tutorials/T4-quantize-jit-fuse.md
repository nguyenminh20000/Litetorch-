# Hướng dẫn T4 — Lượng tử hóa và hợp nhất JIT cho một mô hình, từ đầu đến cuối

Lấy một lớp linear FP32 thuần, huấn luyện nó với nhận biết lượng tử,
hiệu chỉnh các hệ số scale từ activation thật, chạy một phép nhân ma
trận INT8, hợp nhất một đồ thị elementwise thành một kernel JIT duy
nhất, và nén trọng số xuống 4 bit. Mọi bước đều chạy trên CPU với
LiteTorch 88a4ba1, với sai số lượng tử được đo và nêu rõ — không bao
giờ giấu đi.

Bạn sẽ học được:

- Cách fake quantization hoạt động trong `QATLinear`, và sai số của nó
  lớn đến đâu.
- Cách `Calibrator` biến activation quan sát được thành các hệ số scale.
- Cách `w8a8_matmul` lượng tử hóa, nhân, và giải lượng tử.
- Cách trace và hợp nhất một biểu thức JIT, và cách các op đã hợp nhất
  mang gradient.
- Việc cast 4 bit (NF4, FP4) tốn bao nhiêu sai số khứ hồi.

Điều kiện tiên quyết: Hướng dẫn T1–T2. Các lưu ý về số học FP8 được
nêu ở chỗ áp dụng; các đường lượng tử chỉ có trên GPU được đánh dấu
REQUIRES-GPU và kiểm chứng trên Kaggle ở B5, không khẳng định ở đây.

> Các điểm khác biệt với PyTorch trong hướng dẫn này: D-09 (GELU chỉ có
> dạng xấp xỉ tanh — đối chiếu với `F.gelu(x, approximate='tanh')`, không
> bao giờ đối chiếu với GELU erf chính xác mặc định), D-19 (các kiểu dữ
> liệu đã lượng tử được promote lên FP32 — mới chỉ có bằng chứng trên
> cây, chưa đối chiếu với torch; đừng khẳng định parity torch cho
> promotion đã lượng tử), D-21 (trên GPU, phép toán elementwise trên số
> nguyên chạy qua FP32; đường CPU dùng ở đây là chính xác).

## T4.1 — Huấn luyện với nhận biết lượng tử bằng QATLinear

`QATLinear` bọc một `Linear` thuần và fake-quantize trọng số và
activation xuống 8 bit trên mỗi forward pass, để quá trình huấn luyện
nhìn thấy sai số mà suy luận sẽ gặp. Sai số là thật và đo được: với
`w_scale=0.005` và `a_scale=0.01` trên đầu vào dưới đây, đầu ra QAT
khác đầu ra FP32 tối đa 0.0500 (trung bình 0.0133). Đó là sai số fake
quantization kỳ vọng, không phải lỗi — nó tăng theo các hệ số scale mà
bạn chọn.

<!-- ex: id=T4.1-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

lt.manual_seed(2)
device = lt.auto_device()

raw_linear = lt.nn.Linear(512, 256, True)
raw_linear.to(device)

qat_layer = lt.quantization.QATLinear(
    raw_linear,
    w_scale=0.005,
    w_zp=0.0,
    a_scale=0.01,
    a_zp=0.0,
    b=8
)

x = lt.Tensor.from_vector([0.5] * (4 * 512), [4, 512], device, True)
out = qat_layer.forward(x)
loss = lt.Ops.sum(out)
loss.backward()

print("shape đầu ra QAT:", out.shape)
print("các giá trị đầu của gradient trọng số:", qat_layer.weight.grad.to_vector()[:5])

plain = raw_linear.forward(x)
diffs = [abs(a - b) for a, b in zip(out.to_vector(), plain.to_vector())]
print("độ lệch tuyệt đối lớn nhất so với FP32:", max(diffs))
print("độ lệch tuyệt đối trung bình so với FP32:", sum(diffs) / len(diffs))
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: gradient chảy qua bộ fake
quantizer (gradient trọng số khác 0), độ lệch tuyệt đối lớn nhất so với
FP32 là 0.0500, trung bình là 0.0133. Không có khẳng định parity với
torch: torch không có bản tương đương `QATLinear` với các hệ số scale
này; phép so sánh là với chính lớp FP32 được bọc.

## T4.2 — Hiệu chỉnh các hệ số scale từ activation thật

Đoán hệ số scale là sai; hãy đo chúng. `Calibrator` thu thập thống kê
activation, rồi `get_scale` / `get_asymmetric_params` trả về hệ số
scale và điểm 0 cho độ rộng bit đã chọn.

<!-- ex: id=T4.2-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

device = lt.auto_device()
calibrator = lt.quantization.Calibrator()

sample_act = lt.Tensor.from_vector([0.1, -0.5, 2.3, -1.8], [1, 4], device, False)
calibrator.collect("layer1_act", sample_act)

scale = calibrator.get_scale("layer1_act", bits=8)
params = calibrator.get_asymmetric_params("layer1_act", bits=8)
print("hệ số scale:", scale)
print("tham số bất đối xứng (hệ số scale, điểm 0):", params)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: hệ số scale là 0.01811 và
các tham số bất đối xứng là (0.01614, -15.0) cho mẫu này. Trong một
luồng PTQ, các hệ số scale này đưa vào `QATLinear` (hoặc
`quantize_linear`) thay cho các hằng số chọn tay.

## T4.3 — Phép nhân ma trận W8A8

`w8a8_matmul` nhận đầu vào FP32 — truyền tensor INT8 sẽ gây lỗi
(`w8a8_matmul requires FP32 inputs`; khung cũ §10.3 đã làm đúng như
vậy). Nó lượng tử hóa cả hai đầu vào xuống INT8, nhân trong INT8, và
giải lượng tử: out = (round(x / x_scale) @ round(w / w_scale)^T) *
x_scale * w_scale, trong đó `w` có shape `[N, K]`. Kết quả khớp đúng
công thức đó.

<!-- ex: id=T4.3-1 device=cpu backend=cpu runtime=~5.0s torch-parity=exact status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt
import torch

device = lt.auto_device()

x = lt.Tensor.from_vector([10.0, 20.0, -15.0, 30.0], [2, 2], device, False, lt.DataType.FP32)
w = lt.Tensor.from_vector([5.0, -2.0, 1.0, 4.0], [2, 2], device, False, lt.DataType.FP32)

out = lt.Ops.w8a8_matmul(x, w, x_scale=0.01, w_scale=0.02)
print("kết quả W8A8:", out.to_vector())

xt = torch.tensor([[10.0, 20.0], [-15.0, 30.0]])
wt = torch.tensor([[5.0, -2.0], [1.0, 4.0]])
qx = torch.clamp(torch.round(xt / 0.01), -128, 127).to(torch.int32)
qw = torch.clamp(torch.round(wt / 0.02), -128, 127).to(torch.int32)
ref = (qx @ qw.T).float() * 0.01 * 0.02
print("kết quả công thức torch:", ref.flatten().tolist())
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity
exact: cả hai cho [0.6858, 4.4958, -5.7912, 1.9458]. Bản thân phép nhân
INT8 là chính xác; sai số duy nhất so với phép nhân ma trận FP32 thuần
là việc lượng tử hóa đầu vào, vốn bị bão hòa ở đây (giá trị / 0.01 vượt
quá khoảng INT8).

## T4.4 — JIT: trace một biểu thức và hợp nhất nó

Xây dựng một biểu thức tượng trưng bằng `JITVar`, biên dịch nó thành
một `JITFunction`, chạy nó, lưu nó, và nạp lại. `Tracer.trace` làm
điều tương tự từ một hàm Python thông thường.

<!-- ex: id=T4.4-1 device=cpu backend=cpu runtime=~8.7s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt
import torch

device = lt.auto_device()

v1 = lt.jit.JITVar(lt.jit.OpType.INPUT, "x")
v2 = lt.jit.JITVar(lt.jit.OpType.INPUT, "w")
expr = lt.jit.relu(v1 * 2.0 + v2)

jit_fn = lt.jit.JITFunction("fused_relu_add", expr, [v1, v2])

arg1 = lt.Tensor.from_vector([1.0, 2.0], [2], device, False)
arg2 = lt.Tensor.from_vector([-3.0, 1.0], [2], device, False)

res = jit_fn([arg1, arg2])
print("đầu ra JIT:", res.to_vector())

jit_fn.save("/tmp/fused_relu.jit")
loaded_fn = lt.jit.JITFunction.load("/tmp/fused_relu.jit")
print("đầu ra sau khi nạp lại:", loaded_fn([arg1, arg2]).to_vector())

def custom_activation(args):
    return lt.Ops.gelu(lt.Ops.add(args[0], args[1]))

traced_fn = lt.jit.Tracer.trace([arg1, arg2], custom_activation, "traced_gelu")
traced_res = traced_fn([arg1, arg2])
print("đầu ra GELU đã trace:", traced_res.to_vector())

ref = torch.relu(torch.tensor([1.0, 2.0]) * 2.0 + torch.tensor([-3.0, 1.0]))
print("tham chiếu torch:", ref.tolist())
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4
(đầu ra JIT [0.0, 5.0] khớp đúng với torch; đầu ra GELU đã trace khớp
với `F.gelu(x, approximate='tanh')` — xem T4.5 để biết vì sao
`approximate` lại quan trọng).

## T4.5 — Các op đã hợp nhất mang gradient

`JITFunction` tích hợp với autograd: với đầu vào có
`requires_grad=True`, backward lan truyền qua kernel đã hợp nhất. GELU
đã hợp nhất ở đây là dạng xấp xỉ tanh (D-09) — đối chiếu nó với
`F.gelu(approximate='tanh')`, không bao giờ đối chiếu với GELU erf
chính xác mặc định của torch, vốn lệch tới khoảng 4.4e-4.

<!-- ex: id=T4.5-1 device=cpu backend=cpu runtime=~8.0s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt
import torch

device = lt.auto_device()

x = lt.Tensor.from_vector([1.0, -2.0, 3.0], [3], device, True)
bias = lt.Tensor.from_vector([0.5, 0.5, 0.5], [3], device, True)

out = lt.jit.fused_bias_gelu(x, bias)
loss = lt.Ops.sum(out)
loss.backward()

print("đầu ra đã hợp nhất:", out.to_vector())
print("x.grad:", x.grad.to_vector())
print("bias.grad:", bias.grad.to_vector())

xt = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
bt = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
ref = torch.nn.functional.gelu(xt + bt, approximate="tanh")
ref.sum().backward()
print("GELU tanh của torch:", ref.detach().tolist())
print("x.grad của torch:", xt.grad.tolist())
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4:
độ lệch lớn nhất của forward và gradient đều bằng 0.0 so với
`F.gelu(approximate='tanh')`. So với GELU erf chính xác mặc định của
torch, cùng các đầu ra đó lệch tới khoảng 4.4e-4 — đó là D-09, một điểm
khác biệt đã ghi lại, không phải lỗi.

## T4.6 — Trọng số 4 bit: NF4 và FP4

Cast trọng số FP32 xuống 4 bit rồi trở lại. Khứ hồi mất mát theo thiết
kế — đó là cái giá của lưu trữ 4 bit. Trên các giá trị dưới đây, khứ
hồi FP4 lệch tối đa 0.40 và khứ hồi NF4 lệch tối đa 1.40 (giá trị -2.4
nằm ngoài khoảng NF4 bao phủ tốt).

<!-- ex: id=T4.6-1 device=cpu backend=cpu runtime=~0.2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

device = lt.auto_device()

w_fp32 = lt.Tensor.from_vector([0.1, -0.5, 1.2, -2.4], [2, 2], device, False, lt.DataType.FP32)
w_fp4 = w_fp32.cast(lt.DataType.FP4_E2M1)
w_nf4 = w_fp32.cast(lt.DataType.NF4)

print("kiểu dữ liệu FP4:", w_fp4.dtype)
print("kiểu dữ liệu NF4:", w_nf4.dtype)

back_fp4 = w_fp4.cast(lt.DataType.FP32)
back_nf4 = w_nf4.cast(lt.DataType.FP32)
orig = w_fp32.to_vector()
print("khứ hồi FP4:", back_fp4.to_vector())
print("khứ hồi NF4:", back_nf4.to_vector())
print("sai số khứ hồi lớn nhất FP4:", max(abs(a - b) for a, b in zip(orig, back_fp4.to_vector())))
print("sai số khứ hồi lớn nhất NF4:", max(abs(a - b) for a, b in zip(orig, back_nf4.to_vector())))
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: sai số khứ hồi lớn nhất FP4
là 0.40, NF4 là 1.40. Đừng khẳng định parity torch cho promotion đã
lượng tử hay cast 4 bit (D-19, mới chỉ có bằng chứng trên cây): hãy đo
sai số trên khoảng giá trị của riêng bạn trước khi tin một trọng số 4
bit.
