# How-To H3 — Lượng tử hóa một mô hình

Ba cách thu nhỏ một mô hình: huấn luyện nhận biết lượng tử hóa (fake
quantization trong lúc huấn luyện), hiệu chuẩn sau huấn luyện (đo scale
từ activation thật), và phép nhân ma trận trực tiếp ở độ chính xác thấp
(W8A8, FP8, INT4/NF4). Mọi ví dụ dưới đây chạy trên CPU với LiteTorch
88a4ba1.

Điều kiện tiên quyết: LiteTorch đã được cài và import được. Parity
với torch được kiểm bằng `~/torch-venv` (torch 2.14.1+cpu).

> Các khác biệt với PyTorch liên quan ở đây: D-19 (các dtype đã lượng
> tử hóa promote lên FP32 — kiểm tra live vẫn đang mở, xem ghi chú
> H3.1), D-09 (GELU chỉ có dạng tanh-approximation; so sánh với
> `F.gelu(approximate='tanh')`).

## H3.1 — Huấn luyện nhận biết lượng tử hóa với QATLinear

`QATLinear` bọc một `Linear` và áp fake quantization lên trọng số và
activation ở mỗi forward pass, để mô hình học các trọng số chịu được
lượng tử hóa INT8/INT4 thật. Đặt scale và zero point từ lần hiệu chuẩn
của bạn (H3.2); `b` là số bit.

<!-- ex: id=H3.1-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

lt.manual_seed(0)
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

print("QAT output shape:", list(out.shape))
print("Weight grad first 5:", qat_layer.weight.grad.to_vector()[:5])
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: shape
`[4, 256]` và một weight gradient khác không. Sai số fake-quantization
bị chặn bởi nửa weight scale (`w_scale / 2` = 0,0025 ở đây) cho mỗi
trọng số; chính chặn đó làm cho trọng số đã huấn luyện bằng QAT bền
vững khi bạn lượng tử hóa thật sau này.

## H3.2 — Hiệu chuẩn scale từ activation thật

Chạy các input đại diện qua mô hình, thu thập thống kê activation bằng
`Calibrator`, và đọc scale cùng các tham số asymmetric cho số bit của
bạn. Dùng các số này làm scale cho `QATLinear` ở H3.1 hoặc cho một lần
chuyển đổi PTQ.

<!-- ex: id=H3.2-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()
calibrator = lt.quantization.Calibrator()

sample_act = lt.Tensor.from_vector([0.1, -0.5, 2.3, -1.8], [1, 4], device, False)
calibrator.collect("layer1_act", sample_act)

scale = calibrator.get_scale("layer1_act", bits=8)
params = calibrator.get_asymmetric_params("layer1_act", bits=8)
print("Scale (8-bit):", scale)
print("Asymmetric params:", params)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: một scale
dương và một bộ tham số asymmetric cho `layer1_act`. Thu thập từ dữ
liệu thật, không phải từ hằng số tổng hợp — scale chỉ tốt bằng các
activation bạn đã đo. Đối với một pass PTQ đầy đủ, dùng
`lt.quantization.quantize_linear(layer)` sau hiệu chuẩn, hoặc
`quantize_dynamic_linear` / `quantize_dynamic_lstm` /
`quantize_dynamic_gru` cho lượng tử hóa động không cần tập hiệu chuẩn.

## H3.3 — Phép nhân ma trận W8A8: FP32 vào, toán INT8, FP32 ra

`w8a8_matmul(x, w, x_scale, w_scale)` lượng tử hóa cả hai input FP32
sang INT8, nhân chúng bằng số học nguyên, và dequantize bằng
`x_scale * w_scale`. Trọng số có dạng `[N, K]` — kênh đầu ra trước,
như trọng số của `nn.Linear` — và op tính `x @ w.T`, **không phải**
`x @ w`. Cả hai input phải là FP32; input INT8 raise
`w8a8_matmul requires FP32 inputs`.

<!-- ex: id=H3.3-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

x = lt.Tensor.from_vector([10.0, 20.0, -15.0, 30.0], [2, 2], device, False)
w = lt.Tensor.from_vector([5.0, -2.0, 1.0, 4.0], [2, 2], device, False)

w8a8_out = lt.Ops.w8a8_matmul(x, w, 0.01, 0.02)
print("W8A8 result:", w8a8_out.to_vector())
print("W8A8 dtype:", w8a8_out.dtype)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, parity với torch 1e-4 theo
công thức nguyên `round(x / x_scale) @ round(w / w_scale).T * x_scale *
w_scale` với clamp về [-128, 127]. Kết quả mong đợi:
`[0.6858, 4.4958, -5.7912, 1.9458]`. Chú ý phép chuyển vị: với
`x = [[10, 20], [-15, 30]]` và `w = [[5, -2], [1, 4]]`, kết quả là
`x @ w.T` trong số học đã lượng tử hóa — đưa nhầm một trọng số `[K, N]`
sẽ lặng lẽ tính sai tích, nên kiểm tra shape trọng số trước.

## H3.4 — Trọng số 4-bit: cast NF4 và FP4

Cast trọng số FP32 sang NF4 (NormalFloat4, định dạng QLoRA) hoặc
FP4_E2M1 (NVFP4). Cast là một phép chuyển đổi storage; phép tính vẫn
chạy ở FP32 sau một lần upcast ngầm, nên hãy đo sai số round-trip
trước khi tin một mô hình đã lượng tử hóa.

<!-- ex: id=H3.4-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

w_fp32 = lt.Tensor.from_vector([0.1, -0.5, 1.2, -2.4], [2, 2], device, False, lt.DataType.FP32)
w_fp4 = w_fp32.cast(lt.DataType.FP4_E2M1)
w_nf4 = w_fp32.cast(lt.DataType.NF4)

print("FP4 dtype:", w_fp4.dtype)
print("NF4 dtype:", w_nf4.dtype)
print("FP4 round-trip:", w_fp4.cast(lt.DataType.FP32).to_vector())
print("NF4 round-trip:", w_nf4.cast(lt.DataType.FP32).to_vector())
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: dtype
`FP4_E2M1` và `NF4`, với các giá trị round-trip
FP4 `[0.0, -0.5, 1.0, -2.0]` và NF4 `[0.0596, -0.5251, 1.0, -1.0]`.
Hãy đọc chúng một cách trung thực: FP4_E2M1 làm tròn mỗi giá trị về
lưới của nó (0.1 không có điểm lưới gần và làm tròn về 0.0), và các giá
trị NF4 được chuẩn hóa về [-1, 1], nên input ngoài khoảng sẽ clamp
(1.2 → 1.0, -2.4 → -1.0). Một `.cast()` trần không có scale cho mỗi
tensor không phải là một lượng tử hóa dùng được — đó chính là lý do
hiệu chuẩn (H3.2) và QAT (H3.1) tồn tại. Nếu sai số round-trip lớn hơn
dung sai của bạn, lớp đó cần QAT hoặc một định dạng rộng hơn, chứ
không phải một cast mù.
