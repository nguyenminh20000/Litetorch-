# How-To H8 — Debug: Gradcheck, Parity với Torch, Chẩn đoán OOM

Ba kỹ năng debug, theo thứ tự bạn dùng tới: chứng minh gradient đúng
(`gradcheck`), chứng minh số liệu khớp PyTorch (parity), và chứng minh
lỗi bộ nhớ là thật (chẩn đoán OOM). Mọi ví dụ dưới đây chạy trên CPU
với LiteTorch 88a4ba1.

Điều kiện tiên quyết: LiteTorch đã được cài và import được. Các ví dụ
parity import torch — chạy chúng bằng `~/torch-venv` (torch 2.14.1+cpu);
system pip không có torch.

> Các khác biệt với PyTorch liên quan ở đây: D-02 (`.to(device)` lặng
> lẽ fallback về CPU khi accelerator không có), D-09 (GELU chỉ có dạng
> tanh-approximation), D-12 (autograd giữ gradient trên các leaf tensor
> — xem H8.3).

## H8.1 — Gradcheck: chứng minh một gradient là đúng

`lt.autograd.gradcheck(func, inputs, eps, atol, rtol)` so sánh
gradient analytic với gradient finite-difference. Trên FP32, dùng
`eps=1e-3` (hoặc `1e-4`): `eps=1e-6` mặc định quá nhỏ so với làm tròn
FP32 và sẽ fail một gradient đúng. Truyền `raise_exception=True` để
lỗi raise thay vì trả về `False`.

<!-- ex: id=H8.1-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

x = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], device, True)


def quad(t):
    return lt.Ops.sum(t * t)


ok = lt.autograd.gradcheck(quad, [x], eps=1e-3)
print("gradcheck passed:", ok)
loss = quad(x)
loss.backward()
print("analytic grad:", x.grad.to_vector())
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, parity với torch 1e-4
(`torch.autograd.gradcheck` trên cùng hàm trả về `True`). Kết quả mong
đợi: `gradcheck passed: True`, gradient analytic `[2.0, 4.0, 6.0]`.
Nếu gradcheck fail, các nghi phạm thường gặp là: một custom op có
backward không khớp forward, một phép reduce làm rơi một số hạng
broadcast, hoặc một op in-place đã ghi đè một tensor đã lưu — kiểm tra
theo thứ tự đó.

## H8.2 — Parity với torch: pattern so sánh năm dòng

Pattern tham chiếu cho mọi khẳng định số liệu trong tài liệu này: cùng
seed, cùng input, cùng op, độ lệch tuyệt đối lớn nhất, và cả gradient
cũng được so sánh — không chỉ giá trị forward.

<!-- ex: id=H8.2-1 device=cpu backend=cpu runtime=~5s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import torch

import litetorch as lt

torch.manual_seed(0)
lt.manual_seed(0)

x_lt = lt.Tensor.from_vector([0.5, -1.0, 2.0, -0.25], [2, 2], lt.auto_device(), True)
x_pt = torch.tensor([[0.5, -1.0], [2.0, -0.25]], dtype=torch.float32, requires_grad=True)

lin_lt = lt.nn.Linear(2, 2, True)
lin_lt.to(lt.auto_device())
w_lt = lin_lt.weight
b_lt = lin_lt.bias
lin_pt = torch.nn.Linear(2, 2, dtype=torch.float32)
with torch.no_grad():
    lin_pt.weight.copy_(torch.tensor(w_lt.to_vector()).reshape(2, 2))
    lin_pt.bias.copy_(torch.tensor(b_lt.to_vector()))

y_lt = lin_lt(x_lt)
y_pt = lin_pt(x_pt)
loss_lt = lt.Ops.sum(y_lt)
loss_pt = y_pt.sum()
loss_lt.backward()
loss_pt.backward()

fwd_diff = max(abs(a - b) for a, b in zip(y_lt.to_vector(), y_pt.detach().flatten().tolist()))
grad_diff = max(abs(a - b) for a, b in zip(x_lt.grad.to_vector(), x_pt.grad.flatten().tolist()))
print("forward max abs diff:", fwd_diff)
print("grad max abs diff:", grad_diff)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, torch 2.14.1+cpu. Kết quả
mong đợi: cả hai độ lệch đều ở hoặc dưới 1e-4 — trong thực tế đúng
`0.0` đối với một lớp linear. Dung sai là 1e-4 cho FP32, 1e-2 cho
FP16/BF16. Một độ lệch trên dung sai là một divergence thật cho tới
khi chứng minh ngược lại: đừng nới dung sai để nó pass.

## H8.3 — Chẩn đoán OOM và kiểm tra silent CPU fallback

Lỗi out-of-memory và lỗi sai device trông giống nhau trên đường loss,
nên hãy kiểm tra device trước, rồi mới tới bộ nhớ. Ví dụ này chạy ba
kiểm tra quyết định: CUDA có thực sự có không, tensor có nằm đúng chỗ
bạn yêu cầu không, và cache CPU đang giữ bao nhiêu ngay lúc này.

<!-- ex: id=H8.3-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

print("cuda available:", lt.cuda.is_available())
print("cuda devices:", lt.cuda.device_count())

device = lt.Device("cuda")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], device, False)
print("requested cuda, got:", t.device)
print("fallback to CPU:", t.device.type == lt.DeviceType.CPU)

cpu_bytes = lt.get_cached_cpu_bytes()
print("cached CPU bytes at startup:", cpu_bytes)
scratch = lt.Tensor.from_vector([1.0] * 1000000, [1000, 1000], lt.auto_device(), False)
del scratch
print("cached CPU bytes before empty_cache:", lt.get_cached_cpu_bytes())
lt.empty_cache()
print("cached CPU bytes after empty_cache:", lt.get_cached_cpu_bytes())
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi trên máy
chỉ có CPU: `cuda available: False`, `cuda devices: 0`, device của tensor in ra `cpu:0` dù đã yêu cầu
`cuda`, và số byte đã cache giảm từ 4000000 (tensor scratch 4 MB đã
giải phóng, allocator vẫn giữ) về 0 sau `empty_cache()`. Fallback lặng lẽ đó là D-02: nếu lần huấn luyện
của bạn chậm nhưng không crash, đây là điều đầu tiên cần kiểm tra —
bạn có thể đang chạy trên CPU mà không biết.

Khi một OOM thật xảy ra, các knob là: `lt.empty_cache()` (giải phóng
các block caching allocator đang giữ), `lt.set_max_cpu_cache_size(bytes)`
(giới hạn CPU cache để nó không tăng quá ngân sách), và
`lt.get_gpu_memory_used()` (số byte đang giữ trên GPU). Hãy giảm batch
size hoặc bật gradient checkpointing trước khi nâng giới hạn cache —
một cache lớn hơn chỉ dời lỗi sang một tensor lớn hơn.
