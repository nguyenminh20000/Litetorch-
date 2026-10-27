# E2 — Đạo Hàm Tự Động (Autograd)

Autograd ghi lại các phép toán trên các tensor có `requires_grad=True` thành một đồ thị, rồi đi ngược đồ thị để tính gradient. Trang này giải thích cách xây dựng đồ thị, `backward`, `gradcheck`/`gradgradcheck`, các chế độ no-grad, tensor leaf so với non-leaf, và tích lũy gradient — cùng bài học từ lỗi no-learn 2026-10-26, khi các giá trị forward khớp với PyTorch trong khi các gradient đã hỏng.

- **Tree:** integration HEAD `b326f29`; extension chạy được build từ `88a4ba1` (`lt.build_commit()`, được assert trong mọi ví dụ).
- **Tham chiếu:** PyTorch 2.14.1+cpu. Parity với torch là FP32 1e-4 trừ khi header nói khác.
- **Các khác biệt liên quan:** `.grad` của non-leaf được điền mà không cần `retain_grad()` (§4, phát hiện mới trên bản build này); `zero_grad()` để `.grad` là `None` thay vì zeros (§5); `backward()` không có tham số `retain_graph` (§6). Không điểm nào trong số này có trong `plan/differences.md` — chúng được ghi lại ở đây và báo lên Docs leader.

## 1. Xây dựng đồ thị và `backward`

Mọi phép toán trên một tensor có `requires_grad=True` tạo ra một nút trong một đồ thị có hướng không chu trình (DAG) và đánh dấu đầu ra của nó là cần gradient. `.backward()` đi ngược đồ thị từ đầu ra về các leaf, áp dụng quy tắc chuỗi (chain rule), và ghi gradient của mỗi leaf vào `.grad`. Trước bất kỳ backward nào, `.grad` là `None`. `lt.autograd.grad(outputs, inputs)` tính cùng các gradient đó mà không ghi chúng vào `.grad`.

```
x (leaf, requires_grad=True)
│
▼  y = x * x          nút: MulBackward
│
▼  loss = y.sum()     nút: SumBackward
│
backward() đi SumBackward → MulBackward,
tích lũy d loss / d x vào x.grad
```

<!-- ex: id=E2-1 device=cpu backend=cpu runtime=~60s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

x = lt.tensor([2.0, 3.0])
x.requires_grad = True
y = x * x
print("y requires_grad:", y.requires_grad)
print("y.grad trước backward:", y.grad)
y.sum().backward()
print("x.grad:", x.grad.to_vector())
xx = lt.tensor([3.0])
xx.requires_grad = True
gg = lt.autograd.grad(xx * xx, [xx])
print("autograd.grad:", [g.to_vector() for g in gg])
xt = torch.tensor([2.0, 3.0], requires_grad=True)
yt = xt * xt
yt.sum().backward()
assert max(abs(a - b) for a, b in zip(x.grad.to_vector(), xt.grad.tolist())) <= 1e-4
assert gg[0].to_vector() == [6.0]
print("PARITY OK")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.

Kết quả mong đợi:

```
y requires_grad: True
y.grad trước backward: None
x.grad: [4.0, 6.0]
autograd.grad: [[6.0]]
PARITY OK
```

`d(x²)/dx = 2x`, nên tại `x = [2, 3]` các gradient là `[4, 6]`. Cần `y.sum()` vì `backward()` bắt đầu từ một vô hướng: gradient của phép sum toàn là 1, và đó là hạt giống cho quy tắc chuỗi. `autograd.grad` trả về một danh sách các gradient theo thứ tự của các input được yêu cầu, và không chạm tới `.grad`.

## 2. `gradcheck` và `gradgradcheck`: cái bẫy eps

`gradcheck` kiểm chứng một gradient giải tích với một gradient số được tính bằng sai phân trung tâm: `(f(x + ε) − f(x − ε)) / 2ε`. `gradgradcheck` làm điều tương tự cho đạo hàm bậc hai. Cả hai mặc định `eps=1e-6`. **Trên FP32, giá trị mặc định đó thất bại, ở cả hai framework.** Gradient số tại `x = 3` cho `t * t` ra 5.7220458984375 thay vì đạo hàm giải tích 6.0 — sai số 0.28, vượt xa mọi dung sai. Nguyên nhân là làm tròn FP32: tại `x = 3`, `f(x)` gần 9, và FP32 có khoảng 7 chữ số thập phân, nên `f(x + ε)` và `f(x − ε)` mỗi cái mang sai số tuyệt đối gần 1e-6; hiệu của chúng gần 1.2e-5, và chia cho `2ε = 2e-6` khuếch đại sai số lên cỡ 0.5. Với `eps=1e-3` cùng phép tính đó cho 5.999565124511719 và phép kiểm tra đạt.

<!-- ex: id=E2-2 device=cpu backend=cpu runtime=~90s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

def f(t):
    return t * t

x = lt.tensor([3.0])
x.requires_grad = True
print("lt gradcheck eps=1e-6:", lt.autograd.gradcheck(f, (x,), eps=1e-6, raise_exception=False))
print("lt gradcheck eps=1e-3:", lt.autograd.gradcheck(f, (x,), eps=1e-3, raise_exception=False))
xp = lt.tensor([3.0 + 1e-6])
xm = lt.tensor([3.0 - 1e-6])
num6 = ((xp * xp).to_vector()[0] - (xm * xm).to_vector()[0]) / (2e-6)
print("lt numerical grad eps=1e-6:", num6)
xp3 = lt.tensor([3.0 + 1e-3])
xm3 = lt.tensor([3.0 - 1e-3])
num3 = ((xp3 * xp3).to_vector()[0] - (xm3 * xm3).to_vector()[0]) / (2e-3)
print("lt numerical grad eps=1e-3:", num3)
print("đạo hàm giải tích:", 6.0)
xt = torch.tensor([3.0], requires_grad=True)
print("torch gradcheck eps=1e-6:", torch.autograd.gradcheck(lambda t: t * t, (xt,), eps=1e-6, raise_exception=False))
print("torch gradcheck eps=1e-3:", torch.autograd.gradcheck(lambda t: t * t, (xt,), eps=1e-3, raise_exception=False))
assert num6 != 6.0
assert abs(num3 - 6.0) <= 1e-3
print("EPS MẶC ĐỊNH THẤT BẠI TRÊN FP32; EPS=1E-3 ĐẠT (CẢ HAI FRAMEWORK)")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. Không có parity số: điểm cần chứng minh là cả hai framework cho cùng một kết quả — thất bại ở `eps=1e-6`, đạt ở `eps=1e-3`.

Kết quả mong đợi (stdout; `UserWarning` của torch về input không phải double đi ra stderr):

```
lt gradcheck eps=1e-6: False
lt gradcheck eps=1e-3: True
lt numerical grad eps=1e-6: 5.7220458984375
lt numerical grad eps=1e-3: 5.999565124511719
đạo hàm giải tích: 6.0
torch gradcheck eps=1e-6: False
torch gradcheck eps=1e-3: True
EPS MẶC ĐỊNH THẤT BẠI TRÊN FP32; EPS=1E-3 ĐẠT (CẢ HAI FRAMEWORK)
```

Quy tắc: **dùng `eps=1e-3` cho gradcheck FP32, và `eps=1e-6` chỉ cho FP64.** Một gradcheck thất bại ở eps mặc định trên FP32 là bằng chứng về eps, không phải về gradient. Cùng cái bẫy đó áp dụng cho `gradgradcheck`:

<!-- ex: id=E2-3 device=cpu backend=cpu runtime=~60s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

x = lt.tensor([3.0])
x.requires_grad = True
print("lt gradgradcheck eps=1e-3:", lt.autograd.gradgradcheck(lambda t: t * t, (x,), eps=1e-3, raise_exception=False))
y = x * x
g = lt.autograd.grad(y, [x], create_graph=True)[0]
print("grad bậc nhất:", g.to_vector())
g2 = lt.autograd.grad(g, [x])[0]
print("grad bậc hai:", g2.to_vector())
xt = torch.tensor([3.0], requires_grad=True)
print("torch gradgradcheck eps=1e-3:", torch.autograd.gradgradcheck(lambda t: t * t, (xt,), eps=1e-3, raise_exception=False))
assert g2.to_vector() == [2.0]
print("ĐẠO HÀM BẬC HAI OK")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.

Kết quả mong đợi (stdout; `UserWarning` của torch về input không phải double đi ra stderr):

```
lt gradgradcheck eps=1e-3: True
grad bậc nhất: [6.0]
grad bậc hai: [2.0]
torch gradgradcheck eps=1e-3: True
ĐẠO HÀM BẬC HAI OK
```

`d²(x²)/dx² = 2` ở mọi nơi, nên gradient bậc hai là `[2.0]`. Lưu ý `create_graph=True` trên `autograd.grad`: không có nó, gradient bậc nhất là một tensor thường không gắn đồ thị, và việc lấy đạo hàm của nó lần nữa không còn gì để lấy đạo hàm.

## 3. `no_grad` và `inference_mode`

Cả hai context manager đều dừng việc xây dựng đồ thị. Trong `no_grad()`, `is_grad_enabled()` là `False` và các tensor mới không cần gradient ngay cả khi `requires_grad` được đặt. Trong `inference_mode()`, `is_inference_mode()` là `True`. Cả hai khôi phục trạng thái trước đó khi thoát. Dùng `no_grad()` cho suy luận và cho bất kỳ phép tính nào mà gradient của nó sẽ không bao giờ cần tới — nó cũng bỏ qua việc ghi lại, nên nhanh hơn và dùng ít bộ nhớ hơn.

<!-- ex: id=E2-4 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
assert lt.build_commit() == "88a4ba1"

print("grad được bật trước:", lt.is_grad_enabled())
with lt.no_grad():
    n = lt.tensor([1.0])
    n.requires_grad = True
    m = n * 2
    print("trong no_grad, grad được bật:", lt.is_grad_enabled())
    print("trong no_grad, m.requires_grad:", m.requires_grad)
    print("trong no_grad, m.grad:", m.grad)
print("grad được bật sau:", lt.is_grad_enabled())
with lt.inference_mode():
    print("trong inference_mode:", lt.is_inference_mode())
print("inference_mode sau:", lt.is_inference_mode())
assert lt.is_grad_enabled()
assert not lt.is_inference_mode()
print("NO_GRAD VÀ INFERENCE_MODE OK")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Không có parity với torch: ví dụ này quan sát các cờ chế độ của chính LiteTorch.

Kết quả mong đợi:

```
grad được bật trước: True
trong no_grad, grad được bật: False
trong no_grad, m.requires_grad: False
trong no_grad, m.grad: None
grad được bật sau: True
trong inference_mode: True
inference_mode sau: False
NO_GRAD VÀ INFERENCE_MODE OK
```

## 4. Tensor leaf so với non-leaf

Một **leaf** là một tensor do người dùng tạo ra với `requires_grad=True` (một tham số, một input). Một **non-leaf** là đầu ra của một phép toán. Gradient tích lũy vào `.grad` của các leaf. Trong PyTorch, `.grad` của một non-leaf giữ `None` trừ khi `retain_grad()` được gọi trên nó — truy cập nó thậm chí còn cảnh báo. **LiteTorch khác: `.grad` của một non-leaf được điền sau backward mà không cần gọi `retain_grad()`.** Giá trị nó giữ là gradient hạt giống chảy vào nút đó (1.0 với đầu ra vô hướng đã sum), không phải gradient theo bất kỳ thứ gì khác.

<!-- ex: id=E2-5 device=cpu backend=cpu runtime=~60s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

x = lt.tensor([2.0])
x.requires_grad = True
y = x * x
print("x.grad trước backward:", x.grad)
print("y.grad trước backward:", y.grad)
y.backward()
print("x.grad sau backward:", x.grad.to_vector())
print("y.grad sau backward (không retain):", y.grad.to_vector())
z = lt.tensor([2.0])
z.requires_grad = True
w = z * z
w.retain_grad()
w.backward()
print("w.grad với retain_grad:", w.grad.to_vector())
print("z.grad:", z.grad.to_vector())
xt = torch.tensor([2.0], requires_grad=True)
yt = xt * xt
yt.backward()
print("torch y.grad sau backward (không retain):", yt.grad)
assert y.grad.to_vector() == [1.0]
assert w.grad.to_vector() == [1.0]
print("GRAD CỦA NON-LEAF ĐƯỢC ĐIỀN MÀ KHÔNG CẦN RETAIN_GRAD (KHÁC TORCH, xem phần giải thích)")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. Sự khác biệt chính là điểm cần chứng minh: LiteTorch điền nó, torch trả về `None`.

Kết quả mong đợi (stdout; `UserWarning` của torch về truy cập `.grad` của non-leaf đi ra stderr):

```
x.grad trước backward: None
y.grad trước backward: None
x.grad sau backward: [4.0]
y.grad sau backward (không retain): [1.0]
w.grad với retain_grad: [1.0]
z.grad: [4.0]
torch y.grad sau backward (không retain): None
GRAD CỦA NON-LEAF ĐƯỢC ĐIỀN MÀ KHÔNG CẦN RETAIN_GRAD (KHÁC TORCH, xem phần giải thích)
```

`retain_grad()` tồn tại và hoạt động, nhưng trên bản build này nó không thay đổi gì quan sát được: gradient của non-leaf có ở đó dù sao đi nữa. Đừng viết code phụ thuộc vào việc `.grad` của một non-leaf là `None` — giả định đó đúng trong torch và sai ở đây. Ngược lại, đừng cho rằng `retain_grad()` là bắt buộc trước khi đọc gradient của một nút trung gian; trong LiteTorch thì không.

## 5. Tích lũy gradient và `zero_grad`

`.backward()` **cộng dồn** vào `.grad` thay vì thay thế nó. Hai lần backward trên hai forward pass mới sẽ nhân đôi gradient. Đó là mẫu vòng lặp huấn luyện chuẩn: forward, backward, step, xóa gradient, lặp lại. Quên bước xóa sẽ thầm lặng huấn luyện trên các gradient cũ đã tích lũy.

<!-- ex: id=E2-6 device=cpu backend=cpu runtime=~60s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

x = lt.tensor([2.0, 3.0])
x.requires_grad = True
(x * x).sum().backward()
print("sau bước 1:", x.grad.to_vector())
(x * x).sum().backward()
print("sau bước 2 (đã tích lũy):", x.grad.to_vector())
x.zero_grad()
print("sau zero_grad:", x.grad)
xt = torch.tensor([2.0, 3.0], requires_grad=True)
(xt * xt).sum().backward()
(xt * xt).sum().backward()
print("torch sau bước 2:", xt.grad.tolist())
xt.grad.zero_()
print("torch sau zero_:", xt.grad.tolist())
assert x.grad is None
print("TÍCH LŨY OK; zero_grad KHÁC TORCH (None so với zeros, xem phần giải thích)")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. Tích lũy khớp với torch; bước xóa thì không.

Kết quả mong đợi:

```
sau bước 1: [4.0, 6.0]
sau bước 2 (đã tích lũy): [8.0, 12.0]
sau zero_grad: None
torch sau bước 2: [8.0, 12.0]
torch sau zero_: [0.0, 0.0]
TÍCH LŨY OK; zero_grad KHÁC TORCH (None so với zeros, xem phần giải thích)
```

**Khác biệt:** `zero_grad()` của LiteTorch đặt `.grad` thành `None`; torch để lại một tensor zeros trong `.grad`. Code đọc `.grad` sau khi xóa phải xử lý `None` — `x.grad.to_vector()` sau `zero_grad()` sẽ ném lỗi. Lần `backward()` tiếp theo tạo lại `.grad` từ `None`, nên các vòng lặp huấn luyện vẫn chạy; chỉ những lần đọc trực tiếp một gradient vừa được xóa mới hỏng.

## 6. Một backward cho mỗi đồ thị: không có `retain_graph`

`backward()` giải phóng đồ thị. Gọi nó lần thứ hai trên cùng một đầu ra sẽ ném lỗi `Autograd::backward: graph was already freed by a previous backward() call; pass retain_graph=true to run backward multiple times`. Thông điệp lỗi đặt tên cho một tham số không tồn tại: `backward()` chỉ nhận `(gradient=None, create_graph=False)` — LiteTorch không có `retain_graph`. PyTorch chấp nhận `retain_graph=True` và chạy lại cùng đồ thị.

<!-- ex: id=E2-7 device=cpu backend=cpu runtime=~60s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

x = lt.tensor([2.0, 3.0])
x.requires_grad = True
y = (x * x).sum()
y.backward()
print("backward lần 1 ok, x.grad:", x.grad.to_vector())
try:
    y.backward()
    print("second backward ok")
except RuntimeError as e:
    print("backward lần 2 ném lỗi:", e)
print("chữ ký backward:", lt.Tensor.backward.__doc__)
xt = torch.tensor([2.0, 3.0], requires_grad=True)
yt = (xt * xt).sum()
yt.backward(retain_graph=True)
yt.backward()
print("torch backward lần 2 với retain_graph ok, grad:", xt.grad.tolist())
print("LITETORCH KHÔNG CÓ RETAIN_GRAPH (xem phần giải thích)")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1. Khác biệt chính là điểm cần chứng minh: LiteTorch ném lỗi ở chỗ torch chạy lại.

Kết quả mong đợi:

```
backward lần 1 ok, x.grad: [4.0, 6.0]
backward lần 2 ném lỗi: [litetorch Error] Autograd::backward: graph was already freed by a previous backward() call; pass retain_graph=true to run backward multiple times
chữ ký backward: backward(self: litetorch.Tensor, gradient: litetorch.Tensor = None, create_graph: bool = False) -> None

torch backward lần 2 với retain_graph ok, grad: [8.0, 12.0]
LITETORCH KHÔNG CÓ RETAIN_GRAPH (xem phần giải thích)
```

Cách xử lý mang tính cấu trúc, không phải một cờ: chạy lại forward pass để xây một đồ thị mới (đó cũng là cách tích lũy trong §5 hoạt động), hoặc dùng `create_graph=True` khi mục tiêu là đạo hàm bậc cao thay vì một backward lặp lại. Chính thông điệp lỗi gây hiểu lầm và được ghi lại ở đây để không ai đi tìm một đối số `retain_graph` mà binding không chấp nhận.

## 7. Bài học no-learn: parity forward là chưa đủ

Tháng 10 năm 2026, một lần huấn luyện CIFAR đã không học được gì: độ chính xác kẹt ở 0.1000, loss thành NaN, và bước Adam đầu tiên thành NaN. Forward pass khớp với PyTorch — độ chính xác copy-eval khoảng 0.65 trên cả hai framework, giá trị forward từng op nằm trong dung sai. Các gradient đã hỏng. Một backward của `Conv2d` đã ghi ra ngoài biên vào một bộ đệm gradient kề bên, xóa và xáo trộn các gradient trong khi mọi giá trị forward vẫn nguyên vẹn. Nguyên nhân gốc là một va chạm `cached_native_kernel`: các typedef backward FP32 và FP16 giống nhau từng byte, chia sẻ một mục cache tĩnh, và mọi lần gọi "filter-gradient" đã thầm lặng chạy kernel data-gradient thay thế. Bản sửa (tra cứu kernel theo tên) đã merge trong `88a4ba1` sau khi QA phê duyệt (qa-r51) và một lần chạy kiểm chứng trên Kaggle: độ chính xác CIFAR 0.6154 so với torch 0.6433, gradcheck 10/10, không còn gradient hỏng.

Quy tắc rút ra từ đó: **so sánh gradient, không chỉ giá trị forward.** Mọi ví dụ trong cuốn sách này gọi `.backward()` đều so sánh `.grad` với torch ở 1e-4, vì lỗi no-learn chính là một hỏng hóc ở đường gradient với các giá trị forward trông vẫn đúng. Ví dụ dưới đây là mẫu đó — một mô hình tuyến tính nhỏ, forward và cả ba gradient đều được so sánh:

<!-- ex: id=E2-8 device=cpu backend=cpu runtime=~60s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
assert lt.build_commit() == "88a4ba1"

torch.manual_seed(0)
x_t = torch.randn(4, 3, requires_grad=True)
w_t = torch.randn(3, 2, requires_grad=True)
b_t = torch.randn(2, requires_grad=True)
y_t = torch.nn.functional.linear(x_t, w_t.t(), b_t)
loss_t = (y_t * y_t).sum()
loss_t.backward()
x_l = lt.tensor(x_t.detach().numpy().tolist())
w_l = lt.tensor(w_t.detach().numpy().tolist())
b_l = lt.tensor(b_t.detach().numpy().tolist())
x_l.requires_grad = True
w_l.requires_grad = True
b_l.requires_grad = True
y_l = lt.Ops.matmul(x_l, w_l) + b_l
loss_l = (y_l * y_l).sum()
loss_l.backward()
def maxdiff(a, b):
    return max(abs(p - q) for p, q in zip(a, b))
print("sai khác lớn nhất forward:", maxdiff(y_l.to_vector(), y_t.detach().numpy().ravel().tolist()))
print("sai khác lớn nhất x.grad:", maxdiff(x_l.grad.to_vector(), x_t.grad.numpy().ravel().tolist()))
print("sai khác lớn nhất w.grad:", maxdiff(w_l.grad.to_vector(), w_t.grad.numpy().ravel().tolist()))
print("sai khác lớn nhất b.grad:", maxdiff(b_l.grad.to_vector(), b_t.grad.numpy().ravel().tolist()))
assert maxdiff(y_l.to_vector(), y_t.detach().numpy().ravel().tolist()) <= 1e-4
assert maxdiff(x_l.grad.to_vector(), x_t.grad.numpy().ravel().tolist()) <= 1e-4
assert maxdiff(w_l.grad.to_vector(), w_t.grad.numpy().ravel().tolist()) <= 1e-4
assert maxdiff(b_l.grad.to_vector(), b_t.grad.numpy().ravel().tolist()) <= 1e-4
print("PARITY FORWARD VÀ GRAD OK")
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4 cho forward và mọi gradient.

Kết quả mong đợi:

```
sai khác lớn nhất forward: 1.1920928955078125e-07
sai khác lớn nhất x.grad: 4.76837158203125e-07
sai khác lớn nhất w.grad: 9.5367431640625e-07
sai khác lớn nhất b.grad: 2.384185791015625e-07
PARITY FORWARD VÀ GRAD OK
```

Cả bốn sai khác đều nằm gần 1e-7, sâu hơn hai bậc so với tiêu chí 1e-4. Nếu bất kỳ sai khác nào trong số đó thất bại trong khi sai khác forward vẫn đạt, đó sẽ là dấu hiệu của no-learn: một lỗi backward ẩn sau một forward đúng. Đó là lý do phương pháp kiểm chứng (`plan/verification.md` §3) yêu cầu so sánh gradient cho mọi ví dụ gọi `.backward()` — parity chỉ-forward là không đủ.

## 8. Những gì trang này không bao quát

- Bố cục tensor, kiểu dữ liệu, thiết bị và allocator — xem [E1 Mô Hình Tensor và Bộ Nhớ](E1-tensor-memory-model.md).
- Các optimizer, vốn tiêu thụ `.grad` — xem tài liệu tham khảo về optimizer và tutorial T2.
- Toàn bộ bề mặt API của autograd — xem `reference/cuda-tpu-autograd.md`.
