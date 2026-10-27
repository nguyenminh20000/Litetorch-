# Hướng Dẫn Bắt Đầu Nhanh (Quickstart)

Tài liệu này là lối vào 60 phút của LiteTorch. Trong khoảng một giờ, bạn sẽ tạo tensor, tính đạo hàm qua chúng, xây dựng một module mạng nơ-ron, nạp dữ liệu theo batch, huấn luyện mô hình, và lưu một checkpoint. Mọi ví dụ đều tự chứa và chạy được: hãy sao chép một block, chạy nó, và so sánh kết quả của bạn với kết quả kỳ vọng in bên dưới.

Điều kiện tiên quyết: một bản cài đặt đang hoạt động ([Cài đặt](installation.md)). Mọi ví dụ dưới đây chạy trên CPU; trên máy GPU, cùng đoạn mã sẽ chạy trên GPU qua `lt.auto_device()`.

Một lưu ý về tính tái lập: ví dụ huấn luyện dùng `lt.manual_seed(42)` để gieo RNG và chạy DataLoader với `shuffle=False`, nên các giá trị loss dưới đây là chính xác. Trong huấn luyện thực tế hãy dùng `shuffle=True`; các giá trị từng epoch khi đó sẽ khác nhau giữa các lần chạy (việc xáo trộn của DataLoader dùng RNG riêng mà `lt.manual_seed` không điều khiển), nhưng loss vẫn giảm.

---

## 3.1. Tensor và đạo hàm tự động

Một `Tensor` chứa dữ liệu trên một thiết bị; `requires_grad=True` yêu cầu đạo hàm tự động ghi lại các phép toán để `.backward()` có thể tính gradient.

<!-- ex: id=3.1-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

```python
import litetorch as lt

device = lt.auto_device()

a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, True)
b = lt.Tensor.from_vector([0.5, 1.5, -1.0, 2.0], [2, 2], device, True)

c = lt.Ops.matmul(a, b)
loss = lt.Ops.sum(c)

loss.backward()

print("Tich ma tran c:", c.to_vector())
print("Gia tri loss:", loss.item())
print("Dao ham da/dc:", a.grad.to_vector())
print("Dao ham db/dc:", b.grad.to_vector())
```

Đã kiểm tra trên CPU, LiteTorch 88a4ba1 (parity với torch đã kiểm tra 2026-10-27 so với torch 2.14.1+cpu: giá trị, loss và gradient khớp chính xác, max diff 0.0).

Kết quả kỳ vọng:

```
Tich ma tran c: [-1.5, 5.5, -2.5, 12.5]
Gia tri loss: 14.0
Dao ham da/dc: [2.0, 1.0, 2.0, 1.0]
Dao ham db/dc: [4.0, 4.0, 6.0, 6.0]
```

Chuyện gì đã xảy ra: `c = a @ b`, `loss = sum(c) = 14.0`. Gradient của loss theo `a` là ma trận gồm tổng hàng của `b` được broadcast theo cột, và theo `b` là tổng cột của `a` được broadcast theo hàng. Nếu bạn thay đổi `b`, cả loss và gradient đều thay đổi — hãy thử.

---

## 3.2. Xây dựng module mạng nơ-ron

Hãy tạo lớp kế thừa từ `lt.nn.Module` để xây dựng một kiến trúc mô-đun. Một module sở hữu các tham số của nó, di chuyển chúng bằng `.to(device)`, và tính toán đầu ra trong `.forward(x)`.

<!-- ex: id=3.2-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

```python
import litetorch as lt

class MultiLayerPerceptron(lt.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = lt.nn.Linear(in_features, hidden_features, True)
        self.relu = lt.nn.ReLU()
        self.fc2 = lt.nn.Linear(hidden_features, out_features, True)

    def forward(self, x):
        h = self.fc1.forward(x)
        h = self.relu.forward(h)
        out = self.fc2.forward(h)
        return out

    def parameters(self):
        params = []
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def to(self, device):
        self.fc1.to(device)
        self.fc2.to(device)

model = MultiLayerPerceptron(64, 128, 10)
print("So luong tham so:", len(model.parameters()))
```

Đã kiểm tra trên CPU, LiteTorch 88a4ba1 (parity với torch đã kiểm tra 2026-10-27: số lượng tham số khớp với torch, 4 == 4).

Kết quả kỳ vọng:

```
So luong tham so: 4
```

Bốn tensor tham số: trọng số và bias của mỗi lớp `Linear` trong hai lớp. `ReLU` không có tham số.

Hoặc ghép nối các lớp bằng `lt.nn.Sequential` — cùng mô hình đó trong bốn dòng:

<!-- ex: id=3.2-2 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

```python
import litetorch as lt

device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Linear(64, 128, True),
    lt.nn.ReLU(),
    lt.nn.Dropout(0.1),
    lt.nn.Linear(128, 10, True)
])

model.to(device)
print("So luong tham so:", len(model.parameters()))
```

Đã kiểm tra trên CPU, LiteTorch 88a4ba1 (parity với torch đã kiểm tra 2026-10-27: số lượng tham số khớp với torch, 4 == 4).

Kết quả kỳ vọng:

```
So luong tham so: 4
```

`Dropout` không có tham số, và nó chỉ hoạt động ở chế độ huấn luyện — ở chế độ suy luận nó không làm gì. Phần còn lại của tài liệu này dùng `Sequential` cho ngắn gọn; các trang tham chiếu ghi lại đầy đủ mọi class module.

---

## 3.3. Quản lý tập dữ liệu với TensorDataset và DataLoader

`TensorDataset` ghép cặp tensor đặc trưng và tensor nhãn; `DataLoader` chia chúng thành batch, xáo trộn, và chuyển batch sang thiết bị.

<!-- ex: id=3.3-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

```python
import litetorch as lt

device = lt.auto_device()

x_data = [float(i % 10) for i in range(100 * 64)]
y_data = [float(i % 2) for i in range(100 * 1)]

x_tensor = lt.Tensor.from_vector(x_data, [100, 64], device, False)
y_tensor = lt.Tensor.from_vector(y_data, [100, 1], device, False)

dataset = lt.data.TensorDataset(x_tensor, y_tensor)
loader = lt.data.DataLoader(dataset, batch_size=16, shuffle=True, device=device)

loader.reset()
batch = loader.next()
bx, by = batch
print("Kich thuoc batch x:", bx.shape, "Kich thuoc batch y:", by.shape)
```

Đã kiểm tra trên CPU, LiteTorch 88a4ba1 (parity với torch đã kiểm tra 2026-10-27: shape của batch đầu tiên khớp với torch, [16, 64] / [16, 1]).

Kết quả kỳ vọng:

```
Kich thuoc batch x: [16, 64] Kich thuoc batch y: [16, 1]
```

`loader.next()` trả về `None` khi epoch đã hết; hãy gọi `loader.reset()` để bắt đầu epoch mới. Với `shuffle=True`, thứ tự batch khác nhau mỗi epoch.

---

## 3.4. Vòng lặp huấn luyện hoàn chỉnh

Ví dụ này huấn luyện mô hình nhỏ ở §3.2 trên bốn mẫu tổng hợp trong 10 epoch và in ra loss trung bình mỗi epoch. Loss phải giảm — nếu không giảm, vấn đề nằm ở cách thiết lập, không phải ở mô hình.

<!-- ex: id=3.4-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

```python
import litetorch as lt

lt.manual_seed(42)
device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Linear(4, 8, True),
    lt.nn.ReLU(),
    lt.nn.Linear(8, 2, True)
])
model.to(device)

optimizer = lt.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
loss_fn = lt.nn.MSELoss()

x_train = lt.Tensor.from_vector([
    1.0, 2.0, 3.0, 4.0,
    2.0, 3.0, 4.0, 5.0,
    5.0, 6.0, 7.0, 8.0,
    8.0, 7.0, 6.0, 5.0
], [4, 4], device, False)

y_train = lt.Tensor.from_vector([
    1.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 1.0
], [4, 2], device, False)

dataset = lt.data.TensorDataset(x_train, y_train)
loader = lt.data.DataLoader(dataset, batch_size=2, shuffle=False, device=device)

for epoch in range(10):
    loader.reset()
    total_loss = 0.0
    steps = 0
    while True:
        batch = loader.next()
        if batch is None:
            break
        bx, by = batch

        optimizer.zero_grad()
        preds = model.forward(bx)
        loss = loss_fn.forward(preds, by)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    avg_loss = total_loss / max(steps, 1)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
```

Đã kiểm tra trên CPU, LiteTorch 88a4ba1 (parity với torch đã kiểm tra 2026-10-27 so với torch 2.14.1+cpu: quỹ đạo AdamW 10 epoch khớp với cùng trọng số khởi tạo, max diff theo epoch 1.38e-07).

Kết quả kỳ vọng (chính xác với `lt.manual_seed(42)` và `shuffle=False` — việc xáo trộn của DataLoader dùng RNG riêng của nó, nên với `shuffle=True` các giá trị từng epoch sẽ khác nhau giữa các lần chạy trong khi loss vẫn giảm):

```
Epoch 1, Loss: 1.0306
Epoch 2, Loss: 0.7767
Epoch 3, Loss: 0.5915
Epoch 4, Loss: 0.4568
Epoch 5, Loss: 0.3630
Epoch 6, Loss: 0.3002
Epoch 7, Loss: 0.2597
Epoch 8, Loss: 0.2345
Epoch 9, Loss: 0.2197
Epoch 10, Loss: 0.2142
```

Loss giảm từ 1.03 xuống 0.21 trong 10 epoch. Ba lời gọi phải diễn ra theo đúng thứ tự này mỗi bước là `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` — quên `zero_grad()` sẽ khiến gradient tích lũy qua các bước và loss sẽ không hoạt động như kỳ vọng.

---

## 3.5. Lưu trữ và khôi phục mô hình cùng optimizer

Hãy lưu một mô hình và optimizer của nó, rồi nạp chúng vào các instance mới và xác nhận đầu ra giống hệt nhau. Block này tự chứa: nó xây dựng, huấn luyện một bước, lưu, nạp lại, và so sánh.

<!-- ex: id=3.5-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

```python
import litetorch as lt

lt.manual_seed(7)
device = lt.auto_device()

def build_model():
    m = lt.nn.Sequential([
        lt.nn.Linear(4, 8, True),
        lt.nn.ReLU(),
        lt.nn.Linear(8, 2, True)
    ])
    m.to(device)
    return m

model = build_model()
optimizer = lt.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
loss_fn = lt.nn.MSELoss()

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [1, 4], device, False)
y = lt.Tensor.from_vector([1.0, 0.0], [1, 2], device, False)

optimizer.zero_grad()
loss = loss_fn.forward(model.forward(x), y)
loss.backward()
optimizer.step()
print("Loss truoc khi luu:", loss.item())

before = model.forward(x).to_vector()
lt.save(model, "model_checkpoint.lt")
lt.save(optimizer, "optim_checkpoint.lt")

model2 = build_model()
optimizer2 = lt.optim.AdamW(model2.parameters(), lr=0.01, weight_decay=0.001)
lt.load(model2, "model_checkpoint.lt")
lt.load(optimizer2, "optim_checkpoint.lt")

after = model2.forward(x).to_vector()
print("Dau ra khop nhau:", before == after)
```

Đã kiểm tra trên CPU, LiteTorch 88a4ba1 (parity với torch đã kiểm tra 2026-10-27 so với torch 2.14.1+cpu: loss một bước giống hệt từng bit, 0.21786627173423767; vòng lặp save/load chính xác ở cả hai).

Kết quả kỳ vọng (chính xác với `lt.manual_seed(7)`):

```
Loss truoc khi luu: 0.21786627173423767
Dau ra khop nhau: True
```

Các file checkpoint `model_checkpoint.lt` và `optim_checkpoint.lt` được ghi vào thư mục làm việc hiện tại. Việc nạp vào một mô hình mới xây dựng với cùng kiến trúc sẽ khôi phục chính xác các trọng số đã huấn luyện — `before == after` phải là `True`.

---

## Đọc tiếp ở đâu

- **Tensor và đạo hàm tự động chuyên sâu**: Explanations — tensor và đạo hàm tự động (đang viết).
- **Mọi op và module**: API reference (đang viết) — bắt đầu với class Tensor và các module `nn`.
- **Huấn luyện một mô hình phân loại thật**: Tutorial T2 (đang viết) — MNIST / CIFAR với độ chính xác kỳ vọng.
- **Huấn luyện trên 2 GPU**: How-to H1 (đang viết).
- **Gặp lỗi**: trang Xử lý sự cố và FAQ (đang viết).
- **Đến từ PyTorch**: Hướng dẫn migration từ PyTorch (đang viết).
