# Hướng dẫn T1 — Blitz 60 phút: Tensor, Autograd, Huấn luyện

Học LiteTorch bằng cách thực hành: tạo tensor, đạo hàm qua chúng, xây dựng
mô hình, huấn luyện nó, lưu lại, và di chuyển nó giữa các thiết bị. Mọi ví
dụ dưới đây đều tự đầy đủ và chạy được trên CPU. Tổng thời gian: khoảng 60
phút.

Bạn sẽ học được:

- Cách tạo tensor và chạy các op trên chúng.
- Cách autograd ghi lại đồ thị và tính gradient.
- Cách xây dựng mô hình từ `lt.nn.Module` và `lt.nn.Sequential`.
- Cách chạy vòng lặp huấn luyện và đọc đường cong loss.
- Cách lưu và nạp lại mô hình, và cách thiết bị và kiểu dữ liệu hoạt động.

Điều kiện tiên quyết: LiteTorch đã được cài đặt (`pip install litetorch`)
và import được. Mọi ví dụ chạy trên CPU với LiteTorch 88a4ba1; mỗi ví dụ
đều ghi rõ kết quả đối chiếu với PyTorch bên dưới.

> Các điểm khác biệt với PyTorch trong hướng dẫn này: D-03 (`.to()` nhận
> một `Device`, không nhận chuỗi), D-04 (các hàm factory chỉ nhận một danh
> sách shape), D-02 (`.to(device)` lặng lẽ rơi về CPU khi thiết bị tăng tốc
> không có sẵn — luôn kiểm tra `t.device` sau đó).

## T1.1 — Tạo các tensor đầu tiên

Một tensor là một mảng n chiều với một shape, một kiểu dữ liệu, và một
thiết bị. Tạo tensor từ một danh sách Python bằng `Tensor.from_vector`,
hoặc dùng một hàm factory. Các hàm factory chỉ nhận một danh sách shape
duy nhất: `lt.randn([2, 3])`, không bao giờ `lt.randn(2, 3)` (D-04).

<!-- ex: id=T1.1-1 device=cpu backend=cpu runtime=~0.1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

device = lt.auto_device()

a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, False)
b = lt.Tensor.from_vector([0.5, 1.5, -1.0, 2.0], [2, 2], device, False)

c = lt.Ops.matmul(a, b)
print("a @ b:", c.to_vector())
print("shape:", c.shape, "kiểu dữ liệu:", c.dtype, "thiết bị:", c.device)

r = lt.randn([2, 3])
print("shape của randn:", r.shape)

z = lt.zeros([2, 2])
print("zeros:", z.to_vector())
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4
(matmul khớp với `torch.matmul` trên cùng đầu vào).

## T1.2 — Autograd trong năm dòng

Đặt `requires_grad=True` khi tạo tensor, chạy các op trên nó, gọi
`.backward()` trên một loss vô hướng, và đọc `.grad`. LiteTorch ghi lại
một DAG dạng tape trong forward pass và duyệt ngược nó.

<!-- ex: id=T1.2-1 device=cpu backend=cpu runtime=~0.1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

device = lt.auto_device()

x = lt.Tensor.from_vector([2.0, 3.0], [2], device, True)
w = lt.Tensor.from_vector([4.0, 5.0], [2], device, True)

y = lt.Ops.mul(x, w)
loss = lt.Ops.sum(y)
loss.backward()

print("y:", y.to_vector())
print("loss:", loss.item())
print("x.grad:", x.grad.to_vector())
print("w.grad:", w.grad.to_vector())
```

Kết quả kỳ vọng: `y = [8, 15]`, `loss = 23`, `x.grad = [4, 5]`,
`w.grad = [2, 3]`. Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch
2.14.1, parity 1e-4 (giá trị và gradient khớp với `torch` trên cùng
đầu vào).

Lưu ý: các op elementwise và shape nằm trong `litetorch.Ops` (ví dụ
`lt.Ops.relu(t)`), không phải là method của `Tensor` — `t.relu()` không
tồn tại (D-17). `t ** 2.0` hoạt động với số mũ vô hướng; không có method
`.pow()` và không có dạng số mũ là tensor (D-05).

## T1.3 — Xây dựng một mô hình

Kế thừa `lt.nn.Module` cho một khối tùy chỉnh, hoặc ghép các lớp bằng
`lt.nn.Sequential`. Một mô hình sở hữu các tham số của nó;
`parameters()` trả về chúng, `to(device)` di chuyển chúng.

<!-- ex: id=T1.3-1 device=cpu backend=cpu runtime=~0.2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
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
        return self.fc2.forward(h)

    def parameters(self):
        params = []
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def to(self, device):
        self.fc1.to(device)
        self.fc2.to(device)


device = lt.auto_device()
model = MultiLayerPerceptron(4, 8, 2)
model.to(device)

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [1, 4], device, False)
print("shape đầu ra của mô hình tùy chỉnh:", model.forward(x).shape)
print("số lượng tham số:", len(model.parameters()))
```

Cùng mạng đó dưới dạng `Sequential` thì ngắn hơn và tương đương:

<!-- ex: id=T1.3-2 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Linear(4, 8, True),
    lt.nn.ReLU(),
    lt.nn.Linear(8, 2, True)
])
model.to(device)

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [1, 4], device, False)
print("shape đầu ra của sequential:", model.forward(x).shape)
print("số lượng tham số:", len(model.parameters()))
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Không có khẳng định parity
với torch: các trọng số được khởi tạo ngẫu nhiên, nên chỉ shape và số
lượng tham số được kiểm tra.

## T1.4 — Huấn luyện nó: vòng lặp huấn luyện

Vòng lặp luôn gồm năm bước giống nhau: xóa gradient, forward, loss,
backward, bước optimizer. Gieo hạt trước bằng `lt.manual_seed` để lần
chạy có thể lặp lại. Loader chạy với `shuffle=False` nên các loss in
ra có thể tái tạo qua các lần chạy; hãy bật `shuffle=True` khi huấn
luyện thật. Với hạt giống và dữ liệu dưới đây, loss trung bình
giảm từ 1.0306 ở epoch 1 xuống 0.2142 ở epoch 10.

<!-- ex: id=T1.4-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
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
    print("Epoch %d, Loss: %.4f" % (epoch + 1, total_loss / max(steps, 1)))
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Loss giảm mỗi epoch
(1.0306, 0.7767, 0.5915, 0.4568, 0.3630, 0.3002, 0.2597, 0.2345, 0.2197,
0.2142). Không có khẳng định parity từng điểm với torch: động lực huấn
luyện phụ thuộc vào khởi tạo ngẫu nhiên, vốn khác nhau giữa các
framework.

## T1.5 — Lưu và nạp lại một checkpoint

`lt.save` ghi một mô hình (hoặc optimizer) ra file `.lt`; `lt.load`
khôi phục nó vào một mô hình mới có cùng shape. Việc nạp lại phải tái
tạo chính xác đầu ra ban đầu.

<!-- ex: id=T1.5-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
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

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [1, 4], device, False)
before = model.forward(x).to_vector()

lt.save(model, "/tmp/tut_model.lt")

model2 = lt.nn.Sequential([
    lt.nn.Linear(4, 8, True),
    lt.nn.ReLU(),
    lt.nn.Linear(8, 2, True)
])
model2.to(device)
lt.load(model2, "/tmp/tut_model.lt")
after = model2.forward(x).to_vector()

print("độ lệch tuyệt đối lớn nhất sau khi nạp lại:", max(abs(a - b) for a, b in zip(before, after)))
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: độ lệch tuyệt đối lớn nhất
bằng đúng 0.0.

## T1.6 — Thiết bị và kiểu dữ liệu

`lt.auto_device()` chọn thiết bị tốt nhất hiện có (CPU ở đây). `.to()`
nhận một đối tượng `Device` — `model.to("cuda")` gây ra TypeError
(D-03). Và nếu thiết bị tăng tốc được yêu cầu không có sẵn, `.to()` sẽ
lặng lẽ rơi về CPU (D-02): luôn kiểm tra `t.device` sau đó.

<!-- ex: id=T1.6-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt

print("thiết bị tự động:", lt.auto_device())
print("cuda có sẵn:", lt.cuda.is_available())

device = lt.auto_device()
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, False)

gpu = lt.Device("cuda")
y = x.to(gpu)
print("yêu cầu cuda, nhận được thiết bị:", y.device)

half = x.cast(lt.DataType.FP16)
print("kiểu dữ liệu sau khi cast:", half.dtype)
print("trở lại FP32:", half.cast(lt.DataType.FP32).dtype)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Trên máy chỉ có CPU này
`lt.cuda.is_available()` là False và yêu cầu `.to("cuda")` rơi về CPU —
đúng trường hợp rơi lặng lẽ của D-02, được trình bày rõ ràng chứ không
giấu đi. Trên máy có GPU cùng đoạn code trả về một tensor GPU; bước
kiểm tra thiết bị sau `.to()` chính là thứ cho bạn biết trường hợp nào
đã xảy ra.
