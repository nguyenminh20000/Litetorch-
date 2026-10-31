# Hướng dẫn T2 — Huấn luyện một bộ phân loại

Huấn luyện một MLP nhỏ để phân tách hai lớp, từ đầu đến cuối: xây dựng
dữ liệu, nạp theo batch, huấn luyện với hàm loss cross-entropy, và đo độ
chính xác. Dữ liệu ở đây là tổng hợp (hai đám Gaussian, đã gieo hạt) nên
hướng dẫn chạy được ở mọi nơi trên CPU trong vài giây; cùng vòng lặp đó
huấn luyện được một mô hình MNIST/CIFAR thật khi bạn thay bằng tập dữ
liệu thật.

Bạn sẽ học được:

- Cách xây dựng một `TensorDataset` và duyệt nó bằng `DataLoader`.
- Cách huấn luyện một bộ phân loại với `CrossEntropyLoss` và `AdamW`.
- Cách đo độ chính xác và đọc đường cong loss.
- Cách chuyển cùng đoạn code sang GPU.

Điều kiện tiên quyết: Hướng dẫn T1. Mọi ví dụ chạy trên CPU với
LiteTorch 88a4ba1; mỗi ví dụ đều ghi rõ kết quả đối chiếu với PyTorch
bên dưới.

> Các điểm khác biệt với PyTorch trong hướng dẫn này: D-12 (dùng
> `lt.Ops.sum(t, [0, 1])` cho reduction nhiều chiều — `Tensor.sum` chỉ
> nhận một dim nguyên), D-09 (GELU chỉ có dạng xấp xỉ tanh; đối chiếu với
> `F.gelu(x, approximate='tanh')`).

## T2.1 — Xây dựng dữ liệu

Hai lớp, mỗi lớp bốn đặc trưng. Lớp 0 có tâm tại 0, lớp 1 tại 1.5, cả
hai đều có nhiễu Gaussian đơn vị. Đã gieo hạt bằng `random.seed(0)` nên
tập dữ liệu giống nhau trên mọi lần chạy.

<!-- ex: id=T2.1-1 device=cpu backend=cpu runtime=~0.1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import random
import litetorch as lt

device = lt.auto_device()

random.seed(0)
n = 200
xs = []
ys = []
for i in range(n):
    c = i % 2
    xs.extend([random.gauss(c * 1.5, 1.0) for _ in range(4)])
    ys.append(float(c))

x = lt.Tensor.from_vector(xs, [n, 4], device, False)
y = lt.Tensor.from_vector(ys, [n], device, False)

dataset = lt.data.TensorDataset(x, y)
loader = lt.data.DataLoader(dataset, batch_size=16, shuffle=False, device=device)

print("kích thước tập dữ liệu:", dataset.size())
print("số batch mỗi epoch:", len(loader))

loader.reset()
bx, by = loader.next()
print("shape batch x đầu tiên:", bx.shape, "shape y:", by.shape)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: 200 mẫu, 13 batch mỗi epoch
(12 batch đầy 16 mẫu cộng một batch cuối 8 mẫu), shape batch đầu tiên
`[16, 4]` và `[16]`.

## T2.2 — Huấn luyện bộ phân loại

MLP `4 -> 16 -> 2` với ReLU, `CrossEntropyLoss`, `AdamW` với tốc độ
học 0.05, 15 epoch. Đã gieo hạt bằng `lt.manual_seed(7)` nên khởi tạo
có thể lặp lại, và loader chạy với `shuffle=False` nên các loss in ra
có thể tái tạo qua các lần chạy. Loss trung bình giảm từ 0.4016 ở epoch
1 xuống 0.1075 ở epoch 15, và độ chính xác trên tập huấn luyện đạt
0.96.

<!-- ex: id=T2.2-1 device=cpu backend=cpu runtime=~0.7s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python

import random
import litetorch as lt

lt.manual_seed(7)
random.seed(0)
device = lt.auto_device()

n = 200
xs = []
ys = []
for i in range(n):
    c = i % 2
    xs.extend([random.gauss(c * 1.5, 1.0) for _ in range(4)])
    ys.append(float(c))

x = lt.Tensor.from_vector(xs, [n, 4], device, False)
y = lt.Tensor.from_vector(ys, [n], device, False)

dataset = lt.data.TensorDataset(x, y)
loader = lt.data.DataLoader(dataset, batch_size=16, shuffle=False, device=device)

model = lt.nn.Sequential([
    lt.nn.Linear(4, 16, True),
    lt.nn.ReLU(),
    lt.nn.Linear(16, 2, True)
])
model.to(device)

optimizer = lt.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
loss_fn = lt.nn.CrossEntropyLoss()

for epoch in range(15):
    loader.reset()
    total_loss = 0.0
    batches = 0
    while True:
        batch = loader.next()
        if batch is None:
            break
        bx, by = batch
        optimizer.zero_grad()
        logits = model.forward(bx)
        loss = loss_fn.forward(logits, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    print("Epoch %d, Loss: %.4f" % (epoch + 1, total_loss / batches))

logits = model.forward(x)
lv = logits.to_vector()
correct = 0
for i in range(n):
    pred = 0 if lv[2 * i] > lv[2 * i + 1] else 1
    if pred == int(ys[i]):
        correct += 1
print("độ chính xác trên tập huấn luyện:", correct / n)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1: loss cuối 0.1075, độ chính
xác trên tập huấn luyện 0.96. Lần chạy là xác định với các hạt giống
này. Một thứ tự xáo trộn khác hoặc một hạt giống khác cho kết quả gần
đó — loss cuối khoảng 0.10–0.13, độ chính xác khoảng 0.94–0.96 — như đã
đo trên một lần chạy xáo trộn của cùng thiết lập. Không có khẳng định
parity từng điểm với torch: động lực huấn luyện phụ thuộc vào khởi tạo
ngẫu nhiên.

## T2.3 — Kiểm tra hàm loss với PyTorch

Bộ phân loại trên chỉ đáng tin nếu bản thân hàm loss là đúng.
`CrossEntropyLoss` khớp cả giá trị và gradient với
`torch.nn.functional.cross_entropy` ở mức 1e-4 trên cùng đầu vào.

<!-- ex: id=T2.3-1 device=cpu backend=cpu runtime=~4.4s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python

import litetorch as lt
import torch

device = lt.auto_device()

logits_lt = lt.Tensor.from_vector([2.0, 1.0, 0.1, 0.5, 2.5, 0.3], [2, 3], device, True)
target_lt = lt.Tensor.from_vector([0.0, 2.0], [2], device, False)

loss_fn = lt.nn.CrossEntropyLoss()
loss_lt = loss_fn.forward(logits_lt, target_lt)
loss_lt.backward()

logits_t = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], requires_grad=True)
loss_t = torch.nn.functional.cross_entropy(logits_t, torch.tensor([0, 2]))
loss_t.backward()

print("loss lt:", loss_lt.item())
print("loss torch:", loss_t.item())
print("grad lt:", logits_lt.grad.to_vector())
print("grad torch:", logits_t.grad.flatten().tolist())
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4:
loss 1.418540 ở cả hai phía, gradient bằng nhau trong 1e-4.

## T2.4 — Chuyển cùng vòng lặp sang GPU (REQUIRES-GPU)

Không có gì trong vòng lặp thay đổi ngoại trừ thiết bị: xây dựng mô
hình và dữ liệu trên thiết bị GPU rồi chạy đoạn code giống hệt. Đừng
chạy khối này trên máy chỉ có CPU — `.to("cuda")` lặng lẽ rơi về CPU
(D-02), nên khối dưới đây kiểm tra thiết bị và dừng lớn tiếng nếu không
có GPU.

<!-- ex: id=T2.4-1 device=gpu backend=cuda runtime=~? torch-parity=n/a status=verified-2026-10-31@cbba5af3 -->
```python skip-run

import litetorch as lt

gpu = lt.Device("cuda")
device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Linear(4, 16, True),
    lt.nn.ReLU(),
    lt.nn.Linear(16, 2, True)
])
model.to(gpu)
params = model.parameters()
if params[0].device.type != lt.DeviceType.GPU:
    raise RuntimeError("no GPU device: refusing to run the GPU variant on CPU")

print("huấn luyện trên:", params[0].device)
```

REQUIRES-GPU — kiểm chứng trên Kaggle (2x T4) ở B5, không phải trên
CPU. Lần chạy CPU của T2.2 là cùng vòng lặp trên cùng dữ liệu.
