# How-To H7 — Data loading ở quy mô lớn

`DataLoader` tự batch, shuffle, prefetch, và checkpoint state của
chính nó. Trang này trình bày ba điều quan trọng trong thực tế: batch
tái lập được, một dataset tùy chỉnh, và resume từ state loader đã lưu.
Mọi ví dụ dưới đây chạy trên CPU với LiteTorch 88a4ba1.

Điều kiện tiên quyết: LiteTorch đã được cài và import được.

> Các khác biệt với PyTorch liên quan ở đây: D-04 (ví dụ §11.3 cũ
> không standalone — mọi ví dụ dưới đây đều inline setup của mình),
> D-06 (`.long()` trả về INT32 — cast sang INT64 cho các tensor index).

## H7.1 — Một DataLoader standalone có checkpoint và resume

Mọi thứ trong một block: dataset, loader, lặp, lưu state loader xuống
đĩa, và resume từ nó. `save_state` / `load_state` cho phép một job
huấn luyện khởi động lại giữa epoch mà không cần nạp lại các batch đã
tiêu thụ.

<!-- ex: id=H7.1-1 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import os

import litetorch as lt

device = lt.auto_device()
features = lt.Tensor.from_vector([float(i) * 0.1 for i in range(400)], [40, 10], device, False)
labels = lt.Tensor.from_vector([float(i % 2) for i in range(40)], [40], device, False)

dataset = lt.data.TensorDataset(features, labels)
loader = lt.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    device=device,
    prefetch_limit=2,
    num_workers=0,
    drop_last=False,
)

batches = []
for _ in range(5):
    batches.append(loader.next())
print("batches:", len(batches), "first batch shape:", list(batches[0][0].shape))

os.makedirs("/tmp/litetorch-dataloader-demo", exist_ok=True)
loader.save_state("/tmp/litetorch-dataloader-demo/loader_state.json")

loader2 = lt.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    device=device,
    prefetch_limit=2,
    num_workers=0,
    drop_last=False,
)
loader2.load_state("/tmp/litetorch-dataloader-demo/loader_state.json")

resumed = loader2.next()
print("resumed batch first row:", resumed[0].to_vector()[:3])
print("original next batch first row:", loader.next()[0].to_vector()[:3])
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: 5 batch
shape `[4, 10]`, và batch tiếp theo của loader đã resume khớp chính xác
với batch tiếp theo của loader gốc (cùng hàng đầu tiên). Checkpoint chỉ
ghi lại vị trí của loader — dataset tự nó được dựng lại từ cùng các
tensor khi resume, nên hãy giữ cách dựng dataset giống hệt.

Một ghi chú về tính tái lập, đã đo hai lần trong lúc viết lại này:
shuffle của `DataLoader` **không** được seed bởi `lt.manual_seed(42)`.
Hai lần chạy với cùng seed cho thứ tự batch khác nhau khi
`shuffle=True`. Để có số liệu tái lập trong tài liệu, dùng
`shuffle=False` như trên; đối với huấn luyện thực, hãy coi thứ tự
shuffle là không xác định.

## H7.2 — Một dataset tùy chỉnh

Subclass `lt.data.Dataset` và implement `size()` và
`get(index)` trả về cặp tensor `(features, label)`. Trampoline
dispatch tới đúng hai tên phương thức đó — không phải `__len__` /
`__getitem__` — nên hãy định nghĩa `size` và `get` đúng như cách viết
đó. (`len(dataset)` và `dataset[i]` vẫn dùng được: chúng được bind tới
cùng các phương thức C++.) `TensorDataset`
(H7.1) bao quát trường hợp thông thường; subclass khi dữ liệu được sinh
ra, biến đổi, hoặc stream thay vì đã là tensor sẵn.

<!-- ex: id=H7.2-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()


class SquaresDataset(lt.data.Dataset):
    def __init__(self, n, device):
        super().__init__()
        self.n = n
        self.device = device

    def size(self):
        return self.n

    def get(self, idx):
        x = lt.Tensor.from_vector([float(idx)] * 4, [4], self.device, False)
        y = lt.Tensor.from_vector([float(idx * idx)], [1], self.device, False)
        return x, y


dataset = SquaresDataset(10, device)
loader = lt.data.DataLoader(dataset, batch_size=2, shuffle=False, device=device)
first_batch = loader.next()
print("first batch x shape:", list(first_batch[0].shape))
print("first batch y:", first_batch[1].to_vector())
del loader
del dataset
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: batch đầu
`x` shape `[2, 4]`, giá trị `y` là `[0.0, 1.0]` (bình phương của index
0 và 1). Giữ `get` rẻ: nó chạy một lần cho mỗi sample, trên
thread gọi loader khi `num_workers=0`. Lệnh `del loader` /
`del dataset` tường minh ở cuối là bắt buộc, không phải style: một
`Dataset` Python-backed còn bị C++ loader giữ sẽ abort interpreter lúc
shutdown (`FATAL: exception not rethrown`, SIGABRT) nếu để garbage
collection tự dọn. Hãy delete cả hai tường minh trước khi thoát. Đây
là một defect đã biết trong trampoline `Dataset`, đã gửi Core để
triage — `TensorDataset` (H7.1) không cần workaround này.

## H7.3 — Lưu và load một mô hình với định dạng .lt

`.lt` là định dạng checkpoint native của LiteTorch. Lưu mô hình và
optimizer cùng nhau; khi load, khôi phục cả hai trước khi resume huấn
luyện, không optimizer sẽ khởi động lại từ trạng thái ban đầu và các
bước đầu tiên sau resume sẽ sai.

<!-- ex: id=H7.3-1 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import os

import litetorch as lt

device = lt.auto_device()
model = lt.nn.Sequential([
    lt.nn.Linear(4, 8, True),
    lt.nn.GELU(),
    lt.nn.Linear(8, 2, True),
])
model.to(device)

optimizer = lt.optim.Adam(model.parameters(), 0.001)

x = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4], [1, 4], device, True)
y = model(x)
loss = lt.Ops.sum(y)
loss.backward()
optimizer.step()

os.makedirs("/tmp/litetorch-checkpoint-demo", exist_ok=True)
lt.save(model, "/tmp/litetorch-checkpoint-demo/model.lt")
lt.save(optimizer, "/tmp/litetorch-checkpoint-demo/optimizer.lt")

model2 = lt.nn.Sequential([
    lt.nn.Linear(4, 8, True),
    lt.nn.GELU(),
    lt.nn.Linear(8, 2, True),
])
model2.to(device)
optimizer2 = lt.optim.Adam(model2.parameters(), 0.001)
lt.load(model2, "/tmp/litetorch-checkpoint-demo/model.lt")
lt.load(optimizer2, "/tmp/litetorch-checkpoint-demo/optimizer.lt")

x2 = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4], [1, 4], device, True)
print("original output:", model(x2).to_vector())
print("restored output:", model2(x2).to_vector())
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: mô hình
đã khôi phục in ra output giống hệt mô hình gốc (cùng trọng số, cùng
input). `lt.save` / `lt.load` nhận một `Module`, một list tham số,
hoặc một `Optimizer` — không phải một state dict. Lưu optimizer bằng
cùng lời gọi (`lt.save(optimizer, path)`), và giữ các counter
epoch/step trong một file JSON sidecar; một optimizer được khôi phục
từ trạng thái ban đầu làm cho các bước đầu tiên sau resume sai.
`state_dict()` / `load_state_dict(sd)` cũng tồn tại, để chuyển trọng
số giữa hai mô hình đang sống mà không chạm đĩa.
