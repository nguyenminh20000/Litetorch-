# Dữ liệu, Trainer và Serialization — Tài liệu tham khảo API

`litetorch.data` chứa dataset, sampler và `DataLoader`; cấp cao nhất của `litetorch` chứa `Trainer`, `TrainerConfig`, họ hàm `save` / `load`, `AsyncCheckpointSaver` và các hàm RNG.

- **Nguồn xác thực:** `src/bindings/bind_data.cpp`, `include/litetorch/data.h`, `include/litetorch/trainer.h`, `src/serialization/optimizer.cpp`, `src/serialization/parameters.cpp`, `src/serialization/safetensors.cpp`.
- **Cây mã:** agent-v1@02cc271a / HEAD cục bộ 89afd67 (baseline của tài liệu; B5 xác minh lại tại head). Các ví dụ chạy trên extension D3 của Core tại commit 88a4ba1.
- **Quy ước:** `cpu = lt.Device("cpu")` trong mọi ví dụ. Mọi ví dụ đều tự chứa và chạy được trên CPU. Tính tương đương với PyTorch là FP32 1e-4 trừ khi header ghi khác.
- **Điểm khác biệt liên quan:** các lớp dữ liệu nằm dưới `lt.data.*`, không ở cấp cao nhất (`lt.data.TensorDataset`, không phải `lt.TensorDataset`). Lớp con `Dataset` trong Python phải định nghĩa `size()` / `get(idx)` — trampoline điều phối tới hai hàm đó, không tới `__len__` / `__getitem__` — và một loader giữ dataset viết bằng Python (hoặc một `collate_fn` Python) sẽ làm trình thông dịch abort khi tắt trừ khi loader được xóa tường minh (defect H7.2, đã báo cho Core). `lt.save` / `lt.load` là tại chỗ và điều phối theo kiểu; `save_safetensors` / `AsyncCheckpointSaver.save_async` nhận một `dict`, không phải module. `save_optimizer_state` chỉ hỗ trợ Adam, AdamW, SGD, RMSprop.

## `data.Dataset`

- **Chữ ký:** `data.Dataset() -> Dataset`
- **Nguồn:** `src/bindings/bind_data.cpp:8-13`
- **Phương thức:** `size()` — số mục; `get(index)` — trả về tuple `(x, y)` các tensor; `__len__` (bí danh của `size`); `__getitem__` (bí danh của `get`).
- **Kế thừa:** định nghĩa `size()` và `get(idx)` trong lớp con. Chỉ định nghĩa `__len__` / `__getitem__` **không** hoạt động: trampoline C++ điều phối tới `size` / `get`, nên lớp con chỉ định nghĩa dunder sẽ lỗi khi load. `__len__` / `__getitem__` đã bind trên lớp cơ sở gọi `size` / `get`, nên `len(ds)` và `ds[i]` hoạt động khi lớp con đã định nghĩa chúng.
- **Nguy cơ khi tắt (H7.2):** một `DataLoader` giữ `Dataset` viết bằng Python (hoặc một `collate_fn` Python) sẽ làm trình thông dịch abort khi thoát (`FATAL: exception not rethrown`, SIGABRT) trừ khi loader và dataset được xóa tường minh bằng `del loader` / `del dataset` trước khi trình thông dịch tắt. Đã báo cho Core; xem mục `DataLoader` để biết workaround.
- **PyTorch:** `torch.utils.data.Dataset` — lớp con PyTorch định nghĩa `__len__` / `__getitem__`; LiteTorch yêu cầu `size` / `get` thay thế.

<!-- ex: id=r11-dataset device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
class DS(lt.data.Dataset):
    def size(self):
        return 3
    def get(self, idx):
        return (lt.Tensor.from_vector([float(idx)], [1], cpu), lt.Tensor.from_vector([float(idx * 2)], [1], cpu))
ds = DS()
print(len(ds), ds[1][0].to_vector(), ds.get(1)[1].to_vector())
dl = lt.data.DataLoader(ds, batch_size=2, shuffle=False)
for bx, by in dl:
    print(bx.to_vector(), by.to_vector())
del dl, ds
```

## `data.TensorDataset`

- **Chữ ký:** `data.TensorDataset(x, y) -> TensorDataset`
- **Nguồn:** `src/bindings/bind_data.cpp:15-20`
- **Tham số:** `x` — tensor đặc trưng, shape `[n, ...]`. `y` — tensor nhãn, shape `[n, ...]`.
- **Thuộc tính:** `x`, `y` — thuộc tính tensor đọc/ghi.
- **Phương thức:** `size()` (trả về `n`), `get(index)` (trả về cặp hàng thứ `index`), `__len__`, `__getitem__`.
- **Trường hợp biên:** `x` và `y` phải có cùng chiều đầu tiên; không khớp sẽ không được kiểm tra lúc khởi tạo mà lỗi tại thời điểm `get`.
- **PyTorch:** `torch.utils.data.TensorDataset` — tương đương, trừ việc LiteTorch nhận đúng hai tensor (`x`, `y`), không phải danh sách varargs.

<!-- ex: id=r11-tensordataset device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
xs = lt.Tensor.from_vector([float(i) for i in range(8)], [4, 2], cpu)
ys = lt.Tensor.from_vector([0.0, 1.0, 0.0, 1.0], [4], cpu)
ds = lt.data.TensorDataset(xs, ys)
print(ds.size(), ds.get(1)[0].to_vector(), ds[1][1].to_vector())
```

## Sampler (`data.Sampler`, `data.SequentialSampler`, `data.RandomSampler`, `data.BatchSampler`)

- **Chữ ký:** `data.SequentialSampler(n) -> SequentialSampler`; `data.RandomSampler(n, replacement=False, num_samples=0, seed=0) -> RandomSampler`; `data.BatchSampler(sampler, batch_size, drop_last=False) -> BatchSampler`
- **Nguồn:** `src/bindings/bind_data.cpp:22-43`
- **Phương thức:** `indices()` — trả về danh sách chỉ số (danh sách Python các int); `__len__` — số chỉ số (với `BatchSampler`, số batch). `BatchSampler` còn có `batches()` (danh sách các danh sách chỉ số) và thuộc tính chỉ đọc `batch_size` / `drop_last`.
- **Tham số:** `RandomSampler`: `n` — kích thước dataset; `replacement` — lấy mẫu có hoàn lại, mặc định False; `num_samples` — số mẫu cần rút, mặc định 0 (nghĩa là `n`); `seed` — hạt giống RNG, mặc định 0 (0 nghĩa là không gieo hạt). `BatchSampler`: `sampler` — sampler nền; `batch_size` — số chỉ số mỗi batch; `drop_last` — bỏ batch cuối dở dang, mặc định False.
- **Trường hợp biên:** sampler **không thể iterate** từ Python — không có `__iter__`; dùng `indices()` / `batches()` hoặc truyền sampler cho `DataLoader`. `RandomSampler` với `seed=0` rút thứ tự khác nhau mỗi lần gọi.
- **PyTorch:** `torch.utils.data.SequentialSampler` / `RandomSampler` / `BatchSampler` — cùng ngữ nghĩa; sampler của PyTorch có thể iterate, sampler của LiteTorch thì không.

<!-- ex: id=r11-samplers device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
s = lt.data.SequentialSampler(5)
r = lt.data.RandomSampler(5, seed=7)
print(s.indices(), len(s))
print(r.indices())
b = lt.data.BatchSampler(s, 2)
print(b.batches(), len(b))
```

## `data.Subset` và `data.random_split`

- **Chữ ký:** `data.Subset(dataset, indices) -> Subset`; `data.random_split(dataset, lengths, seed=0) -> list[Dataset]`
- **Nguồn:** `src/bindings/bind_data.cpp:45-54`
- **Tham số:** `Subset`: `dataset` — dataset nền; `indices` — danh sách chỉ số cần giữ. `random_split`: `lengths` — danh sách kích thước các phần tách, tổng bằng `len(dataset)`; `seed` — hạt giống xáo trộn, mặc định 0.
- **Phương thức:** `Subset` có `size()`, `get(index)`, `__len__`, `__getitem__` (chỉ số được ánh xạ qua danh sách).
- **Trường hợp biên:** `random_split` với `seed=0` xáo trộn không gieo hạt. `lengths` phải cộng đúng bằng kích thước dataset.
- **PyTorch:** `torch.utils.data.Subset` / `random_split` — tương đương.

<!-- ex: id=r11-subset device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
xs = lt.Tensor.from_vector([float(i) for i in range(8)], [4, 2], cpu)
ys = lt.Tensor.from_vector([0.0, 1.0, 0.0, 1.0], [4], cpu)
ds = lt.data.TensorDataset(xs, ys)
sub = lt.data.Subset(ds, [0, 2])
print(sub.size(), sub.get(1)[0].to_vector())
parts = lt.data.random_split(ds, [3, 1], seed=3)
print([p.size() for p in parts])
```

## `data.DataLoader`

- **Chữ ký:** `data.DataLoader(dataset, batch_size, shuffle=True, device=None, prefetch_limit=2, num_workers=0, drop_last=False, sampler=None, collate_fn=None) -> DataLoader`
- **Nguồn:** `src/bindings/bind_data.cpp:56-112`, `include/litetorch/data.h:109-114`
- **Tham số:** `dataset` — một `Dataset`. `batch_size` — số mục mỗi batch (bắt buộc). `shuffle` — xáo trộn mỗi epoch, mặc định True. `device` — thiết bị đích cho batch, mặc định None (CPU). `prefetch_limit` — số batch nạp trước, mặc định 2. `num_workers` — số luồng nạp trước chạy nền, mặc định 0 (batch được dựng trên luồng gọi; `num_workers > 0` sinh luồng worker nạp trước batch). `drop_last` — bỏ batch cuối dở dang, mặc định False. `sampler` — một `Sampler` ghi đè thứ tự xáo trộn (khi cho, `shuffle` bị bỏ qua). `collate_fn` — một callable Python nhận danh sách cặp `(x, y)` và trả về một tuple `(batch_x, batch_y)`.
- **Phương thức:** `reset()` — khởi động lại epoch; `next()` — trả về tuple `(batch_x, batch_y)` tiếp theo hoặc `None` khi hết; `__iter__` / `__next__` — lặp sẽ reset và cho batch cho đến `StopIteration`; `__len__` / `num_batches()` — số batch; `save_state(filepath)` / `load_state(filepath)` — lưu và khôi phục vị trí lặp (tiếp tục giữa epoch).
- **Thuộc tính:** `dataset`, `batch_size`, `shuffle` — đọc/ghi.
- **Trường hợp biên:** `shuffle=True` **không** được gieo hạt bởi `lt.manual_seed` — thứ tự epoch thay đổi giữa các lần chạy; dùng `shuffle=False` (hoặc một `RandomSampler` có seed tường minh) cho các ví dụ tái lập được. Một loader giữ `Dataset` viết bằng Python hoặc `collate_fn` Python sẽ làm trình thông dịch abort khi tắt (SIGABRT, `FATAL: exception not rethrown`) trừ khi loader được xóa tường minh trước khi thoát — `del loader` (và `del dataset`) ở cuối chương trình. Batch được dựng trên thiết bị truyền trong `device`; collate mặc định xếp các hàng thành tensor `[batch, ...]`.
- **PyTorch:** `torch.utils.data.DataLoader` — cùng vai trò; PyTorch mặc định `shuffle=False` và `num_workers=0`, LiteTorch mặc định `shuffle=True`. `next()` của PyTorch chỉ qua giao thức iterator; LiteTorch còn có phương thức `next()` trực tiếp trả về `None` khi hết.

<!-- ex: id=r11-dataloader device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import tempfile, os
cpu = lt.Device("cpu")
xs = lt.Tensor.from_vector([float(i) for i in range(8)], [4, 2], cpu)
ys = lt.Tensor.from_vector([0.0, 1.0, 0.0, 1.0], [4], cpu)
ds = lt.data.TensorDataset(xs, ys)
dl = lt.data.DataLoader(ds, batch_size=2, shuffle=False)
print(dl.num_batches())
for bx, by in dl:
    print(bx.shape, by.to_vector())
dl2 = lt.data.DataLoader(ds, batch_size=2, shuffle=False)
print(dl2.next()[1].to_vector())
path = os.path.join(tempfile.mkdtemp(), "dlstate")
dl2.save_state(path)
dl3 = lt.data.DataLoader(ds, batch_size=2, shuffle=False)
dl3.load_state(path)
print(dl3.next()[1].to_vector())
```

## `save` / `load`

- **Chữ ký:** `save(obj, filepath) -> None`; `load(obj, filepath) -> None`
- **Nguồn:** `src/bindings/bind_data.cpp:141-169`
- **Tham số:** `obj` — với `save`: một `nn.Module`, một danh sách tensor tham số, hoặc một `Optimizer`; với `load`: một `nn.Module` hoặc danh sách tensor tham số để load **vào** (đối tượng phải đã tồn tại với đúng shape). `filepath` — đường dẫn đích.
- **Hành vi:** `save` điều phối theo kiểu đối tượng: module lưu tham số, danh sách lưu các tensor, optimizer lưu trạng thái optimizer. `load` là tại chỗ — nó không trả về đối tượng. Định dạng là định dạng nhị phân riêng của LiteTorch (magic + CRC32), không phải pickle của `torch.save`.
- **Trường hợp biên:** `save` một tensor trần ném lỗi (`save expects a Module, list of Tensor parameters, or Optimizer`); bọc nó trong danh sách. `load` vào module có số tham số khác sẽ ném lỗi. File không tương thích với PyTorch.
- **PyTorch:** `torch.save` / `torch.load` — cùng vai trò, định dạng khác; `torch.load` trả về đối tượng, `load` của LiteTorch thay đổi tại chỗ.

<!-- ex: id=r11-save-load device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import tempfile, os
cpu = lt.Device("cpu")
lt.manual_seed(0)
m = lt.nn.Linear(4, 2)
path = os.path.join(tempfile.mkdtemp(), "m.lt")
lt.save(m, path)
m2 = lt.nn.Linear(4, 2)
lt.load(m2, path)
print([p.to_vector() for p in m2.parameters()] == [p.to_vector() for p in m.parameters()])
t = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu)
lt.save([t], path)
t2 = lt.Tensor.from_vector([0.0, 0.0, 0.0], [3], cpu)
lt.load([t2], path)
print(t2.to_vector())
```

## `save_parameters` / `load_parameters`

- **Chữ ký:** `save_parameters(params, filepath) -> None`; `load_parameters(params, filepath) -> None`
- **Nguồn:** `src/bindings/bind_data.cpp:114-115`, `src/serialization/parameters.cpp`
- **Tham số:** `params` — một dãy tensor tham số (ví dụ `module.parameters()`), không phải module. `filepath` — đường dẫn đích.
- **Hành vi:** dạng cấp thấp của `save` / `load` cho danh sách tham số. `load_parameters` là tại chỗ.
- **Trường hợp biên:** truyền module thay vì danh sách tham số sẽ ném lỗi kiểu.
- **PyTorch:** `torch.save(module.state_dict(), ...)` / `module.load_state_dict(torch.load(...))` — cùng vai trò, định dạng khác.

<!-- ex: id=r11-save-parameters device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import tempfile, os
cpu = lt.Device("cpu")
lt.manual_seed(0)
m = lt.nn.Linear(4, 2)
path = os.path.join(tempfile.mkdtemp(), "p.lt")
lt.save_parameters(m.parameters(), path)
m2 = lt.nn.Linear(4, 2)
lt.load_parameters(m2.parameters(), path)
print([p.to_vector() for p in m2.parameters()] == [p.to_vector() for p in m.parameters()])
```

## `save_safetensors` / `load_safetensors`

- **Chữ ký:** `save_safetensors(tensors, filepath) -> None`; `load_safetensors(filepath) -> dict[str, Tensor]`
- **Nguồn:** `src/bindings/bind_data.cpp:116-127`, `src/serialization/safetensors.cpp`
- **Tham số:** `tensors` — một **dict** ánh xạ tên tới tensor (không phải module). `filepath` — đường dẫn đích (thường `.safetensors`).
- **Hành vi:** ghi định dạng safetensors (tương thích với `safetensors` của Hugging Face). `load_safetensors` trả về dict các tensor theo tên đã lưu.
- **Trường hợp biên:** truyền module thay vì dict sẽ ném lỗi. Tên được lưu đã sắp xếp; dict trả về có khóa là tên đã lưu.
- **PyTorch:** `safetensors.torch.save_file` / `load_file` — cùng định dạng và vai trò.

<!-- ex: id=r11-safetensors device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import tempfile, os
cpu = lt.Device("cpu")
lt.manual_seed(0)
m = lt.nn.Linear(4, 2)
path = os.path.join(tempfile.mkdtemp(), "m.safetensors")
lt.save_safetensors({"weight": m.parameters()[0], "bias": m.parameters()[1]}, path)
d = lt.load_safetensors(path)
print(sorted(d.keys()), d["weight"].to_vector() == m.parameters()[0].to_vector())
```

## `save_optimizer_state` / `load_optimizer_state`

- **Chữ ký:** `save_optimizer_state(optimizer, filepath) -> None`; `load_optimizer_state(optimizer, filepath) -> None`
- **Nguồn:** `src/bindings/bind_data.cpp:129-130`, `src/serialization/optimizer.cpp:37`
- **Tham số:** `optimizer` — optimizer cần lưu moment. `filepath` — đường dẫn đích.
- **Hành vi:** định dạng nhị phân với magic riêng cho từng kiểu (`LTAD` Adam, `LTAW` AdamW, `LTSG` SGD, `LTRP` RMSprop) cộng CRC32. `load_optimizer_state` là tại chỗ và kiểm tra kiểu file khớp với kiểu optimizer.
- **Trường hợp biên:** chỉ **Adam, AdamW, SGD, RMSprop** được hỗ trợ. Mọi optimizer khác (NAdam, Adagrad, Adamax, Adadelta, ZeRO3Optimizer, AdamW8bit, AdamWFP8) đều ném `Unsupported optimizer type for saving state` khi lưu. Load trạng thái đã lưu từ kiểu optimizer khác sẽ ném lỗi. `Adagrad` và `NAdam` có `state_dict()` / `load_state_dict()` riêng.
- **PyTorch:** `torch.save(optimizer.state_dict(), ...)` — PyTorch hỗ trợ mọi optimizer; LiteTorch hỗ trợ bốn.

<!-- ex: id=r11-optimizer-state device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import tempfile, os
cpu = lt.Device("cpu")
w = lt.Tensor.from_vector([1.0], [1], cpu, True)
w.grad = lt.Tensor.from_vector([0.5], [1], cpu)
adam = lt.optim.Adam([w], lr=0.01)
adam.step()
path = os.path.join(tempfile.mkdtemp(), "opt.lt")
lt.save_optimizer_state(adam, path)
w2 = lt.Tensor.from_vector([1.0], [1], cpu, True)
adam2 = lt.optim.Adam([w2], lr=0.01)
lt.load_optimizer_state(adam2, path)
print("loaded")
try:
    lt.save_optimizer_state(lt.optim.NAdam([w2], lr=0.002), path)
except Exception as e:
    print(type(e).__name__, str(e)[:60])
```

## `AsyncCheckpointSaver`

- **Chữ ký:** `AsyncCheckpointSaver() -> AsyncCheckpointSaver`
- **Nguồn:** `src/bindings/bind_data.cpp:131-139`
- **Phương thức:** `save_async(tensors, filepath)` — xếp một lần ghi checkpoint lên luồng nền; `tensors` là **dict** tên → tensor. `wait()` — chặn cho đến khi mọi lần lưu đã xếp hoàn tất. `is_saving()` — True khi đang có lần lưu chạy.
- **Hành vi:** ghi định dạng safetensors bất đồng bộ; phía gọi phải gọi `wait()` trước khi đọc file hoặc thoát, nếu không file có thể thiếu hoặc dở dang.
- **Trường hợp biên:** `save_async` nhận dict, không phải module. Không có callback khi hoàn tất — thăm dò `is_saving()` hoặc gọi `wait()`. Treo khi tắt không ổn định đã quan sát: trong đợt xác minh, 2 trong ~46 lần chạy treo hơn 60 giây sau khi `wait()` trả về và phải kill (nghi là race khi tháo dỡ luồng nền); xóa saver tường minh bằng `del saver` trước khi thoát như biện pháp phòng ngừa.
- **PyTorch:** không có đối tác trong `torch` (checkpoint bất đồng bộ có trong `torch.distributed.checkpoint`).

<!-- ex: id=r11-async-saver device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import tempfile, os
cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0], [3], cpu)
saver = lt.AsyncCheckpointSaver()
path = os.path.join(tempfile.mkdtemp(), "a.safetensors")
saver.save_async({"t": t}, path)
print(saver.is_saving())
saver.wait()
print(os.path.exists(path), saver.is_saving())
del saver
```

## `TrainerConfig`

- **Chữ ký:** `TrainerConfig() -> TrainerConfig`
- **Nguồn:** `src/bindings/bind_data.cpp:191-197`, `include/litetorch/trainer.h:25-29`
- **Trường (đều đọc/ghi):** `ckpt_dir` — thư mục checkpoint, mặc định `"checkpoints"`. `save_every_steps` — số bước giữa các checkpoint, mặc định 1000. `max_checkpoints` — số checkpoint mới nhất được giữ, mặc định 5. `async_save` — ghi checkpoint trên luồng nền, mặc định True. `elastic` — dùng đường checkpoint nhất quán elastic (chỉ phân tán), mặc định False.
- **PyTorch:** không có lớp tương đương (`torch` không có trainer tích hợp; gần nhất là một tập con của Hugging Face `TrainingArguments`).

<!-- ex: id=r11-trainerconfig device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cfg = lt.TrainerConfig()
print(cfg.ckpt_dir, cfg.save_every_steps, cfg.max_checkpoints, cfg.async_save, cfg.elastic)
cfg.ckpt_dir = "my-checkpoints"
cfg.max_checkpoints = 3
print(cfg.ckpt_dir, cfg.max_checkpoints)
```

## `Trainer`

- **Chữ ký:** `Trainer(model, optimizer, config=TrainerConfig()) -> Trainer`
- **Nguồn:** `src/bindings/bind_data.cpp:199-210`, `src/trainer/trainer.cpp`
- **Tham số:** `model` — một `nn.Module`. `optimizer` — một optimizer. `config` — một `TrainerConfig`, mặc định khi bỏ qua.
- **Phương thức:** `train_step()` — một bước optimizer cộng `zero_grad` (xem cảnh báo dưới); `save_checkpoint(step)` — ghi `step-<n>.safetensors` vào `config.ckpt_dir`; `load_latest()` — load checkpoint mới nhất vào model, trả về bước của nó hoặc -1 khi không có; `garbage_collect()` — xóa mọi checkpoint trừ `max_checkpoints` mới nhất; `latest_step()` — bước checkpoint mới nhất, hoặc -1; `rank()` — hạng của tiến trình (0 khi không phân tán); `world_size()` — 1 khi không phân tán; `is_distributed()` — True chỉ khi có nhóm tiến trình phân tán; `wait_pending_saves()` — chặn cho đến khi các lần ghi checkpoint bất đồng bộ hoàn tất.
- **Cảnh báo — `train_step()` KHÔNG chạy forward/backward.** Nó chỉ gọi `optimizer.step()` rồi `zero_grad()`. Phía gọi phải chạy forward pass, tính loss, gọi `loss.backward()` và đặt gradient trước khi gọi `train_step()`. Gọi `train_step()` trên model không có gradient là no-op lặng lẽ.
- **Trường hợp biên:** `save_checkpoint` là bất đồng bộ khi `config.async_save` là True (mặc định) — file chỉ xuất hiện sau `wait_pending_saves()`; kiểm tra thư mục trước đó sẽ thấy trống dù `latest_step()` đã báo bước. `load_latest()` trả về -1 (không phải ngoại lệ) khi không có checkpoint. `rank()`, `world_size()`, `is_distributed()` là phương thức, không phải thuộc tính. Ở chế độ phân tán, `save_checkpoint` dùng đường checkpoint nhất quán elastic khi `config.elastic` là True.
- **PyTorch:** không có lớp tương đương trong `torch` (gần nhất: vòng huấn luyện viết tay, hoặc Hugging Face `Trainer`).

<!-- ex: id=r11-trainer device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
import tempfile, os
cpu = lt.Device("cpu")
lt.manual_seed(0)
m = lt.nn.Linear(4, 2)
opt = lt.optim.SGD(m.parameters(), lr=0.1)
cfg = lt.TrainerConfig()
cfg.ckpt_dir = os.path.join(tempfile.mkdtemp(), "ckpt")
tr = lt.Trainer(m, opt, cfg)
print(tr.rank(), tr.world_size(), tr.is_distributed())
for p in m.parameters():
    p.grad = lt.Tensor.from_vector([0.1] * p.numel(), p.shape, cpu)
tr.train_step()
tr.save_checkpoint(1)
tr.wait_pending_saves()
print(sorted(os.listdir(cfg.ckpt_dir)), tr.latest_step(), tr.load_latest())
```

## RNG (`manual_seed`, `get_rng_state`, `set_rng_state`, `randn`, `ones`, `zeros`, `arange`)

- **Chữ ký:** `manual_seed(seed) -> None`; `get_rng_state() -> object`; `set_rng_state(state) -> None`; `randn(shape, device=CPU, requires_grad=False, dtype=FP32) -> Tensor`; `ones(shape, device=CPU, requires_grad=False, dtype=FP32) -> Tensor`; `zeros(shape, device=CPU, requires_grad=False, dtype=FP32) -> Tensor`; `arange(start, end, step=1, device=CPU, dtype=FP32) -> Tensor`
- **Nguồn:** `src/bindings/bind_data.cpp:171-184`
- **Hành vi:** `manual_seed` gieo hạt cho RNG CPU toàn cục; `get_rng_state` / `set_rng_state` lưu và khôi phục nó, nên dãy rút sau `set_rng_state(state)` tái lập dãy rút sau `get_rng_state()`. `randn` rút giá trị chuẩn chuẩn.
- **Trường hợp biên:** `manual_seed` **không** gieo hạt cho thứ tự `DataLoader(shuffle=True)` — xáo trộn của loader dùng RNG riêng. `arange` với kiểu dữ liệu nguyên cho ra giá trị nguyên.
- **PyTorch:** `torch.manual_seed` / `torch.get_rng_state` / `torch.set_rng_state` / `torch.randn` / `torch.ones` / `torch.zeros` / `torch.arange` — tương đương (lưu ý LiteTorch viết `arange`, PyTorch cũng `arange`; thứ tự đối số kiểu dữ liệu khác).

<!-- ex: id=r11-rng device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt
cpu = lt.Device("cpu")
lt.manual_seed(42)
a = lt.randn([3], cpu)
st = lt.get_rng_state()
b = lt.randn([3], cpu)
lt.set_rng_state(st)
c = lt.randn([3], cpu)
print(b.to_vector() == c.to_vector())
print(lt.ones([2], cpu).to_vector(), lt.zeros([2], cpu).to_vector(), lt.arange(0, 4, 2, cpu).to_vector())
```
