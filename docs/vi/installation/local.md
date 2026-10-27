# Cài đặt trên máy cá nhân

Trang này hướng dẫn chạy LiteTorch trên phần cứng của chính bạn: máy trạm Linux hoặc máy chủ Windows, CPU hoặc một GPU NVIDIA. Trang này không lặp lại các lệnh cài đặt — các lệnh đó nằm trong [Hướng dẫn cài đặt](../installation.md) và được liên kết từ §1. Trang này cho biết nên chọn hướng nào, cách chứng minh GPU thực sự đang được dùng trước khi huấn luyện, và lần huấn luyện đầu tiên trên máy local trông như thế nào.

> Ghi chú trạng thái: các bước CPU trên trang này đã được chạy ngày 2026-10-27 trên máy kiểm tra của nhóm docs (2 CPU, 7,7 GB RAM, Ubuntu 24.04, Python 3.12.3) với bản build extension D3 của Core, `litetorch.build_commit() == 88a4ba1`. Các bước GPU được đánh dấu `unverified` — máy này không có GPU. Các bước Windows được đánh dấu `unverified` — máy này không có host Windows. Đừng coi một khối `unverified` là đã được chạy cho đến khi một máy có phần cứng đó chạy nó.

---

## 1. Chọn pip hay biên dịch từ mã nguồn

| Trường hợp | Hướng đi |
| :--- | :--- |
| Bạn muốn bản phát hành và sẽ huấn luyện trên CPU hoặc một GPU | pip — [Hướng dẫn cài đặt §2.2 (pip trên Linux)](../installation.md#22-linux-và-google-colab) hoặc [§2.3 (pip trên Windows)](../installation.md#23-windows) |
| Bạn cần cây mã nguồn hiện tại, bạn đang vá framework, hoặc bạn cần thư viện GPU native được biên dịch từ checkout của mình | Biên dịch từ mã nguồn — [Hướng dẫn cài đặt §2.4 (biên dịch từ mã nguồn)](../installation.md#24-biên-dịch-từ-mã-nguồn) |
| Bạn muốn một môi trường tái lập được, còn nguyên sau khi cài lại host | Container — [Cài đặt trên Docker](docker.md) |

Hai sự thật quyết định lựa chọn trên một máy nhỏ:

1. **PyPI phân phối source distribution, không phải wheel.** `pip install litetorch` tải sdist về và biên dịch extension C++ ngay trên máy của bạn. Quan sát ngày 2026-10-27: trong một venv sạch trên máy VM 2 CPU này, pip đã build và cài đặt `litetorch` 0.3.40 trong ~28 phút (wheel `litetorch-0.3.40-cp312-cp312-linux_x86_64.whl`, 23 MB, sha256 `c4b84b0062e8ea082009451a97d9066c5b7a13d581c0b71f9b9ef8776b6247a0`), và gói đã cài import được và chạy smoke test §2.6 trên CPU. Lưu ý bản phát hành PyPI cũ hơn cây mã nguồn hiện tại: nó không có thuộc tính `lt.build_commit()`, nên không thể gắn commit cho nó — các header `verified-2026-10-27@88a4ba1` trên trang này dựa trên bản build extension D3 của Core, không phải gói PyPI. Một lần build đầy đủ cây mã nguồn hiện tại trên máy local không hoàn thành trong một phiên — xem `plan/build-time.md`: bị kill ở phút 44,6 khi mới biên dịch xong 99/167 file nguồn và chưa tạo ra file `.so` nào. Hãy dự trù thời gian tương ứng, hoặc dùng extension đã được build sẵn nếu có cho nền tảng của bạn.
2. **Biên dịch từ mã nguồn chỉ đáng làm nếu bạn build xong được.** Trên máy nhỏ, hãy giới hạn số job biên dịch song song bằng `LT_BUILD_JOBS=1` (đồng thời ép `-O1` cho các translation unit binding, giúp giảm bộ nhớ đỉnh) — xem [Hướng dẫn cài đặt §2.4](../installation.md#24-biên-dịch-từ-mã-nguồn). Các build kiểm tra cho tài liệu này chạy trên Kaggle CPU; các bước configure (`cmake ..`, `cmake --build`) ở đây chỉ được kiểm tra đến mức công cụ có sẵn: `cmake` 3.28.3, `g++` 13.3, `CMakeLists.txt` và `setup.py` có mặt trong cây mã nguồn.

---

## 2. Kiểm tra driver GPU và CUDA toolkit

Quy tắc bắt buộc, không phải gợi ý: kiểm tra (và cài đặt nếu thiếu) CUDA toolkit một cách tường minh trước mọi lần chạy GPU. Thiếu `nvcc` nghĩa là thư viện GPU native không bao giờ được build, và framework sau đó sẽ fallback sang CPU một cách âm thầm.

Trên Linux:

<!-- ex: id=local-2-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```bash
nvidia-smi
nvcc --version
```

Đã kiểm tra ngày 2026-10-27 trên máy kiểm tra của nhóm docs: cả hai lệnh đều vắng mặt (`command not found`) — máy VM này không có GPU và không có CUDA toolkit, và đó là kết quả đúng trên một máy chỉ có CPU. Trên một máy trạm có GPU, cả hai đều phải in ra kết quả: `nvidia-smi` liệt kê GPU và driver, `nvcc --version` in ra phiên bản CUDA toolkit (11.8 hoặc 12.x theo [Hướng dẫn cài đặt §2.1](../installation.md#21-yêu-cầu-hệ-thống)).

Nếu không tìm thấy `nvcc` trên một máy có GPU NVIDIA, hãy cài CUDA Toolkit cho bản phân phối của bạn (ví dụ, kho network repository CUDA 12.x của NVIDIA cho Ubuntu) và chạy lại `nvcc --version` cho đến khi nó in ra một phiên bản. cuDNN là một thư viện riêng, không nằm trong CUDA Toolkit — hãy cài độc lập để tăng tốc convolution; LiteTorch tự phát hiện nó qua `LITETORCH_CUDNN_ROOT` hoặc các đường dẫn chuẩn. Chỉ sau đó mới build hoặc chạy LiteTorch trên GPU.

Trên Windows (chưa kiểm tra — máy VM này không có host Windows):

<!-- ex: id=local-2-2 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=unverified -->

```cmd
nvidia-smi
nvcc --version
```

Quy tắc vẫn vậy: cả hai đều phải in ra kết quả trước một lần chạy GPU. Với GPU AMD, hãy kiểm tra `hipcc` có mặt để tăng tốc ROCm native (các ví dụ ROCm trong tài liệu này được đánh dấu `UNVERIFIED-ROCm` cho đến khi một lần chạy ROCm kiểm tra chúng).

---

## 3. Cổng kiểm tra tiền điều kiện (pre-flight gate)

Trước mọi lần chạy GPU, hãy khẳng định cả ba điều sau. Cổng này bắt lỗi local phổ biến nhất: backend GPU âm thầm không được nạp trong khi quá trình huấn luyện vẫn chạy trên CPU.

1. Dòng khởi tạo `Native GPU (CUDA/HIP)` xuất hiện (KHÔNG phải OpenCL) — điều này bắt lỗi thiếu thư viện native.
2. `liblitetorch_gpu.so` nằm cạnh extension trong thư mục `python/` (thư mục đầu ra của `build_ext --inplace` — xem [Hướng dẫn cài đặt §2.4](../installation.md#24-biên-dịch-từ-mã-nguồn)). Với bố cục build hiện tại, nó nằm trong `build/lib.linux-x86_64-cpython-*/` và phải được chép sang `python/` trước khi import hoạt động.
3. `lt.cuda.device_count()` trả về số lượng đúng như kỳ vọng (1 trên một máy trạm một GPU).

Hãy chạy script pre-flight sau. Trên máy chỉ có CPU, nó báo cáo trạng thái CPU; trên máy GPU, cùng một script phải báo cáo trạng thái GPU — nếu không, hãy dừng lại.

<!-- ex: id=local-3-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "python")
import litetorch as lt

print("Backend:", lt.get_backend_name())
print("Device:", lt.auto_device())
print("GPU available:", lt.is_gpu_available())
print("CUDA device count:", lt.cuda.device_count())
```

Đã kiểm tra ngày 2026-10-27 trên CPU, LiteTorch 88a4ba1 (bản build extension D3; đoạn preamble `sys.path` trỏ tới thư mục extension — hãy chỉnh lại theo cây mã nguồn của bạn). Kết quả đúng trên máy chỉ có CPU:

```
[litetorch Warning] GPU acceleration (OpenCL) is not available. Falling back to CPU.
Backend: CPU
Device: cpu:0
GPU available: False
CUDA device count: 0
```

Dòng cảnh báo là máy CPU báo cho bạn biết không có thiết bị tăng tốc nào để fallback — điều này là đúng ở đây, và chính việc vắng mặt dòng khởi tạo `Native GPU (CUDA/HIP)` xác nhận không có backend GPU nào được nạp. Trên máy GPU, các dòng đầu thay vào đó phải cho thấy dòng khởi tạo `Native GPU (CUDA/HIP)`, `Backend: CUDA` (tên backend chính xác tùy thuộc vào bản build), `Device: gpu:0`, `GPU available: True`, và `CUDA device count: 1`.

### Khi cổng kiểm tra thất bại

| Triệu chứng | Nguyên nhân | Cách sửa |
| :--- | :--- | :--- |
| Không có dòng `Native GPU (CUDA/HIP)`; `Backend: CPU` trên máy có GPU | Thư viện GPU native thiếu hoặc `nvcc` vắng mặt lúc build | Kiểm tra `nvcc --version`, build lại từ mã nguồn, xác nhận dòng khởi tạo xuất hiện |
| `ModuleNotFoundError: No module named 'litetorch'` | Extension không nằm trên `sys.path` | Thêm `<tree>/python` vào `sys.path` — extension nằm trong `python/`, không phải thư mục gốc của cây |
| Không tìm thấy `liblitetorch_gpu.so` khi import | Thư viện nằm trong thư mục build, không nằm cạnh extension | Chép nó từ `build/lib.linux-x86_64-cpython-*/` sang `python/` |
| `CUDA device count: 0` trên máy có GPU | Driver hoặc thiết bị không hiển thị với process | Kiểm tra `nvidia-smi`; kiểm tra thiết bị có đang bị một process hoặc container khác giữ không |
| `CUDA device count: 1` trong khi máy có nhiều GPU hơn | Lỗi phát hiện số lượng thiết bị (B5-DEVICE-COUNT, bản sửa đang được review) | Xem [Khác biệt hành vi so với PyTorch](../behavioral-differences.md); đừng cho rằng con số là đúng cho đến khi bản sửa được merge |
| Huấn luyện chạy nhưng chậm, `Device: cpu:0` | Fallback CPU âm thầm — `.to(device)` fallback sang CPU khi thiết bị tăng tốc không khả dụng | Dừng lại và chạy cổng kiểm tra ở trên; chạy chậm không phải bằng chứng GPU đang được dùng |

Nếu bất kỳ khẳng định nào thất bại, hãy dừng lại và sửa phần cài đặt — đừng coi lỗi đó là vấn đề của mô hình hay dữ liệu. Script kiểm tra cài đặt đầy đủ (matmul + backward) nằm trong [Hướng dẫn cài đặt §2.6](../installation.md#26-script-kiểm-tra-cài-đặt).

---

## 4. Lần huấn luyện đầu tiên trên máy local

### Chạy trên CPU (đã kiểm tra)

Script này huấn luyện mô hình nhỏ trong quickstart trên bốn mẫu tổng hợp trong 5 epoch. Nó tự chứa, ngoại trừ đoạn preamble `sys.path` trỏ tới thư mục extension — hãy chỉnh lại theo cây mã nguồn của bạn. Loss phải giảm; nếu không, có vấn đề với thiết lập, không phải với mô hình.

<!-- ex: id=local-4-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "python")
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

for epoch in range(5):
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
    print(f"Epoch {epoch + 1}, Loss: {total_loss / max(steps, 1):.4f}")
```

Đã kiểm tra ngày 2026-10-27 trên CPU, LiteTorch 88a4ba1. Kết quả đúng (chính xác với `lt.manual_seed(42)` và `shuffle=False`):

```
Epoch 1, Loss: 1.0306
Epoch 2, Loss: 0.7767
Epoch 3, Loss: 0.5915
Epoch 4, Loss: 0.4568
Epoch 5, Loss: 0.3630
```

Năm epoch đầu khớp chính xác với lần chạy 10 epoch trong [Bắt đầu nhanh §3.4](../quickstart.md); lần chạy đó đã được kiểm tra parity với torch ngày 2026-10-27 so với torch 2.14.1+cpu (quỹ đạo AdamW 10 epoch, chênh lệch tối đa mỗi epoch 1.38e-07). Ba lệnh phải gọi theo đúng thứ tự này mỗi bước là `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` — quên `zero_grad()` sẽ làm gradient tích lũy qua các bước và loss sẽ không hoạt động như kỳ vọng.

### Biến thể GPU (chưa kiểm tra — REQUIRES-GPU)

> **REQUIRES-GPU.** Biến thể dưới đây chưa được chạy: máy kiểm tra của nhóm docs không có GPU. Hãy chạy cổng pre-flight trong §3 trước; nếu cổng báo cáo `gpu:0`, thay đổi duy nhất so với lần chạy CPU là chọn thiết bị và đặt dữ liệu — mô hình, optimizer, loss và vòng lặp hoàn toàn giống nhau.

<!-- ex: id=local-4-2 device=gpu backend=cuda runtime=~1s torch-parity=n/a status=unverified -->

```python
import sys
sys.path.insert(0, "python")
import litetorch as lt

lt.manual_seed(42)
device = lt.Device("gpu:0")

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

for epoch in range(5):
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
    print(f"Epoch {epoch + 1}, Loss: {total_loss / max(steps, 1):.4f}")
```

Kết quả đúng trên một GPU là cùng năm giá trị loss như lần chạy CPU (`1.0306` xuống `0.3630`) — cùng seed, cùng dữ liệu, cùng phép toán. Nếu loss trên GPU khác quá 1e-4, đó là một defect về parity: hãy báo cáo, đừng chỉnh xung quanh nó.

---

## 5. Khi nào chuyển sang cloud

Một máy cá nhân không còn là nơi phù hợp khi bất kỳ điều kiện nào sau đây xảy ra:

- **Cần nhiều hơn một GPU.** Huấn luyện đa GPU (collective NCCL, allreduce 2 GPU, mismatch-throw) cần một host có hai GPU trở lên hiển thị với một process — xem [Cài đặt trên Kaggle](kaggle.md) để biết thiết lập 2 GPU tham chiếu.
- **Mô hình hoặc dữ liệu không còn vừa trong RAM / VRAM local.** Activation checkpointing và memory eviction mở rộng những gì một máy chứa được, nhưng chúng không tạo thêm VRAM.
- **Lần chạy kéo dài hơn thời gian máy sẵn sàng.** Huấn luyện dài thuộc về một host không sleep, không reboot để cập nhật, và không chia sẻ GPU với một phiên desktop.
- **Cần TPU.** Không có TPU local; lần chạy TPU đi qua Google Cloud hoặc Colab.

Hướng dẫn triển khai cloud (các trang cùng đợt này):

- [AWS](aws.md)
- [Azure](azure.md)
- [GCP](gcp.md)
- [Các cloud khác](other-clouds.md)

Để thử một lần chạy lớn hơn bằng GPU miễn phí trước khi trả tiền cloud, xem [Cài đặt trên Kaggle](kaggle.md).

---

## 6. Xử lý sự cố

Xem [Xử lý sự cố và FAQ](../troubleshooting.md) để biết danh sách đầy đủ. Các lỗi local gặp thường xuyên nhất:

- `ModuleNotFoundError: No module named 'litetorch'` — extension nằm trong `<tree>/python/`; hãy thêm nó vào `sys.path`.
- Huấn luyện chạy trên CPU dù máy đã có GPU — chạy cổng pre-flight §3; kiểm tra `nvcc --version` và build lại.
- Build từ mã nguồn trên máy local bị kill hoặc không bao giờ xong — build đầy đủ không hoàn thành trong một phiên trên máy nhỏ (`plan/build-time.md`: bị kill ở phút 44,6). Dùng `LT_BUILD_JOBS=1`, hoặc build trên Kaggle CPU.
- `pip install litetorch` có vẻ bị treo — nó đang biên dịch extension C++ từ source distribution; trên máy 2 CPU việc này mất ~28 phút ngày 2026-10-27. Hãy chờ, hoặc build từ mã nguồn một cách tường minh để tiến độ hiển thị rõ.
