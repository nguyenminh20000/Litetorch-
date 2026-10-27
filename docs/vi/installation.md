# Hướng Dẫn Cài Đặt LiteTorch

Tài liệu này bao quát việc cài đặt LiteTorch trên Linux, Windows, Docker và Kaggle, cùng cách biên dịch từ mã nguồn. Để cài đặt trong container, xem [Cài đặt trên Docker](installation/docker.md); để cài đặt trong notebook, xem [Cài đặt trên Kaggle](installation/kaggle.md).

---

## 2.1. Yêu cầu hệ thống

- **Trình biên dịch C++**: hỗ trợ chuẩn C++14 (`g++` >= 7.0, `clang++` >= 6.0, hoặc MSVC v142+).
- **Python**: phiên bản 3.8 đến 3.12, kèm header phát triển (`python3-dev`).
- **CMake**: phiên bản >= 3.10 (để biên dịch bộ test C++ độc lập).
- **pybind11**: bắt buộc để biên dịch extension Python (`pip install pybind11`).
- **Phần cứng tăng tốc (tùy chọn)**:
  - NVIDIA: CUDA Toolkit 11.8 hoặc 12.x với `nvcc`, `cuBLAS`, `cuBLASLt`. cuDNN là thư viện riêng (không kèm trong CUDA Toolkit) — cài độc lập để tăng tốc convolution; LiteTorch tự động nhận diện qua `LITETORCH_CUDNN_ROOT` hoặc đường dẫn chuẩn.
  - Google TPU: Google Cloud TPU / Colab TPU (v2-v5p, v6e Trillium, v7 Ironwood, v8 8t/8i) với `libtpu.so` và PJRT C-API.
  - AMD: ROCm 5.x hoặc 6.x với `hipcc`, `rocBLAS`, `MIOpen`.
  - Thiết bị khác: trình điều khiển hỗ trợ OpenCL 1.2+.

---

## 2.2. Linux và Google Colab

### 2.2.1. Cài đặt các gói phụ thuộc hệ thống

Trên Ubuntu / Debian / Google Colab:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev libopenblas-dev
```

Trên Fedora / RHEL / CentOS:

```bash
sudo dnf groupinstall "Development Tools" -y
sudo dnf install -y python3-devel openblas-devel
```

### 2.2.2. Cài đặt qua pip

```bash
pip install --upgrade litetorch
```

### 2.2.3. Kiểm tra CUDA toolkit trước mọi lần chạy GPU

Đây là quy tắc bắt buộc, không phải gợi ý: luôn kiểm tra (và cài đặt nếu thiếu) CUDA toolkit một cách tường minh trước khi chạy GPU CUDA. Thiếu `nvcc` nghĩa là thư viện GPU native không bao giờ được biên dịch, và framework sẽ fallback sang CPU một cách âm thầm (xem trang Điểm khác biệt hành vi, đang viết — fallback CPU âm thầm của `to(device)`).

```bash
nvcc --version
```

Nếu không tìm thấy `nvcc`, hãy cài CUDA Toolkit cho bản phân phối của bạn (ví dụ: repository mạng CUDA 12.x của NVIDIA cho Ubuntu) và chạy lại `nvcc --version` cho đến khi nó in ra phiên bản. Chỉ sau đó mới biên dịch hoặc chạy LiteTorch trên GPU.

- **GPU NVIDIA**: LiteTorch tự động biên dịch và kích hoạt tăng tốc CUDA native khi có `nvcc`.
- **Google TPU**: trên Google Colab hoặc Cloud TPU VM, LiteTorch tự động phát hiện `COLAB_TPU_ADDR`, `TPU_NAME`, hoặc `libtpu.so` và gán thiết bị từ `lt.Device("tpu:0")` tới `lt.Device("tpu:7")`.
- **GPU AMD**: kiểm tra `hipcc` đã có sẵn để tăng tốc ROCm native. Các bước cài đặt ROCm dưới đây được ghi lại theo bố cục toolchain ROCm; các ví dụ ROCm trong bộ tài liệu này được đánh dấu `UNVERIFIED-ROCm` cho đến khi một lần chạy ROCm kiểm chứng chúng.

---

## 2.3. Windows

### 2.3.1. Cài đặt công cụ biên dịch C++

Python trên Windows cần MSVC hoặc MinGW để biên dịch extension C++. Chọn một trong hai:

**Phương án A: Microsoft Visual C++ Build Tools (khuyến nghị)**

Chạy PowerShell dưới quyền Quản trị viên (Administrator):

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --force --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;includeRecommended"
```

**Phương án B: MinGW-w64 GCC**

1. Tải và cài đặt MinGW-w64 (GCC 10 trở lên).
2. Thêm thư mục `bin` của MinGW vào biến môi trường `PATH`.

### 2.3.2. Cài đặt qua pip

Mở Command Prompt hoặc Terminal:

```cmd
pip install --upgrade litetorch
```

---

## 2.4. Biên dịch từ mã nguồn

```bash
git clone https://github.com/nguyenminh20000/Litetorch-.git
cd Litetorch-
pip install -r requirements.txt
pip install pybind11
```

Biên dịch extension Python tại chỗ:

```bash
python3 setup.py build_ext --inplace
```

Hai sự thật về build gây ra hầu hết lỗi lần đầu:

1. **`setup.py` đặt `package_dir={"": "python"}`, nên `build_ext --inplace` đặt extension import được dưới `<tree>/python/`, không phải thư mục gốc của cây mã nguồn.** Mọi script đều phải thêm `<tree>/python` vào `sys.path` trước `import litetorch`, nếu không import sẽ lỗi `ModuleNotFoundError`.
2. **`setup.py` in ra `BUILD OK` ngay cả khi một số file biên dịch lỗi.** Cổng kiểm tra thực sự là file extension tồn tại dưới `<tree>/python/` và `import litetorch` thành công. Đừng bao giờ chỉ tin vào dòng `BUILD OK`.

Để biên dịch chỉ cho CPU (bỏ qua bước biên dịch thư viện native nvcc / hipcc):

```bash
LITETORCH_NO_NATIVE_GPU=1 python3 setup.py build_ext --inplace
```

Trên máy nhỏ, hãy giới hạn số luồng biên dịch song song (đồng thời ép `-O1` cho các translation unit binding, giúp giảm bộ nhớ đỉnh):

```bash
LITETORCH_NO_NATIVE_GPU=1 LT_BUILD_JOBS=1 python3 setup.py build_ext --inplace
```

Biên dịch bộ test C++ độc lập bằng CMake:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

---

## 2.5. Các biến môi trường

| Tên biến | Giá trị mặc định | Giải thích chức năng |
| :--- | :--- | :--- |
| `LITETORCH_NO_NATIVE_GPU` | `0` | Đặt thành `1` để vô hiệu hóa CUDA / ROCm native và fallback sang OpenCL, TPU hoặc CPU |
| `LITETORCH_NO_TPU` | `0` | Đặt thành `1` để tắt tự động nhận diện phần cứng TPU |
| `LIBTPU_PATH` / `TPU_LIBRARY_PATH` | Không có | Đường dẫn tùy chỉnh tới driver `libtpu.so` |
| `COLAB_TPU_ADDR` / `TPU_NAME` | Không có | Địa chỉ endpoint kết nối Cloud TPU / Colab TPU |
| `LITETORCH_RENDEZVOUS_FILE` | Không có | Đường dẫn tập tin dùng cho cơ chế rendezvous phân tán |
| `LITETORCH_NUM_THREADS` | Số core CPU | Số lượng luồng làm việc trong ThreadPool của CPU |
| `LITETORCH_CUDNN_ROOT` | Không có | Thư mục cài đặt cuDNN tùy chỉnh cho convolution tăng tốc |

---

## 2.6. Script kiểm tra cài đặt

Chạy đoạn mã Python sau để kiểm tra việc cài đặt. Đoạn mã tự chứa: nó chạy được chỉ với preamble `sys.path` do harness cung cấp.

<!-- ex: id=2.6-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

```python
import litetorch as lt

is_gpu = lt.is_gpu_available()
backend = lt.get_backend_name()
device = lt.auto_device()

print(f"GPU Kha dung: {is_gpu}")
print(f"Backend dang su dung: {backend}")
print(f"Thiet bi mac dinh: {device}")

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, True)
y = lt.Ops.matmul(x, x)
loss = lt.Ops.sum(y)
loss.backward()

print("Kich thuoc ma tran ket qua:", y.shape)
print("Gia tri loss:", loss.item())
print("Gia tri dao ham cua x:", x.grad.to_vector())
```

Đã kiểm tra trên CPU, LiteTorch 88a4ba1 (parity với torch đã kiểm tra 2026-10-27 so với torch 2.14.1+cpu: matmul, loss và gradient khớp chính xác, max diff 0.0; các dòng thiết bị/backend là thông tin môi trường).

Kết quả kỳ vọng trên máy chỉ có CPU (các giá trị là chính xác với input này):

```
GPU Available: False
Backend Name: CPU
Active Device: cpu:0
Result shape: [2, 2]
Loss value: 54.0
Gradient: [7.0, 11.0, 9.0, 13.0]
```

Trên máy GPU, ba dòng đầu sẽ khác (`GPU Available: True`, tên backend CUDA, thiết bị `gpu:0`); loss và gradient giữ nguyên. Nếu `Active Device` in ra `cpu:0` trên máy mà bạn kỳ vọng có GPU, nghĩa là backend GPU chưa được nạp — hãy kiểm tra `nvcc --version`, biên dịch lại, và xác nhận dòng khởi tạo `Native GPU (CUDA/HIP)` xuất hiện. Đừng cho rằng GPU đang được dùng: `.to(device)` sẽ fallback sang CPU một cách âm thầm khi thiết bị tăng tốc không khả dụng.

---

## 2.7. Đọc tiếp ở đâu

- Mới với framework: [Bắt đầu nhanh](quickstart.md) (60 phút, chạy được trọn vẹn).
- Cài đặt trong container: [Cài đặt trên Docker](installation/docker.md).
- Cài đặt trong notebook: [Cài đặt trên Kaggle](installation/kaggle.md).
- Gặp lỗi: trang Xử lý sự cố và FAQ (đang viết) — lỗi biên dịch, OOM, fallback CPU âm thầm, nhận diện backend.
