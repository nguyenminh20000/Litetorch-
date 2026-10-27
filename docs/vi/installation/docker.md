# Cài đặt trên Docker

Docker mang lại một bản cài đặt LiteTorch có thể tái lập: trình biên dịch, Python và (đối với image CUDA) CUDA toolkit được ghim trong image thay vì cài trên máy chủ. Có hai image: image CPU và image CUDA.

> Ghi chú trạng thái: các lệnh Docker dưới đây được viết từ quy trình biên dịch từ mã nguồn của LiteTorch và hợp đồng CLI của Docker. Docker không được cài trên máy kiểm tra tài liệu, nên các lần build image trên trang này chưa được chạy tại đây; nội dung `Dockerfile` CPU và CUDA là deliverable đã được xem xét, còn bản thân quá trình build là một hạng mục kiểm tra của Batch B5. Đừng coi các lệnh này là đã chạy kiểm chứng cho đến khi hạng mục đó đóng lại.

---

## Image CPU

`Dockerfile.cpu`:

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential python3-dev python3-pip libopenblas-dev git cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pybind11 numpy

WORKDIR /opt
RUN git clone https://github.com/nguyenminh20000/Litetorch-.git litetorch
WORKDIR /opt/litetorch

RUN LITETORCH_NO_NATIVE_GPU=1 python3 setup.py build_ext --inplace

ENV PYTHONPATH=/opt/litetorch/python:$PYTHONPATH
CMD ["python3", "-c", "import litetorch as lt; print(lt.get_backend_name(), lt.auto_device())"]
```

Build và chạy:

```bash
docker build -f Dockerfile.cpu -t litetorch:cpu .
docker run --rm litetorch:cpu
```

Kết quả kỳ vọng của lệnh chạy:

```
CPU cpu:0
```

Lưu ý:

- Lệnh build đặt `LITETORCH_NO_NATIVE_GPU=1`, nên không cần CUDA toolkit và bước biên dịch thư viện GPU native được bỏ qua.
- `PYTHONPATH` trỏ tới `/opt/litetorch/python` vì `setup.py` dùng `package_dir={"": "python"}` — extension đã biên dịch nằm dưới `<tree>/python/`, không phải thư mục gốc của cây mã nguồn. Thiếu mục này, `import litetorch` sẽ lỗi `ModuleNotFoundError`.
- Trên máy nhỏ, hãy giảm áp lực bộ nhớ khi build bằng cách giới hạn số job biên dịch: truyền `LT_BUILD_JOBS=1` trong dòng `RUN` (nó cũng ép `-O1` cho các translation unit binding).

---

## Image CUDA

`Dockerfile.cuda`:

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential python3-dev python3-pip libopenblas-dev git cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pybind11 numpy

WORKDIR /opt
RUN git clone https://github.com/nguyenminh20000/Litetorch-.git litetorch
WORKDIR /opt/litetorch

RUN python3 setup.py build_ext --inplace

ENV PYTHONPATH=/opt/litetorch/python:$PYTHONPATH
CMD ["python3", "-c", "import litetorch as lt; print(lt.get_backend_name(), lt.auto_device(), lt.is_gpu_available())"]
```

Build và chạy (cần NVIDIA Container Toolkit trên máy chủ và một GPU):

```bash
docker build -f Dockerfile.cuda -t litetorch:cuda .
docker run --rm --gpus all litetorch:cuda
```

Hãy kiểm tra CUDA toolkit bên trong image trước khi tin vào một lần chạy GPU — quy tắc bắt buộc này cũng áp dụng trong container:

```bash
docker run --rm litetorch:cuda nvcc --version
```

Nếu thiếu `nvcc`, thư viện GPU native không bao giờ được biên dịch và framework sẽ fallback sang CPU một cách âm thầm. Kết quả kỳ vọng của lệnh chạy trên máy có GPU:

```
CUDA gpu:0 True
```

(Tên backend chính xác và chỉ số thiết bị phụ thuộc vào GPU của máy chủ. Nếu kết quả hiển thị `CPU cpu:0 False`, nghĩa là backend GPU chưa được nạp — hãy kiểm tra `nvcc --version` bên trong image, biên dịch lại, và xác nhận dòng khởi tạo `Native GPU (CUDA/HIP)` xuất hiện.)

---

## Dùng mã nguồn của bạn với image

Mount một thư mục làm việc và chạy script bên trong container:

```bash
docker run --rm --gpus all -v "$PWD":/work -w /work litetorch:cuda python3 train.py
```

Đối với công việc chỉ dùng CPU, hãy bỏ `--gpus all` và dùng `litetorch:cpu`. Script của bạn chỉ cần thêm `<tree>/python` vào `sys.path` khi nó import một cây mã nguồn tự biên dịch; bên trong các image này `PYTHONPATH` đã bao phủ sẵn.

---

## Xử lý sự cố

- `ModuleNotFoundError: No module named 'litetorch'` — `PYTHONPATH` không bao gồm `<tree>/python`. Hãy kiểm tra `echo $PYTHONPATH` bên trong container và xác nhận file `.so` tồn tại dưới `/opt/litetorch/python/`.
- Quá trình build in ra `BUILD OK` nhưng import thất bại — `setup.py` in ra `BUILD OK` ngay cả khi một số file biên dịch lỗi. Cổng kiểm tra thực sự là file extension tồn tại và `import litetorch` thành công.
- Lần chạy GPU âm thầm dùng CPU — xem mục image CUDA ở trên: kiểm tra `nvcc`, biên dịch lại, và xác nhận dòng khởi tạo `Native GPU (CUDA/HIP)`.
- Hết bộ nhớ trong lúc build — đặt `LT_BUILD_JOBS=1` trong quá trình build image để giảm bộ nhớ đỉnh.
