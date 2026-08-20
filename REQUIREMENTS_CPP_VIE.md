# LiteTorch - Hướng Dẫn Yêu Cầu & Cài Đặt Thư Viện C++ (Bản Tiếng Việt)

Tài liệu này chi tiết hóa toàn bộ các thư viện, trình biên dịch và phần cứng C++ cần thiết để biên dịch và vận hành LiteTorch nguyên bản trên Linux.

---

## Script Cài Đặt Tự Động Thư Viện C++ (Linux Auto-Installer)

LiteTorch cung cấp script cài đặt tự động toàn bộ thư viện C++ cơ bản cho Linux (tự động nhận diện Ubuntu, Debian, RHEL, Fedora, Arch Linux):

```bash
./install_deps.sh
```

---

## Hướng Dẫn Cài Đặt C++ Thủ Công Theo Trình Quản Lý Gói

### 1. Hệ Điều Hành Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y build-essential g++ make ocl-icd-opencl-dev opencl-headers python3-dev python3-pip
```

### 2. Hệ Điều Hành Fedora / RHEL / CentOS
```bash
sudo dnf install -y gcc-c++ make ocl-icd-devel opencl-headers python3-devel python3-pip
```

### 3. Hệ Điều Hành Arch Linux / Manjaro
```bash
sudo pacman -S --needed base-devel gcc make opencl-headers ocl-icd python-pip
```

---

## Hướng Dẫn Cài Đặt Bộ Công Cụ GPU Nguyên Bản (Native GPU Toolkits)

### 1. NVIDIA CUDA Toolkit (Mode 1 - Tăng Tốc GPU NVIDIA)
- **Trang chủ Tải về**: [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- **Yêu cầu Phần cứng**: GPU NVIDIA Compute Capability 6.0 trở lên (Pascal, Volta, Turing, Ampere, Ada Lovelace, Hopper).
- **Lệnh cài đặt trên Ubuntu**:
  ```bash
  sudo apt-get install -y cuda-toolkit-12-0 libcublas-dev libcudnn8-dev
  ```
- **Kiểm tra biên dịch**:
  ```bash
  nvcc --version
  ```

---

### 2. AMD ROCm / HIP Stack (Mode 1 - Tăng Tốc GPU AMD)
- **Trang chủ Tài liệu**: [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- **Yêu cầu Phần cứng**: GPU AMD kiến trúc CDNA, RDNA 2, hoặc RDNA 3 (Instinct MI100/MI200/MI300, Radeon RX 6000/7000 series).
- **Lệnh cài đặt trên Ubuntu**:
  ```bash
  sudo apt-get install -y rocm-hip-sdk librocblas-dev miopen-hip
  ```
- **Kiểm tra biên dịch**:
  ```bash
  hipcc --version
  ```

---

### 3. OpenCL Runtime (Mode 2 - Chế Độ Tương Thích OpenCL GPU / CPU)
- **Cài đặt Header OpenCL (Ubuntu/Debian)**: `sudo apt-get install -y ocl-icd-opencl-dev opencl-headers`
- **Driver Intel iGPU**: `sudo apt-get install -y intel-opencl-icd`
- **Driver MESA (AMD/Intel/Mali)**: `sudo apt-get install -y mesa-opencl-icd`
