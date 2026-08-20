# LiteTorch - C++ Dependency & Installation Requirements

This document provides a comprehensive guide to all required libraries, compilers, and hardware prerequisites to build and run LiteTorch natively on Linux.

---

## Automatic C++ Dependency Installer

LiteTorch includes an automatic script to install all essential C++ build tools and headers across various Linux distributions (Ubuntu, Debian, RHEL, Fedora, Arch Linux):

```bash
./install_deps.sh
```

---

## Manual Installation via Package Managers

### 1. Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y build-essential g++ make ocl-icd-opencl-dev opencl-headers python3-dev python3-pip
```

### 2. Fedora / RHEL / CentOS
```bash
sudo dnf install -y gcc-c++ make ocl-icd-devel opencl-headers python3-devel python3-pip
```

### 3. Arch Linux / Manjaro
```bash
sudo pacman -S --needed base-devel gcc make opencl-headers ocl-icd python-pip
```

---

## Native GPU Toolkit Setup Guide

### 1. NVIDIA CUDA Toolkit (Mode 1 - NVIDIA GPU Acceleration)
- **Official Download**: [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- **Hardware Requirement**: NVIDIA GPU with Compute Capability 6.0 or higher (Pascal, Volta, Turing, Ampere, Ada Lovelace, Hopper).
- **Ubuntu Installation**:
  ```bash
  sudo apt-get install -y cuda-toolkit-12-0 libcublas-dev libcudnn8-dev
  ```
- **Verify Compiler**:
  ```bash
  nvcc --version
  ```

---

### 2. AMD ROCm / HIP Stack (Mode 1 - AMD GPU Acceleration)
- **Official Documentation**: [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- **Hardware Requirement**: AMD GPU with CDNA, RDNA 2, or RDNA 3 architecture (Instinct MI100/MI200/MI300, Radeon RX 6000/7000 series).
- **Ubuntu Installation**:
  ```bash
  sudo apt-get install -y rocm-hip-sdk librocblas-dev miopen-hip
  ```
- **Verify Compiler**:
  ```bash
  hipcc --version
  ```

---

### 3. OpenCL Runtime (Mode 2 - OpenCL GPU / CPU Compatibility Mode)
- **OpenCL Headers (Ubuntu/Debian)**: `sudo apt-get install -y ocl-icd-opencl-dev opencl-headers`
- **Intel iGPU Driver**: `sudo apt-get install -y intel-opencl-icd`
- **MESA Driver (AMD/Intel/Mali)**: `sudo apt-get install -y mesa-opencl-icd`
