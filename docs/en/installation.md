# Installation Guide

This document covers installing LiteTorch on Linux, Windows, Docker, and Kaggle, and building it from source. For container installs see [Installation on Docker](installation/docker.md); for notebook installs see [Installation on Kaggle](installation/kaggle.md).

---

## 2.1. System requirements

- **C++ compiler**: C++14 compliant (`g++` >= 7.0, `clang++` >= 6.0, or MSVC v142+).
- **Python**: 3.8 to 3.12, with development headers (`python3-dev`).
- **CMake**: >= 3.10 (for building the standalone C++ tests).
- **pybind11**: required to build the Python extension (`pip install pybind11`).
- **Accelerators (optional)**:
  - NVIDIA: CUDA Toolkit 11.8 or 12.x with `nvcc`, `cuBLAS`, `cuBLASLt`. cuDNN is a separate library (not included in the CUDA Toolkit) — install it independently for accelerated convolutions; LiteTorch auto-detects it via `LITETORCH_CUDNN_ROOT` or standard paths.
  - Google TPU: Google Cloud TPU / Colab TPU (v2-v5p, v6e Trillium, v7 Ironwood, v8 8t/8i) with `libtpu.so` and the PJRT C-API.
  - AMD: ROCm 5.x or 6.x with `hipcc`, `rocBLAS`, `MIOpen`.
  - Other: OpenCL 1.2+ compatible drivers.

---

## 2.2. Linux and Google Colab

### 2.2.1. Install system dependencies

On Ubuntu / Debian / Google Colab:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev libopenblas-dev
```

On Fedora / RHEL / CentOS:

```bash
sudo dnf groupinstall "Development Tools" -y
sudo dnf install -y python3-devel openblas-devel
```

### 2.2.2. Install via pip

```bash
pip install --upgrade litetorch
```

### 2.2.3. Check the CUDA toolkit before any GPU run

This is a standing rule, not a suggestion: always check (and install if missing) the CUDA toolkit explicitly before a GPU CUDA run. A missing `nvcc` means the native GPU library never builds, and the framework then falls back to CPU silently (see Behavioral differences, in progress — the `to(device)` silent fallback).

```bash
nvcc --version
```

If `nvcc` is not found, install the CUDA Toolkit for your distribution (for example, the NVIDIA CUDA 12.x network repository for Ubuntu) and re-run `nvcc --version` until it prints a version. Only then build or run LiteTorch on GPU.

- **NVIDIA GPU**: LiteTorch automatically compiles and enables native CUDA acceleration when `nvcc` is present.
- **Google TPU**: in Google Colab or a Cloud TPU VM, LiteTorch automatically discovers `COLAB_TPU_ADDR`, `TPU_NAME`, or `libtpu.so` and binds to `lt.Device("tpu:0")` through `lt.Device("tpu:7")`.
- **AMD GPU**: check that `hipcc` is present for native ROCm acceleration. ROCm install steps below are documented from the ROCm toolchain layout; the ROCm examples in these docs are marked `UNVERIFIED-ROCm` until a ROCm run verifies them.

---

## 2.3. Windows

### 2.3.1. Install C++ build tools

Python on Windows requires MSVC or MinGW to compile C++ extensions. Choose one:

**Option A: Microsoft Visual C++ Build Tools (recommended)**

Run PowerShell as Administrator:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --force --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;includeRecommended"
```

**Option B: MinGW-w64 GCC**

1. Download MinGW-w64 (GCC 10+).
2. Add the `bin` directory of MinGW to your system `PATH`.

### 2.3.2. Install via pip

Open Command Prompt or Terminal:

```cmd
pip install --upgrade litetorch
```

---

## 2.4. Build from source

```bash
git clone https://github.com/nguyenminh20000/Litetorch-.git
cd Litetorch-
pip install -r requirements.txt
pip install pybind11
```

Build the Python extension in place:

```bash
python3 setup.py build_ext --inplace
```

Two build facts that cause most first-time failures:

1. **`setup.py` sets `package_dir={"": "python"}`, so `build_ext --inplace` puts the importable extension under `<tree>/python/`, not the tree root.** Every script must add `<tree>/python` to `sys.path` before `import litetorch`, or the import fails with `ModuleNotFoundError`.
2. **`setup.py` prints `BUILD OK` even when per-file compiles fail.** The real gate is the extension file existing under `<tree>/python/` and `import litetorch` succeeding. Never trust the `BUILD OK` line alone.

For a CPU-only build (skips the nvcc / hipcc native-library step):

```bash
LITETORCH_NO_NATIVE_GPU=1 python3 setup.py build_ext --inplace
```

On a small machine, limit build parallelism (also forces `-O1` on the binding translation units, which lowers peak memory):

```bash
LITETORCH_NO_NATIVE_GPU=1 LT_BUILD_JOBS=1 python3 setup.py build_ext --inplace
```

Compile the standalone C++ test binaries with CMake:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

---

## 2.5. Environment variables

| Environment variable | Default | Description |
| :--- | :--- | :--- |
| `LITETORCH_NO_NATIVE_GPU` | `0` | Set to `1` to disable native CUDA / ROCm and fall back to OpenCL, TPU, or CPU |
| `LITETORCH_NO_TPU` | `0` | Set to `1` to disable TPU discovery |
| `LIBTPU_PATH` / `TPU_LIBRARY_PATH` | None | Custom filepath to the `libtpu.so` driver |
| `COLAB_TPU_ADDR` / `TPU_NAME` | None | Cloud TPU / Colab TPU connection endpoint |
| `LITETORCH_RENDEZVOUS_FILE` | None | File path used for distributed rendezvous |
| `LITETORCH_NUM_THREADS` | CPU core count | Worker thread count in the CPU ThreadPool |
| `LITETORCH_CUDNN_ROOT` | None | Custom cuDNN install root for accelerated convolutions |

---

## 2.6. Verification script

Run the following Python code to verify the installation. It is self-contained: it runs with only the harness-provided `sys.path` preamble.

<!-- ex: id=2.6-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

```python
import litetorch as lt

is_gpu = lt.is_gpu_available()
backend = lt.get_backend_name()
device = lt.auto_device()

print(f"GPU Available: {is_gpu}")
print(f"Backend Name: {backend}")
print(f"Active Device: {device}")

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, True)
y = lt.Ops.matmul(x, x)
loss = lt.Ops.sum(y)
loss.backward()

print("Result shape:", y.shape)
print("Loss value:", loss.item())
print("Gradient:", x.grad.to_vector())
```

Verified on CPU, LiteTorch 88a4ba1 (torch parity verified 2026-10-27 vs torch 2.14.1+cpu: matmul, loss and gradients match exactly, max diff 0.0; device/backend lines are environment facts).

Expected output on a CPU-only machine (values are exact for this input):

```
GPU Available: False
Backend Name: CPU
Active Device: cpu:0
Result shape: [2, 2]
Loss value: 54.0
Gradient: [7.0, 11.0, 9.0, 13.0]
```

On a GPU machine the first three lines differ (`GPU Available: True`, a CUDA backend name, a `gpu:0` device); the loss and gradient are the same. If `Active Device` prints `cpu:0` on a machine where you expected a GPU, the GPU backend did not load — check `nvcc --version`, rebuild, and confirm the `Native GPU (CUDA/HIP)` init line appears. Do not assume the GPU is in use: `.to(device)` falls back to CPU silently when the accelerator is unavailable.

---

## 2.7. Where to go next

- New to the framework: [Quickstart](quickstart.md) (60 minutes, runnable end to end).
- Container install: [Installation on Docker](installation/docker.md).
- Notebook install: [Installation on Kaggle](installation/kaggle.md).
- Personal machine: [Installation on a Personal Machine](installation/local.md) — Linux workstation or Windows server, CPU or a single GPU, pre-flight gate, first local training run.
- Cloud (AWS): [Installation on AWS](installation/aws.md) — EC2 GPU instances (DLAMI or plain Ubuntu), SageMaker, ECS, EKS.
- Cloud (Azure): [Installation on Azure](installation/azure.md) — N-series GPU VMs, Azure ML compute.
- Cloud (GCP): [Installation on Google Cloud Platform](installation/gcp.md) — Deep Learning VM or plain-image GPU VMs, Vertex AI.
- Other clouds: [Installation on Other Clouds](installation/other-clouds.md) — Lambda, OCI, RunPod, Paperspace, CoreWeave.
- Something failed: Troubleshooting and FAQ (in progress) — build failures, OOM, silent CPU fallback, backend detection.
