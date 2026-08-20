#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ORIGINAL_DIR="$(pwd)"

cd "$SCRIPT_DIR"
mkdir -p build

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

GPU_MODE=${GPU_DRIVER_MODE:-"2"}
if [ "$1" == "--enable-gpu-native" ] || [ "$1" == "1" ]; then
    GPU_MODE="1"
    shift
elif [ "$1" == "--disable-gpu-native" ] || [ "$1" == "2" ]; then
    GPU_MODE="2"
    shift
fi

if [ "$GPU_MODE" == "1" ]; then
    echo "=================================================="
    echo " Mode 1: Native GPU Backend Auto-Detection (CUDA / ROCm)"
    echo "=================================================="
    if command -v nvcc >/dev/null 2>&1; then
        echo "Nvidia CUDA nvcc detected. Compiling native GPU library..."
        NVCC_FLAGS="-std=c++14 -O3 --shared -Xcompiler -fPIC -Iinclude -Isrc/backend/gpu_native/common"
        NVCC_LIBS="-lcublas"
        if [ -f /usr/local/cuda/lib64/libcudnn.so ] || ldconfig -p | grep -q libcudnn; then
            echo "Found cuDNN. Linking -lcudnn..."
            NVCC_LIBS="$NVCC_LIBS -lcudnn"
            NVCC_FLAGS="$NVCC_FLAGS -DUSE_CUDNN"
        fi
        nvcc $NVCC_FLAGS src/backend/gpu_native/kernels.cu -o build/liblitetorch_gpu.so $NVCC_LIBS
    elif [ -f /usr/local/cuda/bin/nvcc ]; then
        echo "Nvidia CUDA nvcc found in /usr/local/cuda/bin. Compiling native GPU library..."
        NVCC_FLAGS="-std=c++14 -O3 --shared -Xcompiler -fPIC -Iinclude -Isrc/backend/gpu_native/common"
        NVCC_LIBS="-lcublas"
        if [ -f /usr/local/cuda/lib64/libcudnn.so ] || ldconfig -p | grep -q libcudnn; then
            echo "Found cuDNN. Linking -lcudnn..."
            NVCC_LIBS="$NVCC_LIBS -lcudnn"
            NVCC_FLAGS="$NVCC_FLAGS -DUSE_CUDNN"
        fi
        /usr/local/cuda/bin/nvcc $NVCC_FLAGS src/backend/gpu_native/kernels.cu -o build/liblitetorch_gpu.so $NVCC_LIBS
    elif command -v hipcc >/dev/null 2>&1; then
        echo "AMD HIP hipcc detected. Compiling native GPU library..."
        HIPCC_FLAGS="-std=c++14 -O3 --shared -fPIC -Iinclude -Isrc/backend/gpu_native/common"
        HIPCC_LIBS="-lrocblas"
        if [ -f /opt/rocm/lib/libMIOpen.so ] || ldconfig -p | grep -q libMIOpen; then
            echo "Found MIOpen. Linking -lMIOpen..."
            HIPCC_LIBS="$HIPCC_LIBS -lMIOpen"
            HIPCC_FLAGS="$HIPCC_FLAGS -DUSE_MIOPEN"
        fi
        hipcc $HIPCC_FLAGS -D__HIP_PLATFORM_AMD__ src/backend/gpu_native/kernels.cu -o build/liblitetorch_gpu.so $HIPCC_LIBS
    elif [ -f /opt/rocm/bin/hipcc ]; then
        echo "AMD HIP hipcc found in /opt/rocm/bin. Compiling native GPU library..."
        HIPCC_FLAGS="-std=c++14 -O3 --shared -fPIC -Iinclude -Isrc/backend/gpu_native/common"
        HIPCC_LIBS="-lrocblas"
        if [ -f /opt/rocm/lib/libMIOpen.so ] || ldconfig -p | grep -q libMIOpen; then
            echo "Found MIOpen. Linking -lMIOpen..."
            HIPCC_LIBS="$HIPCC_LIBS -lMIOpen"
            HIPCC_FLAGS="$HIPCC_FLAGS -DUSE_MIOPEN"
        fi
        /opt/rocm/bin/hipcc $HIPCC_FLAGS -D__HIP_PLATFORM_AMD__ src/backend/gpu_native/kernels.cu -o build/liblitetorch_gpu.so $HIPCC_LIBS
    else
        echo "No native GPU compiler (nvcc/hipcc) found. Skipping native GPU library compilation."
    fi
else
    echo "=================================================="
    echo " Mode 2: OpenCL / CPU Testing Mode (Native GPU Skipped)"
    echo "=================================================="
fi

echo "--> Building core library liblitetorch.so..."
make -j"$NPROC" --no-print-directory

echo "--> Building litetorch Python Extension via pybind11..."
python3 setup.py build_ext --inplace --quiet

if [ -z "$1" ]; then
    echo ""
    echo "=================================================="
    echo "  LiteTorch Python Runner. Usage:"
    echo "=================================================="
    echo "  Mode 1 (Native GPU CUDA/ROCm): ./run_py.sh 1 <script.py>"
    echo "  Mode 2 (OpenCL / CPU Mode):   ./run_py.sh 2 <script.py>"
    echo "  Example: ./run_py.sh 2 tests/test_litetorch.py"
    echo "=================================================="
    exit 0
fi

PY_FILE="$1"
if [ ! -f "$PY_FILE" ] && [ -f "$ORIGINAL_DIR/$PY_FILE" ]; then
    PY_FILE="$ORIGINAL_DIR/$PY_FILE"
fi

if [ ! -f "$PY_FILE" ]; then
    echo "Error: Python script '$PY_FILE' not found."
    exit 1
fi

echo "--------------------------------------------------"
echo "Executing Python script: $PY_FILE"
echo "--------------------------------------------------"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

if [ "$GPU_MODE" == "2" ]; then
    LITETORCH_NO_NATIVE_GPU=1 python3 "$PY_FILE"
else
    python3 "$PY_FILE"
fi
