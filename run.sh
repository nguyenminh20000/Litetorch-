#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ORIGINAL_DIR="$(pwd)"

cd "$SCRIPT_DIR"
mkdir -p build

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

GPU_MODE="1"
if [ "$1" == "--enable-gpu-native" ]; then
    GPU_MODE="1"
    shift
    if [ "$1" == "1" ]; then
        shift
    fi
elif [ "$1" == "1" ]; then
    GPU_MODE="1"
    shift
elif [ "$1" == "--disable-gpu-native" ]; then
    GPU_MODE="2"
    shift
    if [ "$1" == "2" ]; then
        shift
    fi
elif [ "$1" == "2" ]; then
    GPU_MODE="2"
    shift
fi

if [ "$GPU_MODE" == "1" ]; then
    echo "=================================================="
    echo " Mode 1: Native GPU Backend Auto-Detection (CUDA / ROCm)"
    echo "=================================================="
    if command -v nvcc >/dev/null 2>&1; then
        echo "Nvidia CUDA nvcc detected. Compiling native GPU library..."
        NVCC_FLAGS="-std=c++14 -O3 --shared -Xcompiler -fPIC -Wno-deprecated-gpu-targets -Iinclude -Isrc/backend/gpu_native/common -Isrc/backend/gpu_native"
        NVCC_LIBS="-lcublas"
        if [ -f /usr/local/cuda/lib64/libcudnn.so ] || ldconfig -p 2>/dev/null | grep -q libcudnn; then
            echo "Found cuDNN. Linking -lcudnn..."
            NVCC_LIBS="$NVCC_LIBS -lcudnn"
            NVCC_FLAGS="$NVCC_FLAGS -DUSE_CUDNN"
        fi
        nvcc $NVCC_FLAGS src/backend/gpu_native/kernels.cu -o build/liblitetorch_gpu.so $NVCC_LIBS
    elif [ -f /usr/local/cuda/bin/nvcc ]; then
        echo "Nvidia CUDA nvcc found in /usr/local/cuda/bin. Compiling native GPU library..."
        NVCC_FLAGS="-std=c++14 -O3 --shared -Xcompiler -fPIC -Wno-deprecated-gpu-targets -Iinclude -Isrc/backend/gpu_native/common -Isrc/backend/gpu_native"
        NVCC_LIBS="-lcublas"
        if [ -f /usr/local/cuda/lib64/libcudnn.so ] || ldconfig -p 2>/dev/null | grep -q libcudnn; then
            echo "Found cuDNN. Linking -lcudnn..."
            NVCC_LIBS="$NVCC_LIBS -lcudnn"
            NVCC_FLAGS="$NVCC_FLAGS -DUSE_CUDNN"
        fi
        /usr/local/cuda/bin/nvcc $NVCC_FLAGS src/backend/gpu_native/kernels.cu -o build/liblitetorch_gpu.so $NVCC_LIBS
    elif command -v hipcc >/dev/null 2>&1; then
        echo "AMD HIP hipcc detected. Compiling native GPU library..."
        HIPCC_FLAGS="-std=c++14 -O3 --shared -fPIC -Iinclude -Isrc/backend/gpu_native/common -Isrc/backend/gpu_native"
        HIPCC_LIBS="-lrocblas"
        if [ -f /opt/rocm/lib/libMIOpen.so ] || ldconfig -p 2>/dev/null | grep -q libMIOpen; then
            echo "Found MIOpen. Linking -lMIOpen..."
            HIPCC_LIBS="$HIPCC_LIBS -lMIOpen"
            HIPCC_FLAGS="$HIPCC_FLAGS -DUSE_MIOPEN"
        fi
        hipcc $HIPCC_FLAGS -D__HIP_PLATFORM_AMD__ src/backend/gpu_native/kernels.cu -o build/liblitetorch_gpu.so $HIPCC_LIBS
    elif [ -f /opt/rocm/bin/hipcc ]; then
        echo "AMD HIP hipcc found in /opt/rocm/bin. Compiling native GPU library..."
        HIPCC_FLAGS="-std=c++14 -O3 --shared -fPIC -Iinclude -Isrc/backend/gpu_native/common -Isrc/backend/gpu_native"
        HIPCC_LIBS="-lrocblas"
        if [ -f /opt/rocm/lib/libMIOpen.so ] || ldconfig -p 2>/dev/null | grep -q libMIOpen; then
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

build_core_library() {
    echo "--> Incremental parallel build of core library (liblitetorch.so) via make -j$NPROC..."
    make -j"$NPROC" --no-print-directory
}

build_python_module() {
    if python3 -c "import pybind11" >/dev/null 2>&1; then
        echo "--> Building LiteTorch Python extension module..."
        PY_INCLUDES=$(python3 -m pybind11 --includes 2>/dev/null || echo "-I$(python3 -c 'import pybind11; print(pybind11.get_include())') -I$(python3 -c 'import sysconfig; print(sysconfig.get_path(\"include\"))')")
        PY_SUFFIX=$(python3-config --extension-suffix 2>/dev/null || python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")
        g++ -std=c++14 -O3 -shared -fPIC $PY_INCLUDES -Iinclude src/bindings/python_bindings.cpp -Lbuild -llitetorch -Wl,-rpath,'$ORIGIN/build' -Wl,-rpath,"$SCRIPT_DIR/build" -o "litetorch$PY_SUFFIX"
        echo "--> Python module built: litetorch$PY_SUFFIX"
    fi
}

compile_executable() {
    local src_file=$1
    local out_name=$2
    build_core_library
    g++ -std=c++14 -O3 -Iinclude "$src_file" -Lbuild -llitetorch -Wl,-rpath,"$SCRIPT_DIR/build" -o "build/$out_name" -lpthread -ldl
}

build_core_library
build_python_module

if [ -z "$1" ] || [ "$1" == "build" ] || [ "$1" == "all" ] || [ "$1" == "lib" ]; then
    echo "=================================================="
    echo "LiteTorch C++ & Python core libraries built successfully!"
    echo "=================================================="
    exit 0
fi

INPUT_ARG=$1

if [[ "$INPUT_ARG" == *.cpp ]] || [ -f "${INPUT_ARG}.cpp" ] || [ -f "$ORIGINAL_DIR/${INPUT_ARG}.cpp" ]; then

    if [[ "$INPUT_ARG" == *.cpp ]]; then
        RAW_ARG="$INPUT_ARG"
    else
        RAW_ARG="${INPUT_ARG}.cpp"
    fi

    if [[ "$RAW_ARG" = /* ]]; then
        TARGET_FILE="$RAW_ARG"
    else
        TARGET_FILE="$ORIGINAL_DIR/$RAW_ARG"
    fi

    if [ ! -f "$TARGET_FILE" ]; then
        echo "Error: File '$TARGET_FILE' does not exist."
        exit 1
    fi

    BASE_NAME=$(basename "$TARGET_FILE" .cpp)

    echo "--------------------------------------------------"
    echo "Compiling custom C++ test: $TARGET_FILE"
    echo "--------------------------------------------------"
    
    compile_executable "$TARGET_FILE" "$BASE_NAME"

    echo "--------------------------------------------------"
    echo "Executing custom test: $BASE_NAME"
    echo "--------------------------------------------------"
    if [ "$GPU_MODE" == "2" ]; then
        LITETORCH_NO_NATIVE_GPU=1 "./build/$BASE_NAME"
    else
        "./build/$BASE_NAME"
    fi

else

    TEST_FILE="tests/$INPUT_ARG.cpp"
    if [ -f "$TEST_FILE" ]; then
        echo "--------------------------------------------------"
        echo "Compiling standard test: $INPUT_ARG"
        echo "--------------------------------------------------"
        compile_executable "$TEST_FILE" "$INPUT_ARG"

        echo "--------------------------------------------------"
        echo "Executing test: $INPUT_ARG"
        echo "--------------------------------------------------"
        if [ "$GPU_MODE" == "2" ]; then
            LITETORCH_NO_NATIVE_GPU=1 "./build/$INPUT_ARG"
        else
            "./build/$INPUT_ARG"
        fi
    else
        echo "Error: Test '$INPUT_ARG' not found in tests/ directory."
        echo "Available test names: "
        for f in tests/*.cpp; do
            echo "  - $(basename "$f" .cpp)"
        done
        exit 1
    fi
fi
