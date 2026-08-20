from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class CustomBuildExt(build_ext):
    def run(self):
        nproc = os.cpu_count() or 4
        subprocess.check_call(["make", f"-j{nproc}"], cwd=SCRIPT_DIR)
        
        nvcc_bin = None
        for p in ["nvcc", "/usr/local/cuda/bin/nvcc"]:
            try:
                if subprocess.call(["which", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                    nvcc_bin = p
                    break
            except Exception:
                if os.path.exists(p):
                    nvcc_bin = p
                    break

        if nvcc_bin:
            os.makedirs(os.path.join(SCRIPT_DIR, "build"), exist_ok=True)
            cudnn_flag = ""
            if os.path.exists("/usr/local/cuda/lib64/libcudnn.so") or subprocess.call("ldconfig -p 2>/dev/null | grep -q libcudnn", shell=True) == 0:
                cudnn_flag = "-DUSE_CUDNN -lcudnn"
            
            cmd = (
                f"{nvcc_bin} -std=c++14 -O3 --shared -Xcompiler -fPIC -Wno-deprecated-gpu-targets "
                f"-Iinclude -Isrc/backend/gpu_native/common -Isrc/backend/gpu_native "
                f"src/backend/gpu_native/kernels.cu -o build/liblitetorch_gpu.so -lcublas {cudnn_flag}"
            )
            subprocess.call(cmd, shell=True, cwd=SCRIPT_DIR)

        super().run()

ext_modules = [
    Pybind11Extension(
        "litetorch",
        ["src/bindings/python_bindings.cpp"],
        include_dirs=[os.path.join(SCRIPT_DIR, "include")],
        library_dirs=[os.path.join(SCRIPT_DIR, "build")],
        libraries=["litetorch"],
        extra_compile_args=["-std=c++14", "-O3"],
        extra_link_args=[
            f"-Wl,-rpath,{os.path.join(SCRIPT_DIR, 'build')}",
            "-Wl,-rpath,$ORIGIN/build",
            "-Wl,-rpath,$ORIGIN"
        ],
    ),
]

setup(
    name="litetorch",
    version="0.1.0",
    author="LiteTorch Team",
    description="Python bindings for LiteTorch deep learning framework",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "demo_run.py=tests.demo_run:main",
            "test_litetorch.py=tests.test_litetorch:main",
        ],
    },
)
