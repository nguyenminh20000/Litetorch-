from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import os
import sys
import glob
import shutil
import subprocess
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

cpp_sources = sorted(glob.glob(os.path.join(SCRIPT_DIR, "src", "**", "*.cpp"), recursive=True))

inc_dirs = [
    pybind11.get_include(),
    os.path.join(SCRIPT_DIR, "include"),
    os.path.join(SCRIPT_DIR, "src"),
] + [x[0] for x in os.walk(os.path.join(SCRIPT_DIR, "src"))]

class BuildExt(build_ext):
    def build_extensions(self):
        self.parallel = os.cpu_count() or 4
        compiler_type = self.compiler.compiler_type
        for ext in self.extensions:
            if compiler_type == "msvc":
                ext.extra_compile_args = ["/std:c++14", "/O2", "/EHsc", "/bigobj"]
            else:
                ext.extra_compile_args = ["-std=c++14", "-O3", "-fPIC"]
                if sys.platform.startswith("win"):
                    ext.extra_link_args = ["-static-libgcc", "-static-libstdc++"]
        try:
            super().build_extensions()
            nvcc_bin = None
            for candidate in [shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc", "/usr/local/cuda-12/bin/nvcc", "/usr/local/cuda-11/bin/nvcc"]:
                if candidate and os.path.exists(candidate):
                    nvcc_bin = candidate
                    break
            if nvcc_bin and not os.environ.get("LITETORCH_NO_NATIVE_GPU"):
                cu_src = os.path.join(SCRIPT_DIR, "src", "backend", "gpu_native", "kernels.cu")
                if os.path.exists(cu_src):
                    target_dir = self.build_lib
                    lib_name = "liblitetorch_gpu.dll" if sys.platform.startswith("win") else "liblitetorch_gpu.so"
                    out_so = os.path.join(target_dir, lib_name)
                    inc1 = os.path.join(SCRIPT_DIR, "include")
                    inc2 = os.path.join(SCRIPT_DIR, "src", "backend", "gpu_native")
                    inc3 = os.path.join(SCRIPT_DIR, "src", "backend", "gpu_native", "common")
                    cmd = [
                        nvcc_bin, "-O3", "--shared", "-Xcompiler", "-fPIC",
                        "-arch=native",
                        f"-I{inc1}", f"-I{inc2}", f"-I{inc3}",
                        cu_src, "-o", out_so,
                        "-lcublas", "-lcublasLt"
                    ]
                    try:
                        res = subprocess.run(cmd, capture_output=True, text=True)
                        if res.returncode != 0:
                            cmd_fallback = [
                                nvcc_bin, "-O3", "--shared", "-Xcompiler", "-fPIC",
                                f"-I{inc1}", f"-I{inc2}", f"-I{inc3}",
                                cu_src, "-o", out_so,
                                "-lcublas", "-lcublasLt"
                            ]
                            res = subprocess.run(cmd_fallback, capture_output=True, text=True)
                        if res.returncode == 0:
                            extra_dests = []
                            temp_dir = tempfile.gettempdir()
                            if temp_dir and os.path.exists(temp_dir):
                                extra_dests.append(os.path.join(temp_dir, lib_name))
                            if not sys.platform.startswith("win"):
                                extra_dests.extend(["/tmp/liblitetorch_gpu.so", "/usr/local/lib/liblitetorch_gpu.so"])
                            for extra_dest in extra_dests:
                                try:
                                    shutil.copyfile(out_so, extra_dest)
                                except Exception:
                                    pass
                    except Exception:
                        pass
        except Exception as e:
            sys.stderr.write("\n" + "=" * 70 + "\n")
            sys.stderr.write("LITETORCH BUILD ERROR:\n")
            sys.stderr.write(f"Compiler Type: {compiler_type}\n")
            sys.stderr.write(f"Platform: {sys.platform}\n")
            sys.stderr.write(f"Error Details: {str(e)}\n\n")
            if sys.platform.startswith("win"):
                sys.stderr.write("Windows Troubleshooting:\n")
                sys.stderr.write("1. Ensure Microsoft C++ Build Tools is installed: https://aka.ms/vs/17/release/vs_BuildTools.exe\n")
                sys.stderr.write("2. Select 'Desktop development with C++' during installation.\n")
                sys.stderr.write("3. Alternatively, install via PowerShell (Admin): winget install Microsoft.VisualStudio.2022.BuildTools\n")
            else:
                sys.stderr.write("Linux Troubleshooting:\n")
                sys.stderr.write("1. Install C++ build tools: sudo apt-get install -y build-essential python3-dev\n")
                sys.stderr.write("2. Ensure g++ >= 7.0 is available.\n")
            sys.stderr.write("=" * 70 + "\n\n")
            raise

ext_modules = [
    Extension(
        "litetorch",
        cpp_sources,
        include_dirs=inc_dirs,
        libraries=["pthread", "ws2_32"] if sys.platform.startswith("win") else ["pthread", "dl", "rt"],
        language="c++",
    ),
]

readme_file = os.path.join(SCRIPT_DIR, "README.md")
long_desc = ""
if os.path.exists(readme_file):
    with open(readme_file, "r", encoding="utf-8") as f:
        long_desc = f.read()

setup(
    name="litetorch",
    version="0.3.16",
    author="LiteTorch Team",
    description="Python bindings for LiteTorch deep learning framework",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/nguyenminh20000/Litetorch-",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "demo_run.py=tests.demo_run:main",
            "test_litetorch.py=tests.test_litetorch:main",
        ],
    },
)
