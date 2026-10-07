from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import glob
import shutil
import subprocess
import tempfile

try:
    import pybind11
except ImportError:
    sys.stderr.write("pybind11 is required to build litetorch: pip install pybind11\n")
    raise

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _cudnn_link_files(shared_libs):
    by_base = {}
    for path in sorted(shared_libs):
        base = os.path.basename(path)
        key = base.split(".so")[0].split(".dylib")[0]
        by_base.setdefault(key, []).append(path)
    chosen = []
    for key in sorted(by_base):
        paths = by_base[key]
        plain = [p for p in paths if os.path.basename(p) in (key + ".so", key + ".dylib")]
        chosen.append(plain[0] if plain else sorted(paths)[-1])
    return chosen

def _cudnn_major(include_dir):
    for name in ("cudnn_version.h", "cudnn.h"):
        version_file = os.path.join(include_dir, name)
        if not os.path.isfile(version_file):
            continue
        try:
            with open(version_file, "r", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3 and parts[0] == "#define" and parts[1] == "CUDNN_MAJOR":
                        return parts[2]
        except Exception:
            pass
    return "?"

def _cudnn_layout(root):
    if not root or not os.path.isdir(root):
        return None
    inc = None
    for d in (os.path.join(root, "include"), root):
        if os.path.isfile(os.path.join(d, "cudnn.h")):
            inc = d
            break
    lib = None
    found = []
    for d in (os.path.join(root, "lib"), os.path.join(root, "lib64"), root):
        hits = sorted(glob.glob(os.path.join(d, "libcudnn*.so*"))) + \
               sorted(glob.glob(os.path.join(d, "libcudnn*.dylib")))
        if hits:
            lib = d
            found = hits
            break
    if inc and lib and found:
        return {
            "include_dir": inc,
            "lib_dir": lib,
            "link_files": _cudnn_link_files(found),
            "major": _cudnn_major(inc),
        }
    return None

_CUDNN_SITE_PATTERNS = (
    "/usr/local/lib/python*/dist-packages/nvidia/cudnn",
    "/usr/lib/python*/dist-packages/nvidia/cudnn",
    "/usr/local/lib/python*/site-packages/nvidia/cudnn",
    "/usr/lib/python*/site-packages/nvidia/cudnn",
    os.path.expanduser("~/.local/lib/python*/site-packages/nvidia/cudnn"),
)

def _cudnn_site_roots():
    roots = []
    for pattern in _CUDNN_SITE_PATTERNS:
        roots.extend(sorted(glob.glob(pattern)))
    return roots

def find_cudnn():
    if os.environ.get("LITETORCH_USE_CUDNN", "").strip() == "0":
        print("[litetorch] cuDNN disabled via LITETORCH_USE_CUDNN=0")
        return None
    explicit = os.environ.get("LITETORCH_CUDNN_ROOT", "").strip()
    if explicit:
        info = _cudnn_layout(explicit)
        if info:
            print(f"[litetorch] cuDNN {info['major']} detected via LITETORCH_CUDNN_ROOT={explicit}")
            return info
        print(f"[litetorch] LITETORCH_CUDNN_ROOT={explicit} has no cuDNN layout, trying other locations")
    for root in _cudnn_site_roots():
        info = _cudnn_layout(root)
        if info:
            print(f"[litetorch] cuDNN {info['major']} detected at {root}")
            return info
    candidates = []
    for mod in ("nvidia.cudnn",):
        try:
            m = __import__(mod, fromlist=[""])
            paths = list(getattr(m, "__path__", []))
            candidates.extend(paths)
        except Exception:
            pass
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia.cudnn")
        if spec and spec.submodule_search_locations:
            candidates.extend([str(p) for p in spec.submodule_search_locations])
    except Exception:
        pass
    try:
        import torch
        torch_site = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
        candidates.append(os.path.join(torch_site, "nvidia", "cudnn"))
    except Exception:
        pass
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(cuda_home)
    candidates.extend(["/usr/local/cuda", "/usr", "/opt/cuda"])
    for root in candidates:
        info = _cudnn_layout(root)
        if info:
            return info
    return None

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
            use_rocm = bool(os.environ.get("LITETORCH_ROCM"))
            use_cuda = bool(os.environ.get("LITETORCH_CUDA"))
            nvcc_bin = None
            if not use_rocm:
                for candidate in [shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc", "/usr/local/cuda-12/bin/nvcc", "/usr/local/cuda-11/bin/nvcc"]:
                    if candidate and os.path.exists(candidate):
                        nvcc_bin = candidate
                        break
            hipcc_bin = None
            if not use_cuda:
                rocm_candidates = [shutil.which("hipcc")]
                rocm_env = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH")
                if rocm_env:
                    rocm_candidates.extend([
                        os.path.join(rocm_env, "bin", "hipcc"),
                        os.path.join(rocm_env, "bin", "hipcc.bat"),
                        os.path.join(rocm_env, "bin", "hipcc.exe")
                    ])
                rocm_candidates.extend([
                    "/opt/rocm/bin/hipcc",
                    "/opt/rocm/hip/bin/hipcc",
                    "/usr/bin/hipcc"
                ])
                for candidate in rocm_candidates:
                    if candidate and os.path.exists(candidate):
                        hipcc_bin = candidate
                        break
            if not os.environ.get("LITETORCH_NO_NATIVE_GPU"):
                target_dir = self.build_lib
                lib_name = "liblitetorch_gpu.dll" if sys.platform.startswith("win") else "liblitetorch_gpu.so"
                out_so = os.path.join(target_dir, lib_name)
                inc1 = os.path.join(SCRIPT_DIR, "include")
                inc2 = os.path.join(SCRIPT_DIR, "src", "backend", "gpu_native")
                inc3 = os.path.join(SCRIPT_DIR, "src", "backend", "gpu_native", "common")
                build_success = False
                if nvcc_bin and not use_rocm:
                    cu_src = os.path.join(SCRIPT_DIR, "src", "backend", "gpu_native", "kernels.cu")
                    if os.path.exists(cu_src):
                        cuda_arch = os.environ.get("LITETORCH_CUDA_ARCH", "native")
                        cudnn_args = []
                        cudnn_info = find_cudnn()
                        if cudnn_info:
                            cudnn_args = [
                                "-DUSE_CUDNN",
                                "-I" + cudnn_info["include_dir"],
                                "-L" + cudnn_info["lib_dir"],
                            ] + cudnn_info["link_files"] + [
                                "-Xlinker", "-rpath", "-Xlinker", cudnn_info["lib_dir"],
                            ]
                            sys.stdout.write(
                                "[litetorch] cuDNN %s detected at %s, enabling USE_CUDNN\n"
                                % (cudnn_info["major"], cudnn_info["lib_dir"]))
                            sys.stdout.flush()
                        else:
                            sys.stdout.write("[litetorch] cuDNN not found, building GPU lib without cuDNN\n")
                            sys.stdout.flush()
                        cmd = [
                            nvcc_bin, "-O3", "--shared", "-Xcompiler", "-fPIC",
                            f"-arch={cuda_arch}",
                            f"-I{inc1}", f"-I{inc2}", f"-I{inc3}",
                            cu_src, "-o", out_so,
                            "-lcublas", "-lcublasLt"
                        ] + cudnn_args
                        try:
                            res = subprocess.run(cmd, capture_output=True, text=True)
                            if res.returncode != 0:
                                cmd_fallback = [
                                    nvcc_bin, "-O3", "--shared", "-Xcompiler", "-fPIC",
                                    f"-I{inc1}", f"-I{inc2}", f"-I{inc3}",
                                    cu_src, "-o", out_so,
                                    "-lcublas", "-lcublasLt"
                                ] + cudnn_args
                                res = subprocess.run(cmd_fallback, capture_output=True, text=True)
                            build_success = (res.returncode == 0)
                        except Exception:
                            pass
                elif hipcc_bin:
                    hip_src = os.path.join(SCRIPT_DIR, "src", "backend", "gpu_native", "kernels.hip")
                    if os.path.exists(hip_src):
                        extra_hip_args = ["-lrocblas"]
                        for miopen_path in ["/opt/rocm/lib/libMIOpen.so", "/opt/rocm/lib64/libMIOpen.so"]:
                            if os.path.exists(miopen_path):
                                extra_hip_args.extend(["-lMIOpen", "-DUSE_MIOPEN"])
                                break
                        cmd = [
                            hipcc_bin, "-O3", "--shared", "-fPIC", "-D__HIP_PLATFORM_AMD__",
                            f"-I{inc1}", f"-I{inc2}", f"-I{inc3}",
                            hip_src, "-o", out_so
                        ] + extra_hip_args
                        try:
                            res = subprocess.run(cmd, capture_output=True, text=True)
                            build_success = (res.returncode == 0)
                        except Exception:
                            pass
                if build_success:
                    extra_dests = []
                    temp_dir = tempfile.gettempdir()
                    if temp_dir and os.path.exists(temp_dir):
                        extra_dests.append(os.path.join(temp_dir, lib_name))
                    if not sys.platform.startswith("win"):
                        extra_dests.extend([
                            "/tmp/liblitetorch_gpu.so",
                            "/usr/local/lib/liblitetorch_gpu.so",
                            "/opt/rocm/lib/liblitetorch_gpu.so"
                        ])
                    for extra_dest in extra_dests:
                        try:
                            shutil.copyfile(out_so, extra_dest)
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
    version="0.3.40",
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
)
