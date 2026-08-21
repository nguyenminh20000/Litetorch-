from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import os
import sys
import glob

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
        try:
            super().build_extensions()
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
    version="0.2.4",
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
