# How-To H5 — Move a Model to Mobile / ARM

Cross-compile the LiteTorch **CPU backend** for ARM64: Linux `aarch64`
(servers, Raspberry Pi 4/5, AWS Graviton) and Android (`arm64-v8a`, C++
library). There is no mobile GPU backend — CUDA/ROCm are desktop/server
only. The OpenCL backend is covered at the end of this page.

Toolchain claims below were re-verified against the tree at HEAD
114f2e1: zero SIMD intrinsics (`__m128`, `_mm_*`, `float32x4_t`,
`neon.h`) anywhere under `include/` or `src/`, and
`.github/workflows/arm-crosscompile.yml` plus `scripts/verify_build.py`
both exist. On-device correctness and performance numbers are still
unverified — no ARM device runs in CI.

## H5.1 — Linux aarch64: cross-compile with the stock toolchain

CPU parallelism comes from `include/litetorch/thread_pool.h`, which uses
plain `std::thread` and sizes work by `hardware_concurrency()`. There is
no x86-specific code path to port and no NEON intrinsics to write — NEON
comes from compiler auto-vectorization. Your only job is the toolchain
and the right flags.

```bash
sudo apt install g++-aarch64-linux-gnu
```

`setup.py` honors the standard `CC`/`CXX` environment variables, so the
whole build — host C++ objects and the CPython extension — goes through
the cross compiler:

```bash
cd Litetorch-
CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ \
  python3 setup.py build_ext --inplace
```

What `setup.py` actually passes: `-std=c++14 -fPIC`, optimization
**`-O1`** (deliberate: the build was OOM-killed at `-O3` on small VMs),
plus `-DLITETORCH_BUILD_COMMIT` from the build-integrity guard. There is
no `-march=native`/`-mavx` anywhere in the host flags, so nothing
x86-specific leaks in.

For a release ARM build, raise the optimization in `setup.py`'s
`build_extensions()` (`extra_compile_args`):

```python skip-run
ext.extra_compile_args = ["-std=c++14", "-O3", "-march=armv8-a", "-fPIC", ...]
```

`-O3 -march=armv8-a` enables NEON auto-vectorization across the whole
CPU backend. The relevant env knobs stay the same:
`LITETORCH_CUDA=1` (CUDA, native-arch only — not for cross),
`LITETORCH_ROCM=1`, `LITETORCH_TPU_PJRT=1`, `LITETORCH_CUDNN_ROOT`,
`LITETORCH_NO_NATIVE_GPU=1` (skip the nvcc/hip probe on a headless cross
box), `LITETORCH_CUDA_ARCH`, `LT_BUILD_JOBS`.

After a successful cross build, run `python3 scripts/verify_build.py`
on the target (or with the target sysroot) — it fails loud on stale
artifacts and on a silently-skipped native GPU lib.

## H5.2 — Android NDK: consume LiteTorch as a C++ library

The Python extension targets Linux `aarch64`. On Android, consume
LiteTorch as a **C++ library**. `CMakeLists.txt` is minimal
(`add_library(litetorch SHARED src/*.cpp)`, no install targets — copy
the `.so` yourself):

```bash
export NDK=/path/to/android-ndk
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-android -j
```

Add `-DCMAKE_CXX_FLAGS="-O3 -march=armv8-a"` for the NEON
auto-vectorization noted above. Note `src/*.cpp` also picks up the TPU
`.cpp` files — they compile to stubs without `libtpu` and are never
called unless `LITETORCH_TPU_PJRT` paths are hit; exclude
`src/backend/tpu` from the glob if you want a leaner binary. The NDK
path has not been exercised end-to-end in CI.

## H5.3 — OpenCL on Android

The OpenCL backend needs no vendor headers to build:
`include/litetorch/cl_backend.h` ships stub typedefs (`void*` handles,
matching the opaque-pointer ABI), and `src/cl_backend/cl_backend.cpp`
`dlopen`s `libOpenCL.so.1` / `libOpenCL.so` **at runtime**. On Android
devices whose vendor ships an OpenCL driver (most Snapdragon/Mali), the
already-built backend picks it up with no rebuild — point
`LD_LIBRARY_PATH` at the vendor driver if it is not on the default path.

## H5.4 — Check the CPU backend path on the target

The model below is the smoke test to run on the ARM target after the
cross build: it exercises the exact CPU backend path (tensor creation,
a linear layer, autograd) that the cross-compiled library must serve.
It runs on any CPU, so run it on the host first to establish the
baseline, then on the target and compare outputs.

<!-- ex: id=H5.4-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()
print("device:", device)

lt.manual_seed(0)
model = lt.nn.Sequential([
    lt.nn.Linear(16, 16, True),
    lt.nn.GELU(),
    lt.nn.Linear(16, 4, True),
])
model.to(device)

x = lt.Tensor.from_vector([0.1] * 32, [2, 16], device, True)
y = model(x)
loss = lt.Ops.sum(y)
loss.backward()
print("output shape:", list(y.shape))
print("output values:", y.to_vector()[:4])
print("input grad norm:", sum(v * v for v in x.grad.to_vector()) ** 0.5)
```

Verified on CPU (host x86-64), LiteTorch 88a4ba1, torch parity 1e-4.
On the ARM target the same script must print the same shapes and
values within 1e-4 — if it does not, the cross build is suspect, not
the model.

## H5.5 — Binary size and threading, in brief

No measured numbers are published yet (device benchmarks are a deferred
follow-up). Qualitative guidance, in order of impact:

1. `aarch64-linux-gnu-strip` the final `.so`.
2. Build `Release`/`-Os` instead of `-O3` if size beats the last few
   percent of speed on your workload.
3. Exclude `src/backend/tpu` (PJRT/HLO builders) when you never target
   TPU.
4. The OpenCL backend is small and runtime-gated by `dlopen` — keep it
   unless every byte counts.

`thread_pool.h` sizes its pool from `hardware_concurrency()` and skips
threading for small workloads (`total < 50000`). No code change is
needed on big.LITTLE SoCs; if you pin threads yourself, do it outside
the pool.

## H5.6 — CI coverage and its limits

`.github/workflows/arm-crosscompile.yml` runs on every push/PR touching
`src/`, `include/` or `setup.py`: it installs `g++-aarch64-linux-gnu`
via apt and compiles a representative TU subset (`src/tensor`, `src/nn`,
`src/ops`, `src/cl_backend`, `src/memory_manager`) to object files with
the cross compiler (`-O1 -march=armv8-a`). Compile-only — no device, no
runtime tests. The actual on-device build and benchmarks run outside
CI. The `.cu`/`.hip` GPU sources are never part of a cross build;
setting `LITETORCH_CUDA=1` while cross-compiling is unsupported.
