# LiteTorch on ARM / Mobile — cross-compile guide

## English

### Scope

This document covers cross-compiling the LiteTorch **CPU backend** for
ARM64 targets: Linux `aarch64` (servers, Raspberry Pi 4/5, AWS Graviton)
and Android (`arm64-v8a`, C++ library). It does not cover GPU on mobile —
there is no mobile GPU backend; CUDA/ROCm are desktop/server only. The
OpenCL backend is covered separately below.

### Why the ARM story is toolchain-only

Verified against the tree at this commit: **zero** SIMD intrinsics
(`__m128`, `_mm_*`, `float32x4_t`, `neon.h`) anywhere under `include/` or
`src/`. CPU parallelism comes from `include/litetorch/thread_pool.h`, which
uses plain `std::thread` and sizes work by `hardware_concurrency()`. There
is no x86-specific code path to port and no NEON intrinsics to write —
NEON comes from compiler auto-vectorization. Your only job is the
toolchain and the right flags.

### Linux aarch64 (Debian/Ubuntu)

Install the cross toolchain:

```bash
sudo apt install g++-aarch64-linux-gnu
```

`setup.py` honors the standard `CC`/`CXX` environment variables (via
distutils), so the whole build — host C++ objects and the CPython
extension — goes through the cross compiler:

```bash
cd Litetorch-
CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ \
  python3 setup.py build_ext --inplace
```

What `setup.py` actually passes (re-checked against the flag surface):
`-std=c++14 -fPIC`, optimization **`-O1`** (deliberate: the build was
OOM-killed at `-O3` on small VMs), plus `-DLITETORCH_BUILD_COMMIT` from
the build-integrity guard. There is **no** `-march=native`/`-mavx`
anywhere in the host flags, so nothing x86-specific leaks in.

For a release ARM build, raise the optimization in `setup.py`'s
`build_extensions()` (`extra_compile_args`):

```python
ext.extra_compile_args = ["-std=c++14", "-O3", "-march=armv8-a", "-fPIC", ...]
```

`-O3 -march=armv8-a` enables NEON auto-vectorization across the whole CPU
backend. The relevant env knobs stay the same: `LITETORCH_CUDA=1`
(CUDA, native-arch only — not for cross), `LITETORCH_ROCM=1`,
`LITETORCH_TPU_PJRT=1`, `LITETORCH_CUDNN_ROOT`, `LITETORCH_NO_NATIVE_GPU=1`
(skip the nvcc/hip probe on a headless cross box),
`LITETORCH_CUDA_ARCH`, `LT_BUILD_JOBS`.

After a successful cross build, run `python3 scripts/verify_build.py`
on the target (or with the target sysroot) — it fails loud on stale
artifacts and on a silently-skipped native GPU lib.

### Android NDK (C++ library)

The Python extension targets Linux `aarch64`. On Android, consume
LiteTorch as a **C++ library**. `CMakeLists.txt` is minimal
(`add_library(litetorch SHARED src/*.cpp)`, no install targets — copy the
`.so` yourself):

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
auto-vectorization noted above. Note `src/*.cpp` glob also picks up the
TPU `.cpp` files — they compile to stubs without `libtpu` and are never
called unless `LITETORCH_TPU_PJRT` paths are hit; exclude
`src/backend/tpu` from the glob if you want a leaner binary.

### OpenCL on Android

The OpenCL backend needs no vendor headers to build:
`include/litetorch/cl_backend.h` ships stub typedefs (`void*` handles,
matching the opaque-pointer ABI), and `src/cl_backend/cl_backend.cpp`
`dlopen`s `libOpenCL.so.1` / `libOpenCL.so` **at runtime**. On Android
devices whose vendor ships an OpenCL driver (most Snapdragon/Mali),
the already-built backend picks it up with no rebuild — point
`LD_LIBRARY_PATH` at the vendor driver if it is not on the default path.

### Binary size

No measured numbers are published yet (device benchmarks are a deferred
follow-up). Qualitative guidance, in order of impact:

1. `aarch64-linux-gnu-strip` the final `.so`.
2. Build `Release`/`-Os` instead of `-O3` if size beats the last few
   percent of speed on your workload.
3. Exclude `src/backend/tpu` (PJRT/HLO builders) when you never target TPU.
4. The OpenCL backend is small and runtime-gated by `dlopen` — keep it
   unless every byte counts.

### Threading on mobile

`thread_pool.h` sizes its pool from `hardware_concurrency()` and skips
threading for small workloads (`total < 50000`). No code change is needed
on big.LITTLE SoCs; if you pin threads yourself, do it outside the pool.

### CI

`.github/workflows/arm-crosscompile.yml` runs on every push/PR touching
`src/`, `include/` or `setup.py`: it installs `g++-aarch64-linux-gnu`
via apt and compiles a representative TU subset
(`src/tensor`, `src/nn`, `src/ops`, `src/cl_backend`,
`src/memory_manager`) to object files with the cross compiler
(`-O1 -march=armv8-a`). Compile-only — no device, no runtime tests. The
actual on-device build and benchmarks run outside CI.

### Limitations

- No ARM device in CI: on-device correctness and performance numbers are
  unverified until device benchmarks run (deferred follow-up).
- The `.cu`/`.hip` GPU sources are never part of a cross build; setting
  `LITETORCH_CUDA=1` while cross-compiling is unsupported.
- The NDK path has not been exercised end-to-end in CI.

---

## Tiếng Việt

### Phạm vi

Tài liệu này hướng dẫn cross-compile **CPU backend** của LiteTorch cho
ARM64: Linux `aarch64` (server, Raspberry Pi 4/5, AWS Graviton) và Android
(`arm64-v8a`, thư viện C++). Không có GPU backend cho mobile —
CUDA/ROCm chỉ dành cho desktop/server. OpenCL backend được trình bày riêng
bên dưới.

### Vì sao câu chuyện ARM chỉ là toolchain

Đã kiểm chứng trên cây mã tại commit này: **không có** SIMD intrinsic
nào (`__m128`, `_mm_*`, `float32x4_t`, `neon.h`) trong `include/` hay
`src/`. Song song hóa CPU dùng `include/litetorch/thread_pool.h`
(`std::thread` thuần, cỡ pool theo `hardware_concurrency()`). Không có
code path riêng cho x86 cần port, không cần viết NEON intrinsic — NEON
có được nhờ compiler auto-vectorization. Việc duy nhất là toolchain và
flags đúng.

### Linux aarch64 (Debian/Ubuntu)

Cài toolchain:

```bash
sudo apt install g++-aarch64-linux-gnu
```

`setup.py` tôn trọng biến môi trường chuẩn `CC`/`CXX` (qua distutils), nên
toàn bộ build — object C++ host và CPython extension — đi qua cross
compiler:

```bash
cd Litetorch-
CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ \
  python3 setup.py build_ext --inplace
```

Flags `setup.py` thực sự truyền (đã kiểm tra lại flag surface):
`-std=c++14 -fPIC`, tối ưu **`-O1`** (cố ý: build từng bị OOM-kill ở
`-O3` trên VM nhỏ), cộng `-DLITETORCH_BUILD_COMMIT` từ build-integrity
guard. Không có `-march=native`/`-mavx` trong host flags, nên không có gì
đặc thù x86 lọt vào.

Để build ARM bản release, nâng tối ưu trong `build_extensions()` của
`setup.py` (`extra_compile_args`):

```python
ext.extra_compile_args = ["-std=c++14", "-O3", "-march=armv8-a", "-fPIC", ...]
```

`-O3 -march=armv8-a` bật NEON auto-vectorization cho toàn bộ CPU backend.
Các env knob giữ nguyên: `LITETORCH_CUDA=1` (CUDA, native-arch only —
không dùng khi cross), `LITETORCH_ROCM=1`, `LITETORCH_TPU_PJRT=1`,
`LITETORCH_CUDNN_ROOT`, `LITETORCH_NO_NATIVE_GPU=1` (bỏ qua probe
nvcc/hip trên máy cross headless), `LITETORCH_CUDA_ARCH`, `LT_BUILD_JOBS`.

Sau khi cross build thành công, chạy `python3 scripts/verify_build.py`
trên target (hoặc với target sysroot) — script fail-loud khi artifact cũ
hoặc khi native GPU lib bị bỏ qua trong im lặng.

### Android NDK (thư viện C++)

Python extension nhắm tới Linux `aarch64`. Trên Android, dùng LiteTorch
như **thư viện C++**. `CMakeLists.txt` tối giản
(`add_library(litetorch SHARED src/*.cpp)`, không có install target —
tự copy `.so`):

```bash
export NDK=/path/to/android-ndk
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-android -j
```

Thêm `-DCMAKE_CXX_FLAGS="-O3 -march=armv8-a"` để có NEON
auto-vectorization như trên. Lưu ý glob `src/*.cpp` cũng lấy các file
`.cpp` của TPU — chúng compile thành stub khi không có `libtpu` và không
bao giờ được gọi trừ khi đi vào path `LITETORCH_TPU_PJRT`; loại
`src/backend/tpu` khỏi glob nếu muốn binary gọn hơn.

### OpenCL trên Android

OpenCL backend không cần vendor header để build:
`include/litetorch/cl_backend.h` mang sẵn stub typedef (`void*` handle,
khớp opaque-pointer ABI), và `src/cl_backend/cl_backend.cpp` `dlopen`
`libOpenCL.so.1` / `libOpenCL.so` **lúc runtime**. Trên thiết bị Android
có vendor driver OpenCL (hầu hết Snapdragon/Mali), backend đã build sẵn
sẽ tự nhận driver mà không cần build lại — trỏ `LD_LIBRARY_PATH` tới
driver của vendor nếu nó không nằm trên path mặc định.

### Kích thước binary

Chưa có số đo công bố (benchmark trên thiết bị là follow-up sau). Hướng
dẫn định tính, theo thứ tự tác động:

1. `aarch64-linux-gnu-strip` file `.so` cuối.
2. Build `Release`/`-Os` thay vì `-O3` nếu size quan trọng hơn vài phần
   trăm tốc độ cuối trên workload của bạn.
3. Loại `src/backend/tpu` (PJRT/HLO builders) khi không bao giờ nhắm TPU.
4. OpenCL backend nhỏ và chỉ bật qua `dlopen` lúc runtime — cứ giữ trừ
   khi từng byte đều quý.

### Threading trên mobile

`thread_pool.h` lấy cỡ pool từ `hardware_concurrency()` và bỏ qua
threading với workload nhỏ (`total < 50000`). Không cần sửa code trên
SoC big.LITTLE; nếu tự pin thread thì làm bên ngoài pool.

### CI

`.github/workflows/arm-crosscompile.yml` chạy mỗi push/PR chạm `src/`,
`include/` hoặc `setup.py`: cài `g++-aarch64-linux-gnu` qua apt và biên
dịch một tập TU đại diện (`src/tensor`, `src/nn`, `src/ops`,
`src/cl_backend`, `src/memory_manager`) ra object file bằng cross
compiler (`-O1 -march=armv8-a`). Chỉ compile — không thiết bị, không test
runtime. Build và benchmark thật trên thiết bị nằm ngoài CI.

### Giới hạn

- Không có thiết bị ARM trong CI: tính đúng đắn và số hiệu năng trên
  thiết bị chưa được kiểm chứng cho tới khi chạy benchmark (follow-up sau).
- Các source `.cu`/`.hip` không bao giờ nằm trong cross build; bật
  `LITETORCH_CUDA=1` khi cross-compile là không được hỗ trợ.
- Path NDK chưa được chạy end-to-end trong CI.
