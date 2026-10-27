# How-To H5 — Đưa một mô hình lên Mobile / ARM

Cross-compile **CPU backend** của LiteTorch cho ARM64: Linux `aarch64`
(server, Raspberry Pi 4/5, AWS Graviton) và Android (`arm64-v8a`, thư
viện C++). Không có mobile GPU backend — CUDA/ROCm chỉ dành cho
desktop/server. OpenCL backend được trình bày ở cuối trang này.

Các khẳng định về toolchain dưới đây đã được kiểm chứng lại với cây ở
HEAD 114f2e1: không có SIMD intrinsic nào (`__m128`, `_mm_*`,
`float32x4_t`, `neon.h`) dưới `include/` hay `src/`, và cả
`.github/workflows/arm-crosscompile.yml` lẫn `scripts/verify_build.py`
đều tồn tại. Số liệu đúng đắn và hiệu năng trên thiết bị vẫn chưa được
kiểm chứng — không có thiết bị ARM nào chạy trong CI.

## H5.1 — Linux aarch64: cross-compile với toolchain chuẩn

Song song CPU đến từ `include/litetorch/thread_pool.h`, dùng
`std::thread` thuần và định cỡ công việc theo
`hardware_concurrency()`. Không có code path x86-specific nào cần port
và không có intrinsic NEON nào cần viết — NEON đến từ compiler
auto-vectorization. Việc duy nhất của bạn là toolchain và cờ đúng.

```bash
sudo apt install g++-aarch64-linux-gnu
```

`setup.py` tôn trọng các biến môi trường `CC`/`CXX` chuẩn, nên toàn bộ
build — object C++ host và extension CPython — đi qua cross compiler:

```bash
cd Litetorch-
CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ \
  python3 setup.py build_ext --inplace
```

Những gì `setup.py` thực sự truyền: `-std=c++14 -fPIC`, tối ưu hóa
**`-O1`** (cố ý: build từng bị OOM-kill ở `-O3` trên VM nhỏ), cộng
`-DLITETORCH_BUILD_COMMIT` từ build-integrity guard. Không có
`-march=native`/`-mavx` nào trong các cờ host, nên không có gì
x86-specific lọt vào.

Đối với một bản build ARM release, hãy nâng tối ưu hóa trong
`setup.py`'s `build_extensions()` (`extra_compile_args`):

```python skip-run
ext.extra_compile_args = ["-std=c++14", "-O3", "-march=armv8-a", "-fPIC", ...]
```

`-O3 -march=armv8-a` bật NEON auto-vectorization trên toàn bộ CPU
backend. Các env knob liên quan giữ nguyên:
`LITETORCH_CUDA=1` (CUDA, chỉ native-arch — không dùng để cross),
`LITETORCH_ROCM=1`, `LITETORCH_TPU_PJRT=1`, `LITETORCH_CUDNN_ROOT`,
`LITETORCH_NO_NATIVE_GPU=1` (bỏ qua probe nvcc/hip trên máy cross
headless), `LITETORCH_CUDA_ARCH`, `LT_BUILD_JOBS`.

Sau một cross build thành công, chạy
`python3 scripts/verify_build.py` trên target (hoặc với target
sysroot) — nó fail loud khi artifact lỗi thời và khi lib GPU native bị
bỏ qua lặng lẽ.

## H5.2 — Android NDK: dùng LiteTorch như một thư viện C++

Extension Python nhắm tới Linux `aarch64`. Trên Android, dùng
LiteTorch như một **thư viện C++**. `CMakeLists.txt` tối giản
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
auto-vectorization đã nêu trên. Lưu ý `src/*.cpp` cũng lấy cả các file
`.cpp` TPU — chúng compile thành stub khi không có `libtpu` và không
bao giờ được gọi trừ khi các đường `LITETORCH_TPU_PJRT` bị chạm tới;
loại `src/backend/tpu` khỏi glob nếu bạn muốn binary gọn hơn. Đường
NDK chưa được chạy end-to-end trong CI.

## H5.3 — OpenCL trên Android

OpenCL backend không cần vendor header để build:
`include/litetorch/cl_backend.h` kèm sẵn stub typedef (`void*` handle,
khớp ABI opaque-pointer), và `src/cl_backend/cl_backend.cpp` `dlopen`
`libOpenCL.so.1` / `libOpenCL.so` **lúc runtime**. Trên các thiết bị
Android mà vendor kèm driver OpenCL (hầu hết Snapdragon/Mali), backend
đã build sẵn sẽ nhận nó mà không cần rebuild — trỏ `LD_LIBRARY_PATH`
tới driver của vendor nếu nó không nằm trên đường mặc định.

## H5.4 — Kiểm tra đường CPU backend trên target

Mô hình dưới đây là smoke test chạy trên target ARM sau cross build:
nó luyện đúng đường CPU backend (tạo tensor, một lớp linear, autograd)
mà thư viện đã cross-compile phải phục vụ. Nó chạy trên mọi CPU, nên
hãy chạy trên host trước để lập baseline, rồi chạy trên target và so
sánh output.

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

Đã kiểm chứng trên CPU (host x86-64), LiteTorch 88a4ba1, parity với
torch 1e-4. Trên target ARM cùng script phải in cùng shape và giá trị
trong 1e-4 — nếu không, cross build đáng nghi, chứ không phải mô hình.

## H5.5 — Kích thước binary và threading, ngắn gọn

Chưa có số liệu đo nào được công bố (benchmark trên thiết bị là
follow-up đã hoãn). Hướng dẫn định tính, theo thứ tự tác động:

1. `aarch64-linux-gnu-strip` file `.so` cuối cùng.
2. Build `Release`/`-Os` thay vì `-O3` nếu kích thước quan trọng hơn
   vài phần trăm tốc độ cuối cùng trên workload của bạn.
3. Loại `src/backend/tpu` (PJRT/HLO builders) khi bạn không bao giờ
   nhắm tới TPU.
4. OpenCL backend nhỏ và được gate lúc runtime bằng `dlopen` — giữ nó
   trừ khi từng byte đều quan trọng.

`thread_pool.h` định cỡ pool từ `hardware_concurrency()` và bỏ qua
threading cho workload nhỏ (`total < 50000`). Không cần thay đổi code
trên các SoC big.LITTLE; nếu bạn tự pin thread, hãy làm bên ngoài pool.

## H5.6 — CI coverage và giới hạn của nó

`.github/workflows/arm-crosscompile.yml` chạy trên mỗi push/PR chạm
`src/`, `include/` hoặc `setup.py`: nó cài `g++-aarch64-linux-gnu`
qua apt và compile một tập TU đại diện (`src/tensor`, `src/nn`,
`src/ops`, `src/cl_backend`, `src/memory_manager`) thành object file
bằng cross compiler (`-O1 -march=armv8-a`). Chỉ compile — không có
thiết bị, không có runtime test. Build và benchmark thực tế trên
thiết bị chạy ngoài CI. Các nguồn `.cu`/`.hip` GPU không bao giờ nằm
trong cross build; đặt `LITETORCH_CUDA=1` trong lúc cross-compile là
không được hỗ trợ.
