# API C++ — Tài liệu tham khảo

Trang này ghi lại API C++14 trong `include/litetorch/` (56 header). Nó dành cho tác giả extension và người nhúng; API Python được ghi trong các trang tham khảo khác. Binding Python là một lớp pybind11 mỏng trên các class này — chỗ nào hai bên khác nhau, sự khác biệt được nêu rõ.

- **Nguồn chân lý:** `include/litetorch/*.h` (56 header) và `src/*.cpp`. Đường dẫn header được trích dẫn cho từng mục.
- **Cây mã nguồn:** agent-v1@02cc271a / HEAD cục bộ 89afd67.
- **Quy ước:** C++14, `namespace litetorch`. Tensor là `std::shared_ptr<Tensor>` ở mọi nơi. Lỗi là `std::runtime_error` với tiền tố `[litetorch Error] ` — cùng chuỗi mà Python raise dưới dạng `RuntimeError`.
- **Sai khác liên quan:** D-01 (`Tensor::data_ptr()` throw trên tensor không phải FP32 — xem bên dưới). D-14 (`.numpy()` INT64 không hỗ trợ — biên Python của cùng quy tắc).

## Kiểu cốt lõi

### `Tensor` — `include/litetorch/tensor.h`

- **Thành viên chính:** `Tensor::create(shape, device)` · `Tensor::from_vector(data, shape, device)` · `Tensor::zeros(shape, device)` · `Tensor::ones(shape, device)` · `Tensor::randn(shape, device)` · `shape` / `strides` / `dtype` / `device` / `requires_grad` / `grad` (trường public) · `numel()` · `item()` · `to_vector()` · `to(device)` / `to(dtype)` · `cast(dtype)` · `backward(grad, create_graph)` · `zero_grad()` · `retain_grad()` · `clone()` · `contiguous()` · `view(...)` / `reshape(...)` / `permute(...)` · `numpy()` · `copy_()` / `add_()`.
- **D-01 — `data_ptr()` so với `raw_data_ptr()`:** `float* data_ptr()` (`tensor.h:185`) throw `[litetorch Error] data_ptr(): tensor dtype is not FP32; use raw_data_ptr() for byte-level access` trên bất kỳ tensor nào không phải FP32. `void* raw_data_ptr()` (`tensor.h:186`) trả về con trỏ storage thô cho mọi dtype. Con trỏ là `storage->get_cpu_ptr() + offset` — một con trỏ **CPU** ngay cả với tensor GPU; đừng dereference nó cho bộ nhớ thiết bị. Python không expose cả hai: từ Python, truy cập byte đi qua `.numpy()` (chỉ FP32; INT64 throw — D-14) hoặc `to_vector()`.
- **Storage:** một tensor giữ một `shared_ptr<Storage>` cộng một `offset`; các view chia sẻ storage. `data_ptr()` trả về `storage->get_cpu_ptr() + offset`.
- **PyTorch:** `torch.Tensor` — cùng khái niệm; `data_ptr()` của torch không bao giờ throw theo dtype (nó trả về `void*`), nên kiểu trả về `float*` của LiteTorch là điểm sai khác. Port code bằng cách dùng `raw_data_ptr()` + cast tường minh, hoặc kiểm tra `dtype == DataType::FP32` trước.

<!-- ex: id=r15-boundary device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
f = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
print("fp32 numpy:", f.numpy().tolist())
i = lt.Tensor.from_vector([1, 2], [2], cpu).long()
try:
    i.numpy()
except RuntimeError as e:
    print("int64 numpy raises:", str(e)[:80])
print("raw access via to_vector:", i.to_vector())
```

### `Device` — `include/litetorch/device.h`

- **Chữ ký:** `struct Device { DeviceType type = CPU; int index = 0; Device(DeviceType type = CPU, int index = 0); Device(const std::string& s); }`; `enum class DeviceType { CPU, GPU, META, TPU }`.
- **Trả về:** constructor chuỗi parse `"cpu"`, `"gpu"`, `"cuda"`, `"cuda:N"`, `"tpu"`, `"tpu:N"`, `"meta"` (không phân biệt hoa thường).
- **PyTorch:** `torch.device` / `c10::Device` — cùng hình thức.

### `Ops` — `include/litetorch/ops.h`

- **Hàm chính:** elementwise `add/sub/mul/div/neg/pow`, `relu/gelu/sigmoid/tanh`, `matmul`, `sum/mean/max`, `softmax/log_softmax`, `conv2d`, `max_pool2d`, `embedding`, `linear`, `dropout`, `layernorm`, `batchnorm`, `cross_entropy`, `mse_loss`, cộng `reduce_broadcast` và `from_data` (bên dưới).
- **Chỉ C++ — `Ops::reduce_broadcast`:** `std::shared_ptr<Tensor> reduce_broadcast(std::shared_ptr<Tensor> grad, const std::vector<int64_t>& orig_shape)` (`ops.h:33`). Cộng gradient đã broadcast trở về shape gốc. Không bind sang Python — engine autograd của Python gọi nó nội bộ trong backward.
- **Chỉ C++ — `Ops::from_data`:** `std::shared_ptr<Tensor> from_data(const std::vector<float>& data, const std::vector<int64_t>& shape, ...)` (`ops.h:166`). Wrap dữ liệu host hiện có mà không copy. Không bind sang Python — từ Python dùng `Tensor.from_vector` (nó copy).
- **PyTorch:** các op `at::native` — cùng phép toán; torch chỉ expose `reduce_broadcast` nội bộ (`at::native::sum_to`).

## Autograd — `include/litetorch/autograd.h`

- **Class chính:** `Node` (`autograd.h:52`, node DAG, `enable_shared_from_this`) · `SavedTensor` / `SavedTensorsVector` · `ActiveTensorsGuard` · `NoGradGuard` · `Autograd` (`autograd.h:87`) · `NoGradContext` · `FunctionContext` (`autograd.h:128`) · `Function` (`autograd.h:143`).
- **Hàm chính:** `Autograd::grad(outputs, inputs, grad_outputs, retain_graph, create_graph, allow_unused)` — engine đứng sau `lt.autograd.grad`. `Autograd::retain_grad(t)`. `autograd_gradcheck` / `autograd_gradgradcheck` — bộ kiểm tra số đứng sau `lt.autograd.gradcheck`.
- **Guard:** `NoGradGuard` tắt ghi gradient trong phạm vi của nó (RAII); `ActiveTensorsGuard` theo dõi tensor đang sống cho memory manager. Cả hai đều gắn với phạm vi — không bao giờ giữ một guard quá phạm vi của nó.
- **PyTorch:** `torch::autograd::Node` / `Engine` — cùng thiết kế (DAG các `Node`, điểm vào `grad()`, `NoGradGuard`).

## Quản lý bộ nhớ — `include/litetorch/memory_manager.h`, `allocator.h`, `storage_guard.h`

- **Class chính:** `MemoryManager` (`memory_manager.h:18`) · `StorageImpl` · `StorageGuard` (`storage_guard.h`).
- **Hành vi:** memory manager sở hữu caching allocator và chính sách giải phóng bộ nhớ động (điểm khác biệt của framework): tensor không cần cho đến một backward sau có thể được giải phóng về host và load lại khi cần. Giải phóng là trong suốt với code người dùng — một `Tensor` vẫn hợp lệ sau khi bị giải phóng.
- **Hợp đồng C++:** cấp phát qua manager, không bao giờ `new`/`malloc` bộ đệm tensor trực tiếp; giữ `shared_ptr<Tensor>` (không phải con trỏ thô) qua bất kỳ lệnh gọi nào có thể kích hoạt giải phóng.
- **PyTorch:** `c10::CachingAllocator` + `torch.cuda.memory` — cùng ý tưởng allocator; LiteTorch thêm giải phóng bộ nhớ, mà torch không có.

## Mạng nơ-ron — `include/litetorch/nn.h` (1.260 dòng)

- **Class chính:** `nn::Module` (cơ sở: `forward`, `parameters`, `to`, `train`/`eval`, `state_dict`/`load_state_dict`) · `Linear`, `Conv1d/2d/3d`, `BatchNorm1d/2d`, `LayerNorm`, `Embedding`, `Dropout`, `ReLU/GELU/Sigmoid/Tanh/LeakyReLU/SiLU`, `LSTM/GRU`, `MultiheadAttention`, `TransformerEncoder/Decoder`, `Sequential`, `ModuleList`.
- **Ánh xạ Python:** mọi class `nn` trong `nn.h` đều được bind dưới `lt.nn` (xem các trang tham khảo nn). `Module::forward` của C++ nhận `shared_ptr<Tensor>` và trả về `shared_ptr<Tensor>`; subclass Python ghi đè `forward` bằng Python.
- **PyTorch:** `torch.nn.Module` — cùng hợp đồng class cơ sở, gồm `parameters()` / `to()` / `state_dict()`.

## Optimizer — `include/litetorch/optim.h`

- **Class chính:** `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad`, `LRScheduler` (step / cosine / exponential).
- **Hợp đồng:** `optimizer.step()` cập nhật tham số tại chỗ từ `.grad`; `optimizer.zero_grad()` xóa gradient. Scheduler được step sau `optimizer.step()`.
- **PyTorch:** `torch.optim` — cùng class và thứ tự gọi.

## Phân tán — `include/litetorch/distributed.h`, `device_mesh.h`, `dtensor.h`, `fsdp_wrapper.h`, `fsdp_tp.h`, `flat_param.h`, `dist_checkpoint.h`, `elastic.h`, `rendezvous.h`, `parallel_3d.h`, `pipeline.h`, `tp_collectives.h`, `context_parallel.h`, `p2p_async.h`, `zero3_optimizer.h`

- **Class chính:** `ProcessGroup`, `NCCLBridge`, `DeviceMesh`, `DTensor`, `Placement`, `FSDPUnit` (khởi tạo chỉ C++ — xem `distributed.md`), `FSDPTP`, `FlatParameter`, `Parallel3D`, `PipelineParallelModule`, `InterleavedPipelineModule`, `OverlappedAllReducer`, `RendezvousStore`, `RendezvousClient`, `HeartbeatAgent`.
- **Ánh xạ Python:** mọi thứ trừ khởi tạo `FSDPUnit` đều được bind (xem `distributed.md`). Các header C++ là tham khảo có thẩm quyền cho quy tắc chọn NCCL và hành vi mismatch-throw.
- **PyTorch:** `torch.distributed` — cùng bề mặt, chia trên các header này thay vì một module.

## Lượng tử hóa / JIT / ONNX / HF / suy luận / prune / profiler / hub

- **Header:** `quantization.h` (`Calibrator`, `QATLinear`, `PTQConfig`, `PreparedModel`, `quantize_linear`, `quantize_dynamic_*`) · `jit.h` + `jit_detail.h` (`JITVar`, `JITFunction`, `Tracer`, các helper fusion) · `onnx.h` (`Graph`, `Node`, `ValueInfo`, `export_*`, `trace_module`) · `hf.h` (`ModelConfig`, `PretrainedBundle`, `from_pretrained`, `remap_state_dict`) · `inference.h` (`CudaGraph`, `FusionReport`, `fold_conv_bn`, `fuse_conv_bn`) · `prune.h` (`l1_unstructured`, `random_unstructured`, `remove`, `is_pruned`) · `profiler.h` (`enable/disable/record/report`, `ScopedTimer`) · `hub.h` (`get_dir/set_dir/is_url/download_url_to_file/load_state_dict_from_url`).
- **Ánh xạ Python:** tất cả đều được bind — xem `quant-jit-onnx-hf.md` cho chữ ký Python. Các header C++ bổ sung ghi chú triển khai (ví dụ `fold_conv_bn` yêu cầu chế độ eval; lỗi expired-intermediate của `Tracer`).
- **PyTorch:** `torch.ao.quantization`, `torch.jit`, `torch.onnx`, `transformers`, `torch.nn.utils.prune`, `torch.profiler`, `torch.hub` — cùng vai trò.

## Các header khác

- **`linalg.h`** — namespace `linalg`: `svd`, `qr`, `eigh`, `solve`, `inv`, `det`, `norm`, `cholesky`, `lstsq`, `pinv`. (D-07 đã sửa: đường promotion của linalg giờ khớp torch.)
- **`fft.h`** — `fft`, `ifft`, `rfft`, `irfft`, `fftn`, `fftshift`.
- **`distributions.h`** — `Normal`, `Uniform`, `Bernoulli`, `Categorical`, `Beta`, `Exponential`.
- **`data.h`** — `Dataset`, `DataLoader`, `TensorDataset`, `Subset`, `RandomSampler`, `SequentialSampler`, `BatchSampler`, `collate`. (Trampoline `Dataset` phía Python dispatch tới `size`/`get`, không phải `__len__`/`__getitem__` — xem ghi chú H7.2 trong how-to.)
- **`serialization.h`** — `save` / `load` (file state-dict), `save_checkpoint` / `load_checkpoint`.
- **`checkpoint.h`** — `Checkpoint` (activation checkpointing: `checkpoint(fn, *args)`).
- **`trainer.h`** — `Trainer` (vòng huấn luyện: `train_step`, `eval_step`, `fit`).
- **`amp.h`** — `autocast`, `GradScaler` (mixed precision).
- **`generate.h`** — `generate` (giải mã greedy / sampling cho LLM serving).
- **`dpo.h`** — `DPOTrainer` (direct preference optimization).
- **`llm_serving.h`** — `ServingScheduler`, `PrefixCache`, `ContinuousBatching` (serving kiểu vLLM).
- **`async_checkpoint.h`** — lưu checkpoint bất đồng bộ.
- **`backend.h` / `backend_route.h` / `cl_backend.h`** — dispatch backend: dispatcher định tuyến mỗi op tới CUDA / ROCm / OpenCL / CPU. `backend_route.h` giữ bảng định tuyến.
- **`platform.h`** — phát hiện nền tảng (HĐH, CUDA/ROCm có sẵn).
- **`thread_pool.h`** — thread pool CPU dùng cho các op song song.
- **`flat_param.h`** — `FlatParameter` (xem `distributed.md`).
- **`functional.h`** — các phiên bản functional (không trạng thái) của các op `nn`.
- **`custom_ops.h`** — API đăng ký custom-op (`register_op`).
- **`tpu.h`** — giao diện backend TPU PJRT.
- **`grad_scaler.h`** — `GradScaler` (loss scaling cho AMP).

## Build với LiteTorch (CMake)

Bố cục build thật (`CMakeLists.txt` tại gốc repo, đã xác minh tại HEAD 89afd67):

```cmake
cmake_minimum_required(VERSION 3.10)
project(litetorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cpp")

find_package(Threads REQUIRED)

add_library(litetorch SHARED ${SOURCES})
target_link_libraries(litetorch PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)

file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE litetorch Threads::Threads)
endforeach()
```

Để build chương trình của bạn với thư viện, link shared library và thêm `include/` vào đường dẫn include:

<!-- ex: id=r15-cmake device=cpu backend=cpu runtime=n/a torch-parity=n/a status=fragment -->
```cmake
cmake_minimum_required(VERSION 3.10)
project(my_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(my_app main.cpp)
target_include_directories(my_app PRIVATE /path/to/litetorch/include)
target_link_libraries(my_app PRIVATE /path/to/litetorch/build/liblitetorch.so Threads::Threads)
```

Ghi chú: thư viện glob mọi `src/*.cpp` — không có danh sách target theo từng file cần đồng bộ. Extension Python được build riêng bằng `setup.py` (`python3 setup.py build_ext --inplace`; `package_dir={"": "python"}` đặt extension dưới `python/`). File CUDA (`.cu`) được biên dịch bằng `nvcc` qua `setup.py`, không qua file CMake này — build CMake chỉ cho CPU.

## Một chương trình C++ tối thiểu

<!-- ex: id=r15-cpp-minimal device=cpu backend=cpu runtime=n/a torch-parity=n/a status=fragment -->
```cpp
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <iostream>

int main() {
    using namespace litetorch;
    auto cpu = Device(DeviceType::CPU, 0);
    auto x = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, cpu);
    auto y = Ops::relu(x);
    auto z = Ops::matmul(y, y);
    for (float v : z->to_vector()) std::cout << v << " ";
    std::cout << "\n";
    float* p = z->data_ptr();
    std::cout << "first element via data_ptr: " << p[0] << "\n";
    return 0;
}
```

Biên dịch bằng `g++ -std=c++14 -Iinclude main.cpp -Lbuild -llitetorch -o main` (sau khi build thư viện bằng CMake). `data_ptr()` an toàn ở đây vì `z` là FP32 — trên mọi dtype khác nó throw (D-01).

## Hợp đồng lỗi

- Mọi lỗi C++ đều là `std::runtime_error` với thông điệp bắt đầu bằng `[litetorch Error] ` và đặt tên hàm lỗi, ví dụ `[litetorch Error] data_ptr(): tensor dtype is not FP32; use raw_data_ptr() for byte-level access`.
- Shape/dtype không khớp throw tại điểm gọi (fail-loud), không bao giờ trong im lặng. Có một ngoại lệ cố ý: `Tensor::to(device)` rơi về CPU trong im lặng khi thiết bị không sẵn có (D-02) — kiểm tra `lt.cuda.is_available()` / `lt.tpu.is_available()` trước khi dựa vào một lần di chuyển thiết bị.
- Python thấy các lỗi này dưới dạng `RuntimeError` với cùng văn bản thông điệp. Đừng match trên văn bản thông điệp trong code production — chỉ match trên kiểu exception và tên hàm.

## Ghi chú biên pybind11

- **Quyền sở hữu:** Python giữ `shared_ptr<Tensor>`; một tensor trả về Python vẫn sống chừng nào đối tượng Python còn sống. Code C++ giữ một `shared_ptr` tới tensor do Python tạo không được giả định đối tượng Python sống lâu hơn nó — hãy giữ `shared_ptr`.
- **GIL:** kernel C++ giải phóng GIL trong lúc tính toán; callback Python (`autograd.Function` tùy chỉnh, `Dataset` phía Python) lấy lại nó. Không bao giờ gọi lại vào Python từ một đoạn đã giải phóng GIL.
- **Kiểu:** `std::vector<int64_t>` ánh xạ tới danh sách int Python; `std::vector<float>` ánh xạ tới danh sách float; `std::optional<T>` ánh xạ tới `T | None`. Enum (`DataType`, `DeviceType`, `ReduceOp`, `PlacementType`) ánh xạ tới enum Python với `.name` / `.value`.
- **Không bind:** `Ops::reduce_broadcast`, `Ops::from_data`, khởi tạo `FSDPUnit`, `ProcessGroup::get()` trước init (trả về group chưa khởi tạo). Nếu bạn cần một trong các thứ này từ Python, hãy gửi yêu cầu — đừng với vào header C++ từ Python.
