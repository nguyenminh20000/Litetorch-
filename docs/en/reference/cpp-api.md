# C++ API — Reference

This page documents the C++14 API in `include/litetorch/` (56 headers). It is for extension authors and embedders; the Python API is documented in the other reference pages. The Python bindings are a thin pybind11 layer over these classes — where the two differ, the difference is called out.

- **Source of truth:** `include/litetorch/*.h` (56 headers) and `src/*.cpp`. Header paths are cited per entry.
- **Tree:** agent-v1@02cc271a / local HEAD 89afd67.
- **Conventions:** C++14, `namespace litetorch`. Tensors are `std::shared_ptr<Tensor>` everywhere. Errors are `std::runtime_error` with the prefix `[litetorch Error] ` — the same strings Python raises as `RuntimeError`.
- **Related deviations:** D-01 (`Tensor::data_ptr()` throws on non-FP32 — see below). D-14 (INT64 `.numpy()` unsupported — the Python boundary of the same rule).

## Core types

### `Tensor` — `include/litetorch/tensor.h`

- **Key members:** `Tensor::create(shape, device)` · `Tensor::from_vector(data, shape, device)` · `Tensor::zeros(shape, device)` · `Tensor::ones(shape, device)` · `Tensor::randn(shape, device)` · `shape` / `strides` / `dtype` / `device` / `requires_grad` / `grad` (public fields) · `numel()` · `item()` · `to_vector()` · `to(device)` / `to(dtype)` · `cast(dtype)` · `backward(grad, create_graph)` · `zero_grad()` · `retain_grad()` · `clone()` · `contiguous()` · `view(...)` / `reshape(...)` / `permute(...)` · `numpy()` · `copy_()` / `add_()`.
- **D-01 — `data_ptr()` vs `raw_data_ptr()`:** `float* data_ptr()` (`tensor.h:185`) throws `[litetorch Error] data_ptr(): tensor dtype is not FP32; use raw_data_ptr() for byte-level access` on any non-FP32 tensor. `void* raw_data_ptr()` (`tensor.h:186`) returns the raw storage pointer for any dtype. The pointer is `storage->get_cpu_ptr() + offset` — a **CPU** pointer even for GPU tensors; do not dereference it for device memory. Python exposes neither: from Python, byte access goes through `.numpy()` (FP32 only; INT64 throws — D-14) or `to_vector()`.
- **Storage:** a tensor holds a `shared_ptr<Storage>` plus an `offset`; views share storage. `data_ptr()` returns `storage->get_cpu_ptr() + offset`.
- **PyTorch:** `torch.Tensor` — same concepts; torch's `data_ptr()` never throws on dtype (it returns `void*`), so LiteTorch's typed `float*` return is the divergence. Port code by using `raw_data_ptr()` + an explicit cast, or by checking `dtype == DataType::FP32` first.

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

- **Signature:** `struct Device { DeviceType type = CPU; int index = 0; Device(DeviceType type = CPU, int index = 0); Device(const std::string& s); }`; `enum class DeviceType { CPU, GPU, META, TPU }`.
- **Returns:** the string constructor parses `"cpu"`, `"gpu"`, `"cuda"`, `"cuda:N"`, `"tpu"`, `"tpu:N"`, `"meta"` (case-insensitive).
- **PyTorch:** `torch.device` / `c10::Device` — same shape.

### `Ops` — `include/litetorch/ops.h`

- **Key functions:** elementwise `add/sub/mul/div/neg/pow`, `relu/gelu/sigmoid/tanh`, `matmul`, `sum/mean/max`, `softmax/log_softmax`, `conv2d`, `max_pool2d`, `embedding`, `linear`, `dropout`, `layernorm`, `batchnorm`, `cross_entropy`, `mse_loss`, plus `reduce_broadcast` and `from_data` (below).
- **C++-only — `Ops::reduce_broadcast`:** `std::shared_ptr<Tensor> reduce_broadcast(std::shared_ptr<Tensor> grad, const std::vector<int64_t>& orig_shape)` (`ops.h:33`). Sums a broadcast gradient back to the original shape. Not bound to Python — the Python autograd engine calls it internally during backward.
- **C++-only — `Ops::from_data`:** `std::shared_ptr<Tensor> from_data(const std::vector<float>& data, const std::vector<int64_t>& shape, ...)` (`ops.h:166`). Wraps existing host data without copying. Not bound to Python — use `Tensor.from_vector` from Python (which copies).
- **PyTorch:** `at::native` ops — same operations; torch exposes `reduce_broadcast` only internally (`at::native::sum_to`).

## Autograd — `include/litetorch/autograd.h`

- **Key classes:** `Node` (`autograd.h:52`, the DAG node, `enable_shared_from_this`) · `SavedTensor` / `SavedTensorsVector` · `ActiveTensorsGuard` · `NoGradGuard` · `Autograd` (`autograd.h:87`) · `NoGradContext` · `FunctionContext` (`autograd.h:128`) · `Function` (`autograd.h:143`).
- **Key functions:** `Autograd::grad(outputs, inputs, grad_outputs, retain_graph, create_graph, allow_unused)` — the engine behind `lt.autograd.grad`. `Autograd::retain_grad(t)`. `autograd_gradcheck` / `autograd_gradgradcheck` — the numeric checkers behind `lt.autograd.gradcheck`.
- **Guards:** `NoGradGuard` disables gradient recording in its scope (RAII); `ActiveTensorsGuard` tracks live tensors for the memory manager. Both are scope-bound — never hold one past its scope.
- **PyTorch:** `torch::autograd::Node` / `Engine` — same design (a DAG of `Node`s, `grad()` entry point, `NoGradGuard`).

## Memory management — `include/litetorch/memory_manager.h`, `allocator.h`, `storage_guard.h`

- **Key classes:** `MemoryManager` (`memory_manager.h:18`) · `StorageImpl` · `StorageGuard` (`storage_guard.h`).
- **Behavior:** the memory manager owns the caching allocator and the dynamic eviction policy (the framework's differentiator): tensors not needed until a later backward can be evicted to host and reloaded on demand. Eviction is transparent to user code — a `Tensor` stays valid across eviction.
- **C++ contract:** allocate through the manager, never `new`/`malloc` a tensor buffer directly; hold `shared_ptr<Tensor>` (not raw pointers) across any call that may trigger eviction.
- **PyTorch:** `c10::CachingAllocator` + `torch.cuda.memory` — same allocator idea; LiteTorch adds eviction, which torch does not have.

## Neural networks — `include/litetorch/nn.h` (1,260 lines)

- **Key classes:** `nn::Module` (base: `forward`, `parameters`, `to`, `train`/`eval`, `state_dict`/`load_state_dict`) · `Linear`, `Conv1d/2d/3d`, `BatchNorm1d/2d`, `LayerNorm`, `Embedding`, `Dropout`, `ReLU/GELU/Sigmoid/Tanh/LeakyReLU/SiLU`, `LSTM/GRU`, `MultiheadAttention`, `TransformerEncoder/Decoder`, `Sequential`, `ModuleList`.
- **Python mapping:** every `nn` class in `nn.h` is bound under `lt.nn` (see the nn reference pages). The C++ `Module::forward` takes `shared_ptr<Tensor>` and returns `shared_ptr<Tensor>`; Python subclasses override `forward` in Python.
- **PyTorch:** `torch.nn.Module` — same base-class contract, including `parameters()` / `to()` / `state_dict()`.

## Optimizers — `include/litetorch/optim.h`

- **Key classes:** `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad`, `LRScheduler` (step / cosine / exponential).
- **Contract:** `optimizer.step()` updates parameters in place from `.grad`; `optimizer.zero_grad()` clears grads. Schedulers are stepped after `optimizer.step()`.
- **PyTorch:** `torch.optim` — same classes and call order.

## Distributed — `include/litetorch/distributed.h`, `device_mesh.h`, `dtensor.h`, `fsdp_wrapper.h`, `fsdp_tp.h`, `flat_param.h`, `dist_checkpoint.h`, `elastic.h`, `rendezvous.h`, `parallel_3d.h`, `pipeline.h`, `tp_collectives.h`, `context_parallel.h`, `p2p_async.h`, `zero3_optimizer.h`

- **Key classes:** `ProcessGroup`, `NCCLBridge`, `DeviceMesh`, `DTensor`, `Placement`, `FSDPUnit` (C++-only construction — see `distributed.md`), `FSDPTP`, `FlatParameter`, `Parallel3D`, `PipelineParallelModule`, `InterleavedPipelineModule`, `OverlappedAllReducer`, `RendezvousStore`, `RendezvousClient`, `HeartbeatAgent`.
- **Python mapping:** everything except `FSDPUnit` construction is bound (see `distributed.md`). The C++ headers are the authoritative reference for the NCCL selection rules and the mismatch-throw behavior.
- **PyTorch:** `torch.distributed` — same surface, split across these headers instead of one module.

## Quantization / JIT / ONNX / HF / inference / prune / profiler / hub

- **Headers:** `quantization.h` (`Calibrator`, `QATLinear`, `PTQConfig`, `PreparedModel`, `quantize_linear`, `quantize_dynamic_*`) · `jit.h` + `jit_detail.h` (`JITVar`, `JITFunction`, `Tracer`, fused helpers) · `onnx.h` (`Graph`, `Node`, `ValueInfo`, `export_*`, `trace_module`) · `hf.h` (`ModelConfig`, `PretrainedBundle`, `from_pretrained`, `remap_state_dict`) · `inference.h` (`CudaGraph`, `FusionReport`, `fold_conv_bn`, `fuse_conv_bn`) · `prune.h` (`l1_unstructured`, `random_unstructured`, `remove`, `is_pruned`) · `profiler.h` (`enable/disable/record/report`, `ScopedTimer`) · `hub.h` (`get_dir/set_dir/is_url/download_url_to_file/load_state_dict_from_url`).
- **Python mapping:** all of these are bound — see `quant-jit-onnx-hf.md` for the Python signatures. The C++ headers add the implementation notes (e.g. `fold_conv_bn` requires eval mode; `Tracer`'s expired-intermediate error).
- **PyTorch:** `torch.ao.quantization`, `torch.jit`, `torch.onnx`, `transformers`, `torch.nn.utils.prune`, `torch.profiler`, `torch.hub` — same roles.

## Other headers

- **`linalg.h`** — `linalg` namespace: `svd`, `qr`, `eigh`, `solve`, `inv`, `det`, `norm`, `cholesky`, `lstsq`, `pinv`. (D-07 fixed: the linalg promotion path now matches torch.)
- **`fft.h`** — `fft`, `ifft`, `rfft`, `irfft`, `fftn`, `fftshift`.
- **`distributions.h`** — `Normal`, `Uniform`, `Bernoulli`, `Categorical`, `Beta`, `Exponential`.
- **`data.h`** — `Dataset`, `DataLoader`, `TensorDataset`, `Subset`, `RandomSampler`, `SequentialSampler`, `BatchSampler`, `collate`. (Python-backed `Dataset` trampoline dispatches to `size`/`get`, not `__len__`/`__getitem__` — see the H7.2 note in the how-to guide.)
- **`serialization.h`** — `save` / `load` (state-dict files), `save_checkpoint` / `load_checkpoint`.
- **`checkpoint.h`** — `Checkpoint` (activation checkpointing: `checkpoint(fn, *args)`).
- **`trainer.h`** — `Trainer` (training loop: `train_step`, `eval_step`, `fit`).
- **`amp.h`** — `autocast`, `GradScaler` (mixed precision).
- **`generate.h`** — `generate` (greedy / sampling decoding for LLM serving).
- **`dpo.h`** — `DPOTrainer` (direct preference optimization).
- **`llm_serving.h`** — `ServingScheduler`, `PrefixCache`, `ContinuousBatching` (vLLM-style serving).
- **`async_checkpoint.h`** — async checkpoint saving.
- **`backend.h` / `backend_route.h` / `cl_backend.h`** — backend dispatch: the dispatcher routes each op to CUDA / ROCm / OpenCL / CPU. `backend_route.h` holds the routing table.
- **`platform.h`** — platform detection (OS, CUDA/ROCm availability).
- **`thread_pool.h`** — the CPU thread pool used by parallel ops.
- **`flat_param.h`** — `FlatParameter` (see `distributed.md`).
- **`functional.h`** — functional (stateless) versions of the `nn` ops.
- **`custom_ops.h`** — the custom-op registration API (`register_op`).
- **`tpu.h`** — the PJRT TPU backend interface.
- **`grad_scaler.h`** — `GradScaler` (AMP loss scaling).

## Building against LiteTorch (CMake)

The real build layout (`CMakeLists.txt` at the repo root, verified at HEAD 89afd67):

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

To build your own program against the library, link the shared library and add `include/` to the include path:

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

Notes: the library glob includes every `src/*.cpp` — there is no per-file target list to keep in sync. The Python extension is built separately by `setup.py` (`python3 setup.py build_ext --inplace`; `package_dir={"": "python"}` puts the extension under `python/`). CUDA files (`.cu`) are compiled by `nvcc` through `setup.py`, not through this CMake file — the CMake build is CPU-only.

## A minimal C++ program

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

Compile it with `g++ -std=c++14 -Iinclude main.cpp -Lbuild -llitetorch -o main` (after building the library with CMake). `data_ptr()` is safe here because `z` is FP32 — on any other dtype it throws (D-01).

## Error contract

- Every C++ failure is a `std::runtime_error` whose message starts with `[litetorch Error] ` and names the failing function, e.g. `[litetorch Error] data_ptr(): tensor dtype is not FP32; use raw_data_ptr() for byte-level access`.
- Shape/dtype mismatches throw at the call site (fail-loud), never silently. There is one deliberate exception: `Tensor::to(device)` falls back to CPU silently when the device is unavailable (D-02) — check `lt.cuda.is_available()` / `lt.tpu.is_available()` before relying on a device move.
- Python sees these as `RuntimeError` with the same message text. Do not match on the message text in production code — match on the exception type and the function name only.

## pybind11 boundary notes

- **Ownership:** Python holds `shared_ptr<Tensor>`; a tensor returned to Python stays alive as long as the Python object does. C++ code that keeps a `shared_ptr` to a Python-created tensor must not assume the Python object outlives it — keep the `shared_ptr`.
- **GIL:** C++ kernels release the GIL during compute; Python callbacks (custom `autograd.Function`, Python-backed `Dataset`) re-acquire it. Never call back into Python from a GIL-released section.
- **Types:** `std::vector<int64_t>` maps to a Python list of ints; `std::vector<float>` maps to a list of floats; `std::optional<T>` maps to `T | None`. Enums (`DataType`, `DeviceType`, `ReduceOp`, `PlacementType`) map to Python enums with `.name` / `.value`.
- **Not bound:** `Ops::reduce_broadcast`, `Ops::from_data`, `FSDPUnit` construction, `ProcessGroup::get()` before init (returns an uninitialized group). If you need one of these from Python, file a request — do not reach into the C++ headers from Python.
