#include "bindings_common.h"

void bind_cuda(py::module_& m) {
        auto cuda_mod = m.def_submodule("cuda");
        cuda_mod.def("is_available", []() {
            if (std::getenv("LITETORCH_NO_NATIVE_GPU")) {
                return false;
            }
            auto backend = litetorch::BackendDispatcher::get().get_backend();
            return backend && backend->is_available();
        });
        cuda_mod.def("set_tf32_enabled", [](bool enabled) {
            auto backend = litetorch::BackendDispatcher::get().get_backend();
            if (backend) {
                backend->set_tf32_enabled(enabled);
            }
        }, py::arg("enabled") = true);
        cuda_mod.def("is_tf32_enabled", []() {
            auto backend = litetorch::BackendDispatcher::get().get_backend();
            if (backend) {
                return backend->is_tf32_enabled();
            }
            return false;
        });
        cuda_mod.def("set_fp16_math_enabled", [](bool enabled) {
            auto backend = litetorch::BackendDispatcher::get().get_backend();
            if (backend) {
                backend->set_fp16_math_enabled(enabled);
            }
        }, py::arg("enabled") = true);
        cuda_mod.def("is_fp16_math_enabled", []() {
            auto backend = litetorch::BackendDispatcher::get().get_backend();
            if (backend) {
                return backend->is_fp16_math_enabled();
            }
            return false;
        });
        cuda_mod.def("memory_allocated", []() -> size_t {
            return litetorch::MemoryManager::get().get_gpu_used();
        });
        // === W1-T2 tensor-api ===
        cuda_mod.def("is_available", &litetorch::cuda::is_available);
        cuda_mod.def("device_count", &litetorch::cuda::device_count);
        cuda_mod.def("current_device", &litetorch::cuda::current_device);
        cuda_mod.def("set_device", &litetorch::cuda::set_device, py::arg("device") = 0);
        cuda_mod.def("memory_reserved", &litetorch::cuda::memory_reserved);
        cuda_mod.def("manual_seed_all", &litetorch::cuda::manual_seed_all, py::arg("seed"));
        m.def("equal", &litetorch::tensor_equal, py::arg("a"), py::arg("b"));
        // === W1-T2 tensor-api ===
        m.def("get_gpu_memory_used", []() -> size_t {
            return litetorch::MemoryManager::get().get_gpu_used();
        });
}
