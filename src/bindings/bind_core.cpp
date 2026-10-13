#include "bindings_common.h"

void bind_core(py::module_& m) {
        m.doc() = "LiteTorch Python Bindings";

        m.def("build_commit", []() {
    #ifdef LITETORCH_BUILD_COMMIT
            return std::string(LITETORCH_BUILD_COMMIT);
    #else
            return std::string("unknown");
    #endif
        });

        py::enum_<DeviceType>(m, "DeviceType")
            .value("CPU", DeviceType::CPU)
            .value("GPU", DeviceType::GPU)
            .value("META", DeviceType::META)
            .value("TPU", DeviceType::TPU)
            .export_values();

        py::class_<Device>(m, "Device")
            .def(py::init<DeviceType, int>(), py::arg("type") = DeviceType::CPU, py::arg("index") = 0)
            .def(py::init([](const std::string& dev_str) {
                std::string s = dev_str;
                for (auto& c : s) c = tolower(c);
                if (s == "meta") return Device(DeviceType::META, 0);
                if (s.rfind("tpu", 0) == 0) {
                    int idx = 0;
                    if (s.size() > 4 && s[3] == ':') {
                        try { idx = std::stoi(s.substr(4)); } catch (...) { idx = 0; }
                    }
                    return Device(DeviceType::TPU, idx);
                }
                if (s.rfind("gpu", 0) == 0 || s.rfind("cuda", 0) == 0) {
                    int idx = 0;
                    size_t colon = s.find(':');
                    if (colon != std::string::npos && colon + 1 < s.size()) {
                        try { idx = std::stoi(s.substr(colon + 1)); } catch (...) { idx = 0; }
                    }
                    return Device(DeviceType::GPU, idx);
                }
                return Device(DeviceType::CPU, 0);
            }), py::arg("device_str"))
            .def_readwrite("type", &Device::type)
            .def_readwrite("index", &Device::index)
            .def("to_string", &Device::to_string)
            .def("__repr__", &Device::to_string);

        m.def("is_gpu_available", []() {
            if (std::getenv("LITETORCH_NO_NATIVE_GPU")) {
                return litetorch::CLBackend::get().is_available();
            }
            auto backend = litetorch::BackendDispatcher::get().get_backend();
            if (backend && backend->is_available()) {
                return true;
            }
            return litetorch::CLBackend::get().is_available();
        });

        m.def("cuda_synchronize", []() {
            auto backend = litetorch::BackendDispatcher::get().get_backend();
            if (backend && backend->is_available()) {
                backend->finish();
            }
        });

        m.def("is_tpu_available", &litetorch::tpu::is_available);

        m.def("auto_device", []() {
            if (!std::getenv("LITETORCH_NO_NATIVE_GPU")) {
                auto backend = litetorch::BackendDispatcher::get().get_backend();
                if (backend && backend->is_available()) {
                    return Device(DeviceType::GPU, 0);
                }
            }
            if (litetorch::CLBackend::get().is_available()) {
                return Device(DeviceType::GPU, 0);
            }
            if (litetorch::tpu::is_available()) {
                return Device(DeviceType::TPU, 0);
            }
            return Device(DeviceType::CPU, 0);
        });

        m.def("get_backend_name", []() -> std::string {
            if (!std::getenv("LITETORCH_NO_NATIVE_GPU")) {
                auto backend = litetorch::BackendDispatcher::get().get_backend();
                if (backend && backend->is_available()) {
    #if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
                    return "ROCm";
    #else
                    return "CUDA";
    #endif
                }
            }
            if (litetorch::CLBackend::get().is_available()) {
                return "OpenCL";
            }
            if (litetorch::tpu::is_available()) {
                return "TPU (Google PJRT / libtpu)";
            }
            return "CPU";
        });
}
