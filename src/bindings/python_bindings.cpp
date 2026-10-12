#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cstring>
// === W1-T2 tensor-api: includes ===
#include <pybind11/numpy.h>
#include <algorithm>
#include <cctype>
// === W1-T2 tensor-api ===

#include "litetorch/device.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/optim.h"
#include "litetorch/data.h"
#include "litetorch/checkpoint.h"
#include "litetorch/backend.h"
#include "litetorch/continuous_batching.h"
#include "litetorch/llm_serving.h"
#include "litetorch/distributed.h"
#include "litetorch/serialization.h"
#include "litetorch/grad_scaler.h"
#include "litetorch/amp.h"
#include "litetorch/quantization.h"
#include "litetorch/guided_decoding.h"
#include "litetorch/zero3_optimizer.h"
#include "litetorch/autograd.h"
#include "litetorch/fsdp.h"
#include "litetorch/dtensor.h"
#include "litetorch/device_mesh.h"
#include "litetorch/jit.h"
#include "litetorch/allocator.h"
#include "litetorch/tpu.h"
#include "litetorch/memory_manager.h"

// === W1-T5 functional: includes ===
#include "litetorch/functional.h"



namespace py = pybind11;
using namespace litetorch;

// === W2-T4 autograd: trampoline ===
namespace {
std::vector<std::shared_ptr<litetorch::Tensor>> w2t4_unpack(py::object o, bool allow_none) {
    std::vector<std::shared_ptr<litetorch::Tensor>> out;
    if (py::isinstance<py::tuple>(o) || py::isinstance<py::list>(o)) {
        for (py::handle h : o) {
            if (h.is_none()) {
                if (!allow_none)
                    throw std::runtime_error("[litetorch Error] autograd.Function: forward() must return Tensors, got None");
                out.push_back(nullptr);
                continue;
            }
            out.push_back(h.cast<std::shared_ptr<litetorch::Tensor>>());
        }
    } else {
        if (o.is_none()) {
            if (!allow_none)
                throw std::runtime_error("[litetorch Error] autograd.Function: forward() must return a Tensor, got None");
            out.push_back(nullptr);
        } else {
            out.push_back(o.cast<std::shared_ptr<litetorch::Tensor>>());
        }
    }
    return out;
}

struct PyFunction : public litetorch::Function {
    py::tuple arg_template;
    py::object py_self;
    std::vector<std::shared_ptr<litetorch::Tensor>> forward(litetorch::FunctionContext* ctx,
        const std::vector<std::shared_ptr<litetorch::Tensor>>& inputs) override {
        py::gil_scoped_acquire gil;
        py::function f = py::get_override(static_cast<const litetorch::Function*>(this), "forward");
        if (!f) throw std::runtime_error("[litetorch Error] autograd.Function.forward() must be overridden");
        py::object py_ctx = py::cast(ctx, py::return_value_policy::reference);
        ctx->py_state = std::shared_ptr<void>(new py::object(py_ctx), [](void* p) {
            py::gil_scoped_acquire gil2;
            delete static_cast<py::object*>(p);
        });
        size_t ti = 0;
        py::tuple args(arg_template.size());
        for (size_t i = 0; i < (size_t)arg_template.size(); ++i) {
            if (arg_template[i].is_none()) {
                if (ti >= inputs.size())
                    throw std::runtime_error("[litetorch Error] autograd.Function: tensor arg mismatch in forward()");
                args[i] = py::cast(inputs[ti++]);
            } else {
                args[i] = arg_template[i];
            }
        }
        py::object res = f(py_ctx, *args);
        return w2t4_unpack(res, false);
    }
    std::vector<std::shared_ptr<litetorch::Tensor>> backward(litetorch::FunctionContext* ctx,
        const std::vector<std::shared_ptr<litetorch::Tensor>>& grad_outputs) override {
        py::gil_scoped_acquire gil;
        py::function b = py::get_override(static_cast<const litetorch::Function*>(this), "backward");
        if (!b) throw std::runtime_error("[litetorch Error] autograd.Function.backward() must be overridden");
        py::object py_ctx;
        if (ctx->py_state) py_ctx = *static_cast<py::object*>(ctx->py_state.get());
        else py_ctx = py::cast(ctx, py::return_value_policy::reference);
        py::tuple g(grad_outputs.size());
        for (size_t i = 0; i < grad_outputs.size(); ++i) {
            if (grad_outputs[i]) g[i] = py::cast(grad_outputs[i]);
            else g[i] = py::none();
        }
        py::object res = b(py_ctx, *g);
        auto full = w2t4_unpack(res, true);
        size_t n_tensor = 0;
        for (size_t i = 0; i < (size_t)arg_template.size(); ++i)
            if (arg_template[i].is_none()) ++n_tensor;
        std::vector<std::shared_ptr<litetorch::Tensor>> out;
        if (full.size() == n_tensor) {
            out = full;
        } else if (full.size() == (size_t)arg_template.size()) {
            for (size_t i = 0; i < (size_t)arg_template.size(); ++i)
                if (arg_template[i].is_none()) out.push_back(full[i]);
        } else {
            throw std::runtime_error("[litetorch Error] autograd.Function.backward: must return a grad (or None) per forward input");
        }
        return out;
    }
};
}
// === end W2-T4 autograd ===

class PyModule : public nn::Module {
public:
    using nn::Module::Module;
    using ParamList = std::vector<std::shared_ptr<Tensor>>;

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        PYBIND11_OVERRIDE_PURE(
            std::shared_ptr<Tensor>,
            nn::Module,
            forward,
            input
        );
    }

    ParamList parameters() override {
        PYBIND11_OVERRIDE(
            ParamList,
            nn::Module,
            parameters
        );
    }

    void to(const Device& device) override {
        PYBIND11_OVERRIDE(
            void,
            nn::Module,
            to,
            device
        );
    }
};

class PyDataset : public data::Dataset {
public:
    using data::Dataset::Dataset;
    using ItemType = std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>;

    size_t size() override {
        PYBIND11_OVERRIDE_PURE(
            size_t,
            data::Dataset,
            size
        );
    }

    ItemType get(size_t index) override {
        PYBIND11_OVERRIDE_PURE(
            ItemType,
            data::Dataset,
            get,
            index
        );
    }

    bool is_python_backed() const override { return true; }
};

struct EnableGradContext {
    bool prev = true;
    bool active = false;
    void enter() {
        if (!active) {
            prev = Autograd::is_grad_enabled();
            Autograd::set_grad_enabled(true);
            active = true;
        }
    }
    void exit() {
        if (active) {
            Autograd::set_grad_enabled(prev);
            active = false;
        }
    }
};

inline bool& inference_mode_flag() {
    static thread_local bool flag = false;
    return flag;
}

struct InferenceModeContext {
    bool prev_grad = true;
    bool prev_infer = false;
    bool active = false;
    void enter() {
        if (!active) {
            prev_grad = Autograd::is_grad_enabled();
            prev_infer = inference_mode_flag();
            Autograd::set_grad_enabled(false);
            inference_mode_flag() = true;
            active = true;
        }
    }
    void exit() {
        if (active) {
            Autograd::set_grad_enabled(prev_grad);
            inference_mode_flag() = prev_infer;
            active = false;
        }
    }
};

inline bool gpu_available_for_autocast() {
    if (std::getenv("LITETORCH_NO_NATIVE_GPU")) {
        return litetorch::CLBackend::get().is_available();
    }
    auto backend = litetorch::BackendDispatcher::get().get_backend();
    if (backend && backend->is_available()) {
        return true;
    }
    return litetorch::CLBackend::get().is_available();
}

inline DataType parse_autocast_dtype(py::object dtype_obj, const std::string& device_type) {
    if (dtype_obj.is_none()) {
        return (device_type == "cpu") ? DataType::BF16 : DataType::FP16;
    }
    try {
        return dtype_obj.cast<DataType>();
    } catch (const py::cast_error&) {
    }
    std::string s = dtype_obj.cast<std::string>();
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "float16" || s == "fp16" || s == "half") return DataType::FP16;
    if (s == "bfloat16" || s == "bf16") return DataType::BF16;
    if (s == "float32" || s == "fp32" || s == "float") return DataType::FP32;
    throw std::invalid_argument("[litetorch Error] autocast: unsupported dtype '" + s + "'");
}

struct PyAutocast {
    std::string device_type = "cuda";
    DataType dtype = DataType::FP16;
    bool enabled = true;
    std::unique_ptr<amp::AutocastGuard> guard;

    void enter() {
        std::string dt = device_type;
        std::transform(dt.begin(), dt.end(), dt.begin(), ::tolower);
        if (dt == "cuda" || dt == "gpu") {
            if (!gpu_available_for_autocast()) {
                throw std::runtime_error("[litetorch Error] autocast('cuda') requested but no GPU backend is available on this build");
            }
        } else if (dt != "cpu") {
            throw std::invalid_argument("[litetorch Error] autocast: device_type must be 'cuda' or 'cpu'");
        }
        guard = std::make_unique<amp::AutocastGuard>(enabled, dtype);
    }
    void exit() {
        guard.reset();
    }
};


PYBIND11_MODULE(litetorch, m) {
    m.doc() = "LiteTorch Python Bindings";

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

    auto tpu_mod = m.def_submodule("tpu");
    tpu_mod.def("is_available", &litetorch::tpu::is_available);
    tpu_mod.def("device_count", &litetorch::tpu::device_count);
    tpu_mod.def("current_device", &litetorch::tpu::current_device);
    tpu_mod.def("set_device", &litetorch::tpu::set_device, py::arg("device_id") = 0);
    tpu_mod.def("synchronize", &litetorch::tpu::synchronize);
    tpu_mod.def("get_device_name", &litetorch::tpu::get_device_name, py::arg("device_id") = 0);

    py::enum_<DataType>(m, "DataType")
        .value("FP64", DataType::FP64)
        .value("FP32", DataType::FP32)
        .value("FP16", DataType::FP16)
        .value("BF16", DataType::BF16)
        .value("INT16", DataType::INT16)
        .value("INT8", DataType::INT8)
        .value("INT4", DataType::INT4)
        .value("FP8_E4M3", DataType::FP8_E4M3)
        .value("FP8_E5M2", DataType::FP8_E5M2)
        .value("NF4", DataType::NF4)
        .value("FP4_E2M1", DataType::FP4_E2M1)
        .value("FP4", DataType::FP4_E2M1)
        .value("INT32", DataType::INT32)
        .export_values();

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def_static("create", &Tensor::create, py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32)
        .def_static("from_vector", &Tensor::from_vector, py::arg("data"), py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32)
        .def_static("zeros", static_cast<std::shared_ptr<Tensor>(*)(const std::vector<int64_t>&, const Device&, bool)>(&Tensor::zeros), py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false)
        .def_static("zeros", static_cast<std::shared_ptr<Tensor>(*)(const std::vector<int64_t>&, const Device&, bool, DataType)>(&Tensor::zeros), py::arg("shape"), py::arg("device"), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32)
        .def_static("ones", &Tensor::ones, py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32)
        .def_static("randn", &Tensor::randn, py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32)
        .def_static("arange", &Tensor::arange, py::arg("start"), py::arg("end"), py::arg("step") = 1, py::arg("device") = Device(DeviceType::CPU, 0), py::arg("dtype") = DataType::FP32)
        .def("masked_fill", &Tensor::masked_fill, py::arg("mask"), py::arg("value"))
        .def("index_select", &Tensor::index_select, py::arg("dim"), py::arg("indices"))
        .def("slice", &Tensor::slice, py::arg("dim"), py::arg("start"), py::arg("end"), py::arg("step") = 1)
        .def("__getitem__", [](Tensor& self, py::object key) -> std::shared_ptr<Tensor> {
            if (py::isinstance<py::int_>(key)) {
                int64_t idx = key.cast<int64_t>();
                auto s = self.slice(0, idx, idx + 1);
                return Ops::squeeze(s, 0);
            } else if (py::isinstance<py::slice>(key)) {
                py::slice sl = key.cast<py::slice>();
                int64_t dim_size = self.shape[0];
                int64_t start, stop, step, slicelength;
                if (!sl.compute(dim_size, &start, &stop, &step, &slicelength)) {
                    throw py::index_error("invalid slice");
                }
                return self.slice(0, start, stop, step);
            }
            throw py::type_error("unsupported index type");
        })
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("strides", &Tensor::strides)
        .def_readwrite("dtype", &Tensor::dtype)
        .def_readwrite("device", &Tensor::device)
        .def_readwrite("requires_grad", &Tensor::requires_grad)
        .def_readwrite("grad", &Tensor::grad)
        .def("numel", &Tensor::numel)
        .def("item", &Tensor::item)
        .def("to_vector", &Tensor::to_vector)
        .def("backward", [](Tensor& self, std::shared_ptr<Tensor> grad, bool create_graph) {
            self.backward(grad, create_graph);
        }, py::arg("gradient") = nullptr, py::arg("create_graph") = false)
        .def("zero_grad", &Tensor::zero_grad)
        .def("retain_grad", [](std::shared_ptr<Tensor> self) { Autograd::retain_grad(self); })
        .def("to", py::overload_cast<const Device&>(&Tensor::to), py::arg("device"))
        .def("to", py::overload_cast<DataType>(&Tensor::to), py::arg("dtype"))
        .def("to", py::overload_cast<const Device&, DataType>(&Tensor::to), py::arg("device"), py::arg("dtype"))
        .def("view", [](Tensor& self, py::args args) {
            std::vector<int64_t> shape;
            if (args.size() == 1) {
                py::handle h = args[0];
                if (py::isinstance<py::list>(h) || py::isinstance<py::tuple>(h)) {
                    for (auto o : h.cast<py::sequence>()) shape.push_back(o.cast<int64_t>());
                    return self.view(shape);
                }
            }
            for (auto o : args) shape.push_back(o.cast<int64_t>());
            return self.view(shape);
        })
        .def("reshape", &Tensor::reshape)
        .def("as_strided", &Tensor::as_strided, py::arg("shape"), py::arg("strides"))
        .def("permute", &Tensor::permute, py::arg("dims"))
        .def("detach", &Tensor::detach)
        .def("expand", &Tensor::expand, py::arg("shape"))
        .def("gather", &Tensor::gather, py::arg("dim"), py::arg("index"))
        .def("argmax", &Tensor::argmax, py::arg("dim") = -1, py::arg("keepdim") = false)
        .def("argmin", &Tensor::argmin, py::arg("dim") = -1, py::arg("keepdim") = false)
        .def("topk", &Tensor::topk, py::arg("k"), py::arg("dim") = -1, py::arg("largest") = true, py::arg("sorted") = true)
        .def("eq", &Tensor::eq, py::arg("other"))
        .def("ne", &Tensor::ne, py::arg("other"))
        .def("lt", &Tensor::lt, py::arg("other"))
        .def("gt", &Tensor::gt, py::arg("other"))
        .def("le", &Tensor::le, py::arg("other"))
        .def("ge", &Tensor::ge, py::arg("other"))
        .def("log_softmax", &Tensor::log_softmax, py::arg("dim") = -1)
        .def("silu", &Tensor::silu)
        .def("sum", [](Tensor& self, py::object dim, bool keepdim) -> std::shared_ptr<Tensor> {
            auto s = self.shared_from_this();
            if (dim.is_none()) return Ops::sum(s);
            return s->sum(dim.cast<int64_t>(), keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def("mean", [](Tensor& self, py::object dim, bool keepdim) -> std::shared_ptr<Tensor> {
            auto s = self.shared_from_this();
            if (dim.is_none()) return Ops::mean(s);
            return s->mean(dim.cast<int64_t>(), keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def("max", [](Tensor& self, py::object dim, bool keepdim) -> py::object {
            auto s = self.shared_from_this();
            if (dim.is_none()) return py::cast(Ops::max(s));
            return py::cast(s->max(dim.cast<int64_t>(), keepdim));
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def("__add__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return Ops::add(a, b); })
        .def("__add__", [](std::shared_ptr<Tensor> a, double b) { return Ops::add(a, Tensor::from_vector({(float)b}, {1}, a->device)); })
        .def("__radd__", [](std::shared_ptr<Tensor> a, double b) { return Ops::add(a, Tensor::from_vector({(float)b}, {1}, a->device)); })
        .def("__sub__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return Ops::sub(a, b); })
        .def("__sub__", [](std::shared_ptr<Tensor> a, double b) { return Ops::sub(a, Tensor::from_vector({(float)b}, {1}, a->device)); })
        .def("__rsub__", [](std::shared_ptr<Tensor> a, double b) { return Ops::sub(Tensor::from_vector({(float)b}, {1}, a->device), a); })
        .def("__mul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return Ops::mul(a, b); })
        .def("__mul__", [](std::shared_ptr<Tensor> a, double b) { return Ops::mul(a, Tensor::from_vector({(float)b}, {1}, a->device)); })
        .def("__rmul__", [](std::shared_ptr<Tensor> a, double b) { return Ops::mul(a, Tensor::from_vector({(float)b}, {1}, a->device)); })
        .def("__truediv__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return Ops::div(a, b); })
        .def("__truediv__", [](std::shared_ptr<Tensor> a, double b) { return Ops::div(a, Tensor::from_vector({(float)b}, {1}, a->device)); })
        .def("__rtruediv__", [](std::shared_ptr<Tensor> a, double b) { return Ops::div(Tensor::from_vector({(float)b}, {1}, a->device), a); })
        .def("__neg__", [](std::shared_ptr<Tensor> a) { return Ops::neg(a); })
        .def("__matmul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return Ops::matmul(a, b); })
        .def("__pow__", [](std::shared_ptr<Tensor> a, double b) { return Ops::pow(a, (float)b); })
        .def("transpose", &Tensor::transpose)
        .def("contiguous", &Tensor::contiguous)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("clone", &Tensor::clone)
        .def("to_device_async", &Tensor::to_device_async, py::arg("target_device"))
        .def("cast", &Tensor::cast, py::arg("target_dtype"))
        // === W1-T2 tensor-api ===
        .def("cuda", [](Tensor& self, py::object device) {
            int idx = -1;
            if (!device.is_none()) {
                idx = device.cast<int>();
            }
            return self.cuda(idx);
        }, py::arg("device") = py::none())
        .def("cpu", &Tensor::cpu)
        .def("dim", &Tensor::dim)
        .def_property_readonly("ndim", &Tensor::dim)
        .def("half", &Tensor::half)
        .def("float", &Tensor::to_float)
        .def("double", &Tensor::to_double)
        .def("long", &Tensor::to_int)
        .def("int", &Tensor::to_int)
        .def("type_as", &Tensor::type_as, py::arg("other"))
        .def("numpy", [](Tensor& self) -> py::object {
            if (self.device.type != DeviceType::CPU) {
                throw std::runtime_error("[litetorch Error] numpy(): cannot convert non-CPU tensor to numpy; call .cpu() first");
            }
            size_t esz = self.storage->element_size();
            if (self.dtype == DataType::FP32) {
                std::vector<ssize_t> shape(self.shape.begin(), self.shape.end());
                std::vector<ssize_t> strides;
                for (auto s : self.strides) {
                    strides.push_back(static_cast<ssize_t>(s * esz));
                }
                float* ptr = reinterpret_cast<float*>(
                    reinterpret_cast<char*>(self.storage->get_cpu_ptr()) + self.offset * esz);
                auto* holder = new std::shared_ptr<StorageImpl>(self.storage);
                py::capsule base(holder, [](void* p) {
                    delete static_cast<std::shared_ptr<StorageImpl>*>(p);
                });
                return py::array_t<float>(shape, strides, ptr, base);
            }
            std::shared_ptr<Tensor> t = self.shared_from_this();
            if (t->dtype == DataType::BF16) {
                t = t->cast(DataType::FP32);
            }
            if (!t->is_contiguous()) {
                t = t->contiguous();
            }
            std::vector<ssize_t> shape(t->shape.begin(), t->shape.end());
            size_t n = t->numel();
            size_t tsz = t->storage->element_size();
            const char* src = reinterpret_cast<const char*>(t->storage->get_cpu_ptr()) + t->offset * tsz;
            const char* fmt = nullptr;
            switch (t->dtype) {
                case DataType::FP64: fmt = "float64"; break;
                case DataType::FP32: fmt = "float32"; break;
                case DataType::FP16: fmt = "float16"; break;
                case DataType::INT32: fmt = "int32"; break;
                case DataType::INT16: fmt = "int16"; break;
                case DataType::INT8: fmt = "int8"; break;
                default: throw std::runtime_error("[litetorch Error] numpy(): unsupported dtype for copy conversion");
            }
            py::array arr(py::dtype(fmt), shape);
            std::memcpy(arr.mutable_data(), src, n * tsz);
            return arr;
        })
        .def("tolist", [](Tensor& self) -> py::object {
            std::shared_ptr<Tensor> t = self.shared_from_this();
            if (t->device.type != DeviceType::CPU) {
                t = t->to(Device(DeviceType::CPU, 0));
            }
            if (!t->is_contiguous()) {
                t = t->contiguous();
            }
            bool as_int = (t->dtype == DataType::INT32 || t->dtype == DataType::INT16 ||
                           t->dtype == DataType::INT8);
            std::vector<double> vals = t->flat_doubles();
            std::function<py::object(size_t, size_t)> build = [&](size_t dim, size_t base) -> py::object {
                if (dim == t->shape.size()) {
                    if (as_int) {
                        return py::int_(static_cast<int64_t>(vals[base]));
                    }
                    return py::float_(vals[base]);
                }
                py::list lst;
                size_t stride = 1;
                for (size_t d = dim + 1; d < t->shape.size(); ++d) {
                    stride *= static_cast<size_t>(t->shape[d]);
                }
                for (int64_t i = 0; i < t->shape[dim]; ++i) {
                    lst.append(build(dim + 1, base + static_cast<size_t>(i) * stride));
                }
                return lst;
            };
            return build(0, 0);
        })
        // === W1-T2 tensor-api ===
        .def("copy_", &Tensor::copy_)
        .def("add_", &Tensor::add_)
        .def("register_hook", &Tensor::register_hook)
        .def("__repr__", [](const Tensor& self) {
            return "<litetorch.Tensor shape=" + std::to_string(self.numel()) + " device=" + self.device.to_string() + ">";
        });

    auto ops = m.def_submodule("Ops", "LiteTorch Tensor Operations");
    ops.def("add", &Ops::add);
    ops.def("sub", &Ops::sub);
    ops.def("mul", &Ops::mul);
    ops.def("div", &Ops::div);
    ops.def("matmul", &Ops::matmul);
    ops.def("matmul_bias", &Ops::matmul_bias, py::arg("a"), py::arg("b"), py::arg("bias"));
    ops.def("bmm", &Ops::bmm);
    ops.def("sum", py::overload_cast<std::shared_ptr<Tensor>>(&Ops::sum), py::arg("a"));
    ops.def("sum", py::overload_cast<std::shared_ptr<Tensor>, int64_t, bool>(&Ops::sum), py::arg("a"), py::arg("dim"), py::arg("keepdim") = false);
    ops.def("mean", py::overload_cast<std::shared_ptr<Tensor>>(&Ops::mean), py::arg("a"));
    ops.def("mean", py::overload_cast<std::shared_ptr<Tensor>, int64_t, bool>(&Ops::mean), py::arg("a"), py::arg("dim"), py::arg("keepdim") = false);
    ops.def("max", py::overload_cast<std::shared_ptr<Tensor>>(&Ops::max), py::arg("a"));
    ops.def("max", py::overload_cast<std::shared_ptr<Tensor>, int64_t, bool>(&Ops::max), py::arg("a"), py::arg("dim"), py::arg("keepdim") = false);
    ops.def("max_with_indices", &Ops::max_with_indices, py::arg("a"), py::arg("dim"), py::arg("keepdim") = false);
    ops.def("pow", &Ops::pow, py::arg("a"), py::arg("exponent"));
    ops.def("sqrt", &Ops::sqrt);
    ops.def("exp", &Ops::exp);
    ops.def("log", &Ops::log);
    ops.def("abs", &Ops::abs);
    ops.def("neg", &Ops::neg);
    ops.def("clamp", &Ops::clamp, py::arg("a"), py::arg("min_val"), py::arg("max_val"));
    ops.def("sin", &Ops::sin);
    ops.def("cos", &Ops::cos);
    ops.def("relu", &Ops::relu);
    ops.def("leaky_relu", &Ops::leaky_relu, py::arg("a"), py::arg("negative_slope") = 0.01f);
    ops.def("sigmoid", &Ops::sigmoid);
    ops.def("tanh", &Ops::tanh);
    ops.def("softmax", &Ops::softmax, py::arg("a"), py::arg("dim") = -1);
    ops.def("gelu", &Ops::gelu);
    ops.def("layer_norm", &Ops::layer_norm, py::arg("input"), py::arg("normalized_shape"), py::arg("weight") = nullptr, py::arg("bias") = nullptr, py::arg("eps") = 1e-5f);
    ops.def("fused_add_layernorm", &Ops::fused_add_layernorm, py::arg("input"), py::arg("residual"), py::arg("normalized_shape"), py::arg("weight") = nullptr, py::arg("bias") = nullptr, py::arg("eps") = 1e-5f);
    ops.def("batch_norm2d", &Ops::batch_norm2d, py::arg("input"), py::arg("running_mean"), py::arg("running_var"), py::arg("weight"), py::arg("bias"), py::arg("training"), py::arg("momentum") = 0.1f, py::arg("eps") = 1e-5f);
    ops.def("conv2d", &Ops::conv2d, py::arg("input"), py::arg("weight"), py::arg("bias") = nullptr, py::arg("stride") = 1, py::arg("padding") = 0);
    ops.def("conv2d_relu", &Ops::conv2d_relu, py::arg("input"), py::arg("weight"), py::arg("bias") = nullptr, py::arg("stride") = 1, py::arg("padding") = 0);
    ops.def("conv3d", &Ops::conv3d, py::arg("input"), py::arg("weight"), py::arg("bias") = nullptr, py::arg("stride") = 1, py::arg("padding") = 0);
    ops.def("max_pool2d", &Ops::max_pool2d, py::arg("input"), py::arg("kernel_size"), py::arg("stride") = -1, py::arg("padding") = 0);
    ops.def("max_pool3d", &Ops::max_pool3d, py::arg("input"), py::arg("kernel_size"), py::arg("stride") = -1, py::arg("padding") = 0);
    ops.def("adaptive_avg_pool2d", &Ops::adaptive_avg_pool2d, py::arg("input"), py::arg("output_height"), py::arg("output_width"));
    ops.def("cat", &Ops::cat, py::arg("tensors"), py::arg("dim") = 0);
    ops.def("chunk", &Ops::chunk, py::arg("input"), py::arg("chunks"), py::arg("dim") = 0);
    ops.def("split", &Ops::split, py::arg("input"), py::arg("sizes"), py::arg("dim") = -1);
    ops.def("squeeze", [](std::shared_ptr<Tensor> a, py::object dim) {
        if (dim.is_none()) return Ops::squeeze(a);
        return Ops::squeeze(a, dim.cast<int64_t>());
    }, py::arg("a"), py::arg("dim") = py::none());
    ops.def("unsqueeze", &Ops::unsqueeze, py::arg("a"), py::arg("dim"));
    ops.def("permute", &Ops::permute, py::arg("a"), py::arg("dims"));
    ops.def("expand", &Ops::expand, py::arg("a"), py::arg("shape"));
    ops.def("stack", &Ops::stack, py::arg("tensors"), py::arg("dim") = 0);
    ops.def("gather", &Ops::gather, py::arg("input"), py::arg("dim"), py::arg("index"));
    ops.def("argmax", &Ops::argmax, py::arg("a"), py::arg("dim"), py::arg("keepdim") = false);
    ops.def("argmin", &Ops::argmin, py::arg("a"), py::arg("dim"), py::arg("keepdim") = false);
    ops.def("topk", &Ops::topk, py::arg("a"), py::arg("k"), py::arg("dim") = -1, py::arg("largest") = true, py::arg("sorted") = true);
    ops.def("eq", &Ops::eq, py::arg("a"), py::arg("b"));
    ops.def("ne", &Ops::ne, py::arg("a"), py::arg("b"));
    ops.def("lt", &Ops::lt, py::arg("a"), py::arg("b"));
    ops.def("gt", &Ops::gt, py::arg("a"), py::arg("b"));
    ops.def("le", &Ops::le, py::arg("a"), py::arg("b"));
    ops.def("ge", &Ops::ge, py::arg("a"), py::arg("b"));
    ops.def("where", &Ops::where, py::arg("cond"), py::arg("a"), py::arg("b"));
    ops.def("log_softmax", &Ops::log_softmax, py::arg("a"), py::arg("dim") = -1);
    ops.def("silu", &Ops::silu, py::arg("a"));
    ops.def("cast", &Ops::cast, py::arg("a"), py::arg("target_dtype"));
    ops.def("fake_quantize", &Ops::fake_quantize, py::arg("input"), py::arg("scale"), py::arg("zero_point") = 0.0f, py::arg("bits") = 8);
    ops.def("mse_loss", &Ops::mse_loss);
    ops.def("cross_entropy_loss", &Ops::cross_entropy_loss,
        py::arg("input"), py::arg("target"),
        py::arg("weight") = nullptr,
        py::arg("ignore_index") = -100,
        py::arg("reduction") = "mean",
        py::arg("label_smoothing") = 0.0f);
    ops.def("l1_loss", &Ops::l1_loss);
    ops.def("bce_loss", &Ops::bce_loss);
    ops.def("fused_linear_cross_entropy", &Ops::fused_linear_cross_entropy, py::arg("x"), py::arg("weight"), py::arg("target"), py::arg("bias") = nullptr);
    ops.def("clip_grad_norm_", &Ops::clip_grad_norm_, py::arg("params"), py::arg("max_norm"), py::arg("norm_type") = 2.0f);
    ops.def("rmsnorm", &Ops::rmsnorm, py::arg("input"), py::arg("weight"), py::arg("eps") = 1e-5f);
    ops.def("qkv_extract", &Ops::qkv_extract, py::arg("qkv"), py::arg("num_heads"), py::arg("index"));
    ops.def("rope", &Ops::rope, py::arg("x"), py::arg("cos"), py::arg("sin"));

    // === W1-T1 core-ops ===
    auto w1_parse_dims = [](py::object dim, int64_t ndim) {
        std::vector<int64_t> dims;
        if (py::isinstance<py::int_>(dim)) {
            dims.push_back(dim.cast<int64_t>());
        } else {
            for (auto item : dim.cast<py::sequence>()) dims.push_back(item.cast<int64_t>());
        }
        return dims;
    };
    ops.def("log2", &Ops::log2, py::arg("a"));
    ops.def("log10", &Ops::log10, py::arg("a"));
    ops.def("rsqrt", &Ops::rsqrt, py::arg("a"));
    ops.def("tan", &Ops::tan, py::arg("a"));
    ops.def("reciprocal", &Ops::reciprocal, py::arg("a"));
    ops.def("sign", &Ops::sign, py::arg("a"));
    ops.def("floor", &Ops::floor, py::arg("a"));
    ops.def("ceil", &Ops::ceil, py::arg("a"));
    ops.def("round", &Ops::round, py::arg("a"));
    ops.def("trunc", &Ops::trunc, py::arg("a"));
    ops.def("clamp_min", &Ops::clamp_min, py::arg("a"), py::arg("min"));
    ops.def("clamp_max", &Ops::clamp_max, py::arg("a"), py::arg("max"));
    ops.def("clamp", [](std::shared_ptr<Tensor> a, py::object min_v, py::object max_v) {
        bool has_min = !min_v.is_none();
        bool has_max = !max_v.is_none();
        if (has_min && has_max) return Ops::clamp(a, min_v.cast<float>(), max_v.cast<float>());
        if (has_min) return Ops::clamp_min(a, min_v.cast<float>());
        if (has_max) return Ops::clamp_max(a, max_v.cast<float>());
        throw std::runtime_error("[litetorch Error] clamp: at least one of min or max must be given");
    }, py::arg("a"), py::arg("min") = py::none(), py::arg("max") = py::none());
    ops.def("min", &Ops::min, py::arg("a"));
    ops.def("amax", [w1_parse_dims](std::shared_ptr<Tensor> a, py::object dim, bool keepdim) {
        return Ops::amax(a, w1_parse_dims(dim, (int64_t)a->shape.size()), keepdim);
    }, py::arg("a"), py::arg("dim"), py::arg("keepdim") = false);
    ops.def("amin", [w1_parse_dims](std::shared_ptr<Tensor> a, py::object dim, bool keepdim) {
        return Ops::amin(a, w1_parse_dims(dim, (int64_t)a->shape.size()), keepdim);
    }, py::arg("a"), py::arg("dim"), py::arg("keepdim") = false);
    ops.def("sum", py::overload_cast<std::shared_ptr<Tensor>, const std::vector<int64_t>&, bool>(&Ops::sum),
        py::arg("a"), py::arg("dim"), py::arg("keepdim") = false);
    ops.def("mean", py::overload_cast<std::shared_ptr<Tensor>, const std::vector<int64_t>&, bool>(&Ops::mean),
        py::arg("a"), py::arg("dim"), py::arg("keepdim") = false);
    ops.def("max", py::overload_cast<std::shared_ptr<Tensor>, const std::vector<int64_t>&, bool>(&Ops::max),
        py::arg("a"), py::arg("dim"), py::arg("keepdim") = false);
    ops.def("allclose", &Ops::allclose, py::arg("a"), py::arg("b"),
        py::arg("rtol") = 1e-5f, py::arg("atol") = 1e-8f, py::arg("equal_nan") = false);
    ops.def("isclose", &Ops::isclose, py::arg("a"), py::arg("b"),
        py::arg("rtol") = 1e-5f, py::arg("atol") = 1e-8f, py::arg("equal_nan") = false);
    ops.def("triu", &Ops::triu, py::arg("a"), py::arg("diagonal") = 0);
    ops.def("tril", &Ops::tril, py::arg("a"), py::arg("diagonal") = 0);
    ops.def("sort", &Ops::sort, py::arg("a"), py::arg("dim") = -1, py::arg("descending") = false);
    ops.def("argsort", &Ops::argsort, py::arg("a"), py::arg("dim") = -1, py::arg("descending") = false);
    ops.def("cumsum", &Ops::cumsum, py::arg("a"), py::arg("dim"));
    ops.def("scatter", &Ops::scatter, py::arg("input"), py::arg("dim"), py::arg("index"), py::arg("src"));
    ops.def("scatter_add", &Ops::scatter_add, py::arg("input"), py::arg("dim"), py::arg("index"), py::arg("src"));
    ops.def("flatten", &Ops::flatten, py::arg("a"), py::arg("start_dim") = 0, py::arg("end_dim") = -1);
    ops.def("repeat", &Ops::repeat, py::arg("a"), py::arg("repeats"));
    ops.def("narrow", &Ops::narrow, py::arg("a"), py::arg("dim"), py::arg("start"), py::arg("length"));
    ops.def("t", &Ops::t, py::arg("a"));
    // === end W1-T1 core-ops ===
    ops.def("flash_attention", &Ops::flash_attention, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
    ops.def("flash_attention_qkv", &Ops::flash_attention_qkv, py::arg("qkv"), py::arg("num_heads"), py::arg("causal") = false);
    ops.def("paged_attention", &Ops::paged_attention);
    ops.def("scaled_matmul", &Ops::scaled_matmul, py::arg("a"), py::arg("b"), py::arg("a_scale"), py::arg("b_scale"), py::arg("bias") = nullptr, py::arg("out_dtype") = DataType::FP32);
    ops.def("w8a8_matmul", &Ops::w8a8_matmul, py::arg("x"), py::arg("w"), py::arg("x_scale"), py::arg("w_scale"));
    ops.def("moe_gate", [](std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> gate_weight, int top_k) {
        std::shared_ptr<Tensor> indices_out;
        auto probs = Ops::moe_gate(input, gate_weight, top_k, indices_out);
        return std::make_pair(probs, indices_out);
    });
    ops.def("flash_decoding", &Ops::flash_decoding);
    ops.def("ring_attention", &Ops::ring_attention);
    ops.def("checkpoint", &Ops::checkpoint, py::arg("module"), py::arg("input"));

    m.def("checkpoint", &litetorch::checkpoint, "Activation Checkpointing helper");

    auto nn_mod = m.def_submodule("nn", "LiteTorch Neural Network Modules");
    
    py::class_<nn::Module, PyModule, std::shared_ptr<nn::Module>>(nn_mod, "Module")
        .def(py::init<>())
        .def("forward", &nn::Module::forward)
        .def("parameters", &nn::Module::parameters)
        .def("to", &nn::Module::to)
        .def("half", &nn::Module::half)
        .def("train", &nn::Module::train)
        .def("eval", &nn::Module::eval)
        .def("compile", &nn::Module::compile)
        .def("save_pretrained", &nn::Module::save_pretrained, py::arg("path"), py::arg("num_shards") = 1)
        .def("load_pretrained", &nn::Module::load_pretrained, py::arg("path"))
        .def("__call__", &nn::Module::operator());

    py::class_<nn::Linear, nn::Module, std::shared_ptr<nn::Linear>>(nn_mod, "Linear")
        .def(py::init<int, int, bool>(), py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true)
        .def("forward", &nn::Linear::forward)
        .def("parameters", &nn::Linear::parameters)
        .def("to", &nn::Linear::to)
        .def_readwrite("weight", &nn::Linear::weight)
        .def_readwrite("bias", &nn::Linear::bias)
        .def_readwrite("scales", &nn::Linear::scales);

    py::class_<nn::QLoRALinear, nn::Module, std::shared_ptr<nn::QLoRALinear>>(nn_mod, "QLoRALinear")
        .def(py::init<int, int, int, float, bool>(), py::arg("in_features"), py::arg("out_features"), py::arg("r") = 8, py::arg("lora_alpha") = 16.0f, py::arg("has_bias") = true)
        .def("forward", &nn::QLoRALinear::forward)
        .def("parameters", &nn::QLoRALinear::parameters)
        .def("to", &nn::QLoRALinear::to)
        .def_readwrite("weight", &nn::QLoRALinear::weight)
        .def_readwrite("bias", &nn::QLoRALinear::bias)
        .def_readwrite("lora_A", &nn::QLoRALinear::lora_A)
        .def_readwrite("lora_B", &nn::QLoRALinear::lora_B)
        .def_readwrite("scaling", &nn::QLoRALinear::scaling);

    py::class_<nn::MoELinear, nn::Module, std::shared_ptr<nn::MoELinear>>(nn_mod, "MoELinear")
        .def(py::init<int, int, int, int>(), py::arg("in_features"), py::arg("out_features"), py::arg("num_experts"), py::arg("top_k"))
        .def("forward", &nn::MoELinear::forward)
        .def("parameters", &nn::MoELinear::parameters)
        .def("to", &nn::MoELinear::to)
        .def_readwrite("gate_weight", &nn::MoELinear::gate_weight)
        .def_readwrite("experts", &nn::MoELinear::experts)
        .def_readwrite("num_experts", &nn::MoELinear::num_experts)
        .def_readwrite("top_k", &nn::MoELinear::top_k);

    py::class_<nn::ColumnParallelLinear, nn::Module, std::shared_ptr<nn::ColumnParallelLinear>>(nn_mod, "ColumnParallelLinear")
        .def(py::init<int, int, bool>(), py::arg("in_features"), py::arg("out_features"), py::arg("has_bias") = true)
        .def("forward", &nn::ColumnParallelLinear::forward)
        .def("parameters", &nn::ColumnParallelLinear::parameters)
        .def("to", &nn::ColumnParallelLinear::to)
        .def_readwrite("weight", &nn::ColumnParallelLinear::weight)
        .def_readwrite("bias", &nn::ColumnParallelLinear::bias);

    py::class_<nn::RowParallelLinear, nn::Module, std::shared_ptr<nn::RowParallelLinear>>(nn_mod, "RowParallelLinear")
        .def(py::init<int, int, bool>(), py::arg("in_features"), py::arg("out_features"), py::arg("has_bias") = true)
        .def("forward", &nn::RowParallelLinear::forward)
        .def("parameters", &nn::RowParallelLinear::parameters)
        .def("to", &nn::RowParallelLinear::to)
        .def_readwrite("weight", &nn::RowParallelLinear::weight)
        .def_readwrite("bias", &nn::RowParallelLinear::bias);

    py::class_<nn::Conv2d, nn::Module, std::shared_ptr<nn::Conv2d>>(nn_mod, "Conv2d")
        .def(py::init([](int in_channels, int out_channels, py::object kernel_size,
                         py::object stride, py::object padding, bool has_bias) {
            auto to_vec = [](py::object o) {
                if (py::isinstance<py::int_>(o)) return std::vector<int>{o.cast<int>()};
                return o.cast<std::vector<int>>();
            };
            return std::make_shared<nn::Conv2d>(in_channels, out_channels, to_vec(kernel_size),
                                               to_vec(stride), to_vec(padding), has_bias);
        }), py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"),
           py::arg("stride") = 1, py::arg("padding") = 0, py::arg("has_bias") = true)
        .def("forward", &nn::Conv2d::forward)
        .def("parameters", &nn::Conv2d::parameters)
        .def("to", &nn::Conv2d::to)
        .def_readwrite("weight", &nn::Conv2d::weight)
        .def_readwrite("bias", &nn::Conv2d::bias)
        .def_readwrite("stride", &nn::Conv2d::stride)
        .def_readwrite("padding", &nn::Conv2d::padding)
        .def_readwrite("kernel_size", &nn::Conv2d::kernel_size)
        .def_readwrite("stride_vec", &nn::Conv2d::stride_vec)
        .def_readwrite("padding_vec", &nn::Conv2d::padding_vec);

    py::class_<nn::Conv3d, nn::Module, std::shared_ptr<nn::Conv3d>>(nn_mod, "Conv3d")
        .def(py::init<int, int, int, int, int, bool>(), py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"), py::arg("stride") = 1, py::arg("padding") = 0, py::arg("has_bias") = true)
        .def("forward", &nn::Conv3d::forward)
        .def("parameters", &nn::Conv3d::parameters)
        .def("to", &nn::Conv3d::to)
        .def_readwrite("weight", &nn::Conv3d::weight)
        .def_readwrite("bias", &nn::Conv3d::bias)
        .def_readwrite("stride", &nn::Conv3d::stride)
        .def_readwrite("padding", &nn::Conv3d::padding);

    py::class_<nn::BatchNorm2d, nn::Module, std::shared_ptr<nn::BatchNorm2d>>(nn_mod, "BatchNorm2d")
        .def(py::init<int, float, float, bool, bool>(), py::arg("num_features"), py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f, py::arg("affine") = true, py::arg("track_running_stats") = true)
        .def("forward", &nn::BatchNorm2d::forward)
        .def("parameters", &nn::BatchNorm2d::parameters)
        .def("to", &nn::BatchNorm2d::to)
        .def_readwrite("weight", &nn::BatchNorm2d::weight)
        .def_readwrite("bias", &nn::BatchNorm2d::bias)
        .def_readwrite("running_mean", &nn::BatchNorm2d::running_mean)
        .def_readwrite("running_var", &nn::BatchNorm2d::running_var)
        .def_readwrite("num_features", &nn::BatchNorm2d::num_features)
        .def_readwrite("eps", &nn::BatchNorm2d::eps)
        .def_readwrite("momentum", &nn::BatchNorm2d::momentum)
        .def_readwrite("affine", &nn::BatchNorm2d::affine)
        .def_readwrite("track_running_stats", &nn::BatchNorm2d::track_running_stats);

    py::class_<nn::MaxPool2d, nn::Module, std::shared_ptr<nn::MaxPool2d>>(nn_mod, "MaxPool2d")
        .def(py::init<int, int, int>(), py::arg("kernel_size"), py::arg("stride") = -1, py::arg("padding") = 0)
        .def("forward", &nn::MaxPool2d::forward)
        .def_readwrite("kernel_size", &nn::MaxPool2d::kernel_size)
        .def_readwrite("stride", &nn::MaxPool2d::stride)
        .def_readwrite("padding", &nn::MaxPool2d::padding);

    py::class_<nn::MaxPool3d, nn::Module, std::shared_ptr<nn::MaxPool3d>>(nn_mod, "MaxPool3d")
        .def(py::init<int, int, int>(), py::arg("kernel_size"), py::arg("stride") = -1, py::arg("padding") = 0)
        .def("forward", &nn::MaxPool3d::forward)
        .def_readwrite("kernel_size", &nn::MaxPool3d::kernel_size)
        .def_readwrite("stride", &nn::MaxPool3d::stride)
        .def_readwrite("padding", &nn::MaxPool3d::padding);

    py::class_<nn::AdaptiveAvgPool2d, nn::Module, std::shared_ptr<nn::AdaptiveAvgPool2d>>(nn_mod, "AdaptiveAvgPool2d")
        .def(py::init<int, int>(), py::arg("output_height"), py::arg("output_width"))
        .def("forward", &nn::AdaptiveAvgPool2d::forward)
        .def_readwrite("output_height", &nn::AdaptiveAvgPool2d::output_height)
        .def_readwrite("output_width", &nn::AdaptiveAvgPool2d::output_width);

    py::class_<nn::Dropout, nn::Module, std::shared_ptr<nn::Dropout>>(nn_mod, "Dropout")
        .def(py::init<float>(), py::arg("p") = 0.5f)
        .def("forward", &nn::Dropout::forward)
        .def_readwrite("p", &nn::Dropout::p);

    py::class_<nn::ReLU, nn::Module, std::shared_ptr<nn::ReLU>>(nn_mod, "ReLU")
        .def(py::init<>())
        .def("forward", &nn::ReLU::forward);

    py::class_<nn::LeakyReLU, nn::Module, std::shared_ptr<nn::LeakyReLU>>(nn_mod, "LeakyReLU")
        .def(py::init<float>(), py::arg("negative_slope") = 0.01f)
        .def("forward", &nn::LeakyReLU::forward)
        .def_readwrite("negative_slope", &nn::LeakyReLU::negative_slope);

    py::class_<nn::Sigmoid, nn::Module, std::shared_ptr<nn::Sigmoid>>(nn_mod, "Sigmoid")
        .def(py::init<>())
        .def("forward", &nn::Sigmoid::forward);

    py::class_<nn::Tanh, nn::Module, std::shared_ptr<nn::Tanh>>(nn_mod, "Tanh")
        .def(py::init<>())
        .def("forward", &nn::Tanh::forward);

    py::class_<nn::GELU, nn::Module, std::shared_ptr<nn::GELU>>(nn_mod, "GELU")
        .def(py::init<>())
        .def("forward", &nn::GELU::forward);

    py::class_<nn::Softmax, nn::Module, std::shared_ptr<nn::Softmax>>(nn_mod, "Softmax")
        .def(py::init<int64_t>(), py::arg("dim") = -1)
        .def("forward", &nn::Softmax::forward)
        .def_readwrite("dim", &nn::Softmax::dim);

    py::class_<nn::Flatten, nn::Module, std::shared_ptr<nn::Flatten>>(nn_mod, "Flatten")
        .def(py::init<>())
        .def("forward", &nn::Flatten::forward);

    py::class_<nn::Embedding, nn::Module, std::shared_ptr<nn::Embedding>>(nn_mod, "Embedding")
        .def(py::init([](int num_embeddings, int embedding_dim, py::object padding_idx) {
            int64_t pi = padding_idx.is_none() ? -1 : padding_idx.cast<int64_t>();
            return std::make_shared<nn::Embedding>(num_embeddings, embedding_dim, pi);
        }), py::arg("num_embeddings"), py::arg("embedding_dim"), py::arg("padding_idx") = py::none())
        .def("forward", &nn::Embedding::forward)
        .def("parameters", &nn::Embedding::parameters)
        .def("to", &nn::Embedding::to)
        .def_readwrite("weight", &nn::Embedding::weight)
        .def_readwrite("num_embeddings", &nn::Embedding::num_embeddings)
        .def_readwrite("embedding_dim", &nn::Embedding::embedding_dim)
        .def_readwrite("padding_idx", &nn::Embedding::padding_idx);

    py::class_<nn::LayerNorm, nn::Module, std::shared_ptr<nn::LayerNorm>>(nn_mod, "LayerNorm")
        .def(py::init<const std::vector<int64_t>&, float>(), py::arg("normalized_shape"), py::arg("eps") = 1e-5f)
        .def("forward", &nn::LayerNorm::forward)
        .def("parameters", &nn::LayerNorm::parameters)
        .def("to", &nn::LayerNorm::to)
        .def_readwrite("weight", &nn::LayerNorm::weight)
        .def_readwrite("bias", &nn::LayerNorm::bias)
        .def_readwrite("eps", &nn::LayerNorm::eps)
        .def_readwrite("normalized_shape", &nn::LayerNorm::normalized_shape);

    py::class_<nn::MultiHeadAttention, nn::Module, std::shared_ptr<nn::MultiHeadAttention>>(nn_mod, "MultiHeadAttention")
        .def(py::init<int, int, double, bool, int, int, bool>(), py::arg("embed_dim"), py::arg("num_heads"), py::arg("dropout") = 0.0, py::arg("bias") = true, py::arg("kdim") = -1, py::arg("vdim") = -1, py::arg("batch_first") = false)
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::MultiHeadAttention::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::MultiHeadAttention::forward))
        .def("forward", [](std::shared_ptr<nn::MultiHeadAttention> self, std::shared_ptr<Tensor> query,
                           std::shared_ptr<Tensor> key, std::shared_ptr<Tensor> value,
                           py::object attn_mask, py::object key_padding_mask, bool need_weights) {
            std::shared_ptr<Tensor> am = attn_mask.is_none() ? nullptr : attn_mask.cast<std::shared_ptr<Tensor>>();
            std::shared_ptr<Tensor> kpm = key_padding_mask.is_none() ? nullptr : key_padding_mask.cast<std::shared_ptr<Tensor>>();
            return self->forward(query, key, value, am, kpm, need_weights);
        }, py::arg("query"), py::arg("key"), py::arg("value"), py::arg("attn_mask") = py::none(),
           py::arg("key_padding_mask") = py::none(), py::arg("need_weights") = false)
        .def("parameters", &nn::MultiHeadAttention::parameters)
        .def("to", &nn::MultiHeadAttention::to)
        .def("train", &nn::MultiHeadAttention::train)
        .def("eval", &nn::MultiHeadAttention::eval)
        .def_readwrite("embed_dim", &nn::MultiHeadAttention::embed_dim)
        .def_readwrite("num_heads", &nn::MultiHeadAttention::num_heads)
        .def_readwrite("head_dim", &nn::MultiHeadAttention::head_dim)
        .def_readwrite("kdim", &nn::MultiHeadAttention::kdim)
        .def_readwrite("vdim", &nn::MultiHeadAttention::vdim)
        .def_readwrite("dropout", &nn::MultiHeadAttention::dropout_p)
        .def_readwrite("batch_first", &nn::MultiHeadAttention::batch_first);

    py::class_<nn::TransformerDecoderLayer, nn::Module, std::shared_ptr<nn::TransformerDecoderLayer>>(nn_mod, "TransformerDecoderLayer")
        .def(py::init<int, int, int>(), py::arg("embed_dim"), py::arg("num_heads"), py::arg("dim_feedforward") = 2048)
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::TransformerDecoderLayer::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::TransformerDecoderLayer::forward))
        .def("parameters", &nn::TransformerDecoderLayer::parameters)
        .def("to", &nn::TransformerDecoderLayer::to);

    py::class_<nn::RMSNorm, nn::Module, std::shared_ptr<nn::RMSNorm>>(nn_mod, "RMSNorm")
        .def(py::init<const std::vector<int64_t>&, float>(), py::arg("normalized_shape"), py::arg("eps") = 1e-5f)
        .def("forward", &nn::RMSNorm::forward)
        .def("parameters", &nn::RMSNorm::parameters)
        .def("to", &nn::RMSNorm::to)
        .def_readwrite("weight", &nn::RMSNorm::weight)
        .def_readwrite("eps", &nn::RMSNorm::eps)
        .def_readwrite("normalized_shape", &nn::RMSNorm::normalized_shape);

    py::class_<nn::GPTBlock, nn::Module, std::shared_ptr<nn::GPTBlock>>(nn_mod, "GPTBlock")
        .def(py::init<int, int, int, int, float>(), py::arg("d_model") = 2560, py::arg("n_q_heads") = 20,
             py::arg("n_kv_heads") = 5, py::arg("d_ff") = 6912, py::arg("rope_theta") = 500000.0f)
        .def("forward", &nn::GPTBlock::forward)
        .def("parameters", &nn::GPTBlock::parameters)
        .def("to", &nn::GPTBlock::to)
        .def_readwrite("d_model", &nn::GPTBlock::d_model)
        .def_readwrite("n_q_heads", &nn::GPTBlock::n_q_heads)
        .def_readwrite("n_kv_heads", &nn::GPTBlock::n_kv_heads)
        .def_readwrite("d_ff", &nn::GPTBlock::d_ff)
        .def_readwrite("head_dim", &nn::GPTBlock::head_dim)
        .def_readwrite("rope_theta", &nn::GPTBlock::rope_theta);

    py::class_<nn::GPT2B, nn::Module, std::shared_ptr<nn::GPT2B>>(nn_mod, "GPT2B")
        .def(py::init<int, int, int, int, int, int, float>(), py::arg("vocab_size") = 50257,
             py::arg("d_model") = 2560, py::arg("n_layers") = 28, py::arg("n_q_heads") = 20,
             py::arg("n_kv_heads") = 5, py::arg("d_ff") = 6912, py::arg("rope_theta") = 500000.0f)
        .def("forward", &nn::GPT2B::forward)
        .def("stage_forward", &nn::GPT2B::stage_forward, py::arg("input"), py::arg("stage"))
        .def("parameters", &nn::GPT2B::parameters)
        .def("to", &nn::GPT2B::to)
        .def("tie_weights", &nn::GPT2B::tie_weights)
        .def("pipeline_to", &nn::GPT2B::pipeline_to, py::arg("d0"), py::arg("d1"), py::arg("split") = -1)
        .def_readwrite("vocab_size", &nn::GPT2B::vocab_size)
        .def_readwrite("d_model", &nn::GPT2B::d_model)
        .def_readwrite("n_layers", &nn::GPT2B::n_layers)
        .def_readwrite("n_q_heads", &nn::GPT2B::n_q_heads)
        .def_readwrite("n_kv_heads", &nn::GPT2B::n_kv_heads)
        .def_readwrite("d_ff", &nn::GPT2B::d_ff)
        .def_readwrite("rope_theta", &nn::GPT2B::rope_theta)
        .def_readwrite("tied", &nn::GPT2B::tied)
        .def_readwrite("n_stages", &nn::GPT2B::n_stages)
        .def_readwrite("stage_split", &nn::GPT2B::stage_split)
        .def_readwrite("tok_emb", &nn::GPT2B::tok_emb)
        .def_readwrite("blocks", &nn::GPT2B::blocks)
        .def_readwrite("final_norm", &nn::GPT2B::final_norm)
        .def_readwrite("lm_head", &nn::GPT2B::lm_head);

    py::class_<nn::Sequential, nn::Module, std::shared_ptr<nn::Sequential>>(nn_mod, "Sequential")
        .def(py::init<>())
        .def(py::init<const std::vector<std::shared_ptr<nn::Module>>&>(), py::arg("modules"))
        .def("add", &nn::Sequential::add)
        .def("forward", &nn::Sequential::forward)
        .def("parameters", &nn::Sequential::parameters)
        .def("to", &nn::Sequential::to)
        .def("train", &nn::Sequential::train)
        .def("eval", &nn::Sequential::eval)
        .def("compile", &nn::Sequential::compile);

    // === W1-T3 nn-plumbing ===
    {
        py::object Module_ = nn_mod.attr("Module");
        py::is_method method_tag(Module_);
        Module_.attr("register_parameter") = py::cpp_function(
            [](std::shared_ptr<nn::Module> self, const std::string& name, std::shared_ptr<Tensor> t) {
                self->register_parameter(name, t);
            }, method_tag, py::arg("name"), py::arg("tensor"));
        Module_.attr("register_buffer") = py::cpp_function(
            [](std::shared_ptr<nn::Module> self, const std::string& name, std::shared_ptr<Tensor> t) {
                self->register_buffer(name, t);
            }, method_tag, py::arg("name"), py::arg("tensor"));
        Module_.attr("add_module") = py::cpp_function(
            [](std::shared_ptr<nn::Module> self, const std::string& name, std::shared_ptr<nn::Module> m) {
                self->add_module(name, m);
            }, method_tag, py::arg("name"), py::arg("module"));
        Module_.attr("named_parameters") = py::cpp_function(
            [](std::shared_ptr<nn::Module> self, const std::string& prefix) {
                py::list out;
                for (auto& kv : self->named_parameters(prefix)) out.append(py::make_tuple(kv.first, kv.second));
                return out;
            }, method_tag, py::arg("prefix") = "");
        Module_.attr("named_buffers") = py::cpp_function(
            [](std::shared_ptr<nn::Module> self, const std::string& prefix) {
                py::list out;
                for (auto& kv : self->named_buffers(prefix)) out.append(py::make_tuple(kv.first, kv.second));
                return out;
            }, method_tag, py::arg("prefix") = "");
        Module_.attr("buffers") = py::cpp_function(
            [](std::shared_ptr<nn::Module> self) { return self->buffers(); }, method_tag);
        Module_.attr("named_children") = py::cpp_function(
            [](std::shared_ptr<nn::Module> self) {
                py::list out;
                for (auto& kv : self->named_children()) out.append(py::make_tuple(kv.first, kv.second));
                return out;
            }, method_tag);
        Module_.attr("named_modules") = py::cpp_function(
            [](std::shared_ptr<nn::Module> self, const std::string& prefix) {
                py::list out;
                for (auto& kv : self->named_modules(prefix)) out.append(py::make_tuple(kv.first, kv.second));
                return out;
            }, method_tag, py::arg("prefix") = "");
        Module_.attr("state_dict") = py::cpp_function(
            [](std::shared_ptr<nn::Module> self) {
                py::dict d;
                for (auto& kv : self->state_dict()) d[py::str(kv.first)] = kv.second;
                return d;
            }, method_tag);
        Module_.attr("load_state_dict") = py::cpp_function(
            [](std::shared_ptr<nn::Module> self, py::dict sd, bool strict) {
                std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> vec;
                for (auto kv : sd) vec.push_back({kv.first.cast<std::string>(), kv.second.cast<std::shared_ptr<Tensor>>()});
                auto res = self->load_state_dict(vec, strict);
                return py::make_tuple(res.first, res.second);
            }, method_tag, py::arg("state_dict"), py::arg("strict") = true);
        Module_.attr("zero_grad") = py::cpp_function(
            [](std::shared_ptr<nn::Module> self) { self->zero_grad(); }, method_tag);
        Module_.attr("is_training") = py::cpp_function(
            [](std::shared_ptr<nn::Module> self) { return self->training; }, method_tag);
        Module_.attr("apply") = py::cpp_function(
            [](std::shared_ptr<nn::Module> self, py::function fn) {
                self->apply([&fn](std::shared_ptr<nn::Module> m) { fn(m); });
            }, method_tag, py::arg("fn"));
    }

    py::class_<nn::Parameter, Tensor, std::shared_ptr<nn::Parameter>>(nn_mod, "Parameter")
        .def(py::init([](std::shared_ptr<Tensor> data, bool requires_grad) {
            auto p = std::make_shared<nn::Parameter>(data->shape, data->device, data->dtype);
            p->copy_(data);
            p->requires_grad = requires_grad;
            return p;
        }), py::arg("data"), py::arg("requires_grad") = true);

    py::class_<nn::Buffer, Tensor, std::shared_ptr<nn::Buffer>>(nn_mod, "Buffer")
        .def(py::init([](std::shared_ptr<Tensor> data) {
            auto b = std::make_shared<nn::Buffer>(data->shape, data->device, data->dtype);
            b->copy_(data);
            return b;
        }), py::arg("data"));

    py::class_<nn::ModuleList, nn::Module, std::shared_ptr<nn::ModuleList>>(nn_mod, "ModuleList")
        .def(py::init<>())
        .def(py::init<const std::vector<std::shared_ptr<nn::Module>>&>(), py::arg("modules"))
        .def("append", &nn::ModuleList::append, py::arg("module"))
        .def("extend", &nn::ModuleList::extend, py::arg("modules"))
        .def("__len__", &nn::ModuleList::size)
        .def("__getitem__", [](nn::ModuleList& self, int64_t i) {
            int64_t n = static_cast<int64_t>(self.size());
            if (i < 0) i += n;
            if (i < 0 || i >= n) throw py::index_error();
            return self.modules[static_cast<size_t>(i)];
        }, py::arg("index"))
        .def("__setitem__", [](nn::ModuleList& self, int64_t i, std::shared_ptr<nn::Module> m) {
            int64_t n = static_cast<int64_t>(self.size());
            if (i < 0) i += n;
            if (i < 0 || i >= n) throw py::index_error();
            self.modules[static_cast<size_t>(i)] = m;
        }, py::arg("index"), py::arg("module"))
        .def("__iter__", [](nn::ModuleList& self) {
            return py::make_iterator(self.modules.begin(), self.modules.end());
        }, py::keep_alive<0, 1>());

    py::class_<nn::ModuleDict, nn::Module, std::shared_ptr<nn::ModuleDict>>(nn_mod, "ModuleDict")
        .def(py::init<>())
        .def("put", &nn::ModuleDict::put, py::arg("key"), py::arg("module"))
        .def("get", &nn::ModuleDict::get, py::arg("key"))
        .def("keys", &nn::ModuleDict::keys)
        .def("__len__", &nn::ModuleDict::size)
        .def("__getitem__", [](nn::ModuleDict& self, const std::string& key) {
            auto m = self.get(key);
            if (!m) throw py::key_error("ModuleDict key not found: '" + key + "'");
            return m;
        }, py::arg("key"))
        .def("__setitem__", &nn::ModuleDict::put, py::arg("key"), py::arg("module"))
        .def("__delitem__", &nn::ModuleDict::erase, py::arg("key"))
        .def("__contains__", &nn::ModuleDict::contains, py::arg("key"))
        .def("__iter__", [](nn::ModuleDict& self) {
            return py::make_iterator(self.items.begin(), self.items.end());
        }, py::keep_alive<0, 1>());
    // === end W1-T3 nn-plumbing ===

    py::class_<nn::MSELoss, nn::Module, std::shared_ptr<nn::MSELoss>>(nn_mod, "MSELoss")
        .def(py::init<const std::string&>(), py::arg("reduction") = "mean")
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::MSELoss::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::MSELoss::forward))
        .def_readwrite("reduction", &nn::MSELoss::reduction);

    py::class_<nn::CrossEntropyLoss, nn::Module, std::shared_ptr<nn::CrossEntropyLoss>>(nn_mod, "CrossEntropyLoss")
        .def(py::init([](py::object weight, int64_t ignore_index, const std::string& reduction, float label_smoothing) {
            std::shared_ptr<Tensor> w = weight.is_none() ? nullptr : weight.cast<std::shared_ptr<Tensor>>();
            return std::make_shared<nn::CrossEntropyLoss>(w, ignore_index, reduction, label_smoothing);
        }), py::arg("weight") = py::none(), py::arg("ignore_index") = -100, py::arg("reduction") = "mean", py::arg("label_smoothing") = 0.0f)
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::CrossEntropyLoss::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::CrossEntropyLoss::forward))
        .def("__call__", [](nn::CrossEntropyLoss& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
            return self.forward(input, target);
        })
        .def_readwrite("weight", &nn::CrossEntropyLoss::weight)
        .def_readwrite("ignore_index", &nn::CrossEntropyLoss::ignore_index)
        .def_readwrite("reduction", &nn::CrossEntropyLoss::reduction)
        .def_readwrite("label_smoothing", &nn::CrossEntropyLoss::label_smoothing);

    py::class_<nn::BCELoss, nn::Module, std::shared_ptr<nn::BCELoss>>(nn_mod, "BCELoss")
        .def(py::init<const std::string&>(), py::arg("reduction") = "mean")
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::BCELoss::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::BCELoss::forward))
        .def_readwrite("reduction", &nn::BCELoss::reduction);

    py::class_<nn::L1Loss, nn::Module, std::shared_ptr<nn::L1Loss>>(nn_mod, "L1Loss")
        .def(py::init<const std::string&>(), py::arg("reduction") = "mean")
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::L1Loss::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::L1Loss::forward))
        .def_readwrite("reduction", &nn::L1Loss::reduction);

    // === W1-T4 nn-layers ===
    ops.def("conv1d", &Ops::conv1d, py::arg("input"), py::arg("weight"), py::arg("bias") = nullptr, py::arg("stride") = 1, py::arg("padding") = 0);
    ops.def("avg_pool2d", &Ops::avg_pool2d, py::arg("input"), py::arg("kernel_size"), py::arg("stride") = -1, py::arg("padding") = 0);
    ops.def("nll_loss", &Ops::nll_loss, py::arg("input"), py::arg("target"), py::arg("weight") = nullptr,
        py::arg("ignore_index") = -100, py::arg("reduction") = "mean");
    ops.def("bce_with_logits_loss", &Ops::bce_with_logits_loss, py::arg("input"), py::arg("target"),
        py::arg("weight") = nullptr, py::arg("reduction") = "mean", py::arg("pos_weight") = nullptr);

    py::class_<nn::GroupNorm, nn::Module, std::shared_ptr<nn::GroupNorm>>(nn_mod, "GroupNorm")
        .def(py::init<int, int, float, bool>(), py::arg("num_groups"), py::arg("num_channels"), py::arg("eps") = 1e-5f, py::arg("affine") = true)
        .def("forward", &nn::GroupNorm::forward)
        .def("parameters", &nn::GroupNorm::parameters)
        .def("to", &nn::GroupNorm::to)
        .def_readwrite("weight", &nn::GroupNorm::weight)
        .def_readwrite("bias", &nn::GroupNorm::bias)
        .def_readwrite("num_groups", &nn::GroupNorm::num_groups)
        .def_readwrite("num_channels", &nn::GroupNorm::num_channels)
        .def_readwrite("eps", &nn::GroupNorm::eps)
        .def_readwrite("affine", &nn::GroupNorm::affine);

    py::class_<nn::BatchNorm1d, nn::Module, std::shared_ptr<nn::BatchNorm1d>>(nn_mod, "BatchNorm1d")
        .def(py::init<int, float, float, bool, bool>(), py::arg("num_features"), py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f, py::arg("affine") = true, py::arg("track_running_stats") = true)
        .def("forward", &nn::BatchNorm1d::forward)
        .def("parameters", &nn::BatchNorm1d::parameters)
        .def("to", &nn::BatchNorm1d::to)
        .def_readwrite("weight", &nn::BatchNorm1d::weight)
        .def_readwrite("bias", &nn::BatchNorm1d::bias)
        .def_readwrite("running_mean", &nn::BatchNorm1d::running_mean)
        .def_readwrite("running_var", &nn::BatchNorm1d::running_var)
        .def_readwrite("num_features", &nn::BatchNorm1d::num_features)
        .def_readwrite("eps", &nn::BatchNorm1d::eps)
        .def_readwrite("momentum", &nn::BatchNorm1d::momentum)
        .def_readwrite("affine", &nn::BatchNorm1d::affine)
        .def_readwrite("track_running_stats", &nn::BatchNorm1d::track_running_stats);

    py::class_<nn::Conv1d, nn::Module, std::shared_ptr<nn::Conv1d>>(nn_mod, "Conv1d")
        .def(py::init([](int in_channels, int out_channels, py::object kernel_size,
                         py::object stride, py::object padding, bool has_bias) {
            auto to_int = [](py::object o) {
                if (py::isinstance<py::int_>(o)) return o.cast<int>();
                auto v = o.cast<std::vector<int>>();
                if (v.size() != 1) throw std::runtime_error("[litetorch Error] Conv1d kernel_size/stride/padding must be an int or single-element tuple");
                return v[0];
            };
            return std::make_shared<nn::Conv1d>(in_channels, out_channels, to_int(kernel_size),
                                               to_int(stride), to_int(padding), has_bias);
        }), py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"),
           py::arg("stride") = 1, py::arg("padding") = 0, py::arg("has_bias") = true)
        .def("forward", &nn::Conv1d::forward)
        .def("parameters", &nn::Conv1d::parameters)
        .def("to", &nn::Conv1d::to)
        .def_readwrite("weight", &nn::Conv1d::weight)
        .def_readwrite("bias", &nn::Conv1d::bias)
        .def_readwrite("kernel_size", &nn::Conv1d::kernel_size)
        .def_readwrite("stride", &nn::Conv1d::stride)
        .def_readwrite("padding", &nn::Conv1d::padding);

    py::class_<nn::AvgPool2d, nn::Module, std::shared_ptr<nn::AvgPool2d>>(nn_mod, "AvgPool2d")
        .def(py::init<int, int, int>(), py::arg("kernel_size"), py::arg("stride") = -1, py::arg("padding") = 0)
        .def("forward", &nn::AvgPool2d::forward)
        .def_readwrite("kernel_size", &nn::AvgPool2d::kernel_size)
        .def_readwrite("stride", &nn::AvgPool2d::stride)
        .def_readwrite("padding", &nn::AvgPool2d::padding);

    py::class_<nn::NLLLoss, nn::Module, std::shared_ptr<nn::NLLLoss>>(nn_mod, "NLLLoss")
        .def(py::init([](py::object weight, int64_t ignore_index, const std::string& reduction) {
            std::shared_ptr<Tensor> w = weight.is_none() ? nullptr : weight.cast<std::shared_ptr<Tensor>>();
            return std::make_shared<nn::NLLLoss>(w, ignore_index, reduction);
        }), py::arg("weight") = py::none(), py::arg("ignore_index") = -100, py::arg("reduction") = "mean")
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::NLLLoss::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::NLLLoss::forward))
        .def("__call__", [](nn::NLLLoss& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
            return self.forward(input, target);
        })
        .def_readwrite("weight", &nn::NLLLoss::weight)
        .def_readwrite("ignore_index", &nn::NLLLoss::ignore_index)
        .def_readwrite("reduction", &nn::NLLLoss::reduction);

    py::class_<nn::BCEWithLogitsLoss, nn::Module, std::shared_ptr<nn::BCEWithLogitsLoss>>(nn_mod, "BCEWithLogitsLoss")
        .def(py::init([](py::object weight, const std::string& reduction, py::object pos_weight) {
            std::shared_ptr<Tensor> w = weight.is_none() ? nullptr : weight.cast<std::shared_ptr<Tensor>>();
            std::shared_ptr<Tensor> pw = pos_weight.is_none() ? nullptr : pos_weight.cast<std::shared_ptr<Tensor>>();
            return std::make_shared<nn::BCEWithLogitsLoss>(w, reduction, pw);
        }), py::arg("weight") = py::none(), py::arg("reduction") = "mean", py::arg("pos_weight") = py::none())
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::BCEWithLogitsLoss::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::BCEWithLogitsLoss::forward))
        .def("__call__", [](nn::BCEWithLogitsLoss& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
            return self.forward(input, target);
        })
        .def_readwrite("weight", &nn::BCEWithLogitsLoss::weight)
        .def_readwrite("reduction", &nn::BCEWithLogitsLoss::reduction)
        .def_readwrite("pos_weight", &nn::BCEWithLogitsLoss::pos_weight);

    py::class_<nn::SiLU, nn::Module, std::shared_ptr<nn::SiLU>>(nn_mod, "SiLU")
        .def(py::init<>())
        .def("forward", &nn::SiLU::forward);

    py::class_<nn::LogSoftmax, nn::Module, std::shared_ptr<nn::LogSoftmax>>(nn_mod, "LogSoftmax")
        .def(py::init<int64_t>(), py::arg("dim") = -1)
        .def("forward", &nn::LogSoftmax::forward)
        .def_readwrite("dim", &nn::LogSoftmax::dim);
    // === end W1-T4 nn-layers ===

    py::class_<nn::GuidedDecoder>(nn_mod, "GuidedDecoder")
        .def(py::init<const std::vector<std::string>&>(), py::arg("vocabulary"))
        .def("apply_mask", &nn::GuidedDecoder::apply_mask, py::arg("logits"), py::arg("prefix"), py::arg("target_pattern"))
        .def_readwrite("vocab", &nn::GuidedDecoder::vocab);

    py::class_<nn::Request, std::shared_ptr<nn::Request>>(nn_mod, "Request")
        .def(py::init<uint64_t, const std::vector<int64_t>&, int64_t>(), py::arg("id"), py::arg("prompt"), py::arg("max_new_tokens") = 32)
        .def_readwrite("id", &nn::Request::id)
        .def_readwrite("prompt_tokens", &nn::Request::prompt_tokens)
        .def_readwrite("generated_tokens", &nn::Request::generated_tokens)
        .def_readwrite("max_new_tokens", &nn::Request::max_new_tokens)
        .def_readwrite("prefill_processed", &nn::Request::prefill_processed)
        .def_readwrite("is_finished", &nn::Request::is_finished)
        .def_readwrite("block_ids", &nn::Request::block_ids);

    py::class_<nn::ServingScheduler, std::shared_ptr<nn::ServingScheduler>>(nn_mod, "ServingScheduler")
        .def(py::init<int, int, int, int>(), py::arg("num_blocks"), py::arg("block_size"), py::arg("max_num_seqs"), py::arg("max_num_batched_tokens"))
        .def("add_request", &nn::ServingScheduler::add_request)
        .def("step", &nn::ServingScheduler::step)
        .def("get_block_table_tensor", &nn::ServingScheduler::get_block_table_tensor)
        .def("get_context_lens_tensor", &nn::ServingScheduler::get_context_lens_tensor)
        .def_readwrite("block_size", &nn::ServingScheduler::block_size)
        .def_readwrite("max_num_seqs", &nn::ServingScheduler::max_num_seqs)
        .def_readwrite("max_num_batched_tokens", &nn::ServingScheduler::max_num_batched_tokens)
        .def_readwrite("pending_queue", &nn::ServingScheduler::pending_queue)
        .def_readwrite("running_queue", &nn::ServingScheduler::running_queue)
        .def_readwrite("free_blocks", &nn::ServingScheduler::free_blocks)
        .def_readwrite("num_blocks", &nn::ServingScheduler::num_blocks);

    py::class_<nn::PrefixCache>(nn_mod, "PrefixCache")
        .def_static("get", &nn::PrefixCache::get, py::return_value_policy::reference)
        .def("put", &nn::PrefixCache::put)
        .def("get_cache", &nn::PrefixCache::get_cache)
        .def("clear", &nn::PrefixCache::clear);

    py::class_<nn::SpeculativeEngine, std::shared_ptr<nn::SpeculativeEngine>>(nn_mod, "SpeculativeEngine")
        .def(py::init<std::shared_ptr<nn::Module>, std::shared_ptr<nn::Module>>())
        .def("generate", &nn::SpeculativeEngine::generate, py::arg("prompt"), py::arg("max_new_tokens"), py::arg("lookahead") = 4)
        .def_readwrite("draft_model", &nn::SpeculativeEngine::draft_model)
        .def_readwrite("target_model", &nn::SpeculativeEngine::target_model);

    py::class_<nn::MedusaHead, nn::Module, std::shared_ptr<nn::MedusaHead>>(nn_mod, "MedusaHead")
        .def(py::init<int, int>())
        .def("forward", &nn::MedusaHead::forward)
        .def("parameters", &nn::MedusaHead::parameters)
        .def("to", &nn::MedusaHead::to)
        .def_readwrite("weight", &nn::MedusaHead::weight)
        .def_readwrite("bias", &nn::MedusaHead::bias);

    py::class_<nn::MedusaEngine, std::shared_ptr<nn::MedusaEngine>>(nn_mod, "MedusaEngine")
        .def(py::init<std::shared_ptr<nn::Module>, const std::vector<std::shared_ptr<nn::MedusaHead>>&>())
        .def("generate", &nn::MedusaEngine::generate, py::arg("prompt"), py::arg("max_new_tokens"), py::arg("lookahead") = 2)
        .def_readwrite("base_model", &nn::MedusaEngine::base_model)
        .def_readwrite("medusa_heads", &nn::MedusaEngine::medusa_heads);

    auto init_mod = nn_mod.def_submodule("init", "Weight initialization");
    init_mod.def("kaiming_uniform_", &nn::init::kaiming_uniform_, py::arg("tensor"), py::arg("a") = 0.0f, py::arg("mode") = "fan_in", py::arg("nonlinearity") = "relu");
    init_mod.def("kaiming_normal_", &nn::init::kaiming_normal_, py::arg("tensor"), py::arg("a") = 0.0f, py::arg("mode") = "fan_in", py::arg("nonlinearity") = "relu");
    init_mod.def("xavier_uniform_", &nn::init::xavier_uniform_, py::arg("tensor"), py::arg("gain") = 1.0f);
    init_mod.def("xavier_normal_", &nn::init::xavier_normal_, py::arg("tensor"), py::arg("gain") = 1.0f);
    init_mod.def("normal_", &nn::init::normal_, py::arg("tensor"), py::arg("mean") = 0.0f, py::arg("std") = 1.0f);
    init_mod.def("uniform_", &nn::init::uniform_, py::arg("tensor"), py::arg("a") = 0.0f, py::arg("b") = 1.0f);
    init_mod.def("zeros_", &nn::init::zeros_, py::arg("tensor"));
    init_mod.def("ones_", &nn::init::ones_, py::arg("tensor"));
    init_mod.def("fill_", &nn::init::fill_, py::arg("tensor"), py::arg("value"));

    // === W1-T5 functional ===
    auto functional_mod = nn_mod.def_submodule("functional", "LiteTorch nn.functional");
    functional_mod.def("dropout", &functional::dropout,
        py::arg("x"), py::arg("p") = 0.5, py::arg("training") = true);
    functional_mod.def("linear", &functional::linear,
        py::arg("x"), py::arg("weight"), py::arg("bias") = nullptr);
    functional_mod.def("pad", &functional::pad,
        py::arg("x"), py::arg("pad"), py::arg("mode") = "constant", py::arg("value") = 0.0);
    functional_mod.def("interpolate",
        [](std::shared_ptr<Tensor> x, py::object size, py::object scale_factor,
           const std::string& mode, bool align_corners) {
            std::vector<int64_t> sz;
            std::vector<double> sf;
            if (!size.is_none()) {
                if (py::isinstance<py::int_>(size)) {
                    sz.push_back(size.cast<int64_t>());
                } else {
                    for (auto v : size.cast<py::sequence>()) sz.push_back(v.cast<int64_t>());
                }
            }
            if (!scale_factor.is_none()) {
                if (py::isinstance<py::float_>(scale_factor) ||
                    py::isinstance<py::int_>(scale_factor)) {
                    sf.push_back(scale_factor.cast<double>());
                } else {
                    for (auto v : scale_factor.cast<py::sequence>()) sf.push_back(v.cast<double>());
                }
            }
            return functional::interpolate(x, sz, sf, mode, align_corners);
        },
        py::arg("x"), py::arg("size") = py::none(), py::arg("scale_factor") = py::none(),
        py::arg("mode") = "nearest", py::arg("align_corners") = false);
    functional_mod.def("nll_loss", &functional::nll_loss,
        py::arg("input"), py::arg("target"), py::arg("weight") = nullptr,
        py::arg("reduction") = "mean", py::arg("ignore_index") = -100);
    functional_mod.def("bce_with_logits", &functional::bce_with_logits,
        py::arg("input"), py::arg("target"), py::arg("reduction") = "mean");
    functional_mod.def("normalize", &functional::normalize,
        py::arg("x"), py::arg("p") = 2.0, py::arg("dim") = 1, py::arg("eps") = 1e-12);
    functional_mod.def("glu", &functional::glu,
        py::arg("x"), py::arg("dim") = -1);
    functional_mod.def("elu", &functional::elu,
        py::arg("x"), py::arg("alpha") = 1.0);
    // === end W1-T5 functional ===

    auto optim_mod = m.def_submodule("optim", "LiteTorch Optimizers");
    
    py::class_<optim::Optimizer, std::shared_ptr<optim::Optimizer>>(optim_mod, "Optimizer")
        .def("step", &optim::Optimizer::step)
        .def("zero_grad", &optim::Optimizer::zero_grad)
        .def("get_lr", &optim::Optimizer::get_lr)
        .def("set_lr", &optim::Optimizer::set_lr, py::arg("new_lr"));

    py::class_<optim::SGD, optim::Optimizer, std::shared_ptr<optim::SGD>>(optim_mod, "SGD")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float, bool>(), py::arg("params"), py::arg("lr") = 0.01, py::arg("momentum") = 0.0f, py::arg("dampening") = 0.0f, py::arg("weight_decay") = 0.0f, py::arg("nesterov") = false)
        .def("step", &optim::SGD::step)
        .def("zero_grad", &optim::SGD::zero_grad)
        .def("get_lr", &optim::SGD::get_lr)
        .def("set_lr", &optim::SGD::set_lr, py::arg("new_lr"))
        .def_readwrite("lr", &optim::SGD::lr)
        .def_readwrite("momentum", &optim::SGD::momentum)
        .def_readwrite("dampening", &optim::SGD::dampening)
        .def_readwrite("nesterov", &optim::SGD::nesterov)
        .def_readwrite("weight_decay", &optim::SGD::weight_decay);

    py::class_<optim::Adam, optim::Optimizer, std::shared_ptr<optim::Adam>>(optim_mod, "Adam")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.001, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def("step", &optim::Adam::step)
        .def("zero_grad", &optim::Adam::zero_grad)
        .def("get_lr", &optim::Adam::get_lr)
        .def("set_lr", &optim::Adam::set_lr, py::arg("new_lr"))
        .def_readwrite("lr", &optim::Adam::lr)
        .def_readwrite("beta1", &optim::Adam::beta1)
        .def_readwrite("beta2", &optim::Adam::beta2)
        .def_readwrite("eps", &optim::Adam::eps)
        .def_readwrite("weight_decay", &optim::Adam::weight_decay);

    py::class_<optim::AdamW, optim::Optimizer, std::shared_ptr<optim::AdamW>>(optim_mod, "AdamW")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float, float, bool>(), py::arg("params"), py::arg("lr") = 0.001, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.01f, py::arg("offload_to_cpu") = false)
        .def("step", &optim::AdamW::step)
        .def("zero_grad", &optim::AdamW::zero_grad)
        .def("get_lr", &optim::AdamW::get_lr)
        .def("set_lr", &optim::AdamW::set_lr, py::arg("new_lr"))
        .def_readwrite("lr", &optim::AdamW::lr)
        .def_readwrite("beta1", &optim::AdamW::beta1)
        .def_readwrite("beta2", &optim::AdamW::beta2)
        .def_readwrite("eps", &optim::AdamW::eps)
        .def_readwrite("weight_decay", &optim::AdamW::weight_decay);

    py::class_<optim::AdamW8bit, optim::Optimizer, std::shared_ptr<optim::AdamW8bit>>(optim_mod, "AdamW8bit")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.001, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def("step", &optim::AdamW8bit::step)
        .def("zero_grad", &optim::AdamW8bit::zero_grad)
        .def("get_lr", &optim::AdamW8bit::get_lr)
        .def("set_lr", &optim::AdamW8bit::set_lr, py::arg("new_lr"))
        .def_readwrite("lr", &optim::AdamW8bit::lr)
        .def_readwrite("weight_decay", &optim::AdamW8bit::weight_decay);

    py::class_<optim::AdamWFP8, optim::Optimizer, std::shared_ptr<optim::AdamWFP8>>(optim_mod, "AdamWFP8")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.001, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def("step", &optim::AdamWFP8::step)
        .def("zero_grad", &optim::AdamWFP8::zero_grad)
        .def("get_lr", &optim::AdamWFP8::get_lr)
        .def("set_lr", &optim::AdamWFP8::set_lr, py::arg("new_lr"))
        .def_readwrite("lr", &optim::AdamWFP8::lr)
        .def_readwrite("weight_decay", &optim::AdamWFP8::weight_decay);

    py::class_<optim::RMSprop, optim::Optimizer, std::shared_ptr<optim::RMSprop>>(optim_mod, "RMSprop")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float>(), py::arg("params"), py::arg("lr") = 0.01, py::arg("alpha") = 0.99f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def("step", &optim::RMSprop::step)
        .def("zero_grad", &optim::RMSprop::zero_grad)
        .def("get_lr", &optim::RMSprop::get_lr)
        .def("set_lr", &optim::RMSprop::set_lr, py::arg("new_lr"))
        .def_readwrite("lr", &optim::RMSprop::lr)
        .def_readwrite("weight_decay", &optim::RMSprop::weight_decay);

    py::class_<optim::ZeRO3Optimizer, optim::Optimizer, std::shared_ptr<optim::ZeRO3Optimizer>>(optim_mod, "ZeRO3Optimizer")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.001, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def("step", &optim::ZeRO3Optimizer::step)
        .def("zero_grad", &optim::ZeRO3Optimizer::zero_grad)
        .def_readwrite("lr", &optim::ZeRO3Optimizer::lr)
        .def_readwrite("beta1", &optim::ZeRO3Optimizer::beta1)
        .def_readwrite("beta2", &optim::ZeRO3Optimizer::beta2)
        .def_readwrite("eps", &optim::ZeRO3Optimizer::eps)
        .def_readwrite("weight_decay", &optim::ZeRO3Optimizer::weight_decay);

    py::class_<optim::StepLR>(optim_mod, "StepLR")
        .def(py::init<optim::Optimizer*, int, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("step_size"), py::arg("gamma") = 0.1f)
        .def("step", &optim::StepLR::step)
        .def_readwrite("step_size", &optim::StepLR::step_size)
        .def_readwrite("gamma", &optim::StepLR::gamma)
        .def_readwrite("last_epoch", &optim::StepLR::last_epoch);

    py::class_<optim::CosineAnnealingLR>(optim_mod, "CosineAnnealingLR")
        .def(py::init<optim::Optimizer*, int, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("T_max"), py::arg("eta_min") = 0.0f)
        .def("step", &optim::CosineAnnealingLR::step)
        .def_readwrite("T_max", &optim::CosineAnnealingLR::T_max)
        .def_readwrite("eta_min", &optim::CosineAnnealingLR::eta_min)
        .def_readwrite("last_epoch", &optim::CosineAnnealingLR::last_epoch);

    auto lr_sched_mod = optim_mod.def_submodule("lr_scheduler", "Learning rate schedulers");
    lr_sched_mod.def("StepLR", [](optim::Optimizer* opt, int step_size, float gamma) {
        return std::unique_ptr<optim::StepLR>(new optim::StepLR(opt, step_size, gamma));
    }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("step_size"), py::arg("gamma") = 0.1f);
    lr_sched_mod.def("CosineAnnealingLR", [](optim::Optimizer* opt, int T_max, float eta_min) {
        return std::unique_ptr<optim::CosineAnnealingLR>(new optim::CosineAnnealingLR(opt, T_max, eta_min));
    }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("T_max"), py::arg("eta_min") = 0.0f);
    py::class_<optim::LinearWarmup>(lr_sched_mod, "LinearWarmup")
        .def(py::init<optim::Optimizer*, int, int, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("warmup_steps"), py::arg("total_steps"), py::arg("min_lr") = 0.0f)
        .def("step", &optim::LinearWarmup::step)
        .def("get_lr", &optim::LinearWarmup::get_lr)
        .def_readwrite("warmup_steps", &optim::LinearWarmup::warmup_steps)
        .def_readwrite("total_steps", &optim::LinearWarmup::total_steps)
        .def_readwrite("last_step", &optim::LinearWarmup::last_step);

    py::class_<optim::Adagrad, optim::Optimizer, std::shared_ptr<optim::Adagrad>>(optim_mod, "Adagrad")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float>(), py::arg("params"), py::arg("lr") = 0.01, py::arg("lr_decay") = 0.0f, py::arg("weight_decay") = 0.0f, py::arg("eps") = 1e-10f)
        .def("step", &optim::Adagrad::step)
        .def("zero_grad", &optim::Adagrad::zero_grad)
        .def("get_lr", &optim::Adagrad::get_lr)
        .def("set_lr", &optim::Adagrad::set_lr, py::arg("new_lr"))
        .def("add_param_group", &optim::Adagrad::add_param_group, py::arg("params"), py::arg("lr"), py::arg("weight_decay") = 0.0f)
        .def("param_groups", [](optim::Adagrad& o) {
            py::list groups;
            for (auto& g : o.param_groups) {
                py::dict gd;
                py::list ps;
                for (auto& p : g.params) ps.append(py::cast(p));
                gd["params"] = ps;
                gd["lr"] = g.lr;
                gd["weight_decay"] = g.weight_decay;
                groups.append(gd);
            }
            return groups;
        })
        .def("state_dict", [](optim::Adagrad& o) {
            py::dict d;
            d["lr"] = o.lr;
            d["lr_decay"] = o.lr_decay;
            d["weight_decay"] = o.weight_decay;
            d["eps"] = o.eps;
            py::list states;
            size_t idx = 0;
            for (auto& g : o.param_groups) {
                for (size_t i = 0; i < g.params.size(); ++i, ++idx) {
                    py::dict s;
                    s["step"] = o.step_count;
                    s["sum"] = py::cast(o.sum[idx]->clone());
                    states.append(s);
                }
            }
            d["state"] = states;
            py::list groups;
            for (auto& g : o.param_groups) {
                py::dict gd;
                gd["lr"] = g.lr;
                gd["weight_decay"] = g.weight_decay;
                gd["num_params"] = static_cast<int>(g.params.size());
                groups.append(gd);
            }
            d["param_groups"] = groups;
            return d;
        })
        .def("load_state_dict", [](optim::Adagrad& o, py::dict d) {
            py::list groups = d["param_groups"];
            if (groups.size() != static_cast<py::ssize_t>(o.param_groups.size()))
                throw std::runtime_error("[litetorch Error] Adagrad.load_state_dict: param_groups size mismatch");
            for (py::ssize_t gi = 0; gi < groups.size(); ++gi) {
                py::dict gd = groups[gi];
                o.param_groups[gi].lr = gd["lr"].cast<double>();
                o.param_groups[gi].weight_decay = gd["weight_decay"].cast<float>();
            }
            o.lr = o.param_groups[0].lr;
            o.weight_decay = o.param_groups[0].weight_decay;
            if (d.contains("lr_decay")) o.lr_decay = d["lr_decay"].cast<float>();
            if (d.contains("eps")) o.eps = d["eps"].cast<float>();
            py::list states = d["state"];
            if (states.size() != static_cast<py::ssize_t>(o.sum.size()))
                throw std::runtime_error("[litetorch Error] Adagrad.load_state_dict: state size mismatch");
            for (py::ssize_t i = 0; i < states.size(); ++i) {
                py::dict s = states[i];
                o.step_count = s["step"].cast<int>();
                auto src = py::cast<std::shared_ptr<Tensor>>(s["sum"]);
                auto dst = o.sum[i];
                if (src->numel() != dst->numel())
                    throw std::runtime_error("[litetorch Error] Adagrad.load_state_dict: state tensor size mismatch");
                std::memcpy(dst->data_ptr(), src->data_ptr(), static_cast<size_t>(src->numel()) * sizeof(float));
            }
        }, py::arg("state_dict"))
        .def_readwrite("lr", &optim::Adagrad::lr)
        .def_readwrite("lr_decay", &optim::Adagrad::lr_decay)
        .def_readwrite("weight_decay", &optim::Adagrad::weight_decay)
        .def_readwrite("eps", &optim::Adagrad::eps);

    py::class_<optim::NAdam, optim::Optimizer, std::shared_ptr<optim::NAdam>>(optim_mod, "NAdam")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, std::pair<double, double>, float, float, float>(), py::arg("params"), py::arg("lr") = 0.002, py::arg("betas") = std::pair<double, double>(0.9, 0.999), py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f, py::arg("momentum_decay") = 0.004f)
        .def("step", &optim::NAdam::step)
        .def("zero_grad", &optim::NAdam::zero_grad)
        .def("get_lr", &optim::NAdam::get_lr)
        .def("set_lr", &optim::NAdam::set_lr, py::arg("new_lr"))
        .def("add_param_group", &optim::NAdam::add_param_group, py::arg("params"), py::arg("lr"), py::arg("weight_decay") = 0.0f)
        .def("param_groups", [](optim::NAdam& o) {
            py::list groups;
            for (auto& g : o.param_groups) {
                py::dict gd;
                py::list ps;
                for (auto& p : g.params) ps.append(py::cast(p));
                gd["params"] = ps;
                gd["lr"] = g.lr;
                gd["weight_decay"] = g.weight_decay;
                groups.append(gd);
            }
            return groups;
        })
        .def("state_dict", [](optim::NAdam& o) {
            py::dict d;
            d["lr"] = o.lr;
            d["betas"] = py::make_tuple(o.beta1, o.beta2);
            d["eps"] = o.eps;
            d["weight_decay"] = o.weight_decay;
            d["momentum_decay"] = o.momentum_decay;
            py::list states;
            size_t idx = 0;
            for (auto& g : o.param_groups) {
                for (size_t i = 0; i < g.params.size(); ++i, ++idx) {
                    py::dict s;
                    s["step"] = o.step_count;
                    s["exp_avg"] = py::cast(o.exp_avg[idx]->clone());
                    s["exp_avg_sq"] = py::cast(o.exp_avg_sq[idx]->clone());
                    states.append(s);
                }
            }
            d["state"] = states;
            py::list groups;
            for (auto& g : o.param_groups) {
                py::dict gd;
                gd["lr"] = g.lr;
                gd["weight_decay"] = g.weight_decay;
                gd["num_params"] = static_cast<int>(g.params.size());
                groups.append(gd);
            }
            d["param_groups"] = groups;
            return d;
        })
        .def("load_state_dict", [](optim::NAdam& o, py::dict d) {
            py::list groups = d["param_groups"];
            if (groups.size() != static_cast<py::ssize_t>(o.param_groups.size()))
                throw std::runtime_error("[litetorch Error] NAdam.load_state_dict: param_groups size mismatch");
            for (py::ssize_t gi = 0; gi < groups.size(); ++gi) {
                py::dict gd = groups[gi];
                o.param_groups[gi].lr = gd["lr"].cast<double>();
                o.param_groups[gi].weight_decay = gd["weight_decay"].cast<float>();
            }
            o.lr = o.param_groups[0].lr;
            o.weight_decay = o.param_groups[0].weight_decay;
            if (d.contains("betas")) {
                auto b = d["betas"].cast<std::pair<double, double>>();
                o.beta1 = b.first;
                o.beta2 = b.second;
            }
            if (d.contains("eps")) o.eps = d["eps"].cast<float>();
            if (d.contains("momentum_decay")) o.momentum_decay = d["momentum_decay"].cast<float>();
            py::list states = d["state"];
            if (states.size() != static_cast<py::ssize_t>(o.exp_avg.size()))
                throw std::runtime_error("[litetorch Error] NAdam.load_state_dict: state size mismatch");
            for (py::ssize_t i = 0; i < states.size(); ++i) {
                py::dict s = states[i];
                o.step_count = s["step"].cast<int>();
                auto m_src = py::cast<std::shared_ptr<Tensor>>(s["exp_avg"]);
                auto v_src = py::cast<std::shared_ptr<Tensor>>(s["exp_avg_sq"]);
                auto m_dst = o.exp_avg[i];
                auto v_dst = o.exp_avg_sq[i];
                if (m_src->numel() != m_dst->numel() || v_src->numel() != v_dst->numel())
                    throw std::runtime_error("[litetorch Error] NAdam.load_state_dict: state tensor size mismatch");
                std::memcpy(m_dst->data_ptr(), m_src->data_ptr(), static_cast<size_t>(m_src->numel()) * sizeof(float));
                std::memcpy(v_dst->data_ptr(), v_src->data_ptr(), static_cast<size_t>(v_src->numel()) * sizeof(float));
            }
        }, py::arg("state_dict"))
        .def_readwrite("lr", &optim::NAdam::lr)
        .def_readwrite("beta1", &optim::NAdam::beta1)
        .def_readwrite("beta2", &optim::NAdam::beta2)
        .def_readwrite("eps", &optim::NAdam::eps)
        .def_readwrite("weight_decay", &optim::NAdam::weight_decay)
        .def_readwrite("momentum_decay", &optim::NAdam::momentum_decay);

    py::class_<optim::MultiStepLR>(optim_mod, "MultiStepLR")
        .def(py::init<optim::Optimizer*, std::vector<int>, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("milestones"), py::arg("gamma") = 0.1f)
        .def("step", &optim::MultiStepLR::step)
        .def("get_last_lr", &optim::MultiStepLR::get_last_lr)
        .def("state_dict", [](optim::MultiStepLR& s) {
            py::dict d;
            for (auto& kv : s.state_dict()) d[kv.first.c_str()] = kv.second;
            py::list ms;
            for (int m : s.milestones) ms.append(m);
            d["milestones"] = ms;
            return d;
        })
        .def("load_state_dict", [](optim::MultiStepLR& s, py::dict d) {
            std::map<std::string, double> m;
            for (auto item : d) {
                std::string k = py::cast<std::string>(item.first);
                if (k == "milestones") continue;
                m[k] = py::cast<double>(item.second);
            }
            s.load_state_dict(m);
            if (d.contains("milestones")) {
                s.milestones.clear();
                for (auto v : d["milestones"]) s.milestones.push_back(py::cast<int>(v));
            }
        }, py::arg("state_dict"))
        .def_readwrite("gamma", &optim::MultiStepLR::gamma)
        .def_readwrite("last_epoch", &optim::MultiStepLR::last_epoch);

    py::class_<optim::ExponentialLR>(optim_mod, "ExponentialLR")
        .def(py::init<optim::Optimizer*, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("gamma"))
        .def("step", &optim::ExponentialLR::step)
        .def("get_last_lr", &optim::ExponentialLR::get_last_lr)
        .def("state_dict", [](optim::ExponentialLR& s) {
            py::dict d;
            for (auto& kv : s.state_dict()) d[kv.first.c_str()] = kv.second;
            return d;
        })
        .def("load_state_dict", [](optim::ExponentialLR& s, py::dict d) {
            std::map<std::string, double> m;
            for (auto item : d) {
                m[py::cast<std::string>(item.first)] = py::cast<double>(item.second);
            }
            s.load_state_dict(m);
        }, py::arg("state_dict"))
        .def_readwrite("gamma", &optim::ExponentialLR::gamma)
        .def_readwrite("last_epoch", &optim::ExponentialLR::last_epoch);

    py::class_<optim::ReduceLROnPlateau>(optim_mod, "ReduceLROnPlateau")
        .def(py::init<optim::Optimizer*, std::string, float, int, float, std::string, int, float, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("mode") = "min", py::arg("factor") = 0.1f, py::arg("patience") = 10, py::arg("threshold") = 1e-4f, py::arg("threshold_mode") = "rel", py::arg("cooldown") = 0, py::arg("min_lr") = 0.0f, py::arg("eps") = 1e-8f)
        .def("step", &optim::ReduceLROnPlateau::step, py::arg("metric"))
        .def("get_last_lr", &optim::ReduceLROnPlateau::get_last_lr)
        .def("state_dict", [](optim::ReduceLROnPlateau& s) {
            py::dict d;
            for (auto& kv : s.state_dict()) d[kv.first.c_str()] = kv.second;
            d["mode"] = s.mode;
            d["threshold_mode"] = s.threshold_mode;
            return d;
        })
        .def("load_state_dict", [](optim::ReduceLROnPlateau& s, py::dict d) {
            std::map<std::string, double> m;
            for (auto item : d) {
                std::string k = py::cast<std::string>(item.first);
                if (k == "mode" || k == "threshold_mode") continue;
                m[k] = py::cast<double>(item.second);
            }
            s.load_state_dict(m);
            if (d.contains("mode")) s.mode = py::cast<std::string>(d["mode"]);
            if (d.contains("threshold_mode")) s.threshold_mode = py::cast<std::string>(d["threshold_mode"]);
        }, py::arg("state_dict"))
        .def_readwrite("patience", &optim::ReduceLROnPlateau::patience)
        .def_readwrite("last_epoch", &optim::ReduceLROnPlateau::last_epoch)
        .def_readwrite("mode", &optim::ReduceLROnPlateau::mode)
        .def_readwrite("best", &optim::ReduceLROnPlateau::best);

    lr_sched_mod.def("MultiStepLR", [](optim::Optimizer* opt, std::vector<int> milestones, float gamma) {
        return std::unique_ptr<optim::MultiStepLR>(new optim::MultiStepLR(opt, milestones, gamma));
    }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("milestones"), py::arg("gamma") = 0.1f);
    lr_sched_mod.def("ExponentialLR", [](optim::Optimizer* opt, float gamma) {
        return std::unique_ptr<optim::ExponentialLR>(new optim::ExponentialLR(opt, gamma));
    }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("gamma"));
    lr_sched_mod.def("ReduceLROnPlateau", [](optim::Optimizer* opt, std::string mode, float factor, int patience, float threshold, std::string threshold_mode, int cooldown, float min_lr, float eps) {
        return std::unique_ptr<optim::ReduceLROnPlateau>(new optim::ReduceLROnPlateau(opt, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps));
    }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("mode") = "min", py::arg("factor") = 0.1f, py::arg("patience") = 10, py::arg("threshold") = 1e-4f, py::arg("threshold_mode") = "rel", py::arg("cooldown") = 0, py::arg("min_lr") = 0.0f, py::arg("eps") = 1e-8f);

    py::class_<optim::GradScaler>(optim_mod, "GradScaler")
        .def(py::init<float, float, float, int, bool>(), py::arg("init_scale") = 65536.0f, py::arg("growth_factor") = 2.0f, py::arg("backoff_factor") = 0.5f, py::arg("growth_interval") = 2000, py::arg("enabled") = true)
        .def("scale", &optim::GradScaler::scale, py::arg("loss"))
        .def("scale_loss", &optim::GradScaler::scale_loss, py::arg("loss"))
        .def("unscale_", &optim::GradScaler::unscale_, py::arg("optimizer"))
        .def("step", &optim::GradScaler::step, py::arg("optimizer"))
        .def("update", &optim::GradScaler::update)
        .def("get_scale", &optim::GradScaler::get_scale)
        .def("set_scale", &optim::GradScaler::set_scale, py::arg("scale"))
        .def("is_enabled", &optim::GradScaler::is_enabled);

    auto amp_mod = m.def_submodule("amp", "LiteTorch Automatic Mixed Precision");
    
    py::class_<amp::GradScaler, optim::GradScaler>(amp_mod, "GradScaler")
        .def(py::init<float, float, float, int, bool>(), py::arg("init_scale") = 65536.0f, py::arg("growth_factor") = 2.0f, py::arg("backoff_factor") = 0.5f, py::arg("growth_interval") = 2000, py::arg("enabled") = true)
        .def("scale", &amp::GradScaler::scale, py::arg("loss"))
        .def("unscale_", &amp::GradScaler::unscale_, py::arg("optimizer"))
        .def("step", &amp::GradScaler::step, py::arg("optimizer"))
        .def("update", &amp::GradScaler::update)
        .def("get_scale", &amp::GradScaler::get_scale)
        .def("set_scale", &amp::GradScaler::set_scale, py::arg("scale"))
        .def("is_enabled", &amp::GradScaler::is_enabled);

    py::class_<amp::AutocastGuard>(amp_mod, "AutocastGuard")
        .def(py::init<bool, DataType>(), py::arg("enabled") = true, py::arg("dtype") = DataType::FP16)
        .def_static("is_enabled", &amp::AutocastGuard::is_enabled)
        .def_static("get_dtype", &amp::AutocastGuard::get_dtype);

    py::class_<PyAutocast>(amp_mod, "autocast")
        .def(py::init([](const std::string& device_type, py::object dtype_obj, bool enabled) {
            PyAutocast s;
            s.device_type = device_type;
            s.dtype = parse_autocast_dtype(dtype_obj, device_type);
            s.enabled = enabled;
            return s;
        }), py::arg("device_type") = "cuda", py::arg("dtype") = py::none(), py::arg("enabled") = true)
        .def("__enter__", [](PyAutocast& self) -> PyAutocast& { self.enter(); return self; })
        .def("__exit__", [](PyAutocast& self, py::object, py::object, py::object) { self.exit(); })
        .def_property_readonly("device_type", [](const PyAutocast& self) { return self.device_type; })
        .def_property_readonly("dtype", [](const PyAutocast& self) { return self.dtype; })
        .def_property_readonly("enabled", [](const PyAutocast& self) { return self.enabled; });

    amp_mod.def("is_autocast_enabled", &amp::AutocastGuard::is_enabled);
    amp_mod.def("get_autocast_dtype", &amp::AutocastGuard::get_dtype);

    auto quant_mod = m.def_submodule("quantization", "LiteTorch Quantization");
    quant_mod.def("quantize_linear", &quantization::quantize_linear, py::arg("layer"));

    py::class_<quantization::Calibrator>(quant_mod, "Calibrator")
        .def(py::init<>())
        .def("collect", &quantization::Calibrator::collect, py::arg("name"), py::arg("tensor"))
        .def("get_asymmetric_params", &quantization::Calibrator::get_asymmetric_params, py::arg("name"), py::arg("bits") = 8)
        .def("get_scale", &quantization::Calibrator::get_scale, py::arg("name"), py::arg("bits") = 8);

    py::class_<quantization::QATLinear, nn::Module, std::shared_ptr<quantization::QATLinear>>(quant_mod, "QATLinear")
        .def(py::init<std::shared_ptr<nn::Linear>, float, float, float, float, int>(),
             py::arg("linear"), py::arg("w_scale") = 0.01f, py::arg("w_zp") = 0.0f, py::arg("a_scale") = 0.01f, py::arg("a_zp") = 0.0f, py::arg("b") = 8)
        .def("forward", &quantization::QATLinear::forward)
        .def("parameters", &quantization::QATLinear::parameters)
        .def("to", &quantization::QATLinear::to)
        .def_readwrite("weight", &quantization::QATLinear::weight)
        .def_readwrite("bias", &quantization::QATLinear::bias)
        .def_readwrite("weight_scale", &quantization::QATLinear::weight_scale)
        .def_readwrite("weight_zero_point", &quantization::QATLinear::weight_zero_point)
        .def_readwrite("act_scale", &quantization::QATLinear::act_scale)
        .def_readwrite("act_zero_point", &quantization::QATLinear::act_zero_point)
        .def_readwrite("bits", &quantization::QATLinear::bits)
        .def_readwrite("qat_enabled", &quantization::QATLinear::qat_enabled);

    auto dist_mod = m.def_submodule("distributed", "LiteTorch Distributed Modules");

    py::class_<distributed::ProcessGroup>(dist_mod, "ProcessGroup")
        .def_static("get", &distributed::ProcessGroup::get, py::return_value_policy::reference)
        .def("init", &distributed::ProcessGroup::init, py::arg("rank"), py::arg("world_size"), py::arg("master_addr"), py::arg("master_port"))
        .def("shutdown", &distributed::ProcessGroup::shutdown)
        .def("all_reduce", py::overload_cast<std::shared_ptr<Tensor>>(&distributed::ProcessGroup::all_reduce), py::arg("tensor"))
        .def("all_reduce", py::overload_cast<std::shared_ptr<Tensor>, distributed::ReduceOp>(&distributed::ProcessGroup::all_reduce), py::arg("tensor"), py::arg("op"))
        .def("barrier", &distributed::ProcessGroup::barrier)
        .def("all_to_all", &distributed::ProcessGroup::all_to_all, py::arg("input"), py::arg("output"))
        .def("scatter", &distributed::ProcessGroup::scatter, py::arg("output"), py::arg("input"), py::arg("src"))
        .def("gather", &distributed::ProcessGroup::gather, py::arg("input"), py::arg("output"), py::arg("dst"))
        .def("broadcast", &distributed::ProcessGroup::broadcast, py::arg("tensor"), py::arg("src"))
        .def("all_gather", &distributed::ProcessGroup::all_gather, py::arg("shard"), py::arg("full"))
        .def("reduce_scatter", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&distributed::ProcessGroup::reduce_scatter), py::arg("shard"), py::arg("full"))
        .def("reduce_scatter", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, distributed::ReduceOp>(&distributed::ProcessGroup::reduce_scatter), py::arg("shard"), py::arg("full"), py::arg("op"))
        .def("send_tensor", &distributed::ProcessGroup::send_tensor, py::arg("tensor"), py::arg("dst"))
        .def("recv_tensor", &distributed::ProcessGroup::recv_tensor, py::arg("tensor"), py::arg("src"))
        .def("sync_comm", &distributed::ProcessGroup::sync_comm)
        .def("get_rank", &distributed::ProcessGroup::get_rank)
        .def("get_world_size", &distributed::ProcessGroup::get_world_size)
        .def("is_initialized", &distributed::ProcessGroup::is_initialized);

    py::class_<distributed::DeviceMesh, std::shared_ptr<distributed::DeviceMesh>>(dist_mod, "DeviceMesh")
        .def(py::init<const std::vector<int>&, const std::string&>(), py::arg("shape"), py::arg("dim_names"))
        .def("get_rank", &distributed::DeviceMesh::get_rank, py::arg("coords"))
        .def_readwrite("mesh_shape", &distributed::DeviceMesh::mesh_shape)
        .def_readwrite("mesh_dim_names", &distributed::DeviceMesh::mesh_dim_names)
        .def_readwrite("mesh_topology", &distributed::DeviceMesh::mesh_topology);

    py::enum_<distributed::PlacementType>(dist_mod, "PlacementType")
        .value("SHARD", distributed::PlacementType::SHARD)
        .value("REPLICATE", distributed::PlacementType::REPLICATE)
        .value("PARTIAL", distributed::PlacementType::PARTIAL)
        .export_values();

    py::class_<distributed::Placement>(dist_mod, "Placement")
        .def(py::init<>())
        .def_readwrite("type", &distributed::Placement::type)
        .def_readwrite("dim", &distributed::Placement::dim);

    py::class_<distributed::DTensor, std::shared_ptr<distributed::DTensor>>(dist_mod, "DTensor")
        .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<distributed::DeviceMesh>, const std::vector<distributed::Placement>&>(),
             py::arg("local_tensor"), py::arg("mesh"), py::arg("placements"))
        .def("redistribute", &distributed::DTensor::redistribute, py::arg("new_mesh"), py::arg("new_placements"))
        .def_readwrite("device_mesh", &distributed::DTensor::device_mesh)
        .def_readwrite("placements", &distributed::DTensor::placements)
        .def_readwrite("local_tensor", &distributed::DTensor::local_tensor);

    py::class_<distributed::FullyShardedDataParallel, std::shared_ptr<distributed::FullyShardedDataParallel>>(dist_mod, "FullyShardedDataParallel")
        .def(py::init<std::shared_ptr<nn::Module>>(), py::arg("module"))
        .def("gather_parameters", &distributed::FullyShardedDataParallel::gather_parameters)
        .def("shard_parameters", &distributed::FullyShardedDataParallel::shard_parameters)
        .def("get_module", &distributed::FullyShardedDataParallel::get_module);

    py::class_<distributed::OverlappedAllReducer>(dist_mod, "OverlappedAllReducer")
        .def_static("get", &distributed::OverlappedAllReducer::get, py::return_value_policy::reference)
        .def("push_and_all_reduce", &distributed::OverlappedAllReducer::push_and_all_reduce, py::arg("tensor"))
        .def("push_and_reduce_scatter", &distributed::OverlappedAllReducer::push_and_reduce_scatter, py::arg("shard"), py::arg("full"))
        .def("sync", &distributed::OverlappedAllReducer::sync);

    dist_mod.def("overlapped_all_reduce", &distributed::overlapped_all_reduce, py::arg("tensor"));
    dist_mod.def("overlapped_reduce_scatter", &distributed::overlapped_reduce_scatter, py::arg("shard"), py::arg("full"));
    dist_mod.def("sync_overlapped_all_reduce", &distributed::sync_overlapped_all_reduce);

    py::class_<distributed::PipelineParallelModule, nn::Module, std::shared_ptr<distributed::PipelineParallelModule>>(dist_mod, "PipelineParallelModule")
        .def(py::init<std::shared_ptr<nn::Module>>(), py::arg("sub_module"))
        .def("forward", &distributed::PipelineParallelModule::forward, py::arg("input"))
        .def("forward_microbatches", &distributed::PipelineParallelModule::forward_microbatches, py::arg("microbatches"))
        .def("schedule_1f1b", &distributed::PipelineParallelModule::schedule_1f1b, py::arg("microbatches"))
        .def("train_step_1f1b", &distributed::PipelineParallelModule::train_step_1f1b, py::arg("microbatches"), py::arg("loss_fn") = nullptr)
        .def("run_1f1b_with_backward", &distributed::PipelineParallelModule::run_1f1b_with_backward, py::arg("microbatches"), py::arg("loss_fn"))
        .def("parameters", &distributed::PipelineParallelModule::parameters)
        .def("to", &distributed::PipelineParallelModule::to, py::arg("device"));

    py::enum_<distributed::ReduceOp>(dist_mod, "ReduceOp")
        .value("SUM", distributed::ReduceOp::SUM)
        .value("AVG", distributed::ReduceOp::AVG)
        .value("PRODUCT", distributed::ReduceOp::PRODUCT)
        .value("MIN", distributed::ReduceOp::MIN)
        .value("MAX", distributed::ReduceOp::MAX)
        .export_values();

    dist_mod.def("init_process_group", &distributed::init_process_group,
                 py::arg("rank"), py::arg("world_size"),
                 py::arg("master_addr") = "127.0.0.1", py::arg("master_port") = 29500);
    dist_mod.def("shutdown", &distributed::shutdown);
    dist_mod.def("barrier", &distributed::barrier);
    dist_mod.def("all_reduce",
                 py::overload_cast<std::shared_ptr<Tensor>>(&distributed::all_reduce),
                 py::arg("tensor"));
    dist_mod.def("all_reduce",
                 py::overload_cast<std::shared_ptr<Tensor>, distributed::ReduceOp>(&distributed::all_reduce),
                 py::arg("tensor"), py::arg("op") = distributed::ReduceOp::SUM);
    dist_mod.def("all_gather", &distributed::all_gather, py::arg("shard"), py::arg("full"));
    dist_mod.def("reduce_scatter",
                 py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&distributed::reduce_scatter),
                 py::arg("shard"), py::arg("full"));
    dist_mod.def("reduce_scatter",
                 py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, distributed::ReduceOp>(&distributed::reduce_scatter),
                 py::arg("shard"), py::arg("full"), py::arg("op"));
    dist_mod.def("broadcast", &distributed::broadcast, py::arg("tensor"), py::arg("src_rank"));
    dist_mod.def("all_to_all", &distributed::all_to_all, py::arg("input"), py::arg("output"));
    dist_mod.def("scatter", &distributed::scatter, py::arg("output"), py::arg("input"), py::arg("src"));
    dist_mod.def("gather", &distributed::gather, py::arg("input"), py::arg("output"), py::arg("dst"));
    dist_mod.def("send_tensor", &distributed::send_tensor, py::arg("tensor"), py::arg("dst"));
    dist_mod.def("recv_tensor", &distributed::recv_tensor, py::arg("tensor"), py::arg("src"));

    py::class_<distributed::NCCLBridge, std::unique_ptr<distributed::NCCLBridge, py::nodelete>>(dist_mod, "NCCLBridge")
        .def_static("get", &distributed::NCCLBridge::get, py::return_value_policy::reference)
        .def("is_available", &distributed::NCCLBridge::is_available)
        .def("init", &distributed::NCCLBridge::init, py::arg("rank"), py::arg("world_size"))
        .def("shutdown", &distributed::NCCLBridge::shutdown)
        .def("sync_comm", &distributed::NCCLBridge::sync_comm);

    py::class_<distributed::DistributedDataParallel, nn::Module, std::shared_ptr<distributed::DistributedDataParallel>>(dist_mod, "DistributedDataParallel")
        .def(py::init<std::shared_ptr<nn::Module>, distributed::ReduceOp>(), py::arg("module"), py::arg("op") = distributed::ReduceOp::AVG)
        .def("forward", &distributed::DistributedDataParallel::forward, py::arg("input"))
        .def("backward", py::overload_cast<std::shared_ptr<Tensor>>(&distributed::DistributedDataParallel::backward), py::arg("loss"))
        .def("backward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&distributed::DistributedDataParallel::backward), py::arg("loss"), py::arg("grad_output"))
        .def("sync_grads", &distributed::DistributedDataParallel::sync_grads)
        .def("parameters", &distributed::DistributedDataParallel::parameters)
        .def("to", &distributed::DistributedDataParallel::to, py::arg("device"))
        .def_readwrite("reduce_op", &distributed::DistributedDataParallel::reduce_op);

    auto jit_mod = m.def_submodule("jit", "LiteTorch JIT Compilation");

    py::enum_<JITVar::OpType>(jit_mod, "OpType")
        .value("INPUT", JITVar::OpType::INPUT)
        .value("CONST", JITVar::OpType::CONST)
        .value("ADD", JITVar::OpType::ADD)
        .value("SUB", JITVar::OpType::SUB)
        .value("MUL", JITVar::OpType::MUL)
        .value("DIV", JITVar::OpType::DIV)
        .value("RELU", JITVar::OpType::RELU)
        .value("GELU", JITVar::OpType::GELU)
        .value("SIGMOID", JITVar::OpType::SIGMOID)
        .value("TANH", JITVar::OpType::TANH)
        .value("SQRT", JITVar::OpType::SQRT)
        .value("EXP", JITVar::OpType::EXP)
        .value("LOG", JITVar::OpType::LOG)
        .value("ABS", JITVar::OpType::ABS)
        .value("NEG", JITVar::OpType::NEG)
        .export_values();

    py::class_<JITVar, std::shared_ptr<JITVar>>(jit_mod, "JITVar")
        .def(py::init<JITVar::OpType, const std::string&>(), py::arg("op"), py::arg("name") = "")
        .def(py::init<float>(), py::arg("val"))
        .def_readwrite("op", &JITVar::op)
        .def_readwrite("name", &JITVar::name)
        .def_readwrite("val", &JITVar::val)
        .def("__add__", [](std::shared_ptr<JITVar> a, std::shared_ptr<JITVar> b) { return a + b; })
        .def("__add__", [](std::shared_ptr<JITVar> a, float b) { return a + b; })
        .def("__radd__", [](std::shared_ptr<JITVar> a, float b) { return a + b; })
        .def("__sub__", [](std::shared_ptr<JITVar> a, std::shared_ptr<JITVar> b) { return a - b; })
        .def("__sub__", [](std::shared_ptr<JITVar> a, float b) { return a - b; })
        .def("__rsub__", [](std::shared_ptr<JITVar> a, float b) { return std::make_shared<JITVar>(b) - a; })
        .def("__mul__", [](std::shared_ptr<JITVar> a, std::shared_ptr<JITVar> b) { return a * b; })
        .def("__mul__", [](std::shared_ptr<JITVar> a, float b) { return a * b; })
        .def("__rmul__", [](std::shared_ptr<JITVar> a, float b) { return a * b; })
        .def("__truediv__", [](std::shared_ptr<JITVar> a, std::shared_ptr<JITVar> b) { return a / b; })
        .def("__truediv__", [](std::shared_ptr<JITVar> a, float b) { return a / b; })
        .def("__neg__", [](std::shared_ptr<JITVar> a) { return -a; });

    jit_mod.def("relu", &JIT::relu, py::arg("a"));
    jit_mod.def("gelu", &JIT::gelu, py::arg("a"));
    jit_mod.def("sigmoid", &JIT::sigmoid, py::arg("a"));
    jit_mod.def("tanh", &JIT::tanh, py::arg("a"));
    jit_mod.def("sqrt", &JIT::sqrt, py::arg("a"));
    jit_mod.def("exp", &JIT::exp, py::arg("a"));
    jit_mod.def("log", &JIT::log, py::arg("a"));
    jit_mod.def("abs", &JIT::abs, py::arg("a"));
    jit_mod.def("fused_bias_relu", [](std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> bias) {
        auto v_x = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
        auto v_b = std::make_shared<JITVar>(JITVar::OpType::INPUT, "b");
        auto expr = JIT::relu(v_x + v_b);
        JITFunction fn("fused_bias_relu", expr, {v_x, v_b});
        return fn({x, bias});
    }, py::arg("x"), py::arg("bias"));
    jit_mod.def("fused_bias_gelu", [](std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> bias) {
        auto v_x = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
        auto v_b = std::make_shared<JITVar>(JITVar::OpType::INPUT, "b");
        auto expr = JIT::gelu(v_x + v_b);
        JITFunction fn("fused_bias_gelu", expr, {v_x, v_b});
        return fn({x, bias});
    }, py::arg("x"), py::arg("bias"));
    jit_mod.def("fused_residual_add", [](std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> residual) {
        auto v_x = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
        auto v_r = std::make_shared<JITVar>(JITVar::OpType::INPUT, "r");
        auto expr = v_x + v_r;
        JITFunction fn("fused_residual_add", expr, {v_x, v_r});
        return fn({x, residual});
    }, py::arg("x"), py::arg("residual"));
    jit_mod.def("fused_residual_gelu", [](std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> residual) {
        auto v_x = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
        auto v_r = std::make_shared<JITVar>(JITVar::OpType::INPUT, "r");
        auto expr = JIT::gelu(v_x + v_r);
        JITFunction fn("fused_residual_gelu", expr, {v_x, v_r});
        return fn({x, residual});
    }, py::arg("x"), py::arg("residual"));

    py::class_<JITFunction, std::shared_ptr<JITFunction>>(jit_mod, "JITFunction")
        .def(py::init<const std::string&, std::shared_ptr<JITVar>, const std::vector<std::shared_ptr<JITVar>>&>(),
             py::arg("name"), py::arg("expr"), py::arg("inputs"))
        .def("__call__", &JITFunction::operator(), py::arg("args"))
        .def("forward", &JITFunction::operator(), py::arg("args"))
        .def("save", &JITFunction::save, py::arg("filepath"))
        .def_static("load", &JITFunction::load, py::arg("filepath"));

    py::class_<Tracer>(jit_mod, "Tracer")
        .def_static("trace", &Tracer::trace, py::arg("inputs"), py::arg("func"), py::arg("name") = "traced_fn")
        .def_static("derivative", &Tracer::derivative, py::arg("var"), py::arg("wrt"))
        .def_static("serialize_var", &Tracer::serialize_var, py::arg("var"))
        .def_static("deserialize_var", [](const std::string& str) {
            size_t pos = 0;
            return Tracer::deserialize_var(str, pos);
        }, py::arg("str"));

    auto data_mod = m.def_submodule("data", "LiteTorch Data Loading and Datasets");

    py::class_<data::Dataset, PyDataset, std::shared_ptr<data::Dataset>>(data_mod, "Dataset")
        .def(py::init<>())
        .def("size", &data::Dataset::size)
        .def("get", &data::Dataset::get, py::arg("index"))
        .def("__len__", &data::Dataset::size)
        .def("__getitem__", &data::Dataset::get, py::arg("index"));

    py::class_<data::TensorDataset, data::Dataset, std::shared_ptr<data::TensorDataset>>(data_mod, "TensorDataset")
        .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("x"), py::arg("y"))
        .def("size", &data::TensorDataset::size)
        .def("get", &data::TensorDataset::get, py::arg("index"))
        .def_readwrite("x", &data::TensorDataset::x)
        .def_readwrite("y", &data::TensorDataset::y);

    py::class_<data::Sampler, std::shared_ptr<data::Sampler>>(data_mod, "Sampler")
        .def("indices", &data::Sampler::indices)
        .def("__len__", &data::Sampler::size);

    py::class_<data::SequentialSampler, data::Sampler, std::shared_ptr<data::SequentialSampler>>(data_mod, "SequentialSampler")
        .def(py::init<size_t>(), py::arg("n"))
        .def("indices", &data::SequentialSampler::indices)
        .def("__len__", &data::SequentialSampler::size);

    py::class_<data::RandomSampler, data::Sampler, std::shared_ptr<data::RandomSampler>>(data_mod, "RandomSampler")
        .def(py::init<size_t, bool, size_t, uint64_t>(),
             py::arg("n"), py::arg("replacement") = false, py::arg("num_samples") = 0, py::arg("seed") = 0)
        .def("indices", &data::RandomSampler::indices)
        .def("__len__", &data::RandomSampler::size);

    py::class_<data::BatchSampler, data::Sampler, std::shared_ptr<data::BatchSampler>>(data_mod, "BatchSampler")
        .def(py::init<std::shared_ptr<data::Sampler>, size_t, bool>(),
             py::arg("sampler"), py::arg("batch_size"), py::arg("drop_last") = false)
        .def("batches", &data::BatchSampler::batches)
        .def("__len__", &data::BatchSampler::size)
        .def_property_readonly("batch_size", &data::BatchSampler::batch_size)
        .def_property_readonly("drop_last", &data::BatchSampler::drop_last);

    py::class_<data::Subset, data::Dataset, std::shared_ptr<data::Subset>>(data_mod, "Subset")
        .def(py::init<std::shared_ptr<data::Dataset>, std::vector<size_t>>(),
             py::arg("dataset"), py::arg("indices"))
        .def("size", &data::Subset::size)
        .def("get", &data::Subset::get, py::arg("index"))
        .def("__len__", &data::Subset::size)
        .def("__getitem__", &data::Subset::get, py::arg("index"));

    data_mod.def("random_split", &data::random_split,
                 py::arg("dataset"), py::arg("lengths"), py::arg("seed") = 0);

    py::class_<data::DataLoader, std::shared_ptr<data::DataLoader>>(data_mod, "DataLoader")
        .def(py::init([](std::shared_ptr<data::Dataset> dataset, size_t batch_size, bool shuffle,
                         py::object device_obj, size_t prefetch_limit, size_t num_workers,
                         bool drop_last, py::object sampler_obj, py::object collate_obj) {
            Device dev = device_obj.is_none() ? Device(DeviceType::CPU, 0) : device_obj.cast<Device>();
            std::shared_ptr<data::Sampler> sampler;
            if (!sampler_obj.is_none()) {
                sampler = sampler_obj.cast<std::shared_ptr<data::Sampler>>();
            }
            auto dl = std::make_shared<data::DataLoader>(
                dataset, batch_size, shuffle, dev, prefetch_limit, num_workers, drop_last, sampler);
            if (!collate_obj.is_none()) {
                py::function f = collate_obj.cast<py::function>();
                dl->set_collate_fn([f](const std::vector<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>& items) {
                    py::list lst;
                    for (auto& it : items) {
                        lst.append(py::make_tuple(it.first, it.second));
                    }
                    py::tuple res = f(lst).cast<py::tuple>();
                    return std::make_pair(res[0].cast<std::shared_ptr<Tensor>>(),
                                          res[1].cast<std::shared_ptr<Tensor>>());
                }, true);
            }
            return dl;
        }), py::arg("dataset"), py::arg("batch_size"), py::arg("shuffle") = true,
            py::arg("device") = py::none(), py::arg("prefetch_limit") = 2,
            py::arg("num_workers") = 0, py::arg("drop_last") = false,
            py::arg("sampler") = py::none(), py::arg("collate_fn") = py::none())
        .def("reset", &data::DataLoader::reset)
        .def("next", [](data::DataLoader& self) -> py::object {
            std::shared_ptr<Tensor> bx;
            std::shared_ptr<Tensor> by;
            bool ok;
            { py::gil_scoped_release rel; ok = self.next(bx, by); }
            if (ok) {
                return py::make_tuple(bx, by);
            }
            return py::none();
        })
        .def("__iter__", [](data::DataLoader& self) -> data::DataLoader& {
            self.reset();
            return self;
        })
        .def("__next__", [](data::DataLoader& self) -> py::object {
            std::shared_ptr<Tensor> bx;
            std::shared_ptr<Tensor> by;
            bool ok;
            { py::gil_scoped_release rel; ok = self.next(bx, by); }
            if (!ok) {
                throw py::stop_iteration();
            }
            return py::make_tuple(bx, by);
        })
        .def("__len__", &data::DataLoader::num_batches)
        .def("num_batches", &data::DataLoader::num_batches)
        .def("save_state", &data::DataLoader::save_state, py::arg("filepath"))
        .def("load_state", &data::DataLoader::load_state, py::arg("filepath"));

    m.def("save_parameters", &litetorch::save_parameters, py::arg("params"), py::arg("filepath"));
    m.def("load_parameters", &litetorch::load_parameters, py::arg("params"), py::arg("filepath"));
    m.def("save_safetensors", [](py::dict tensors, const std::string& filepath) {
        std::map<std::string, std::shared_ptr<Tensor>> m;
        for (auto kv : tensors) {
            m[kv.first.cast<std::string>()] = kv.second.cast<std::shared_ptr<Tensor>>();
        }
        litetorch::save_safetensors(m, filepath);
    }, py::arg("tensors"), py::arg("filepath"));
    m.def("load_safetensors", [](const std::string& filepath) {
        auto m = litetorch::load_safetensors(filepath);
        py::dict d;
        for (auto& kv : m) d[py::str(kv.first)] = kv.second;
        return d;
    }, py::arg("filepath"));
    m.def("save_optimizer_state", &litetorch::save_optimizer_state, py::arg("optimizer"), py::arg("filepath"));
    m.def("load_optimizer_state", &litetorch::load_optimizer_state, py::arg("optimizer"), py::arg("filepath"));

    m.def("save", [](py::object obj, const std::string& filepath) {
        if (py::isinstance<nn::Module>(obj)) {
            auto mod = obj.cast<nn::Module*>();
            litetorch::save_parameters(mod->parameters(), filepath);
        } else if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
            auto params = obj.cast<std::vector<std::shared_ptr<Tensor>>>();
            litetorch::save_parameters(params, filepath);
        } else if (py::isinstance<optim::Optimizer>(obj)) {
            auto opt = obj.cast<optim::Optimizer*>();
            litetorch::save_optimizer_state(opt, filepath);
        } else {
            throw std::runtime_error("save expects a Module, list of Tensor parameters, or Optimizer");
        }
    }, py::arg("obj"), py::arg("filepath"));

    m.def("load", [](py::object obj, const std::string& filepath) {
        if (py::isinstance<nn::Module>(obj)) {
            auto mod = obj.cast<nn::Module*>();
            litetorch::load_parameters(mod->parameters(), filepath);
        } else if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
            auto params = obj.cast<std::vector<std::shared_ptr<Tensor>>>();
            litetorch::load_parameters(params, filepath);
        } else if (py::isinstance<optim::Optimizer>(obj)) {
            auto opt = obj.cast<optim::Optimizer*>();
            litetorch::load_optimizer_state(opt, filepath);
        } else {
            throw std::runtime_error("load expects a Module, list of Tensor parameters, or Optimizer");
        }
    }, py::arg("obj"), py::arg("filepath"));

    m.def("manual_seed", &litetorch::manual_seed, py::arg("seed"));
    m.def("get_rng_state", &litetorch::get_rng_state);
    m.def("set_rng_state", &litetorch::set_rng_state, py::arg("state"));

    m.def("randn", [](const std::vector<int64_t>& shape, const Device& device, bool requires_grad, DataType dtype) {
        return Tensor::randn(shape, device, requires_grad, dtype);
    }, py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32);
    m.def("ones", [](const std::vector<int64_t>& shape, const Device& device, bool requires_grad, DataType dtype) {
        return Tensor::ones(shape, device, requires_grad, dtype);
    }, py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32);
    m.def("zeros", [](const std::vector<int64_t>& shape, const Device& device, bool requires_grad, DataType dtype) {
        return Tensor::zeros(shape, device, requires_grad, dtype);
    }, py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32);
    m.def("arange", [](int64_t start, int64_t end, int64_t step, const Device& device, DataType dtype) {
        return Tensor::arange(start, end, step, device, dtype);
    }, py::arg("start"), py::arg("end"), py::arg("step") = 1, py::arg("device") = Device(DeviceType::CPU, 0), py::arg("dtype") = DataType::FP32);

    // === W1-T1 core-ops ===
    auto w1_opt_device = [](py::object device) -> Device {
        return device.is_none() ? Device(DeviceType::CPU, 0) : device.cast<Device>();
    };
    auto w1_opt_dtype = [](py::object dtype, DataType dflt) -> DataType {
        return dtype.is_none() ? dflt : dtype.cast<DataType>();
    };
    auto w1_shape_arg = [](py::object shape) {
        std::vector<int64_t> s;
        if (py::isinstance<py::int_>(shape)) {
            s.push_back(shape.cast<int64_t>());
        } else {
            for (auto item : shape.cast<py::sequence>()) s.push_back(item.cast<int64_t>());
        }
        return s;
    };
    m.def("tensor", [w1_opt_device, w1_opt_dtype](py::object data, py::object device, bool requires_grad, py::object dtype) {
        std::vector<float> flat;
        std::vector<int64_t> shape;
        std::function<void(py::handle, size_t)> rec = [&](py::handle obj, size_t depth) {
            if (py::isinstance<py::sequence>(obj) && !py::isinstance<py::str>(obj)) {
                auto seq = py::reinterpret_borrow<py::sequence>(obj);
                size_t n = seq.size();
                if (depth == shape.size()) shape.push_back((int64_t)n);
                else if (shape[depth] != (int64_t)n)
                    throw std::runtime_error("[litetorch Error] tensor(): inconsistent nested sequence sizes");
                for (auto item : seq) rec(item, depth + 1);
            } else {
                double v = py::isinstance<py::bool_>(obj) ? (obj.cast<bool>() ? 1.0 : 0.0) : obj.cast<double>();
                flat.push_back((float)v);
            }
        };
        rec(data, 0);
        if (shape.empty()) shape.push_back(1);
        return Ops::from_data(flat, shape, w1_opt_device(device), requires_grad, w1_opt_dtype(dtype, DataType::FP32));
    }, py::arg("data"), py::arg("device") = py::none(), py::arg("requires_grad") = false, py::arg("dtype") = py::none());
    m.def("empty", [w1_opt_device, w1_opt_dtype, w1_shape_arg](py::object shape, py::object device, bool requires_grad, py::object dtype) {
        return Ops::empty(w1_shape_arg(shape), w1_opt_device(device), requires_grad, w1_opt_dtype(dtype, DataType::FP32));
    }, py::arg("shape"), py::arg("device") = py::none(), py::arg("requires_grad") = false, py::arg("dtype") = py::none());
    m.def("full", [w1_opt_device, w1_opt_dtype, w1_shape_arg](py::object shape, double fill_value, py::object device, bool requires_grad, py::object dtype) {
        return Ops::full(w1_shape_arg(shape), (float)fill_value, w1_opt_device(device), requires_grad, w1_opt_dtype(dtype, DataType::FP32));
    }, py::arg("shape"), py::arg("fill_value"), py::arg("device") = py::none(), py::arg("requires_grad") = false, py::arg("dtype") = py::none());
    m.def("linspace", [w1_opt_device, w1_opt_dtype](double start, double end, int64_t steps, py::object device, py::object dtype) {
        return Ops::linspace((float)start, (float)end, steps, w1_opt_device(device), w1_opt_dtype(dtype, DataType::FP32));
    }, py::arg("start"), py::arg("end"), py::arg("steps"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
    m.def("logspace", [w1_opt_device, w1_opt_dtype](double start, double end, int64_t steps, double base, py::object device, py::object dtype) {
        return Ops::logspace((float)start, (float)end, steps, (float)base, w1_opt_device(device), w1_opt_dtype(dtype, DataType::FP32));
    }, py::arg("start"), py::arg("end"), py::arg("steps"), py::arg("base") = 10.0, py::arg("device") = py::none(), py::arg("dtype") = py::none());
    m.def("eye", [w1_opt_device, w1_opt_dtype](int64_t n, py::object m_arg, py::object device, py::object dtype) {
        int64_t m = m_arg.is_none() ? -1 : m_arg.cast<int64_t>();
        return Ops::eye(n, m, w1_opt_device(device), w1_opt_dtype(dtype, DataType::FP32));
    }, py::arg("n"), py::arg("m") = py::none(), py::arg("device") = py::none(), py::arg("dtype") = py::none());
    m.def("rand", [w1_opt_device, w1_opt_dtype, w1_shape_arg](py::object shape, py::object device, bool requires_grad, py::object dtype) {
        return Ops::rand(w1_shape_arg(shape), w1_opt_device(device), requires_grad, w1_opt_dtype(dtype, DataType::FP32));
    }, py::arg("shape"), py::arg("device") = py::none(), py::arg("requires_grad") = false, py::arg("dtype") = py::none());
    m.def("randint", [w1_opt_device, w1_opt_dtype, w1_shape_arg](int64_t low, int64_t high, py::object shape, py::object device, py::object dtype) {
        return Ops::randint(low, high, w1_shape_arg(shape), w1_opt_device(device), w1_opt_dtype(dtype, DataType::INT32));
    }, py::arg("low"), py::arg("high"), py::arg("shape"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
    m.def("randint", [w1_opt_device, w1_opt_dtype, w1_shape_arg](int64_t high, py::object shape, py::object device, py::object dtype) {
        return Ops::randint(0, high, w1_shape_arg(shape), w1_opt_device(device), w1_opt_dtype(dtype, DataType::INT32));
    }, py::arg("high"), py::arg("shape"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
    m.def("randperm", [w1_opt_device, w1_opt_dtype](int64_t n, py::object device, py::object dtype) {
        return Ops::randperm(n, w1_opt_device(device), w1_opt_dtype(dtype, DataType::INT32));
    }, py::arg("n"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
    m.def("zeros_like", [w1_opt_dtype](std::shared_ptr<Tensor> a, py::object device, py::object dtype) {
        Device d = device.is_none() ? a->device : device.cast<Device>();
        return Ops::zeros_like(a, d, w1_opt_dtype(dtype, a->dtype));
    }, py::arg("a"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
    m.def("ones_like", [w1_opt_dtype](std::shared_ptr<Tensor> a, py::object device, py::object dtype) {
        Device d = device.is_none() ? a->device : device.cast<Device>();
        return Ops::ones_like(a, d, w1_opt_dtype(dtype, a->dtype));
    }, py::arg("a"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
    m.def("empty_like", [w1_opt_dtype](std::shared_ptr<Tensor> a, py::object device, py::object dtype) {
        Device d = device.is_none() ? a->device : device.cast<Device>();
        return Ops::empty_like(a, d, w1_opt_dtype(dtype, a->dtype));
    }, py::arg("a"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
    m.def("full_like", [w1_opt_dtype](std::shared_ptr<Tensor> a, double fill_value, py::object device, py::object dtype) {
        Device d = device.is_none() ? a->device : device.cast<Device>();
        return Ops::full_like(a, (float)fill_value, d, w1_opt_dtype(dtype, a->dtype));
    }, py::arg("a"), py::arg("fill_value"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
    m.def("rand_like", [w1_opt_dtype](std::shared_ptr<Tensor> a, py::object device, py::object dtype) {
        Device d = device.is_none() ? a->device : device.cast<Device>();
        return Ops::rand_like(a, d, w1_opt_dtype(dtype, a->dtype));
    }, py::arg("a"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
    m.def("randn_like", [w1_opt_dtype](std::shared_ptr<Tensor> a, py::object device, py::object dtype) {
        Device d = device.is_none() ? a->device : device.cast<Device>();
        return Ops::randn_like(a, d, w1_opt_dtype(dtype, a->dtype));
    }, py::arg("a"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
    // === end W1-T1 core-ops ===

    m.def("empty_cache", []() {
        CachingAllocator::get().empty_cache();
        auto backend = litetorch::BackendDispatcher::get().get_backend();
        if (backend && backend->is_available()) {
            backend->empty_cache();
        }
    });

    m.def("set_max_cpu_cache_size", [](size_t bytes) {
        CachingAllocator::get().set_max_cpu_cache_size(bytes);
    }, py::arg("bytes"));

    m.def("get_cached_cpu_bytes", []() -> size_t {
        return CachingAllocator::get().get_cached_cpu_bytes();
    });

    m.def("get_cached_gpu_bytes", []() -> size_t {
        return CachingAllocator::get().get_cached_gpu_bytes();
    });

    py::class_<NoGradContext>(m, "no_grad")
        .def(py::init<>())
        .def("__enter__", [](NoGradContext& self) -> NoGradContext& { self.enter(); return self; })
        .def("__exit__", [](NoGradContext& self, py::object, py::object, py::object) { self.exit(); });

    py::class_<EnableGradContext>(m, "enable_grad")
        .def(py::init<>())
        .def("__enter__", [](EnableGradContext& self) -> EnableGradContext& { self.enter(); return self; })
        .def("__exit__", [](EnableGradContext& self, py::object, py::object, py::object) { self.exit(); });

    py::class_<InferenceModeContext>(m, "inference_mode")
        .def(py::init<>())
        .def("__enter__", [](InferenceModeContext& self) -> InferenceModeContext& { self.enter(); return self; })
        .def("__exit__", [](InferenceModeContext& self, py::object, py::object, py::object) { self.exit(); });

    m.def("is_inference_mode", []() -> bool {
        return inference_mode_flag();
    });

    m.def("is_grad_enabled", []() -> bool {
        return Autograd::is_grad_enabled();
    });
    m.def("set_grad_enabled", [](bool enabled) {
        Autograd::set_grad_enabled(enabled);
    }, py::arg("enabled"));

    // === W2-T4 autograd ===
    auto autograd_mod = m.def_submodule("autograd", "LiteTorch Autograd");
    py::class_<FunctionContext, std::shared_ptr<FunctionContext>>(autograd_mod, "FunctionContext", py::dynamic_attr())
        .def(py::init<>())
        .def("save_for_backward", [](FunctionContext& self, py::args args) {
            std::vector<std::shared_ptr<Tensor>> ts;
            for (auto a : args) ts.push_back(a.cast<std::shared_ptr<Tensor>>());
            self.save_for_backward(ts);
        })
        .def_property_readonly("saved_tensors", [](FunctionContext& self) {
            const auto& st = self.get_saved_tensors();
            py::tuple t(st.size());
            for (size_t i = 0; i < st.size(); ++i) t[i] = py::cast(st[i]);
            return t;
        })
        .def_property_readonly("needs_input_grad", [](FunctionContext& self) {
            py::list l;
            for (bool b : self.needs_input_grad()) l.append(b);
            return l;
        });
    py::class_<Function, PyFunction, std::shared_ptr<Function>> pyFn(autograd_mod, "Function");
    pyFn.def(py::init<>());
    py::object apply_fn = py::cpp_function([](py::object cls, py::args args) -> py::object {
        py::object inst = cls();
        auto fn = inst.cast<std::shared_ptr<Function>>();
        auto pyfn = std::dynamic_pointer_cast<PyFunction>(fn);
        if (!pyfn)
            throw std::runtime_error("[litetorch Error] autograd.Function.apply: could not access Python subclass state");
        std::vector<std::shared_ptr<Tensor>> tensor_inputs;
        py::tuple tmpl(args.size());
        for (size_t i = 0; i < (size_t)args.size(); ++i) {
            bool is_tensor = false;
            try {
                tensor_inputs.push_back(args[i].cast<std::shared_ptr<Tensor>>());
                is_tensor = true;
            } catch (py::cast_error&) {}
            if (is_tensor) tmpl[i] = py::none();
            else tmpl[i] = args[i];
        }
        pyfn->arg_template = tmpl;
        pyfn->py_self = inst;
        std::vector<std::shared_ptr<Tensor>> outputs = Function::apply(fn, tensor_inputs);
        if (outputs.size() == 1) return py::cast(outputs[0]);
        py::tuple out(outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) out[i] = py::cast(outputs[i]);
        return out;
    }, py::arg("cls"));
    pyFn.attr("apply") = py::module::import("builtins").attr("classmethod")(apply_fn);
    autograd_mod.def("grad", [](py::object outputs, py::object inputs, py::object grad_outputs,
            bool retain_graph, bool create_graph, bool allow_unused) -> py::object {
        auto outs = w2t4_unpack(outputs, false);
        auto ins = w2t4_unpack(inputs, false);
        std::vector<std::shared_ptr<Tensor>> gos;
        if (!grad_outputs.is_none()) gos = w2t4_unpack(grad_outputs, false);
        std::vector<std::shared_ptr<Tensor>> res =
            Autograd::grad(outs, ins, gos, retain_graph, create_graph, allow_unused);
        py::tuple t(res.size());
        for (size_t i = 0; i < res.size(); ++i) {
            if (res[i]) t[i] = py::cast(res[i]);
            else t[i] = py::none();
        }
        return t;
    }, py::arg("outputs"), py::arg("inputs"), py::arg("grad_outputs") = py::none(),
       py::arg("retain_graph") = false, py::arg("create_graph") = false, py::arg("allow_unused") = false);
    autograd_mod.def("gradcheck", [](py::function func, py::object inputs, double eps, double atol,
            double rtol, bool raise_exception) -> bool {
        auto ins = w2t4_unpack(inputs, false);
        AutogradFunc f = [&](const std::vector<std::shared_ptr<Tensor>>& xs)
            -> std::vector<std::shared_ptr<Tensor>> {
            py::gil_scoped_acquire gil;
            py::object res = func(*py::cast(xs));
            return w2t4_unpack(res, false);
        };
        std::string err;
        return autograd_gradcheck(f, ins, eps, atol, rtol, raise_exception, err);
    }, py::arg("func"), py::arg("inputs"), py::arg("eps") = 1e-6, py::arg("atol") = 1e-5,
       py::arg("rtol") = 1e-3, py::arg("raise_exception") = true);
    autograd_mod.def("gradgradcheck", [](py::function func, py::object inputs, double eps, double atol,
            double rtol, bool raise_exception) -> bool {
        auto ins = w2t4_unpack(inputs, false);
        AutogradFunc f = [&](const std::vector<std::shared_ptr<Tensor>>& xs)
            -> std::vector<std::shared_ptr<Tensor>> {
            py::gil_scoped_acquire gil;
            py::object res = func(*py::cast(xs));
            return w2t4_unpack(res, false);
        };
        std::string err;
        return autograd_gradgradcheck(f, ins, eps, atol, rtol, raise_exception, err);
    }, py::arg("func"), py::arg("inputs"), py::arg("eps") = 1e-6, py::arg("atol") = 1e-5,
       py::arg("rtol") = 1e-3, py::arg("raise_exception") = true);
    autograd_mod.def("retain_grad", [](std::shared_ptr<Tensor> t) { Autograd::retain_grad(t); },
        py::arg("tensor"));
    // === end W2-T4 autograd ===
}
