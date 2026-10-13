#ifndef LITETORCH_BINDINGS_COMMON_H
#define LITETORCH_BINDINGS_COMMON_H

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
#include "litetorch/dtensor.h"
#include "litetorch/device_mesh.h"
#include "litetorch/jit.h"
#include "litetorch/allocator.h"
#include "litetorch/tpu.h"
#include "litetorch/memory_manager.h"

// === W1-T5 functional: includes ===
#include "litetorch/functional.h"
// === W3-T1 linalg: includes ===
#include "litetorch/linalg.h"
// === W3-T2 distributions: includes ===
#include "litetorch/distributions.h"
// === DPO (LT-ENG-039): includes ===
#include "litetorch/dpo.h"
// === I2 inference (LT-ENG-051): includes ===
#include "litetorch/inference.h"



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



#endif
