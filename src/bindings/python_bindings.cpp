#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "litetorch/device.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/optim.h"
#include "litetorch/checkpoint.h"
#include "litetorch/backend.h"
#include "litetorch/continuous_batching.h"
#include "litetorch/llm_serving.h"
#include "litetorch/distributed.h"
#include "litetorch/serialization.h"

namespace py = pybind11;
using namespace litetorch;

class PyModule : public nn::Module {
public:
    using nn::Module::Module;

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        PYBIND11_OVERRIDE_PURE(
            std::shared_ptr<Tensor>,
            nn::Module,
            forward,
            input
        );
    }

    std::vector<std::shared_ptr<Tensor>> parameters() override {
        PYBIND11_OVERRIDE(
            std::vector<std::shared_ptr<Tensor>>,
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

PYBIND11_MODULE(litetorch, m) {
    m.doc() = "LiteTorch Python Bindings";

    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("GPU", DeviceType::GPU)
        .value("META", DeviceType::META)
        .export_values();

    py::class_<Device>(m, "Device")
        .def(py::init<DeviceType, int>(), py::arg("type") = DeviceType::CPU, py::arg("index") = 0)
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

    m.def("auto_device", []() {
        if (std::getenv("LITETORCH_NO_NATIVE_GPU")) {
            if (litetorch::CLBackend::get().is_available()) {
                return Device(DeviceType::GPU, 0);
            }
            return Device(DeviceType::CPU, 0);
        }
        auto backend = litetorch::BackendDispatcher::get().get_backend();
        if (backend && backend->is_available()) {
            return Device(DeviceType::GPU, 0);
        }
        if (litetorch::CLBackend::get().is_available()) {
            return Device(DeviceType::GPU, 0);
        }
        return Device(DeviceType::CPU, 0);
    });

    auto cuda_mod = m.def_submodule("cuda");
    cuda_mod.def("is_available", []() {
        if (std::getenv("LITETORCH_NO_NATIVE_GPU")) {
            return false;
        }
        auto backend = litetorch::BackendDispatcher::get().get_backend();
        return backend && backend->is_available();
    });

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
        .export_values();

    py::class_<distributed::ProcessGroup, std::shared_ptr<distributed::ProcessGroup>>(m, "ProcessGroup");

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def_static("create", &Tensor::create, py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32)
        .def_static("from_vector", &Tensor::from_vector, py::arg("data"), py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32)
        .def_static("zeros", &Tensor::zeros, py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false)
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
        .def("to", &Tensor::to)
        .def("view", &Tensor::view)
        .def("transpose", &Tensor::transpose)
        .def("contiguous", &Tensor::contiguous)
        .def("clone", &Tensor::clone)
        .def("copy_", &Tensor::copy_)
        .def("add_", &Tensor::add_)
        .def("__repr__", [](const Tensor& self) {
            return "<litetorch.Tensor shape=" + std::to_string(self.numel()) + " device=" + self.device.to_string() + ">";
        });

    auto ops = m.def_submodule("Ops", "LiteTorch Tensor Operations");
    ops.def("add", &Ops::add);
    ops.def("sub", &Ops::sub);
    ops.def("mul", &Ops::mul);
    ops.def("div", &Ops::div);
    ops.def("matmul", &Ops::matmul);
    ops.def("bmm", &Ops::bmm);
    ops.def("sum", &Ops::sum);
    ops.def("mean", &Ops::mean);
    ops.def("max", &Ops::max);
    ops.def("relu", &Ops::relu);
    ops.def("sigmoid", &Ops::sigmoid);
    ops.def("tanh", &Ops::tanh);
    ops.def("softmax", &Ops::softmax, py::arg("a"), py::arg("dim") = -1);
    ops.def("gelu", &Ops::gelu);
    ops.def("layer_norm", &Ops::layer_norm, py::arg("input"), py::arg("normalized_shape"), py::arg("weight") = nullptr, py::arg("bias") = nullptr, py::arg("eps") = 1e-5f);
    ops.def("fused_add_layernorm", &Ops::fused_add_layernorm, py::arg("input"), py::arg("residual"), py::arg("normalized_shape"), py::arg("weight") = nullptr, py::arg("bias") = nullptr, py::arg("eps") = 1e-5f);
    ops.def("mse_loss", &Ops::mse_loss);
    ops.def("cross_entropy_loss", &Ops::cross_entropy_loss);
    ops.def("rope", &Ops::rope);
    ops.def("flash_attention", &Ops::flash_attention);
    ops.def("paged_attention", &Ops::paged_attention);
    ops.def("w8a8_matmul", &Ops::w8a8_matmul);
    ops.def("moe_gate", [](std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> gate_weight, int top_k) {
        std::shared_ptr<Tensor> indices_out;
        auto probs = Ops::moe_gate(input, gate_weight, top_k, indices_out);
        return std::make_pair(probs, indices_out);
    });
    ops.def("flash_decoding", &Ops::flash_decoding);
    ops.def("ring_attention", &Ops::ring_attention);

    m.def("checkpoint", &checkpoint, "Activation Checkpointing helper");

    auto nn_mod = m.def_submodule("nn", "LiteTorch Neural Network Modules");
    
    py::class_<nn::Module, PyModule, std::shared_ptr<nn::Module>>(nn_mod, "Module")
        .def(py::init<>())
        .def("forward", &nn::Module::forward)
        .def("parameters", &nn::Module::parameters)
        .def("to", &nn::Module::to);

    py::class_<nn::Linear, nn::Module, std::shared_ptr<nn::Linear>>(nn_mod, "Linear")
        .def(py::init<int, int, bool>(), py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true)
        .def("forward", &nn::Linear::forward)
        .def("parameters", &nn::Linear::parameters)
        .def("to", &nn::Linear::to)
        .def_readwrite("weight", &nn::Linear::weight)
        .def_readwrite("bias", &nn::Linear::bias);

    py::class_<nn::LayerNorm, nn::Module, std::shared_ptr<nn::LayerNorm>>(nn_mod, "LayerNorm")
        .def(py::init<const std::vector<int64_t>&, float>(), py::arg("normalized_shape"), py::arg("eps") = 1e-5f)
        .def("forward", &nn::LayerNorm::forward)
        .def("parameters", &nn::LayerNorm::parameters)
        .def("to", &nn::LayerNorm::to)
        .def_readwrite("weight", &nn::LayerNorm::weight)
        .def_readwrite("bias", &nn::LayerNorm::bias);

    py::class_<nn::MultiHeadAttention, nn::Module, std::shared_ptr<nn::MultiHeadAttention>>(nn_mod, "MultiHeadAttention")
        .def(py::init<int, int>(), py::arg("embed_dim"), py::arg("num_heads"))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::MultiHeadAttention::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::MultiHeadAttention::forward))
        .def("parameters", &nn::MultiHeadAttention::parameters)
        .def("to", &nn::MultiHeadAttention::to);

    py::class_<nn::TransformerDecoderLayer, nn::Module, std::shared_ptr<nn::TransformerDecoderLayer>>(nn_mod, "TransformerDecoderLayer")
        .def(py::init<int, int, int>(), py::arg("embed_dim"), py::arg("num_heads"), py::arg("dim_feedforward") = 2048)
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::TransformerDecoderLayer::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::TransformerDecoderLayer::forward))
        .def("parameters", &nn::TransformerDecoderLayer::parameters)
        .def("to", &nn::TransformerDecoderLayer::to);

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

    auto optim_mod = m.def_submodule("optim", "LiteTorch Optimizers");
    
    py::class_<optim::Optimizer, std::shared_ptr<optim::Optimizer>>(optim_mod, "Optimizer")
        .def("step", &optim::Optimizer::step)
        .def("zero_grad", &optim::Optimizer::zero_grad);

    py::class_<optim::SGD, optim::Optimizer, std::shared_ptr<optim::SGD>>(optim_mod, "SGD")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float, float>(), py::arg("params"), py::arg("lr") = 0.01f, py::arg("momentum") = 0.0f)
        .def("step", &optim::SGD::step)
        .def("zero_grad", &optim::SGD::zero_grad);

    py::class_<optim::Adam, optim::Optimizer, std::shared_ptr<optim::Adam>>(optim_mod, "Adam")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.001f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def("step", &optim::Adam::step)
        .def("zero_grad", &optim::Adam::zero_grad);

    py::class_<optim::AdamW, optim::Optimizer, std::shared_ptr<optim::AdamW>>(optim_mod, "AdamW")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.001f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.01f)
        .def("step", &optim::AdamW::step)
        .def("zero_grad", &optim::AdamW::zero_grad);

    m.def("save_parameters", &litetorch::save_parameters, py::arg("params"), py::arg("filepath"));
    m.def("load_parameters", &litetorch::load_parameters, py::arg("params"), py::arg("filepath"));
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
}
