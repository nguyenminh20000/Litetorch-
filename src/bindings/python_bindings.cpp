#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

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
#include "litetorch/fsdp.h"
#include "litetorch/dtensor.h"
#include "litetorch/device_mesh.h"
#include "litetorch/jit.h"
#include "litetorch/allocator.h"

namespace py = pybind11;
using namespace litetorch;

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
        .export_values();

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
        .def("reshape", &Tensor::reshape)
        .def("transpose", &Tensor::transpose)
        .def("contiguous", &Tensor::contiguous)
        .def("clone", &Tensor::clone)
        .def("cast", &Tensor::cast, py::arg("target_dtype"))
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
    ops.def("conv3d", &Ops::conv3d, py::arg("input"), py::arg("weight"), py::arg("bias") = nullptr, py::arg("stride") = 1, py::arg("padding") = 0);
    ops.def("max_pool2d", &Ops::max_pool2d, py::arg("input"), py::arg("kernel_size"), py::arg("stride") = -1, py::arg("padding") = 0);
    ops.def("max_pool3d", &Ops::max_pool3d, py::arg("input"), py::arg("kernel_size"), py::arg("stride") = -1, py::arg("padding") = 0);
    ops.def("adaptive_avg_pool2d", &Ops::adaptive_avg_pool2d, py::arg("input"), py::arg("output_height"), py::arg("output_width"));
    ops.def("cat", &Ops::cat, py::arg("tensors"), py::arg("dim") = 0);
    ops.def("squeeze", &Ops::squeeze, py::arg("a"), py::arg("dim") = -1);
    ops.def("unsqueeze", &Ops::unsqueeze, py::arg("a"), py::arg("dim"));
    ops.def("cast", &Ops::cast, py::arg("a"), py::arg("target_dtype"));
    ops.def("fake_quantize", &Ops::fake_quantize, py::arg("input"), py::arg("scale"), py::arg("zero_point") = 0.0f, py::arg("bits") = 8);
    ops.def("mse_loss", &Ops::mse_loss);
    ops.def("cross_entropy_loss", &Ops::cross_entropy_loss);
    ops.def("l1_loss", &Ops::l1_loss);
    ops.def("bce_loss", &Ops::bce_loss);
    ops.def("fused_linear_cross_entropy", &Ops::fused_linear_cross_entropy, py::arg("x"), py::arg("weight"), py::arg("target"), py::arg("bias") = nullptr);
    ops.def("clip_grad_norm_", &Ops::clip_grad_norm_, py::arg("params"), py::arg("max_norm"), py::arg("norm_type") = 2.0f);
    ops.def("rope", &Ops::rope);
    ops.def("flash_attention", &Ops::flash_attention);
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
        .def("train", &nn::Module::train)
        .def("eval", &nn::Module::eval)
        .def("compile", &nn::Module::compile)
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
        .def(py::init<int, int, int, int, int, bool>(), py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"), py::arg("stride") = 1, py::arg("padding") = 0, py::arg("has_bias") = true)
        .def("forward", &nn::Conv2d::forward)
        .def("parameters", &nn::Conv2d::parameters)
        .def("to", &nn::Conv2d::to)
        .def_readwrite("weight", &nn::Conv2d::weight)
        .def_readwrite("bias", &nn::Conv2d::bias)
        .def_readwrite("stride", &nn::Conv2d::stride)
        .def_readwrite("padding", &nn::Conv2d::padding);

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
        .def(py::init<int, float, float>(), py::arg("num_features"), py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f)
        .def("forward", &nn::BatchNorm2d::forward)
        .def("parameters", &nn::BatchNorm2d::parameters)
        .def("to", &nn::BatchNorm2d::to)
        .def_readwrite("weight", &nn::BatchNorm2d::weight)
        .def_readwrite("bias", &nn::BatchNorm2d::bias)
        .def_readwrite("running_mean", &nn::BatchNorm2d::running_mean)
        .def_readwrite("running_var", &nn::BatchNorm2d::running_var)
        .def_readwrite("num_features", &nn::BatchNorm2d::num_features)
        .def_readwrite("eps", &nn::BatchNorm2d::eps)
        .def_readwrite("momentum", &nn::BatchNorm2d::momentum);

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
        .def(py::init<int, int>(), py::arg("num_embeddings"), py::arg("embedding_dim"))
        .def("forward", &nn::Embedding::forward)
        .def("parameters", &nn::Embedding::parameters)
        .def("to", &nn::Embedding::to)
        .def_readwrite("weight", &nn::Embedding::weight)
        .def_readwrite("num_embeddings", &nn::Embedding::num_embeddings)
        .def_readwrite("embedding_dim", &nn::Embedding::embedding_dim);

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
        .def(py::init<int, int>(), py::arg("embed_dim"), py::arg("num_heads"))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::MultiHeadAttention::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::MultiHeadAttention::forward))
        .def("parameters", &nn::MultiHeadAttention::parameters)
        .def("to", &nn::MultiHeadAttention::to)
        .def_readwrite("embed_dim", &nn::MultiHeadAttention::embed_dim)
        .def_readwrite("num_heads", &nn::MultiHeadAttention::num_heads)
        .def_readwrite("head_dim", &nn::MultiHeadAttention::head_dim);

    py::class_<nn::TransformerDecoderLayer, nn::Module, std::shared_ptr<nn::TransformerDecoderLayer>>(nn_mod, "TransformerDecoderLayer")
        .def(py::init<int, int, int>(), py::arg("embed_dim"), py::arg("num_heads"), py::arg("dim_feedforward") = 2048)
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::TransformerDecoderLayer::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::TransformerDecoderLayer::forward))
        .def("parameters", &nn::TransformerDecoderLayer::parameters)
        .def("to", &nn::TransformerDecoderLayer::to);

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

    py::class_<nn::MSELoss, nn::Module, std::shared_ptr<nn::MSELoss>>(nn_mod, "MSELoss")
        .def(py::init<>())
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::MSELoss::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::MSELoss::forward));

    py::class_<nn::CrossEntropyLoss, nn::Module, std::shared_ptr<nn::CrossEntropyLoss>>(nn_mod, "CrossEntropyLoss")
        .def(py::init<>())
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::CrossEntropyLoss::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::CrossEntropyLoss::forward));

    py::class_<nn::BCELoss, nn::Module, std::shared_ptr<nn::BCELoss>>(nn_mod, "BCELoss")
        .def(py::init<>())
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::BCELoss::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::BCELoss::forward));

    py::class_<nn::L1Loss, nn::Module, std::shared_ptr<nn::L1Loss>>(nn_mod, "L1Loss")
        .def(py::init<>())
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::L1Loss::forward))
        .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::L1Loss::forward));

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

    auto optim_mod = m.def_submodule("optim", "LiteTorch Optimizers");
    
    py::class_<optim::Optimizer, std::shared_ptr<optim::Optimizer>>(optim_mod, "Optimizer")
        .def("step", &optim::Optimizer::step)
        .def("zero_grad", &optim::Optimizer::zero_grad)
        .def("get_lr", &optim::Optimizer::get_lr)
        .def("set_lr", &optim::Optimizer::set_lr, py::arg("new_lr"));

    py::class_<optim::SGD, optim::Optimizer, std::shared_ptr<optim::SGD>>(optim_mod, "SGD")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float, float, float>(), py::arg("params"), py::arg("lr") = 0.01f, py::arg("momentum") = 0.0f, py::arg("weight_decay") = 0.0f)
        .def("step", &optim::SGD::step)
        .def("zero_grad", &optim::SGD::zero_grad)
        .def("get_lr", &optim::SGD::get_lr)
        .def("set_lr", &optim::SGD::set_lr, py::arg("new_lr"))
        .def_readwrite("lr", &optim::SGD::lr)
        .def_readwrite("momentum", &optim::SGD::momentum)
        .def_readwrite("weight_decay", &optim::SGD::weight_decay);

    py::class_<optim::Adam, optim::Optimizer, std::shared_ptr<optim::Adam>>(optim_mod, "Adam")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.001f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
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
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float, float, float, float, float, bool>(), py::arg("params"), py::arg("lr") = 0.001f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.01f, py::arg("offload_to_cpu") = false)
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
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.001f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def("step", &optim::AdamW8bit::step)
        .def("zero_grad", &optim::AdamW8bit::zero_grad)
        .def("get_lr", &optim::AdamW8bit::get_lr)
        .def("set_lr", &optim::AdamW8bit::set_lr, py::arg("new_lr"))
        .def_readwrite("lr", &optim::AdamW8bit::lr)
        .def_readwrite("weight_decay", &optim::AdamW8bit::weight_decay);

    py::class_<optim::AdamWFP8, optim::Optimizer, std::shared_ptr<optim::AdamWFP8>>(optim_mod, "AdamWFP8")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.001f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def("step", &optim::AdamWFP8::step)
        .def("zero_grad", &optim::AdamWFP8::zero_grad)
        .def("get_lr", &optim::AdamWFP8::get_lr)
        .def("set_lr", &optim::AdamWFP8::set_lr, py::arg("new_lr"))
        .def_readwrite("lr", &optim::AdamWFP8::lr)
        .def_readwrite("weight_decay", &optim::AdamWFP8::weight_decay);

    py::class_<optim::RMSprop, optim::Optimizer, std::shared_ptr<optim::RMSprop>>(optim_mod, "RMSprop")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.01f, py::arg("alpha") = 0.99f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def("step", &optim::RMSprop::step)
        .def("zero_grad", &optim::RMSprop::zero_grad)
        .def("get_lr", &optim::RMSprop::get_lr)
        .def("set_lr", &optim::RMSprop::set_lr, py::arg("new_lr"))
        .def_readwrite("lr", &optim::RMSprop::lr)
        .def_readwrite("weight_decay", &optim::RMSprop::weight_decay);

    py::class_<optim::ZeRO3Optimizer, optim::Optimizer, std::shared_ptr<optim::ZeRO3Optimizer>>(optim_mod, "ZeRO3Optimizer")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.001f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def("step", &optim::ZeRO3Optimizer::step)
        .def("zero_grad", &optim::ZeRO3Optimizer::zero_grad)
        .def_readwrite("lr", &optim::ZeRO3Optimizer::lr)
        .def_readwrite("beta1", &optim::ZeRO3Optimizer::beta1)
        .def_readwrite("beta2", &optim::ZeRO3Optimizer::beta2)
        .def_readwrite("eps", &optim::ZeRO3Optimizer::eps)
        .def_readwrite("weight_decay", &optim::ZeRO3Optimizer::weight_decay);

    py::class_<optim::StepLR>(optim_mod, "StepLR")
        .def(py::init<optim::Optimizer*, int, float>(), py::arg("optimizer"), py::arg("step_size"), py::arg("gamma") = 0.1f)
        .def("step", &optim::StepLR::step)
        .def_readwrite("step_size", &optim::StepLR::step_size)
        .def_readwrite("gamma", &optim::StepLR::gamma)
        .def_readwrite("last_epoch", &optim::StepLR::last_epoch);

    py::class_<optim::CosineAnnealingLR>(optim_mod, "CosineAnnealingLR")
        .def(py::init<optim::Optimizer*, int, float>(), py::arg("optimizer"), py::arg("T_max"), py::arg("eta_min") = 0.0f)
        .def("step", &optim::CosineAnnealingLR::step)
        .def_readwrite("T_max", &optim::CosineAnnealingLR::T_max)
        .def_readwrite("eta_min", &optim::CosineAnnealingLR::eta_min)
        .def_readwrite("last_epoch", &optim::CosineAnnealingLR::last_epoch);

    py::class_<optim::GradScaler>(optim_mod, "GradScaler")
        .def(py::init<float, float, float, int>(), py::arg("init_scale") = 65536.0f, py::arg("growth_factor") = 2.0f, py::arg("backoff_factor") = 0.5f, py::arg("growth_interval") = 2000)
        .def("scale_loss", &optim::GradScaler::scale_loss, py::arg("loss"))
        .def("step", &optim::GradScaler::step, py::arg("optimizer"))
        .def("update", &optim::GradScaler::update)
        .def_readwrite("scale", &optim::GradScaler::scale);

    auto amp_mod = m.def_submodule("amp", "LiteTorch Automatic Mixed Precision");
    
    py::class_<amp::GradScaler>(amp_mod, "GradScaler")
        .def(py::init<float, float, float, int>(), py::arg("init_scale") = 65536.0f, py::arg("growth_factor") = 2.0f, py::arg("backoff_factor") = 0.5f, py::arg("growth_interval") = 2000)
        .def("scale", &amp::GradScaler::scale, py::arg("loss"))
        .def("step", &amp::GradScaler::step, py::arg("optimizer"))
        .def("update", &amp::GradScaler::update)
        .def("get_scale", &amp::GradScaler::get_scale);

    py::class_<amp::AutocastGuard>(amp_mod, "AutocastGuard")
        .def(py::init<bool, DataType>(), py::arg("enabled") = true, py::arg("dtype") = DataType::FP16)
        .def_static("is_enabled", &amp::AutocastGuard::is_enabled)
        .def_static("get_dtype", &amp::AutocastGuard::get_dtype);

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
        .def("all_reduce", &distributed::ProcessGroup::all_reduce, py::arg("tensor"))
        .def("broadcast", &distributed::ProcessGroup::broadcast, py::arg("tensor"), py::arg("src"))
        .def("all_gather", &distributed::ProcessGroup::all_gather, py::arg("shard"), py::arg("full"))
        .def("reduce_scatter", &distributed::ProcessGroup::reduce_scatter, py::arg("shard"), py::arg("full"))
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
        .def_readwrite("val", &JITVar::val);

    py::class_<JITFunction, std::shared_ptr<JITFunction>>(jit_mod, "JITFunction")
        .def(py::init<const std::string&, std::shared_ptr<JITVar>, const std::vector<std::shared_ptr<JITVar>>&>(),
             py::arg("name"), py::arg("expr"), py::arg("inputs"))
        .def("__call__", &JITFunction::operator(), py::arg("args"))
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

    py::class_<data::DataLoader, std::shared_ptr<data::DataLoader>>(data_mod, "DataLoader")
        .def(py::init<std::shared_ptr<data::Dataset>, size_t, bool, const Device&, size_t>(),
             py::arg("dataset"), py::arg("batch_size"), py::arg("shuffle") = true, py::arg("device") = Device(DeviceType::CPU, 0), py::arg("prefetch_limit") = 2)
        .def("reset", &data::DataLoader::reset)
        .def("next", [](data::DataLoader& self) -> py::object {
            std::shared_ptr<Tensor> bx;
            std::shared_ptr<Tensor> by;
            if (self.next(bx, by)) {
                return py::make_tuple(bx, by);
            }
            return py::none();
        })
        .def("save_state", &data::DataLoader::save_state, py::arg("filepath"))
        .def("load_state", &data::DataLoader::load_state, py::arg("filepath"));

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

    m.def("empty_cache", []() {
        CachingAllocator::get().empty_cache();
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

    py::module_ jit_mod = m.def_submodule("jit");

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

    py::class_<JITFunction, std::shared_ptr<JITFunction>>(jit_mod, "JITFunction")
        .def(py::init<const std::string&, std::shared_ptr<JITVar>, const std::vector<std::shared_ptr<JITVar>>&>(),
             py::arg("name"), py::arg("expr"), py::arg("inputs"))
        .def("__call__", &JITFunction::operator(), py::arg("args"))
        .def("forward", &JITFunction::operator(), py::arg("args"))
        .def("save", &JITFunction::save, py::arg("filepath"))
        .def_static("load", &JITFunction::load, py::arg("filepath"));

    jit_mod.def("trace", [](py::object py_fn, const std::vector<std::shared_ptr<Tensor>>& inputs, const std::string& name) {
        auto func = [py_fn](const std::vector<std::shared_ptr<Tensor>>& args) -> std::shared_ptr<Tensor> {
            py::list py_args;
            for (auto& a : args) py_args.append(a);
            py::object res = py_fn(*py_args);
            return res.cast<std::shared_ptr<Tensor>>();
        };
        return Tracer::trace(inputs, func, name);
    }, py::arg("func"), py::arg("inputs"), py::arg("name") = "traced_fn");
}
