#include "bindings_common.h"

void bind_nn(py::module_& m) {
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

        // === DT-A2 (LT-ENG-019) ===
        py::class_<nn::TPAttention, nn::Module, std::shared_ptr<nn::TPAttention>>(nn_mod, "TPAttention")
            .def(py::init<int, int, int, bool>(), py::arg("embed_dim"), py::arg("num_heads"),
                 py::arg("n_kv_heads") = -1, py::arg("has_bias") = true)
            .def("forward", &nn::TPAttention::forward)
            .def("parameters", &nn::TPAttention::parameters)
            .def("to", &nn::TPAttention::to)
            .def_readwrite("w_q", &nn::TPAttention::w_q)
            .def_readwrite("b_q", &nn::TPAttention::b_q)
            .def_readwrite("w_k", &nn::TPAttention::w_k)
            .def_readwrite("b_k", &nn::TPAttention::b_k)
            .def_readwrite("w_v", &nn::TPAttention::w_v)
            .def_readwrite("b_v", &nn::TPAttention::b_v)
            .def_readwrite("w_o", &nn::TPAttention::w_o)
            .def_readwrite("b_o", &nn::TPAttention::b_o);

        py::class_<nn::VocabParallelEmbedding, nn::Module, std::shared_ptr<nn::VocabParallelEmbedding>>(nn_mod, "VocabParallelEmbedding")
            .def(py::init<int, int, int64_t>(), py::arg("vocab_size"), py::arg("embed_dim"),
                 py::arg("padding_idx") = -1)
            .def("forward", &nn::VocabParallelEmbedding::forward)
            .def("logits", &nn::VocabParallelEmbedding::logits)
            .def("sharded_cross_entropy", &nn::VocabParallelEmbedding::sharded_cross_entropy)
            .def("parameters", &nn::VocabParallelEmbedding::parameters)
            .def("to", &nn::VocabParallelEmbedding::to)
            .def_readwrite("weight", &nn::VocabParallelEmbedding::weight);

        // === end DT-A2 ===

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

        // === W3-T4 losses-misc ===
        py::class_<nn::KLDivLoss, nn::Module, std::shared_ptr<nn::KLDivLoss>>(nn_mod, "KLDivLoss")
            .def(py::init<const std::string&>(), py::arg("reduction") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::KLDivLoss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::KLDivLoss::forward))
            .def("__call__", [](nn::KLDivLoss& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
                return self.forward(input, target);
            })
            .def_readwrite("reduction", &nn::KLDivLoss::reduction);

        py::class_<nn::SmoothL1Loss, nn::Module, std::shared_ptr<nn::SmoothL1Loss>>(nn_mod, "SmoothL1Loss")
            .def(py::init<float, const std::string&>(), py::arg("beta") = 1.0f, py::arg("reduction") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::SmoothL1Loss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::SmoothL1Loss::forward))
            .def("__call__", [](nn::SmoothL1Loss& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
                return self.forward(input, target);
            })
            .def_readwrite("beta", &nn::SmoothL1Loss::beta)
            .def_readwrite("reduction", &nn::SmoothL1Loss::reduction);

        py::class_<nn::HuberLoss, nn::Module, std::shared_ptr<nn::HuberLoss>>(nn_mod, "HuberLoss")
            .def(py::init<float, const std::string&>(), py::arg("delta") = 1.0f, py::arg("reduction") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::HuberLoss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::HuberLoss::forward))
            .def("__call__", [](nn::HuberLoss& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
                return self.forward(input, target);
            })
            .def_readwrite("delta", &nn::HuberLoss::delta)
            .def_readwrite("reduction", &nn::HuberLoss::reduction);

        py::class_<nn::CTCLoss, nn::Module, std::shared_ptr<nn::CTCLoss>>(nn_mod, "CTCLoss")
            .def(py::init<int, const std::string&, bool>(), py::arg("blank") = 0, py::arg("reduction") = "mean", py::arg("zero_infinity") = false)
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::CTCLoss::forward))
            .def("forward", [](nn::CTCLoss& self, std::shared_ptr<Tensor> log_probs, std::shared_ptr<Tensor> targets,
                               std::shared_ptr<Tensor> input_lengths, std::shared_ptr<Tensor> target_lengths) {
                return self.forward(log_probs, targets, input_lengths, target_lengths);
            }, py::arg("log_probs"), py::arg("targets"), py::arg("input_lengths"), py::arg("target_lengths"))
            .def("__call__", [](nn::CTCLoss& self, std::shared_ptr<Tensor> log_probs, std::shared_ptr<Tensor> targets,
                                std::shared_ptr<Tensor> input_lengths, std::shared_ptr<Tensor> target_lengths) {
                return self.forward(log_probs, targets, input_lengths, target_lengths);
            })
            .def_readwrite("blank", &nn::CTCLoss::blank)
            .def_readwrite("reduction", &nn::CTCLoss::reduction)
            .def_readwrite("zero_infinity", &nn::CTCLoss::zero_infinity);

        // === W3-T3 recurrent+ ===
        py::class_<nn::RNNCell, nn::Module, std::shared_ptr<nn::RNNCell>>(nn_mod, "RNNCell")
            .def(py::init<int, int, bool, const std::string&>(), py::arg("input_size"), py::arg("hidden_size"),
                 py::arg("bias") = true, py::arg("nonlinearity") = "tanh")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::RNNCell::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::RNNCell::forward),
                 py::arg("input"), py::arg("hx"))
            .def("__call__", [](nn::RNNCell& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> hx) {
                return hx ? self.forward(input, hx) : self.forward(input);
            }, py::arg("input"), py::arg("hx") = nullptr)
            .def("parameters", &nn::RNNCell::parameters)
            .def("to", &nn::RNNCell::to)
            .def_readwrite("weight_ih", &nn::RNNCell::weight_ih)
            .def_readwrite("weight_hh", &nn::RNNCell::weight_hh)
            .def_readwrite("bias_ih", &nn::RNNCell::bias_ih)
            .def_readwrite("bias_hh", &nn::RNNCell::bias_hh);

        py::class_<nn::LSTMCell, nn::Module, std::shared_ptr<nn::LSTMCell>>(nn_mod, "LSTMCell")
            .def(py::init<int, int, bool>(), py::arg("input_size"), py::arg("hidden_size"), py::arg("bias") = true)
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::LSTMCell::forward))
            .def("forward", [](nn::LSTMCell& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> hx,
                               std::shared_ptr<Tensor> cx) { return self.forward(input, hx, cx); },
                 py::arg("input"), py::arg("hx"), py::arg("cx"))
            .def("parameters", &nn::LSTMCell::parameters)
            .def("to", &nn::LSTMCell::to)
            .def_readwrite("weight_ih", &nn::LSTMCell::weight_ih)
            .def_readwrite("weight_hh", &nn::LSTMCell::weight_hh)
            .def_readwrite("bias_ih", &nn::LSTMCell::bias_ih)
            .def_readwrite("bias_hh", &nn::LSTMCell::bias_hh);

        py::class_<nn::GRUCell, nn::Module, std::shared_ptr<nn::GRUCell>>(nn_mod, "GRUCell")
            .def(py::init<int, int, bool>(), py::arg("input_size"), py::arg("hidden_size"), py::arg("bias") = true)
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::GRUCell::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::GRUCell::forward),
                 py::arg("input"), py::arg("hx"))
            .def("__call__", [](nn::GRUCell& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> hx) {
                return hx ? self.forward(input, hx) : self.forward(input);
            }, py::arg("input"), py::arg("hx") = nullptr)
            .def("parameters", &nn::GRUCell::parameters)
            .def("to", &nn::GRUCell::to)
            .def_readwrite("weight_ih", &nn::GRUCell::weight_ih)
            .def_readwrite("weight_hh", &nn::GRUCell::weight_hh)
            .def_readwrite("bias_ih", &nn::GRUCell::bias_ih)
            .def_readwrite("bias_hh", &nn::GRUCell::bias_hh);

        py::class_<nn::RNN, nn::Module, std::shared_ptr<nn::RNN>>(nn_mod, "RNN")
            .def(py::init<int, int, int, const std::string&, bool, bool, float, bool>(), py::arg("input_size"),
                 py::arg("hidden_size"), py::arg("num_layers") = 1, py::arg("nonlinearity") = "tanh",
                 py::arg("bias") = true, py::arg("batch_first") = false, py::arg("dropout") = 0.0f,
                 py::arg("bidirectional") = false)
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::RNN::forward))
            .def("forward_full", [](nn::RNN& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> h0) {
                auto r = self.forward_full(input, h0);
                return std::make_pair(r.first, r.second);
            }, py::arg("input"), py::arg("h0") = nullptr)
            .def("parameters", &nn::RNN::parameters)
            .def("to", &nn::RNN::to)
            .def_readwrite("cells", &nn::RNN::cells)
            .def("__call__", [](nn::RNN& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> h0) {
                auto r = self.forward_full(input, h0);
                return std::make_pair(r.first, r.second);
            }, py::arg("input"), py::arg("hx") = nullptr);

        py::class_<nn::LSTM, nn::Module, std::shared_ptr<nn::LSTM>>(nn_mod, "LSTM")
            .def(py::init<int, int, int, bool, bool, float, bool>(), py::arg("input_size"),
                 py::arg("hidden_size"), py::arg("num_layers") = 1, py::arg("bias") = true,
                 py::arg("batch_first") = false, py::arg("dropout") = 0.0f, py::arg("bidirectional") = false)
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::LSTM::forward))
            .def("forward_full", [](nn::LSTM& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> h0,
                                    std::shared_ptr<Tensor> c0) {
                auto r = self.forward_full(input, h0, c0);
                return std::make_tuple(r.first, r.second.first, r.second.second);
            }, py::arg("input"), py::arg("h0") = nullptr, py::arg("c0") = nullptr)
            .def("parameters", &nn::LSTM::parameters)
            .def("to", &nn::LSTM::to)
            .def_readwrite("cells", &nn::LSTM::cells)
            .def("__call__", [](nn::LSTM& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> h0,
                                std::shared_ptr<Tensor> c0) {
                auto r = self.forward_full(input, h0, c0);
                return std::make_tuple(r.first, std::make_pair(r.second.first, r.second.second));
            }, py::arg("input"), py::arg("hx") = nullptr, py::arg("cx") = nullptr);

        py::class_<nn::GRU, nn::Module, std::shared_ptr<nn::GRU>>(nn_mod, "GRU")
            .def(py::init<int, int, int, bool, bool, float, bool>(), py::arg("input_size"),
                 py::arg("hidden_size"), py::arg("num_layers") = 1, py::arg("bias") = true,
                 py::arg("batch_first") = false, py::arg("dropout") = 0.0f, py::arg("bidirectional") = false)
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::GRU::forward))
            .def("forward_full", [](nn::GRU& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> h0) {
                auto r = self.forward_full(input, h0);
                return std::make_pair(r.first, r.second);
            }, py::arg("input"), py::arg("h0") = nullptr)
            .def("parameters", &nn::GRU::parameters)
            .def("to", &nn::GRU::to)
            .def_readwrite("cells", &nn::GRU::cells)
            .def("__call__", [](nn::GRU& self, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> h0) {
                auto r = self.forward_full(input, h0);
                return std::make_pair(r.first, r.second);
            }, py::arg("input"), py::arg("hx") = nullptr);

        py::class_<nn::TransformerEncoderLayer, nn::Module, std::shared_ptr<nn::TransformerEncoderLayer>>(
            nn_mod, "TransformerEncoderLayer")
            .def(py::init<int, int, int, float, const std::string&, bool>(), py::arg("d_model"),
                 py::arg("nhead"), py::arg("dim_feedforward") = 2048, py::arg("dropout") = 0.1f,
                 py::arg("activation") = "relu", py::arg("batch_first") = false)
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::TransformerEncoderLayer::forward))
            .def("forward", [](nn::TransformerEncoderLayer& self, std::shared_ptr<Tensor> src,
                               std::shared_ptr<Tensor> mask, std::shared_ptr<Tensor> key_mask) {
                return self.forward(src, mask, key_mask);
            }, py::arg("src"), py::arg("mask") = nullptr, py::arg("src_key_padding_mask") = nullptr)
            .def("parameters", &nn::TransformerEncoderLayer::parameters)
            .def("to", &nn::TransformerEncoderLayer::to);

        py::class_<nn::TransformerEncoder, nn::Module, std::shared_ptr<nn::TransformerEncoder>>(
            nn_mod, "TransformerEncoder")
            .def(py::init<std::shared_ptr<nn::TransformerEncoderLayer>, int, std::shared_ptr<nn::LayerNorm>>(),
                 py::arg("encoder_layer"), py::arg("num_layers"), py::arg("norm") = nullptr)
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::TransformerEncoder::forward))
            .def("forward", [](nn::TransformerEncoder& self, std::shared_ptr<Tensor> src,
                               std::shared_ptr<Tensor> mask, std::shared_ptr<Tensor> key_mask) {
                return self.forward(src, mask, key_mask);
            }, py::arg("src"), py::arg("mask") = nullptr, py::arg("src_key_padding_mask") = nullptr)
            .def("parameters", &nn::TransformerEncoder::parameters)
            .def("to", &nn::TransformerEncoder::to);

        py::class_<nn::EmbeddingBag, nn::Module, std::shared_ptr<nn::EmbeddingBag>>(nn_mod, "EmbeddingBag")
            .def(py::init<int, int, const std::string&>(), py::arg("num_embeddings"),
                 py::arg("embedding_dim"), py::arg("mode") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::EmbeddingBag::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(
                     &nn::EmbeddingBag::forward),
                 py::arg("input"), py::arg("offsets"))
            .def("parameters", &nn::EmbeddingBag::parameters)
            .def("to", &nn::EmbeddingBag::to)
            .def_readwrite("weight", &nn::EmbeddingBag::weight);

        py::class_<nn::ConvTranspose1d, nn::Module, std::shared_ptr<nn::ConvTranspose1d>>(nn_mod, "ConvTranspose1d")
            .def(py::init<int, int, int, int, int, int, int, bool, int>(), py::arg("in_channels"),
                 py::arg("out_channels"), py::arg("kernel_size"), py::arg("stride") = 1,
                 py::arg("padding") = 0, py::arg("output_padding") = 0, py::arg("groups") = 1,
                 py::arg("bias") = true, py::arg("dilation") = 1)
            .def("forward", &nn::ConvTranspose1d::forward)
            .def("parameters", &nn::ConvTranspose1d::parameters)
            .def("to", &nn::ConvTranspose1d::to)
            .def_readwrite("weight", &nn::ConvTranspose1d::weight)
            .def_readwrite("bias", &nn::ConvTranspose1d::bias);

        py::class_<nn::ConvTranspose2d, nn::Module, std::shared_ptr<nn::ConvTranspose2d>>(nn_mod, "ConvTranspose2d")
            .def(py::init<int, int, int, int, int, int, int, bool, int>(), py::arg("in_channels"),
                 py::arg("out_channels"), py::arg("kernel_size"), py::arg("stride") = 1,
                 py::arg("padding") = 0, py::arg("output_padding") = 0, py::arg("groups") = 1,
                 py::arg("bias") = true, py::arg("dilation") = 1)
            .def("forward", &nn::ConvTranspose2d::forward)
            .def("parameters", &nn::ConvTranspose2d::parameters)
            .def("to", &nn::ConvTranspose2d::to)
            .def_readwrite("weight", &nn::ConvTranspose2d::weight)
            .def_readwrite("bias", &nn::ConvTranspose2d::bias);

        py::class_<nn::ConvTranspose3d, nn::Module, std::shared_ptr<nn::ConvTranspose3d>>(nn_mod, "ConvTranspose3d")
            .def(py::init<int, int, int, int, int, int, int, bool, int>(), py::arg("in_channels"),
                 py::arg("out_channels"), py::arg("kernel_size"), py::arg("stride") = 1,
                 py::arg("padding") = 0, py::arg("output_padding") = 0, py::arg("groups") = 1,
                 py::arg("bias") = true, py::arg("dilation") = 1)
            .def("forward", &nn::ConvTranspose3d::forward)
            .def("parameters", &nn::ConvTranspose3d::parameters)
            .def("to", &nn::ConvTranspose3d::to)
            .def_readwrite("weight", &nn::ConvTranspose3d::weight)
            .def_readwrite("bias", &nn::ConvTranspose3d::bias);

        py::class_<nn::InstanceNorm1d, nn::Module, std::shared_ptr<nn::InstanceNorm1d>>(nn_mod, "InstanceNorm1d")
            .def(py::init<int, float, float, bool, bool>(), py::arg("num_features"), py::arg("eps") = 1e-5f,
                 py::arg("momentum") = 0.1f, py::arg("affine") = false, py::arg("track_running_stats") = false)
            .def("forward", &nn::InstanceNorm1d::forward)
            .def("parameters", &nn::InstanceNorm1d::parameters)
            .def("to", &nn::InstanceNorm1d::to)
            .def_readwrite("weight", &nn::InstanceNorm1d::weight)
            .def_readwrite("bias", &nn::InstanceNorm1d::bias);

        py::class_<nn::InstanceNorm2d, nn::Module, std::shared_ptr<nn::InstanceNorm2d>>(nn_mod, "InstanceNorm2d")
            .def(py::init<int, float, float, bool, bool>(), py::arg("num_features"), py::arg("eps") = 1e-5f,
                 py::arg("momentum") = 0.1f, py::arg("affine") = false, py::arg("track_running_stats") = false)
            .def("forward", &nn::InstanceNorm2d::forward)
            .def("parameters", &nn::InstanceNorm2d::parameters)
            .def("to", &nn::InstanceNorm2d::to)
            .def_readwrite("weight", &nn::InstanceNorm2d::weight)
            .def_readwrite("bias", &nn::InstanceNorm2d::bias);

        py::class_<nn::InstanceNorm3d, nn::Module, std::shared_ptr<nn::InstanceNorm3d>>(nn_mod, "InstanceNorm3d")
            .def(py::init<int, float, float, bool, bool>(), py::arg("num_features"), py::arg("eps") = 1e-5f,
                 py::arg("momentum") = 0.1f, py::arg("affine") = false, py::arg("track_running_stats") = false)
            .def("forward", &nn::InstanceNorm3d::forward)
            .def("parameters", &nn::InstanceNorm3d::parameters)
            .def("to", &nn::InstanceNorm3d::to)
            .def_readwrite("weight", &nn::InstanceNorm3d::weight)
            .def_readwrite("bias", &nn::InstanceNorm3d::bias);

        py::class_<nn::BatchNorm3d, nn::Module, std::shared_ptr<nn::BatchNorm3d>>(nn_mod, "BatchNorm3d")
            .def(py::init<int, float, float, bool, bool>(), py::arg("num_features"), py::arg("eps") = 1e-5f,
                 py::arg("momentum") = 0.1f, py::arg("affine") = true, py::arg("track_running_stats") = true)
            .def("forward", &nn::BatchNorm3d::forward)
            .def("parameters", &nn::BatchNorm3d::parameters)
            .def("to", &nn::BatchNorm3d::to)
            .def_readwrite("weight", &nn::BatchNorm3d::weight)
            .def_readwrite("bias", &nn::BatchNorm3d::bias)
            .def_readwrite("running_mean", &nn::BatchNorm3d::running_mean)
            .def_readwrite("running_var", &nn::BatchNorm3d::running_var);

        py::class_<nn::SyncBatchNorm, nn::Module, std::shared_ptr<nn::SyncBatchNorm>>(nn_mod, "SyncBatchNorm")
            .def(py::init<int, float, float, bool, bool>(), py::arg("num_features"), py::arg("eps") = 1e-5f,
                 py::arg("momentum") = 0.1f, py::arg("affine") = true, py::arg("track_running_stats") = true)
            .def("forward", &nn::SyncBatchNorm::forward)
            .def("parameters", &nn::SyncBatchNorm::parameters)
            .def("to", &nn::SyncBatchNorm::to)
            .def_readwrite("weight", &nn::SyncBatchNorm::weight)
            .def_readwrite("bias", &nn::SyncBatchNorm::bias);

        py::class_<nn::ELU, nn::Module, std::shared_ptr<nn::ELU>>(nn_mod, "ELU")
            .def(py::init<float>(), py::arg("alpha") = 1.0f)
            .def("forward", &nn::ELU::forward)
            .def_readwrite("alpha", &nn::ELU::alpha);

        py::class_<nn::PReLU, nn::Module, std::shared_ptr<nn::PReLU>>(nn_mod, "PReLU")
            .def(py::init<int, float>(), py::arg("num_parameters") = 1, py::arg("init") = 0.25f)
            .def("forward", &nn::PReLU::forward)
            .def("parameters", &nn::PReLU::parameters)
            .def("to", &nn::PReLU::to)
            .def_readwrite("weight", &nn::PReLU::weight);

        py::class_<nn::GLU, nn::Module, std::shared_ptr<nn::GLU>>(nn_mod, "GLU")
            .def(py::init<int64_t>(), py::arg("dim") = -1)
            .def("forward", &nn::GLU::forward)
            .def_readwrite("dim", &nn::GLU::dim);

        py::class_<nn::PixelShuffle, nn::Module, std::shared_ptr<nn::PixelShuffle>>(nn_mod, "PixelShuffle")
            .def(py::init<int>(), py::arg("upscale_factor"))
            .def("forward", &nn::PixelShuffle::forward)
            .def_readwrite("upscale_factor", &nn::PixelShuffle::upscale_factor);

        py::class_<nn::PixelUnshuffle, nn::Module, std::shared_ptr<nn::PixelUnshuffle>>(nn_mod, "PixelUnshuffle")
            .def(py::init<int>(), py::arg("downscale_factor"))
            .def("forward", &nn::PixelUnshuffle::forward)
            .def_readwrite("downscale_factor", &nn::PixelUnshuffle::downscale_factor);

        py::class_<nn::Identity, nn::Module, std::shared_ptr<nn::Identity>>(nn_mod, "Identity")
            .def(py::init<>())
            .def("forward", &nn::Identity::forward);

        py::class_<nn::Unflatten, nn::Module, std::shared_ptr<nn::Unflatten>>(nn_mod, "Unflatten")
            .def(py::init<int64_t, const std::vector<int64_t>&>(), py::arg("dim"), py::arg("unflattened_size"))
            .def("forward", &nn::Unflatten::forward)
            .def_readwrite("dim", &nn::Unflatten::dim)
            .def_readwrite("unflattened_size", &nn::Unflatten::unflattened_size);
        // === end W3-T3 ===

        auto utils_mod = nn_mod.def_submodule("utils", "LiteTorch nn.utils");
        utils_mod.def("clip_grad_value_", &nn::utils::clip_grad_value_, py::arg("parameters"), py::arg("clip_value"));
        utils_mod.def("parameters_to_vector", &nn::utils::parameters_to_vector, py::arg("parameters"));
        utils_mod.def("vector_to_parameters", &nn::utils::vector_to_parameters, py::arg("vec"), py::arg("parameters"));
        utils_mod.def("weight_norm", &nn::utils::weight_norm, py::arg("v"), py::arg("g"), py::arg("dim") = 0);
        utils_mod.def("spectral_norm", &nn::utils::spectral_norm, py::arg("weight"), py::arg("n_power_iterations") = 1);
        // === end W3-T4 losses-misc ===

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
        // === W3-T4 losses-misc ===
        init_mod.def("trunc_normal_", &nn::init::trunc_normal_, py::arg("tensor"), py::arg("mean") = 0.0f,
            py::arg("std") = 1.0f, py::arg("a") = -2.0f, py::arg("b") = 2.0f);
        init_mod.def("orthogonal_", &nn::init::orthogonal_, py::arg("tensor"), py::arg("gain") = 1.0f);
        // === end W3-T4 losses-misc ===

        // === W4-T3 (LT-ENG-028): extra activations ===
        py::class_<nn::ReLU6, nn::Module, std::shared_ptr<nn::ReLU6>>(nn_mod, "ReLU6")
            .def(py::init<>())
            .def("forward", &nn::ReLU6::forward);
        py::class_<nn::RReLU, nn::Module, std::shared_ptr<nn::RReLU>>(nn_mod, "RReLU")
            .def(py::init<float, float>(), py::arg("lower") = 0.125f, py::arg("upper") = 0.3333333333333333f)
            .def("forward", &nn::RReLU::forward)
            .def_readwrite("lower", &nn::RReLU::lower)
            .def_readwrite("upper", &nn::RReLU::upper);
        py::class_<nn::CELU, nn::Module, std::shared_ptr<nn::CELU>>(nn_mod, "CELU")
            .def(py::init<float>(), py::arg("alpha") = 1.0f)
            .def("forward", &nn::CELU::forward)
            .def_readwrite("alpha", &nn::CELU::alpha);
        py::class_<nn::SELU, nn::Module, std::shared_ptr<nn::SELU>>(nn_mod, "SELU")
            .def(py::init<>())
            .def("forward", &nn::SELU::forward);
        py::class_<nn::Mish, nn::Module, std::shared_ptr<nn::Mish>>(nn_mod, "Mish")
            .def(py::init<>())
            .def("forward", &nn::Mish::forward);
        py::class_<nn::Softplus, nn::Module, std::shared_ptr<nn::Softplus>>(nn_mod, "Softplus")
            .def(py::init<float, float>(), py::arg("beta") = 1.0f, py::arg("threshold") = 20.0f)
            .def("forward", &nn::Softplus::forward)
            .def_readwrite("beta", &nn::Softplus::beta)
            .def_readwrite("threshold", &nn::Softplus::threshold);
        py::class_<nn::Softshrink, nn::Module, std::shared_ptr<nn::Softshrink>>(nn_mod, "Softshrink")
            .def(py::init<float>(), py::arg("lambd") = 0.5f)
            .def("forward", &nn::Softshrink::forward)
            .def_readwrite("lambd", &nn::Softshrink::lambd);
        py::class_<nn::Hardshrink, nn::Module, std::shared_ptr<nn::Hardshrink>>(nn_mod, "Hardshrink")
            .def(py::init<float>(), py::arg("lambd") = 0.5f)
            .def("forward", &nn::Hardshrink::forward)
            .def_readwrite("lambd", &nn::Hardshrink::lambd);
        py::class_<nn::Tanhshrink, nn::Module, std::shared_ptr<nn::Tanhshrink>>(nn_mod, "Tanhshrink")
            .def(py::init<>())
            .def("forward", &nn::Tanhshrink::forward);
        py::class_<nn::Softsign, nn::Module, std::shared_ptr<nn::Softsign>>(nn_mod, "Softsign")
            .def(py::init<>())
            .def("forward", &nn::Softsign::forward);
        py::class_<nn::Hardtanh, nn::Module, std::shared_ptr<nn::Hardtanh>>(nn_mod, "Hardtanh")
            .def(py::init<float, float>(), py::arg("min_val") = -1.0f, py::arg("max_val") = 1.0f)
            .def("forward", &nn::Hardtanh::forward)
            .def_readwrite("min_val", &nn::Hardtanh::min_val)
            .def_readwrite("max_val", &nn::Hardtanh::max_val);
        py::class_<nn::Threshold, nn::Module, std::shared_ptr<nn::Threshold>>(nn_mod, "Threshold")
            .def(py::init<float, float>(), py::arg("threshold"), py::arg("value"))
            .def("forward", &nn::Threshold::forward)
            .def_readwrite("threshold", &nn::Threshold::threshold)
            .def_readwrite("value", &nn::Threshold::value);
        // === end W4-T3 activations ===

        // === W4-T3 (LT-ENG-028): extra losses ===
        py::class_<nn::HingeEmbeddingLoss, nn::Module, std::shared_ptr<nn::HingeEmbeddingLoss>>(nn_mod, "HingeEmbeddingLoss")
            .def(py::init<float, const std::string&>(), py::arg("margin") = 1.0f, py::arg("reduction") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::HingeEmbeddingLoss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::HingeEmbeddingLoss::forward))
            .def_readwrite("margin", &nn::HingeEmbeddingLoss::margin)
            .def_readwrite("reduction", &nn::HingeEmbeddingLoss::reduction);
        py::class_<nn::CosineEmbeddingLoss, nn::Module, std::shared_ptr<nn::CosineEmbeddingLoss>>(nn_mod, "CosineEmbeddingLoss")
            .def(py::init<float, const std::string&>(), py::arg("margin") = 0.0f, py::arg("reduction") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::CosineEmbeddingLoss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::CosineEmbeddingLoss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::CosineEmbeddingLoss::forward))
            .def_readwrite("margin", &nn::CosineEmbeddingLoss::margin)
            .def_readwrite("reduction", &nn::CosineEmbeddingLoss::reduction);
        py::class_<nn::MarginRankingLoss, nn::Module, std::shared_ptr<nn::MarginRankingLoss>>(nn_mod, "MarginRankingLoss")
            .def(py::init<float, const std::string&>(), py::arg("margin") = 0.0f, py::arg("reduction") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::MarginRankingLoss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::MarginRankingLoss::forward))
            .def_readwrite("margin", &nn::MarginRankingLoss::margin)
            .def_readwrite("reduction", &nn::MarginRankingLoss::reduction);
        py::class_<nn::MultiMarginLoss, nn::Module, std::shared_ptr<nn::MultiMarginLoss>>(nn_mod, "MultiMarginLoss")
            .def(py::init<int, float, const std::string&>(), py::arg("p") = 1, py::arg("margin") = 1.0f, py::arg("reduction") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::MultiMarginLoss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::MultiMarginLoss::forward))
            .def_readwrite("p", &nn::MultiMarginLoss::p)
            .def_readwrite("margin", &nn::MultiMarginLoss::margin)
            .def_readwrite("reduction", &nn::MultiMarginLoss::reduction);
        py::class_<nn::MultiLabelMarginLoss, nn::Module, std::shared_ptr<nn::MultiLabelMarginLoss>>(nn_mod, "MultiLabelMarginLoss")
            .def(py::init<const std::string&>(), py::arg("reduction") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::MultiLabelMarginLoss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::MultiLabelMarginLoss::forward))
            .def_readwrite("reduction", &nn::MultiLabelMarginLoss::reduction);
        py::class_<nn::SoftMarginLoss, nn::Module, std::shared_ptr<nn::SoftMarginLoss>>(nn_mod, "SoftMarginLoss")
            .def(py::init<const std::string&>(), py::arg("reduction") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::SoftMarginLoss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::SoftMarginLoss::forward))
            .def_readwrite("reduction", &nn::SoftMarginLoss::reduction);
        py::class_<nn::MultiLabelSoftMarginLoss, nn::Module, std::shared_ptr<nn::MultiLabelSoftMarginLoss>>(nn_mod, "MultiLabelSoftMarginLoss")
            .def(py::init<const std::string&>(), py::arg("reduction") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::MultiLabelSoftMarginLoss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::MultiLabelSoftMarginLoss::forward))
            .def_readwrite("reduction", &nn::MultiLabelSoftMarginLoss::reduction);
        py::class_<nn::TripletMarginLoss, nn::Module, std::shared_ptr<nn::TripletMarginLoss>>(nn_mod, "TripletMarginLoss")
            .def(py::init<float, int, float, bool, const std::string&>(), py::arg("margin") = 1.0f, py::arg("p") = 2, py::arg("eps") = 1e-6f, py::arg("swap") = false, py::arg("reduction") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::TripletMarginLoss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::TripletMarginLoss::forward))
            .def_readwrite("margin", &nn::TripletMarginLoss::margin)
            .def_readwrite("reduction", &nn::TripletMarginLoss::reduction);
        py::class_<nn::TripletMarginWithDistanceLoss, nn::Module, std::shared_ptr<nn::TripletMarginWithDistanceLoss>>(nn_mod, "TripletMarginWithDistanceLoss")
            .def(py::init<float, const std::string&>(), py::arg("margin") = 1.0f, py::arg("reduction") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::TripletMarginWithDistanceLoss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::TripletMarginWithDistanceLoss::forward))
            .def_readwrite("margin", &nn::TripletMarginWithDistanceLoss::margin)
            .def_readwrite("reduction", &nn::TripletMarginWithDistanceLoss::reduction);
        py::class_<nn::PoissonNLLLoss, nn::Module, std::shared_ptr<nn::PoissonNLLLoss>>(nn_mod, "PoissonNLLLoss")
            .def(py::init<bool, bool, float, const std::string&>(), py::arg("log_input") = true, py::arg("full") = false, py::arg("eps") = 1e-8f, py::arg("reduction") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::PoissonNLLLoss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::PoissonNLLLoss::forward))
            .def_readwrite("reduction", &nn::PoissonNLLLoss::reduction);
        py::class_<nn::GaussianNLLLoss, nn::Module, std::shared_ptr<nn::GaussianNLLLoss>>(nn_mod, "GaussianNLLLoss")
            .def(py::init<bool, float, const std::string&>(), py::arg("full") = false, py::arg("eps") = 1e-6f, py::arg("reduction") = "mean")
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::GaussianNLLLoss::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::GaussianNLLLoss::forward))
            .def_readwrite("reduction", &nn::GaussianNLLLoss::reduction);
        // === end W4-T3 losses ===

        // === W4-T3 (LT-ENG-028): Bilinear, Fold, Unfold ===
        py::class_<nn::Bilinear, nn::Module, std::shared_ptr<nn::Bilinear>>(nn_mod, "Bilinear")
            .def(py::init<int, int, int, bool>(), py::arg("in1_features"), py::arg("in2_features"), py::arg("out_features"), py::arg("bias") = true)
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>>(&nn::Bilinear::forward))
            .def("forward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&nn::Bilinear::forward));
        py::class_<nn::Fold, nn::Module, std::shared_ptr<nn::Fold>>(nn_mod, "Fold")
            .def(py::init<const std::vector<int64_t>&, const std::vector<int64_t>&, const std::vector<int64_t>&, const std::vector<int64_t>&, const std::vector<int64_t>&>(),
                 py::arg("output_size"), py::arg("kernel_size"), py::arg("dilation") = std::vector<int64_t>{1, 1},
                 py::arg("padding") = std::vector<int64_t>{0, 0}, py::arg("stride") = std::vector<int64_t>{1, 1})
            .def("forward", &nn::Fold::forward);
        py::class_<nn::Unfold, nn::Module, std::shared_ptr<nn::Unfold>>(nn_mod, "Unfold")
            .def(py::init<const std::vector<int64_t>&, const std::vector<int64_t>&, const std::vector<int64_t>&, const std::vector<int64_t>&>(),
                 py::arg("kernel_size"), py::arg("dilation") = std::vector<int64_t>{1, 1},
                 py::arg("padding") = std::vector<int64_t>{0, 0}, py::arg("stride") = std::vector<int64_t>{1, 1})
            .def("forward", &nn::Unfold::forward);
        // === end W4-T3 misc ===

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
}
