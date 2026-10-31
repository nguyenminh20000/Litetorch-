#include "bindings_common.h"

void bind_nn_base(py::module_& m) {
    auto nn_mod = m.attr("nn").cast<py::module_>();

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
            .def("__call__", [](std::shared_ptr<nn::MultiHeadAttention> self, std::shared_ptr<Tensor> input) {
                return self->forward(input);
            }, py::arg("input"))
            .def("__call__", [](std::shared_ptr<nn::MultiHeadAttention> self, std::shared_ptr<Tensor> query,
                               std::shared_ptr<Tensor> key, std::shared_ptr<Tensor> value) {
                return self->forward(query, key, value);
            }, py::arg("query"), py::arg("key"), py::arg("value"))
            .def("__call__", [](std::shared_ptr<nn::MultiHeadAttention> self, std::shared_ptr<Tensor> query,
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
            .def("__call__", [](nn::TransformerDecoderLayer& self, std::shared_ptr<Tensor> input) {
                return self.forward(input);
            }, py::arg("input"))
            .def("__call__", [](nn::TransformerDecoderLayer& self, std::shared_ptr<Tensor> tgt, std::shared_ptr<Tensor> memory) {
                return self.forward(tgt, memory);
            }, py::arg("tgt"), py::arg("memory") = nullptr)
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

}
