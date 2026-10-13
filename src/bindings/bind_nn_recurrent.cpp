#include "bindings_common.h"

void bind_nn_recurrent(py::module_& m) {
    auto nn_mod = m.attr("nn").cast<py::module_>();
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

}
