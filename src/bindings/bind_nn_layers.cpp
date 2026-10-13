#include "bindings_common.h"

void bind_nn_layers(py::module_& m) {
    auto nn_mod = m.attr("nn").cast<py::module_>();
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

}
