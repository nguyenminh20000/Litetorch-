#include "bindings_common.h"
#include "litetorch/generate.h"

void bind_nn_utils(py::module_& m) {
    auto nn_mod = m.attr("nn").cast<py::module_>();
        auto utils_mod = nn_mod.def_submodule("utils", "LiteTorch nn.utils");
        utils_mod.def("clip_grad_value_", &nn::utils::clip_grad_value_, py::arg("parameters"), py::arg("clip_value"));
        utils_mod.def("parameters_to_vector", &nn::utils::parameters_to_vector, py::arg("parameters"));
        utils_mod.def("vector_to_parameters", &nn::utils::vector_to_parameters, py::arg("vec"), py::arg("parameters"));
        utils_mod.def("weight_norm", &nn::utils::weight_norm, py::arg("v"), py::arg("g"), py::arg("dim") = 0);
        utils_mod.def("spectral_norm", &nn::utils::spectral_norm, py::arg("weight"), py::arg("n_power_iterations") = 1);
        // === end W3-T4 losses-misc ===

        py::class_<nn::GenerateConfig>(nn_mod, "GenerateConfig")
            .def(py::init<>())
            .def_readwrite("max_new_tokens", &nn::GenerateConfig::max_new_tokens)
            .def_readwrite("do_sample", &nn::GenerateConfig::do_sample)
            .def_readwrite("temperature", &nn::GenerateConfig::temperature)
            .def_readwrite("top_k", &nn::GenerateConfig::top_k)
            .def_readwrite("top_p", &nn::GenerateConfig::top_p)
            .def_readwrite("num_beams", &nn::GenerateConfig::num_beams)
            .def_readwrite("repetition_penalty", &nn::GenerateConfig::repetition_penalty)
            .def_readwrite("eos_token_id", &nn::GenerateConfig::eos_token_id)
            .def_readwrite("stop_token_ids", &nn::GenerateConfig::stop_token_ids)
            .def_readwrite("stop_sequences", &nn::GenerateConfig::stop_sequences)
            .def_readwrite("seed", &nn::GenerateConfig::seed);
        nn_mod.def("generate", &nn::generate, py::arg("model"), py::arg("prompt_tokens"),
                   py::arg("config") = nn::GenerateConfig());

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
        // === DPO (LT-ENG-039) ===
        functional_mod.def("dpo_loss", &functional::dpo_loss,
            py::arg("policy_chosen_logps"), py::arg("policy_rejected_logps"),
            py::arg("ref_chosen_logps") = nullptr, py::arg("ref_rejected_logps") = nullptr,
            py::arg("beta") = 0.1f, py::arg("reference_free") = false,
            py::arg("reduction") = "mean");
        // === end DPO ===
}
