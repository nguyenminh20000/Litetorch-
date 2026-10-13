#include "bindings_common.h"

void bind_ops(py::module_& m) {
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
        ops.def("fp8_cast_scaled", &Ops::fp8_cast_scaled, py::arg("x"), py::arg("scale"), py::arg("dtype") = DataType::FP8_E4M3);
        ops.def("fp8_uncast_scaled", &Ops::fp8_uncast_scaled, py::arg("x"), py::arg("scale"));
        ops.def("fp8_matmul", &Ops::fp8_matmul, py::arg("a"), py::arg("b"), py::arg("a_scale"), py::arg("b_scale"), py::arg("out_scale"), py::arg("out_dtype") = DataType::FP8_E4M3);
        ops.def("int4_pack", &Ops::int4_pack, py::arg("x"));
        ops.def("int4_unpack", &Ops::int4_unpack, py::arg("packed"), py::arg("numel"));
        ops.def("int4_weight_only_linear", &Ops::int4_weight_only_linear, py::arg("x"), py::arg("packed_w"), py::arg("w_scale"), py::arg("bias") = nullptr);
        ops.def("moe_gate", [](std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> gate_weight, int top_k) {
            std::shared_ptr<Tensor> indices_out;
            auto probs = Ops::moe_gate(input, gate_weight, top_k, indices_out);
            return std::make_pair(probs, indices_out);
        });
        ops.def("flash_decoding", &Ops::flash_decoding);
        ops.def("ring_attention", &Ops::ring_attention);
        ops.def("checkpoint", &Ops::checkpoint, py::arg("module"), py::arg("input"));
        // === LT-ENG-022 fused MFU kernels ===
        ops.def("fused_swiglu", &Ops::fused_swiglu, py::arg("gate"), py::arg("up"));
        ops.def("fused_qkv_proj", &Ops::fused_qkv_proj, py::arg("x"), py::arg("wq"), py::arg("wk"), py::arg("wv"));
        // === end LT-ENG-022 ===
        // === W4-T3 (LT-ENG-028): bilinear, fold, unfold ===
        ops.def("bilinear", &Ops::bilinear, py::arg("input1"), py::arg("input2"), py::arg("weight"), py::arg("bias") = nullptr);
        ops.def("fold", &Ops::fold, py::arg("input"), py::arg("output_size"), py::arg("kernel_size"),
            py::arg("dilation"), py::arg("padding"), py::arg("stride"));
        ops.def("unfold", &Ops::unfold, py::arg("input"), py::arg("kernel_size"),
            py::arg("dilation"), py::arg("padding"), py::arg("stride"));
        // === end W4-T3 ===

        ops.def("conv1d", &Ops::conv1d, py::arg("input"), py::arg("weight"), py::arg("bias") = nullptr, py::arg("stride") = 1, py::arg("padding") = 0);
        ops.def("avg_pool2d", &Ops::avg_pool2d, py::arg("input"), py::arg("kernel_size"), py::arg("stride") = -1, py::arg("padding") = 0);
        ops.def("nll_loss", &Ops::nll_loss, py::arg("input"), py::arg("target"), py::arg("weight") = nullptr,
            py::arg("ignore_index") = -100, py::arg("reduction") = "mean");
        ops.def("bce_with_logits_loss", &Ops::bce_with_logits_loss, py::arg("input"), py::arg("target"),
            py::arg("weight") = nullptr, py::arg("reduction") = "mean", py::arg("pos_weight") = nullptr);

        m.def("checkpoint", &litetorch::checkpoint, "Activation Checkpointing helper");
}
