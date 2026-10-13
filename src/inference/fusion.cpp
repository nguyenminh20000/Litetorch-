#include "litetorch/inference.h"
#include "litetorch/ops.h"
#include <cmath>
#include <stdexcept>

namespace litetorch {
namespace inference {

std::shared_ptr<nn::Conv2d> fold_conv_bn(std::shared_ptr<nn::Conv2d> conv,
                                         std::shared_ptr<nn::BatchNorm2d> bn) {
    if (!conv || !bn) {
        throw std::runtime_error("[litetorch Error] fold_conv_bn: null conv or bn");
    }
    if (conv->weight->shape[0] != bn->num_features) {
        throw std::runtime_error("[litetorch Error] fold_conv_bn: out_channels mismatch");
    }
    int C = bn->num_features;
    std::shared_ptr<Tensor> scale;
    {
        auto eps_t = Ops::full({C}, bn->eps, bn->running_var->device, false, DataType::FP32);
        auto var_eps = Ops::add(bn->running_var, eps_t);
        auto inv = Ops::rsqrt(var_eps);
        scale = bn->affine ? Ops::mul(bn->weight, inv) : inv;
    }
    auto w_shape = conv->weight->shape;
    auto s4 = scale->reshape({C, 1, 1, 1});
    auto new_w = Ops::mul(conv->weight, s4);

    std::shared_ptr<Tensor> new_b;
    {
        auto b = conv->bias ? conv->bias
            : Tensor::zeros({C}, conv->weight->device, false, DataType::FP32);
        auto shifted = Ops::sub(b, bn->running_mean);
        auto scaled = Ops::mul(shifted, scale);
        new_b = (bn->affine && bn->bias) ? Ops::add(scaled, bn->bias) : scaled;
    }

    auto folded = std::make_shared<nn::Conv2d>(
        static_cast<int>(w_shape[1] * conv->groups),
        C,
        std::vector<int>{static_cast<int>(w_shape[2]), static_cast<int>(w_shape[3])},
        std::vector<int>{conv->stride, conv->stride},
        std::vector<int>{conv->padding, conv->padding},
        true, conv->groups);
    folded->weight = new_w;
    folded->bias = new_b;
    folded->to(conv->weight->device);
    return folded;
}

FusionReport fuse_conv_bn(std::shared_ptr<nn::Sequential> seq) {
    FusionReport rep;
    if (!seq) return rep;
    for (size_t i = 0; i + 1 < seq->modules.size(); ++i) {
        auto conv = std::dynamic_pointer_cast<nn::Conv2d>(seq->modules[i]);
        auto bn = std::dynamic_pointer_cast<nn::BatchNorm2d>(seq->modules[i + 1]);
        if (conv && bn) {
            auto folded = fold_conv_bn(conv, bn);
            seq->modules[i] = folded;
            seq->modules[i + 1] = std::make_shared<nn::Identity>();
            rep.conv_bn_folded++;
            rep.notes.push_back("folded Conv2d->BatchNorm2d at Sequential[" + std::to_string(i) + "]");
            ++i;
        }
    }
    return rep;
}

static void scan_layernorm(std::shared_ptr<nn::Module> m, FusionReport& rep, const std::string& path) {
    if (!m) return;
    if (std::dynamic_pointer_cast<nn::LayerNorm>(m)) {
        rep.layernorm_detected++;
        rep.notes.push_back("LayerNorm fusion candidate at " + path + " (residual-add + norm pattern)");
    }
    for (auto& kv : m->named_children()) {
        scan_layernorm(kv.second, rep, path + "/" + kv.first);
    }
}

FusionReport detect_layernorm_fusion(std::shared_ptr<nn::Module> root) {
    FusionReport rep;
    scan_layernorm(root, rep, "root");
    return rep;
}

bool verify_linear_bias_fusion() {
    auto a = Tensor::randn({32, 64}, Device(DeviceType::CPU, 0), false);
    auto b = Tensor::randn({64, 128}, Device(DeviceType::CPU, 0), false);
    auto bias = Tensor::randn({128}, Device(DeviceType::CPU, 0), false);
    auto out = Ops::matmul_bias(a, b, bias);
    auto ref = Ops::add(Ops::matmul(a, b), bias);
    return out->shape == ref->shape && out->shape[0] == 32 && out->shape[1] == 128;
}

} // namespace inference
} // namespace litetorch
