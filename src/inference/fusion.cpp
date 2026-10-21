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
    if (bn->training) {
        throw std::runtime_error("[litetorch Error] fold_conv_bn: BatchNorm2d must be in eval mode; folding training-mode batch statistics is mathematically wrong (call eval() first)");
    }
    if (!bn->running_mean || !bn->running_var) {
        throw std::runtime_error("[litetorch Error] fold_conv_bn: BatchNorm2d running_mean/running_var is null (untrained BN or track_running_stats=false); cannot fold");
    }
    if (!conv->weight) {
        throw std::runtime_error("[litetorch Error] fold_conv_bn: conv weight is null");
    }
    if (bn->affine && !bn->weight) {
        throw std::runtime_error("[litetorch Error] fold_conv_bn: BatchNorm2d affine=true but weight (gamma) is null");
    }
    if (conv->weight->shape[0] != bn->num_features) {
        throw std::runtime_error("[litetorch Error] fold_conv_bn: out_channels mismatch");
    }
    const Device& wdev = conv->weight->device;
    if (bn->running_mean->device != wdev || bn->running_var->device != wdev ||
        (bn->weight && bn->weight->device != wdev) ||
        (bn->bias && bn->bias->device != wdev)) {
        throw std::runtime_error("[litetorch Error] fold_conv_bn: conv weight and BatchNorm2d tensors must be on the same device; move them to a common device first");
    }
    if (conv->weight->dtype != DataType::FP32) {
        throw std::runtime_error("[litetorch Error] fold_conv_bn: conv weight must be FP32; mixed-dtype folding (e.g. FP16 conv weight with FP32 BN stats) would silently promote to FP32 and is unverified");
    }
    if (bn->running_mean->dtype != DataType::FP32 || bn->running_var->dtype != DataType::FP32 ||
        (bn->weight && bn->weight->dtype != DataType::FP32) ||
        (bn->bias && bn->bias->dtype != DataType::FP32)) {
        throw std::runtime_error("[litetorch Error] fold_conv_bn: BatchNorm2d statistics/affine tensors must be FP32");
    }
    int C = bn->num_features;
    std::shared_ptr<Tensor> scale;
    {
        auto eps_t = Ops::full({C}, bn->eps, wdev, false, DataType::FP32);
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
        conv->kernel_size,
        conv->stride_vec,
        conv->padding_vec,
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
            if (conv.use_count() > 3) {
                throw std::runtime_error("[litetorch Error] fuse_conv_bn: Conv2d at Sequential[" + std::to_string(i) + "] is shared with another owner (use_count=" + std::to_string(conv.use_count()) + " > 3: a Sequential built with add() holds each module twice - modules vector + named-children map - plus this function's local handle); refusing in-place rewrite. Clone the module or pass an unshared Sequential.");
            }
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

FusionReport detect_layernorm_candidates(std::shared_ptr<nn::Module> root) {
    FusionReport rep;
    scan_layernorm(root, rep, "root");
    return rep;
}

FusionReport detect_layernorm_fusion(std::shared_ptr<nn::Module> root) {
    return detect_layernorm_candidates(root);
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
