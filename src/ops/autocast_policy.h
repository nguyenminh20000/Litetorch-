#ifndef LITETORCH_AUTOCAST_POLICY_H
#define LITETORCH_AUTOCAST_POLICY_H

#include <string>
#include "litetorch/amp.h"

namespace litetorch {
namespace amp {

enum class AutocastPolicy {
    FP16_ELIGIBLE,
    FP32_ONLY,
    PASSTHROUGH
};

inline AutocastPolicy autocast_policy_for(const std::string& op) {
    if (op == "softmax" || op == "log_softmax" || op == "layer_norm" ||
        op == "batch_norm" || op == "batch_norm2d" ||
        op == "cross_entropy" || op == "cross_entropy_loss" || op == "nll_loss" ||
        op == "mse_loss" || op == "bce_loss" || op == "bce_with_logits_loss" ||
        op == "fused_linear_cross_entropy") {
        return AutocastPolicy::FP32_ONLY;
    }
    if (op == "matmul" || op == "linear" || op == "conv2d" || op == "conv3d" ||
        op == "attention" || op == "flash_attention" || op == "flash_attention_qkv" ||
        op == "mul" || op == "add" || op == "matmul_bias" || op == "bmm") {
        return AutocastPolicy::FP16_ELIGIBLE;
    }
    return AutocastPolicy::PASSTHROUGH;
}

inline bool autocast_fp16_eligible(const std::string& op) {
    return AutocastGuard::is_enabled() &&
           autocast_policy_for(op) == AutocastPolicy::FP16_ELIGIBLE;
}

inline bool autocast_fp32_only(const std::string& op) {
    return AutocastGuard::is_enabled() &&
           autocast_policy_for(op) == AutocastPolicy::FP32_ONLY;
}

}
}

#endif
