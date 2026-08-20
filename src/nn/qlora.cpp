#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/tensor.h"
#include <random>
#include <cmath>

namespace litetorch {
namespace nn {

QLoRALinear::QLoRALinear(int in_features, int out_features, int r, float lora_alpha, bool has_bias) {
    std::vector<float> w_data(out_features * in_features);
    std::default_random_engine generator(42);
    std::normal_distribution<float> distribution(0.0f, 1.0f / std::sqrt(static_cast<float>(in_features)));
    for (int i = 0; i < out_features * in_features; ++i) {
        w_data[i] = distribution(generator);
    }
    auto w_fp32 = Tensor::from_vector(w_data, {out_features, in_features}, Device(DeviceType::CPU));
    weight = w_fp32->cast(DataType::NF4);

    if (has_bias) {
        bias = Tensor::zeros({out_features}, Device(DeviceType::CPU), true);
    } else {
        bias = nullptr;
    }

    std::vector<float> a_data(r * in_features);
    std::normal_distribution<float> a_dist(0.0f, 1.0f / std::sqrt(static_cast<float>(in_features)));
    for (int i = 0; i < r * in_features; ++i) {
        a_data[i] = a_dist(generator);
    }
    lora_A = Tensor::from_vector(a_data, {r, in_features}, Device(DeviceType::CPU), true);
    lora_B = Tensor::zeros({out_features, r}, Device(DeviceType::CPU), true);

    scaling = lora_alpha / static_cast<float>(r);
}

std::shared_ptr<Tensor> QLoRALinear::forward(std::shared_ptr<Tensor> input) {
    auto dequant_w = weight->cast(DataType::FP32);
    auto out_base = Ops::matmul(input, dequant_w->transpose(0, 1));
    if (bias) {
        out_base = Ops::add(out_base, bias);
    }
    auto lora_x = Ops::matmul(input, lora_A->transpose(0, 1));
    auto lora_out = Ops::matmul(lora_x, lora_B->transpose(0, 1));
    auto scale_t = Tensor::from_vector({scaling}, {1}, input->device);
    auto lora_out_scaled = Ops::mul(lora_out, scale_t);
    return Ops::add(out_base, lora_out_scaled);
}

std::vector<std::shared_ptr<Tensor>> QLoRALinear::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    params.push_back(lora_A);
    params.push_back(lora_B);
    if (bias) {
        params.push_back(bias);
    }
    return params;
}

void QLoRALinear::to(const Device& device) {
    weight = weight->to(device);
    if (bias) {
        bias = bias->to(device);
    }
    lora_A = lora_A->to(device);
    lora_B = lora_B->to(device);
}

}
}
