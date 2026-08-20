#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <algorithm>

namespace litetorch {
namespace nn {

BatchNorm2d::BatchNorm2d(int num_features, float eps, float momentum)
    : num_features(num_features), eps(eps), momentum(momentum) {
    weight = Tensor::create({num_features}, Device(DeviceType::CPU, 0), true);
    bias = Tensor::create({num_features}, Device(DeviceType::CPU, 0), true);
    running_mean = Tensor::create({num_features}, Device(DeviceType::CPU, 0), false);
    running_var = Tensor::create({num_features}, Device(DeviceType::CPU, 0), false);

    std::fill(weight->data_ptr(), weight->data_ptr() + num_features, 1.0f);
    std::fill(bias->data_ptr(), bias->data_ptr() + num_features, 0.0f);
    std::fill(running_mean->data_ptr(), running_mean->data_ptr() + num_features, 0.0f);
    std::fill(running_var->data_ptr(), running_var->data_ptr() + num_features, 1.0f);
}

std::shared_ptr<Tensor> BatchNorm2d::forward(std::shared_ptr<Tensor> input) {
    return Ops::batch_norm2d(input, running_mean, running_var, weight, bias, training, momentum, eps);
}

std::vector<std::shared_ptr<Tensor>> BatchNorm2d::parameters() {
    return { weight, bias, running_mean, running_var };
}

void BatchNorm2d::to(const Device& device) {
    weight = weight->to(device);
    bias = bias->to(device);
    running_mean = running_mean->to(device);
    running_var = running_var->to(device);
}

LayerNorm::LayerNorm(const std::vector<int64_t>& normalized_shape, float eps)
    : normalized_shape(normalized_shape), eps(eps) {
    int64_t M = 1;
    for (auto s : normalized_shape) M *= s;
    weight = Tensor::create({M}, Device(DeviceType::CPU, 0), true);
    bias = Tensor::create({M}, Device(DeviceType::CPU, 0), true);
    std::fill(weight->data_ptr(), weight->data_ptr() + M, 1.0f);
    std::fill(bias->data_ptr(), bias->data_ptr() + M, 0.0f);
}

std::shared_ptr<Tensor> LayerNorm::forward(std::shared_ptr<Tensor> input) {
    return Ops::layer_norm(input, normalized_shape, weight, bias, eps);
}

std::vector<std::shared_ptr<Tensor>> LayerNorm::parameters() {
    return { weight, bias };
}

void LayerNorm::to(const Device& device) {
    weight = weight->to(device);
    bias = bias->to(device);
}

}
}
