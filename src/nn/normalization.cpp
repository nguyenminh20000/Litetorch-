#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <algorithm>

namespace litetorch {
namespace nn {

BatchNorm2d::BatchNorm2d(int num_features, float eps, float momentum, bool affine, bool track_running_stats)
    : num_features(num_features), eps(eps), momentum(momentum), affine(affine), track_running_stats(track_running_stats) {
    if (affine) {
        weight = Tensor::create({num_features}, Device(DeviceType::CPU, 0), true);
        bias = Tensor::create({num_features}, Device(DeviceType::CPU, 0), true);
        std::fill(weight->data_ptr(), weight->data_ptr() + num_features, 1.0f);
        std::fill(bias->data_ptr(), bias->data_ptr() + num_features, 0.0f);
    }
    if (track_running_stats) {
        running_mean = Tensor::create({num_features}, Device(DeviceType::CPU, 0), false);
        running_var = Tensor::create({num_features}, Device(DeviceType::CPU, 0), false);
        std::fill(running_mean->data_ptr(), running_mean->data_ptr() + num_features, 0.0f);
        std::fill(running_var->data_ptr(), running_var->data_ptr() + num_features, 1.0f);
    }
}

std::shared_ptr<Tensor> BatchNorm2d::forward(std::shared_ptr<Tensor> input) {
    std::shared_ptr<Tensor> w = weight;
    std::shared_ptr<Tensor> b = bias;
    std::shared_ptr<Tensor> rm = running_mean;
    std::shared_ptr<Tensor> rv = running_var;
    bool train = training;
    if (!affine) {
        w = Tensor::ones({num_features}, input->device, false, input->dtype);
        b = Tensor::zeros({num_features}, input->device, false, input->dtype);
    } else if (w && w->device != input->device) {
        w = w->to(input->device);
        b = b->to(input->device);
    }
    if (!track_running_stats) {
        rm = Tensor::zeros({num_features}, input->device, false, input->dtype);
        rv = Tensor::ones({num_features}, input->device, false, input->dtype);
        train = true;
    } else if (rm && rm->device != input->device) {
        rm = rm->to(input->device);
        rv = rv->to(input->device);
    }
    return Ops::batch_norm2d(input, rm, rv, w, b, train, momentum, eps);
}

std::vector<std::shared_ptr<Tensor>> BatchNorm2d::parameters() {
    if (!affine) return {};
    return { weight, bias };
}

void BatchNorm2d::to(const Device& device) {
    if (weight) weight = weight->to(device);
    if (bias) bias = bias->to(device);
    if (running_mean) running_mean = running_mean->to(device);
    if (running_var) running_var = running_var->to(device);
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
