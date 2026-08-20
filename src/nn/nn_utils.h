#ifndef LITETORCH_NN_UTILS_H
#define LITETORCH_NN_UTILS_H

#include "litetorch/tensor.h"
#include <random>
#include <cmath>

namespace litetorch {
extern std::mt19937& get_generator();

namespace nn {
namespace {
void kaiming_uniform(std::shared_ptr<Tensor> tensor, float a = 0.0) {
    int64_t fan_in = tensor->shape[1];
    if (tensor->shape.size() > 2) {
        int64_t receptive_field_size = 1;
        for (size_t i = 2; i < tensor->shape.size(); ++i) {
            receptive_field_size *= tensor->shape[i];
        }
        fan_in *= receptive_field_size;
    }
    float gain = std::sqrt(2.0f / (1.0f + a * a));
    float std = gain / std::sqrt(static_cast<float>(fan_in));
    float bound = std::sqrt(3.0f) * std;

    std::mt19937& gen = get_generator();
    std::uniform_real_distribution<float> dis(-bound, bound);

    if (tensor->device.type == DeviceType::GPU) {
        std::vector<float> temp_vec(tensor->numel());
        for (size_t i = 0; i < tensor->numel(); ++i) {
            temp_vec[i] = dis(gen);
        }
        auto temp_t = Tensor::from_vector(temp_vec, tensor->shape, Device(DeviceType::CPU, 0));
        tensor->copy_(temp_t);
    } else {
        float* data = tensor->data_ptr();
        for (size_t i = 0; i < tensor->numel(); ++i) {
            data[i] = dis(gen);
        }
    }
}
}
}
}
#endif
