#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "nn_utils.h"

namespace litetorch {
namespace nn {

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool has_bias)
    : stride(stride), padding(padding) {
    weight = Tensor::create({out_channels, in_channels, kernel_size, kernel_size}, Device(DeviceType::CPU, 0), true);
    kaiming_uniform(weight, std::sqrt(5.0f));

    if (has_bias) {
        bias = Tensor::create({out_channels}, Device(DeviceType::CPU, 0), true);
        float fan_in = in_channels * kernel_size * kernel_size;
        float bound = 1.0f / std::sqrt(fan_in);
        std::mt19937& gen = get_generator();
        std::uniform_real_distribution<float> dis(-bound, bound);
        float* data = bias->data_ptr();
        for (size_t i = 0; i < bias->numel(); ++i) {
            data[i] = dis(gen);
        }
    }
}

std::shared_ptr<Tensor> Conv2d::forward(std::shared_ptr<Tensor> input) {
    return Ops::conv2d(input, weight, bias, stride, padding);
}

std::vector<std::shared_ptr<Tensor>> Conv2d::parameters() {
    if (bias) return { weight, bias };
    return { weight };
}

void Conv2d::to(const Device& device) {
    weight = weight->to(device);
    if (bias) bias = bias->to(device);
}

Conv3d::Conv3d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool has_bias)
    : stride(stride), padding(padding) {
    weight = Tensor::create({out_channels, in_channels, kernel_size, kernel_size, kernel_size}, Device(DeviceType::CPU, 0), true);
    kaiming_uniform(weight, std::sqrt(5.0f));

    if (has_bias) {
        bias = Tensor::create({out_channels}, Device(DeviceType::CPU, 0), true);
        float fan_in = in_channels * kernel_size * kernel_size * kernel_size;
        float bound = 1.0f / std::sqrt(fan_in);
        std::mt19937& gen = get_generator();
        std::uniform_real_distribution<float> dis(-bound, bound);
        float* data = bias->data_ptr();
        for (size_t i = 0; i < bias->numel(); ++i) {
            data[i] = dis(gen);
        }
    }
}

std::shared_ptr<Tensor> Conv3d::forward(std::shared_ptr<Tensor> input) {
    return Ops::conv3d(input, weight, bias, stride, padding);
}

std::vector<std::shared_ptr<Tensor>> Conv3d::parameters() {
    if (bias) return { weight, bias };
    return { weight };
}

void Conv3d::to(const Device& device) {
    weight = weight->to(device);
    if (bias) bias = bias->to(device);
}

}
}
