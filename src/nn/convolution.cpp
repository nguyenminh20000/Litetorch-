#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "nn_utils.h"
#include <stdexcept>

namespace litetorch {
namespace nn {

static std::vector<int> normalize_pair(const std::vector<int>& v, const char* name) {
    if (v.size() == 1) return {v[0], v[0]};
    if (v.size() == 2) return v;
    throw std::runtime_error(std::string("[litetorch Error] Conv2d ") + name + " must be an int or a pair of ints");
}

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool has_bias)
    : Conv2d(in_channels, out_channels, std::vector<int>{kernel_size},
             std::vector<int>{stride}, std::vector<int>{padding}, has_bias) {}

Conv2d::Conv2d(int in_channels, int out_channels, const std::vector<int>& kernel_size,
               const std::vector<int>& stride, const std::vector<int>& padding, bool has_bias) {
    auto ks = normalize_pair(kernel_size, "kernel_size");
    auto st = normalize_pair(stride, "stride");
    auto pd = normalize_pair(padding, "padding");
    if (st[0] != st[1] || pd[0] != pd[1]) {
        throw std::runtime_error("[litetorch Error] Conv2d requires square stride/padding (got asymmetric tuple); the underlying conv op only supports a single stride/padding value");
    }
    int kh = ks[0], kw = ks[1];
    this->stride = st[0];
    this->padding = pd[0];
    this->kernel_size = ks;
    stride_vec = st;
    padding_vec = pd;
    weight = Tensor::create({out_channels, in_channels, kh, kw}, Device(DeviceType::CPU, 0), true);
    kaiming_uniform(weight, std::sqrt(5.0f));

    if (has_bias) {
        bias = Tensor::create({out_channels}, Device(DeviceType::CPU, 0), true);
        float fan_in = in_channels * kh * kw;
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
