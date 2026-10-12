#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "nn_utils.h"
#include <stdexcept>
#include <cmath>
#include <random>

namespace litetorch {
extern std::mt19937& get_generator();
namespace nn {

static std::vector<int> normalize_pair(const std::vector<int>& v, const char* name) {
    if (v.size() == 1) return {v[0], v[0]};
    if (v.size() == 2) return v;
    throw std::runtime_error(std::string("[litetorch Error] Conv2d ") + name + " must be an int or a pair of ints");
}

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool has_bias, int groups)
    : Conv2d(in_channels, out_channels, std::vector<int>{kernel_size},
             std::vector<int>{stride}, std::vector<int>{padding}, has_bias, groups) {}

Conv2d::Conv2d(int in_channels, int out_channels, const std::vector<int>& kernel_size,
               const std::vector<int>& stride, const std::vector<int>& padding, bool has_bias, int groups) {
    auto ks = normalize_pair(kernel_size, "kernel_size");
    auto st = normalize_pair(stride, "stride");
    auto pd = normalize_pair(padding, "padding");
    if (st[0] != st[1] || pd[0] != pd[1]) {
        throw std::runtime_error("[litetorch Error] Conv2d requires square stride/padding (got asymmetric tuple); the underlying conv op only supports a single stride/padding value");
    }
    if (groups < 1) {
        throw std::runtime_error("[litetorch Error] Conv2d groups must be >= 1");
    }
    if (in_channels % groups != 0 || out_channels % groups != 0) {
        throw std::runtime_error("[litetorch Error] Conv2d in_channels and out_channels must be divisible by groups");
    }
    int kh = ks[0], kw = ks[1];
    this->stride = st[0];
    this->padding = pd[0];
    this->kernel_size = ks;
    this->groups = groups;
    stride_vec = st;
    padding_vec = pd;
    weight = Tensor::create({out_channels, in_channels / groups, kh, kw}, Device(DeviceType::CPU, 0), true);
    kaiming_uniform(weight, std::sqrt(5.0f));

    if (has_bias) {
        bias = Tensor::create({out_channels}, Device(DeviceType::CPU, 0), true);
        float fan_in = (in_channels / groups) * kh * kw;
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
    if (groups == 1) {
        return Ops::conv2d(input, weight, bias, stride, padding);
    }
    auto xs = Ops::chunk(input, groups, 1);
    auto ws = Ops::chunk(weight, groups, 0);
    std::vector<std::shared_ptr<Tensor>> bs;
    if (bias) bs = Ops::chunk(bias, groups, 0);
    std::vector<std::shared_ptr<Tensor>> ys;
    ys.reserve(static_cast<size_t>(groups));
    for (int g = 0; g < groups; ++g) {
        size_t gi = static_cast<size_t>(g);
        std::shared_ptr<Tensor> b = bias ? bs[gi] : nullptr;
        ys.push_back(Ops::conv2d(xs[gi], ws[gi], b, stride, padding));
    }
    return Ops::cat(ys, 1);
}

std::vector<std::shared_ptr<Tensor>> Conv2d::parameters() {
    if (bias) return { weight, bias };
    return { weight };
}

void Conv2d::to(const Device& device) {
    weight = weight->to(device);
    if (bias) bias = bias->to(device);
}

Conv3d::Conv3d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool has_bias, int groups)
    : stride(stride), padding(padding), groups(groups) {
    if (groups < 1) {
        throw std::runtime_error("[litetorch Error] Conv3d groups must be >= 1");
    }
    if (in_channels % groups != 0 || out_channels % groups != 0) {
        throw std::runtime_error("[litetorch Error] Conv3d in_channels and out_channels must be divisible by groups");
    }
    weight = Tensor::create({out_channels, in_channels / groups, kernel_size, kernel_size, kernel_size}, Device(DeviceType::CPU, 0), true);
    kaiming_uniform(weight, std::sqrt(5.0f));

    if (has_bias) {
        bias = Tensor::create({out_channels}, Device(DeviceType::CPU, 0), true);
        float fan_in = (in_channels / groups) * kernel_size * kernel_size * kernel_size;
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
    if (groups == 1) {
        return Ops::conv3d(input, weight, bias, stride, padding);
    }
    auto xs = Ops::chunk(input, groups, 1);
    auto ws = Ops::chunk(weight, groups, 0);
    std::vector<std::shared_ptr<Tensor>> bs;
    if (bias) bs = Ops::chunk(bias, groups, 0);
    std::vector<std::shared_ptr<Tensor>> ys;
    ys.reserve(static_cast<size_t>(groups));
    for (int g = 0; g < groups; ++g) {
        size_t gi = static_cast<size_t>(g);
        std::shared_ptr<Tensor> b = bias ? bs[gi] : nullptr;
        ys.push_back(Ops::conv3d(xs[gi], ws[gi], b, stride, padding));
    }
    return Ops::cat(ys, 1);
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
