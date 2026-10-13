#include "litetorch/nn.h"
#include "litetorch/ops.h"

namespace litetorch {
namespace nn {

namespace {
std::shared_ptr<Tensor> sc(float v, std::shared_ptr<Tensor> ref) {
    return Tensor::from_vector({v}, {1}, ref->device, false, ref->dtype);
}
}

std::shared_ptr<Tensor> ReLU6::forward(std::shared_ptr<Tensor> input) {
    return Ops::clamp(input, 0.0f, 6.0f);
}

RReLU::RReLU(float lower, float upper) : lower(lower), upper(upper) {}

std::shared_ptr<Tensor> RReLU::forward(std::shared_ptr<Tensor> input) {
    auto zero = sc(0.0f, input);
    auto is_pos = Ops::ge(input, zero);
    std::shared_ptr<Tensor> slope;
    if (training) {
        auto r = Ops::rand(input->shape, input->device, false, input->dtype);
        slope = Ops::add(Ops::mul(r, sc(upper - lower, input)), sc(lower, input));
    } else {
        slope = Ops::mul(sc((lower + upper) / 2.0f, input),
                         Ops::add(sc(1.0f, input), Ops::mul(input, sc(0.0f, input))));
    }
    auto neg_part = Ops::mul(input, slope);
    return Ops::where(is_pos, input, neg_part);
}

CELU::CELU(float alpha) : alpha(alpha) {}

std::shared_ptr<Tensor> CELU::forward(std::shared_ptr<Tensor> input) {
    auto zero = sc(0.0f, input);
    auto pos = Ops::where(Ops::gt(input, zero), input, zero);
    auto exp_term = Ops::sub(Ops::exp(Ops::div(input, sc(alpha, input))), sc(1.0f, input));
    auto neg_part = Ops::mul(sc(alpha, input), exp_term);
    auto neg = Ops::where(Ops::lt(input, zero), neg_part, zero);
    return Ops::add(pos, neg);
}

std::shared_ptr<Tensor> SELU::forward(std::shared_ptr<Tensor> input) {
    const float scale = 1.0507009873554805f;
    const float alpha = 1.6732632423543772f;
    auto zero = sc(0.0f, input);
    auto pos = Ops::where(Ops::gt(input, zero), input, zero);
    auto exp_term = Ops::sub(Ops::exp(input), sc(1.0f, input));
    auto neg_part = Ops::mul(sc(alpha, input), exp_term);
    auto neg = Ops::where(Ops::lt(input, zero), neg_part, zero);
    return Ops::mul(sc(scale, input), Ops::add(pos, neg));
}

std::shared_ptr<Tensor> Mish::forward(std::shared_ptr<Tensor> input) {
    auto sp = Softplus(1.0f, 20.0f).forward(input);
    return Ops::mul(input, Ops::tanh(sp));
}

Softplus::Softplus(float beta, float threshold) : beta(beta), threshold(threshold) {}

std::shared_ptr<Tensor> Softplus::forward(std::shared_ptr<Tensor> input) {
    auto bx = Ops::mul(input, sc(beta, input));
    auto large = Ops::gt(bx, sc(threshold, input));
    auto small_result = Ops::div(
        Ops::log(Ops::add(sc(1.0f, input), Ops::exp(bx))),
        sc(beta, input));
    return Ops::where(large, input, small_result);
}

Softshrink::Softshrink(float lambd) : lambd(lambd) {}

std::shared_ptr<Tensor> Softshrink::forward(std::shared_ptr<Tensor> input) {
    auto lam = sc(lambd, input);
    auto nlam = sc(-lambd, input);
    auto zero = Ops::mul(input, sc(0.0f, input));
    auto pos = Ops::gt(input, lam);
    auto neg = Ops::lt(input, nlam);
    auto pos_val = Ops::sub(input, lam);
    auto neg_val = Ops::add(input, lam);
    return Ops::where(pos, pos_val, Ops::where(neg, neg_val, zero));
}

Hardshrink::Hardshrink(float lambd) : lambd(lambd) {}

std::shared_ptr<Tensor> Hardshrink::forward(std::shared_ptr<Tensor> input) {
    auto lam = sc(lambd, input);
    auto nlam = sc(-lambd, input);
    auto zero = Ops::mul(input, sc(0.0f, input));
    auto keep = Ops::where(Ops::gt(input, lam), input,
                           Ops::where(Ops::lt(input, nlam), input, zero));
    return keep;
}

std::shared_ptr<Tensor> Tanhshrink::forward(std::shared_ptr<Tensor> input) {
    return Ops::sub(input, Ops::tanh(input));
}

std::shared_ptr<Tensor> Softsign::forward(std::shared_ptr<Tensor> input) {
    return Ops::div(input, Ops::add(sc(1.0f, input), Ops::abs(input)));
}

Hardtanh::Hardtanh(float min_val, float max_val) : min_val(min_val), max_val(max_val) {}

std::shared_ptr<Tensor> Hardtanh::forward(std::shared_ptr<Tensor> input) {
    return Ops::clamp(input, min_val, max_val);
}

Threshold::Threshold(float threshold, float value) : threshold(threshold), value(value) {}

std::shared_ptr<Tensor> Threshold::forward(std::shared_ptr<Tensor> input) {
    return Ops::where(Ops::gt(input, sc(threshold, input)), input, sc(value, input));
}

Bilinear::Bilinear(int in1_features, int in2_features, int out_features, bool use_bias)
    : in1_features(in1_features), in2_features(in2_features), out_features(out_features) {
    weight = Tensor::from_vector(std::vector<float>((size_t)out_features * in1_features * in2_features, 0.1f),
                                 {(int64_t)out_features, (int64_t)in1_features, (int64_t)in2_features},
                                 Device(DeviceType::CPU, 0), true);
    if (use_bias) {
        bias = Tensor::from_vector(std::vector<float>((size_t)out_features, 0.0f),
                                   {(int64_t)out_features}, Device(DeviceType::CPU, 0), true);
    }
}

std::shared_ptr<Tensor> Bilinear::forward(std::shared_ptr<Tensor> input) {
    throw std::runtime_error("Bilinear requires two inputs");
}

std::shared_ptr<Tensor> Bilinear::forward(std::shared_ptr<Tensor> input1,
                                           std::shared_ptr<Tensor> input2) {
    return Ops::bilinear(input1, input2, weight, bias);
}

std::vector<std::shared_ptr<Tensor>> Bilinear::parameters() {
    if (bias) return {weight, bias};
    return {weight};
}

void Bilinear::to(const Device& device) {
    weight = weight->to(device);
    if (bias) bias = bias->to(device);
}

Fold::Fold(const std::vector<int64_t>& output_size, const std::vector<int64_t>& kernel_size,
           const std::vector<int64_t>& dilation, const std::vector<int64_t>& padding,
           const std::vector<int64_t>& stride)
    : output_size(output_size), kernel_size(kernel_size), dilation(dilation),
      padding(padding), stride(stride) {}

std::shared_ptr<Tensor> Fold::forward(std::shared_ptr<Tensor> input) {
    return Ops::fold(input, output_size, kernel_size, dilation, padding, stride);
}

Unfold::Unfold(const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& dilation,
               const std::vector<int64_t>& padding, const std::vector<int64_t>& stride)
    : kernel_size(kernel_size), dilation(dilation), padding(padding), stride(stride) {}

std::shared_ptr<Tensor> Unfold::forward(std::shared_ptr<Tensor> input) {
    return Ops::unfold(input, kernel_size, dilation, padding, stride);
}

}
}
