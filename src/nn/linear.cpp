#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "nn_utils.h"

namespace litetorch {
namespace nn {

Linear::Linear(int in_features, int out_features, bool has_bias) {
    weight = Tensor::create({out_features, in_features}, Device(DeviceType::CPU, 0), true);
    kaiming_uniform(weight, std::sqrt(5.0f));

    if (has_bias) {
        bias = Tensor::create({out_features}, Device(DeviceType::CPU, 0), true);
        float bound = 1.0f / std::sqrt(static_cast<float>(in_features));
        std::mt19937& gen = get_generator();
        std::uniform_real_distribution<float> dis(-bound, bound);
        float* data = bias->data_ptr();
        for (size_t i = 0; i < bias->numel(); ++i) {
            data[i] = dis(gen);
        }
    }
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    auto w = weight;
    if (weight->dtype == DataType::INT8 && scales) {
        auto w_fp32 = weight->cast(DataType::FP32);
        w = Ops::mul(w_fp32, scales->view({weight->shape[0], 1}));
    }
    auto w_t = w->transpose(0, 1);
    auto out = Ops::matmul(input, w_t);
    if (bias) {
        auto b_view = bias->view({1, bias->shape[0]});
        out = Ops::add(out, b_view);
    }
    return out;
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters() {
    if (bias) return { weight, bias };
    return { weight };
}

void Linear::to(const Device& device) {
    weight = weight->to(device);
    if (bias) bias = bias->to(device);
    if (scales) scales = scales->to(device);
}

}
}
