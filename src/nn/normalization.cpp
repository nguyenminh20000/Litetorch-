#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <algorithm>
#include <stdexcept>

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
        throw std::runtime_error("[litetorch Error] BatchNorm2d: param device differs from input device; call .to(device) on the module");
    }
    if (!track_running_stats) {
        rm = Tensor::zeros({num_features}, input->device, false, input->dtype);
        rv = Tensor::ones({num_features}, input->device, false, input->dtype);
        train = true;
    } else if (rm && rm->device != input->device) {
        throw std::runtime_error("[litetorch Error] BatchNorm2d: running stats device differs from input device; call .to(device) on the module");
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
    if (weight->device != input->device || bias->device != input->device) {
        throw std::runtime_error("[litetorch Error] LayerNorm: param device differs from input device; call .to(device) on the module");
    }
    return Ops::layer_norm(input, normalized_shape, weight, bias, eps);
}

std::vector<std::shared_ptr<Tensor>> LayerNorm::parameters() {
    return { weight, bias };
}

void LayerNorm::to(const Device& device) {
    weight = weight->to(device);
    bias = bias->to(device);
}

GroupNorm::GroupNorm(int num_groups, int num_channels, float eps, bool affine)
    : num_groups(num_groups), num_channels(num_channels), eps(eps), affine(affine) {
    if (num_groups <= 0) {
        throw std::runtime_error("[litetorch Error] GroupNorm num_groups must be positive");
    }
    if (num_channels % num_groups != 0) {
        throw std::runtime_error("[litetorch Error] GroupNorm num_channels must be divisible by num_groups");
    }
    if (affine) {
        weight = Tensor::create({num_channels}, Device(DeviceType::CPU, 0), true);
        bias = Tensor::create({num_channels}, Device(DeviceType::CPU, 0), true);
        std::fill(weight->data_ptr(), weight->data_ptr() + num_channels, 1.0f);
        std::fill(bias->data_ptr(), bias->data_ptr() + num_channels, 0.0f);
    }
}

std::shared_ptr<Tensor> GroupNorm::forward(std::shared_ptr<Tensor> input) {
    if (input->shape.size() < 2) {
        throw std::runtime_error("[litetorch Error] GroupNorm requires input with at least 2 dims (N, C, ...)");
    }
    int64_t N = input->shape[0];
    int64_t C = input->shape[1];
    if (C != num_channels) {
        throw std::runtime_error("[litetorch Error] GroupNorm channel mismatch: input has " +
            std::to_string(C) + " channels, expected " + std::to_string(num_channels));
    }
    if (weight && weight->device != input->device) {
        throw std::runtime_error("[litetorch Error] GroupNorm: param device differs from input device; call .to(device) on the module");
    }
    int64_t G = num_groups;
    int64_t D = input->numel() / (N * G);
    auto xc = input->is_contiguous() ? input : input->contiguous();
    auto xr = xc->view({N, G, D});
    auto mean = Ops::mean(xr, 2, true);
    auto diff = Ops::sub(xr, mean);
    auto var = Ops::mean(Ops::mul(diff, diff), 2, true);
    auto eps_t = Tensor::from_vector({eps}, {1}, var->device, false, var->dtype);
    auto inv_std = Ops::pow(Ops::add(var, eps_t), -0.5f);
    auto x_hat = Ops::mul(diff, inv_std);
    auto xh_c = x_hat->is_contiguous() ? x_hat : x_hat->contiguous();
    auto out = xh_c->view(input->shape);
    if (!affine) return out;
    std::vector<int64_t> wshape(input->shape.size(), 1);
    wshape[1] = C;
    auto wc = weight->is_contiguous() ? weight : weight->contiguous();
    auto bc = bias->is_contiguous() ? bias : bias->contiguous();
    return Ops::add(Ops::mul(out, wc->view(wshape)), bc->view(wshape));
}

std::vector<std::shared_ptr<Tensor>> GroupNorm::parameters() {
    if (!affine) return {};
    return { weight, bias };
}

void GroupNorm::to(const Device& device) {
    if (weight) weight = weight->to(device);
    if (bias) bias = bias->to(device);
}

BatchNorm1d::BatchNorm1d(int num_features, float eps, float momentum, bool affine, bool track_running_stats)
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

std::shared_ptr<Tensor> BatchNorm1d::forward(std::shared_ptr<Tensor> input) {
    size_t ndim = input->shape.size();
    if (ndim != 2 && ndim != 3) {
        throw std::runtime_error("[litetorch Error] BatchNorm1d requires 2D (N, C) or 3D (N, C, L) input");
    }
    int64_t N = input->shape[0];
    int64_t C = input->shape[1];
    if (C != num_features) {
        throw std::runtime_error("[litetorch Error] BatchNorm1d feature mismatch: input has " +
            std::to_string(C) + " features, expected " + std::to_string(num_features));
    }
    std::shared_ptr<Tensor> w = weight;
    std::shared_ptr<Tensor> b = bias;
    std::shared_ptr<Tensor> rm = running_mean;
    std::shared_ptr<Tensor> rv = running_var;
    bool train = training;
    if (!affine) {
        w = Tensor::ones({num_features}, input->device, false, input->dtype);
        b = Tensor::zeros({num_features}, input->device, false, input->dtype);
    } else if (w && w->device != input->device) {
        throw std::runtime_error("[litetorch Error] BatchNorm1d: param device differs from input device; call .to(device) on the module");
    }
    if (!track_running_stats) {
        rm = Tensor::zeros({num_features}, input->device, false, input->dtype);
        rv = Tensor::ones({num_features}, input->device, false, input->dtype);
        train = true;
    } else if (rm && rm->device != input->device) {
        throw std::runtime_error("[litetorch Error] BatchNorm1d: running stats device differs from input device; call .to(device) on the module");
    }
    auto xc = input->is_contiguous() ? input : input->contiguous();
    std::shared_ptr<Tensor> x4;
    if (ndim == 2) {
        x4 = xc->view({N, C, 1, 1});
    } else {
        x4 = xc->view({N, C, input->shape[2], 1});
    }
    auto y4 = Ops::batch_norm2d(x4, rm, rv, w, b, train, momentum, eps);
    auto y4_c = y4->is_contiguous() ? y4 : y4->contiguous();
    return y4_c->view(input->shape);
}

std::vector<std::shared_ptr<Tensor>> BatchNorm1d::parameters() {
    if (!affine) return {};
    return { weight, bias };
}

void BatchNorm1d::to(const Device& device) {
    if (weight) weight = weight->to(device);
    if (bias) bias = bias->to(device);
    if (running_mean) running_mean = running_mean->to(device);
    if (running_var) running_var = running_var->to(device);
}

}
}
