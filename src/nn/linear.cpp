#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/distributed.h"
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

ColumnParallelLinear::ColumnParallelLinear(int in_features, int out_features, bool has_bias) {
    int world_size = 1;
    if (distributed::ProcessGroup::get().is_initialized()) {
        world_size = distributed::ProcessGroup::get().get_world_size();
    }
    int out_features_per_rank = out_features / world_size;

    weight = Tensor::create({out_features_per_rank, in_features}, Device(DeviceType::CPU, 0), true);
    kaiming_uniform(weight, std::sqrt(5.0f));

    if (has_bias) {
        bias = Tensor::create({out_features_per_rank}, Device(DeviceType::CPU, 0), true);
        float bound = 1.0f / std::sqrt(static_cast<float>(in_features));
        std::mt19937& gen = get_generator();
        std::uniform_real_distribution<float> dis(-bound, bound);
        float* data = bias->data_ptr();
        for (size_t i = 0; i < bias->numel(); ++i) {
            data[i] = dis(gen);
        }
    }
}

std::shared_ptr<Tensor> ColumnParallelLinear::forward(std::shared_ptr<Tensor> input) {
    auto w_t = weight->transpose(0, 1);
    auto out = Ops::matmul(input, w_t);
    if (bias) {
        auto b_view = bias->view({1, bias->shape[0]});
        out = Ops::add(out, b_view);
    }
    if (distributed::ProcessGroup::get().is_initialized()) {
        int world_size = distributed::ProcessGroup::get().get_world_size();
        if (world_size > 1) {
            std::vector<int64_t> full_shape = out->shape;
            full_shape[full_shape.size() - 1] *= world_size;
            auto full_out = Tensor::create(full_shape, out->device);
            distributed::ProcessGroup::get().all_gather(out, full_out);
            return full_out;
        }
    }
    return out;
}

std::vector<std::shared_ptr<Tensor>> ColumnParallelLinear::parameters() {
    if (bias) return { weight, bias };
    return { weight };
}

void ColumnParallelLinear::to(const Device& device) {
    weight = weight->to(device);
    if (bias) bias = bias->to(device);
}

RowParallelLinear::RowParallelLinear(int in_features, int out_features, bool has_bias) {
    int world_size = 1;
    if (distributed::ProcessGroup::get().is_initialized()) {
        world_size = distributed::ProcessGroup::get().get_world_size();
    }
    int in_features_per_rank = in_features / world_size;

    weight = Tensor::create({out_features, in_features_per_rank}, Device(DeviceType::CPU, 0), true);
    kaiming_uniform(weight, std::sqrt(5.0f));

    if (has_bias) {
        bias = Tensor::create({out_features}, Device(DeviceType::CPU, 0), true);
        float bound = 1.0f / std::sqrt(static_cast<float>(in_features_per_rank));
        std::mt19937& gen = get_generator();
        std::uniform_real_distribution<float> dis(-bound, bound);
        float* data = bias->data_ptr();
        for (size_t i = 0; i < bias->numel(); ++i) {
            data[i] = dis(gen);
        }
    }
}

std::shared_ptr<Tensor> RowParallelLinear::forward(std::shared_ptr<Tensor> input) {
    auto w_t = weight->transpose(0, 1);
    auto out = Ops::matmul(input, w_t);
    if (distributed::ProcessGroup::get().is_initialized()) {
        int world_size = distributed::ProcessGroup::get().get_world_size();
        if (world_size > 1) {
            distributed::ProcessGroup::get().all_reduce(out);
        }
    }
    if (bias) {
        auto b_view = bias->view({1, bias->shape[0]});
        out = Ops::add(out, b_view);
    }
    return out;
}

std::vector<std::shared_ptr<Tensor>> RowParallelLinear::parameters() {
    if (bias) return { weight, bias };
    return { weight };
}

void RowParallelLinear::to(const Device& device) {
    weight = weight->to(device);
    if (bias) bias = bias->to(device);
}

}
}
