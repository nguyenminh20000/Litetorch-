#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/distributed.h"
#include "litetorch/tp_collectives.h"
#include "nn_utils.h"
#include <cmath>
#include <random>
#include <stdexcept>

namespace litetorch {

extern std::mt19937& get_generator();

namespace nn {
namespace {

class BwdAllReduceFn : public Function {
public:
    std::vector<std::shared_ptr<Tensor>> forward(FunctionContext*,
        const std::vector<std::shared_ptr<Tensor>>& inputs) override {
        auto t = inputs[0];
        auto out = Tensor::create(t->shape, t->device, false, t->dtype);
        out->copy_(t);
        return {out};
    }
    std::vector<std::shared_ptr<Tensor>> backward(FunctionContext*,
        const std::vector<std::shared_ptr<Tensor>>& grad_outputs) override {
        auto g = grad_outputs[0];
        auto out = Tensor::create(g->shape, g->device, false, g->dtype);
        out->copy_(g);
        distributed::tp_allreduce_inplace(out);
        return {out};
    }
};

std::shared_ptr<Tensor> bwd_allreduce_autograd(std::shared_ptr<Tensor> t) {
    if (!distributed::tp_active()) return t;
    auto fn = std::make_shared<BwdAllReduceFn>();
    return Function::apply(fn, {t})[0];
}

void init_tp_weight(std::shared_ptr<Tensor> w, int in_features) {
    kaiming_uniform(w, std::sqrt(5.0f));
}

void init_tp_bias(std::shared_ptr<Tensor> b, int in_features) {
    float bound = 1.0f / std::sqrt(static_cast<float>(in_features));
    std::mt19937& gen = get_generator();
    std::uniform_real_distribution<float> dis(-bound, bound);
    float* data = b->data_ptr();
    for (size_t i = 0; i < b->numel(); ++i) data[i] = dis(gen);
}

int tp_world_size() {
    if (distributed::ProcessGroup::get().is_initialized())
        return distributed::ProcessGroup::get().get_world_size();
    return 1;
}

}

ColumnParallelLinear::ColumnParallelLinear(int in_features, int out_features, bool has_bias) {
    int world_size = tp_world_size();
    if (out_features % world_size != 0)
        throw std::runtime_error("[litetorch Error] ColumnParallelLinear: out_features must be divisible by world_size");
    int out_per_rank = out_features / world_size;
    weight = Tensor::create({out_per_rank, in_features}, Device(DeviceType::CPU, 0), true);
    init_tp_weight(weight, in_features);
    if (has_bias) {
        bias = Tensor::create({out_per_rank}, Device(DeviceType::CPU, 0), true);
        init_tp_bias(bias, in_features);
    }
}

std::shared_ptr<Tensor> ColumnParallelLinear::forward(std::shared_ptr<Tensor> input) {
    auto xg = bwd_allreduce_autograd(input);
    auto w_t = weight->transpose(0, 1);
    auto out = Ops::matmul(xg, w_t);
    if (bias) {
        auto b_view = bias->view({1, bias->shape[0]});
        out = Ops::add(out, b_view);
    }
    return out;
}

std::vector<std::shared_ptr<Tensor>> ColumnParallelLinear::parameters() {
    if (bias) return {weight, bias};
    return {weight};
}

void ColumnParallelLinear::to(const Device& device) {
    weight = weight->to(device);
    if (bias) bias = bias->to(device);
}

RowParallelLinear::RowParallelLinear(int in_features, int out_features, bool has_bias) {
    int world_size = tp_world_size();
    if (in_features % world_size != 0)
        throw std::runtime_error("[litetorch Error] RowParallelLinear: in_features must be divisible by world_size");
    int in_per_rank = in_features / world_size;
    weight = Tensor::create({out_features, in_per_rank}, Device(DeviceType::CPU, 0), true);
    init_tp_weight(weight, in_features);
    if (has_bias) {
        bias = Tensor::create({out_features}, Device(DeviceType::CPU, 0), true);
        init_tp_bias(bias, in_features);
    }
}

std::shared_ptr<Tensor> RowParallelLinear::forward(std::shared_ptr<Tensor> input) {
    auto w_t = weight->transpose(0, 1);
    auto partial = Ops::matmul(input, w_t);
    auto out = distributed::tp_all_reduce_autograd(partial);
    if (bias) {
        auto b_view = bias->view({1, bias->shape[0]});
        out = Ops::add(out, b_view);
    }
    return out;
}

std::vector<std::shared_ptr<Tensor>> RowParallelLinear::parameters() {
    if (bias) return {weight, bias};
    return {weight};
}

void RowParallelLinear::to(const Device& device) {
    weight = weight->to(device);
    if (bias) bias = bias->to(device);
}

}
}
