#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/distributed.h"
#include "nn_utils.h"
#include <cmath>
#include <random>

namespace litetorch {

extern std::mt19937& get_generator();

namespace nn {
namespace {

bool dist_active() {
    return distributed::ProcessGroup::get().is_initialized()
        && distributed::ProcessGroup::get().get_world_size() > 1;
}

bool use_nccl() {
    return distributed::NCCLBridge::get().is_available();
}

class AllGatherNode : public Node {
public:
    std::vector<int64_t> shard_shape;
    AllGatherNode(const std::vector<int64_t>& shard_shape)
        : Node("AllGather"), shard_shape(shard_shape) {}

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto grad_shard = Tensor::create(shard_shape, grad_output->device, false, grad_output->dtype);
        if (use_nccl()) {
            distributed::NCCLBridge::get().reduce_scatter(grad_shard, grad_output);
        } else {
            distributed::ProcessGroup::get().reduce_scatter(grad_shard, grad_output);
        }
        return {grad_shard};
    }
};

class AllReduceNode : public Node {
public:
    AllReduceNode() : Node("AllReduce") {}

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto g = Tensor::create(grad_output->shape, grad_output->device, false, grad_output->dtype);
        g->copy_(grad_output);
        if (use_nccl()) {
            distributed::OverlappedAllReducer::get().push_and_all_reduce(g);
            distributed::OverlappedAllReducer::get().sync();
        } else {
            distributed::ProcessGroup::get().all_reduce(g);
        }
        return {g};
    }
};

std::shared_ptr<Tensor> all_gather_autograd(std::shared_ptr<Tensor> shard, int world_size) {
    std::vector<int64_t> full_shape = shard->shape;
    full_shape.back() *= world_size;
    auto full = Tensor::create(full_shape, shard->device, false, shard->dtype);
    if (use_nccl()) {
        distributed::NCCLBridge::get().tp_all_gather(shard, full);
    } else {
        distributed::ProcessGroup::get().all_gather(shard, full);
    }
    if (Autograd::is_grad_enabled() && shard->requires_grad) {
        auto node = std::make_shared<AllGatherNode>(shard->shape);
        node->next_nodes = {shard->creator};
        node->inputs = {{shard, shard->requires_grad}};
        node->output = full;
        full->creator = node;
        full->requires_grad = true;
    }
    return full;
}

std::shared_ptr<Tensor> all_reduce_autograd(std::shared_ptr<Tensor> t) {
    auto out = Tensor::create(t->shape, t->device, false, t->dtype);
    out->copy_(t);
    if (use_nccl()) {
        distributed::OverlappedAllReducer::get().push_and_all_reduce(out);
        distributed::OverlappedAllReducer::get().sync();
    } else {
        distributed::ProcessGroup::get().all_reduce(out);
    }
    if (Autograd::is_grad_enabled() && t->requires_grad) {
        auto node = std::make_shared<AllReduceNode>();
        node->next_nodes = {t->creator};
        node->inputs = {{t, t->requires_grad}};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

}

ColumnParallelLinear::ColumnParallelLinear(int in_features, int out_features, bool has_bias) {
    int world_size = 1;
    if (distributed::ProcessGroup::get().is_initialized()) {
        world_size = distributed::ProcessGroup::get().get_world_size();
    }
    if (out_features % world_size != 0) {
        throw std::runtime_error("[litetorch Error] ColumnParallelLinear: out_features must be divisible by world_size, got " + std::to_string(out_features) + " / " + std::to_string(world_size));
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
    if (dist_active()) {
        int world_size = distributed::ProcessGroup::get().get_world_size();
        out = all_gather_autograd(out, world_size);
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
    if (in_features % world_size != 0) {
        throw std::runtime_error("[litetorch Error] RowParallelLinear: in_features must be divisible by world_size, got " + std::to_string(in_features) + " / " + std::to_string(world_size));
    }
    int in_features_per_rank = in_features / world_size;

    weight = Tensor::create({out_features, in_features_per_rank}, Device(DeviceType::CPU, 0), true);
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

std::shared_ptr<Tensor> RowParallelLinear::forward(std::shared_ptr<Tensor> input) {
    auto w_t = weight->transpose(0, 1);
    auto out = Ops::matmul(input, w_t);
    if (dist_active()) {
        out = all_reduce_autograd(out);
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
