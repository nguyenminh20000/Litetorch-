#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/distributed.h"
#include "nn_utils.h"
#include <cmath>
#include <random>
#include <stdexcept>

namespace litetorch {

extern std::mt19937& get_generator();

namespace nn {
namespace {

int dist_world_size() {
    if (distributed::ProcessGroup::get().is_initialized())
        return distributed::ProcessGroup::get().get_world_size();
    return 1;
}

int dist_rank() {
    if (distributed::ProcessGroup::get().is_initialized())
        return distributed::ProcessGroup::get().get_rank();
    return 0;
}

bool tp_dist_active() {
    return distributed::ProcessGroup::get().is_initialized()
        && distributed::ProcessGroup::get().get_world_size() > 1;
}

bool tp_use_nccl() {
    return distributed::NCCLBridge::get().is_available();
}

std::shared_ptr<Tensor> ensure_contig(std::shared_ptr<Tensor> t) {
    return t->is_contiguous() ? t : t->contiguous();
}

class TPAllReduceNode : public Node {
public:
    TPAllReduceNode() : Node("TPAllReduce") {}

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        return {grad_output};
    }
};

class TPAllGatherNode : public Node {
public:
    std::vector<int64_t> shard_shape;
    TPAllGatherNode(const std::vector<int64_t>& shard_shape)
        : Node("TPAllGather"), shard_shape(shard_shape) {}

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto grad_shard = Tensor::create(shard_shape, grad_output->device, false, grad_output->dtype);
        if (tp_use_nccl()) {
            distributed::NCCLBridge::get().reduce_scatter(grad_shard, grad_output);
        } else {
            distributed::ProcessGroup::get().reduce_scatter(grad_shard, grad_output);
        }
        return {grad_shard};
    }
};

std::shared_ptr<Tensor> tp_all_reduce(std::shared_ptr<Tensor> t) {
    auto out = Tensor::create(t->shape, t->device, false, t->dtype);
    out->copy_(t);
    if (tp_use_nccl()) {
        distributed::OverlappedAllReducer::get().push_and_all_reduce(out);
        distributed::OverlappedAllReducer::get().sync();
    } else {
        distributed::ProcessGroup::get().all_reduce(out);
    }
    if (Autograd::is_grad_enabled() && t->requires_grad) {
        auto node = std::make_shared<TPAllReduceNode>();
        node->next_nodes = {t->creator};
        node->inputs = {{t, t->requires_grad}};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> tp_all_gather(std::shared_ptr<Tensor> shard, int world_size) {
    std::vector<int64_t> full_shape = shard->shape;
    full_shape.back() *= world_size;
    auto full = Tensor::create(full_shape, shard->device, false, shard->dtype);
    if (tp_use_nccl()) {
        distributed::NCCLBridge::get().tp_all_gather(shard, full);
    } else {
        distributed::ProcessGroup::get().all_gather(shard, full);
    }
    if (Autograd::is_grad_enabled() && shard->requires_grad) {
        auto node = std::make_shared<TPAllGatherNode>(shard->shape);
        node->next_nodes = {shard->creator};
        node->inputs = {{shard, shard->requires_grad}};
        node->output = full;
        full->creator = node;
        full->requires_grad = true;
    }
    return full;
}

void init_sharded_weight(std::shared_ptr<Tensor> w, int fan_in) {
    kaiming_uniform(w, std::sqrt(5.0f));
}

void init_sharded_bias(std::shared_ptr<Tensor> b, int fan_in) {
    float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
    std::mt19937& gen = get_generator();
    std::uniform_real_distribution<float> dis(-bound, bound);
    float* data = b->data_ptr();
    for (size_t i = 0; i < b->numel(); ++i) data[i] = dis(gen);
}

}

TPAttention::TPAttention(int embed_dim_, int num_heads_, int n_kv_heads_, bool has_bias_)
    : embed_dim(embed_dim_), num_heads(num_heads_),
      n_kv_heads(n_kv_heads_ < 0 ? num_heads_ : n_kv_heads_),
      has_bias(has_bias_) {
    tp_size = dist_world_size();
    tp_rank = dist_rank();
    if (embed_dim % num_heads != 0)
        throw std::runtime_error("[litetorch Error] TPAttention: embed_dim must be divisible by num_heads");
    if (num_heads % tp_size != 0)
        throw std::runtime_error("[litetorch Error] TPAttention: num_heads must be divisible by tp_size, got " +
                                 std::to_string(num_heads) + " / " + std::to_string(tp_size));
    if (n_kv_heads % tp_size != 0)
        throw std::runtime_error("[litetorch Error] TPAttention: n_kv_heads must be divisible by tp_size, got " +
                                 std::to_string(n_kv_heads) + " / " + std::to_string(tp_size));
    if (num_heads % n_kv_heads != 0)
        throw std::runtime_error("[litetorch Error] TPAttention: num_heads must be divisible by n_kv_heads for GQA");
    head_dim = embed_dim / num_heads;
    q_heads_per_rank = num_heads / tp_size;
    kv_heads_per_rank = n_kv_heads / tp_size;

    Device cpu(DeviceType::CPU, 0);
    w_q = Tensor::create({(int64_t)q_heads_per_rank * head_dim, (int64_t)embed_dim}, cpu, true);
    w_k = Tensor::create({(int64_t)kv_heads_per_rank * head_dim, (int64_t)embed_dim}, cpu, true);
    w_v = Tensor::create({(int64_t)kv_heads_per_rank * head_dim, (int64_t)embed_dim}, cpu, true);
    w_o = Tensor::create({(int64_t)embed_dim, (int64_t)q_heads_per_rank * head_dim}, cpu, true);
    init_sharded_weight(w_q, embed_dim);
    init_sharded_weight(w_k, embed_dim);
    init_sharded_weight(w_v, embed_dim);
    init_sharded_weight(w_o, q_heads_per_rank * head_dim);
    if (has_bias) {
        b_q = Tensor::create({(int64_t)q_heads_per_rank * head_dim}, cpu, true);
        b_k = Tensor::create({(int64_t)kv_heads_per_rank * head_dim}, cpu, true);
        b_v = Tensor::create({(int64_t)kv_heads_per_rank * head_dim}, cpu, true);
        b_o = Tensor::create({(int64_t)embed_dim}, cpu, true);
        init_sharded_bias(b_q, embed_dim);
        init_sharded_bias(b_k, embed_dim);
        init_sharded_bias(b_v, embed_dim);
        init_sharded_bias(b_o, q_heads_per_rank * head_dim);
    }
}

static std::shared_ptr<Tensor> linear_shard(std::shared_ptr<Tensor> x,
                                            std::shared_ptr<Tensor> w,
                                            std::shared_ptr<Tensor> b) {
    auto w_t = ensure_contig(w)->transpose(0, 1);
    auto out = Ops::matmul(x, w_t);
    if (b) {
        std::vector<int64_t> bs = {1};
        for (size_t i = 1; i < out->shape.size(); ++i) bs.push_back(out->shape[i]);
        bs.back() = b->shape[0];
        auto bv = b->view(bs);
        out = Ops::add(out, bv);
    }
    return out;
}

std::shared_ptr<Tensor> TPAttention::forward(std::shared_ptr<Tensor> input) {
    if (input->shape.size() != 3)
        throw std::runtime_error("[litetorch Error] TPAttention: input must be (B, T, E) batch-first");
    int64_t B = input->shape[0];
    int64_t T = input->shape[1];
    if (input->shape[2] != embed_dim)
        throw std::runtime_error("[litetorch Error] TPAttention: last dim must equal embed_dim");

    auto q = linear_shard(input, w_q, has_bias ? b_q : nullptr);
    auto k = linear_shard(input, w_k, has_bias ? b_k : nullptr);
    auto v = linear_shard(input, w_v, has_bias ? b_v : nullptr);

    auto split = [&](std::shared_ptr<Tensor> x, int64_t heads) {
        auto r = ensure_contig(x)->view({B, T, heads, head_dim});
        return ensure_contig(r->transpose(1, 2));
    };
    auto q4 = split(q, q_heads_per_rank);
    auto k4 = split(k, kv_heads_per_rank);
    auto v4 = split(v, kv_heads_per_rank);

    int64_t rep = q_heads_per_rank / kv_heads_per_rank;
    std::shared_ptr<Tensor> k4r = k4;
    std::shared_ptr<Tensor> v4r = v4;
    if (rep > 1) {
        std::vector<std::shared_ptr<Tensor>> kheads, vheads;
        for (int64_t h = 0; h < kv_heads_per_rank; ++h) {
            auto kh = Ops::narrow(k4, 1, h, 1);
            auto vh = Ops::narrow(v4, 1, h, 1);
            for (int64_t r = 0; r < rep; ++r) {
                kheads.push_back(kh);
                vheads.push_back(vh);
            }
        }
        k4r = Ops::cat(kheads, 1);
        v4r = Ops::cat(vheads, 1);
    }

    auto qbh = ensure_contig(q4)->view({B * q_heads_per_rank, T, head_dim});
    auto kbh = ensure_contig(k4r)->view({B * q_heads_per_rank, T, head_dim});
    auto vbh = ensure_contig(v4r)->view({B * q_heads_per_rank, T, head_dim});
    auto kbh_t = ensure_contig(kbh->transpose(1, 2));
    auto scores = Ops::bmm(qbh, kbh_t);
    auto scale = Tensor::from_vector({1.0f / std::sqrt(static_cast<float>(head_dim))}, {1},
                                     scores->device, false, scores->dtype);
    scores = Ops::mul(scores, scale);
    auto probs = Ops::softmax(scores, -1);
    auto ctx = Ops::bmm(probs, vbh);
    auto ctx4 = ensure_contig(ctx)->view({B, q_heads_per_rank, T, head_dim});
    auto ctx_t = ensure_contig(ctx4->transpose(1, 2));
    auto ctx_flat = ensure_contig(ctx_t)->view({B, T, (int64_t)q_heads_per_rank * head_dim});

    auto out_shard = linear_shard(ctx_flat, w_o, nullptr);
    std::shared_ptr<Tensor> out = out_shard;
    if (tp_dist_active()) out = tp_all_reduce(out_shard);
    if (has_bias) {
        auto bv = b_o->view({1, 1, (int64_t)embed_dim});
        out = Ops::add(out, bv);
    }
    return out;
}

std::vector<std::shared_ptr<Tensor>> TPAttention::parameters() {
    std::vector<std::shared_ptr<Tensor>> p = {w_q, w_k, w_v, w_o};
    if (has_bias) { p.push_back(b_q); p.push_back(b_k); p.push_back(b_v); p.push_back(b_o); }
    return p;
}

void TPAttention::to(const Device& device) {
    w_q = w_q->to(device); w_k = w_k->to(device);
    w_v = w_v->to(device); w_o = w_o->to(device);
    if (has_bias) { b_q = b_q->to(device); b_k = b_k->to(device); b_v = b_v->to(device); b_o = b_o->to(device); }
}

}
}
