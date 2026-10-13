#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/distributed.h"
#include "nn_utils.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <cstdint>

namespace litetorch {

extern std::mt19937& get_generator();

namespace nn {
namespace {

int ve_world_size() {
    if (distributed::ProcessGroup::get().is_initialized())
        return distributed::ProcessGroup::get().get_world_size();
    return 1;
}

int ve_rank() {
    if (distributed::ProcessGroup::get().is_initialized())
        return distributed::ProcessGroup::get().get_rank();
    return 0;
}

bool ve_dist_active() {
    return distributed::ProcessGroup::get().is_initialized()
        && distributed::ProcessGroup::get().get_world_size() > 1;
}

bool ve_use_nccl() {
    return distributed::NCCLBridge::get().is_available();
}

class VEAllReduceNode : public Node {
public:
    VEAllReduceNode() : Node("VEAllReduce") {}

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        return {grad_output};
    }
};

std::shared_ptr<Tensor> ve_all_reduce(std::shared_ptr<Tensor> t) {
    auto out = Tensor::create(t->shape, t->device, false, t->dtype);
    out->copy_(t);
    if (ve_use_nccl()) {
        distributed::OverlappedAllReducer::get().push_and_all_reduce(out);
        distributed::OverlappedAllReducer::get().sync();
    } else {
        distributed::ProcessGroup::get().all_reduce(out);
    }
    if (Autograd::is_grad_enabled() && t->requires_grad) {
        auto node = std::make_shared<VEAllReduceNode>();
        node->next_nodes = {t->creator};
        node->inputs = {{t, t->requires_grad}};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

void ve_all_reduce_max_inplace(std::shared_ptr<Tensor> t) {
    if (ve_use_nccl()) {
        distributed::NCCLBridge::get().all_reduce(t, distributed::ReduceOp::MAX);
    } else {
        distributed::ProcessGroup::get().all_reduce(t, distributed::ReduceOp::MAX);
    }
}

}

VocabParallelEmbedding::VocabParallelEmbedding(int vocab_size_, int embed_dim_, int64_t padding_idx_)
    : vocab_size(vocab_size_), embed_dim(embed_dim_), padding_idx(padding_idx_) {
    tp_size = ve_world_size();
    tp_rank = ve_rank();
    if (vocab_size % tp_size != 0)
        throw std::runtime_error("[litetorch Error] VocabParallelEmbedding: vocab_size must be divisible by tp_size, got " +
                                 std::to_string(vocab_size) + " / " + std::to_string(tp_size));
    vocab_per_rank = vocab_size / tp_size;
    vocab_start = (int64_t)tp_rank * vocab_per_rank;
    vocab_end = vocab_start + vocab_per_rank;

    Device cpu(DeviceType::CPU, 0);
    weight = Tensor::create({(int64_t)vocab_per_rank, (int64_t)embed_dim}, cpu, true);
    kaiming_uniform(weight, std::sqrt(5.0f));
    if (padding_idx >= vocab_start && padding_idx < vocab_end) {
        NoGradGuard ng;
        float* d = weight->data_ptr();
        int64_t row = padding_idx - vocab_start;
        for (int64_t j = 0; j < embed_dim; ++j) d[row * embed_dim + j] = 0.0f;
    }
}

std::shared_ptr<Tensor> VocabParallelEmbedding::forward(std::shared_ptr<Tensor> input) {
    auto in_c = input->is_contiguous() ? input : input->contiguous();
    auto vals = in_c->to_vector();
    int64_t n = (int64_t)vals.size();

    Device cpu(DeviceType::CPU, 0);
    auto safe_ids = Tensor::create(in_c->shape, cpu, false, DataType::INT32);
    auto mask = Tensor::create(in_c->shape, cpu, false, DataType::FP32);
    int32_t* sp = reinterpret_cast<int32_t*>(safe_ids->data_ptr());
    float* mp = mask->data_ptr();
    for (int64_t i = 0; i < n; ++i) {
        int64_t id = (int64_t)vals[i];
        bool in_range = (id >= vocab_start && id < vocab_end);
        sp[i] = in_range ? (id - vocab_start) : 0;
        mp[i] = in_range ? 1.0f : 0.0f;
    }
    if (in_c->device.type != DeviceType::CPU) {
        safe_ids = safe_ids->to(in_c->device);
        mask = mask->to(in_c->device);
    }

    std::vector<int64_t> gshape = in_c->shape;
    gshape.push_back(embed_dim);
    int64_t n_total = 1;
    for (size_t i = 0; i < in_c->shape.size(); ++i) n_total *= in_c->shape[i];
    auto flat_ids = safe_ids->view({n_total});
    auto flat_idx = Ops::unsqueeze(flat_ids, -1)->expand({n_total, embed_dim});
    auto flat_gathered = Ops::gather(weight, 0, flat_idx);
    auto gathered = flat_gathered->view(gshape);
    std::vector<int64_t> mshape = in_c->shape;
    mshape.push_back(1);
    auto m = mask->view(mshape)->expand(gshape);
    if (m->dtype != gathered->dtype) m = m->cast(gathered->dtype);
    auto out = Ops::mul(gathered, m);
    if (ve_dist_active()) out = ve_all_reduce(out);
    return out;
}

std::shared_ptr<Tensor> VocabParallelEmbedding::logits(std::shared_ptr<Tensor> hidden) {
    if (hidden->shape.empty() || hidden->shape.back() != embed_dim)
        throw std::runtime_error("[litetorch Error] VocabParallelEmbedding::logits: last dim must equal embed_dim");
    std::vector<int64_t> hshape = hidden->shape;
    int64_t vd = hshape.back();
    int64_t n = 1;
    for (size_t i = 0; i + 1 < hshape.size(); ++i) n *= hshape[i];
    auto flat = hidden->view({n, vd});
    auto w_t = weight->transpose(0, 1);
    auto l = Ops::matmul(flat, w_t);
    std::vector<int64_t> lshape = hshape;
    lshape.back() = vocab_per_rank;
    return l->view(lshape);
}

std::shared_ptr<Tensor> VocabParallelEmbedding::sharded_cross_entropy(
    std::shared_ptr<Tensor> logits_shard, std::shared_ptr<Tensor> targets) {
    if (logits_shard->shape.size() != 2 || logits_shard->shape[1] != vocab_per_rank)
        throw std::runtime_error("[litetorch Error] sharded_cross_entropy: logits must be (N, vocab_per_rank)");
    if (targets->shape.size() != 1 || targets->shape[0] != logits_shard->shape[0])
        throw std::runtime_error("[litetorch Error] sharded_cross_entropy: targets must be (N,)");
    int64_t N = logits_shard->shape[0];
    auto tvals = targets->to_vector();

    std::shared_ptr<Tensor> global_max;
    {
        NoGradGuard ng;
        auto local_max = Ops::max(logits_shard, 1, false);
        global_max = Tensor::create(local_max->shape, local_max->device, false, local_max->dtype);
        global_max->copy_(local_max);
        if (ve_dist_active()) ve_all_reduce_max_inplace(global_max);
    }
    auto gm = global_max->view({N, 1})->expand({N, vocab_per_rank});
    auto shifted = Ops::sub(logits_shard, gm);
    auto sum_local = Ops::sum(Ops::exp(shifted), 1, false);
    std::shared_ptr<Tensor> sum_global = sum_local;
    if (ve_dist_active()) sum_global = ve_all_reduce(sum_local);

    Device cpu(DeviceType::CPU, 0);
    auto onehot = Tensor::create({N, (int64_t)vocab_per_rank}, cpu, false, DataType::FP32);
    float* op = onehot->data_ptr();
    for (int64_t i = 0; i < N; ++i) {
        int64_t t = (int64_t)tvals[i];
        for (int64_t j = 0; j < vocab_per_rank; ++j)
            op[i * vocab_per_rank + j] = (t >= vocab_start && t < vocab_end && t - vocab_start == j) ? 1.0f : 0.0f;
    }
    if (logits_shard->device.type != DeviceType::CPU) onehot = onehot->to(logits_shard->device);
    if (onehot->dtype != logits_shard->dtype) onehot = onehot->cast(logits_shard->dtype);
    auto pred_local = Ops::sum(Ops::mul(logits_shard, onehot), 1, false);
    std::shared_ptr<Tensor> pred_global = pred_local;
    if (ve_dist_active()) pred_global = ve_all_reduce(pred_local);

    auto lse = Ops::add(Ops::log(sum_global), global_max);
    auto per = Ops::sub(lse, pred_global);
    return Ops::mean(per);
}

std::vector<std::shared_ptr<Tensor>> VocabParallelEmbedding::parameters() {
    return {weight};
}

void VocabParallelEmbedding::to(const Device& device) {
    weight = weight->to(device);
}

}
}
