#include "litetorch/zero3_optimizer.h"

namespace litetorch {
namespace optim {

ZeRO3Optimizer::ZeRO3Optimizer(const std::vector<std::shared_ptr<Tensor>>& params, float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(params), lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay) {}

void ZeRO3Optimizer::ensure_masters() {
    bool changed = false;
    for (auto p : params) {
        auto active = p->shard ? p->shard : p;
        if (!active) continue;
        if (master.find(active.get()) != master.end()) continue;
        std::shared_ptr<Tensor> mst;
        if (active->dtype != DataType::FP32) {
            mst = active->cast(DataType::FP32);
            mst->requires_grad = false;
        } else {
            mst = active;
        }
        master_index[active.get()] = master_list.size();
        master_list.push_back(mst);
        master[active.get()] = mst;
        changed = true;
    }
    if (changed || !adamw_) {
        adamw_ = std::make_unique<AdamW>(master_list, static_cast<double>(lr), beta1, beta2, eps, weight_decay);
    }
}

void ZeRO3Optimizer::step() {
    ensure_masters();
    if (master_list.empty() || !adamw_) return;

    adamw_->lr = static_cast<double>(lr);
    adamw_->beta1 = beta1;
    adamw_->beta2 = beta2;
    adamw_->eps = eps;
    adamw_->weight_decay = weight_decay;

    int rank = 0;
    int world_size = 1;
    bool pg_init = distributed::ProcessGroup::get().is_initialized();
    if (pg_init) {
        rank = distributed::ProcessGroup::get().get_rank();
        world_size = distributed::ProcessGroup::get().get_world_size();
    }

    size_t rep_idx = 0;
    for (auto p : params) {
        auto active = p->shard ? p->shard : p;
        if (!active) continue;
        auto mit = master.find(active.get());
        if (mit == master.end()) continue;
        auto mst = mit->second;
        mst->grad = nullptr;

        bool is_replicated = !p->shard;
        int owner_rank = 0;
        if (is_replicated) {
            owner_rank = static_cast<int>(rep_idx % static_cast<size_t>(world_size));
            rep_idx++;
            if (pg_init && rank != owner_rank) continue;
        }

        if (!active->grad) continue;
        auto g = active->grad;
        std::shared_ptr<Tensor> g32;
        if (g->dtype != DataType::FP32) {
            g32 = g->cast(DataType::FP32);
        } else {
            g32 = g;
        }
        mst->grad = g32->is_contiguous() ? g32 : g32->contiguous();
    }

    adamw_->step();

    rep_idx = 0;
    for (auto p : params) {
        auto active = p->shard ? p->shard : p;
        if (!active) continue;
        auto mit = master.find(active.get());
        if (mit == master.end()) continue;
        auto mst = mit->second;

        bool is_replicated = !p->shard;
        int owner_rank = 0;
        if (is_replicated) {
            owner_rank = static_cast<int>(rep_idx % static_cast<size_t>(world_size));
            rep_idx++;
        }

        if (!is_replicated || !pg_init || rank == owner_rank) {
            if (mst.get() != active.get()) {
                active->copy_(mst);
            }
        }

        if (is_replicated && pg_init) {
            distributed::ProcessGroup::get().broadcast(p, owner_rank);
            if (rank != owner_rank && mst.get() != active.get()) {
                mst->copy_(active);
            }
        }
    }
}

void ZeRO3Optimizer::zero_grad() {
    for (auto p : params) {
        if (p->grad) p->grad = nullptr;
        if (p->shard && p->shard->grad) p->shard->grad = nullptr;
    }
    for (auto& kv : master) {
        if (kv.second->grad) kv.second->grad = nullptr;
    }
}

}
}
