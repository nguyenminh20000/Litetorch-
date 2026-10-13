#include "litetorch/fsdp_tp.h"
#include "litetorch/fsdp_wrapper.h"
#include "litetorch/distributed.h"
#include "litetorch/tp_collectives.h"
#include "litetorch/ops.h"
#include <cstring>
#include <functional>
#include <stdexcept>

namespace litetorch {
namespace distributed {

bool FSDPTP::initialized_ = false;
int FSDPTP::tp_size_ = 1;
int FSDPTP::dp_size_ = 1;
std::map<const Tensor*, TPShardInfo> FSDPTP::tp_shard_registry_;
std::mutex FSDPTP::registry_mutex_;

void FSDPTP::init(int tp_size, int dp_size) {
    if (tp_size < 1 || dp_size < 1)
        throw std::runtime_error("[litetorch Error] FSDPTP::init: sizes must be >= 1");
    auto& pg = ProcessGroup::get();
    int world = pg.is_initialized() ? pg.get_world_size() : 1;
    if (world != tp_size * dp_size)
        throw std::runtime_error("[litetorch Error] FSDPTP::init: world_size must equal tp_size * dp_size");
    tp_size_ = tp_size;
    dp_size_ = dp_size;
    initialized_ = true;
}

int FSDPTP::tp_rank() {
    if (!initialized_) return 0;
    auto& pg = ProcessGroup::get();
    int rank = pg.is_initialized() ? pg.get_rank() : 0;
    return rank % tp_size_;
}

int FSDPTP::dp_rank() {
    if (!initialized_) return 0;
    auto& pg = ProcessGroup::get();
    int rank = pg.is_initialized() ? pg.get_rank() : 0;
    return rank / tp_size_;
}

void FSDPTP::register_tp_shard(std::shared_ptr<Tensor> param, int shard_dim,
                               const std::vector<int64_t>& full_shape) {
    if (!param) return;
    std::lock_guard<std::mutex> lk(registry_mutex_);
    TPShardInfo info;
    info.shard_dim = shard_dim;
    info.full_shape = full_shape;
    info.is_tp_sharded = true;
    tp_shard_registry_[param.get()] = info;
}

TPShardInfo FSDPTP::get_tp_shard_info(std::shared_ptr<Tensor> param) {
    std::lock_guard<std::mutex> lk(registry_mutex_);
    auto it = tp_shard_registry_.find(param.get());
    if (it != tp_shard_registry_.end()) return it->second;
    return TPShardInfo();
}

namespace {

std::shared_ptr<Tensor> gather_tp_shard(std::shared_ptr<Tensor> shard, const TPShardInfo& info) {
    if (!info.is_tp_sharded) return shard;
    if (!tp_active()) return shard;
    int dim = info.shard_dim;
    if (dim == 0) {
        std::vector<int64_t> flat_shape = {(int64_t)shard->numel()};
        auto flat = shard->contiguous()->view(flat_shape);
        auto gathered_flat = tp_all_gather_autograd(flat);
        auto full = gathered_flat->contiguous()->view(info.full_shape);
        return full;
    } else {
        auto tr = shard->transpose(0, dim)->contiguous();
        std::vector<int64_t> flat_shape = {(int64_t)tr->numel()};
        auto flat = tr->view(flat_shape);
        auto gathered_flat = tp_all_gather_autograd(flat);
        std::vector<int64_t> tr_full_shape = info.full_shape;
        std::swap(tr_full_shape[0], tr_full_shape[dim]);
        auto tr_full = gathered_flat->contiguous()->view(tr_full_shape);
        auto full = tr_full->transpose(0, dim)->contiguous();
        return full;
    }
}

std::shared_ptr<Tensor> slice_tp_shard(std::shared_ptr<Tensor> full, const TPShardInfo& info, int tp_rank, int tp_size) {
    if (!info.is_tp_sharded) return full;
    int dim = info.shard_dim;
    int64_t full_dim = info.full_shape[dim];
    int64_t shard_dim = full_dim / tp_size;
    int64_t start = (int64_t)tp_rank * shard_dim;
    return full->slice(dim, start, start + shard_dim)->contiguous();
}

}

void FSDPTP::fully_shard_tp(std::shared_ptr<nn::Module> module) {
    if (!initialized_)
        throw std::runtime_error("[litetorch Error] FSDPTP::fully_shard_tp: call FSDPTP::init first");

    std::function<void(std::shared_ptr<nn::Module>)> register_tp_params;
    register_tp_params = [&](std::shared_ptr<nn::Module> mod) {
        if (auto col = std::dynamic_pointer_cast<nn::ColumnParallelLinear>(mod)) {
            if (tp_size_ > 1 && col->weight) {
                std::vector<int64_t> full_shape = col->weight->shape;
                full_shape[0] *= tp_size_;
                register_tp_shard(col->weight, 0, full_shape);
            }
            if (tp_size_ > 1 && col->bias) {
                std::vector<int64_t> full_shape = col->bias->shape;
                full_shape[0] *= tp_size_;
                register_tp_shard(col->bias, 0, full_shape);
            }
        } else if (auto row = std::dynamic_pointer_cast<nn::RowParallelLinear>(mod)) {
            if (tp_size_ > 1 && row->weight) {
                std::vector<int64_t> full_shape = row->weight->shape;
                full_shape[1] *= tp_size_;
                register_tp_shard(row->weight, 1, full_shape);
            }
        }
        for (auto& child : mod->children()) {
            if (child) register_tp_params(child);
        }
    };
    register_tp_params(module);

    if (dp_size_ == 1) {
        return;
    }

    FSDP::fully_shard(module);
}

std::map<std::string, std::shared_ptr<Tensor>> FSDPTP::state_dict(
    std::shared_ptr<nn::Module> module) {
    std::map<std::string, std::shared_ptr<Tensor>> out;

    auto units = FSDP::active_units();
    for (auto& u : units) {
        u->ensure_gathered();
        u->copy_flat_to_params();
    }

    auto named = module->named_parameters();
    for (auto& kv : named) {
        auto param = kv.second;
        auto info = get_tp_shard_info(param);
        auto full = gather_tp_shard(param, info);
        out[kv.first] = full;
    }

    for (auto& u : units) {
        u->release_gathered();
    }

    return out;
}

void FSDPTP::load_state_dict(
    std::shared_ptr<nn::Module> module,
    const std::map<std::string, std::shared_ptr<Tensor>>& sd) {
    auto named = module->named_parameters();
    int tr = tp_rank();

    for (auto& kv : named) {
        auto it = sd.find(kv.first);
        if (it == sd.end()) continue;
        auto full = it->second;
        auto param = kv.second;
        auto info = get_tp_shard_info(param);

        std::shared_ptr<Tensor> local = slice_tp_shard(full, info, tr, tp_size_);

        if (local->numel() != param->numel())
            throw std::runtime_error("[litetorch Error] FSDPTP::load_state_dict: shape mismatch for " + kv.first);

        size_t n = (size_t)local->numel();
        size_t esz = param->storage->element_size();
        param->storage->ensure_cpu();
        local->storage->ensure_cpu();
        std::memcpy(reinterpret_cast<char*>(param->storage->get_cpu_ptr()) + (size_t)param->offset * esz,
                    reinterpret_cast<char*>(local->storage->get_cpu_ptr()) + (size_t)local->offset * esz,
                    n * esz);
    }
}

}
}
