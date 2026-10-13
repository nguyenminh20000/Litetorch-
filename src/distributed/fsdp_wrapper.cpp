#include "litetorch/fsdp_wrapper.h"
#include "litetorch/distributed.h"
#include "litetorch/autograd.h"
#include "litetorch/backend.h"
#include "litetorch/backend_route.h"
#include "litetorch/allocator.h"
#include <cstring>
#include <stdexcept>

namespace litetorch {
namespace distributed {

std::atomic<int> FSDPUnit::prefetch_limit_{2};
std::atomic<int> FSDPUnit::inflight_gathers_{0};

namespace {

thread_local std::vector<std::shared_ptr<FSDPUnit>> t_forward_stack;
std::mutex g_units_mutex;
std::vector<std::weak_ptr<FSDPUnit>> g_all_units;

void copy_tensor_data(std::shared_ptr<Tensor> dst, std::shared_ptr<Tensor> src) {
    if (dst->numel() != src->numel())
        throw std::runtime_error("[litetorch Error] FSDP copy: numel mismatch");
    size_t n = (size_t)dst->numel();
    size_t esz = dst->storage->element_size();
    if (dst->device.type == DeviceType::GPU) {
        if (!dst->storage->gpu_data) dst->storage->gpu_data = (cl_mem)rt_gpu_allocate(n * esz);
        if (!src->storage->gpu_data) throw std::runtime_error("[litetorch Error] FSDP copy: src has no GPU data");
        rt_gpu_copy(src->gpu_data(), dst->gpu_data(), n * esz,
                    (size_t)src->offset * esz, (size_t)dst->offset * esz);
        return;
    }
    dst->storage->ensure_cpu();
    src->storage->ensure_cpu();
    std::memcpy(reinterpret_cast<char*>(dst->storage->get_cpu_ptr()) + (size_t)dst->offset * esz,
                reinterpret_cast<char*>(src->storage->get_cpu_ptr()) + (size_t)src->offset * esz,
                n * esz);
}

}

FSDPUnit::FSDPUnit(std::shared_ptr<nn::Module> mod) : module(mod) {
    params = mod->parameters();
    if (params.empty())
        throw std::runtime_error("[litetorch Error] FSDPUnit: module has no parameters");
    flat = make_flat_param(params);
    for (auto& p : params) {
        p->shard = flat->local_shard();
    }
}

void FSDPUnit::prefetch_gather() {
    bool expected = false;
    if (!gather_issued_.compare_exchange_strong(expected, true)) return;
    auto self = shared_from_this();
    gather_future_ = std::async(std::launch::async, [self]() {
        {
            std::lock_guard<std::mutex> lk(self->gather_mutex_);
            self->flat->all_gather_flat();
            self->gathered_.store(true);
        }
        FSDPUnit::inflight_gathers_.fetch_add(1);
    });
}

void FSDPUnit::ensure_gathered() {
    if (gathered_.load()) return;
    if (gather_issued_.load()) {
        if (gather_future_.valid()) gather_future_.wait();
        return;
    }
    std::lock_guard<std::mutex> lk(gather_mutex_);
    if (!gathered_.load()) {
        flat->all_gather_flat();
        gathered_.store(true);
        inflight_gathers_.fetch_add(1);
    }
}

void FSDPUnit::copy_flat_to_params() {
    auto views = flat->param_views();
    if (views.size() != params.size())
        throw std::runtime_error("[litetorch Error] FSDPUnit: view/param count mismatch");
    for (size_t i = 0; i < params.size(); ++i) {
        copy_tensor_data(params[i], views[i]);
    }
}

void FSDPUnit::release_gathered() {
    bool was = gathered_.exchange(false);
    if (was) inflight_gathers_.fetch_sub(1);
    gather_issued_.store(false);
}

void FSDPUnit::on_param_grad(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad) {
    std::lock_guard<std::mutex> lk(grad_mutex_);
    pending_grads_.emplace_back(p, grad);
    if (pending_grads_.size() >= params.size()) {
        reduce_scatter_grads();
        pending_grads_.clear();
    }
}

void FSDPUnit::reduce_scatter_grads() {
    auto full_grad = Tensor::create({flat->flat_numel()}, flat->device(), false, flat->dtype());
    size_t esz = full_grad->storage->element_size();
    full_grad->storage->ensure_cpu();
    auto& metas = flat->metadata();
    for (auto& pg : pending_grads_) {
        auto p = pg.first;
        auto g = pg.second;
        size_t idx = 0;
        for (; idx < params.size(); ++idx) {
            if (params[idx].get() == p.get()) break;
        }
        if (idx >= params.size()) continue;
        const auto& m = metas[idx];
        auto gc = g->is_contiguous() ? g : g->contiguous();
        gc->storage->ensure_cpu();
        std::memcpy(reinterpret_cast<char*>(full_grad->storage->get_cpu_ptr()) + (size_t)m.offset * esz,
                    reinterpret_cast<char*>(gc->storage->get_cpu_ptr()) + (size_t)gc->offset * esz,
                    (size_t)m.numel * esz);
    }
    auto grad_shard = flat->reduce_scatter_flat(full_grad);
    flat->local_shard()->grad = grad_shard;
}

void FSDPUnit::set_prefetch_limit(int n) {
    if (n < 1) n = 1;
    prefetch_limit_.store(n);
}

int FSDPUnit::prefetch_limit() {
    return prefetch_limit_.load();
}

std::vector<std::shared_ptr<FSDPUnit>>& FSDPUnit::forward_stack() {
    return t_forward_stack;
}

class FSDPBackwardNode : public Node {
public:
    std::shared_ptr<Node> orig_creator;
    std::shared_ptr<FSDPUnit> unit;

    FSDPBackwardNode(std::shared_ptr<Node> orig, std::shared_ptr<FSDPUnit> u)
        : Node("FSDPBackward"), orig_creator(orig), unit(u) {
        if (orig) {
            this->inputs = orig->inputs;
            this->next_nodes = orig->next_nodes;
        }
    }

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        {
            std::unique_ptr<NoGradGuard> ng;
            if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();

            auto& stack = FSDPUnit::forward_stack();
            if (!stack.empty() && stack.back().get() == unit.get()) {
                stack.pop_back();
            }
            if (!stack.empty()) {
                stack.back()->prefetch_gather();
            }

            unit->ensure_gathered();
        }

        std::vector<std::shared_ptr<Tensor>> out;
        if (orig_creator) {
            out = orig_creator->backward(grad_output);
            if (!Autograd::is_create_graph_) {
                orig_creator->saved_tensors.clear();
                orig_creator->output = SavedTensor();
            }
        } else {
            out = { grad_output };
        }

        while (FSDPUnit::inflight_count() > FSDPUnit::prefetch_limit()) {
            unit->release_gathered();
            break;
        }
        return out;
    }
};

void FSDP::fully_shard(std::shared_ptr<nn::Module> module) {
    auto children = module->children();
    if (!children.empty()) {
        for (auto& child : children) {
            fully_shard(child);
        }
        return;
    }

    auto params = module->parameters();
    if (params.empty()) return;

    auto unit = std::make_shared<FSDPUnit>(module);
    {
        std::lock_guard<std::mutex> lk(g_units_mutex);
        g_all_units.push_back(unit);
    }

    for (auto& p : params) {
        if (p->requires_grad) {
            std::weak_ptr<FSDPUnit> wunit = unit;
            p->register_hook([wunit, p](std::shared_ptr<Tensor> grad) {
                if (auto u = wunit.lock()) {
                    u->on_param_grad(p, grad);
                }
                return grad;
            });
        }
    }

    module->register_forward_pre_hook([unit](nn::Module*, std::shared_ptr<Tensor>) {
        unit->ensure_gathered();
        unit->copy_flat_to_params();
        auto& stack = FSDPUnit::forward_stack();
        stack.push_back(unit);
        if ((int)stack.size() > 1) {
            auto prev = stack[stack.size() - 2];
            if (FSDPUnit::inflight_count() >= FSDPUnit::prefetch_limit()) {
                prev->release_gathered();
            }
        }
    });

    module->register_forward_hook([unit](nn::Module*, std::shared_ptr<Tensor>, std::shared_ptr<Tensor> out) {
        if (out->requires_grad) {
            auto bwd = std::make_shared<FSDPBackwardNode>(out->creator, unit);
            out->creator = bwd;
        }
    });
}

void FSDP::set_prefetch_limit(int n) {
    FSDPUnit::set_prefetch_limit(n);
}

std::vector<std::shared_ptr<FSDPUnit>> FSDP::active_units() {
    std::lock_guard<std::mutex> lk(g_units_mutex);
    std::vector<std::shared_ptr<FSDPUnit>> out;
    for (auto& w : g_all_units) {
        if (auto u = w.lock()) out.push_back(u);
    }
    return out;
}

}
}
