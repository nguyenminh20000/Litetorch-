#include "litetorch/tp_collectives.h"
#include "litetorch/distributed.h"
#include "litetorch/autograd.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/device.h"
#include <stdexcept>

namespace litetorch {
namespace distributed {

bool tp_active() {
    auto& pg = ProcessGroup::get();
    return pg.is_initialized() && pg.get_world_size() > 1;
}

void tp_allreduce_inplace(std::shared_ptr<Tensor> t) {
    if (!t) throw std::runtime_error("[litetorch Error] tp_allreduce_inplace: null tensor");
    if (!tp_active()) return;
    if (t->device.type == DeviceType::GPU) {
        if (!NCCLBridge::get().is_available())
            throw std::runtime_error("[litetorch Error] tp_allreduce_inplace: GPU tensor but NCCL unavailable");
        if (!NCCLBridge::get().tp_all_reduce(t))
            throw std::runtime_error("[litetorch Error] tp_allreduce_inplace: NCCL tp_all_reduce failed");
        return;
    }
    ProcessGroup::get().all_reduce(t);
}

void tp_allgather_into(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    if (!shard || !full) throw std::runtime_error("[litetorch Error] tp_allgather_into: null tensor");
    if (!tp_active()) {
        full->copy_(shard);
        return;
    }
    if (shard->device.type == DeviceType::GPU) {
        if (!NCCLBridge::get().is_available())
            throw std::runtime_error("[litetorch Error] tp_allgather_into: GPU tensor but NCCL unavailable");
        if (!NCCLBridge::get().tp_all_gather(shard, full))
            throw std::runtime_error("[litetorch Error] tp_allgather_into: NCCL tp_all_gather failed");
        return;
    }
    ProcessGroup::get().all_gather(shard, full);
}

std::vector<std::shared_ptr<Tensor>> AllReduceFn::forward(FunctionContext*,
    const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto t = inputs[0];
    auto out = Tensor::create(t->shape, t->device, false, t->dtype);
    out->copy_(t);
    tp_allreduce_inplace(out);
    return {out};
}

std::vector<std::shared_ptr<Tensor>> AllReduceFn::backward(FunctionContext*,
    const std::vector<std::shared_ptr<Tensor>>& grad_outputs) {
    return {grad_outputs[0]};
}

std::vector<std::shared_ptr<Tensor>> AllGatherFn::forward(FunctionContext* ctx,
    const std::vector<std::shared_ptr<Tensor>>& inputs) {
    auto shard = inputs[0];
    int world = tp_active() ? ProcessGroup::get().get_world_size() : 1;
    std::vector<int64_t> full_shape = shard->shape;
    if (!full_shape.empty()) full_shape.back() *= world;
    auto full = Tensor::create(full_shape, shard->device, false, shard->dtype);
    tp_allgather_into(shard, full);
    ctx->save_for_backward(shard);
    return {full};
}

std::vector<std::shared_ptr<Tensor>> AllGatherFn::backward(FunctionContext* ctx,
    const std::vector<std::shared_ptr<Tensor>>& grad_outputs) {
    auto g = grad_outputs[0];
    const auto& saved = ctx->get_saved_tensors();
    auto shard = saved[0];
    int64_t shard_n = shard->shape.empty() ? 1 : shard->shape.back();
    int64_t dim = (int64_t)g->shape.size() - 1;
    int rank = tp_active() ? ProcessGroup::get().get_rank() : 0;
    auto grad_shard = Ops::narrow(g, dim, (int64_t)rank * shard_n, shard_n);
    auto out = Tensor::create(shard->shape, grad_shard->device, false, grad_shard->dtype);
    out->copy_(grad_shard);
    tp_allreduce_inplace(out);
    return {out};
}

std::shared_ptr<Tensor> tp_all_reduce_autograd(std::shared_ptr<Tensor> t) {
    if (!t) throw std::runtime_error("[litetorch Error] tp_all_reduce_autograd: null tensor");
    if (!tp_active()) return t;
    auto fn = std::make_shared<AllReduceFn>();
    return Function::apply(fn, {t})[0];
}

std::shared_ptr<Tensor> tp_all_gather_autograd(std::shared_ptr<Tensor> shard) {
    if (!shard) throw std::runtime_error("[litetorch Error] tp_all_gather_autograd: null tensor");
    if (!tp_active()) return shard;
    auto fn = std::make_shared<AllGatherFn>();
    return Function::apply(fn, {shard})[0];
}

}
}
