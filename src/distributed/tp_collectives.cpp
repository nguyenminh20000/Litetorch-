#include "litetorch/tp_collectives.h"
#include "litetorch/distributed.h"
#include "litetorch/autograd.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/device.h"
#include "litetorch/fsdp_tp.h"
#include <cstring>
#include <stdexcept>

namespace litetorch {
namespace distributed {

namespace {

struct TPGroupInfo {
    int size = 1;
    int position = 0;
    int color = 0;
};

TPGroupInfo current_tp_group() {
    TPGroupInfo g;
    auto& pg = ProcessGroup::get();
    if (!pg.is_initialized()) return g;
    int world = pg.get_world_size();
    int rank = pg.get_rank();
    int tp_size = world;
    if (FSDPTP::is_initialized() && FSDPTP::tp_size() >= 1) tp_size = FSDPTP::tp_size();
    if (tp_size <= 1) return g;
    if (tp_size >= world) {
        g.size = world;
        g.position = rank;
        g.color = 0;
        return g;
    }
    g.size = tp_size;
    g.color = rank / tp_size;
    g.position = rank % tp_size;
    return g;
}

uint16_t float_to_bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t rounding = ((u >> 16) & 1u) + 0x7FFFu;
    return (uint16_t)((u + rounding) >> 16);
}

float bf16_bits_to_float(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

template <typename T>
void subgroup_sum_typed(char* acc, const char* seg, int64_t n) {
    T* a = reinterpret_cast<T*>(acc);
    const T* s = reinterpret_cast<const T*>(seg);
    for (int64_t i = 0; i < n; ++i) a[i] = (T)(a[i] + s[i]);
}

void subgroup_sum_accumulate(char* acc, const char* seg, int64_t n, DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            subgroup_sum_typed<float>(acc, seg, n);
            return;
        case DataType::FP64:
            subgroup_sum_typed<double>(acc, seg, n);
            return;
        case DataType::INT32:
            subgroup_sum_typed<int32_t>(acc, seg, n);
            return;
        case DataType::INT64:
            subgroup_sum_typed<int64_t>(acc, seg, n);
            return;
        case DataType::INT16:
            subgroup_sum_typed<int16_t>(acc, seg, n);
            return;
        case DataType::INT8:
            subgroup_sum_typed<int8_t>(acc, seg, n);
            return;
        case DataType::UINT8:
            subgroup_sum_typed<uint8_t>(acc, seg, n);
            return;
        case DataType::FP16: {
            uint16_t* a = reinterpret_cast<uint16_t*>(acc);
            const uint16_t* s = reinterpret_cast<const uint16_t*>(seg);
            for (int64_t i = 0; i < n; ++i)
                a[i] = float_to_half(half_to_float(a[i]) + half_to_float(s[i]));
            return;
        }
        case DataType::BF16: {
            uint16_t* a = reinterpret_cast<uint16_t*>(acc);
            const uint16_t* s = reinterpret_cast<const uint16_t*>(seg);
            for (int64_t i = 0; i < n; ++i)
                a[i] = float_to_bf16_bits(bf16_bits_to_float(a[i]) + bf16_bits_to_float(s[i]));
            return;
        }
        default:
            throw std::runtime_error("[litetorch Error] tp subgroup all_reduce SUM: unsupported dtype");
    }
}

}

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
    auto& pg = ProcessGroup::get();
    TPGroupInfo g = current_tp_group();
    if (g.size <= 1) return;
    if (g.size == pg.get_world_size()) {
        pg.all_reduce(t);
        return;
    }
    int64_t n = (int64_t)t->numel();
    if (n <= 0) throw std::runtime_error("[litetorch Error] tp_allreduce_inplace: empty tensor");
    int world = pg.get_world_size();
    auto wfull = Tensor::create({(int64_t)world * n}, t->device, false, t->dtype);
    pg.all_gather(t, wfull);
    size_t esz = t->storage->element_size();
    std::vector<char> acc((size_t)n * esz);
    const char* base = static_cast<const char*>(wfull->raw_data_ptr());
    std::memcpy(acc.data(), base + (int64_t)g.color * g.size * n * (int64_t)esz, (size_t)n * esz);
    for (int k = 1; k < g.size; ++k)
        subgroup_sum_accumulate(acc.data(),
            base + ((int64_t)g.color * g.size + k) * n * (int64_t)esz, n, t->dtype);
    auto out = Tensor::create(t->shape, t->device, false, t->dtype);
    std::memcpy(out->raw_data_ptr(), acc.data(), (size_t)n * esz);
    t->copy_(out);
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
    auto& pg = ProcessGroup::get();
    TPGroupInfo g = current_tp_group();
    if (g.size <= 1) {
        full->copy_(shard);
        return;
    }
    if (g.size == pg.get_world_size()) {
        pg.all_gather(shard, full);
        return;
    }
    int64_t n = (int64_t)shard->numel();
    if (n <= 0) throw std::runtime_error("[litetorch Error] tp_allgather_into: empty shard");
    int world = pg.get_world_size();
    auto wfull = Tensor::create({(int64_t)world * n}, shard->device, false, shard->dtype);
    pg.all_gather(shard, wfull);
    int64_t need = (int64_t)g.size * n;
    if ((int64_t)full->numel() != need)
        throw std::runtime_error("[litetorch Error] tp_allgather_into: full numel does not match TP group size");
    auto seg = Tensor::create(full->shape, shard->device, false, shard->dtype);
    size_t esz = shard->storage->element_size();
    std::memcpy(seg->raw_data_ptr(),
        static_cast<const char*>(wfull->raw_data_ptr()) + (int64_t)g.color * g.size * n * (int64_t)esz,
        (size_t)need * esz);
    full->copy_(seg);
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
    int gs = tp_active() ? current_tp_group().size : 1;
    std::vector<int64_t> full_shape = shard->shape;
    if (!full_shape.empty()) full_shape.back() *= gs;
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
    auto gsum = Tensor::create(g->shape, g->device, false, g->dtype);
    gsum->copy_(g);
    tp_allreduce_inplace(gsum);
    int pos = tp_active() ? current_tp_group().position : 0;
    auto grad_shard = Ops::narrow(gsum, dim, (int64_t)pos * shard_n, shard_n);
    auto out = Tensor::create(shard->shape, grad_shard->device, false, grad_shard->dtype);
    out->copy_(grad_shard);
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
    auto gfn = std::make_shared<AllGatherFn>();
    return Function::apply(gfn, {shard})[0];
}

}
}
