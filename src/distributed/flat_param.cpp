#include "litetorch/flat_param.h"
#include "litetorch/distributed.h"
#include "litetorch/device.h"
#include "litetorch/backend_route.h"
#include <cstring>
#include <stdexcept>

namespace litetorch {
namespace distributed {

namespace {

std::vector<int64_t> contig_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t s = 1;
    for (int64_t i = (int64_t)shape.size() - 1; i >= 0; --i) {
        strides[(size_t)i] = s;
        s *= shape[(size_t)i];
    }
    return strides;
}

void zero_range(std::shared_ptr<Tensor> t, int64_t off, int64_t n) {
    size_t esz = t->storage->element_size();
    if (t->device.type == DeviceType::GPU) {
        std::vector<char> zeros((size_t)n * esz, 0);
        rt_gpu_write(t->gpu_data(), (size_t)n * esz, zeros.data(), (size_t)(t->offset + off) * esz);
        return;
    }
    t->storage->ensure_cpu();
    std::memset(reinterpret_cast<char*>(t->storage->get_cpu_ptr()) + (t->offset + off) * esz, 0, (size_t)n * esz);
}

}

FlatParameter::FlatParameter(const std::vector<std::shared_ptr<Tensor>>& params) {
    if (params.empty())
        throw std::runtime_error("[litetorch Error] FlatParameter: empty param list");
    dtype_ = params[0]->dtype;
    device_ = params[0]->device;
    int64_t off = 0;
    for (auto& p : params) {
        if (!p) throw std::runtime_error("[litetorch Error] FlatParameter: null param");
        if (p->dtype != dtype_)
            throw std::runtime_error("[litetorch Error] FlatParameter: mixed dtypes not supported");
        if (p->device.type != device_.type || p->device.index != device_.index)
            throw std::runtime_error("[litetorch Error] FlatParameter: mixed devices not supported");
        FlatParamMeta m;
        m.offset = off;
        m.numel = p->numel();
        m.shape = p->shape;
        metas_.push_back(m);
        off += m.numel;
    }
    unpadded_numel_ = off;
    auto& pg = ProcessGroup::get();
    world_ = pg.is_initialized() ? pg.get_world_size() : 1;
    rank_ = pg.is_initialized() ? pg.get_rank() : 0;
    if (world_ < 1) world_ = 1;
    shard_numel_ = (unpadded_numel_ + world_ - 1) / world_;
    flat_numel_ = shard_numel_ * world_;

    shard_ = Tensor::create({shard_numel_}, device_, false, dtype_);
    full_ = Tensor::create({flat_numel_}, device_, false, dtype_);
    zero_range(shard_, 0, shard_numel_);
    zero_range(full_, 0, flat_numel_);

    int64_t base = (int64_t)rank_ * shard_numel_;
    for (size_t i = 0; i < params.size(); ++i) {
        auto src = params[i]->is_contiguous() ? params[i] : params[i]->contiguous();
        const auto& m = metas_[i];
        int64_t seg_start = m.offset;
        int64_t seg_end = m.offset + m.numel;
        int64_t ov_start = seg_start > base ? seg_start : base;
        int64_t ov_end = seg_end < base + shard_numel_ ? seg_end : base + shard_numel_;
        if (ov_end <= ov_start) continue;
        int64_t src_off = ov_start - seg_start;
        int64_t dst_off = ov_start - base;
        int64_t n = ov_end - ov_start;
        size_t esz = shard_->storage->element_size();
        if (device_.type == DeviceType::GPU) {
            rt_gpu_copy(src->gpu_data(), shard_->gpu_data(), (size_t)n * esz,
                        (size_t)(src->offset + src_off) * esz, (size_t)(shard_->offset + dst_off) * esz);
        } else {
            shard_->storage->ensure_cpu();
            src->storage->ensure_cpu();
            std::memcpy(reinterpret_cast<char*>(shard_->storage->get_cpu_ptr()) + (shard_->offset + dst_off) * esz,
                        reinterpret_cast<char*>(src->storage->get_cpu_ptr()) + (src->offset + src_off) * esz,
                        (size_t)n * esz);
        }
    }
}

std::vector<std::shared_ptr<Tensor>> FlatParameter::param_views() {
    std::vector<std::shared_ptr<Tensor>> views;
    views.reserve(metas_.size());
    for (const auto& m : metas_) {
        auto v = std::make_shared<Tensor>(full_->storage, m.shape, contig_strides(m.shape), m.offset, device_, false);
        views.push_back(v);
    }
    return views;
}

void FlatParameter::all_gather_flat() {
    if (world_ <= 1) {
        full_->copy_(shard_);
        return;
    }
    if (device_.type == DeviceType::GPU) {
        if (!NCCLBridge::get().is_available())
            throw std::runtime_error("[litetorch Error] FlatParameter::all_gather_flat: GPU tensor but NCCL unavailable");
        if (!NCCLBridge::get().dp_all_gather(shard_, full_))
            throw std::runtime_error("[litetorch Error] FlatParameter::all_gather_flat: NCCL dp_all_gather failed");
        return;
    }
    ProcessGroup::get().all_gather(shard_, full_);
}

std::shared_ptr<Tensor> FlatParameter::reduce_scatter_flat(std::shared_ptr<Tensor> full_grad) {
    if (!full_grad) throw std::runtime_error("[litetorch Error] FlatParameter::reduce_scatter_flat: null grad");
    if (full_grad->numel() != flat_numel_)
        throw std::runtime_error("[litetorch Error] FlatParameter::reduce_scatter_flat: grad numel mismatch");
    auto grad_shard = Tensor::create({shard_numel_}, device_, false, dtype_);
    if (world_ <= 1) {
        grad_shard->copy_(full_grad);
        return grad_shard;
    }
    if (device_.type == DeviceType::GPU) {
        if (!NCCLBridge::get().is_available())
            throw std::runtime_error("[litetorch Error] FlatParameter::reduce_scatter_flat: GPU tensor but NCCL unavailable");
        if (!NCCLBridge::get().dp_reduce_scatter(grad_shard, full_grad))
            throw std::runtime_error("[litetorch Error] FlatParameter::reduce_scatter_flat: NCCL dp_reduce_scatter failed");
        return grad_shard;
    }
    ProcessGroup::get().reduce_scatter(grad_shard, full_grad);
    return grad_shard;
}

std::shared_ptr<FlatParameter> make_flat_param(const std::vector<std::shared_ptr<Tensor>>& params) {
    return std::make_shared<FlatParameter>(params);
}

}
}
