#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/backend_route.h"
#include "litetorch/thread_pool.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>

#include "litetorch/storage_guard.h"
namespace litetorch {
namespace {

std::shared_ptr<Tensor> pack4_indices(std::shared_ptr<Tensor> x, DataType idx_dtype) {
    auto xc = x->device.type == DeviceType::CPU ? x : x->to(Device(DeviceType::CPU));
    if (xc->dtype == DataType::FP32) {
        xc = xc->cast(idx_dtype);
    } else if (xc->dtype != idx_dtype && xc->dtype != DataType::INT8) {
        throw std::runtime_error("[litetorch Error] pack4: expects NF4/FP4_E2M1, INT8 indices, or FP32 input");
    }
    auto xcc = xc->is_contiguous() ? xc : xc->contiguous();
    int64_t n = xcc->numel();
    int64_t packed_n = (n + 1) / 2;
    auto out = Tensor::create({packed_n}, x->device, false, DataType::INT8);
    StorageUseGuard guard({xcc->storage, out->storage});
    uint8_t* src = (uint8_t*)xcc->storage->get_cpu_ptr();
    int8_t* dst = (int8_t*)out->storage->get_cpu_ptr();
    for (int64_t i = 0; i < packed_n; ++i) {
        uint8_t lo = src[2 * i];
        uint8_t hi = (2 * i + 1 < n) ? src[2 * i + 1] : 0;
        if (lo > 15) lo = 15;
        if (hi > 15) hi = 15;
        dst[i] = (int8_t)((hi << 4) | (lo & 0x0F));
    }
    if (x->device.type != DeviceType::CPU) {
        auto out_gpu = out->to(x->device);
        return out_gpu;
    }
    return out;
}

std::shared_ptr<Tensor> unpack4_indices(std::shared_ptr<Tensor> packed, int64_t numel, DataType idx_dtype) {
    if (numel <= 0)
        throw std::runtime_error("[litetorch Error] unpack4: numel must be positive");
    if (packed->dtype != DataType::INT8)
        throw std::runtime_error("[litetorch Error] unpack4: packed tensor must have INT8 dtype");
    if (packed->numel() < (numel + 1) / 2)
        throw std::runtime_error("[litetorch Error] unpack4: packed tensor too small for numel");
    auto pc = packed->device.type == DeviceType::CPU ? packed : packed->to(Device(DeviceType::CPU));
    auto pcc = pc->is_contiguous() ? pc : pc->contiguous();
    auto out = Tensor::create({numel}, packed->device, false, idx_dtype);
    StorageUseGuard guard({pcc->storage, out->storage});
    int8_t* src = (int8_t*)pcc->storage->get_cpu_ptr();
    uint8_t* dst = (uint8_t*)out->storage->get_cpu_ptr();
    for (int64_t i = 0; i < numel; ++i) {
        uint8_t byte = (uint8_t)src[i / 2];
        uint8_t nibble = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        dst[i] = nibble;
    }
    if (packed->device.type != DeviceType::CPU) return out->to(packed->device);
    return out;
}

std::shared_ptr<Tensor> pack4_weight_only_linear(std::shared_ptr<Tensor> x,
                                                 std::shared_ptr<Tensor> packed_w,
                                                 std::shared_ptr<Tensor> w_scale,
                                                 std::shared_ptr<Tensor> bias,
                                                 DataType idx_dtype) {
    if (x->shape.size() != 2)
        throw std::runtime_error("[litetorch Error] pack4_weight_only_linear: x must be 2D");
    int64_t K = x->shape[1];
    int64_t N = w_scale->shape[0];
    int64_t packed_k = (K + 1) / 2;
    if (packed_w->shape.size() != 2 || packed_w->shape[0] != N || packed_w->shape[1] != packed_k)
        throw std::runtime_error("[litetorch Error] pack4_weight_only_linear: packed_w shape mismatch");
    auto xc = x->device.type == DeviceType::CPU ? x : x->to(Device(DeviceType::CPU), DataType::FP32);
    if (xc->dtype != DataType::FP32) xc = xc->cast(DataType::FP32);
    xc = xc->is_contiguous() ? xc : xc->contiguous();
    auto sc = w_scale->device.type == DeviceType::CPU ? w_scale : w_scale->to(Device(DeviceType::CPU), DataType::FP32);
    if (sc->dtype != DataType::FP32) sc = sc->cast(DataType::FP32);
    auto pwc = packed_w->device.type == DeviceType::CPU ? packed_w : packed_w->to(Device(DeviceType::CPU));
    auto pwcc = pwc->is_contiguous() ? pwc : pwc->contiguous();
    auto w_unpacked_cpu = Tensor::create({N, K}, Device(DeviceType::CPU), false, idx_dtype);
    StorageUseGuard guard({pwcc->storage, w_unpacked_cpu->storage});
    int8_t* psrc = (int8_t*)pwcc->storage->get_cpu_ptr();
    uint8_t* pdst = (uint8_t*)w_unpacked_cpu->storage->get_cpu_ptr();
    for (int64_t r = 0; r < N; ++r) {
        auto row = Tensor::create({packed_k}, Device(DeviceType::CPU), false, DataType::INT8);
        int8_t* row_data = (int8_t*)row->storage->get_cpu_ptr();
        std::memcpy(row_data, psrc + r * packed_k, packed_k);
        auto row_unpacked = unpack4_indices(row, K, idx_dtype);
        uint8_t* row_out = (uint8_t*)row_unpacked->storage->get_cpu_ptr();
        std::memcpy(pdst + r * K, row_out, K);
    }
    auto w_unpacked = packed_w->device.type != DeviceType::CPU ? w_unpacked_cpu->to(packed_w->device) : w_unpacked_cpu;
    auto w_fp32 = w_unpacked->cast(DataType::FP32)->reshape({N, K});
    float* w_data = (float*)w_fp32->storage->get_cpu_ptr();
    float* s_data = (float*)sc->storage->get_cpu_ptr();
    for (int64_t i = 0; i < N; ++i)
        for (int64_t j = 0; j < K; ++j)
            w_data[i * K + j] *= s_data[i];
    auto out = Ops::matmul(xc, w_fp32->transpose(0, 1));
    if (bias) out = Ops::add(out, bias);
    if (x->device.type != DeviceType::CPU) out = out->to(x->device);
    return out;
}

}

namespace Ops {

std::shared_ptr<Tensor> nf4_pack(std::shared_ptr<Tensor> x) {
    return pack4_indices(x, DataType::NF4);
}

std::shared_ptr<Tensor> nf4_unpack(std::shared_ptr<Tensor> packed, int64_t numel) {
    return unpack4_indices(packed, numel, DataType::NF4);
}

std::shared_ptr<Tensor> nf4_weight_only_linear(std::shared_ptr<Tensor> x,
                                               std::shared_ptr<Tensor> packed_w,
                                               std::shared_ptr<Tensor> w_scale,
                                               std::shared_ptr<Tensor> bias) {
    return pack4_weight_only_linear(x, packed_w, w_scale, bias, DataType::NF4);
}

std::shared_ptr<Tensor> fp4_pack(std::shared_ptr<Tensor> x) {
    return pack4_indices(x, DataType::FP4_E2M1);
}

std::shared_ptr<Tensor> fp4_unpack(std::shared_ptr<Tensor> packed, int64_t numel) {
    return unpack4_indices(packed, numel, DataType::FP4_E2M1);
}

std::shared_ptr<Tensor> fp4_weight_only_linear(std::shared_ptr<Tensor> x,
                                               std::shared_ptr<Tensor> packed_w,
                                               std::shared_ptr<Tensor> w_scale,
                                               std::shared_ptr<Tensor> bias) {
    return pack4_weight_only_linear(x, packed_w, w_scale, bias, DataType::FP4_E2M1);
}

}
}
