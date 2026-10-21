#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/backend_route.h"
#include "litetorch/thread_pool.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "litetorch/storage_guard.h"
namespace litetorch {
namespace {

}

namespace Ops {

std::shared_ptr<Tensor> int4_pack(std::shared_ptr<Tensor> x) {
    auto xc = x->device.type == DeviceType::CPU ? x : x->to(Device(DeviceType::CPU));
    if (xc->dtype != DataType::INT8) {
        if (xc->dtype == DataType::FP32) {
            xc = xc->cast(DataType::INT8);
        } else {
            throw std::runtime_error("[litetorch Error] int4_pack: expects INT8 or FP32 input");
        }
    }
    auto xcc = xc->is_contiguous() ? xc : xc->contiguous();
    int64_t n = xcc->numel();
    int64_t packed_n = (n + 1) / 2;
    auto out = Tensor::create({packed_n}, x->device, false, DataType::INT8);
    StorageUseGuard guard({xcc->storage, out->storage});
    int8_t* src = (int8_t*)xcc->storage->get_cpu_ptr();
    int8_t* dst = (int8_t*)out->storage->get_cpu_ptr();
    for (int64_t i = 0; i < packed_n; ++i) {
        int8_t lo = src[2 * i];
        int8_t hi = (2 * i + 1 < n) ? src[2 * i + 1] : 0;
        lo = std::max<int8_t>(-8, std::min<int8_t>(7, lo));
        hi = std::max<int8_t>(-8, std::min<int8_t>(7, hi));
        uint8_t ulo = (uint8_t)(lo + 8);
        uint8_t uhi = (uint8_t)(hi + 8);
        dst[i] = (int8_t)((uhi << 4) | (ulo & 0x0F));
    }
    if (x->device.type != DeviceType::CPU) {
        auto out_gpu = out->to(x->device);
        return out_gpu;
    }
    return out;
}

std::shared_ptr<Tensor> int4_unpack(std::shared_ptr<Tensor> packed, int64_t numel) {
    if (numel <= 0)
        throw std::runtime_error("[litetorch Error] int4_unpack: numel must be positive");
    auto pc = packed->device.type == DeviceType::CPU ? packed : packed->to(Device(DeviceType::CPU));
    auto pcc = pc->is_contiguous() ? pc : pc->contiguous();
    auto out = Tensor::create({numel}, packed->device, false, DataType::INT8);
    StorageUseGuard guard({pcc->storage, out->storage});
    int8_t* src = (int8_t*)pcc->storage->get_cpu_ptr();
    int8_t* dst = (int8_t*)out->storage->get_cpu_ptr();
    for (int64_t i = 0; i < numel; ++i) {
        uint8_t byte = (uint8_t)src[i / 2];
        uint8_t nibble = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        dst[i] = (int8_t)nibble - 8;
    }
    if (packed->device.type != DeviceType::CPU) return out->to(packed->device);
    return out;
}

std::shared_ptr<Tensor> int4_weight_only_linear(std::shared_ptr<Tensor> x,
                                                 std::shared_ptr<Tensor> packed_w,
                                                 std::shared_ptr<Tensor> w_scale,
                                                 std::shared_ptr<Tensor> bias) {
    if (x->shape.size() != 2)
        throw std::runtime_error("[litetorch Error] int4_weight_only_linear: x must be 2D");
    int64_t M = x->shape[0];
    int64_t K = x->shape[1];
    int64_t N = w_scale->shape[0];
    int64_t packed_k = (K + 1) / 2;
    if (packed_w->shape.size() != 2 || packed_w->shape[0] != N || packed_w->shape[1] != packed_k)
        throw std::runtime_error("[litetorch Error] int4_weight_only_linear: packed_w shape mismatch");
    auto xc = x->device.type == DeviceType::CPU ? x : x->to(Device(DeviceType::CPU), DataType::FP32);
    if (xc->dtype != DataType::FP32) xc = xc->cast(DataType::FP32);
    xc = xc->is_contiguous() ? xc : xc->contiguous();
    auto sc = w_scale->device.type == DeviceType::CPU ? w_scale : w_scale->to(Device(DeviceType::CPU), DataType::FP32);
    if (sc->dtype != DataType::FP32) sc = sc->cast(DataType::FP32);
    auto w_unpacked = int4_unpack(packed_w->reshape({N * packed_k}), N * K);
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
}
