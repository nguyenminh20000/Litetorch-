#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/backend_route.h"
#include "litetorch/thread_pool.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace litetorch {
namespace {

struct StorageUseGuard {
    std::vector<std::shared_ptr<StorageImpl>> storages;
    StorageUseGuard(const std::vector<std::shared_ptr<StorageImpl>>& list) : storages(list) {
        for (auto& s : storages) if (s) s->in_use = true;
    }
    ~StorageUseGuard() {
        for (auto& s : storages) if (s) s->in_use = false;
    }
};

}

namespace Ops {

std::shared_ptr<Tensor> fp8_cast_scaled(std::shared_ptr<Tensor> x, float scale, DataType dtype) {
    if (dtype != DataType::FP8_E4M3 && dtype != DataType::FP8_E5M2)
        throw std::runtime_error("[litetorch Error] fp8_cast_scaled: dtype must be FP8_E4M3 or FP8_E5M2");
    if (scale == 0.0f)
        throw std::runtime_error("[litetorch Error] fp8_cast_scaled: scale must be non-zero");
    auto x_cpu = x->device.type == DeviceType::CPU ? x : x->to(Device(DeviceType::CPU), DataType::FP32);
    if (x_cpu->dtype != DataType::FP32) x_cpu = x_cpu->cast(DataType::FP32);
    auto xc = x_cpu->is_contiguous() ? x_cpu : x_cpu->contiguous();
    auto scale_t = Ops::full_like(xc, scale, xc->device, DataType::FP32);
    auto scaled = Ops::div(xc, scale_t);
    auto out = scaled->cast(dtype);
    if (x->device.type != DeviceType::CPU) out = out->to(x->device);
    return out;
}

std::shared_ptr<Tensor> fp8_uncast_scaled(std::shared_ptr<Tensor> x, float scale) {
    if (x->dtype != DataType::FP8_E4M3 && x->dtype != DataType::FP8_E5M2)
        throw std::runtime_error("[litetorch Error] fp8_uncast_scaled: input must be FP8");
    auto x_cpu = x->device.type == DeviceType::CPU ? x : x->to(Device(DeviceType::CPU));
    auto fp32 = x_cpu->cast(DataType::FP32);
    auto scale_t = Ops::full_like(fp32, scale, fp32->device, DataType::FP32);
    auto out = Ops::mul(fp32, scale_t);
    if (x->device.type != DeviceType::CPU) out = out->to(x->device);
    return out;
}

std::shared_ptr<Tensor> fp8_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
                                    float a_scale, float b_scale, float out_scale,
                                    DataType out_dtype) {
    if (a->shape.size() != 2 || b->shape.size() != 2)
        throw std::runtime_error("[litetorch Error] fp8_matmul: expects 2D inputs");
    if (a->shape[1] != b->shape[0])
        throw std::runtime_error("[litetorch Error] fp8_matmul: inner dimension mismatch");
    if (out_dtype != DataType::FP8_E4M3 && out_dtype != DataType::FP8_E5M2 && out_dtype != DataType::FP32)
        throw std::runtime_error("[litetorch Error] fp8_matmul: out_dtype must be FP8 or FP32");
    auto a_fp32 = fp8_uncast_scaled(a, a_scale);
    auto b_fp32 = fp8_uncast_scaled(b, b_scale);
    auto out_fp32 = Ops::matmul(a_fp32, b_fp32);
    if (out_dtype == DataType::FP32) return out_fp32;
    return fp8_cast_scaled(out_fp32, out_scale, out_dtype);
}

}
}
