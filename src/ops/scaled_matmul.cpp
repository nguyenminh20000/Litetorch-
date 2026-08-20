#include "litetorch/ops.h"
#include "litetorch/tensor.h"
#include <stdexcept>

namespace litetorch {
namespace Ops {

std::shared_ptr<Tensor> scaled_matmul(
    std::shared_ptr<Tensor> a,
    std::shared_ptr<Tensor> b,
    float a_scale,
    float b_scale,
    std::shared_ptr<Tensor> bias,
    DataType out_dtype)
{
    auto a_fp32 = a->dtype == DataType::FP32 ? a : a->cast(DataType::FP32);
    auto b_fp32 = b->dtype == DataType::FP32 ? b : b->cast(DataType::FP32);

    auto out_fp32 = matmul(a_fp32, b_fp32);

    float total_scale = a_scale * b_scale;
    if (total_scale != 1.0f) {
        float* out_ptr = out_fp32->data_ptr();
        for (size_t i = 0; i < out_fp32->numel(); ++i) {
            out_ptr[i] *= total_scale;
        }
    }

    if (bias) {
        auto bias_fp32 = bias->dtype == DataType::FP32 ? bias : bias->cast(DataType::FP32);
        float* out_ptr = out_fp32->data_ptr();
        float* bias_ptr = bias_fp32->data_ptr();
        size_t out_cols = out_fp32->shape.back();
        for (size_t i = 0; i < out_fp32->numel(); ++i) {
            out_ptr[i] += bias_ptr[i % out_cols];
        }
    }

    if (out_dtype != DataType::FP32) {
        return out_fp32->cast(out_dtype);
    }
    return out_fp32;
}

}
}
