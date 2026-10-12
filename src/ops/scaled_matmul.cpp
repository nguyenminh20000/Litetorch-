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
    auto out = matmul(a, b);

    float total_scale = a_scale * b_scale;
    if (total_scale != 1.0f) {
        auto scale_t = Tensor::from_vector({total_scale}, {1}, out->device, false, out->dtype);
        out = mul(out, scale_t);
    }

    if (bias) {
        out = add(out, bias);
    }

    if (out_dtype != out->dtype) {
        out = out->cast(out_dtype);
    }
    return out;
}

}
}
