#include "litetorch/ops.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace litetorch {
namespace Ops {

float clip_grad_norm_(const std::vector<std::shared_ptr<Tensor>>& params, float max_norm, float norm_type) {
    double total_sum = 0.0;
    bool has_inf = std::isinf(norm_type);

    for (auto& p : params) {
        if (!p || !p->grad) continue;
        auto vec = p->grad->to_vector();
        if (has_inf) {
            for (float val : vec) {
                total_sum = std::max(total_sum, static_cast<double>(std::abs(val)));
            }
        } else if (norm_type == 2.0f) {
            for (float val : vec) {
                total_sum += val * val;
            }
        } else if (norm_type == 1.0f) {
            for (float val : vec) {
                total_sum += std::abs(val);
            }
        } else {
            for (float val : vec) {
                total_sum += std::pow(std::abs(val), norm_type);
            }
        }
    }

    float total_norm = 0.0f;
    if (has_inf) {
        total_norm = total_sum;
    } else {
        total_norm = std::pow(total_sum, 1.0 / norm_type);
    }

    float clip_coef = max_norm / (total_norm + 1e-6f);
    if (clip_coef < 1.0f) {
        for (auto& p : params) {
            if (!p || !p->grad) continue;
            auto scale_t = Tensor::from_vector({clip_coef}, {1}, p->grad->device);
            auto scaled = Ops::mul(p->grad, scale_t);
            p->grad->copy_(scaled);
        }
    }

    return total_norm;
}

}
}
