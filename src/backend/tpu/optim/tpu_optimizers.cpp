#include "../common/tpu_common.h"
#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace litetorch {
namespace tpu_internal {

void tpu_adamw_update(float* p, const float* g, float* m, float* v, int64_t size,
                      float lr_t, float beta1, float beta2, float eps, float weight_decay) {
    if (!p || !g || !m || !v || size <= 0) return;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < size; ++i) {
        p[i] -= lr_t * weight_decay * p[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
        p[i] -= lr_t * m[i] / (std::sqrt(v[i]) + eps);
    }
}

}
}
