#include "../common/tpu_common.h"
#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace litetorch {
namespace tpu_internal {

void tpu_adamw_update(float* p, const float* g, float* m, float* v, int64_t size,
                      float lr, float beta1, float beta2, float eps, float weight_decay,
                      float bias_correction1, float bias_correction2) {
    if (!p || !g || !m || !v || size <= 0) return;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < size; ++i) {
        float grad_val = g[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad_val;
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad_val * grad_val;

        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;
        float update = m_hat / (std::sqrt(v_hat) + eps);
        if (weight_decay != 0.0f) {
            p[i] -= lr * (weight_decay * p[i] + update);
        } else {
            p[i] -= lr * update;
        }
    }
}

}
}
