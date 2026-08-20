#include "litetorch/optim.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "optim_utils.h"
#include <cmath>

namespace litetorch {
namespace optim {

Adam::Adam(const std::vector<std::shared_ptr<Tensor>>& params, float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(params), lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay) {
    for (auto& p : params) {
        m.push_back(Tensor::zeros(p->shape, p->device));
        v.push_back(Tensor::zeros(p->shape, p->device));
    }
}

void Adam::step() {
    step_count++;
    float bias_correction1 = 1.0f - std::pow(beta1, step_count);
    float bias_correction2 = 1.0f - std::pow(beta2, step_count);

    for (size_t i = 0; i < params.size(); ++i) {
        auto p = params[i];
        if (!p || !p->grad) continue;
        auto g = p->grad;

        if (!p->is_contiguous()) {
            throw std::runtime_error("[litetorch Error] Optimizer parameter must be contiguous");
        }
        auto g_c = g->is_contiguous() ? g : g->contiguous();

        if (p->device.type == DeviceType::GPU) {
            StorageUseGuard guard({p->storage, g_c->storage, m[i]->storage, v[i]->storage});
            cl_mem p_mem = p->gpu_data();
            int p_off = p->offset;
            cl_mem g_mem = g_c->gpu_data();
            int g_off = g_c->offset;
            cl_mem m_mem = m[i]->gpu_data();
            int m_off = m[i]->offset;
            cl_mem v_mem = v[i]->gpu_data();
            int v_off = v[i]->offset;
            int size = p->numel();

            auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "adam_step_kernel");
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                {&p_mem, &p_off, &g_mem, &g_off, &m_mem, &m_off, &v_mem, &v_off, &beta1, &beta2, &lr, &eps, &weight_decay, &bias_correction1, &bias_correction2, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(int)});
        } else {
            float* p_ptr = p->data_ptr();
            float* g_ptr = g_c->data_ptr();
            float* m_ptr = m[i]->data_ptr();
            float* v_ptr = v[i]->data_ptr();
            size_t size = p->numel();

            ThreadPool::get().parallel_for(0, size, [&](int64_t j) {
                float grad_val = g_ptr[j];
                if (weight_decay != 0.0f) {
                    grad_val += weight_decay * p_ptr[j];
                }
                m_ptr[j] = beta1 * m_ptr[j] + (1.0f - beta1) * grad_val;
                v_ptr[j] = beta2 * v_ptr[j] + (1.0f - beta2) * grad_val * grad_val;

                float m_hat = m_ptr[j] / bias_correction1;
                float v_hat = v_ptr[j] / bias_correction2;
                p_ptr[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            });
        }
    }
}

}
}
