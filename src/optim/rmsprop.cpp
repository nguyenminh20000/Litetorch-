#include "litetorch/optim.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "optim_utils.h"
#include <cmath>

namespace litetorch {
namespace optim {

RMSprop::RMSprop(const std::vector<std::shared_ptr<Tensor>>& params, float lr, float alpha, float eps, float weight_decay)
    : Optimizer(params), lr(lr), alpha(alpha), eps(eps), weight_decay(weight_decay) {
    for (auto& p : params) {
        square_avg.push_back(Tensor::zeros(p->shape, p->device));
    }
}

void RMSprop::step() {
    for (size_t i = 0; i < params.size(); ++i) {
        auto p = params[i];
        if (!p || !p->grad) continue;
        auto g = p->grad;

        if (!p->is_contiguous()) {
            throw std::runtime_error("[litetorch Error] Optimizer parameter must be contiguous");
        }
        auto g_c = g->is_contiguous() ? g : g->contiguous();

        if (p->device.type == DeviceType::GPU) {
            StorageUseGuard guard({p->storage, g_c->storage, square_avg[i]->storage});
            cl_mem p_mem = p->gpu_data();
            int p_off = p->offset;
            cl_mem g_mem = g_c->gpu_data();
            int g_off = g_c->offset;
            cl_mem sq_mem = square_avg[i]->gpu_data();
            int sq_off = square_avg[i]->offset;
            int size = p->numel();

            auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "rmsprop_step_kernel");
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                {&p_mem, &p_off, &g_mem, &g_off, &sq_mem, &sq_off, &alpha, &lr, &eps, &weight_decay, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(int)});
        } else {
            float* p_ptr = p->data_ptr();
            float* g_ptr = g_c->data_ptr();
            float* sq_ptr = square_avg[i]->data_ptr();
            size_t size = p->numel();

            ThreadPool::get().parallel_for(0, size, [&](int64_t j) {
                float grad_val = g_ptr[j];
                if (weight_decay != 0.0f) {
                    grad_val += weight_decay * p_ptr[j];
                }
                sq_ptr[j] = alpha * sq_ptr[j] + (1.0f - alpha) * grad_val * grad_val;
                p_ptr[j] -= lr * grad_val / (std::sqrt(sq_ptr[j]) + eps);
            });
        }
    }
}

}
}
