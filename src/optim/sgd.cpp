#include "litetorch/optim.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend_route.h"
#include "optim_utils.h"

namespace litetorch {
namespace optim {

SGD::SGD(const std::vector<std::shared_ptr<Tensor>>& params, double lr, float momentum,
         float dampening, float weight_decay, bool nesterov)
    : Optimizer(params), lr(lr), momentum(momentum), dampening(dampening),
      nesterov(nesterov), weight_decay(weight_decay) {
    if (momentum != 0.0f) {
        for (auto& p : params) {
            velocity.push_back(Tensor::zeros(p->shape, p->device));
        }
    }
}

void SGD::step() {
    for (size_t i = 0; i < params.size(); ++i) {
        auto p = params[i];
        if (!p || !p->grad) continue;
        auto g = p->grad;

        if (!p->is_contiguous()) {
            throw std::runtime_error("[litetorch Error] Optimizer parameter must be contiguous");
        }
        bool need_cast = (p->dtype != DataType::FP32);
        auto p32 = need_cast ? p->cast(DataType::FP32) : p;
        auto g32 = g->cast(DataType::FP32);
        auto g_c = g32->is_contiguous() ? g32 : g32->contiguous();

        if (p32->device.type == DeviceType::GPU) {
            StorageUseGuard guard({p32->storage, g_c->storage});
            cl_mem p_mem = p32->gpu_data();
            int p_off = p32->offset;
            cl_mem g_mem = g_c->gpu_data();
            int g_off = g_c->offset;
            int size = p32->numel();

            cl_mem v_mem = nullptr;
            int v_off = 0;
            int has_momentum = 0;
            if (momentum != 0.0f) {
                has_momentum = 1;
                v_mem = velocity[i]->gpu_data();
                v_off = velocity[i]->offset;
                guard.storages.push_back(velocity[i]->storage);
                velocity[i]->storage->in_use = true;
            }

            auto kernel = rt_gpu_get_kernel("litetorch_kernels", litetorch_kernels_src, "sgd_step_kernel");
            int nesterov_val = nesterov ? 1 : 0;
            float lr_f = static_cast<float>(lr);
            rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                {&p_mem, &p_off, &g_mem, &g_off, &v_mem, &v_off, &has_momentum, &momentum, &dampening, &nesterov_val, &lr_f, &weight_decay, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(float), sizeof(float), sizeof(int), sizeof(float), sizeof(float), sizeof(int)});
        } else {
            float* p_ptr = p32->data_ptr();
            float* g_ptr = g_c->data_ptr();
            size_t size = p32->numel();

            if (momentum != 0.0f) {
                float* v_ptr = velocity[i]->data_ptr();
                ThreadPool::get().parallel_for(0, size, [&](int64_t j) {
                    float d_p = g_ptr[j];
                    if (weight_decay != 0.0f) {
                        d_p += weight_decay * p_ptr[j];
                    }
                    float buf = momentum * v_ptr[j] + (1.0f - dampening) * d_p;
                    v_ptr[j] = buf;
                    float upd = nesterov ? (d_p + momentum * buf) : buf;
                    p_ptr[j] -= lr * upd;
                });
            } else {
                ThreadPool::get().parallel_for(0, size, [&](int64_t j) {
                    float grad_val = g_ptr[j];
                    if (weight_decay != 0.0f) {
                        grad_val += weight_decay * p_ptr[j];
                    }
                    p_ptr[j] -= lr * grad_val;
                });
            }
        }

        if (need_cast) {
            p->copy_(p32);
        }
    }
}

}
}
