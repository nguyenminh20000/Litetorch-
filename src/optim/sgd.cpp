#include "litetorch/optim.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "optim_utils.h"

namespace litetorch {
namespace optim {

SGD::SGD(const std::vector<std::shared_ptr<Tensor>>& params, float lr, float momentum, float weight_decay)
    : Optimizer(params), lr(lr), momentum(momentum), weight_decay(weight_decay) {
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
        auto g_c = g->is_contiguous() ? g : g->contiguous();

        if (p->device.type == DeviceType::GPU) {
            StorageUseGuard guard({p->storage, g_c->storage});
            cl_mem p_mem = p->gpu_data();
            int p_off = p->offset;
            cl_mem g_mem = g_c->gpu_data();
            int g_off = g_c->offset;
            int size = p->numel();

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

            auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "sgd_step_kernel");
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                {&p_mem, &p_off, &g_mem, &g_off, &v_mem, &v_off, &has_momentum, &momentum, &lr, &weight_decay, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(float), sizeof(float), sizeof(float), sizeof(int)});
        } else {
            float* p_ptr = p->data_ptr();
            float* g_ptr = g_c->data_ptr();
            size_t size = p->numel();

            if (momentum != 0.0f) {
                float* v_ptr = velocity[i]->data_ptr();
                ThreadPool::get().parallel_for(0, size, [&](int64_t j) {
                    float grad_val = g_ptr[j];
                    if (weight_decay != 0.0f) {
                        grad_val += weight_decay * p_ptr[j];
                    }
                    v_ptr[j] = momentum * v_ptr[j] + grad_val;
                    p_ptr[j] -= lr * v_ptr[j];
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
    }
}

}
}
