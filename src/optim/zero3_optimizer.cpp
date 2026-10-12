#include "litetorch/zero3_optimizer.h"
#include "litetorch/backend.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend_route.h"
#include "optim_utils.h"
#include <cmath>

namespace litetorch {
namespace optim {

ZeRO3Optimizer::ZeRO3Optimizer(const std::vector<std::shared_ptr<Tensor>>& params, float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(params), lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay) {}

void ZeRO3Optimizer::step() {
    t++;
    float bias_correction1 = 1.0f - std::pow(beta1, t);
    float bias_correction2 = 1.0f - std::pow(beta2, t);
    int rank = 0;
    int world_size = 1;
    if (distributed::ProcessGroup::get().is_initialized()) {
        rank = distributed::ProcessGroup::get().get_rank();
        world_size = distributed::ProcessGroup::get().get_world_size();
    }

    std::vector<std::shared_ptr<Tensor>> sharded_params;
    std::vector<std::shared_ptr<Tensor>> replicated_params;
    for (auto p : params) {
        if (p->shard) {
            sharded_params.push_back(p);
        } else {
            replicated_params.push_back(p);
        }
    }

    for (auto p : sharded_params) {
        auto active_p = p->shard;
        if (!active_p->grad) continue;

        if (m.find(active_p.get()) == m.end()) {
            m[active_p.get()] = Tensor::zeros(active_p->shape, active_p->device, false);
            v[active_p.get()] = Tensor::zeros(active_p->shape, active_p->device, false);
        }

        bool need_cast = (active_p->dtype != DataType::FP32);
        auto p32 = need_cast ? active_p->cast(DataType::FP32) : active_p;
        auto g32 = active_p->grad->cast(DataType::FP32);
        auto grad_c = g32->is_contiguous() ? g32 : g32->contiguous();
        size_t active_size = p32->numel();

        if (p32->device.type == DeviceType::GPU) {
            auto backend = BackendDispatcher::get().get_backend();
            if (backend && backend->is_available()) {
                backend->adamw_step(p32->gpu_data(), p32->offset, grad_c->gpu_data(), grad_c->offset,
                                    m[active_p.get()]->gpu_data(), m[active_p.get()]->offset,
                                    v[active_p.get()]->gpu_data(), v[active_p.get()]->offset,
                                    active_size, lr, beta1, beta2, eps, weight_decay,
                                    bias_correction1, bias_correction2);
            } else {
                StorageUseGuard guard({p32->storage, grad_c->storage, m[active_p.get()]->storage, v[active_p.get()]->storage});
                cl_mem p_mem = p32->gpu_data();
                int p_off = p32->offset;
                cl_mem g_mem = grad_c->gpu_data();
                int g_off = grad_c->offset;
                cl_mem m_mem = m[active_p.get()]->gpu_data();
                int m_off = m[active_p.get()]->offset;
                cl_mem v_mem = v[active_p.get()]->gpu_data();
                int v_off = v[active_p.get()]->offset;
                int size = active_size;
                auto kernel = rt_gpu_get_kernel("litetorch_kernels", litetorch_kernels_src, "adamw_step_kernel");
                if (kernel) {
                    rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                        {&p_mem, &p_off, &g_mem, &g_off, &m_mem, &m_off, &v_mem, &v_off, &beta1, &beta2, &lr, &eps, &weight_decay, &bias_correction1, &bias_correction2, &size},
                        {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(int)});
                }
            }
        } else {
            float* p_ptr = p32->data_ptr();
            float* g_ptr = grad_c->data_ptr();
            float* m_ptr = m[active_p.get()]->data_ptr();
            float* v_ptr = v[active_p.get()]->data_ptr();
            for (size_t i = 0; i < active_size; ++i) {
                float g = g_ptr[i];
                m_ptr[i] = beta1 * m_ptr[i] + (1.0f - beta1) * g;
                v_ptr[i] = beta2 * v_ptr[i] + (1.0f - beta2) * g * g;
                float m_hat = m_ptr[i] / bias_correction1;
                float v_hat = v_ptr[i] / bias_correction2;
                float update = m_hat / (std::sqrt(v_hat) + eps);
                p_ptr[i] -= lr * (weight_decay * p_ptr[i] + update);
            }
        }

        if (need_cast) {
            active_p->copy_(p32);
        }
    }

    for (size_t idx = 0; idx < replicated_params.size(); ++idx) {
        auto p = replicated_params[idx];
        int owner_rank = idx % world_size;

        if (rank == owner_rank) {
            if (!p->grad) continue;

            if (m.find(p.get()) == m.end()) {
                m[p.get()] = Tensor::zeros(p->shape, p->device, false);
                v[p.get()] = Tensor::zeros(p->shape, p->device, false);
            }

            bool need_cast = (p->dtype != DataType::FP32);
            auto p32 = need_cast ? p->cast(DataType::FP32) : p;
            auto g32 = p->grad->cast(DataType::FP32);
            auto grad_c = g32->is_contiguous() ? g32 : g32->contiguous();
            size_t active_size = p32->numel();

            if (p32->device.type == DeviceType::GPU) {
                auto backend = BackendDispatcher::get().get_backend();
                if (backend && backend->is_available()) {
                    backend->adamw_step(p32->gpu_data(), p32->offset, grad_c->gpu_data(), grad_c->offset,
                                        m[p.get()]->gpu_data(), m[p.get()]->offset,
                                        v[p.get()]->gpu_data(), v[p.get()]->offset,
                                        active_size, lr, beta1, beta2, eps, weight_decay,
                                        bias_correction1, bias_correction2);
                } else {
                    StorageUseGuard guard({p32->storage, grad_c->storage, m[p.get()]->storage, v[p.get()]->storage});
                    cl_mem p_mem = p32->gpu_data();
                    int p_off = p32->offset;
                    cl_mem g_mem = grad_c->gpu_data();
                    int g_off = grad_c->offset;
                    cl_mem m_mem = m[p.get()]->gpu_data();
                    int m_off = m[p.get()]->offset;
                    cl_mem v_mem = v[p.get()]->gpu_data();
                    int v_off = v[p.get()]->offset;
                    int size = active_size;
                    auto kernel = rt_gpu_get_kernel("litetorch_kernels", litetorch_kernels_src, "adamw_step_kernel");
                    if (kernel) {
                        rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                            {&p_mem, &p_off, &g_mem, &g_off, &m_mem, &m_off, &v_mem, &v_off, &beta1, &beta2, &lr, &eps, &weight_decay, &bias_correction1, &bias_correction2, &size},
                            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(int)});
                    }
                }
            } else {
                float* p_ptr = p32->data_ptr();
                float* g_ptr = grad_c->data_ptr();
                float* m_ptr = m[p.get()]->data_ptr();
                float* v_ptr = v[p.get()]->data_ptr();
                for (size_t i = 0; i < active_size; ++i) {
                    float g = g_ptr[i];
                    m_ptr[i] = beta1 * m_ptr[i] + (1.0f - beta1) * g;
                    v_ptr[i] = beta2 * v_ptr[i] + (1.0f - beta2) * g * g;
                    float m_hat = m_ptr[i] / bias_correction1;
                    float v_hat = v_ptr[i] / bias_correction2;
                    float update = m_hat / (std::sqrt(v_hat) + eps);
                    p_ptr[i] -= lr * (weight_decay * p_ptr[i] + update);
                }
            }

            if (need_cast) {
                p->copy_(p32);
            }
        }

        if (distributed::ProcessGroup::get().is_initialized()) {
            distributed::ProcessGroup::get().broadcast(p, owner_rank);
        }
    }
}

void ZeRO3Optimizer::zero_grad() {
    for (auto p : params) {
        if (p->grad) p->grad = nullptr;
        if (p->shard && p->shard->grad) p->shard->grad = nullptr;
    }
}

}
}
