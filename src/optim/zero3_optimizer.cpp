#include "litetorch/zero3_optimizer.h"
#include "litetorch/backend.h"
#include <cmath>

namespace litetorch {
namespace optim {

ZeRO3Optimizer::ZeRO3Optimizer(const std::vector<std::shared_ptr<Tensor>>& params, float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(params), lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay) {}

void ZeRO3Optimizer::step() {
    t++;
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

        size_t active_size = active_p->numel();
        float lr_t = lr * std::sqrt(1.0f - std::pow(beta2, t)) / (1.0f - std::pow(beta1, t));

        if (active_p->device.type == DeviceType::GPU) {
            auto backend = BackendDispatcher::get().get_backend();
            if (backend && backend->is_available()) {
                backend->adamw_step(active_p->gpu_data(), active_p->offset, active_p->grad->gpu_data(), active_p->grad->offset,
                                    m[active_p.get()]->gpu_data(), m[active_p.get()]->offset,
                                    v[active_p.get()]->gpu_data(), v[active_p.get()]->offset,
                                    active_size, lr_t, beta1, beta2, eps, weight_decay);
            }
        } else {
            float* p_ptr = active_p->data_ptr();
            float* g_ptr = active_p->grad->data_ptr();
            float* m_ptr = m[active_p.get()]->data_ptr();
            float* v_ptr = v[active_p.get()]->data_ptr();
            for (size_t i = 0; i < active_size; ++i) {
                float g = g_ptr[i] + weight_decay * p_ptr[i];
                m_ptr[i] = beta1 * m_ptr[i] + (1.0f - beta1) * g;
                v_ptr[i] = beta2 * v_ptr[i] + (1.0f - beta2) * g * g;
                p_ptr[i] -= lr_t * m_ptr[i] / (std::sqrt(v_ptr[i]) + eps);
            }
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

            size_t active_size = p->numel();
            float lr_t = lr * std::sqrt(1.0f - std::pow(beta2, t)) / (1.0f - std::pow(beta1, t));

            if (p->device.type == DeviceType::GPU) {
                auto backend = BackendDispatcher::get().get_backend();
                if (backend && backend->is_available()) {
                    backend->adamw_step(p->gpu_data(), p->offset, p->grad->gpu_data(), p->grad->offset,
                                        m[p.get()]->gpu_data(), m[p.get()]->offset,
                                        v[p.get()]->gpu_data(), v[p.get()]->offset,
                                        active_size, lr_t, beta1, beta2, eps, weight_decay);
                }
            } else {
                float* p_ptr = p->data_ptr();
                float* g_ptr = p->grad->data_ptr();
                float* m_ptr = m[p.get()]->data_ptr();
                float* v_ptr = v[p.get()]->data_ptr();
                for (size_t i = 0; i < active_size; ++i) {
                    float g = g_ptr[i] + weight_decay * p_ptr[i];
                    m_ptr[i] = beta1 * m_ptr[i] + (1.0f - beta1) * g;
                    v_ptr[i] = beta2 * v_ptr[i] + (1.0f - beta2) * g * g;
                    p_ptr[i] -= lr_t * m_ptr[i] / (std::sqrt(v_ptr[i]) + eps);
                }
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
