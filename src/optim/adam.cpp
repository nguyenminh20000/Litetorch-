#include "litetorch/optim.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
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

    auto native = BackendDispatcher::get().get_backend();
    bool use_foreach = false;
    typedef void (*AdamForeachFn)(void**, int*, void**, int*, void**, int*, void**, int*, int*, int,
        float, float, float, float, float, float, float, int);
    AdamForeachFn foreach_fn = nullptr;
    if (native && native->is_available()) {
        foreach_fn = reinterpret_cast<AdamForeachFn>(native->get_kernel("", "", "adam_foreach"));
        use_foreach = (foreach_fn != nullptr);
    }
    std::vector<void*> P_list, G_list, M_list, V_list;
    std::vector<int> p_offs, g_offs, m_offs, v_offs, sizes;
    std::vector<std::shared_ptr<Tensor>> foreach_keepalive;
    std::vector<std::shared_ptr<StorageImpl>> foreach_storages;
    int max_size = 0;
    if (use_foreach) {
        P_list.reserve(params.size());
        G_list.reserve(params.size());
        M_list.reserve(params.size());
        V_list.reserve(params.size());
        p_offs.reserve(params.size());
        g_offs.reserve(params.size());
        m_offs.reserve(params.size());
        v_offs.reserve(params.size());
        sizes.reserve(params.size());
        foreach_keepalive.reserve(params.size());
        foreach_storages.reserve(params.size() * 4);
    }

    for (size_t i = 0; i < params.size(); ++i) {
        auto p = params[i];
        if (!p || !p->grad) continue;
        auto g = p->grad;

        if (!p->is_contiguous()) {
            throw std::runtime_error("[litetorch Error] Optimizer parameter must be contiguous");
        }
        auto g_c = g->is_contiguous() ? g : g->contiguous();

        if (p->device.type == DeviceType::GPU) {
            if (use_foreach) {
                P_list.push_back(p->gpu_data());
                G_list.push_back(g_c->gpu_data());
                M_list.push_back(m[i]->gpu_data());
                V_list.push_back(v[i]->gpu_data());
                p_offs.push_back(p->offset);
                g_offs.push_back(g_c->offset);
                m_offs.push_back(m[i]->offset);
                v_offs.push_back(v[i]->offset);
                int sz = static_cast<int>(p->numel());
                sizes.push_back(sz);
                if (sz > max_size) max_size = sz;
                foreach_keepalive.push_back(g_c);
                foreach_storages.push_back(p->storage);
                foreach_storages.push_back(g_c->storage);
                foreach_storages.push_back(m[i]->storage);
                foreach_storages.push_back(v[i]->storage);
                continue;
            }
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

    if (use_foreach && !P_list.empty()) {
        StorageUseGuard guard(foreach_storages);
        foreach_fn(P_list.data(), p_offs.data(), G_list.data(), g_offs.data(),
                   M_list.data(), m_offs.data(), V_list.data(), v_offs.data(),
                   sizes.data(), static_cast<int>(P_list.size()),
                   beta1, beta2, lr, eps, weight_decay,
                   bias_correction1, bias_correction2, max_size);
    }
}

}
}
