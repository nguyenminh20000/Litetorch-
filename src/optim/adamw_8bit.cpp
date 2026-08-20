#include "litetorch/optim.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "optim_utils.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace litetorch {
namespace optim {

AdamW8bit::AdamW8bit(const std::vector<std::shared_ptr<Tensor>>& params, float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(params), lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay) {
    for (auto& p : params) {
        auto m_tensor = Tensor::create(p->shape, Device(DeviceType::CPU, 0), false, DataType::INT8);
        auto v_tensor = Tensor::create(p->shape, Device(DeviceType::CPU, 0), false, DataType::INT8);
        
        std::memset(m_tensor->storage->cpu_data, 0, m_tensor->numel());
        std::memset(v_tensor->storage->cpu_data, 0, v_tensor->numel());

        m.push_back(m_tensor);
        v.push_back(v_tensor);
        m_scales.push_back(1.0f);
        v_scales.push_back(1.0f);
    }
}

void AdamW8bit::step() {
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

        auto p_cpu = p->device.type == DeviceType::CPU ? p : p->to(Device(DeviceType::CPU, 0));
        auto g_cpu = g_c->device.type == DeviceType::CPU ? g_c : g_c->to(Device(DeviceType::CPU, 0));

        float* p_ptr = p_cpu->data_ptr();
        float* g_ptr = g_cpu->data_ptr();
        int8_t* m_ptr = reinterpret_cast<int8_t*>(m[i]->storage->cpu_data);
        int8_t* v_ptr = reinterpret_cast<int8_t*>(v[i]->storage->cpu_data);
        float m_scale = m_scales[i];
        float v_scale = v_scales[i];
        size_t size = p->numel();

        std::vector<float> new_m(size);
        std::vector<float> new_v(size);

        ThreadPool::get().parallel_for(0, size, [&](int64_t j) {
            float grad_val = g_ptr[j];
            float m_val = static_cast<float>(m_ptr[j]) * m_scale;
            float v_val = static_cast<float>(v_ptr[j]) * v_scale;

            m_val = beta1 * m_val + (1.0f - beta1) * grad_val;
            v_val = beta2 * v_val + (1.0f - beta2) * grad_val * grad_val;

            new_m[j] = m_val;
            new_v[j] = v_val;

            float m_hat = m_val / bias_correction1;
            float v_hat = v_val / bias_correction2;
            float update = m_hat / (std::sqrt(v_hat) + eps);

            if (weight_decay != 0.0f) {
                p_ptr[j] -= lr * (weight_decay * p_ptr[j] + update);
            } else {
                p_ptr[j] -= lr * update;
            }
        });

        float max_m = 0.0f;
        float max_v = 0.0f;
        for (size_t j = 0; j < size; ++j) {
            float abs_m = std::abs(new_m[j]);
            float abs_v = std::abs(new_v[j]);
            if (abs_m > max_m) max_m = abs_m;
            if (abs_v > max_v) max_v = abs_v;
        }

        float new_m_scale = max_m > 0.0f ? max_m / 127.0f : 1.0f;
        float new_v_scale = max_v > 0.0f ? max_v / 127.0f : 1.0f;

        for (size_t j = 0; j < size; ++j) {
            m_ptr[j] = static_cast<int8_t>(std::round(new_m[j] / new_m_scale));
            v_ptr[j] = static_cast<int8_t>(std::round(new_v[j] / new_v_scale));
        }

        m_scales[i] = new_m_scale;
        v_scales[i] = new_v_scale;

        if (p->device.type != DeviceType::CPU) {
            p->copy_(p_cpu);
        }
    }
}

}
}
