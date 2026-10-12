#include "litetorch/optim.h"
#include "litetorch/thread_pool.h"
#include <cmath>
#include <stdexcept>

namespace litetorch {
namespace optim {

NAdam::NAdam(const std::vector<std::shared_ptr<Tensor>>& params, float lr,
             std::pair<float, float> betas, float eps, float weight_decay, float momentum_decay)
    : Optimizer(params), lr(lr), beta1(betas.first), beta2(betas.second), eps(eps),
      weight_decay(weight_decay), momentum_decay(momentum_decay) {
    param_groups.push_back({params, lr, weight_decay});
    for (auto& p : params) {
        exp_avg.push_back(Tensor::zeros(p->shape, p->device));
        exp_avg_sq.push_back(Tensor::zeros(p->shape, p->device));
    }
}

void NAdam::add_param_group(const std::vector<std::shared_ptr<Tensor>>& new_params, float group_lr,
                            float group_weight_decay) {
    param_groups.push_back({new_params, group_lr, group_weight_decay});
    for (auto& p : new_params) {
        params.push_back(p);
        exp_avg.push_back(Tensor::zeros(p->shape, p->device));
        exp_avg_sq.push_back(Tensor::zeros(p->shape, p->device));
    }
}

float NAdam::get_lr() const {
    if (!param_groups.empty()) return param_groups[0].lr;
    return lr;
}

void NAdam::set_lr(float new_lr) {
    lr = new_lr;
    for (auto& g : param_groups) g.lr = new_lr;
}

void NAdam::step() {
    step_count++;
    float bias_correction1 = 1.0f - std::pow(beta1, step_count);
    float bias_correction2 = 1.0f - std::pow(beta2, step_count);
    float bias_correction2_sqrt = std::sqrt(bias_correction2);
    float mu = beta1 * (1.0f - 0.5f * std::pow(0.96f, step_count * momentum_decay));
    float mu_next = beta1 * (1.0f - 0.5f * std::pow(0.96f, (step_count + 1) * momentum_decay));
    float mu_product = mu_next / mu;
    float one_minus_mu = 1.0f - mu;

    size_t state_idx = 0;
    for (auto& group : param_groups) {
        float step_size = group.lr / bias_correction1;
        float wd = group.weight_decay;
        float b1 = beta1;
        float b2 = beta2;
        float e = eps;
        for (size_t i = 0; i < group.params.size(); ++i, ++state_idx) {
            auto p = group.params[i];
            if (!p || !p->grad) continue;
            if (p->device.type == DeviceType::GPU) {
                throw std::runtime_error("[litetorch Error] NAdam: GPU step not implemented, use CPU tensors");
            }
            if (!p->is_contiguous()) {
                throw std::runtime_error("[litetorch Error] NAdam: parameter must be contiguous");
            }
            auto g = p->grad;
            auto g_c = g->is_contiguous() ? g : g->contiguous();
            float* p_ptr = p->data_ptr();
            float* g_ptr = g_c->data_ptr();
            float* m_ptr = exp_avg[state_idx]->data_ptr();
            float* v_ptr = exp_avg_sq[state_idx]->data_ptr();
            size_t n = p->numel();
            ThreadPool::get().parallel_for(0, n, [&](int64_t j) {
                float gv = g_ptr[j];
                if (wd != 0.0f) gv += wd * p_ptr[j];
                float m = m_ptr[j] = b1 * m_ptr[j] + (1.0f - b1) * gv;
                float v = v_ptr[j] = b2 * v_ptr[j] + (1.0f - b2) * gv * gv;
                float numer = m * mu_product + one_minus_mu * gv;
                float denom = (std::sqrt(v) + e) / bias_correction2_sqrt;
                p_ptr[j] -= step_size * numer / denom;
            });
        }
    }
}

}
}
