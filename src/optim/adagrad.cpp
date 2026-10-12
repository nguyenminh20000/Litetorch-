#include "litetorch/optim.h"
#include "litetorch/thread_pool.h"
#include <cmath>
#include <stdexcept>

namespace litetorch {
namespace optim {

Adagrad::Adagrad(const std::vector<std::shared_ptr<Tensor>>& params, float lr, float lr_decay,
                 float weight_decay, float eps)
    : Optimizer(params), lr(lr), lr_decay(lr_decay), weight_decay(weight_decay), eps(eps) {
    param_groups.push_back({params, lr, weight_decay});
    for (auto& p : params) {
        sum.push_back(Tensor::zeros(p->shape, p->device));
    }
}

void Adagrad::add_param_group(const std::vector<std::shared_ptr<Tensor>>& new_params, float group_lr,
                              float group_weight_decay) {
    param_groups.push_back({new_params, group_lr, group_weight_decay});
    for (auto& p : new_params) {
        params.push_back(p);
        sum.push_back(Tensor::zeros(p->shape, p->device));
    }
}

float Adagrad::get_lr() const {
    if (!param_groups.empty()) return param_groups[0].lr;
    return lr;
}

void Adagrad::set_lr(float new_lr) {
    lr = new_lr;
    for (auto& g : param_groups) g.lr = new_lr;
}

void Adagrad::step() {
    step_count++;
    size_t state_idx = 0;
    for (auto& group : param_groups) {
        float group_clr = group.lr / (1.0f + (step_count - 1) * lr_decay);
        for (size_t i = 0; i < group.params.size(); ++i, ++state_idx) {
            auto p = group.params[i];
            if (!p || !p->grad) continue;
            if (p->device.type == DeviceType::GPU) {
                throw std::runtime_error("[litetorch Error] Adagrad: GPU step not implemented, use CPU tensors");
            }
            if (!p->is_contiguous()) {
                throw std::runtime_error("[litetorch Error] Adagrad: parameter must be contiguous");
            }
            auto g = p->grad;
            auto g_c = g->is_contiguous() ? g : g->contiguous();
            float* p_ptr = p->data_ptr();
            float* g_ptr = g_c->data_ptr();
            float* s_ptr = sum[state_idx]->data_ptr();
            size_t n = p->numel();
            float wd = group.weight_decay;
            float e = eps;
            ThreadPool::get().parallel_for(0, n, [&](int64_t j) {
                float gv = g_ptr[j];
                if (wd != 0.0f) gv += wd * p_ptr[j];
                s_ptr[j] += gv * gv;
                float denom = std::sqrt(s_ptr[j]) + e;
                p_ptr[j] -= group_clr * gv / denom;
            });
        }
    }
}

}
}
