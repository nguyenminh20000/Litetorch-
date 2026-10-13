#include "litetorch/optim.h"
#include "litetorch/thread_pool.h"
#include <cmath>
#include <stdexcept>

namespace litetorch {
namespace optim {

Adamax::Adamax(const std::vector<std::shared_ptr<Tensor>>& params, double lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(params), lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay) {
    for (auto& p : params) {
        exp_avg.push_back(Tensor::zeros(p->shape, p->device));
        exp_inf.push_back(Tensor::zeros(p->shape, p->device));
    }
}

void Adamax::step() {
    step_count++;
    double bias_correction = 1.0 - std::pow((double)beta1, step_count);
    for (size_t i = 0; i < params.size(); ++i) {
        auto p = params[i];
        if (!p || !p->grad) continue;
        auto g = p->grad;
        if (!p->is_contiguous())
            throw std::runtime_error("[litetorch Error] Optimizer parameter must be contiguous");
        if (p->device.type != DeviceType::CPU)
            throw std::runtime_error("[litetorch Error] Adamax: only CPU supported");
        bool need_cast = (p->dtype != DataType::FP32);
        auto p32 = need_cast ? p->cast(DataType::FP32) : p;
        auto g32 = g->cast(DataType::FP32);
        auto g_c = g32->is_contiguous() ? g32 : g32->contiguous();
        float* p_ptr = p32->data_ptr();
        float* g_ptr = g_c->data_ptr();
        float* m_ptr = exp_avg[i]->data_ptr();
        float* u_ptr = exp_inf[i]->data_ptr();
        size_t size = p32->numel();
        double lr_d = lr;
        float b1 = beta1;
        float b2 = beta2;
        float eps_f = eps;
        float wd = weight_decay;
        double bc = bias_correction;
        ThreadPool::get().parallel_for(0, size, [&](int64_t j) {
            float grad_val = g_ptr[j];
            if (wd != 0.0f) grad_val += wd * p_ptr[j];
            m_ptr[j] = b1 * m_ptr[j] + (1.0f - b1) * grad_val;
            float ag = std::fabs(grad_val);
            float bu = b2 * u_ptr[j];
            u_ptr[j] = bu > ag ? bu : ag;
            p_ptr[j] -= (float)(lr_d / bc * m_ptr[j] / (u_ptr[j] + eps_f));
        });
        if (need_cast) p->copy_(p32);
    }
}

}
}
