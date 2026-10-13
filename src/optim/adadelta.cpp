#include "litetorch/optim.h"
#include "litetorch/thread_pool.h"
#include <cmath>
#include <stdexcept>

namespace litetorch {
namespace optim {

Adadelta::Adadelta(const std::vector<std::shared_ptr<Tensor>>& params, double lr, float rho, float eps, float weight_decay)
    : Optimizer(params), lr(lr), rho(rho), eps(eps), weight_decay(weight_decay) {
    for (auto& p : params) {
        square_avg.push_back(Tensor::zeros(p->shape, p->device));
        acc_delta.push_back(Tensor::zeros(p->shape, p->device));
    }
}

void Adadelta::step() {
    for (size_t i = 0; i < params.size(); ++i) {
        auto p = params[i];
        if (!p || !p->grad) continue;
        auto g = p->grad;
        if (!p->is_contiguous())
            throw std::runtime_error("[litetorch Error] Optimizer parameter must be contiguous");
        if (p->device.type != DeviceType::CPU)
            throw std::runtime_error("[litetorch Error] Adadelta: only CPU supported");
        bool need_cast = (p->dtype != DataType::FP32);
        auto p32 = need_cast ? p->cast(DataType::FP32) : p;
        auto g32 = g->cast(DataType::FP32);
        auto g_c = g32->is_contiguous() ? g32 : g32->contiguous();
        float* p_ptr = p32->data_ptr();
        float* g_ptr = g_c->data_ptr();
        float* sq_ptr = square_avg[i]->data_ptr();
        float* ad_ptr = acc_delta[i]->data_ptr();
        size_t size = p32->numel();
        double lr_d = lr;
        float rho_f = rho;
        float eps_f = eps;
        float wd = weight_decay;
        ThreadPool::get().parallel_for(0, size, [&](int64_t j) {
            float grad_val = g_ptr[j];
            if (wd != 0.0f) grad_val += wd * p_ptr[j];
            sq_ptr[j] = rho_f * sq_ptr[j] + (1.0f - rho_f) * grad_val * grad_val;
            float std_ = std::sqrt(sq_ptr[j] + eps_f);
            float delta = std::sqrt(ad_ptr[j] + eps_f) / std_ * grad_val;
            ad_ptr[j] = rho_f * ad_ptr[j] + (1.0f - rho_f) * delta * delta;
            p_ptr[j] -= (float)(lr_d * delta);
        });
        if (need_cast) p->copy_(p32);
    }
}

}
}
