#include "litetorch/grad_scaler.h"
#include "litetorch/ops.h"
#include <cmath>

namespace litetorch {
namespace optim {

GradScaler::GradScaler(float init_scale, float growth_factor, float backoff_factor, int growth_interval, bool enabled)
    : scale_(init_scale), growth_factor(growth_factor), backoff_factor(backoff_factor), growth_interval(growth_interval), unskipped_steps(0), enabled(enabled), found_inf(false) {}

std::shared_ptr<Tensor> GradScaler::scale(std::shared_ptr<Tensor> loss) {
    if (!enabled || !loss) return loss;
    auto scale_tensor = Tensor::from_vector({scale_}, {}, loss->device, false, loss->dtype);
    return Ops::mul(loss, scale_tensor);
}

std::shared_ptr<Tensor> GradScaler::scale_loss(std::shared_ptr<Tensor> loss) {
    return scale(loss);
}

bool GradScaler::check_inf_nan(const std::vector<std::shared_ptr<Tensor>>& grads) {
    for (const auto& g : grads) {
        if (!g) continue;
        if (g->device.type == DeviceType::CPU) {
            float* ptr = g->data_ptr();
            size_t total = g->numel();
            for (size_t i = 0; i < total; ++i) {
                if (std::isnan(ptr[i]) || std::isinf(ptr[i])) {
                    return true;
                }
            }
        } else {
            auto max_abs = Ops::max(Ops::abs(g));
            float val = max_abs->item();
            if (std::isnan(val) || std::isinf(val)) {
                return true;
            }
        }
    }
    return false;
}

void GradScaler::unscale_grads(const std::vector<std::shared_ptr<Tensor>>& grads) {
    for (auto& g : grads) {
        if (!g) continue;
        // Always unscale in FP32: a scale > 65504 overflows FP16 to inf,
        // which would silently zero all gradients (div by inf = 0).
        auto scale_tensor = Tensor::from_vector({scale_}, {}, g->device, false, DataType::FP32);
        auto g_fp32 = (g->dtype == DataType::FP32) ? g : g->cast(DataType::FP32);
        auto unscaled = Ops::div(g_fp32, scale_tensor);
        g->copy_(unscaled->cast(g->dtype));
    }
}

void GradScaler::unscale_(Optimizer& optimizer) {
    if (!enabled) return;
    std::vector<std::shared_ptr<Tensor>> grads;
    for (auto& p : optimizer.params) {
        if (p && p->grad) {
            grads.push_back(p->grad);
        }
    }
    unscale_grads(grads);
}

void GradScaler::step(Optimizer& optimizer) {
    if (!enabled) {
        optimizer.step();
        return;
    }
    std::vector<std::shared_ptr<Tensor>> grads;
    for (auto& p : optimizer.params) {
        if (p && p->grad) {
            grads.push_back(p->grad);
        }
    }

    unscale_(optimizer);
    found_inf = check_inf_nan(grads);
    if (found_inf) {
        return;
    }

    optimizer.step();
}

void GradScaler::update() {
    if (!enabled) return;
    if (found_inf) {
        scale_ *= backoff_factor;
        if (scale_ < 1.0f) {
            scale_ = 1.0f;
        }
        unskipped_steps = 0;
    } else {
        unskipped_steps++;
        if (unskipped_steps >= growth_interval) {
            scale_ *= growth_factor;
            unskipped_steps = 0;
        }
    }
}

}
}
