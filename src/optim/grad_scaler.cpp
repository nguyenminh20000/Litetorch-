#include "litetorch/grad_scaler.h"
#include "litetorch/ops.h"
#include <cmath>

namespace litetorch {
namespace optim {

GradScaler::GradScaler(float init_scale, float growth_factor, float backoff_factor, int growth_interval)
    : scale(init_scale), growth_factor(growth_factor), backoff_factor(backoff_factor), growth_interval(growth_interval), unskipped_steps(0), found_inf(false) {}

std::shared_ptr<Tensor> GradScaler::scale_loss(std::shared_ptr<Tensor> loss) {
    if (!loss) return nullptr;
    auto scale_tensor = Tensor::from_vector({scale}, {}, loss->device, false, loss->dtype);
    return Ops::mul(loss, scale_tensor);
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
            std::vector<float> vals = g->to_vector();
            for (float v : vals) {
                if (std::isnan(v) || std::isinf(v)) {
                    return true;
                }
            }
        }
    }
    return false;
}

void GradScaler::unscale_grads(const std::vector<std::shared_ptr<Tensor>>& grads) {
    for (auto& g : grads) {
        if (!g) continue;
        auto scale_tensor = Tensor::from_vector({scale}, {}, g->device, false, g->dtype);
        auto unscaled = Ops::div(g, scale_tensor);
        g->copy_(unscaled);
    }
}

void GradScaler::step(Optimizer& optimizer) {
    std::vector<std::shared_ptr<Tensor>> grads;
    for (auto& p : optimizer.params) {
        if (p && p->grad) {
            grads.push_back(p->grad);
        }
    }

    found_inf = check_inf_nan(grads);
    if (found_inf) {
        return;
    }

    unscale_grads(grads);
    optimizer.step();
}

void GradScaler::update() {
    if (found_inf) {
        scale *= backoff_factor;
        if (scale < 1.0f) {
            scale = 1.0f;
        }
        unskipped_steps = 0;
    } else {
        unskipped_steps++;
        if (unskipped_steps >= growth_interval) {
            scale *= growth_factor;
            unskipped_steps = 0;
        }
    }
}

}
}
