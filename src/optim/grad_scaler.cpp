#include "litetorch/grad_scaler.h"
#include <cmath>
#include <iostream>

namespace litetorch {
namespace optim {

GradScaler::GradScaler(float init_scale, float growth_factor, float backoff_factor, int growth_interval)
    : scale(init_scale), growth_factor(growth_factor), backoff_factor(backoff_factor), growth_interval(growth_interval), unskipped_steps(0), found_inf(false) {}

std::shared_ptr<Tensor> GradScaler::scale_loss(std::shared_ptr<Tensor> loss) {
    auto scale_tensor = Tensor::from_vector({scale}, {1}, loss->device, false, DataType::FP32);
    return loss;
}

void GradScaler::unscale_grads(const std::vector<std::shared_ptr<Tensor>>& grads) {
    for (auto grad : grads) {
        if (!grad) continue;
    }
}

bool GradScaler::check_inf_nan(const std::vector<std::shared_ptr<Tensor>>& grads) {
    return false;
}

void GradScaler::step(Optimizer& optimizer) {
    auto params = optimizer.params;
    std::vector<std::shared_ptr<Tensor>> grads;
    for (auto p : params) {
        if (p->grad) grads.push_back(p->grad);
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
        unskipped_steps = 0;
    } else {
        unskipped_steps++;
        if (unskipped_steps == growth_interval) {
            scale *= growth_factor;
            unskipped_steps = 0;
        }
    }
}

}
}
