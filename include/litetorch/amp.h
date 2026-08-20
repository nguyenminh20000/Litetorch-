#ifndef LITETORCH_AMP_H
#define LITETORCH_AMP_H

#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/optim.h"
#include <cmath>
#include <vector>
#include <memory>
#include <stdexcept>

namespace litetorch {
namespace amp {

class GradScaler {
public:
    GradScaler(float init_scale = 65536.0f, float growth_factor = 2.0f, float backoff_factor = 0.5f, int growth_interval = 2000)
        : scale_(init_scale), growth_factor_(growth_factor), backoff_factor_(backoff_factor), growth_interval_(growth_interval) {}

    std::shared_ptr<Tensor> scale(std::shared_ptr<Tensor> loss) {
        auto scale_tensor = Tensor::from_vector({scale_}, {}, loss->device);
        return Ops::mul(loss, scale_tensor);
    }

    void step(optim::Optimizer& optimizer) {
        bool has_nan_or_inf = false;
        for (auto& p : optimizer.params) {
            if (p->grad) {
                std::vector<float> grads = p->grad->to_vector();
                for (float g : grads) {
                    if (std::isnan(g) || std::isinf(g)) {
                        has_nan_or_inf = true;
                        break;
                    }
                }
                if (has_nan_or_inf) break;
            }
        }

        if (!has_nan_or_inf) {
            for (auto& p : optimizer.params) {
                if (p->grad) {
                    auto scale_tensor = Tensor::from_vector({scale_}, {}, p->grad->device);
                    auto unscaled = Ops::div(p->grad, scale_tensor);
                    p->grad->copy_(unscaled);
                }
            }
            optimizer.step();
            skipped_last_step_ = false;
        } else {
            skipped_last_step_ = true;
        }
    }

    void update() {
        if (skipped_last_step_) {
            scale_ *= backoff_factor_;
            if (scale_ < 1.0f) scale_ = 1.0f;
            consecutive_good_steps_ = 0;
        } else {
            consecutive_good_steps_++;
            if (consecutive_good_steps_ >= growth_interval_) {
                scale_ *= growth_factor_;
                consecutive_good_steps_ = 0;
            }
        }
    }

    float get_scale() const { return scale_; }

private:
    float scale_;
    float growth_factor_;
    float backoff_factor_;
    int growth_interval_;
    int consecutive_good_steps_ = 0;
    bool skipped_last_step_ = false;
};

class AutocastGuard {
public:
    AutocastGuard(bool enabled = true, DataType dtype = DataType::FP16) {
        prev_enabled_ = enabled_;
        prev_dtype_ = dtype_;
        enabled_ = enabled;
        dtype_ = dtype;
    }
    ~AutocastGuard() {
        enabled_ = prev_enabled_;
        dtype_ = prev_dtype_;
    }
    static bool is_enabled() { return enabled_; }
    static DataType get_dtype() { return dtype_; }

private:
    static thread_local bool enabled_;
    static thread_local DataType dtype_;
    bool prev_enabled_;
    DataType prev_dtype_;
};

}
}

#endif
