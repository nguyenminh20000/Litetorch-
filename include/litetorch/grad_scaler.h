#ifndef LITETORCH_GRAD_SCALER_H
#define LITETORCH_GRAD_SCALER_H

#include "litetorch/tensor.h"
#include "litetorch/optim.h"

namespace litetorch {
namespace optim {

class GradScaler {
public:
    float scale;
    float backoff_factor;
    float growth_factor;
    int growth_interval;
    int unskipped_steps;

    GradScaler(float init_scale = 65536.0f, float growth_factor = 2.0f, float backoff_factor = 0.5f, int growth_interval = 2000);

    std::shared_ptr<Tensor> scale_loss(std::shared_ptr<Tensor> loss);
    void step(Optimizer& optimizer);
    void update();

private:
    bool check_inf_nan(const std::vector<std::shared_ptr<Tensor>>& grads);
    void unscale_grads(const std::vector<std::shared_ptr<Tensor>>& grads);
    bool found_inf;
};

}
}

#endif
