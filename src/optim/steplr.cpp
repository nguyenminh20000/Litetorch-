#include "litetorch/optim.h"

namespace litetorch {
namespace optim {

StepLR::StepLR(Optimizer* optimizer, int step_size, float gamma)
    : optimizer(optimizer), step_size(step_size), gamma(gamma) {}

void StepLR::step() {
    last_epoch++;
    if (last_epoch % step_size == 0) {
        float current_lr = optimizer->get_lr();
        optimizer->set_lr(current_lr * gamma);
    }
}

}
}
