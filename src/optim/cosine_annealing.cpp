#include "litetorch/optim.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace litetorch {
namespace optim {

CosineAnnealingLR::CosineAnnealingLR(Optimizer* optimizer, int T_max, float eta_min)
    : optimizer(optimizer), T_max(T_max), eta_min(eta_min) {
    base_lr = optimizer->get_lr();
}

void CosineAnnealingLR::step() {
    last_epoch++;
    float cos_val = std::cos(M_PI * last_epoch / T_max);
    float new_lr = eta_min + 0.5f * (base_lr - eta_min) * (1.0f + cos_val);
    optimizer->set_lr(new_lr);
}

}
}
