#include "litetorch/optim.h"

namespace litetorch {
namespace optim {

LinearWarmup::LinearWarmup(Optimizer* optimizer, int warmup_steps, int total_steps, float min_lr)
    : optimizer(optimizer), warmup_steps(warmup_steps), total_steps(total_steps), min_lr(min_lr) {
    base_lr = optimizer->get_lr();
}

void LinearWarmup::step() {
    last_step++;
    double new_lr;
    if (last_step <= warmup_steps) {
        new_lr = base_lr * static_cast<double>(last_step) / static_cast<double>(warmup_steps);
    } else {
        int decay_steps = total_steps - warmup_steps;
        if (decay_steps <= 0) {
            new_lr = base_lr;
        } else {
            int s = last_step - warmup_steps;
            if (s >= decay_steps) {
                new_lr = min_lr;
            } else {
                new_lr = base_lr - (base_lr - min_lr) * static_cast<double>(s) / static_cast<double>(decay_steps);
            }
        }
    }
    optimizer->set_lr(new_lr);
}

double LinearWarmup::get_lr() const {
    return optimizer->get_lr();
}

}
}
