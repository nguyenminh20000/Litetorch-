#include "litetorch/optim.h"

namespace litetorch {
namespace optim {

ExponentialLR::ExponentialLR(Optimizer* optimizer, float gamma)
    : optimizer(optimizer), gamma(gamma), base_lr(optimizer->get_lr()) {}

void ExponentialLR::step() {
    last_epoch++;
    optimizer->set_lr(optimizer->get_lr() * gamma);
}

double ExponentialLR::get_last_lr() const {
    return optimizer->get_lr();
}

std::map<std::string, double> ExponentialLR::state_dict() const {
    return {{"last_epoch", last_epoch}, {"gamma", gamma}, {"base_lr", base_lr}};
}

void ExponentialLR::load_state_dict(const std::map<std::string, double>& state) {
    auto it = state.find("last_epoch");
    if (it != state.end()) last_epoch = static_cast<int>(it->second);
    it = state.find("gamma");
    if (it != state.end()) gamma = static_cast<float>(it->second);
    it = state.find("base_lr");
    if (it != state.end()) base_lr = it->second;
}

}
}
