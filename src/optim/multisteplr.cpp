#include "litetorch/optim.h"
#include <algorithm>

namespace litetorch {
namespace optim {

MultiStepLR::MultiStepLR(Optimizer* optimizer, const std::vector<int>& milestones, float gamma)
    : optimizer(optimizer), milestones(milestones), gamma(gamma), base_lr(optimizer->get_lr()) {}

void MultiStepLR::step() {
    last_epoch++;
    if (std::find(milestones.begin(), milestones.end(), last_epoch) != milestones.end()) {
        optimizer->set_lr(optimizer->get_lr() * gamma);
    }
}

double MultiStepLR::get_last_lr() const {
    return optimizer->get_lr();
}

std::map<std::string, double> MultiStepLR::state_dict() const {
    return {{"last_epoch", last_epoch}, {"gamma", gamma}, {"base_lr", base_lr},
            {"num_milestones", static_cast<double>(milestones.size())}};
}

void MultiStepLR::load_state_dict(const std::map<std::string, double>& state) {
    auto it = state.find("last_epoch");
    if (it != state.end()) last_epoch = static_cast<int>(it->second);
    it = state.find("gamma");
    if (it != state.end()) gamma = static_cast<float>(it->second);
    it = state.find("base_lr");
    if (it != state.end()) base_lr = it->second;
}

}
}
