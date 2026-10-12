#include "litetorch/optim.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace litetorch {
namespace optim {

ReduceLROnPlateau::ReduceLROnPlateau(Optimizer* optimizer, const std::string& mode, float factor,
                                     int patience, float threshold, const std::string& threshold_mode,
                                     int cooldown, float min_lr, float eps)
    : optimizer(optimizer), mode(mode), factor(factor), patience(patience), threshold(threshold),
      threshold_mode(threshold_mode), cooldown(cooldown), min_lr(min_lr), eps(eps),
      base_lr(optimizer->get_lr()) {
    if (mode != "min" && mode != "max") {
        throw std::runtime_error("[litetorch Error] ReduceLROnPlateau: mode must be 'min' or 'max'");
    }
    if (threshold_mode != "rel" && threshold_mode != "abs") {
        throw std::runtime_error("[litetorch Error] ReduceLROnPlateau: threshold_mode must be 'rel' or 'abs'");
    }
    best = (mode == "min") ? std::numeric_limits<float>::infinity()
                           : -std::numeric_limits<float>::infinity();
}

bool ReduceLROnPlateau::is_better(float a, float best_val) const {
    if (mode == "min" && threshold_mode == "rel") return a < best_val * (1.0f - threshold);
    if (mode == "min" && threshold_mode == "abs") return a < best_val - threshold;
    if (mode == "max" && threshold_mode == "rel") return a > best_val * (1.0f + threshold);
    return a > best_val + threshold;
}

void ReduceLROnPlateau::step(float metric) {
    last_epoch++;
    if (is_better(metric, best)) {
        best = metric;
        num_bad_epochs = 0;
    } else {
        num_bad_epochs++;
    }
    if (cooldown_counter > 0) {
        cooldown_counter--;
        num_bad_epochs = 0;
    }
    if (num_bad_epochs > patience) {
        double old_lr = optimizer->get_lr();
        double new_lr = std::max(old_lr * factor, static_cast<double>(min_lr));
        if (old_lr - new_lr > eps) {
            optimizer->set_lr(new_lr);
        }
        cooldown_counter = cooldown;
        num_bad_epochs = 0;
    }
}

double ReduceLROnPlateau::get_last_lr() const {
    return optimizer->get_lr();
}

std::map<std::string, double> ReduceLROnPlateau::state_dict() const {
    return {{"last_epoch", last_epoch},
            {"factor", factor},
            {"patience", patience},
            {"threshold", threshold},
            {"cooldown", cooldown},
            {"min_lr", min_lr},
            {"eps", eps},
            {"base_lr", base_lr},
            {"best", best},
            {"cooldown_counter", cooldown_counter},
            {"num_bad_epochs", num_bad_epochs},
            {"mode_min", mode == "min" ? 1.0 : 0.0},
            {"threshold_rel", threshold_mode == "rel" ? 1.0 : 0.0}};
}

void ReduceLROnPlateau::load_state_dict(const std::map<std::string, double>& state) {
    auto get = [&](const char* k, double dflt) {
        auto it = state.find(k);
        return it != state.end() ? it->second : dflt;
    };
    last_epoch = static_cast<int>(get("last_epoch", 0));
    factor = static_cast<float>(get("factor", 0.1));
    patience = static_cast<int>(get("patience", 10));
    threshold = static_cast<float>(get("threshold", 1e-4));
    cooldown = static_cast<int>(get("cooldown", 0));
    min_lr = static_cast<float>(get("min_lr", 0.0));
    eps = static_cast<float>(get("eps", 1e-8));
    base_lr = get("base_lr", base_lr);
    best = static_cast<float>(get("best", best));
    cooldown_counter = static_cast<int>(get("cooldown_counter", 0));
    num_bad_epochs = static_cast<int>(get("num_bad_epochs", 0));
    mode = get("mode_min", 1.0) > 0.5 ? "min" : "max";
    threshold_mode = get("threshold_rel", 1.0) > 0.5 ? "rel" : "abs";
}

}
}
