#include "litetorch/optim.h"
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace litetorch {
namespace optim {

LambdaLR::LambdaLR(Optimizer* optimizer, std::function<double(int)> lr_lambda)
    : optimizer(optimizer), lr_lambda(lr_lambda) {
    base_lr = optimizer->get_lr();
}

void LambdaLR::step() {
    last_epoch++;
    optimizer->set_lr(base_lr * lr_lambda(last_epoch));
}

double LambdaLR::get_last_lr() const { return optimizer->get_lr(); }

LinearLR::LinearLR(Optimizer* optimizer, float start_factor, float end_factor, int total_iters)
    : optimizer(optimizer), start_factor(start_factor), end_factor(end_factor), total_iters(total_iters) {
    base_lr = optimizer->get_lr();
}

void LinearLR::step() {
    last_epoch++;
    double factor;
    if (last_epoch >= total_iters) factor = end_factor;
    else factor = start_factor + (end_factor - start_factor) * last_epoch / total_iters;
    optimizer->set_lr(base_lr * factor);
}

double LinearLR::get_last_lr() const { return optimizer->get_lr(); }

PolynomialLR::PolynomialLR(Optimizer* optimizer, int total_iters, float power)
    : optimizer(optimizer), total_iters(total_iters), power(power) {
    base_lr = optimizer->get_lr();
}

void PolynomialLR::step() {
    last_epoch++;
    double decay;
    if (last_epoch >= total_iters) decay = 0.0;
    else decay = std::pow(1.0 - (double)last_epoch / total_iters, power);
    optimizer->set_lr(base_lr * decay);
}

double PolynomialLR::get_last_lr() const { return optimizer->get_lr(); }

CosineAnnealingWarmRestarts::CosineAnnealingWarmRestarts(Optimizer* optimizer, int T_0, int T_mult, float eta_min)
    : optimizer(optimizer), T_0(T_0), T_mult(T_mult), eta_min(eta_min), T_i(T_0) {
    if (T_0 <= 0) throw std::runtime_error("[litetorch Error] CosineAnnealingWarmRestarts: T_0 must be positive");
    if (T_mult < 1) throw std::runtime_error("[litetorch Error] CosineAnnealingWarmRestarts: T_mult must be >= 1");
    base_lr = optimizer->get_lr();
}

void CosineAnnealingWarmRestarts::step() {
    last_epoch++;
    T_cur++;
    if (T_cur >= T_i) {
        T_cur = 0;
        T_i = T_i * T_mult;
    }
    double new_lr = eta_min + 0.5 * (base_lr - eta_min) * (1.0 + std::cos(M_PI * T_cur / T_i));
    optimizer->set_lr(new_lr);
}

double CosineAnnealingWarmRestarts::get_last_lr() const { return optimizer->get_lr(); }

OneCycleLR::OneCycleLR(Optimizer* optimizer, double max_lr, int total_steps, int epochs,
                       int steps_per_epoch, float pct_start, const std::string& anneal_strategy,
                       float div_factor, float final_div_factor)
    : optimizer(optimizer), max_lr(max_lr), pct_start(pct_start), anneal_strategy(anneal_strategy),
      div_factor(div_factor), final_div_factor(final_div_factor) {
    if (anneal_strategy != "cos" && anneal_strategy != "linear")
        throw std::runtime_error("[litetorch Error] OneCycleLR: anneal_strategy must be 'cos' or 'linear'");
    if (total_steps > 0) {
        this->total_steps = total_steps;
    } else if (epochs > 0 && steps_per_epoch > 0) {
        this->total_steps = epochs * steps_per_epoch;
    } else {
        throw std::runtime_error("[litetorch Error] OneCycleLR: provide total_steps or (epochs and steps_per_epoch)");
    }
    initial_lr = max_lr / div_factor;
    min_lr = initial_lr / final_div_factor;
    step_size_up = (int)(this->total_steps * pct_start);
    if (step_size_up < 1) step_size_up = 1;
    optimizer->set_lr(initial_lr);
}

static double anneal_cos(double start, double end, double pct) {
    return end + 0.5 * (start - end) * (1.0 + std::cos(M_PI * pct));
}

static double anneal_linear(double start, double end, double pct) {
    return start + (end - start) * pct;
}

void OneCycleLR::step() {
    last_epoch++;
    int step_num = last_epoch;
    if (step_num > total_steps) step_num = total_steps;
    double new_lr;
    if (step_num <= step_size_up) {
        double pct = (double)step_num / step_size_up;
        new_lr = (anneal_strategy == "cos") ? anneal_cos(initial_lr, max_lr, pct)
                                            : anneal_linear(initial_lr, max_lr, pct);
    } else {
        int down_steps = total_steps - step_size_up;
        double pct = down_steps > 0 ? (double)(step_num - step_size_up) / down_steps : 1.0;
        new_lr = (anneal_strategy == "cos") ? anneal_cos(max_lr, min_lr, pct)
                                            : anneal_linear(max_lr, min_lr, pct);
    }
    optimizer->set_lr(new_lr);
}

double OneCycleLR::get_last_lr() const { return optimizer->get_lr(); }

}
}
