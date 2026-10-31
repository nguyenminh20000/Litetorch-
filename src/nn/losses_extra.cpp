#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <stdexcept>
#include <cmath>
#include <vector>

namespace litetorch {
namespace nn {

namespace {
void check_red(const std::string& reduction) {
    if (reduction != "mean" && reduction != "sum" && reduction != "none") {
        throw std::invalid_argument("[litetorch Error] reduction must be 'mean', 'sum' or 'none'");
    }
}

std::shared_ptr<Tensor> sc(float v, std::shared_ptr<Tensor> ref) {
    return Tensor::from_vector({v}, {1}, ref->device, false, ref->dtype);
}

std::shared_ptr<Tensor> apply_reduction(std::shared_ptr<Tensor> x, const std::string& reduction) {
    if (reduction == "none") return x;
    if (reduction == "sum") return Ops::sum(x);
    return Ops::mean(x);
}
}

HingeEmbeddingLoss::HingeEmbeddingLoss(float margin, const std::string& reduction)
    : margin(margin), reduction(reduction) { check_red(reduction); }

std::shared_ptr<Tensor> HingeEmbeddingLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("HingeEmbeddingLoss requires input and target");
}

std::shared_ptr<Tensor> HingeEmbeddingLoss::forward(std::shared_ptr<Tensor> input,
                                                     std::shared_ptr<Tensor> target) {
    auto m = sc(margin, input);
    auto one = sc(1.0f, input);
    auto is_pos = Ops::eq(target, one);
    auto neg_candidate = Ops::sub(m, input);
    auto neg_loss = Ops::where(Ops::gt(neg_candidate, sc(0.0f, input)),
                               neg_candidate, Ops::mul(input, sc(0.0f, input)));
    auto loss = Ops::where(is_pos, input, neg_loss);
    return apply_reduction(loss, reduction);
}

CosineEmbeddingLoss::CosineEmbeddingLoss(float margin, const std::string& reduction)
    : margin(margin), reduction(reduction) { check_red(reduction); }

std::shared_ptr<Tensor> CosineEmbeddingLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("CosineEmbeddingLoss requires input1, input2 and target");
}

std::shared_ptr<Tensor> CosineEmbeddingLoss::forward(std::shared_ptr<Tensor> input,
                                                      std::shared_ptr<Tensor> target) {
    throw std::runtime_error("CosineEmbeddingLoss requires two inputs");
}

std::shared_ptr<Tensor> CosineEmbeddingLoss::forward(std::shared_ptr<Tensor> x1,
                                                      std::shared_ptr<Tensor> x2,
                                                      std::shared_ptr<Tensor> y) {
    auto dot = Ops::sum(Ops::mul(x1, x2), 1);
    auto n1 = Ops::sqrt(Ops::sum(Ops::mul(x1, x1), 1));
    auto n2 = Ops::sqrt(Ops::sum(Ops::mul(x2, x2), 1));
    auto cos = Ops::div(dot, Ops::mul(n1, n2));
    auto one = Tensor::from_vector({1.0f}, {1}, x1->device, false, x1->dtype);
    auto zero = Tensor::from_vector({0.0f}, {1}, x1->device, false, x1->dtype);
    auto m = Tensor::from_vector({margin}, {1}, x1->device, false, x1->dtype);
    auto is_pos = Ops::eq(y, one);
    auto pos_loss = Ops::sub(one, cos);
    auto neg_candidate = Ops::sub(cos, m);
    auto neg_loss = Ops::where(Ops::gt(neg_candidate, zero), neg_candidate, zero);
    auto loss = Ops::where(is_pos, pos_loss, neg_loss);
    return apply_reduction(loss, reduction);
}

MarginRankingLoss::MarginRankingLoss(float margin, const std::string& reduction)
    : margin(margin), reduction(reduction) { check_red(reduction); }

std::shared_ptr<Tensor> MarginRankingLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("MarginRankingLoss requires input1, input2 and target");
}

std::shared_ptr<Tensor> MarginRankingLoss::forward(std::shared_ptr<Tensor> input1,
                                                    std::shared_ptr<Tensor> input2,
                                                    std::shared_ptr<Tensor> target) {
    auto diff = Ops::sub(input1, input2);
    auto prod = Ops::mul(Ops::neg(target), diff);
    auto m = sc(margin, input1);
    auto zero = sc(0.0f, input1);
    auto candidate = Ops::add(prod, m);
    auto loss = Ops::where(Ops::gt(candidate, zero), candidate, zero);
    return apply_reduction(loss, reduction);
}

MultiMarginLoss::MultiMarginLoss(double p, float margin, const std::string& reduction)
    : p(p), margin(margin), reduction(reduction) {
    check_red(reduction);
    if (p != 1 && p != 2) throw std::invalid_argument("[litetorch Error] MultiMarginLoss p must be 1 or 2");
}

std::shared_ptr<Tensor> MultiMarginLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("MultiMarginLoss requires input and target");
}

std::shared_ptr<Tensor> MultiMarginLoss::forward(std::shared_ptr<Tensor> input,
                                                  std::shared_ptr<Tensor> target) {
    if (p != 1 && p != 2) throw std::invalid_argument("[litetorch Error] MultiMarginLoss p must be 1 or 2");
    auto shape = input->shape;
    int64_t C = shape[1];
    auto m = sc(margin, input);
    auto zero = sc(0.0f, input);
    auto total = Ops::mul(Ops::narrow(input, 1, 0, 1), zero);
    auto t2d = Ops::unsqueeze(target, 1);
    for (int64_t i = 0; i < C; i++) {
        auto xi = Ops::narrow(input, 1, i, 1);
        auto is_target = Ops::eq(t2d, sc((float)i, target));
        auto diff = Ops::sub(Ops::add(m, xi), Ops::gather(input, 1, t2d));
        auto term = Ops::where(Ops::gt(diff, zero), diff, zero);
        if (p != 1) term = Ops::pow(term, (float)p);
        auto mask = Ops::where(is_target, zero, sc(1.0f, input));
        total = Ops::add(total, Ops::mul(term, mask));
    }
    auto loss = Ops::div(total, sc((float)C, input));
    if (reduction == "none") return Ops::squeeze(loss, 1);
    return apply_reduction(loss, reduction);
}

MultiLabelMarginLoss::MultiLabelMarginLoss(const std::string& reduction)
    : reduction(reduction) { check_red(reduction); }

std::shared_ptr<Tensor> MultiLabelMarginLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("MultiLabelMarginLoss requires input and target");
}

std::shared_ptr<Tensor> MultiLabelMarginLoss::forward(std::shared_ptr<Tensor> input,
                                                       std::shared_ptr<Tensor> target) {
    auto shape = input->shape;
    int64_t C = shape[1];
    auto zero = sc(0.0f, input);
    auto one = sc(1.0f, input);
    auto tz = sc(0.0f, target);
    auto col0 = Ops::mul(Ops::narrow(input, 1, 0, 1), zero);
    auto col1 = Ops::add(col0, one);
    std::vector<std::shared_ptr<Tensor>> tcol((size_t)C);
    std::vector<std::shared_ptr<Tensor>> act((size_t)C);
    std::vector<std::shared_ptr<Tensor>> xt((size_t)C);
    auto running = col1;
    for (int64_t j = 0; j < C; j++) {
        auto tj = Ops::narrow(target, 1, j, 1);
        tcol[(size_t)j] = tj;
        running = Ops::mul(running, Ops::ge(tj, tz));
        act[(size_t)j] = running;
        auto safe_tj = Ops::where(running, tj, tz);
        xt[(size_t)j] = Ops::gather(input, 1, safe_tj);
    }
    std::vector<std::shared_ptr<Tensor>> neg((size_t)C);
    for (int64_t i = 0; i < C; i++) {
        auto neg_i = col1;
        for (int64_t j = 0; j < C; j++) {
            auto ne_ij = Ops::ne(tcol[(size_t)j], sc((float)i, target));
            neg_i = Ops::mul(neg_i, Ops::where(act[(size_t)j], ne_ij, col1));
        }
        neg[(size_t)i] = neg_i;
    }
    auto total = col0;
    for (int64_t j = 0; j < C; j++) {
        auto xj = xt[(size_t)j];
        for (int64_t i = 0; i < C; i++) {
            auto xi = Ops::narrow(input, 1, i, 1);
            auto diff = Ops::sub(one, Ops::sub(xj, xi));
            auto term = Ops::where(Ops::gt(diff, zero), diff, zero);
            auto mask = Ops::mul(act[(size_t)j], neg[(size_t)i]);
            total = Ops::add(total, Ops::mul(term, mask));
        }
    }
    auto loss = Ops::div(total, sc((float)C, input));
    if (reduction == "none") return Ops::squeeze(loss, 1);
    return apply_reduction(loss, reduction);
}

SoftMarginLoss::SoftMarginLoss(const std::string& reduction)
    : reduction(reduction) { check_red(reduction); }

std::shared_ptr<Tensor> SoftMarginLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("SoftMarginLoss requires input and target");
}

std::shared_ptr<Tensor> SoftMarginLoss::forward(std::shared_ptr<Tensor> input,
                                                 std::shared_ptr<Tensor> target) {
    auto prod = Ops::mul(target, input);
    auto loss = Ops::log(Ops::add(sc(1.0f, input), Ops::exp(Ops::neg(prod))));
    return apply_reduction(loss, reduction);
}

MultiLabelSoftMarginLoss::MultiLabelSoftMarginLoss(const std::string& reduction)
    : reduction(reduction) { check_red(reduction); }

std::shared_ptr<Tensor> MultiLabelSoftMarginLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("MultiLabelSoftMarginLoss requires input and target");
}

std::shared_ptr<Tensor> MultiLabelSoftMarginLoss::forward(std::shared_ptr<Tensor> input,
                                                           std::shared_ptr<Tensor> target) {
    auto one = sc(1.0f, input);
    auto sig = Ops::sigmoid(input);
    auto term1 = Ops::mul(target, Ops::log(Ops::add(sig, sc(1e-12f, input))));
    auto term2 = Ops::mul(Ops::sub(one, target),
                          Ops::log(Ops::add(Ops::sub(one, sig), sc(1e-12f, input))));
    auto loss = Ops::neg(Ops::add(term1, term2));
    auto per_sample = Ops::mean(loss, 1);
    return apply_reduction(per_sample, reduction);
}

TripletMarginLoss::TripletMarginLoss(float margin, int p, float eps, bool swap,
                                     const std::string& reduction)
    : margin(margin), p(p), eps(eps), swap(swap), reduction(reduction) { check_red(reduction); }

std::shared_ptr<Tensor> TripletMarginLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("TripletMarginLoss requires anchor, positive and negative");
}

std::shared_ptr<Tensor> TripletMarginLoss::forward(std::shared_ptr<Tensor> anchor,
                                                    std::shared_ptr<Tensor> positive,
                                                    std::shared_ptr<Tensor> negative) {
    auto ap = Ops::pow(Ops::sum(Ops::pow(Ops::abs(Ops::sub(anchor, positive)), (float)p), 1),
                       1.0f / (float)p);
    auto an = Ops::pow(Ops::sum(Ops::pow(Ops::abs(Ops::sub(anchor, negative)), (float)p), 1),
                       1.0f / (float)p);
    if (swap) {
        auto pn = Ops::pow(Ops::sum(Ops::pow(Ops::abs(Ops::sub(positive, negative)), (float)p), 1),
                           1.0f / (float)p);
        auto min_pn = Ops::where(Ops::lt(pn, an), pn, an);
        an = min_pn;
    }
    auto diff = Ops::add(Ops::sub(ap, an), sc(margin, anchor));
    auto zero = sc(0.0f, anchor);
    auto loss = Ops::where(Ops::gt(diff, zero), diff, zero);
    return apply_reduction(loss, reduction);
}

TripletMarginWithDistanceLoss::TripletMarginWithDistanceLoss(float margin,
                                                             const std::string& reduction)
    : margin(margin), reduction(reduction) { check_red(reduction); }

std::shared_ptr<Tensor> TripletMarginWithDistanceLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("TripletMarginWithDistanceLoss requires anchor, positive and negative");
}

std::shared_ptr<Tensor> TripletMarginWithDistanceLoss::forward(std::shared_ptr<Tensor> anchor,
                                                                std::shared_ptr<Tensor> positive,
                                                                std::shared_ptr<Tensor> negative) {
    auto ap = Ops::sqrt(Ops::add(Ops::sum(Ops::mul(Ops::sub(anchor, positive),
                                                   Ops::sub(anchor, positive)), 1),
                                 sc(1e-12f, anchor)));
    auto an = Ops::sqrt(Ops::add(Ops::sum(Ops::mul(Ops::sub(anchor, negative),
                                                   Ops::sub(anchor, negative)), 1),
                                 sc(1e-12f, anchor)));
    auto diff = Ops::add(Ops::sub(ap, an), sc(margin, anchor));
    auto zero = sc(0.0f, anchor);
    auto loss = Ops::where(Ops::gt(diff, zero), diff, zero);
    return apply_reduction(loss, reduction);
}

PoissonNLLLoss::PoissonNLLLoss(bool log_input, bool full, float eps, const std::string& reduction)
    : log_input(log_input), full(full), eps(eps), reduction(reduction) { check_red(reduction); }

std::shared_ptr<Tensor> PoissonNLLLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("PoissonNLLLoss requires input and target");
}

std::shared_ptr<Tensor> PoissonNLLLoss::forward(std::shared_ptr<Tensor> input,
                                                 std::shared_ptr<Tensor> target) {
    std::shared_ptr<Tensor> loss;
    if (log_input) {
        loss = Ops::sub(Ops::exp(input), Ops::mul(target, input));
    } else {
        loss = Ops::sub(input, Ops::mul(target, Ops::log(Ops::add(input, sc(eps, input)))));
    }
    if (full) {
        auto stirling = Ops::sub(Ops::mul(target, Ops::log(Ops::add(target, sc(eps, input)))),
                                 target);
        auto tzero = Ops::eq(target, sc(0.0f, target));
        stirling = Ops::where(tzero, Ops::mul(target, sc(0.0f, target)), stirling);
        loss = Ops::add(loss, stirling);
    }
    return apply_reduction(loss, reduction);
}

GaussianNLLLoss::GaussianNLLLoss(bool full, float eps, const std::string& reduction)
    : full(full), eps(eps), reduction(reduction) { check_red(reduction); }

std::shared_ptr<Tensor> GaussianNLLLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("GaussianNLLLoss requires input, target and var");
}

std::shared_ptr<Tensor> GaussianNLLLoss::forward(std::shared_ptr<Tensor> input,
                                                  std::shared_ptr<Tensor> target,
                                                  std::shared_ptr<Tensor> var) {
    auto v = Ops::where(Ops::lt(var, sc(eps, var)), sc(eps, var), var);
    auto diff = Ops::sub(input, target);
    auto loss = Ops::add(Ops::log(v),
                         Ops::div(Ops::mul(diff, diff), v));
    loss = Ops::mul(loss, sc(0.5f, input));
    if (full) {
        loss = Ops::add(loss, sc(0.5f * std::log(2.0f * 3.141592653589793f), input));
    }
    return apply_reduction(loss, reduction);
}

}
}
