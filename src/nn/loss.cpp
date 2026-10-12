#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include <stdexcept>
#include <cmath>
#include <limits>
#include <algorithm>

namespace litetorch {
namespace nn {

namespace {
void check_reduction(const std::string& reduction) {
    if (reduction != "mean" && reduction != "sum" && reduction != "none") {
        throw std::invalid_argument("[litetorch Error] reduction must be 'mean', 'sum' or 'none', got '" + reduction + "'");
    }
}

std::shared_ptr<Tensor> apply_sum_reduction(std::shared_ptr<Tensor> mean_out, size_t numel) {
    auto scale = Tensor::from_vector({static_cast<float>(numel)}, {1}, mean_out->device, false, mean_out->dtype);
    return Ops::mul(mean_out, scale);
}
}

MSELoss::MSELoss(const std::string& reduction) : reduction(reduction) {
    check_reduction(reduction);
}

std::shared_ptr<Tensor> MSELoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("MSELoss requires both input and target");
}

std::shared_ptr<Tensor> MSELoss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    if (reduction == "none") {
        auto diff = Ops::sub(input, target);
        return Ops::mul(diff, diff);
    }
    auto out = Ops::mse_loss(input, target);
    if (reduction == "sum") return apply_sum_reduction(out, input->numel());
    return out;
}

CrossEntropyLoss::CrossEntropyLoss(std::shared_ptr<Tensor> weight, int64_t ignore_index, const std::string& reduction, float label_smoothing)
    : weight(weight), ignore_index(ignore_index), reduction(reduction), label_smoothing(label_smoothing) {
    check_reduction(reduction);
    if (label_smoothing < 0.0f || label_smoothing > 1.0f) {
        throw std::invalid_argument("[litetorch Error] CrossEntropyLoss label_smoothing must be in [0, 1]");
    }
}

std::shared_ptr<Tensor> CrossEntropyLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("CrossEntropyLoss requires both input and target");
}

std::shared_ptr<Tensor> CrossEntropyLoss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    return Ops::cross_entropy_loss(input, target, weight, ignore_index, reduction, label_smoothing);
}

BCELoss::BCELoss(const std::string& reduction) : reduction(reduction) {
    check_reduction(reduction);
}

std::shared_ptr<Tensor> BCELoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("BCELoss requires both input and target");
}

std::shared_ptr<Tensor> BCELoss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    if (reduction == "none") {
        auto x = Ops::clamp(input, 1e-7f, 1.0f - 1e-7f);
        auto one = Tensor::from_vector({1.0f}, {1}, input->device);
        auto term1 = Ops::mul(target, Ops::log(x));
        auto term2 = Ops::mul(Ops::sub(one, target), Ops::log(Ops::sub(one, x)));
        return Ops::neg(Ops::add(term1, term2));
    }
    auto out = Ops::bce_loss(input, target);
    if (reduction == "sum") return apply_sum_reduction(out, input->numel());
    return out;
}

L1Loss::L1Loss(const std::string& reduction) : reduction(reduction) {
    check_reduction(reduction);
}

std::shared_ptr<Tensor> L1Loss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("L1Loss requires both input and target");
}

std::shared_ptr<Tensor> L1Loss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    if (reduction == "none") {
        return Ops::abs(Ops::sub(input, target));
    }
    auto out = Ops::l1_loss(input, target);
    if (reduction == "sum") return apply_sum_reduction(out, input->numel());
    return out;
}

NLLLoss::NLLLoss(std::shared_ptr<Tensor> weight, int64_t ignore_index, const std::string& reduction)
    : weight(weight), ignore_index(ignore_index), reduction(reduction) {
    check_reduction(reduction);
}

std::shared_ptr<Tensor> NLLLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("NLLLoss requires both input and target");
}

std::shared_ptr<Tensor> NLLLoss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    return Ops::nll_loss(input, target, weight, ignore_index, reduction);
}

BCEWithLogitsLoss::BCEWithLogitsLoss(std::shared_ptr<Tensor> weight, const std::string& reduction,
                                     std::shared_ptr<Tensor> pos_weight)
    : weight(weight), reduction(reduction), pos_weight(pos_weight) {
    check_reduction(reduction);
}

std::shared_ptr<Tensor> BCEWithLogitsLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("BCEWithLogitsLoss requires both input and target");
}

std::shared_ptr<Tensor> BCEWithLogitsLoss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    return Ops::bce_with_logits_loss(input, target, weight, reduction, pos_weight);
}

// === W3-T4 ===

namespace {
void check_kl_reduction(const std::string& reduction) {
    if (reduction != "mean" && reduction != "batchmean" && reduction != "sum" && reduction != "none") {
        throw std::invalid_argument("[litetorch Error] KLDivLoss reduction must be 'mean', 'batchmean', 'sum' or 'none', got '" + reduction + "'");
    }
}

std::shared_ptr<Tensor> make_scalar(float v, const Device& dev) {
    return Tensor::from_vector({v}, {1}, dev, false, DataType::FP32);
}
}

KLDivLoss::KLDivLoss(const std::string& reduction) : reduction(reduction) {
    check_kl_reduction(reduction);
}

std::shared_ptr<Tensor> KLDivLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("KLDivLoss requires both input and target");
}

std::shared_ptr<Tensor> KLDivLoss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    auto y = Ops::clamp(target, 1e-10f, 1e10f);
    auto log_y = Ops::log(y);
    auto diff = Ops::sub(log_y, input);
    auto elem = Ops::mul(target, diff);
    if (reduction == "none") return elem;
    if (reduction == "sum") return Ops::sum(elem);
    auto total = Ops::sum(elem);
    if (reduction == "batchmean") {
        int64_t n = input->shape.empty() ? 1 : input->shape[0];
        return Ops::div(total, make_scalar(static_cast<float>(n), input->device));
    }
    return Ops::div(total, make_scalar(static_cast<float>(input->numel()), input->device));
}

SmoothL1Loss::SmoothL1Loss(float beta, const std::string& reduction) : beta(beta), reduction(reduction) {
    check_reduction(reduction);
    if (beta <= 0.0f) throw std::invalid_argument("[litetorch Error] SmoothL1Loss beta must be positive");
}

std::shared_ptr<Tensor> SmoothL1Loss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("SmoothL1Loss requires both input and target");
}

std::shared_ptr<Tensor> SmoothL1Loss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    auto diff = Ops::sub(input, target);
    auto abs_diff = Ops::abs(diff);
    auto beta_t = Tensor::from_vector({beta}, {1}, input->device, false, input->dtype);
    auto half_beta = Tensor::from_vector({0.5f * beta}, {1}, input->device, false, input->dtype);
    auto quad = Ops::div(Ops::mul(diff, diff), Tensor::from_vector({2.0f * beta}, {1}, input->device, false, input->dtype));
    auto lin = Ops::sub(abs_diff, half_beta);
    auto elem = Ops::where(Ops::lt(abs_diff, beta_t), quad, lin);
    if (reduction == "none") return elem;
    auto out = Ops::mean(elem);
    if (reduction == "sum") return apply_sum_reduction(out, input->numel());
    return out;
}

HuberLoss::HuberLoss(float delta, const std::string& reduction) : delta(delta), reduction(reduction) {
    check_reduction(reduction);
    if (delta <= 0.0f) throw std::invalid_argument("[litetorch Error] HuberLoss delta must be positive");
}

std::shared_ptr<Tensor> HuberLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("HuberLoss requires both input and target");
}

std::shared_ptr<Tensor> HuberLoss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    SmoothL1Loss base(delta, reduction);
    return base.forward(input, target);
}

namespace {
const float NEG_INF_F = -1e30f;

float logaddexp_f(float a, float b) {
    if (a <= NEG_INF_F / 2) return b;
    if (b <= NEG_INF_F / 2) return a;
    float m = a > b ? a : b;
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

struct CTCBatchResult {
    float loss;
    std::vector<float> grad;
    bool is_inf;
};

CTCBatchResult ctc_forward_backward(const float* lp, int64_t T, int64_t C,
                                    const std::vector<int>& labels, int blank) {
    CTCBatchResult r;
    int64_t S = static_cast<int64_t>(labels.size());
    int64_t L = 2 * S + 1;
    std::vector<int> ext(L);
    for (int64_t i = 0; i < S; ++i) { ext[2 * i] = blank; ext[2 * i + 1] = labels[i]; }
    ext[L - 1] = blank;

    std::vector<float> alpha(T * L, NEG_INF_F), beta(T * L, NEG_INF_F);
    auto A = [&](int64_t t, int64_t s) -> float& { return alpha[t * L + s]; };
    auto B = [&](int64_t t, int64_t s) -> float& { return beta[t * L + s]; };
    auto LP = [&](int64_t t, int c) -> float { return lp[t * C + c]; };

    A(0, 0) = LP(0, blank);
    if (S > 0) A(0, 1) = LP(0, labels[0]);
    for (int64_t t = 1; t < T; ++t) {
        int64_t s_max = std::min(L - 1, 2 * t + 1);
        for (int64_t s = 0; s <= s_max; ++s) {
            float v = A(t - 1, s);
            if (s > 0) v = logaddexp_f(v, A(t - 1, s - 1));
            if (s > 1 && ext[s] != blank && ext[s] != ext[s - 2]) v = logaddexp_f(v, A(t - 1, s - 2));
            A(t, s) = LP(t, ext[s]) + v;
        }
    }
    float log_total = (S == 0) ? A(T - 1, 0) : logaddexp_f(A(T - 1, L - 1), A(T - 1, L - 2));
    r.is_inf = !(log_total > NEG_INF_F / 2);
    r.loss = r.is_inf ? std::numeric_limits<float>::infinity() : -log_total;

    r.grad.assign(T * C, 0.0f);
    if (!r.is_inf) {
        B(T - 1, L - 1) = LP(T - 1, blank);
        if (S > 0) B(T - 1, L - 2) = LP(T - 1, ext[L - 2]);
        for (int64_t t = T - 2; t >= 0; --t) {
            int64_t s_min = std::max<int64_t>(0, L - 2 * (T - 1 - t) - 1);
            for (int64_t s = L - 1; s >= s_min; --s) {
                float v = B(t + 1, s);
                if (s + 1 < L) v = logaddexp_f(v, B(t + 1, s + 1));
                if (s + 2 < L && ext[s] != blank && ext[s] != ext[s + 2]) v = logaddexp_f(v, B(t + 1, s + 2));
                B(t, s) = LP(t, ext[s]) + v;
            }
        }
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t s = 0; s < L; ++s) {
                float a = A(t, s), b = B(t, s);
                if (a <= NEG_INF_F / 2 || b <= NEG_INF_F / 2) continue;
                float post = std::exp(a + b - LP(t, ext[s]) - log_total);
                r.grad[t * C + ext[s]] -= post;
            }
        }
    }
    return r;
}

class CTCNode : public Node {
public:
    std::vector<CTCBatchResult> batch_results;
    std::vector<int64_t> input_lengths;
    std::vector<int64_t> target_lengths;
    int64_t T, N, C;
    std::string reduction;
    bool zero_infinity;

    CTCNode(const std::vector<CTCBatchResult>& br, const std::vector<int64_t>& il,
            const std::vector<int64_t>& tl, int64_t T_, int64_t N_, int64_t C_,
            const std::string& red, bool zi)
        : Node("CTCNode"), batch_results(br), input_lengths(il), target_lengths(tl),
          T(T_), N(N_), C(C_), reduction(red), zero_infinity(zi) {}

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto g = Tensor::zeros({T, N, C}, Device(DeviceType::CPU, 0), false, DataType::FP32);
        float* gp = g->data_ptr();
        std::vector<float> go = grad_output->to_vector();
        float out_scale = go.empty() ? 1.0f : go[0];
        for (int64_t n = 0; n < N; ++n) {
            float scale = (reduction == "none") ? (n < (int64_t)go.size() ? go[n] : 1.0f) : out_scale;
            if (reduction == "mean") scale /= static_cast<float>(N);
            const auto& br = batch_results[n];
            if (br.is_inf && zero_infinity) continue;
            for (int64_t t = 0; t < input_lengths[n]; ++t)
                for (int64_t c = 0; c < C; ++c)
                    gp[(t * N + n) * C + c] = br.grad[t * C + c] * scale;
        }
        return {g};
    }
};
}

CTCLoss::CTCLoss(int blank, const std::string& reduction, bool zero_infinity)
    : blank(blank), reduction(reduction), zero_infinity(zero_infinity) {
    check_reduction(reduction);
    if (blank < 0) throw std::invalid_argument("[litetorch Error] CTCLoss blank must be non-negative");
}

std::shared_ptr<Tensor> CTCLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("CTCLoss requires log_probs, targets, input_lengths, target_lengths");
}

std::shared_ptr<Tensor> CTCLoss::forward(std::shared_ptr<Tensor> log_probs, std::shared_ptr<Tensor> targets,
                                         std::shared_ptr<Tensor> input_lengths, std::shared_ptr<Tensor> target_lengths) {
    if (log_probs->shape.size() != 3) throw std::runtime_error("[litetorch Error] CTCLoss log_probs must be (T, N, C)");
    int64_t T = log_probs->shape[0], N = log_probs->shape[1], C = log_probs->shape[2];
    if (blank >= C) throw std::runtime_error("[litetorch Error] CTCLoss blank must be < num classes");
    auto lp_cpu = log_probs->to(Device(DeviceType::CPU, 0));
    const float* lp = lp_cpu->data_ptr();
    std::vector<float> il_f = input_lengths->to_vector(), tl_f = target_lengths->to_vector();
    if ((int64_t)il_f.size() != N || (int64_t)tl_f.size() != N)
        throw std::runtime_error("[litetorch Error] CTCLoss input_lengths/target_lengths must have size N");
    std::vector<float> tgt = targets->to_vector();
    bool tgt_1d = targets->shape.size() == 1;

    std::vector<CTCBatchResult> results;
    results.reserve(N);
    std::vector<float> losses;
    losses.reserve(N);
    std::vector<int64_t> il(N), tl(N);
    for (int64_t n = 0; n < N; ++n) {
        il[n] = (int64_t)il_f[n]; tl[n] = (int64_t)tl_f[n];
        if (il[n] > T || tl[n] < 0) throw std::runtime_error("[litetorch Error] CTCLoss invalid input/target length");
        std::vector<int> labels;
        labels.reserve(tl[n]);
        if (tgt_1d) {
            int64_t off = 0;
            for (int64_t m = 0; m < n; ++m) off += (int64_t)tl_f[m];
            for (int64_t i = 0; i < tl[n]; ++i) labels.push_back((int)tgt[off + i]);
        } else {
            int64_t Smax = targets->shape[1];
            for (int64_t i = 0; i < tl[n]; ++i) labels.push_back((int)tgt[n * Smax + i]);
        }
        for (int lab : labels)
            if (lab < 0 || lab >= C || lab == blank) throw std::runtime_error("[litetorch Error] CTCLoss target label out of range or equals blank");
        std::vector<float> lp_n(il[n] * C);
        for (int64_t t = 0; t < il[n]; ++t)
            for (int64_t c = 0; c < C; ++c) lp_n[t * C + c] = lp[(t * N + n) * C + c];
        auto r = ctc_forward_backward(lp_n.data(), il[n], C, labels, blank);
        if (r.is_inf && zero_infinity) { r.loss = 0.0f; }
        results.push_back(std::move(r));
        losses.push_back(results.back().loss);
    }

    std::shared_ptr<Tensor> out;
    bool out_requires_grad = log_probs->requires_grad && Autograd::is_grad_enabled();
    if (reduction == "none") {
        out = Tensor::from_vector(losses, {N}, log_probs->device, out_requires_grad, DataType::FP32);
    } else {
        float total = 0.0f;
        for (float v : losses) total += v;
        if (reduction == "mean") total /= static_cast<float>(N);
        out = Tensor::from_vector({total}, {1}, log_probs->device, out_requires_grad, DataType::FP32);
    }

    if (out_requires_grad) {
        auto node = std::make_shared<CTCNode>(results, il, tl, T, N, C, reduction, zero_infinity);
        node->inputs = {{log_probs, true}};
        node->next_nodes = {log_probs->creator};
        node->saved_tensors = {log_probs};
        out->creator = node;
        node->output = out;
    }
    return out;
}

// === end W3-T4 ===

}
}
