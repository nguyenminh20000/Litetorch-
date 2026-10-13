#include "litetorch/dpo.h"
#include "litetorch/ops.h"
#include <stdexcept>

namespace litetorch {

namespace {
void check_dpo_reduction(const std::string& reduction) {
    if (reduction != "mean" && reduction != "sum" && reduction != "none") {
        throw std::invalid_argument(
            "[litetorch Error] DPOLoss reduction must be 'mean', 'sum' or 'none', got '" +
            reduction + "'");
    }
}

std::shared_ptr<Tensor> scalar_like(float v, std::shared_ptr<Tensor> ref) {
    return Tensor::from_vector({v}, {1}, ref->device, false, ref->dtype);
}

std::shared_ptr<Tensor> log_odds(std::shared_ptr<Tensor> logp) {
    auto p = Ops::exp(logp);
    auto one_minus_p = Ops::clamp(Ops::sub(scalar_like(1.0f, logp), p), 1e-7f, 1.0f);
    return Ops::sub(logp, Ops::log(one_minus_p));
}
}

namespace nn {


DPOLoss::DPOLoss(float beta, bool reference_free, const std::string& reduction)
    : beta(beta), reference_free(reference_free), reduction(reduction) {
    check_dpo_reduction(reduction);
    if (beta <= 0.0f) {
        throw std::invalid_argument("[litetorch Error] DPOLoss beta must be positive");
    }
}

std::shared_ptr<Tensor> DPOLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error(
        "[litetorch Error] DPOLoss requires policy_chosen_logps and policy_rejected_logps");
}

std::shared_ptr<Tensor> DPOLoss::forward(std::shared_ptr<Tensor> policy_chosen_logps,
                                         std::shared_ptr<Tensor> policy_rejected_logps,
                                         std::shared_ptr<Tensor> ref_chosen_logps,
                                         std::shared_ptr<Tensor> ref_rejected_logps) {
    return functional::dpo_loss(policy_chosen_logps, policy_rejected_logps,
                                ref_chosen_logps, ref_rejected_logps,
                                beta, reference_free, reduction);
}

}

namespace functional {

std::shared_ptr<Tensor> dpo_loss(std::shared_ptr<Tensor> policy_chosen_logps,
                                 std::shared_ptr<Tensor> policy_rejected_logps,
                                 std::shared_ptr<Tensor> ref_chosen_logps,
                                 std::shared_ptr<Tensor> ref_rejected_logps,
                                 float beta,
                                 bool reference_free,
                                 const std::string& reduction) {
    if (!policy_chosen_logps || !policy_rejected_logps) {
        throw std::invalid_argument(
            "[litetorch Error] dpo_loss requires policy_chosen_logps and policy_rejected_logps");
    }
    if (reduction != "mean" && reduction != "sum" && reduction != "none") {
        throw std::invalid_argument(
            "[litetorch Error] dpo_loss reduction must be 'mean', 'sum' or 'none'");
    }
    if (beta <= 0.0f) {
        throw std::invalid_argument("[litetorch Error] dpo_loss beta must be positive");
    }

    std::shared_ptr<Tensor> logits;
    if (reference_free) {
        auto chosen_odds = log_odds(policy_chosen_logps);
        auto rejected_odds = log_odds(policy_rejected_logps);
        logits = Ops::sub(chosen_odds, rejected_odds);
    } else {
        if (!ref_chosen_logps || !ref_rejected_logps) {
            throw std::invalid_argument(
                "[litetorch Error] dpo_loss requires ref_chosen_logps and ref_rejected_logps "
                "unless reference_free=True");
        }
        auto pi_logratios = Ops::sub(policy_chosen_logps, policy_rejected_logps);
        auto ref_logratios = Ops::sub(ref_chosen_logps, ref_rejected_logps);
        auto delta = Ops::sub(pi_logratios, ref_logratios);
        logits = Ops::mul(delta, scalar_like(beta, delta));
    }

    auto losses = Ops::neg(Ops::log(Ops::sigmoid(logits)));
    if (reduction == "none") return losses;
    if (reduction == "sum") return Ops::sum(losses);
    return Ops::mean(losses);
}

}
}
