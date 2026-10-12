#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <stdexcept>

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

CrossEntropyLoss::CrossEntropyLoss(std::shared_ptr<Tensor> weight, int64_t ignore_index, const std::string& reduction)
    : weight(weight), ignore_index(ignore_index), reduction(reduction) {
    check_reduction(reduction);
}

std::shared_ptr<Tensor> CrossEntropyLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("CrossEntropyLoss requires both input and target");
}

std::shared_ptr<Tensor> CrossEntropyLoss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    return Ops::cross_entropy_loss(input, target, weight, ignore_index, reduction);
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

}
}
