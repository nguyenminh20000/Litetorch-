#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <stdexcept>

namespace litetorch {
namespace nn {

std::shared_ptr<Tensor> MSELoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("MSELoss requires both input and target");
}

std::shared_ptr<Tensor> MSELoss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    return Ops::mse_loss(input, target);
}

std::shared_ptr<Tensor> CrossEntropyLoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("CrossEntropyLoss requires both input and target");
}

std::shared_ptr<Tensor> CrossEntropyLoss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    return Ops::cross_entropy_loss(input, target);
}

std::shared_ptr<Tensor> BCELoss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("BCELoss requires both input and target");
}

std::shared_ptr<Tensor> BCELoss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    return Ops::bce_loss(input, target);
}

std::shared_ptr<Tensor> L1Loss::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("L1Loss requires both input and target");
}

std::shared_ptr<Tensor> L1Loss::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    return Ops::l1_loss(input, target);
}

}
}
