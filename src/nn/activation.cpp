#include "litetorch/nn.h"
#include "litetorch/ops.h"

namespace litetorch {
namespace nn {

LeakyReLU::LeakyReLU(float negative_slope) : negative_slope(negative_slope) {}

std::shared_ptr<Tensor> LeakyReLU::forward(std::shared_ptr<Tensor> input) {
    return Ops::leaky_relu(input, negative_slope);
}

std::shared_ptr<Tensor> Sigmoid::forward(std::shared_ptr<Tensor> input) {
    return Ops::sigmoid(input);
}

std::shared_ptr<Tensor> Tanh::forward(std::shared_ptr<Tensor> input) {
    return Ops::tanh(input);
}

std::shared_ptr<Tensor> ReLU::forward(std::shared_ptr<Tensor> input) {
    return Ops::relu(input);
}

Softmax::Softmax(int64_t dim) : dim(dim) {}

std::shared_ptr<Tensor> Softmax::forward(std::shared_ptr<Tensor> input) {
    return Ops::softmax(input, dim);
}

std::shared_ptr<Tensor> GELU::forward(std::shared_ptr<Tensor> input) {
    return Ops::gelu(input);
}

}
}
