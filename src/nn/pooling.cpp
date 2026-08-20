#include "litetorch/nn.h"
#include "litetorch/ops.h"

namespace litetorch {
namespace nn {

MaxPool2d::MaxPool2d(int kernel_size, int stride, int padding)
    : kernel_size(kernel_size), stride(stride), padding(padding) {}

std::shared_ptr<Tensor> MaxPool2d::forward(std::shared_ptr<Tensor> input) {
    return Ops::max_pool2d(input, kernel_size, stride, padding);
}

MaxPool3d::MaxPool3d(int kernel_size, int stride, int padding)
    : kernel_size(kernel_size), stride(stride), padding(padding) {}

std::shared_ptr<Tensor> MaxPool3d::forward(std::shared_ptr<Tensor> input) {
    return Ops::max_pool3d(input, kernel_size, stride, padding);
}

AdaptiveAvgPool2d::AdaptiveAvgPool2d(int output_height, int output_width)
    : output_height(output_height), output_width(output_width) {}

std::shared_ptr<Tensor> AdaptiveAvgPool2d::forward(std::shared_ptr<Tensor> input) {
    return Ops::adaptive_avg_pool2d(input, output_height, output_width);
}

}
}
