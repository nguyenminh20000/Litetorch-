#ifndef LITETORCH_NN_UTILS_H
#define LITETORCH_NN_UTILS_H

#include "litetorch/nn.h"

namespace litetorch {
namespace nn {
namespace {
void kaiming_uniform(std::shared_ptr<Tensor> tensor, float a = 0.0) {
    init::kaiming_uniform_(tensor, a, "fan_in", "relu");
}
}
}
}
#endif
