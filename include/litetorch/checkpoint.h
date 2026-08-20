#ifndef LITETORCH_CHECKPOINT_H
#define LITETORCH_CHECKPOINT_H

#include "litetorch/tensor.h"
#include "litetorch/autograd.h"
#include <functional>
#include <memory>
#include <vector>

namespace litetorch {

using CheckpointFunction = std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>)>;

std::shared_ptr<Tensor> checkpoint(CheckpointFunction function, std::shared_ptr<Tensor> input);

}

#endif
