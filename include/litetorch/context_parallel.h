#ifndef LITETORCH_CONTEXT_PARALLEL_H
#define LITETORCH_CONTEXT_PARALLEL_H

#include "litetorch/tensor.h"

namespace litetorch {
namespace distributed {

std::shared_ptr<Tensor> context_parallel_forward(std::shared_ptr<Tensor> input, int cp_group_size);
std::shared_ptr<Tensor> context_parallel_backward(std::shared_ptr<Tensor> grad_output, int cp_group_size);

}
}

#endif
