#include "litetorch/context_parallel.h"
#include "litetorch/distributed.h"

namespace litetorch {
namespace distributed {

std::shared_ptr<Tensor> context_parallel_forward(std::shared_ptr<Tensor> input, int cp_group_size) {
    return input;
}

std::shared_ptr<Tensor> context_parallel_backward(std::shared_ptr<Tensor> grad_output, int cp_group_size) {
    return grad_output;
}

}
}
