#include "litetorch/optim.h"

namespace litetorch {
namespace optim {

Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>>& params) : params(params) {}

void Optimizer::zero_grad() {
    for (auto& p : params) {
        if (p) p->zero_grad();
    }
}

}
}
