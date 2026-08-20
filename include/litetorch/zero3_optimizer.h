#ifndef LITETORCH_ZERO3_OPTIMIZER_H
#define LITETORCH_ZERO3_OPTIMIZER_H

#include "litetorch/optim.h"
#include "litetorch/distributed.h"
#include <unordered_map>

namespace litetorch {
namespace optim {

class ZeRO3Optimizer : public Optimizer {
public:
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;

    ZeRO3Optimizer(const std::vector<std::shared_ptr<Tensor>>& params, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f);

    void step() override;
    void zero_grad();

private:
    std::unordered_map<Tensor*, std::shared_ptr<Tensor>> m;
    std::unordered_map<Tensor*, std::shared_ptr<Tensor>> v;
    int t = 0;
};

}
}

#endif
