#ifndef LITETORCH_SERIALIZATION_H
#define LITETORCH_SERIALIZATION_H

#include "litetorch/tensor.h"
#include <vector>
#include <string>
#include <memory>

namespace litetorch {

namespace optim {
class Optimizer;
}

void save_parameters(const std::vector<std::shared_ptr<Tensor>>& params, const std::string& filepath);
void load_parameters(const std::vector<std::shared_ptr<Tensor>>& params, const std::string& filepath);

void save_optimizer_state(optim::Optimizer* opt, const std::string& filepath);
void load_optimizer_state(optim::Optimizer* opt, const std::string& filepath);

}

#endif
