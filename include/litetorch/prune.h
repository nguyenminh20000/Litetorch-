#ifndef LITETORCH_PRUNE_H
#define LITETORCH_PRUNE_H

#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <cstdint>
#include <memory>
#include <string>

namespace litetorch {
namespace prune {

void l1_unstructured(nn::Module& module, const std::string& name, int64_t amount);
void l1_unstructured(nn::Module& module, const std::string& name, double amount);
void random_unstructured(nn::Module& module, const std::string& name, int64_t amount, int64_t seed = -1);
void random_unstructured(nn::Module& module, const std::string& name, double amount, int64_t seed = -1);
void remove(nn::Module& module, const std::string& name);
bool is_pruned(nn::Module& module, const std::string& name);

}
}

#endif
