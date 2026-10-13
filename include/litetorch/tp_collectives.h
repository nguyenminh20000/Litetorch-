#ifndef LITETORCH_TP_COLLECTIVES_H
#define LITETORCH_TP_COLLECTIVES_H

#include <cstdint>
#include <memory>
#include <vector>
#include "litetorch/autograd.h"

namespace litetorch {
namespace distributed {

bool tp_active();

void tp_allreduce_inplace(std::shared_ptr<::litetorch::Tensor> t);
void tp_allgather_into(std::shared_ptr<::litetorch::Tensor> shard, std::shared_ptr<::litetorch::Tensor> full);

class AllReduceFn : public ::litetorch::Function {
public:
    std::vector<std::shared_ptr<::litetorch::Tensor>> forward(::litetorch::FunctionContext* ctx,
        const std::vector<std::shared_ptr<::litetorch::Tensor>>& inputs) override;
    std::vector<std::shared_ptr<::litetorch::Tensor>> backward(::litetorch::FunctionContext* ctx,
        const std::vector<std::shared_ptr<::litetorch::Tensor>>& grad_outputs) override;
};

class AllGatherFn : public ::litetorch::Function {
public:
    std::vector<std::shared_ptr<::litetorch::Tensor>> forward(::litetorch::FunctionContext* ctx,
        const std::vector<std::shared_ptr<::litetorch::Tensor>>& inputs) override;
    std::vector<std::shared_ptr<::litetorch::Tensor>> backward(::litetorch::FunctionContext* ctx,
        const std::vector<std::shared_ptr<::litetorch::Tensor>>& grad_outputs) override;
};

std::shared_ptr<::litetorch::Tensor> tp_all_reduce_autograd(std::shared_ptr<::litetorch::Tensor> t);
std::shared_ptr<::litetorch::Tensor> tp_all_gather_autograd(std::shared_ptr<::litetorch::Tensor> shard);

}
}

#endif
