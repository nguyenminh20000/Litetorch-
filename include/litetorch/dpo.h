#ifndef LITETORCH_DPO_H
#define LITETORCH_DPO_H

#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include <memory>
#include <string>

namespace litetorch {
namespace nn {

class DPOLoss : public Module {
public:
    float beta;
    bool reference_free;
    std::string reduction;

    DPOLoss(float beta = 0.1f, bool reference_free = false,
            const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> policy_chosen_logps,
                                    std::shared_ptr<Tensor> policy_rejected_logps,
                                    std::shared_ptr<Tensor> ref_chosen_logps = nullptr,
                                    std::shared_ptr<Tensor> ref_rejected_logps = nullptr);
};

}

namespace functional {

std::shared_ptr<Tensor> dpo_loss(std::shared_ptr<Tensor> policy_chosen_logps,
                                 std::shared_ptr<Tensor> policy_rejected_logps,
                                 std::shared_ptr<Tensor> ref_chosen_logps = nullptr,
                                 std::shared_ptr<Tensor> ref_rejected_logps = nullptr,
                                 float beta = 0.1f,
                                 bool reference_free = false,
                                 const std::string& reduction = "mean");

}
}

#endif
