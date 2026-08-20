#include "litetorch/checkpoint.h"
#include "litetorch/autograd.h"
#include <iostream>

namespace litetorch {

class CheckpointNode : public Node {
public:
    CheckpointFunction function_;
    std::shared_ptr<Tensor> input_copy_;

    CheckpointNode(CheckpointFunction fn, std::shared_ptr<Tensor> input)
        : Node("CheckpointNode"), function_(fn), input_copy_(input) {
        if (input && input->requires_grad) {
            inputs.push_back({ input, true });
            next_nodes.push_back(input->creator ? input->creator.ptr : nullptr);
        }
    }

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        if (!input_copy_) return { nullptr };

        std::shared_ptr<Tensor> recomputed_input = Tensor::create(input_copy_->shape, input_copy_->device, input_copy_->requires_grad, input_copy_->dtype);
        recomputed_input->copy_(input_copy_);

        std::shared_ptr<Tensor> recomputed_output = function_(recomputed_input);

        if (recomputed_output && recomputed_output->requires_grad) {
            recomputed_output->grad = grad_output;
            Autograd::backward(recomputed_output, false);
        }

        if (input_copy_->requires_grad && recomputed_input->grad) {
            if (!input_copy_->grad) {
                input_copy_->grad = Tensor::create(recomputed_input->grad->shape, recomputed_input->grad->device, false, recomputed_input->grad->dtype);
                input_copy_->grad->copy_(recomputed_input->grad);
            } else {
                input_copy_->grad->add_(recomputed_input->grad);
            }
        }

        return { recomputed_input->grad };
    }
};

std::shared_ptr<Tensor> checkpoint(CheckpointFunction function, std::shared_ptr<Tensor> input) {
    if (!Autograd::is_grad_enabled() || !input || !input->requires_grad) {
        return function(input);
    }

    std::shared_ptr<Tensor> output;
    {
        NoGradGuard guard;
        output = function(input);
    }

    if (output) {
        auto node = std::make_shared<CheckpointNode>(function, input);
        output->requires_grad = true;
        output->creator = node;
    }

    return output;
}

}
