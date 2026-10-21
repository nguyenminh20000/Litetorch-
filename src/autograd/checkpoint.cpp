#include "litetorch/checkpoint.h"
#include "litetorch/autograd.h"
#include <iostream>

namespace litetorch {

class CheckpointNode : public Node {
public:
    CheckpointFunction function_;
    std::shared_ptr<Tensor> input_copy_;
    std::string rng_state_;
    bool input_requires_grad_ = false;
    bool create_graph_ = false;

    CheckpointNode(CheckpointFunction fn, std::shared_ptr<Tensor> input, const std::string& rng_state)
        : Node("CheckpointNode"), function_(fn), rng_state_(rng_state) {
        if (input) {
            input_copy_ = Tensor::create(input->shape, input->device, false, input->dtype);
            input_copy_->copy_(input);
            input_requires_grad_ = input->requires_grad;
        }
        if (input && input->requires_grad) {
            inputs.push_back({ input, true });
            next_nodes.push_back(input->creator ? input->creator.ptr : nullptr);
        }
    }

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        if (!input_copy_) return { nullptr };

        std::string saved_rng = get_rng_state();
        std::vector<std::weak_ptr<Tensor>> saved_active = Autograd::active_tensors;
        set_rng_state(rng_state_);

        std::shared_ptr<Tensor> recomputed_input = Tensor::create(input_copy_->shape, input_copy_->device, input_requires_grad_, input_copy_->dtype);
        {
            NoGradGuard guard;
            recomputed_input->copy_(input_copy_);
        }

        std::shared_ptr<Tensor> recomputed_output = function_(recomputed_input);

        create_graph_ = Autograd::is_create_graph_;
        if (recomputed_output && recomputed_output->requires_grad) {
            recomputed_output->grad = grad_output;
            Autograd::backward(recomputed_output, create_graph_);
        }

        set_rng_state(saved_rng);
        Autograd::active_tensors = std::move(saved_active);
        return { recomputed_input->grad };
    }
};

std::shared_ptr<Tensor> checkpoint(CheckpointFunction function, std::shared_ptr<Tensor> input) {
    if (!Autograd::is_grad_enabled() || !input || !input->requires_grad) {
        return function(input);
    }

    std::string rng_state = get_rng_state();

    std::shared_ptr<Tensor> output;
    {
        NoGradGuard guard;
        output = function(input);
    }

    if (output) {
        auto node = std::make_shared<CheckpointNode>(function, input, rng_state);
        output->requires_grad = true;
        output->creator = node;
    }

    return output;
}

}
