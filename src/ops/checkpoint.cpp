#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/nn.h"
#include <functional>

namespace litetorch {

class ModuleCheckpointNode : public Node {
public:
    std::shared_ptr<nn::Module> module;
    std::vector<std::shared_ptr<Tensor>> saved_inputs;
    std::vector<std::shared_ptr<Tensor>> saved_params;
    std::vector<bool> orig_requires_grad_inputs;
    std::vector<bool> orig_requires_grad_params;

    ModuleCheckpointNode(std::shared_ptr<nn::Module> mod,
                   const std::vector<std::shared_ptr<Tensor>>& inputs,
                   const std::vector<std::shared_ptr<Tensor>>& params)
        : Node("ModuleCheckpoint"), module(mod), saved_inputs(inputs), saved_params(params) {
        for (auto& t : inputs) orig_requires_grad_inputs.push_back(t->requires_grad);
        for (auto& t : params) orig_requires_grad_params.push_back(t->requires_grad);
    }

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::vector<std::shared_ptr<Tensor>> detached_inputs;
        detached_inputs.reserve(saved_inputs.size());
        for (size_t i = 0; i < saved_inputs.size(); ++i) {
            auto& t = saved_inputs[i];
            auto d = Tensor::create(t->shape, t->device, false, t->dtype);
            d->storage = t->storage;
            d->offset = t->offset;
            d->strides = t->strides;
            d->requires_grad = orig_requires_grad_inputs[i];
            detached_inputs.push_back(d);
        }

        auto local_out = module->forward(detached_inputs[0]);
        local_out->backward(grad_output, Autograd::is_create_graph_);

        std::vector<std::shared_ptr<Tensor>> grads;
        for (auto& d : detached_inputs) {
            grads.push_back(d->grad);
            d->grad = nullptr;
        }
        for (auto& p : saved_params) {
            grads.push_back(p->grad);
            p->grad = nullptr;
        }

        return grads;
    }
};

namespace Ops {

std::shared_ptr<Tensor> checkpoint(std::shared_ptr<nn::Module> module, std::shared_ptr<Tensor> input) {
    auto params = module->parameters();
    bool input_requires_grad = input->requires_grad;
    std::vector<bool> params_requires_grad;
    for (auto& p : params) {
        params_requires_grad.push_back(p->requires_grad);
        p->requires_grad = false;
    }
    input->requires_grad = false;

    std::shared_ptr<Tensor> output;
    {
        NoGradGuard no_grad;
        output = module->forward(input);
    }

    input->requires_grad = input_requires_grad;
    for (size_t i = 0; i < params.size(); ++i) {
        params[i]->requires_grad = params_requires_grad[i];
    }

    bool any_requires_grad = input_requires_grad;
    for (auto r : params_requires_grad) {
        if (r) any_requires_grad = true;
    }

    if (Autograd::is_grad_enabled() && (any_requires_grad)) {
        auto node = std::make_shared<ModuleCheckpointNode>(module, std::vector<std::shared_ptr<Tensor>>{input}, params);
        node->inputs.push_back({input, input_requires_grad});
        node->next_nodes.push_back(input->creator);
        for (size_t i = 0; i < params.size(); ++i) {
            node->inputs.push_back({params[i], params_requires_grad[i]});
            node->next_nodes.push_back(params[i]->creator);
        }
        node->output = output;
        output->creator = node;
        output->requires_grad = true;
    }

    return output;
}

}
}
