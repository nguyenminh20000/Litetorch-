#include "litetorch/custom_ops.h"

namespace litetorch {
namespace custom_ops {

class CustomOpNode : public Node {
public:
    std::string op_name;
    CustomOpBackwardFunc backward_func;

    CustomOpNode(const std::string& name, CustomOpBackwardFunc bw)
        : Node("CustomOp_" + name), op_name(name), backward_func(bw) {}

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        if (!backward_func) {
            throw std::runtime_error("[litetorch Error] Custom operator " + op_name + " backward was called but no backward function was registered");
        }
        auto inps = saved_tensors;
        auto out = output.lock();
        return backward_func(grad_output, inps, out);
    }
};

std::shared_ptr<Tensor> Registry::call(const std::string& name, const std::vector<std::shared_ptr<Tensor>>& args) {
    auto it = registry_.find(name);
    if (it == registry_.end()) {
        throw std::runtime_error("[litetorch Error] Custom operator " + name + " is not registered");
    }

    if (args.empty()) {
        throw std::runtime_error("[litetorch Error] Custom operator call must have at least one input tensor");
    }

    Device dev = args[0]->device;
    std::shared_ptr<Tensor> out;
    if (dev.type == DeviceType::GPU) {
        if (it->second.gpu_forward) {
            out = it->second.gpu_forward(args);
        } else {
            out = it->second.cpu_forward(args);
        }
    } else {
        out = it->second.cpu_forward(args);
    }

    bool any_requires_grad = false;
    for (auto& a : args) {
        if (a->requires_grad) {
            any_requires_grad = true;
            break;
        }
    }

    if (any_requires_grad && it->second.backward) {
        auto node = std::make_shared<CustomOpNode>(name, it->second.backward);
        for (auto& a : args) {
            node->inputs.push_back({a, a->requires_grad});
            node->next_nodes.push_back(a->creator);
            node->saved_tensors.push_back(a);
        }
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }

    return out;
}

}
}
