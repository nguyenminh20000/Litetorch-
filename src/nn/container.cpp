#include "litetorch/nn.h"
#include "litetorch/backend.h"

namespace litetorch {
namespace nn {

Sequential::Sequential(const std::vector<std::shared_ptr<Module>>& modules) : modules(modules) {}

void Sequential::add(std::shared_ptr<Module> module) {
    modules.push_back(module);
}

std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> input) {
    if (is_compiled && input->device.type == DeviceType::GPU) {
        auto backend = BackendDispatcher::get().get_backend();
        if (backend && backend->is_available()) {
            if (compiled_graph_exec) {
                backend->launch_graph(compiled_graph_exec);
                auto out = input;
                for (auto& module : modules) {
                    out = module->forward(out);
                }
                return out;
            } else if (!is_graph_recorded) {
                backend->start_recording();
                auto out = input;
                for (auto& module : modules) {
                    out = module->forward(out);
                }
                compiled_graph_exec = backend->stop_recording(nullptr);
                is_graph_recorded = true;
                return out;
            }
        }
    }
    auto out = input;
    for (auto& module : modules) {
        out = module->forward(out);
    }
    return out;
}

std::vector<std::shared_ptr<Tensor>> Sequential::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto& module : modules) {
        auto m_params = module->parameters();
        params.insert(params.end(), m_params.begin(), m_params.end());
    }
    return params;
}

void Sequential::to(const Device& device) {
    for (auto& module : modules) {
        module->to(device);
    }
}

void Sequential::train() {
    Module::train();
    for (auto& module : modules) {
        module->train();
    }
}

void Sequential::eval() {
    Module::eval();
    for (auto& module : modules) {
        module->eval();
    }
}

std::shared_ptr<Tensor> Flatten::forward(std::shared_ptr<Tensor> input) {
    int64_t batch_size = input->shape[0];
    int64_t remaining = input->numel() / batch_size;
    return input->view({batch_size, remaining});
}

}
}
