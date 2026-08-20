#ifndef LITETORCH_CUSTOM_OPS_H
#define LITETORCH_CUSTOM_OPS_H

#include "litetorch/tensor.h"
#include "litetorch/autograd.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <stdexcept>

namespace litetorch {
namespace custom_ops {

using CustomOpForwardFunc = std::function<std::shared_ptr<Tensor>(const std::vector<std::shared_ptr<Tensor>>&)>;
using CustomOpBackwardFunc = std::function<std::vector<std::shared_ptr<Tensor>>(std::shared_ptr<Tensor>, const std::vector<std::shared_ptr<Tensor>>&, std::shared_ptr<Tensor>)>;

struct CustomOpDef {
    std::string name;
    CustomOpForwardFunc cpu_forward;
    CustomOpForwardFunc gpu_forward;
    CustomOpBackwardFunc backward;
};

class Registry {
private:
    std::unordered_map<std::string, CustomOpDef> registry_;
    Registry() = default;

public:
    static Registry& get() {
        static Registry instance;
        return instance;
    }

    void register_op(const std::string& name,
                     CustomOpForwardFunc cpu_forward,
                     CustomOpForwardFunc gpu_forward = nullptr,
                     CustomOpBackwardFunc backward = nullptr) {
        registry_[name] = {name, cpu_forward, gpu_forward, backward};
    }

    std::shared_ptr<Tensor> call(const std::string& name, const std::vector<std::shared_ptr<Tensor>>& args);
};

}
}

#endif
