#include "litetorch/checkpoint.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <iostream>
#include <cassert>

using namespace litetorch;

int main() {
    std::cout << "[Checkpoint Test] Starting Activation Checkpointing test...\n";

    std::vector<float> data(16, 1.0f);
    auto x = Tensor::from_vector(data, {4, 4}, Device(DeviceType::CPU), true);
    
    auto fn = [](std::shared_ptr<Tensor> input) {
        auto h1 = Ops::mul(input, input);
        auto h2 = Ops::add(h1, input);
        return h2;
    };

    auto out = checkpoint(fn, x);
    assert(out != nullptr);
    assert(out->requires_grad == true);

    auto loss = Ops::sum(out);
    loss->backward();

    assert(x->grad != nullptr);
    std::cout << "[Checkpoint Test] Activation Checkpointing Test Passed! Grad sum: " << Ops::sum(x->grad)->item() << "\n";

    return 0;
}
