#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace litetorch;

void test_attention(const Device& dev) {
    std::cout << "\n--- Testing MultiHeadAttention on " << (dev.type == DeviceType::GPU ? "GPU" : "CPU") << " ---\n";

    int B = 2;
    int T = 4;
    int C = 8;
    int num_heads = 2;

    std::vector<float> input_data;
    for (int i = 0; i < B * T * C; ++i) {
        input_data.push_back(sin(static_cast<float>(i)));
    }
    auto input = Tensor::from_vector(input_data, {B, T, C}, dev, true);

    auto mha = std::make_shared<nn::MultiHeadAttention>(C, num_heads);
    mha->to(dev);

    auto output = mha->forward(input);

    std::cout << "Input shape: (" << input->shape[0] << ", " << input->shape[1] << ", " << input->shape[2] << ")\n";
    std::cout << "Output shape: (" << output->shape[0] << ", " << output->shape[1] << ", " << output->shape[2] << ")\n";

    assert(output->shape[0] == B);
    assert(output->shape[1] == T);
    assert(output->shape[2] == C);

    auto loss = Ops::sum(output);
    std::cout << "Forward success. Mock Loss: " << loss->item() << "\n";

    loss->backward();

    float input_grad_sum = 0.0f;
    if (input->grad) {
        auto grad_vec = input->grad->to_vector();
        for (float g : grad_vec) input_grad_sum += std::abs(g);
    }
    std::cout << "Backward success. Input Gradient absolute sum: " << input_grad_sum << "\n";

    float q_proj_weight_grad_sum = 0.0f;
    if (mha->q_proj->weight->grad) {
        auto q_grad_vec = mha->q_proj->weight->grad->to_vector();
        for (float g : q_grad_vec) q_proj_weight_grad_sum += std::abs(g);
    }
    std::cout << "Q Proj Weight Gradient absolute sum: " << q_proj_weight_grad_sum << "\n";

    assert(input_grad_sum > 0.0f);
    assert(q_proj_weight_grad_sum > 0.0f);
    std::cout << "[RESULT] SUCCESS: MultiHeadAttention forward/backward passed on " << (dev.type == DeviceType::GPU ? "GPU" : "CPU") << "\n";
}

int main() {
    std::cout << "==================================================\n";
    std::cout << "       LITETORCH MULTI-HEAD ATTENTION TEST        \n";
    std::cout << "==================================================\n";

    std::vector<Device> devices = { Device(DeviceType::CPU, 0) };
    if (CLBackend::get().is_available()) {
        devices.push_back(Device(DeviceType::GPU, 0));
    }

    for (const auto& dev : devices) {
        test_attention(dev);
    }

    std::cout << "==================================================\n";
    return 0;
}
