#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace litetorch;

void test_conv3d_non_contiguous(const Device& dev) {
    std::cout << "\n--- 1. Testing conv3d on " << (dev.type == DeviceType::GPU ? "GPU" : "CPU") << " with non-contiguous input ---\n";

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto base = Tensor::from_vector(data, {1, 1, 2, 2, 2}, dev);

    auto input = base->transpose(3, 4);
    assert(!input->is_contiguous());

    std::vector<float> expected = {1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 7.0f, 6.0f, 8.0f};

    auto weight = Tensor::from_vector({1.0f}, {1, 1, 1, 1, 1}, dev);

    auto output = Ops::conv3d(input, weight, nullptr, 1, 0);

    std::vector<float> out_data = output->to_vector();
    std::cout << "Expected: ";
    for (float v : expected) std::cout << v << " ";
    std::cout << "\nActual output: ";
    for (float v : out_data) std::cout << v << " ";
    std::cout << "\n";

    bool bug_present = false;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(out_data[i] - expected[i]) > 1e-5f) {
            bug_present = true;
        }
    }

    if (bug_present) {
        std::cout << "[RESULT] BUG CONFIRMED: conv3d on " << (dev.type == DeviceType::GPU ? "GPU" : "CPU") << " ignores non-contiguity!\n";
    } else {
        std::cout << "[RESULT] SUCCESS: conv3d on " << (dev.type == DeviceType::GPU ? "GPU" : "CPU") << " handled non-contiguity correctly.\n";
    }
}

void test_autograd_storage_sharing(const Device& dev) {
    std::cout << "\n--- 2. Testing autograd gradient accumulation storage sharing on " << (dev.type == DeviceType::GPU ? "GPU" : "CPU") << " ---\n";

    auto a = Tensor::from_vector({2.0f}, {1}, dev, true);
    auto b = Tensor::from_vector({3.0f}, {1}, dev, true);
    auto x = Ops::add(a, b);
    auto y = Ops::add(x, a);

    y->backward();

    float grad_a = a->grad->item();
    float grad_b = b->grad->item();

    std::cout << "Mathematical expected gradients: dy/da = 2.0, dy/db = 1.0\n";
    std::cout << "Actual gradients: dy/da = " << grad_a << ", dy/db = " << grad_b << "\n";

    if (std::abs(grad_b - 1.0f) > 1e-5f) {
        std::cout << "[RESULT] BUG CONFIRMED: autograd corrupted dy/db (got " << grad_b << ") due to storage leakage on " << (dev.type == DeviceType::GPU ? "GPU" : "CPU") << "!\n";
    } else {
        std::cout << "[RESULT] SUCCESS: gradients accumulated correctly on " << (dev.type == DeviceType::GPU ? "GPU" : "CPU") << ".\n";
    }
}

int main() {
    std::cout << "==================================================\n";
    std::cout << "       LITETORCH AUDIT BUGS VERIFICATION TEST     \n";
    std::cout << "==================================================\n";

    std::vector<Device> devices = { Device(DeviceType::CPU, 0) };
    if (CLBackend::get().is_available()) {
        devices.push_back(Device(DeviceType::GPU, 0));
    }

    for (const auto& dev : devices) {
        test_conv3d_non_contiguous(dev);
        test_autograd_storage_sharing(dev);
    }

    std::cout << "==================================================\n";
    return 0;
}
