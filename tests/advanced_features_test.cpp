#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/optim.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>

using namespace litetorch;

void test_meta_device() {
    std::cout << "--- Testing Meta Device ---\n";
    auto t = Tensor::create({2, 3, 4}, Device(DeviceType::META, 0));
    assert(t->shape[0] == 2);
    assert(t->shape[1] == 3);
    assert(t->shape[2] == 4);
    assert(t->device.type == DeviceType::META);

    bool caught = false;
    try {
        t->data_ptr();
    } catch (const std::runtime_error& e) {
        caught = true;
    }
    assert(caught);
    std::cout << "Meta Device test passed!\n";
}

void test_tensor_backward_hooks(const Device& dev) {
    std::cout << "--- Testing Tensor Backward Hooks on " << dev.to_string() << " ---\n";
    auto x = Tensor::from_vector({1.0f, 2.0f}, {2}, dev, true);
    auto y = Ops::mul(x, Tensor::from_vector({3.0f, 4.0f}, {2}, dev));
    
    x->register_hook([](std::shared_ptr<Tensor> grad) {
        return Ops::mul(grad, Tensor::from_vector({2.0f, 2.0f}, {2}, grad->device));
    });

    auto loss = Ops::sum(y);
    loss->backward();

    auto grad_x = x->grad->to_vector();
    assert(std::abs(grad_x[0] - 6.0f) < 1e-4f);
    assert(std::abs(grad_x[1] - 8.0f) < 1e-4f);
    std::cout << "Tensor Backward Hooks test passed!\n";
}

class MockModule : public nn::Module {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        return Ops::mul(input, Tensor::from_vector({2.0f}, {1}, input->device));
    }
};

void test_module_hooks(const Device& dev) {
    std::cout << "--- Testing Module Hooks on " << dev.to_string() << " ---\n";
    auto module = std::make_shared<MockModule>();
    bool pre_hook_run = false;
    bool post_hook_run = false;

    module->register_forward_pre_hook([&](nn::Module* m, std::shared_ptr<Tensor> input) {
        pre_hook_run = true;
    });

    module->register_forward_hook([&](nn::Module* m, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) {
        post_hook_run = true;
    });

    auto x = Tensor::from_vector({1.0f, 2.0f}, {2}, dev);
    auto y = (*module)(x);

    assert(pre_hook_run);
    assert(post_hook_run);
    auto y_vec = y->to_vector();
    assert(std::abs(y_vec[0] - 2.0f) < 1e-4f);
    assert(std::abs(y_vec[1] - 4.0f) < 1e-4f);
    std::cout << "Module Hooks test passed!\n";
}

void test_adamw_optimizer(const Device& dev) {
    std::cout << "--- Testing AdamW Optimizer on " << dev.to_string() << " ---\n";
    auto p = Tensor::from_vector({1.0f, 2.0f}, {2}, dev, true);
    p->grad = Tensor::from_vector({0.1f, 0.2f}, {2}, dev);

    optim::AdamW opt({p}, 1e-1f, 0.9f, 0.999f, 1e-8f, 0.01f, false);
    opt.step();

    auto p_vec = p->to_vector();
    assert(p_vec[0] < 1.0f);
    assert(p_vec[1] < 2.0f);
    std::cout << "AdamW Standard step passed!\n";

    if (dev.type == DeviceType::GPU) {
        std::cout << "--- Testing AdamW CPU-Offload from GPU ---\n";
        auto p_gpu = Tensor::from_vector({1.0f, 2.0f}, {2}, dev, true);
        p_gpu->grad = Tensor::from_vector({0.1f, 0.2f}, {2}, dev);

        optim::AdamW opt_offload({p_gpu}, 1e-1f, 0.9f, 0.999f, 1e-8f, 0.01f, true);
        opt_offload.step();

        auto p_gpu_vec = p_gpu->to_vector();
        assert(p_gpu_vec[0] < 1.0f);
        assert(p_gpu_vec[1] < 2.0f);
        std::cout << "AdamW ZeRO-Offload step passed!\n";
    }
}

void test_cosine_annealing() {
    std::cout << "--- Testing CosineAnnealingLR Schedulers ---\n";
    auto p = Tensor::from_vector({1.0f}, {1}, Device(DeviceType::CPU, 0), true);
    auto opt = std::make_shared<optim::AdamW>(std::vector<std::shared_ptr<Tensor>>{p}, 1.0f);
    optim::CosineAnnealingLR scheduler(opt.get(), 10, 0.0f);

    scheduler.step();
    float lr1 = opt->get_lr();
    assert(lr1 < 1.0f);

    for (int i = 0; i < 9; ++i) {
        scheduler.step();
    }
    float lr10 = opt->get_lr();
    assert(lr10 < 1e-5f);
    std::cout << "CosineAnnealingLR test passed!\n";
}

void test_gradient_clipping(const Device& dev) {
    std::cout << "--- Testing Gradient Clipping on " << dev.to_string() << " ---\n";
    auto p1 = Tensor::from_vector({1.0f}, {1}, dev, true);
    p1->grad = Tensor::from_vector({10.0f}, {1}, dev);

    auto p2 = Tensor::from_vector({1.0f}, {1}, dev, true);
    p2->grad = Tensor::from_vector({20.0f}, {1}, dev);

    float norm = Ops::clip_grad_norm_({p1, p2}, 5.0f);
    assert(norm > 22.0f);

    auto g1 = p1->grad->to_vector()[0];
    auto g2 = p2->grad->to_vector()[0];
    float new_norm = std::sqrt(g1 * g1 + g2 * g2);
    assert(std::abs(new_norm - 5.0f) < 1e-2f);
    std::cout << "Gradient Clipping test passed!\n";
}

void test_rope(const Device& dev) {
    std::cout << "--- Testing RoPE on " << dev.to_string() << " ---\n";
    int B = 1, H = 1, T = 2, D = 4;
    auto x = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {B, H, T, D}, dev, true);
    auto cos = Tensor::from_vector({1.0f, 0.0f, 0.5f, 0.87f}, {T, D / 2}, dev);
    auto sin = Tensor::from_vector({0.0f, 1.0f, 0.87f, 0.5f}, {T, D / 2}, dev);

    auto y = Ops::rope(x, cos, sin);
    auto y_vec = y->to_vector();

    assert(std::abs(y_vec[0] - 1.0f) < 1e-4f);
    assert(std::abs(y_vec[1] - 2.0f) < 1e-4f);

    auto loss = Ops::sum(y);
    loss->backward();

    auto grad_x = x->grad->to_vector();
    assert(grad_x.size() == 8);
    std::cout << "RoPE test passed!\n";
}

void test_gqa(const Device& dev) {
    std::cout << "--- Testing GQA on " << dev.to_string() << " ---\n";
    int B = 1, Hq = 4, Hkv = 2, Tq = 2, Tk = 2, D = 4;
    auto q = Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                  1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {B, Hq, Tq, D}, dev, true);
    auto k = Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {B, Hkv, Tk, D}, dev, true);
    auto v = Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {B, Hkv, Tk, D}, dev, true);

    auto out = Ops::flash_attention(q, k, v);
    assert(out->shape[0] == B);
    assert(out->shape[1] == Hq);
    assert(out->shape[2] == Tq);
    assert(out->shape[3] == D);

    auto loss = Ops::sum(out);
    loss->backward();

    assert(q->grad);
    assert(k->grad);
    assert(v->grad);

    assert(k->grad->shape[1] == Hkv);
    assert(v->grad->shape[1] == Hkv);
    std::cout << "GQA test passed!\n";
}

int main() {
    std::vector<Device> devices = { Device(DeviceType::CPU, 0) };
    if (CLBackend::get().is_available()) {
        devices.push_back(Device(DeviceType::GPU, 0));
    }

    test_meta_device();
    test_cosine_annealing();

    for (auto& dev : devices) {
        test_tensor_backward_hooks(dev);
        test_module_hooks(dev);
        test_adamw_optimizer(dev);
        test_gradient_clipping(dev);
        test_rope(dev);
        test_gqa(dev);
    }

    std::cout << "ALL ADVANCED FEATURES TESTS PASSED SUCCESSFULY!\n";
    return 0;
}
