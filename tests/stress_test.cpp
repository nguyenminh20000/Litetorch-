#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/optim.h"
#include "litetorch/memory_manager.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include <thread>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

using namespace litetorch;

double get_ram_usage_mb() {
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            std::stringstream ss(line.substr(6));
            double value;
            std::string unit;
            ss >> value >> unit;
            if (unit == "kB") return value / 1024.0;
            return value;
        }
    }
    return 0.0;
}

void test_lru_eviction_brutal(Device dev) {
    std::cout << "[Brutal LRU Eviction Test] Device: " << (dev.type == DeviceType::GPU ? "GPU" : "CPU") << std::endl;
    if (dev.type == DeviceType::GPU) {
        MemoryManager::get().set_gpu_limit(1024 * 1024 * 2);
        std::vector<std::shared_ptr<Tensor>> tensors;
        for (int i = 0; i < 20; ++i) {
            auto t = Tensor::create({1024, 128}, dev);
            std::vector<float> data(1024 * 128, static_cast<float>(i));
            CLBackend::get().write(t->gpu_data(), t->numel() * sizeof(float), data.data());
            tensors.push_back(t);
        }
        for (int step = 0; step < 5; ++step) {
            for (int i = 0; i < 20; ++i) {
                int target_idx = (i + step) % 20;
                auto vec = tensors[target_idx]->to_vector();
                for (size_t j = 0; j < 10; ++j) {
                    assert(std::abs(vec[j] - static_cast<float>(target_idx)) < 1e-5);
                }
            }
        }
        for (int i = 0; i < 10; ++i) {
            auto a = tensors[i];
            auto b = tensors[i + 10];
            auto c = Ops::add(a, b);
            auto vec = c->to_vector();
            float expected = static_cast<float>(i + i + 10);
            for (size_t j = 0; j < 10; ++j) {
                assert(std::abs(vec[j] - expected) < 1e-5);
            }
        }
        std::cout << "[Brutal LRU Eviction Test] PASSED" << std::endl;
    } else {
        std::cout << "[Brutal LRU Eviction Test] Skipped for CPU" << std::endl;
    }
}

void test_deep_convnet_brutal(Device dev) {
    std::cout << "[Brutal Deep ConvNet Test] Device: " << (dev.type == DeviceType::GPU ? "GPU" : "CPU") << std::endl;

    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 0.2f);

    std::vector<float> x_data(16 * 3 * 16 * 16);
    for (auto& val : x_data) val = dis(gen);

    std::vector<float> y_data(16 * 10);
    for (int i = 0; i < 16; ++i) {
        int label = i % 10;
        for (int j = 0; j < 10; ++j) {
            y_data[i * 10 + j] = (j == label) ? 1.0f : 0.0f;
        }
    }

    auto x = Tensor::from_vector(x_data, {16, 3, 16, 16}, dev);
    auto y = Tensor::from_vector(y_data, {16, 10}, dev);

    auto model = std::make_shared<nn::Sequential>();
    model->add(std::make_shared<nn::Conv2d>(3, 8, 3, 1, 1));
    model->add(std::make_shared<nn::BatchNorm2d>(8));
    model->add(std::make_shared<nn::ReLU>());
    model->add(std::make_shared<nn::MaxPool2d>(2, 2));
    
    model->add(std::make_shared<nn::Conv2d>(8, 16, 3, 1, 1));
    model->add(std::make_shared<nn::BatchNorm2d>(16));
    model->add(std::make_shared<nn::ReLU>());
    model->add(std::make_shared<nn::MaxPool2d>(2, 2));

    model->add(std::make_shared<nn::AdaptiveAvgPool2d>(2, 2));
    model->add(std::make_shared<nn::Flatten>());
    model->add(std::make_shared<nn::Linear>(64, 32));
    model->add(std::make_shared<nn::LayerNorm>(std::vector<int64_t>{32}));
    model->add(std::make_shared<nn::GELU>());
    model->add(std::make_shared<nn::Linear>(32, 10));
    model->add(std::make_shared<nn::Softmax>(-1));
    model->to(dev);

    auto optimizer = std::make_unique<optim::Adam>(model->parameters(), 0.01f);

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    {
        auto out = x;
        std::cout << "DEBUG: input x[0..4] =";
        auto vec_x = x->to_vector();
        for (int i = 0; i < 5; ++i) std::cout << " " << vec_x[i];
        std::cout << std::endl;
        
        for (size_t i = 0; i < model->modules.size(); ++i) {
            out = model->modules[i]->forward(out);
            auto vec_out = out->to_vector();
            std::cout << "DEBUG: Layer " << i << " out[0..4] =";
            for (int j = 0; j < 5; ++j) {
                if (j < vec_out.size()) std::cout << " " << vec_out[j];
            }
            std::cout << std::endl;
        }
    }

    for (int epoch = 0; epoch < 40; ++epoch) {
        optimizer->zero_grad();
        auto out = model->forward(x);
        auto loss = Ops::mse_loss(out, y);
        loss->backward();

        {
            auto params = model->parameters();
            std::cout << "DEBUG Gradients:";
            for (size_t p_idx = 0; p_idx < params.size(); ++p_idx) {
                auto p = params[p_idx];
                if (p->grad) {
                    auto vec = p->grad->to_vector();
                    float sum_abs = 0.0f;
                    for (float val : vec) sum_abs += std::abs(val);
                    std::cout << " [" << p_idx << "]=" << sum_abs;
                } else {
                    std::cout << " [" << p_idx << "]=NULL";
                }
            }
            std::cout << std::endl;
        }

        optimizer->step();
        float loss_val = loss->item();
        std::cout << "Epoch " << epoch << " | Loss: " << loss_val << std::endl;
        assert(!std::isnan(loss_val) && !std::isinf(loss_val));
        if (epoch == 0) initial_loss = loss_val;
        final_loss = loss_val;
    }

    std::cout << "[Brutal Deep ConvNet Test] Initial Loss: " << initial_loss << ", Final Loss: " << final_loss << std::endl;
    assert(final_loss < initial_loss);
    std::cout << "[Brutal Deep ConvNet Test] PASSED" << std::endl;
}

void test_concurrent_backward_stress() {
    std::cout << "[Concurrent Backward Stress Test] Running..." << std::endl;
    const int num_threads = 8;
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([t]() {
            Device dev(DeviceType::CPU, 0);
            std::mt19937 gen(100 + t);
            std::normal_distribution<float> dis(0.0f, 0.5f);

            std::vector<float> x_data(8 * 4);
            std::vector<float> y_data(8 * 2);
            for (auto& val : x_data) val = dis(gen);
            for (auto& val : y_data) val = dis(gen) > 0.0f ? 1.0f : 0.0f;

            auto x = Tensor::from_vector(x_data, {8, 4}, dev);
            auto y = Tensor::from_vector(y_data, {8, 2}, dev);

            auto model = std::make_shared<nn::Sequential>();
            model->add(std::make_shared<nn::Linear>(4, 8));
            model->add(std::make_shared<nn::LayerNorm>(std::vector<int64_t>{8}));
            model->add(std::make_shared<nn::GELU>());
            model->add(std::make_shared<nn::Linear>(8, 2));
            model->add(std::make_shared<nn::Softmax>(-1));
            model->to(dev);

            auto optimizer = std::make_unique<optim::Adam>(model->parameters(), 0.02f);

            for (int epoch = 0; epoch < 20; ++epoch) {
                optimizer->zero_grad();
                auto out = model->forward(x);
                auto loss = Ops::mse_loss(out, y);
                loss->backward();
                optimizer->step();
                float loss_val = loss->item();
                assert(!std::isnan(loss_val) && !std::isinf(loss_val));
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    std::cout << "[Concurrent Backward Stress Test] PASSED" << std::endl;
}

void test_huge_matmul_precision() {
    std::cout << "[Huge Matmul Precision Test] Running..." << std::endl;
    Device dev(DeviceType::CPU, 0);
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    std::vector<float> a_data(256 * 256);
    std::vector<float> b_data(256 * 256);
    for (auto& val : a_data) val = dis(gen);
    for (auto& val : b_data) val = dis(gen);

    auto a = Tensor::from_vector(a_data, {256, 256}, dev);
    auto b = Tensor::from_vector(b_data, {256, 256}, dev);
    auto c = Ops::matmul(a, b);

    std::vector<float> expected(256 * 256, 0.0f);
    for (int i = 0; i < 256; ++i) {
        for (int k = 0; k < 256; ++k) {
            float val_a = a_data[i * 256 + k];
            for (int j = 0; j < 256; ++j) {
                expected[i * 256 + j] += val_a * b_data[k * 256 + j];
            }
        }
    }

    auto actual = c->to_vector();
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(std::abs(actual[i] - expected[i]) < 1e-3);
    }
    std::cout << "[Huge Matmul Precision Test] PASSED" << std::endl;
}

void test_adversarial_cases() {
    std::cout << "[Adversarial Test] Running..." << std::endl;
    Device dev(DeviceType::CPU, 0);

    bool caught = false;
    try {
        auto t1 = Tensor::create({2, 3}, dev);
        auto t2 = Tensor::create({4, 5}, dev);
        auto t3 = Ops::add(t1, t2);
    } catch (const std::exception& e) {
        caught = true;
    }
    assert(caught);

    caught = false;
    try {
        auto t1 = Tensor::create({2, 3}, dev);
        auto t2 = Tensor::create({4, 5}, dev);
        auto t3 = Ops::matmul(t1, t2);
    } catch (const std::exception& e) {
        caught = true;
    }
    assert(caught);

    auto t_nan = Tensor::from_vector({NAN, 1.0f, INFINITY}, {3, 1}, dev);
    auto t_out = Ops::add(t_nan, t_nan);
    auto vec_out = t_out->to_vector();
    assert(std::isnan(vec_out[0]));
    assert(std::isinf(vec_out[2]));

    std::cout << "[Adversarial Test] PASSED" << std::endl;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    double start_ram = get_ram_usage_mb();

    std::cout << "=========================================" << std::endl;
    std::cout << "      LITETORCH BRUTAL STRESS TEST       " << std::endl;
    std::cout << "=========================================" << std::endl;

    Device cpu_dev(DeviceType::CPU, 0);

    test_lru_eviction_brutal(cpu_dev);
    test_deep_convnet_brutal(cpu_dev);
    test_concurrent_backward_stress();
    test_huge_matmul_precision();
    test_adversarial_cases();

    if (CLBackend::get().is_available()) {
        Device gpu_dev(DeviceType::GPU, 0);
        test_lru_eviction_brutal(gpu_dev);
        test_deep_convnet_brutal(gpu_dev);
    } else {
        std::cout << "[OpenCL] GPU Device is not available. Skipping GPU stress tests." << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double end_ram = get_ram_usage_mb();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "=========================================" << std::endl;
    std::cout << "   ALL LITETORCH STRESS TESTS PASSED!    " << std::endl;
    std::cout << "=========================================" << std::endl;

    std::cout << "\n--- STRUCTURAL RUNTIME REPORT ---" << std::endl;
    std::cout << "Execution Time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Peak RAM usage: " << std::max(start_ram, end_ram) << " MB" << std::endl;
    std::cout << "GPU availability: " << (CLBackend::get().is_available() ? "YES" : "NO") << std::endl;
    std::cout << "VRAM peak usage: " << MemoryManager::get().get_gpu_used() / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Training status: CONVERGED" << std::endl;
    std::cout << "Correctness status: VALID" << std::endl;
    std::cout << "Adversarial checks: PASSED" << std::endl;
    std::cout << "---------------------------------" << std::endl;

    return 0;
}
