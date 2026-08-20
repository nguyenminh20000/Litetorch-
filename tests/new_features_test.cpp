#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/serialization.h"
#include "litetorch/memory_manager.h"
#include "litetorch/optim.h"
#include "litetorch/data.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <algorithm>

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

void test_serialization(double& save_time_ms, double& load_time_ms, double& ser_ram_mb) {
    Device dev_cpu(DeviceType::CPU, 0);
    Device dev_gpu(DeviceType::CPU, 0);
    if (CLBackend::get().is_available()) {
        dev_gpu = Device(DeviceType::GPU, 0);
    }

    std::vector<float> w1 = {1.5f, -2.0f, 3.1f, 0.4f};
    std::vector<float> w2 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

    auto t1 = Tensor::from_vector(w1, {2, 2}, dev_gpu);
    auto t2 = Tensor::from_vector(w2, {2, 3}, dev_gpu);

    std::vector<std::shared_ptr<Tensor>> params = {t1, t2};
    std::string path = "test_model_weights.lt";

    double ram_start = get_ram_usage_mb();
    auto start_save = std::chrono::high_resolution_clock::now();
    save_parameters(params, path);
    auto end_save = std::chrono::high_resolution_clock::now();
    save_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_save - start_save).count() / 1000.0;

    auto t1_load = Tensor::create({2, 2}, dev_gpu);
    auto t2_load = Tensor::create({2, 3}, dev_gpu);
    std::vector<std::shared_ptr<Tensor>> params_load = {t1_load, t2_load};

    auto start_load = std::chrono::high_resolution_clock::now();
    load_parameters(params_load, path);
    auto end_load = std::chrono::high_resolution_clock::now();
    load_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_load - start_load).count() / 1000.0;
    double ram_end = get_ram_usage_mb();
    ser_ram_mb = std::max(ram_start, ram_end);

    std::vector<float> t1_val = t1_load->to_vector();
    std::vector<float> t2_val = t2_load->to_vector();

    for (size_t i = 0; i < w1.size(); ++i) {
        assert(std::abs(t1_val[i] - w1[i]) < 1e-5f);
    }
    for (size_t i = 0; i < w2.size(); ++i) {
        assert(std::abs(t2_val[i] - w2[i]) < 1e-5f);
    }

    try {
        auto bad_t = Tensor::create({3, 3}, dev_gpu);
        std::vector<std::shared_ptr<Tensor>> bad_params = {bad_t, t2_load};
        load_parameters(bad_params, path);
        assert(false);
    } catch (const std::runtime_error&) {}

    try {
        std::ofstream bad_file("bad_file.lt", std::ios::binary);
        bad_file.write("BADMAGIC", 8);
        bad_file.close();
        load_parameters(params_load, "bad_file.lt");
        assert(false);
    } catch (const std::runtime_error&) {}

    try {
        std::ofstream short_file("short_file.lt", std::ios::binary);
        short_file.write("LTW1", 4);
        uint32_t num_tensors = 2;
        short_file.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));
        uint32_t ndims = 2;
        short_file.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
        int64_t dim = 2;
        short_file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        short_file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        uint64_t numel = 4;
        short_file.write(reinterpret_cast<const char*>(&numel), sizeof(numel));
        float partial_data[2] = {1.0f, 2.0f};
        short_file.write(reinterpret_cast<const char*>(partial_data), sizeof(partial_data));
        short_file.close();
        load_parameters(params_load, "short_file.lt");
        assert(false);
    } catch (const std::runtime_error&) {}

    std::remove("test_model_weights.lt");
    std::remove("bad_file.lt");
    std::remove("short_file.lt");

    {
        auto t_opt = Tensor::create({2, 2}, dev_gpu, true);
        std::vector<std::shared_ptr<Tensor>> opt_params = {t_opt};
        auto adam = std::make_shared<optim::Adam>(opt_params, 1e-3f);
        adam->step_count = 42;
        std::vector<float> m_data = {0.5f, 0.6f, 0.7f, 0.8f};
        std::vector<float> v_data = {0.01f, 0.02f, 0.03f, 0.04f};
        auto m_tensor = Tensor::from_vector(m_data, {2, 2}, dev_gpu);
        auto v_tensor = Tensor::from_vector(v_data, {2, 2}, dev_gpu);
        adam->m[0]->copy_(m_tensor);
        adam->v[0]->copy_(v_tensor);

        std::string opt_path = "test_opt_state.lt";
        save_optimizer_state(adam.get(), opt_path);

        auto adam_load = std::make_shared<optim::Adam>(opt_params, 1e-3f);
        load_optimizer_state(adam_load.get(), opt_path);

        assert(adam_load->step_count == 42);
        std::vector<float> m_loaded = adam_load->m[0]->to_vector();
        std::vector<float> v_loaded = adam_load->v[0]->to_vector();
        for (size_t i = 0; i < m_data.size(); ++i) {
            assert(std::abs(m_loaded[i] - m_data[i]) < 1e-5f);
            assert(std::abs(v_loaded[i] - v_data[i]) < 1e-5f);
        }

        std::remove(opt_path.c_str());
    }
}

void test_bmm(double& cpu_fwd_ms, double& cpu_bwd_ms, double& gpu_fwd_ms, double& gpu_bwd_ms,
              float& max_fwd_err, float& max_bwd_a_err, float& max_bwd_b_err, double& bmm_ram_mb, double& vram_peak_mb) {
    Device dev_cpu(DeviceType::CPU, 0);
    Device dev_gpu(DeviceType::CPU, 0);
    if (CLBackend::get().is_available()) {
        dev_gpu = Device(DeviceType::GPU, 0);
    }

    int B = 16;
    int M = 64;
    int K = 128;
    int N = 64;

    std::vector<float> av(B * M * K);
    std::vector<float> bv(B * K * N);
    for (int i = 0; i < B * M * K; ++i) av[i] = static_cast<float>(i % 100) / 50.0f - 1.0f;
    for (int i = 0; i < B * K * N; ++i) bv[i] = static_cast<float>(i % 100) / 100.0f - 0.5f;

    double ram_start = get_ram_usage_mb();

    auto a_cpu = Tensor::from_vector(av, {B, M, K}, dev_cpu, true);
    auto b_cpu = Tensor::from_vector(bv, {B, K, N}, dev_cpu, true);

    auto start_cpu_fwd = std::chrono::high_resolution_clock::now();
    auto c_cpu = Ops::bmm(a_cpu, b_cpu);
    auto end_cpu_fwd = std::chrono::high_resolution_clock::now();
    cpu_fwd_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu_fwd - start_cpu_fwd).count() / 1000.0;

    auto grad_out_cpu = Tensor::from_vector(std::vector<float>(B * M * N, 0.1f), {B, M, N}, dev_cpu);
    auto start_cpu_bwd = std::chrono::high_resolution_clock::now();
    c_cpu->backward(grad_out_cpu);
    auto end_cpu_bwd = std::chrono::high_resolution_clock::now();
    cpu_bwd_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu_bwd - start_cpu_bwd).count() / 1000.0;

    auto a_gpu = Tensor::from_vector(av, {B, M, K}, dev_gpu, true);
    auto b_gpu = Tensor::from_vector(bv, {B, K, N}, dev_gpu, true);

    auto start_gpu_fwd = std::chrono::high_resolution_clock::now();
    auto c_gpu = Ops::bmm(a_gpu, b_gpu);
    auto end_gpu_fwd = std::chrono::high_resolution_clock::now();
    gpu_fwd_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu_fwd - start_gpu_fwd).count() / 1000.0;

    auto grad_out_gpu = Tensor::from_vector(std::vector<float>(B * M * N, 0.1f), {B, M, N}, dev_gpu);
    auto start_gpu_bwd = std::chrono::high_resolution_clock::now();
    c_gpu->backward(grad_out_gpu);
    auto end_gpu_bwd = std::chrono::high_resolution_clock::now();
    gpu_bwd_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu_bwd - start_gpu_bwd).count() / 1000.0;

    vram_peak_mb = MemoryManager::get().get_gpu_used() / (1024.0 * 1024.0);
    double ram_end = get_ram_usage_mb();
    bmm_ram_mb = std::max(ram_start, ram_end);

    std::vector<float> cpu_res = c_cpu->to_vector();
    std::vector<float> gpu_res = c_gpu->to_vector();
    max_fwd_err = 0.0f;
    for (size_t i = 0; i < cpu_res.size(); ++i) {
        float err = std::abs(cpu_res[i] - gpu_res[i]);
        if (err > max_fwd_err) max_fwd_err = err;
    }

    std::vector<float> grad_a_cpu = a_cpu->grad->to_vector();
    std::vector<float> grad_a_gpu = a_gpu->grad->to_vector();
    max_bwd_a_err = 0.0f;
    for (size_t i = 0; i < grad_a_cpu.size(); ++i) {
        float err = std::abs(grad_a_cpu[i] - grad_a_gpu[i]);
        if (err > max_bwd_a_err) max_bwd_a_err = err;
    }

    std::vector<float> grad_b_cpu = b_cpu->grad->to_vector();
    std::vector<float> grad_b_gpu = b_gpu->grad->to_vector();
    max_bwd_b_err = 0.0f;
    for (size_t i = 0; i < grad_b_cpu.size(); ++i) {
        float err = std::abs(grad_b_cpu[i] - grad_b_gpu[i]);
        if (err > max_bwd_b_err) max_bwd_b_err = err;
    }

    assert(max_fwd_err < 1e-3f);
    assert(max_bwd_a_err < 1e-3f);
    assert(max_bwd_b_err < 1e-3f);
}

void test_5d_and_transformers() {
    Device dev_cpu(DeviceType::CPU, 0);
    Device dev_gpu(DeviceType::CPU, 0);
    if (CLBackend::get().is_available()) {
        dev_gpu = Device(DeviceType::GPU, 0);
    }

    
    {
        int N = 2, C_in = 2, D_in = 4, H_in = 4, W_in = 4;
        int C_out = 3, KD = 3, KH = 3, KW = 3;
        int stride = 1, padding = 1;

        std::vector<float> in_data(N * C_in * D_in * H_in * W_in);
        for (size_t i = 0; i < in_data.size(); ++i) in_data[i] = static_cast<float>(i % 10) / 10.0f;

        std::vector<float> w_data(C_out * C_in * KD * KH * KW);
        for (size_t i = 0; i < w_data.size(); ++i) w_data[i] = static_cast<float>(i % 5) / 5.0f - 0.4f;

        std::vector<float> b_data(C_out);
        for (size_t i = 0; i < b_data.size(); ++i) b_data[i] = 0.2f * i;

        
        auto in_cpu = Tensor::from_vector(in_data, {N, C_in, D_in, H_in, W_in}, dev_cpu, true);
        auto w_cpu = Tensor::from_vector(w_data, {C_out, C_in, KD, KH, KW}, dev_cpu, true);
        auto b_cpu = Tensor::from_vector(b_data, {C_out}, dev_cpu, true);

        auto out_cpu = Ops::conv3d(in_cpu, w_cpu, b_cpu, stride, padding);
        auto grad_out_cpu = Tensor::create(out_cpu->shape, dev_cpu);
        std::vector<float> grad_out_data(out_cpu->numel(), 0.1f);
        grad_out_cpu->copy_(Tensor::from_vector(grad_out_data, out_cpu->shape, dev_cpu));
        out_cpu->backward(grad_out_cpu);

        
        auto in_gpu = Tensor::from_vector(in_data, {N, C_in, D_in, H_in, W_in}, dev_gpu, true);
        auto w_gpu = Tensor::from_vector(w_data, {C_out, C_in, KD, KH, KW}, dev_gpu, true);
        auto b_gpu = Tensor::from_vector(b_data, {C_out}, dev_gpu, true);

        auto out_gpu = Ops::conv3d(in_gpu, w_gpu, b_gpu, stride, padding);
        auto grad_out_gpu = Tensor::create(out_gpu->shape, dev_gpu);
        grad_out_gpu->copy_(Tensor::from_vector(grad_out_data, out_gpu->shape, dev_gpu));
        out_gpu->backward(grad_out_gpu);

        
        auto res_cpu = out_cpu->to_vector();
        auto res_gpu = out_gpu->to_vector();
        assert(res_cpu.size() == res_gpu.size());
        for (size_t i = 0; i < res_cpu.size(); ++i) {
            assert(std::abs(res_cpu[i] - res_gpu[i]) < 1e-3f);
        }

        
        auto gin_cpu = in_cpu->grad->to_vector();
        auto gin_gpu = in_gpu->grad->to_vector();
        assert(gin_cpu.size() == gin_gpu.size());
        for (size_t i = 0; i < gin_cpu.size(); ++i) {
            assert(std::abs(gin_cpu[i] - gin_gpu[i]) < 1e-3f);
        }

        auto gw_cpu = w_cpu->grad->to_vector();
        auto gw_gpu = w_gpu->grad->to_vector();
        assert(gw_cpu.size() == gw_gpu.size());
        for (size_t i = 0; i < gw_cpu.size(); ++i) {
            assert(std::abs(gw_cpu[i] - gw_gpu[i]) < 1e-3f);
        }

        auto gb_cpu = b_cpu->grad->to_vector();
        auto gb_gpu = b_gpu->grad->to_vector();
        assert(gb_cpu.size() == gb_gpu.size());
        for (size_t i = 0; i < gb_cpu.size(); ++i) {
            assert(std::abs(gb_cpu[i] - gb_gpu[i]) < 1e-3f);
        }
    }

    
    {
        int N = 2, C = 2, D_in = 4, H_in = 4, W_in = 4;
        int kernel_size = 2, stride = 2, padding = 0;

        std::vector<float> in_data(N * C * D_in * H_in * W_in);
        for (size_t i = 0; i < in_data.size(); ++i) in_data[i] = static_cast<float>(i % 20) / 10.0f;

        
        auto in_cpu = Tensor::from_vector(in_data, {N, C, D_in, H_in, W_in}, dev_cpu, true);
        auto out_cpu = Ops::max_pool3d(in_cpu, kernel_size, stride, padding);
        auto grad_out_cpu = Tensor::create(out_cpu->shape, dev_cpu);
        std::vector<float> grad_out_data(out_cpu->numel(), 0.5f);
        grad_out_cpu->copy_(Tensor::from_vector(grad_out_data, out_cpu->shape, dev_cpu));
        out_cpu->backward(grad_out_cpu);

        
        auto in_gpu = Tensor::from_vector(in_data, {N, C, D_in, H_in, W_in}, dev_gpu, true);
        auto out_gpu = Ops::max_pool3d(in_gpu, kernel_size, stride, padding);
        auto grad_out_gpu = Tensor::create(out_gpu->shape, dev_gpu);
        grad_out_gpu->copy_(Tensor::from_vector(grad_out_data, out_gpu->shape, dev_gpu));
        out_gpu->backward(grad_out_gpu);

        
        auto res_cpu = out_cpu->to_vector();
        auto res_gpu = out_gpu->to_vector();
        assert(res_cpu.size() == res_gpu.size());
        for (size_t i = 0; i < res_cpu.size(); ++i) {
            assert(std::abs(res_cpu[i] - res_gpu[i]) < 1e-3f);
        }

        
        auto gin_cpu = in_cpu->grad->to_vector();
        auto gin_gpu = in_gpu->grad->to_vector();
        assert(gin_cpu.size() == gin_gpu.size());
        for (size_t i = 0; i < gin_cpu.size(); ++i) {
            assert(std::abs(gin_cpu[i] - gin_gpu[i]) < 1e-3f);
        }
    }

    
    {
        int B = 2, S = 3, D_in = 4, D_out = 5;
        std::vector<float> in_data(B * S * D_in);
        for (size_t i = 0; i < in_data.size(); ++i) in_data[i] = static_cast<float>(i) / 10.0f;

        
        auto in_cpu = Tensor::from_vector(in_data, {B, S, D_in}, dev_cpu, true);
        auto linear_cpu = nn::Linear(D_in, D_out);
        auto out_cpu = linear_cpu.forward(in_cpu);
        
        
        assert(out_cpu->shape.size() == 3);
        assert(out_cpu->shape[0] == B);
        assert(out_cpu->shape[1] == S);
        assert(out_cpu->shape[2] == D_out);

        auto grad_out_cpu = Tensor::create(out_cpu->shape, dev_cpu);
        std::vector<float> grad_out_data(out_cpu->numel(), 0.2f);
        grad_out_cpu->copy_(Tensor::from_vector(grad_out_data, out_cpu->shape, dev_cpu));
        out_cpu->backward(grad_out_cpu);

        
        auto in_gpu = Tensor::from_vector(in_data, {B, S, D_in}, dev_gpu, true);
        auto linear_gpu = nn::Linear(D_in, D_out);
        
        linear_gpu.weight->copy_(linear_cpu.weight);
        if (linear_cpu.bias) linear_gpu.bias->copy_(linear_cpu.bias);
        linear_gpu.to(dev_gpu);

        auto out_gpu = linear_gpu.forward(in_gpu);
        auto grad_out_gpu = Tensor::create(out_gpu->shape, dev_gpu);
        grad_out_gpu->copy_(Tensor::from_vector(grad_out_data, out_gpu->shape, dev_gpu));
        out_gpu->backward(grad_out_gpu);

        
        auto res_cpu = out_cpu->to_vector();
        auto res_gpu = out_gpu->to_vector();
        assert(res_cpu.size() == res_gpu.size());
        for (size_t i = 0; i < res_cpu.size(); ++i) {
            assert(std::abs(res_cpu[i] - res_gpu[i]) < 1e-3f);
        }

        
        auto gin_cpu = in_cpu->grad->to_vector();
        auto gin_gpu = in_gpu->grad->to_vector();
        assert(gin_cpu.size() == gin_gpu.size());
        for (size_t i = 0; i < gin_cpu.size(); ++i) {
            assert(std::abs(gin_cpu[i] - gin_gpu[i]) < 1e-3f);
        }

        auto gw_cpu = linear_cpu.weight->grad->to_vector();
        auto gw_gpu = linear_gpu.weight->grad->to_vector();
        assert(gw_cpu.size() == gw_gpu.size());
        for (size_t i = 0; i < gw_cpu.size(); ++i) {
            assert(std::abs(gw_cpu[i] - gw_gpu[i]) < 1e-3f);
        }

        if (linear_cpu.bias) {
            auto gb_cpu = linear_cpu.bias->grad->to_vector();
            auto gb_gpu = linear_gpu.bias->grad->to_vector();
            assert(gb_cpu.size() == gb_gpu.size());
            for (size_t i = 0; i < gb_cpu.size(); ++i) {
                assert(std::abs(gb_cpu[i] - gb_gpu[i]) < 1e-3f);
            }
        }
    }
}

void test_new_fixes() {
    Device dev_cpu(DeviceType::CPU, 0);
    Device dev_gpu = dev_cpu;
    if (CLBackend::get().is_available()) {
        dev_gpu = Device(DeviceType::GPU, 0);
    }

    std::cout << "\n==================================================\n";
    std::cout << "    RUNNING CODEBASE INTEGRITY AUDIT FIXES TESTS   \n";
    std::cout << "==================================================\n";

    for (auto dev : {dev_cpu, dev_gpu}) {
        auto bn = nn::BatchNorm2d(1);
        bn.to(dev);
        auto x = Tensor::from_vector({3.5f}, {1, 1, 1, 1}, dev, true);
        auto out = bn.forward(x);
        
        auto run_var = bn.running_var->to_vector();
        assert(!std::isnan(run_var[0]) && !std::isinf(run_var[0]));

        auto grad_out = Tensor::from_vector({1.0f}, {1, 1, 1, 1}, dev);
        out->backward(grad_out);
        assert(x->grad != nullptr);
    }
    std::cout << " -> Fix 1 (BatchNorm2d M=1 division-by-zero protection) passed.\n";

    {
        auto x = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, dev_cpu, true);
        auto x_t = x->transpose(0, 1);
        bool threw = false;
        try {
            auto y = x_t->view({6});
        } catch (const std::runtime_error& e) {
            threw = true;
        }
        assert(threw);
    }
    std::cout << " -> Fix 2 (Non-contiguous view throws error) passed.\n";

    {
        auto x = Tensor::from_vector({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {2, 3}, dev_cpu, false);
        auto x_t = x->transpose(0, 1);
        auto x_c = x_t->clone();
        auto c_vals = x_c->to_vector();
        std::vector<float> expected = {0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f};
        assert(c_vals.size() == expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            assert(std::abs(c_vals[i] - expected[i]) < 1e-5f);
        }
        assert(x_c->is_contiguous());
    }
    std::cout << " -> Fix 3 (Non-contiguous clone layout correctness) passed.\n";

    {
        auto model_param = Tensor::create({2, 2}, dev_cpu, true);
        std::vector<std::shared_ptr<Tensor>> params = {model_param};
        auto opt = std::make_shared<optim::Adam>(params, 0.001f);
        opt->step();

        std::string test_path = "temp_adam_state.lt";
        save_optimizer_state(opt.get(), test_path);

        auto mismatched_param = Tensor::create({2, 3}, dev_cpu, true);
        std::vector<std::shared_ptr<Tensor>> mismatched_params = {mismatched_param};
        auto opt_mismatched = std::make_shared<optim::Adam>(mismatched_params, 0.001f);

        bool caught_error = false;
        try {
            load_optimizer_state(opt_mismatched.get(), test_path);
        } catch (const std::runtime_error& e) {
            caught_error = true;
            std::cout << " -> Fix 4 (Caught expected error: " << e.what() << ")\n";
        }
        assert(caught_error);
    }
    std::cout << " -> Fix 4 (Optimizer state deserialization shape safety validation) passed.\n";

    {
        auto x_data = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4, 1}, dev_gpu);
        auto y_data = Tensor::from_vector({10.0f, 20.0f, 30.0f, 40.0f}, {4, 1}, dev_gpu);
        auto dataset = std::make_shared<data::TensorDataset>(x_data, y_data);
        auto sample = dataset->get(2);
        assert(sample.first->device == dev_gpu);
        assert(sample.second->device == dev_gpu);
        auto x_val = sample.first->to_vector();
        auto y_val = sample.second->to_vector();
        assert(std::abs(x_val[0] - 3.0f) < 1e-5f);
        assert(std::abs(y_val[0] - 30.0f) < 1e-5f);
    }
    std::cout << " -> Fix 5 (TensorDataset GPU synchronization) passed.\n";
}

int main() {
    double ser_save_time = 0.0, ser_load_time = 0.0, ser_ram = 0.0;
    test_serialization(ser_save_time, ser_load_time, ser_ram);

    double cpu_fwd = 0.0, cpu_bwd = 0.0, gpu_fwd = 0.0, gpu_bwd = 0.0;
    float fwd_err = 0.0f, bwd_a_err = 0.0f, bwd_b_err = 0.0f;
    double bmm_ram = 0.0, vram_peak = 0.0;
    test_bmm(cpu_fwd, cpu_bwd, gpu_fwd, gpu_bwd, fwd_err, bwd_a_err, bwd_b_err, bmm_ram, vram_peak);

    test_5d_and_transformers();
    test_new_fixes();

    std::cout << "\n==================================================\n";
    std::cout << "          LITETORCH NEW FEATURES METRICS REPORT     \n";
    std::cout << "==================================================\n\n";

    std::cout << "--- 1. WEIGHT SERIALIZATION (Save/Load) ---\n";
    std::cout << "Save Execution Time: " << ser_save_time << " ms\n";
    std::cout << "Load Execution Time: " << ser_load_time << " ms\n";
    std::cout << "Peak System RAM:     " << ser_ram << " MB\n\n";

    std::cout << "--- 2. BATCHED MATRIX MULTIPLICATION (BMM) ---\n";
    std::cout << "Batch configuration: " << "B=16, M=64, K=128, N=64\n";
    std::cout << "CPU Forward Time:    " << cpu_fwd << " ms\n";
    std::cout << "CPU Backward Time:   " << cpu_bwd << " ms\n";
    std::cout << "GPU Forward Time:    " << gpu_fwd << " ms\n";
    std::cout << "GPU Backward Time:   " << gpu_bwd << " ms\n";
    if (gpu_fwd > 0.0) {
        std::cout << "Forward Speedup:     " << (cpu_fwd / gpu_fwd) << " x\n";
        std::cout << "Backward Speedup:    " << (cpu_bwd / gpu_bwd) << " x\n";
    }
    std::cout << "Max Forward Error:   " << fwd_err << "\n";
    std::cout << "Max Grad A Error:    " << bwd_a_err << "\n";
    std::cout << "Max Grad B Error:    " << bwd_b_err << "\n";
    std::cout << "Peak System RAM:     " << bmm_ram << " MB\n";
    std::cout << "Peak VRAM Usage:     " << vram_peak << " MB\n\n";

    std::cout << "==================================================\n";
    std::cout << "   ALL TESTS PASSED & MEASURED SUCCESSFULLY!     \n";
    std::cout << "==================================================\n";

    return 0;
}

