#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/jit.h"
#include "litetorch/backend.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

using namespace litetorch;

int main() {
    auto dev_cpu = Device(DeviceType::CPU);
    auto dev_gpu = Device(DeviceType::GPU);

    auto a_sym = std::make_shared<JITVar>(JITVar::OpType::INPUT, "a");
    auto b_sym = std::make_shared<JITVar>(JITVar::OpType::INPUT, "b");
    auto c_sym = std::make_shared<JITVar>(JITVar::OpType::INPUT, "c");
    auto expr = JIT::gelu(a_sym * b_sym + c_sym);

    auto fused_fn = JITFunction("fused_gelu_add", expr, {a_sym, b_sym, c_sym});

    std::cout << "--- Testing JIT on CPU ---" << std::endl;
    auto a_cpu = Tensor::from_vector({1.0f, 2.0f, -3.0f, 4.0f}, {4}, dev_cpu);
    auto b_cpu = Tensor::from_vector({0.5f, 1.5f, 2.0f, -1.0f}, {4}, dev_cpu);
    auto c_cpu = Tensor::from_vector({1.0f, -1.0f, 0.0f, 2.0f}, {4}, dev_cpu);

    auto out_cpu_sequential = Ops::gelu(Ops::add(Ops::mul(a_cpu, b_cpu), c_cpu));
    auto out_cpu_jit = fused_fn({a_cpu, b_cpu, c_cpu});

    auto v_seq_cpu = out_cpu_sequential->to_vector();
    auto v_jit_cpu = out_cpu_jit->to_vector();

    for (size_t i = 0; i < v_seq_cpu.size(); ++i) {
        std::cout << "CPU Element " << i << ": Sequential=" << v_seq_cpu[i] << ", JIT=" << v_jit_cpu[i] << std::endl;
        if (std::abs(v_seq_cpu[i] - v_jit_cpu[i]) >= 1e-5f) {
            std::cerr << "CPU mismatch!" << std::endl;
            return 1;
        }
    }
    std::cout << "CPU JIT matches Sequential!" << std::endl;

    auto active_backend = BackendDispatcher::get().get_backend();
    if (active_backend && active_backend->is_available()) {
        std::cout << "\n--- Testing JIT on GPU ---" << std::endl;
        auto a_gpu = a_cpu->to(dev_gpu);
        auto b_gpu = b_cpu->to(dev_gpu);
        auto c_gpu = c_cpu->to(dev_gpu);

        auto out_gpu_sequential = Ops::gelu(Ops::add(Ops::mul(a_gpu, b_gpu), c_gpu));
        auto out_gpu_jit = fused_fn({a_gpu, b_gpu, c_gpu});

        auto v_seq_gpu = out_gpu_sequential->to_vector();
        auto v_jit_gpu = out_gpu_jit->to_vector();

        for (size_t i = 0; i < v_seq_gpu.size(); ++i) {
            std::cout << "GPU Element " << i << ": Sequential=" << v_seq_gpu[i] << ", JIT=" << v_jit_gpu[i] << std::endl;
            if (std::abs(v_seq_gpu[i] - v_jit_gpu[i]) >= 1e-5f) {
                std::cerr << "GPU mismatch!" << std::endl;
                return 1;
            }
        }
        std::cout << "GPU JIT matches Sequential!" << std::endl;
    } else {
        std::cout << "\nGPU not available, skipping GPU tests." << std::endl;
    }

    std::cout << "\n--- Testing Automated Graph Capture / Tracing ---" << std::endl;
    auto x0 = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, dev_cpu);
    auto x1 = Tensor::from_vector({0.5f, 1.0f, 2.0f}, {3}, dev_cpu);

    auto traced_fn = Tracer::trace({x0, x1}, [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
        auto y = Ops::add(inputs[0], inputs[1]);
        return Ops::relu(y);
    }, "traced_relu_add");

    auto eager_out = Ops::relu(Ops::add(x0, x1));
    auto traced_out = (*traced_fn)({x0, x1});

    auto v_eager = eager_out->to_vector();
    auto v_traced = traced_out->to_vector();
    for (size_t i = 0; i < v_eager.size(); ++i) {
        std::cout << "Traced Element " << i << ": Eager=" << v_eager[i] << ", Traced=" << v_traced[i] << std::endl;
        assert(std::abs(v_eager[i] - v_traced[i]) < 1e-5f);
    }
    std::cout << "Automated Tracing matched eager execution!" << std::endl;

    if (active_backend && active_backend->is_available()) {
        auto x0_gpu = x0->to(dev_gpu);
        auto x1_gpu = x1->to(dev_gpu);
        auto traced_out_gpu = (*traced_fn)({x0_gpu, x1_gpu});
        auto v_traced_gpu = traced_out_gpu->to_vector();
        for (size_t i = 0; i < v_eager.size(); ++i) {
            std::cout << "GPU Traced Element " << i << ": Eager=" << v_eager[i] << ", Traced=" << v_traced_gpu[i] << std::endl;
            assert(std::abs(v_eager[i] - v_traced_gpu[i]) < 1e-5f);
        }
        std::cout << "GPU Automated Tracing matched eager execution!" << std::endl;
    }

    std::cout << "\nSUCCESS: JIT Elementwise compilation verified!" << std::endl;
    return 0;
}
