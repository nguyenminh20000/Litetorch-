#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/quantization.h"
#include "litetorch/jit.h"
#include "litetorch/distributed.h"
#include "litetorch/cl_backend.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace litetorch;

void test_asymmetric_quantization() {
    auto t = Tensor::from_vector({-10.0f, -5.0f, 0.0f, 2.0f}, {4});
    quantization::Calibrator cal;
    cal.collect("test_layer", t);

    auto params = cal.get_asymmetric_params("test_layer", 8);
    float scale = params.first;
    float zero_point = params.second;

    assert(scale > 0.0f);
    assert(zero_point >= -127.0f && zero_point <= 127.0f);

    auto q_cpu = Ops::fake_quantize(t, scale, zero_point, 8);
    std::vector<float> res_cpu = q_cpu->to_vector();

    assert(std::abs(res_cpu[0] - (-10.0f)) < 0.1f);
    assert(std::abs(res_cpu[3] - 2.0f) < 0.1f);

    if (CLBackend::get().is_available()) {
        auto t_gpu = t->to(Device(DeviceType::GPU));
        auto q_gpu = Ops::fake_quantize(t_gpu, scale, zero_point, 8);
        std::vector<float> res_gpu = q_gpu->to_vector();
        assert(std::abs(res_gpu[0] - (-10.0f)) < 0.1f);
        assert(std::abs(res_gpu[3] - 2.0f) < 0.1f);
    }
}

void test_jit_shape_guards_and_derivative() {
    auto x_var = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
    auto y_var = std::make_shared<JITVar>(JITVar::OpType::INPUT, "y");
    auto expr = JIT::relu(x_var * y_var);

    auto d_expr = Tracer::derivative(expr, x_var);

    auto forward_fn = std::make_shared<JITFunction>("fwd", expr, std::vector<std::shared_ptr<JITVar>>{x_var, y_var});
    auto backward_fn = std::make_shared<JITFunction>("bwd", d_expr, std::vector<std::shared_ptr<JITVar>>{x_var, y_var});

    auto x1 = Tensor::from_vector({-2.0f, 3.0f, 4.0f, -5.0f}, {4});
    auto y1 = Tensor::from_vector({2.0f, 2.0f, -2.0f, 2.0f}, {4});

    if (CLBackend::get().is_available()) {
        x1 = x1->to(Device(DeviceType::GPU));
        y1 = y1->to(Device(DeviceType::GPU));
    }

    auto out1 = (*forward_fn)({x1, y1});
    std::vector<float> res_out1 = out1->to_vector();
    assert(res_out1[0] == 0.0f);
    assert(res_out1[1] == 6.0f);
    assert(res_out1[2] == 0.0f);
    assert(res_out1[3] == 0.0f);

    forward_fn->save("jit_func_test.txt");
    auto loaded_forward_fn = JITFunction::load("jit_func_test.txt");
    auto out1_loaded = (*loaded_forward_fn)({x1, y1});
    std::vector<float> res_out1_loaded = out1_loaded->to_vector();
    assert(res_out1_loaded[0] == 0.0f);
    assert(res_out1_loaded[1] == 6.0f);
    assert(res_out1_loaded[2] == 0.0f);
    assert(res_out1_loaded[3] == 0.0f);

    auto grad1 = (*backward_fn)({x1, y1});
    std::vector<float> res_grad1 = grad1->to_vector();
    assert(res_grad1[0] == 0.0f);
    assert(res_grad1[1] == 2.0f);
    assert(res_grad1[2] == 0.0f);
    assert(res_grad1[3] == 0.0f);

    auto x2 = Tensor::from_vector({-1.0f, 2.0f, 3.0f, 4.0f, 5.0f, -6.0f}, {6});
    auto y2 = Tensor::from_vector({3.0f, 3.0f, 3.0f, 3.0f, -3.0f, 3.0f}, {6});

    if (CLBackend::get().is_available()) {
        x2 = x2->to(Device(DeviceType::GPU));
        y2 = y2->to(Device(DeviceType::GPU));
    }

    auto out2 = (*forward_fn)({x2, y2});
    std::vector<float> res_out2 = out2->to_vector();
    assert(res_out2[0] == 0.0f);
    assert(res_out2[1] == 6.0f);
    assert(res_out2[2] == 9.0f);
    assert(res_out2[3] == 12.0f);
    assert(res_out2[4] == 0.0f);

    auto grad2 = (*backward_fn)({x2, y2});
    std::vector<float> res_grad2 = grad2->to_vector();
    assert(res_grad2[0] == 0.0f);
    assert(res_grad2[1] == 3.0f);
    assert(res_grad2[2] == 3.0f);
    assert(res_grad2[3] == 3.0f);
    assert(res_grad2[4] == 0.0f);
}

void test_nccl_bridge() {
    auto& bridge = distributed::NCCLBridge::get();
    assert(!bridge.is_available());
    bridge.init(0, 1);
    
    auto t = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3});
    bridge.all_reduce(t);
    std::vector<float> res = t->to_vector();
    assert(res[0] == 1.0f);
    
    bridge.shutdown();
}

int main() {
    test_asymmetric_quantization();
    test_jit_shape_guards_and_derivative();
    test_nccl_bridge();
    std::cout << "All Stage 4 advanced features passed successfully!" << std::endl;
    return 0;
}
