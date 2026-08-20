#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/cl_backend.h"
#include "litetorch/guided_decoding.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace litetorch;
using namespace litetorch::nn;

void test_opencl_command_graphs() {
    std::cout << "Running test_opencl_command_graphs..." << std::endl;

    if (!CLBackend::get().is_available()) {
        std::cout << "OpenCL is not available. Skipping OpenCL Command Graphs test." << std::endl;
        return;
    }

    auto device = Device(DeviceType::GPU);
    auto t1 = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, device);
    auto t2 = Tensor::from_vector({10.0f, 20.0f, 30.0f, 40.0f}, {4}, device);

    OpenCLCommandGraph graph;
    graph.start_recording();

    auto out_tmp = Ops::add(t1, t2);

    graph.stop_recording();

    assert(graph.commands.size() > 0);

    float* clear_ptr = new float[4]{0.0f, 0.0f, 0.0f, 0.0f};
    CLBackend::get().write(out_tmp->storage->get_gpu_ptr(), 4 * sizeof(float), clear_ptr);
    delete[] clear_ptr;

    graph.replay();
    CLBackend::get().finish();

    std::vector<float> res(4);
    CLBackend::get().read(out_tmp->storage->get_gpu_ptr(), 4 * sizeof(float), res.data());

    assert(std::abs(res[0] - 11.0f) < 1e-5f);
    assert(std::abs(res[1] - 22.0f) < 1e-5f);
    assert(std::abs(res[2] - 33.0f) < 1e-5f);
    assert(std::abs(res[3] - 44.0f) < 1e-5f);

    std::cout << "SUCCESS: test_opencl_command_graphs passed!" << std::endl;
}

void test_guided_decoding() {
    std::cout << "Running test_guided_decoding..." << std::endl;

    std::vector<std::string> vocabulary = {"123", "abc", "789", "xyz"};
    GuidedDecoder decoder(vocabulary);

    auto logits = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device(DeviceType::CPU));
    decoder.apply_mask(logits, "", "digits");

    float* log_ptr = logits->data_ptr();
    assert(log_ptr[0] == 1.0f);
    assert(log_ptr[1] == -1e9f);
    assert(log_ptr[2] == 3.0f);
    assert(log_ptr[3] == -1e9f);

    auto logits2 = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device(DeviceType::CPU));
    decoder.apply_mask(logits2, "", "prefix:abc");

    float* log_ptr2 = logits2->data_ptr();
    assert(log_ptr2[0] == -1e9f);
    assert(log_ptr2[1] == 2.0f);
    assert(log_ptr2[2] == -1e9f);
    assert(log_ptr2[3] == -1e9f);

    std::cout << "SUCCESS: test_guided_decoding passed!" << std::endl;
}

void test_w8a8_quantization() {
    std::cout << "Running test_w8a8_quantization..." << std::endl;

    auto x = Tensor::from_vector({1.5f, 2.5f, -3.5f, 4.5f}, {2, 2}, Device(DeviceType::CPU));
    auto w = Tensor::from_vector({0.5f, -1.5f, 2.0f, 1.0f}, {2, 2}, Device(DeviceType::CPU));

    auto out = Ops::w8a8_matmul(x, w, 0.5f, 0.5f);

    float* out_ptr = out->data_ptr();
    assert(out->shape[0] == 2);
    assert(out->shape[1] == 2);

    float val0 = (std::round(1.5f/0.5f)*std::round(0.5f/0.5f) + std::round(2.5f/0.5f)*std::round(-1.5f/0.5f)) * 0.25f;
    assert(std::abs(out_ptr[0] - val0) < 1e-5f);

    if (CLBackend::get().is_available()) {
        auto dev_gpu = Device(DeviceType::GPU);
        auto x_gpu = x->to(dev_gpu);
        auto w_gpu = w->to(dev_gpu);
        auto out_gpu = Ops::w8a8_matmul(x_gpu, w_gpu, 0.5f, 0.5f);
        float* out_gpu_ptr = out_gpu->data_ptr();
        assert(std::abs(out_gpu_ptr[0] - val0) < 1e-5f);
    }

    std::cout << "SUCCESS: test_w8a8_quantization passed!" << std::endl;
}

int main() {
    test_opencl_command_graphs();
    test_guided_decoding();
    test_w8a8_quantization();
    std::cout << "ALL STAGE 2 ADVANCED OPTIMIZATION TESTS PASSED!" << std::endl;
    return 0;
}
