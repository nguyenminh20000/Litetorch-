#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/optim.h"
#include "litetorch/distributed.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace litetorch;

void test_fp8_cast() {
    std::cout << "Running test_fp8_cast..." << std::endl;

    auto x = Tensor::from_vector({1.0f, -0.5f, 2.5f, 0.0f, 15.0f}, {1, 5}, Device(DeviceType::CPU));
    
    auto x_e4m3 = x->cast(DataType::FP8_E4M3);
    assert(x_e4m3->dtype == DataType::FP8_E4M3);
    assert(x_e4m3->storage->element_size() == 1);

    auto x_e4m3_back = x_e4m3->cast(DataType::FP32);
    auto v_e4m3 = x_e4m3_back->to_vector();

    assert(std::abs(v_e4m3[0] - 1.0f) < 0.1f);
    assert(std::abs(v_e4m3[1] - (-0.5f)) < 0.1f);
    assert(std::abs(v_e4m3[2] - 2.5f) < 0.2f);
    assert(v_e4m3[3] == 0.0f);

    auto x_e5m2 = x->cast(DataType::FP8_E5M2);
    assert(x_e5m2->dtype == DataType::FP8_E5M2);
    assert(x_e5m2->storage->element_size() == 1);

    auto x_e5m2_back = x_e5m2->cast(DataType::FP32);
    auto v_e5m2 = x_e5m2_back->to_vector();

    assert(std::abs(v_e5m2[0] - 1.0f) < 0.1f);
    assert(std::abs(v_e5m2[1] - (-0.5f)) < 0.1f);
    assert(std::abs(v_e5m2[2] - 2.5f) < 0.5f);
    assert(v_e5m2[3] == 0.0f);

    std::cout << "SUCCESS: test_fp8_cast passed!" << std::endl;
}

void test_scaled_matmul() {
    std::cout << "Running test_scaled_matmul..." << std::endl;

    auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto b = Tensor::from_vector({1.0f, 0.0f, 0.0f, 1.0f}, {2, 2});

    auto out = Ops::scaled_matmul(a, b, 0.5f, 2.0f);
    auto v = out->to_vector();

    assert(std::abs(v[0] - 1.0f) < 1e-5f);
    assert(std::abs(v[1] - 2.0f) < 1e-5f);
    assert(std::abs(v[2] - 3.0f) < 1e-5f);
    assert(std::abs(v[3] - 4.0f) < 1e-5f);

    std::cout << "SUCCESS: test_scaled_matmul passed!" << std::endl;
}

void test_ring_attention() {
    std::cout << "Running test_ring_attention..." << std::endl;

    auto q = Tensor::from_vector(std::vector<float>(1 * 2 * 4 * 4, 0.1f), {1, 2, 4, 4});
    auto k = Tensor::from_vector(std::vector<float>(1 * 2 * 4 * 4, 0.2f), {1, 2, 4, 4});
    auto v = Tensor::from_vector(std::vector<float>(1 * 2 * 4 * 4, 0.3f), {1, 2, 4, 4});

    auto out = Ops::ring_attention(q, k, v, nullptr);
    auto val = out->to_vector();

    for (float x : val) {
        assert(std::abs(x - 0.3f) < 1e-4f);
    }

    std::cout << "SUCCESS: test_ring_attention passed!" << std::endl;
}

void test_adamw_fp8() {
    std::cout << "Running test_adamw_fp8..." << std::endl;

    auto p = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {1, 3}, Device(DeviceType::CPU), true);
    p->grad = Tensor::from_vector({0.1f, -0.2f, 0.3f}, {1, 3}, Device(DeviceType::CPU));

    optim::AdamWFP8 opt({p}, 0.1f, 0.9f, 0.999f, 1e-8f, 0.01f);
    opt.step();

    auto v = p->to_vector();
    assert(v[0] < 1.0f);
    assert(v[1] > 2.0f);
    assert(v[2] < 3.0f);

    assert(opt.m[0]->dtype == DataType::FP8_E4M3);
    assert(opt.v[0]->dtype == DataType::FP8_E4M3);

    std::cout << "SUCCESS: test_adamw_fp8 passed!" << std::endl;
}

int main() {
    test_fp8_cast();
    test_scaled_matmul();
    test_ring_attention();
    test_adamw_fp8();
    std::cout << "ALL TESTS PASSED!" << std::endl;
    return 0;
}
