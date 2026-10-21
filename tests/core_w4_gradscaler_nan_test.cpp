#include "litetorch/tensor.h"
#include "litetorch/optim.h"
#include "litetorch/grad_scaler.h"
#include "litetorch/ops.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

using namespace litetorch;

static const float QNAN = std::numeric_limits<float>::quiet_NaN();
static const float INF = std::numeric_limits<float>::infinity();
static Device cpu(DeviceType::CPU, 0);

void test_nan_grad_skips_step_and_backs_off() {
    auto p = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
    p->grad = Tensor::from_vector({QNAN, 1.0f}, {2}, cpu, false);
    optim::SGD opt({p}, 0.1);
    optim::GradScaler scaler(4.0f, 2.0f, 0.5f, 2000, true);
    float before = p->to_vector()[0];
    scaler.step(opt);
    scaler.update();
    float after = p->to_vector()[0];
    assert(before == after);
    assert(scaler.get_scale() == 2.0f);
    std::cout << "nan_grad_skips_step_and_backs_off ok" << std::endl;
}

void test_inf_grad_detected() {
    auto p = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
    p->grad = Tensor::from_vector({0.5f, -INF}, {2}, cpu, false);
    optim::SGD opt({p}, 0.1);
    optim::GradScaler scaler(4.0f, 2.0f, 0.5f, 2000, true);
    float before = p->to_vector()[1];
    scaler.step(opt);
    scaler.update();
    assert(p->to_vector()[1] == before);
    assert(scaler.get_scale() == 2.0f);
    std::cout << "inf_grad_detected ok" << std::endl;
}

void test_clean_grad_steps_and_grows() {
    auto p = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
    optim::SGD opt({p}, 0.1);
    optim::GradScaler scaler(4.0f, 2.0f, 0.5f, 2, true);
    float before = p->to_vector()[0];
    p->grad = Tensor::from_vector({1.0f, 1.0f}, {2}, cpu, false);
    scaler.step(opt);
    scaler.update();
    p->grad = Tensor::from_vector({1.0f, 1.0f}, {2}, cpu, false);
    scaler.step(opt);
    scaler.update();
    assert(p->to_vector()[0] != before);
    assert(scaler.get_scale() == 8.0f);
    std::cout << "clean_grad_steps_and_grows ok" << std::endl;
}

void test_noncontiguous_nan_grad_detected() {
    auto p = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, cpu, true);
    auto base = Tensor::from_vector({0.1f, 0.2f, QNAN, 0.4f}, {2, 2}, cpu, false);
    auto nc = base->transpose(0, 1);
    assert(!nc->is_contiguous());
    p->grad = nc;
    optim::SGD opt({p}, 0.1);
    optim::GradScaler scaler(4.0f, 2.0f, 0.5f, 2000, true);
    float before = p->to_vector()[0];
    scaler.step(opt);
    scaler.update();
    assert(p->to_vector()[0] == before);
    assert(scaler.get_scale() == 2.0f);
    std::cout << "noncontiguous_nan_grad_detected ok" << std::endl;
}

void test_fp16_nan_grad_detected() {
    auto p = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
    auto g32 = Tensor::from_vector({1.0f, QNAN}, {2}, cpu, false);
    p->grad = Ops::cast(g32, DataType::FP16);
    assert(p->grad->dtype == DataType::FP16);
    optim::SGD opt({p}, 0.1);
    optim::GradScaler scaler(4.0f, 2.0f, 0.5f, 2000, true);
    float before = p->to_vector()[0];
    scaler.step(opt);
    scaler.update();
    assert(p->to_vector()[0] == before);
    assert(scaler.get_scale() == 2.0f);
    std::cout << "fp16_nan_grad_detected ok" << std::endl;
}

int main() {
    test_nan_grad_skips_step_and_backs_off();
    test_inf_grad_detected();
    test_clean_grad_steps_and_grows();
    test_noncontiguous_nan_grad_detected();
    test_fp16_nan_grad_detected();
    {
        auto g = Tensor::from_vector({1.0f, QNAN}, {2}, cpu, false);
        float m = Ops::max(Ops::abs(g))->item();
        assert(std::isnan(m));
        std::cout << "max_propagates_nan ok" << std::endl;
    }
    std::cout << "ALL core_w4 gradscaler tests passed" << std::endl;
    return 0;
}
