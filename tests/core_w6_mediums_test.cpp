#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/dtensor.h"
#include "litetorch/storage_guard.h"
#include "litetorch/allocator.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace litetorch;

static const float QNAN = std::numeric_limits<float>::quiet_NaN();
static Device cpu(DeviceType::CPU, 0);

void test_from_vector_short_throws() {
    bool threw = false;
    try {
        Tensor::from_vector({1.0f, 2.0f}, {4}, cpu);
    } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    std::cout << "from_vector_short_throws ok" << std::endl;
}

void test_from_vector_long_throws() {
    bool threw = false;
    try {
        Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {2}, cpu);
    } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    std::cout << "from_vector_long_throws ok" << std::endl;
}

void test_from_vector_exact_ok() {
    auto t = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, cpu);
    auto v = t->to_vector();
    assert(v.size() == 3 && v[0] == 1.0f && v[2] == 3.0f);
    std::cout << "from_vector_exact_ok ok" << std::endl;
}

void test_slice_zero_step_throws() {
    auto t = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, cpu);
    bool threw = false;
    try {
        t->slice(0, 0, 3, 0);
    } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    std::cout << "slice_zero_step_throws ok" << std::endl;
}

void test_slice_negative_step_throws() {
    auto t = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, cpu);
    bool threw = false;
    try {
        t->slice(0, 2, 0, -1);
    } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    std::cout << "slice_negative_step_throws ok" << std::endl;
}

void test_slice_positive_step_ok() {
    auto t = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, cpu);
    auto s = t->slice(0, 0, 4, 2);
    assert(s->shape[0] == 2);
    auto v = s->to_vector();
    assert(v[0] == 1.0f && v[1] == 3.0f);
    std::cout << "slice_positive_step_ok ok" << std::endl;
}

void test_max_propagates_nan() {
    auto t = Tensor::from_vector({1.0f, QNAN, 2.0f}, {3}, cpu);
    auto m = Ops::max(t);
    assert(std::isnan(m->to_vector()[0]));
    std::cout << "max_propagates_nan ok" << std::endl;
}

void test_max_no_nan_unchanged() {
    auto t = Tensor::from_vector({1.0f, 5.0f, 2.0f}, {3}, cpu);
    auto m = Ops::max(t);
    assert(m->to_vector()[0] == 5.0f);
    std::cout << "max_no_nan_unchanged ok" << std::endl;
}

void test_min_propagates_nan() {
    auto t = Tensor::from_vector({1.0f, QNAN, 2.0f}, {3}, cpu);
    auto m = Ops::min(t);
    assert(std::isnan(m->to_vector()[0]));
    std::cout << "min_propagates_nan ok" << std::endl;
}

void test_amax_dim_propagates_nan() {
    auto t = Tensor::from_vector({1.0f, 2.0f, QNAN, 4.0f}, {2, 2}, cpu);
    auto m = Ops::amax(t, {1}, false);
    auto v = m->to_vector();
    assert(v[0] == 2.0f && std::isnan(v[1]));
    std::cout << "amax_dim_propagates_nan ok" << std::endl;
}

void test_max_backward_routes_to_nan() {
    auto x = Tensor::from_vector({1.0f, QNAN, 3.0f}, {3}, cpu, true);
    auto y = Ops::max(x);
    assert(std::isnan(y->to_vector()[0]));
    y->backward();
    auto g = x->grad->to_vector();
    assert(g[0] == 0.0f && g[1] == 1.0f && g[2] == 0.0f);
    std::cout << "max_backward_routes_to_nan ok" << std::endl;
}

void test_redistribute_throws_loudly() {
    auto local = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu);
    auto mesh = std::make_shared<distributed::DeviceMesh>(std::vector<int>{1}, "x");
    distributed::DTensor dt(local, mesh, {});
    bool threw = false;
    try {
        dt.redistribute(mesh, {});
    } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    std::cout << "redistribute_throws_loudly ok" << std::endl;
}

void test_free_cpu_unknown_throws() {
    int stack_var = 0;
    bool threw = false;
    try {
        CachingAllocator::get().free_cpu(&stack_var);
    } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    std::cout << "free_cpu_unknown_throws ok" << std::endl;
}

void test_free_gpu_unknown_throws() {
    int stack_var = 0;
    bool threw = false;
    try {
        CachingAllocator::get().free_gpu(&stack_var);
    } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    std::cout << "free_gpu_unknown_throws ok" << std::endl;
}

void test_storage_use_guard_shared_header() {
    auto t = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu);
    {
        StorageUseGuard guard({t->storage});
        assert(t->storage->in_use == true);
    }
    assert(t->storage->in_use == false);
    std::cout << "storage_use_guard_shared_header ok" << std::endl;
}

int main() {
    test_from_vector_short_throws();
    test_from_vector_long_throws();
    test_from_vector_exact_ok();
    test_slice_zero_step_throws();
    test_slice_negative_step_throws();
    test_slice_positive_step_ok();
    test_max_propagates_nan();
    test_max_no_nan_unchanged();
    test_min_propagates_nan();
    test_amax_dim_propagates_nan();
    test_max_backward_routes_to_nan();
    test_redistribute_throws_loudly();
    test_free_cpu_unknown_throws();
    test_free_gpu_unknown_throws();
    test_storage_use_guard_shared_header();
    std::cout << "ALL CORE_W6 TESTS PASSED" << std::endl;
    return 0;
}
