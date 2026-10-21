#include "litetorch/tensor.h"
#include "litetorch/autograd.h"
#include "litetorch/ops.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>

using namespace litetorch;

static Device cpu_dev(DeviceType::CPU, 0);

static bool vec_eq(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-5f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::cout << "FAIL " << __func__ << ":" << __LINE__ << std::endl; ++failures; } } while (0)

static void test_slice_grad() {
    auto x = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {6}, cpu_dev, true);
    auto s = x->slice(0, 1, 4);
    CHECK((bool)s->creator);
    CHECK(s->shape == std::vector<int64_t>({3}));
    s->backward(Tensor::ones({3}, cpu_dev));
    CHECK(x->grad != nullptr);
    CHECK(vec_eq(x->grad->to_vector(), {0, 1, 1, 1, 0, 0}));
    std::cout << "PASS test_slice_grad" << std::endl;
}

static void test_slice_grad_2d() {
    auto x = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3}, cpu_dev, true);
    auto s = x->slice(1, 0, 2);
    CHECK(s->shape == (std::vector<int64_t>{2, 2}));
    s->backward(Tensor::ones({2, 2}, cpu_dev));
    CHECK(vec_eq(x->grad->to_vector(), {1, 1, 0, 1, 1, 0}));
    std::cout << "PASS test_slice_grad_2d" << std::endl;
}

static void test_index_select_grad() {
    auto x = Tensor::from_vector({10, 20, 30, 40, 50, 60}, {6}, cpu_dev, true);
    auto idx = Tensor::from_vector({2, 0, 2}, {3}, cpu_dev, false);
    auto y = x->index_select(0, idx);
    CHECK((bool)y->creator);
    CHECK(vec_eq(y->to_vector(), {30, 10, 30}));
    y->backward(Tensor::ones({3}, cpu_dev));
    CHECK(vec_eq(x->grad->to_vector(), {1, 0, 2, 0, 0, 0}));
    std::cout << "PASS test_index_select_grad" << std::endl;
}

static void test_masked_fill_grad() {
    auto x = Tensor::ones({4}, cpu_dev, true);
    auto mask = Tensor::from_vector({0, 1, 0, 1}, {4}, cpu_dev, false);
    auto y = x->masked_fill(mask, 0.0f);
    CHECK((bool)y->creator);
    CHECK(vec_eq(y->to_vector(), {1, 0, 1, 0}));
    y->backward(Tensor::ones({4}, cpu_dev));
    CHECK(vec_eq(x->grad->to_vector(), {1, 0, 1, 0}));
    std::cout << "PASS test_masked_fill_grad" << std::endl;
}

static void test_version_counter() {
    auto t = Tensor::create({3}, cpu_dev, true);
    uint64_t v0 = t->_version();
    {
        NoGradGuard ng;
        t->add_(Tensor::ones({3}, cpu_dev));
    }
    CHECK(t->_version() == v0 + 1);
    bool threw = false;
    try {
        t->copy_(Tensor::ones({3}, cpu_dev));
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    CHECK(t->_version() == v0 + 1);
    std::cout << "PASS test_version_counter" << std::endl;
}

static void test_as_strided_bounds() {
    auto x = Tensor::create({12}, cpu_dev, false);
    auto ok = x->as_strided({3, 4}, {1, 3});
    CHECK(ok->numel() == 12);
    bool threw = false;
    try {
        x->as_strided({3, 4}, {2, 3});
    } catch (const std::out_of_range&) { threw = true; }
    CHECK(threw);
    threw = false;
    try {
        x->as_strided({3, 4}, {-2, 1});
    } catch (const std::out_of_range&) { threw = true; }
    CHECK(threw);
    std::cout << "PASS test_as_strided_bounds" << std::endl;
}

static void test_as_strided_grad_sum() {
    auto x = Tensor::from_vector({1, 2, 3, 4}, {4}, cpu_dev, true);
    auto v = x->as_strided({2, 2}, {0, 0});
    v->backward(Tensor::ones({2, 2}, cpu_dev));
    CHECK(vec_eq(x->grad->to_vector(), {4, 0, 0, 0}));
    std::cout << "PASS test_as_strided_grad_sum" << std::endl;
}

static void test_data_ptr_guard() {
    auto fp32 = Tensor::create({2}, cpu_dev, false, DataType::FP32);
    CHECK(fp32->data_ptr() != nullptr);
    auto fp16 = Tensor::create({2}, cpu_dev, false, DataType::FP16);
    bool threw = false;
    try { fp16->data_ptr(); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    auto i32 = Tensor::create({2}, cpu_dev, false, DataType::INT32);
    threw = false;
    try { i32->data_ptr(); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    std::cout << "PASS test_data_ptr_guard" << std::endl;
}

static void test_int64_dtype() {
    auto t = Tensor::create({3}, cpu_dev, false, DataType::INT64);
    CHECK(t->storage->element_size() == 8);
    auto x = Tensor::from_vector({1, 2, 3}, {3}, cpu_dev, false);
    auto l = x->to_long();
    CHECK(l->dtype == DataType::INT64);
    CHECK(l->flat_doubles() == std::vector<double>({1.0, 2.0, 3.0}));
    auto a = Tensor::from_vector({1, 2}, {2}, cpu_dev, false, DataType::INT64);
    auto b = Tensor::from_vector({10, 20}, {2}, cpu_dev, false, DataType::INT64);
    a->add_(b);
    CHECK(a->flat_doubles() == std::vector<double>({11.0, 22.0}));
    auto u8 = Tensor::from_vector({1, 2}, {2}, cpu_dev, false, DataType::UINT8);
    CHECK(u8->storage->element_size() == 1);
    auto bl = Tensor::from_vector({0, 5}, {2}, cpu_dev, false, DataType::BOOL);
    CHECK(bl->flat_doubles() == std::vector<double>({0.0, 1.0}));
    std::cout << "PASS test_int64_dtype" << std::endl;
}

static void test_retain_graph() {
    auto x = Tensor::from_vector({1, 2, 3}, {3}, cpu_dev, true);
    auto c = x->clone();
    auto ones3 = Tensor::ones({3}, cpu_dev);
    c->backward(ones3, false, true);
    CHECK(vec_eq(x->grad->to_vector(), {1, 1, 1}));
    c->zero_grad();
    x->zero_grad();
    c->backward(Tensor::ones({3}, cpu_dev), false, true);
    CHECK(x->grad != nullptr);
    CHECK(vec_eq(x->grad->to_vector(), {1, 1, 1}));
    std::cout << "PASS test_retain_graph" << std::endl;
}

static void test_backward_nograd_throws() {
    auto x = Tensor::from_vector({1, 2, 3}, {3}, cpu_dev, false);
    bool threw = false;
    try { x->backward(); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    std::cout << "PASS test_backward_nograd_throws" << std::endl;
}

static void test_inplace_saved_tensor_detected() {
    auto a = Tensor::from_vector({1, 2, 3, 4}, {2, 2}, cpu_dev, true);
    auto b = Tensor::from_vector({1, 0, 0, 1}, {2, 2}, cpu_dev, true);
    auto c = Ops::matmul(a, b);
    {
        NoGradGuard ng;
        a->add_(Tensor::ones({2, 2}, cpu_dev));
    }
    bool threw = false;
    try {
        c->backward(Tensor::ones({2, 2}, cpu_dev));
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("inplace") != std::string::npos;
    }
    CHECK(threw);
    std::cout << "PASS test_inplace_saved_tensor_detected" << std::endl;
}

int main() {
    test_slice_grad();
    test_slice_grad_2d();
    test_index_select_grad();
    test_masked_fill_grad();
    test_version_counter();
    test_as_strided_bounds();
    test_as_strided_grad_sum();
    test_data_ptr_guard();
    test_int64_dtype();
    test_retain_graph();
    test_backward_nograd_throws();
    test_inplace_saved_tensor_detected();
    if (failures == 0) {
        std::cout << "ALL CORE_W1 TESTS PASSED" << std::endl;
        return 0;
    }
    std::cout << failures << " CHECKS FAILED" << std::endl;
    return 1;
}
