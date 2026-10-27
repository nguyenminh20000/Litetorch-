#include "litetorch/tensor.h"
#include "litetorch/optim.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace litetorch;

static Device cpu(DeviceType::CPU, 0);
static int fails = 0;

static void check(const char* name, float got, float want, float tol) {
    if (std::fabs(got - want) <= tol) {
        std::printf("PASS: %s got %.6f want %.6f\n", name, got, want);
    } else {
        std::printf("FAIL: %s got %.6f want %.6f\n", name, got, want);
        ++fails;
    }
}

static void step_once(optim::NAdam& opt, const std::vector<std::shared_ptr<Tensor>>& ps,
                      const std::vector<std::vector<float>>& gs) {
    for (size_t i = 0; i < ps.size(); ++i) {
        ps[i]->grad = Tensor::from_vector(gs[i], ps[i]->shape, cpu, false);
    }
    opt.step();
}

static void test_docs_three_steps() {
    auto w = Tensor::from_vector({1.0f}, {1}, cpu, true);
    optim::NAdam opt({w}, 0.01);
    const float want[3] = {0.989435f, 0.981599f, 0.974276f};
    for (int s = 0; s < 3; ++s) {
        step_once(opt, {w}, {{0.5f}});
        char name[64];
        std::snprintf(name, sizeof(name), "docs 3-step lr=0.01 step %d", s + 1);
        check(name, w->to_vector()[0], want[s], 1e-5f);
    }
}

static void test_docs_one_step_lr01() {
    auto w = Tensor::from_vector({1.0f}, {1}, cpu, true);
    optim::NAdam opt({w}, 0.1);
    step_once(opt, {w}, {{0.5f}});
    check("docs 1-step lr=0.1", w->to_vector()[0], 0.894355f, 1e-5f);
}

static void test_twenty_steps() {
    auto w = Tensor::from_vector({1.0f}, {1}, cpu, true);
    optim::NAdam opt({w}, 0.01);
    const float want[20] = {0.989435f, 0.981599f, 0.974276f, 0.966966f, 0.959506f,
                            0.951845f, 0.943975f, 0.935903f, 0.927643f, 0.919212f,
                            0.910625f, 0.901898f, 0.893043f, 0.884074f, 0.875003f,
                            0.865838f, 0.856590f, 0.847267f, 0.837876f, 0.828424f};
    for (int s = 0; s < 20; ++s) {
        step_once(opt, {w}, {{0.5f}});
        char name[64];
        std::snprintf(name, sizeof(name), "20-step lr=0.01 step %d", s + 1);
        check(name, w->to_vector()[0], want[s], 1e-5f);
    }
}

static void test_multi_param() {
    auto w1 = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
    auto w2 = Tensor::from_vector({0.5f}, {1}, cpu, true);
    optim::NAdam opt({w1, w2}, 0.01);
    for (int s = 0; s < 3; ++s) {
        step_once(opt, {w1, w2}, {{0.5f, -0.25f}, {1.0f}});
    }
    check("multi w1[0] 3 steps", w1->to_vector()[0], 0.974276f, 1e-5f);
    check("multi w1[1] 3 steps", w1->to_vector()[1], 2.025724f, 1e-5f);
    check("multi w2[0] 3 steps", w2->to_vector()[0], 0.474276f, 1e-5f);
}

static void test_weight_decay() {
    auto w = Tensor::from_vector({1.0f}, {1}, cpu, true);
    optim::NAdam opt({w}, 0.01, {0.9, 0.999}, 1e-8f, 0.01f);
    for (int s = 0; s < 3; ++s) {
        step_once(opt, {w}, {{0.5f}});
    }
    check("weight_decay=0.01 3 steps", w->to_vector()[0], 0.974278f, 1e-5f);
}

static void test_momentum_decay() {
    auto w = Tensor::from_vector({1.0f}, {1}, cpu, true);
    optim::NAdam opt({w}, 0.01, {0.9, 0.999}, 1e-8f, 0.0f, 0.01f);
    for (int s = 0; s < 5; ++s) {
        step_once(opt, {w}, {{0.5f}});
    }
    check("momentum_decay=0.01 5 steps", w->to_vector()[0], 0.959512f, 1e-5f);
}

static void test_betas() {
    auto w = Tensor::from_vector({1.0f}, {1}, cpu, true);
    optim::NAdam opt({w}, 0.01, {0.85, 0.99});
    for (int s = 0; s < 5; ++s) {
        step_once(opt, {w}, {{0.5f}});
    }
    check("betas 0.85/0.99 5 steps", w->to_vector()[0], 0.956793f, 1e-5f);
}

static void test_add_param_group() {
    auto w1 = Tensor::from_vector({1.0f}, {1}, cpu, true);
    optim::NAdam opt({w1}, 0.01);
    for (int s = 0; s < 3; ++s) {
        step_once(opt, {w1}, {{0.5f}});
    }
    auto w2 = Tensor::from_vector({2.0f}, {1}, cpu, true);
    opt.add_param_group({w2}, 0.01, 0.0f);
    step_once(opt, {w1, w2}, {{0.5f}, {-0.25f}});
    check("add_param_group w1 step 4", w1->to_vector()[0], 0.966966f, 1e-5f);
    check("add_param_group w2 step 1", w2->to_vector()[0], 2.010565f, 1e-5f);
    if (opt.param_steps.size() != 2 || opt.param_steps[0] != 4 || opt.param_steps[1] != 1) {
        std::printf("FAIL: add_param_group param_steps not per-param\n");
        ++fails;
    } else {
        std::printf("PASS: add_param_group param_steps per-param\n");
    }
    if (opt.mu_products.size() != 2 || std::fabs(opt.mu_products[1] - opt.mu_products[0]) < 1e-9f) {
        std::printf("FAIL: add_param_group mu_products not per-param\n");
        ++fails;
    } else {
        std::printf("PASS: add_param_group mu_products per-param\n");
    }
}

static void test_step_count_state() {
    auto w = Tensor::from_vector({1.0f}, {1}, cpu, true);
    optim::NAdam opt({w}, 0.01);
    if (opt.mu_products.size() != 1 || opt.mu_products[0] != 1.0f) {
        std::printf("FAIL: mu_products init\n");
        ++fails;
    } else {
        std::printf("PASS: mu_products init 1.0\n");
    }
    step_once(opt, {w}, {{0.5f}});
    if (opt.step_count != 1 || opt.mu_products[0] >= 1.0f || opt.mu_products[0] <= 0.0f) {
        std::printf("FAIL: mu_products running product\n");
        ++fails;
    } else {
        std::printf("PASS: mu_products running product %.6f\n", opt.mu_products[0]);
    }
}

int main() {
    test_docs_three_steps();
    test_docs_one_step_lr01();
    test_twenty_steps();
    test_multi_param();
    test_weight_decay();
    test_momentum_decay();
    test_betas();
    test_add_param_group();
    test_step_count_state();
    if (fails == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d FAILURES\n", fails);
    return 1;
}
