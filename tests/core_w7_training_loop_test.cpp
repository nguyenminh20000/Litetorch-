#include "litetorch/tensor.h"
#include "litetorch/autograd.h"
#include "litetorch/ops.h"
#include <cstdio>
#include <string>
#include <vector>

using namespace litetorch;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); ++failures; } } while (0)

static bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

static bool near(float a, float b, float tol = 1e-4f) {
    float d = a > b ? a - b : b - a;
    return d < tol;
}

int main() {
    Device cpu0(DeviceType::CPU, 0);
    Device cpu1(DeviceType::CPU, 1);

    {
        auto w0 = Tensor::randn({8, 4}, cpu0, true);
        auto w = w0->to(cpu1);
        CHECK((bool)w->creator);
        auto b0 = Tensor::randn({4}, cpu0, true);
        auto b = b0->to(cpu1);
        Autograd::retain_grad(w);
        Autograd::retain_grad(b);
        auto lr = Tensor::from_vector({-0.01f}, {1}, cpu1, false);
        for (int it = 0; it < 3; ++it) {
            auto x = Tensor::randn({2, 8}, cpu1, false);
            auto out = Ops::matmul(x, w);
            auto outb = Ops::add(out, b);
            auto loss = Ops::sum(outb);
            bool threw = false;
            try {
                loss->backward();
            } catch (const std::exception& e) {
                threw = true;
                std::printf("iter %d threw: %s\n", it, e.what());
            }
            CHECK(!threw);
            CHECK(w->grad != nullptr);
            {
                NoGradGuard ng;
                w->add_(Ops::mul(w->grad, lr));
                b->add_(Ops::mul(b->grad, lr));
            }
            w->zero_grad();
            b->zero_grad();
        }
        std::printf("training_loop_with_moved_params done\n");
    }

    {
        auto w0 = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, cpu0, true);
        auto w = w0->to(cpu1);
        auto b0 = Tensor::from_vector({0.5f, -0.5f}, {2}, cpu0, true);
        auto b = b0->to(cpu1);
        Autograd::retain_grad(w);
        Autograd::retain_grad(b);
        auto x = Tensor::from_vector({1.0f, 1.0f, 2.0f, 2.0f}, {2, 2}, cpu1, false);
        for (int it = 0; it < 2; ++it) {
            auto out = Ops::matmul(x, w);
            auto outb = Ops::add(out, b);
            auto loss = Ops::sum(outb);
            loss->backward();
            float* wg = w->grad->data_ptr();
            float* bg = b->grad->data_ptr();
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    CHECK(near(wg[i * 2 + j], 3.0f));
            CHECK(near(bg[0], 2.0f));
            CHECK(near(bg[1], 2.0f));
            w->zero_grad();
            b->zero_grad();
        }
        std::printf("grad_correctness_per_iteration done\n");
    }

    {
        auto x = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu0, true);
        auto y = Ops::mul(x, x);
        auto loss = Ops::sum(y);
        loss->backward();
        bool threw = false;
        std::string msg;
        try {
            loss->backward();
        } catch (const std::runtime_error& e) {
            threw = true;
            msg = e.what();
        }
        CHECK(threw);
        CHECK(contains(msg, "already freed"));
        std::printf("double_backward_same_graph_throws done\n");
    }

    {
        auto x = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu0, true);
        auto y = Ops::mul(x, x);
        auto loss = Ops::sum(y);
        Autograd::backward(loss, false, true);
        float g1 = x->grad->data_ptr()[0];
        Autograd::backward(loss, false, true);
        float g2 = x->grad->data_ptr()[0];
        CHECK(near(g1, 2.0f));
        CHECK(near(g2, 4.0f));
        std::printf("retain_graph_allows_double_backward done\n");
    }

    {
        auto w = Tensor::randn({1, 1}, cpu0, true);
        auto b = Tensor::zeros({1}, cpu0, true);
        auto lr = Tensor::from_vector({-0.1f}, {1}, cpu0, false);
        auto x = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4, 1}, cpu0, false);
        auto t = Tensor::from_vector({2.0f, 4.0f, 6.0f, 8.0f}, {4, 1}, cpu0, false);
        float first = 0.0f, last = 0.0f;
        for (int it = 0; it < 20; ++it) {
            auto pred = Ops::add(Ops::matmul(x, w), b);
            auto diff = Ops::sub(pred, t);
            auto loss = Ops::mean(Ops::mul(diff, diff));
            float lv = loss->data_ptr()[0];
            if (it == 0) first = lv;
            last = lv;
            loss->backward();
            {
                NoGradGuard ng;
                w->add_(Ops::mul(w->grad, lr));
                b->add_(Ops::mul(b->grad, lr));
            }
            w->zero_grad();
            b->zero_grad();
        }
        CHECK(last < first);
        std::printf("loss_decreases done first=%f last=%f\n", first, last);
    }

    if (failures == 0) std::printf("ALL CORE_W7 TESTS PASSED\n");
    return failures == 0 ? 0 : 1;
}
