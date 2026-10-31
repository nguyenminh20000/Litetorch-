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
    Device cpu(DeviceType::CPU, 0);

    {
        auto x = Tensor::randn({2, 2}, cpu, true, DataType::FP64);
        AutogradFunc f = [](const std::vector<std::shared_ptr<Tensor>>& in) -> std::vector<std::shared_ptr<Tensor>> {
            return { Ops::sum(Ops::mul(in[0], in[0])) };
        };
        std::string err;
        bool ok = false;
        try {
            ok = autograd_gradcheck(f, {x}, 1e-4, 5e-3, 1e-2, true, err);
        } catch (const std::exception& e) {
            std::printf("fp64 gradcheck threw: %s\n", e.what());
        }
        CHECK(ok);
        std::printf("fp64_gradcheck done ok=%d err=%s\n", (int)ok, err.c_str());
    }

    {
        auto x = Tensor::randn({2, 3}, cpu, true);
        auto w = Tensor::randn({3, 2}, cpu, true);
        auto y = Ops::matmul(x, w);
        auto l1 = Ops::sum(y);
        l1->backward();
        auto z = Ops::add(y, y);
        auto l2 = Ops::sum(z);
        bool threw = false;
        std::string msg;
        try {
            l2->backward();
        } catch (const std::exception& e) {
            threw = true;
            msg = e.what();
        }
        CHECK(threw);
        CHECK(contains(msg, "already freed"));
        CHECK(!contains(msg, "_M_range_check"));
        std::printf("freed_stateful_node_throws done: %s\n", msg.c_str());
    }

    {
        Device cpu1(DeviceType::CPU, 1);
        auto w0 = Tensor::randn({4, 3}, cpu, true);
        auto w = w0->to(cpu1);
        auto b0 = Tensor::randn({3}, cpu, true);
        auto b = b0->to(cpu1);
        for (int it = 0; it < 3; ++it) {
            auto x = Tensor::randn({2, 4}, cpu1, false);
            auto out = Ops::matmul(x, w);
            auto loss = Ops::sum(Ops::add(out, b));
            bool threw = false;
            try {
                loss->backward();
            } catch (const std::exception& e) {
                threw = true;
                std::printf("iter %d threw: %s\n", it, e.what());
            }
            CHECK(!threw);
            CHECK(w->grad == nullptr);
            CHECK(b->grad == nullptr);
            CHECK(w0->grad != nullptr);
            CHECK(b0->grad != nullptr);
            if (w0->grad && b0->grad) {
                auto w0g = w0->grad->to_vector();
                auto xv = x->to_vector();
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 3; ++j)
                        CHECK(near(w0g[i * 3 + j], xv[i] + xv[4 + i]));
                auto b0g = b0->grad->to_vector();
                for (int j = 0; j < 3; ++j)
                    CHECK(near(b0g[j], 2.0f));
            }
            w->zero_grad();
            b->zero_grad();
            w0->zero_grad();
            b0->zero_grad();
        }
        std::printf("stateless_shared_node_reexec done\n");
    }

    {
        auto x = Tensor::randn({2, 2}, cpu, true);
        auto l = Ops::sum(Ops::mul(x, x));
        l->backward();
        bool threw = false;
        std::string msg;
        try {
            l->backward();
        } catch (const std::exception& e) {
            threw = true;
            msg = e.what();
        }
        CHECK(threw);
        CHECK(contains(msg, "already freed"));
        std::printf("double_backward_throws done\n");
    }

    {
        auto x = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
        auto l = Ops::sum(Ops::mul(x, x));
        l->backward(nullptr, false, true);
        l->backward(nullptr, false, true);
        auto g = x->grad->to_vector();
        CHECK(near(g[0], 4.0f));
        CHECK(near(g[1], 8.0f));
        std::printf("retain_graph_double_backward done\n");
    }

    if (failures == 0) std::printf("ALL CORE_W9 TESTS PASSED\n");
    return failures ? 1 : 0;
}
