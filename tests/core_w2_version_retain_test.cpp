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

static bool near(float a, float b) {
    float d = a > b ? a - b : b - a;
    return d < 1e-4f;
}

int main() {
    Device cpu(DeviceType::CPU);

    {
        auto x = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
        auto h = Ops::mul(x, x);
        auto k = Ops::mul(h, h);
        auto z = Ops::sum(k);
        h->add_(Tensor::ones({2}, cpu));
        bool threw = false;
        std::string msg;
        try {
            Autograd::backward(z);
        } catch (const std::runtime_error& e) {
            threw = true;
            msg = e.what();
        }
        CHECK(threw);
        CHECK(contains(msg, "modified by an inplace operation"));
        CHECK(contains(msg, "expected version"));
    }

    {
        auto x = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
        auto h = Ops::mul(x, x);
        auto k = Ops::mul(h, h);
        auto z = Ops::sum(k);
        Autograd::backward(z);
        CHECK(x->grad != nullptr);
        std::vector<float> g = x->grad->to_vector();
        CHECK(g.size() == 2);
        CHECK(near(g[0], 4.0f));
        CHECK(near(g[1], 32.0f));
    }

    {
        auto x = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
        auto y = Ops::sum(Ops::mul(x, x));
        Autograd::backward(y, false, true);
        CHECK(x->grad != nullptr);
        std::vector<float> g1 = x->grad->to_vector();
        CHECK(near(g1[0], 2.0f));
        CHECK(near(g1[1], 4.0f));
        Autograd::backward(y);
        std::vector<float> g2 = x->grad->to_vector();
        CHECK(near(g2[0], 4.0f));
        CHECK(near(g2[1], 8.0f));
    }

    {
        auto x = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
        auto y = Ops::sum(Ops::mul(x, x));
        Autograd::backward(y);
        bool threw = false;
        std::string msg;
        try {
            Autograd::backward(y);
        } catch (const std::runtime_error& e) {
            threw = true;
            msg = e.what();
        }
        CHECK(threw);
        CHECK(contains(msg, "already freed"));
        CHECK(contains(msg, "retain_graph"));
        CHECK(!contains(msg, "out_of_range"));
    }

    {
        auto x = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
        auto h = Ops::mul(x, x);
        auto k = Ops::mul(h, h);
        auto z = Ops::sum(k);
        h->add_(Tensor::ones({2}, cpu));
        bool threw = false;
        try {
            Autograd::backward(z, true);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw);
        CHECK(!Autograd::is_create_graph_);
        auto x2 = Tensor::from_vector({3.0f}, {1}, cpu, true);
        auto y2 = Ops::sum(Ops::mul(x2, x2));
        Autograd::backward(y2);
        CHECK(x2->grad != nullptr);
        std::vector<float> g = x2->grad->to_vector();
        CHECK(g.size() == 1);
        CHECK(near(g[0], 6.0f));
    }

    if (failures == 0) std::printf("core_w2_version_retain_test: ALL PASS\n");
    else std::printf("core_w2_version_retain_test: %d FAILURES\n", failures);
    return failures == 0 ? 0 : 1;
}
