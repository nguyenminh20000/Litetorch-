#include "litetorch/tensor.h"
#include "litetorch/autograd.h"
#include "litetorch/ops.h"
#include "litetorch/custom_ops.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace litetorch;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); ++failures; } } while (0)
static bool near(float a, float b, float tol = 1e-4f) { float d = a > b ? a - b : b - a; return d < tol; }

int main() {
    Device cpu(DeviceType::CPU);

    {
        auto x = Tensor::from_vector({0.5f}, {1}, cpu, true);
        auto h = Ops::sigmoid(x);
        auto y = Ops::sum(h);
        h.reset();
        Autograd::backward(y);
        CHECK(x->grad != nullptr);
        double s = 1.0 / (1.0 + std::exp(-0.5));
        auto g = x->grad->to_vector();
        CHECK(g.size() == 1 && near(g[0], (float)(s * (1.0 - s))));
    }

    {
        auto x = Tensor::from_vector({0.5f}, {1}, cpu, true);
        auto h = Ops::tanh(x);
        auto y = Ops::sum(h);
        h.reset();
        Autograd::backward(y);
        CHECK(x->grad != nullptr);
        double t = std::tanh(0.5);
        auto g = x->grad->to_vector();
        CHECK(g.size() == 1 && near(g[0], (float)(1.0 - t * t)));
    }

    {
        auto x = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
        auto h = Ops::softmax(x, -1);
        auto y = Ops::sum(h);
        h.reset();
        Autograd::backward(y);
        CHECK(x->grad != nullptr);
        auto g = x->grad->to_vector();
        CHECK(g.size() == 2 && near(g[0], 0.0f) && near(g[1], 0.0f));
    }

    {
        auto x = Tensor::from_vector({4.0f}, {1}, cpu, true);
        auto h = Ops::sqrt(x);
        auto y = Ops::sum(h);
        h.reset();
        Autograd::backward(y);
        CHECK(x->grad != nullptr);
        auto g = x->grad->to_vector();
        CHECK(g.size() == 1 && near(g[0], 0.25f));
    }

    {
        auto x = Tensor::from_vector({1.0f}, {1}, cpu, true);
        auto h = Ops::exp(x);
        auto y = Ops::sum(h);
        h.reset();
        Autograd::backward(y);
        CHECK(x->grad != nullptr);
        auto g = x->grad->to_vector();
        CHECK(g.size() == 1 && near(g[0], (float)std::exp(1.0)));
    }

    {
        auto x = Tensor::from_vector({4.0f}, {1}, cpu, true);
        auto h = Ops::rsqrt(x);
        auto y = Ops::sum(h);
        h.reset();
        Autograd::backward(y);
        CHECK(x->grad != nullptr);
        auto g = x->grad->to_vector();
        CHECK(g.size() == 1 && near(g[0], -0.0625f));
    }

    {
        auto x = Tensor::from_vector({0.0f}, {1}, cpu, true);
        auto h = Ops::tan(x);
        auto y = Ops::sum(h);
        h.reset();
        Autograd::backward(y);
        CHECK(x->grad != nullptr);
        auto g = x->grad->to_vector();
        CHECK(g.size() == 1 && near(g[0], 1.0f));
    }

    {
        auto x = Tensor::from_vector({2.0f}, {1}, cpu, true);
        auto h = Ops::reciprocal(x);
        auto y = Ops::sum(h);
        h.reset();
        Autograd::backward(y);
        CHECK(x->grad != nullptr);
        auto g = x->grad->to_vector();
        CHECK(g.size() == 1 && near(g[0], -0.25f));
    }

    {
        auto x = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 2, 2}, cpu, true);
        auto h = Ops::max_pool2d(x, 2);
        auto y = Ops::sum(h);
        h.reset();
        Autograd::backward(y);
        CHECK(x->grad != nullptr);
        auto g = x->grad->to_vector();
        CHECK(g.size() == 4 && near(g[0], 0.0f) && near(g[1], 0.0f) && near(g[2], 0.0f) && near(g[3], 1.0f));
    }

    {
        auto x = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 2, 2}, cpu, true);
        auto h = Ops::avg_pool2d(x, 2);
        auto y = Ops::sum(h);
        h.reset();
        Autograd::backward(y);
        CHECK(x->grad != nullptr);
        auto g = x->grad->to_vector();
        CHECK(g.size() == 4 && near(g[0], 0.25f) && near(g[1], 0.25f) && near(g[2], 0.25f) && near(g[3], 0.25f));
    }

    {
        bool saw_inps = false;
        bool saw_out = false;
        size_t inps_size = 0;
        custom_ops::Registry::get().register_op("w3_square",
            [](const std::vector<std::shared_ptr<Tensor>>& args) -> std::shared_ptr<Tensor> {
                auto x = args[0];
                auto out = Tensor::create(x->shape, x->device, false, x->dtype);
                const float* xp = x->data_ptr();
                float* op = out->data_ptr();
                for (size_t i = 0; i < (size_t)x->numel(); ++i) op[i] = xp[i] * xp[i];
                return out;
            },
            nullptr,
            [&](std::shared_ptr<Tensor> grad_output, const std::vector<std::shared_ptr<Tensor>>& inps, std::shared_ptr<Tensor> out) -> std::vector<std::shared_ptr<Tensor>> {
                saw_inps = true;
                saw_out = (out != nullptr);
                inps_size = inps.size();
                auto x = inps[0];
                auto g = Tensor::create(x->shape, x->device, false, x->dtype);
                const float* xp = x->data_ptr();
                const float* gp = grad_output->data_ptr();
                float* rp = g->data_ptr();
                for (size_t i = 0; i < (size_t)x->numel(); ++i) rp[i] = 2.0f * xp[i] * gp[i];
                return {g};
            });
        auto x = Tensor::from_vector({3.0f}, {1}, cpu, true);
        auto h = custom_ops::Registry::get().call("w3_square", {x});
        auto y = Ops::sum(h);
        h.reset();
        Autograd::backward(y);
        CHECK(saw_inps);
        CHECK(saw_out);
        CHECK(inps_size == 1);
        CHECK(x->grad != nullptr);
        auto g = x->grad->to_vector();
        CHECK(g.size() == 1 && near(g[0], 6.0f));
    }

    if (failures == 0) std::printf("core_w3_h4_test: ALL PASS\n");
    return failures;
}
