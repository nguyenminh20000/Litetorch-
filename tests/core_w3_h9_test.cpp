#include "litetorch/tensor.h"
#include "litetorch/autograd.h"
#include "litetorch/ops.h"
#include "litetorch/amp.h"
#include "../src/ops/autocast_policy.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace litetorch;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); ++failures; } } while (0)
static bool near(float a, float b, float tol = 1e-3f) { float d = a > b ? a - b : b - a; return d < tol; }

int main() {
    Device cpu(DeviceType::CPU);

    {
        using amp::AutocastPolicy;
        using amp::autocast_policy_for;
        CHECK(autocast_policy_for("softmax") == AutocastPolicy::FP32_ONLY);
        CHECK(autocast_policy_for("log_softmax") == AutocastPolicy::FP32_ONLY);
        CHECK(autocast_policy_for("layer_norm") == AutocastPolicy::FP32_ONLY);
        CHECK(autocast_policy_for("batch_norm") == AutocastPolicy::FP32_ONLY);
        CHECK(autocast_policy_for("batch_norm2d") == AutocastPolicy::FP32_ONLY);
        CHECK(autocast_policy_for("cross_entropy_loss") == AutocastPolicy::FP32_ONLY);
        CHECK(autocast_policy_for("nll_loss") == AutocastPolicy::FP32_ONLY);
        CHECK(autocast_policy_for("matmul") == AutocastPolicy::FP16_ELIGIBLE);
        CHECK(autocast_policy_for("linear") == AutocastPolicy::FP16_ELIGIBLE);
        CHECK(autocast_policy_for("conv2d") == AutocastPolicy::FP16_ELIGIBLE);
        CHECK(autocast_policy_for("attention") == AutocastPolicy::FP16_ELIGIBLE);
        CHECK(autocast_policy_for("mul") == AutocastPolicy::FP16_ELIGIBLE);
        CHECK(autocast_policy_for("add") == AutocastPolicy::FP16_ELIGIBLE);
        CHECK(autocast_policy_for("cast") == AutocastPolicy::PASSTHROUGH);
        CHECK(autocast_policy_for("reshape") == AutocastPolicy::PASSTHROUGH);
        CHECK(autocast_policy_for("some_unknown_op") == AutocastPolicy::PASSTHROUGH);
    }

    {
        auto x32 = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, cpu, false);
        auto x16 = Ops::cast(x32, DataType::FP16);
        std::shared_ptr<Tensor> y;
        {
            amp::AutocastGuard guard(true, DataType::FP16);
            y = Ops::softmax(x16, -1);
        }
        CHECK(y->dtype == DataType::FP16);
        auto v = y->to_vector();
        double s = std::exp(1.0) + std::exp(2.0) + std::exp(3.0);
        CHECK(v.size() == 3);
        CHECK(near(v[0], (float)(std::exp(1.0) / s)));
        CHECK(near(v[1], (float)(std::exp(2.0) / s)));
        CHECK(near(v[2], (float)(std::exp(3.0) / s)));
    }

    {
        auto x32 = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, cpu, false);
        auto w32 = Tensor::from_vector({1.0f, 0.0f, 0.0f, 1.0f}, {2, 2}, cpu, false);
        auto x16 = Ops::cast(x32, DataType::FP16);
        auto w16 = Ops::cast(w32, DataType::FP16);
        std::shared_ptr<Tensor> y;
        {
            amp::AutocastGuard guard(true, DataType::FP16);
            y = Ops::matmul(x16, w16);
        }
        CHECK(y->dtype == DataType::FP16);
        auto v = y->to_vector();
        CHECK(v.size() == 4);
        CHECK(near(v[0], 1.0f) && near(v[1], 2.0f) && near(v[2], 3.0f) && near(v[3], 4.0f));
    }

    {
        auto x32 = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, cpu, false);
        auto x16 = Ops::cast(x32, DataType::FP16);
        std::shared_ptr<Tensor> y;
        {
            amp::AutocastGuard guard(true, DataType::FP16);
            y = Ops::layer_norm(x16, {2}, nullptr, nullptr, 1e-5f);
        }
        CHECK(y->dtype == DataType::FP16);
        auto v = y->to_vector();
        CHECK(v.size() == 4);
        double m = 1.5;
        double var = ((1.0 - m) * (1.0 - m) + (2.0 - m) * (2.0 - m)) / 2.0;
        double inv = 1.0 / std::sqrt(var + 1e-5);
        CHECK(near(v[0], (float)((1.0 - m) * inv)));
        CHECK(near(v[1], (float)((2.0 - m) * inv)));
    }

    if (failures == 0) std::printf("core_w3_h9_test: ALL PASS\n");
    return failures;
}
