#include "litetorch/tensor.h"
#include "litetorch/autograd.h"
#include "litetorch/ops.h"
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
        auto input = Tensor::from_vector({1.0f, 3.0f, 10.0f, 20.0f}, {1, 2, 1, 2}, cpu, true);
        auto rm = Tensor::zeros({2}, cpu);
        auto rv = Tensor::ones({2}, cpu);
        auto out = Ops::batch_norm2d(input, rm, rv, nullptr, nullptr, true, 0.1f, 0.0f);
        auto v = out->to_vector();
        CHECK(v.size() == 4);
        CHECK(near(v[0], -1.0f) && near(v[1], 1.0f));
        CHECK(near(v[2], -1.0f) && near(v[3], 1.0f));
        auto y = Ops::sum(out);
        Autograd::backward(y);
        CHECK(input->grad != nullptr);
    }

    {
        auto input = Tensor::from_vector({1.0f, 3.0f, 10.0f, 20.0f}, {1, 2, 1, 2}, cpu, false);
        auto rm = Tensor::zeros({2}, cpu);
        auto rv = Tensor::ones({2}, cpu);
        auto w = Tensor::from_vector({2.0f, 0.5f}, {2}, cpu, false);
        auto b = Tensor::from_vector({1.0f, -1.0f}, {2}, cpu, false);
        auto out = Ops::batch_norm2d(input, rm, rv, w, b, true, 0.1f, 0.0f);
        auto v = out->to_vector();
        CHECK(v.size() == 4);
        CHECK(near(v[0], -1.0f) && near(v[1], 3.0f));
        CHECK(near(v[2], -1.5f) && near(v[3], -0.5f));
    }

    if (failures == 0) std::printf("core_w3_bn_affine_test: ALL PASS\n");
    return failures;
}
