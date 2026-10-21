#include "litetorch/checkpoint.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <iostream>
#include <cmath>

using namespace litetorch;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::cout << "FAIL line " << __LINE__ << ": " #cond "\n"; ++failures; } } while (0)

int main() {
    std::cout << "[core-w2b] checkpoint in-place-guard regression\n";

    {
        std::vector<float> data(16, 1.0f);
        auto x = Tensor::from_vector(data, {4, 4}, Device(DeviceType::CPU), true);
        auto fn = [](std::shared_ptr<Tensor> input) {
            return Ops::add(Ops::mul(input, input), input);
        };
        bool threw = false;
        try {
            auto out = checkpoint(fn, x);
            auto loss = Ops::sum(out);
            loss->backward();
        } catch (const std::exception& e) {
            threw = true;
            std::cout << "unexpected throw: " << e.what() << "\n";
        }
        CHECK(!threw);
        CHECK(x->grad != nullptr);
        float g = Ops::sum(x->grad)->item();
        CHECK(std::fabs(g - 48.0f) < 1e-4f);
    }

    {
        std::vector<float> data(4, 2.0f);
        auto x = Tensor::from_vector(data, {4}, Device(DeviceType::CPU), true);
        auto y = Ops::mul(x, x);
        auto fn = [](std::shared_ptr<Tensor> input) { return Ops::sum(Ops::mul(input, input)); };
        bool threw = false;
        try {
            auto out = checkpoint(fn, y);
            out->backward();
        } catch (const std::exception& e) {
            threw = true;
            std::cout << "unexpected throw: " << e.what() << "\n";
        }
        CHECK(!threw);
        CHECK(x->grad != nullptr);
        float g = Ops::sum(x->grad)->item();
        CHECK(std::fabs(g - 128.0f) < 1e-3f);
    }

    if (failures == 0) std::cout << "core_w2b_checkpoint_test: ALL PASS\n";
    return failures ? 1 : 0;
}
