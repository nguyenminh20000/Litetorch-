#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/jit.h"
#include "litetorch/autograd.h"
#include <iostream>
#include <string>
#include <vector>

using namespace litetorch;

static int failures = 0;

static void check_shape(const std::string& label, const std::shared_ptr<Tensor>& t, const std::vector<int64_t>& want) {
    int64_t want_numel = 1;
    for (auto d : want) want_numel *= d;
    if (t->shape != want || t->numel() != want_numel) {
        std::cout << "FAIL " << label << ": got shape [";
        for (size_t i = 0; i < t->shape.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << t->shape[i];
        }
        std::cout << "] numel " << t->numel() << ", want shape [";
        for (size_t i = 0; i < want.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << want[i];
        }
        std::cout << "] numel " << want_numel << std::endl;
        ++failures;
        return;
    }
    std::cout << "PASS " << label << std::endl;
}

int main() {
    auto cpu = Device(DeviceType::CPU, 0);
    auto x = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
    auto y = std::make_shared<JITVar>(JITVar::OpType::INPUT, "y");

    {
        JITFunction add_fn("add", x + y, {x, y});
        auto a = Tensor::from_vector({}, {2, 0}, cpu);
        auto b = Tensor::from_vector({10.0f, 20.0f}, {2, 1}, cpu);
        auto out = add_fn({a, b});
        check_shape("[2,0]+[2,1]->[2,0]", out, {2, 0});
        auto out2 = add_fn({a, b});
        check_shape("[2,0]+[2,1] repeat->[2,0]", out2, {2, 0});
        if (add_fn.shape_cache_hits() < 1) {
            std::cout << "FAIL [2,0]+[2,1] cache hit expected on repeat call" << std::endl;
            ++failures;
        } else {
            std::cout << "PASS [2,0]+[2,1] cache hit on repeat call" << std::endl;
        }
    }

    {
        JITFunction add_fn("add", x + y, {x, y});
        auto a = Tensor::from_vector({}, {0, 3}, cpu);
        auto b = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, cpu);
        auto out = add_fn({a, b});
        check_shape("[0,3]+[3]->[0,3]", out, {0, 3});
    }

    {
        JITFunction add_fn("add", x + y, {x, y});
        auto a = Tensor::from_vector({}, {0}, cpu);
        auto b = Tensor::from_vector({}, {0}, cpu);
        auto out = add_fn({a, b});
        check_shape("[0]+[0]->[0]", out, {0});
    }

    {
        JITFunction add_fn("add", x + y, {x, y});
        auto a = Tensor::from_vector({1.0f, 2.0f}, {2, 1}, cpu);
        auto b = Tensor::from_vector({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, {2, 3}, cpu);
        auto out = add_fn({a, b});
        check_shape("[2,1]+[2,3]->[2,3] (nonzero sanity)", out, {2, 3});
        auto want = Ops::add(a, b)->to_vector();
        auto got = out->to_vector();
        bool ok = got.size() == want.size();
        for (size_t i = 0; ok && i < got.size(); ++i) {
            if (std::abs(got[i] - want[i]) >= 1e-5f) ok = false;
        }
        if (!ok) {
            std::cout << "FAIL [2,1]+[2,3] values differ from eager" << std::endl;
            ++failures;
        } else {
            std::cout << "PASS [2,1]+[2,3] values match eager" << std::endl;
        }
    }

    if (failures == 0) {
        std::cout << "ALL_PASS" << std::endl;
        return 0;
    }
    std::cout << "FAILURES " << failures << std::endl;
    return 1;
}
