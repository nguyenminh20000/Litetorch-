#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <limits>

using namespace litetorch;

static int g_pass = 0, g_fail = 0;

static void expect_throw(const std::string& name, void (*fn)()) {
    bool threw = false;
    try { fn(); } catch (const std::exception&) { threw = true; }
    if (threw) { ++g_pass; std::cout << "PASS " << name << "\n"; }
    else { ++g_fail; std::cout << "FAIL " << name << " :: did not throw\n"; }
}

static std::shared_ptr<Tensor> e0() {
    return Tensor::from_vector(std::vector<float>{}, {0}, Device(DeviceType::CPU, 0));
}
static std::shared_ptr<Tensor> e03() {
    return Tensor::from_vector(std::vector<float>{}, {0, 3}, Device(DeviceType::CPU, 0));
}

static void t1() { Ops::max(e0()); }
static void t2() { Ops::max(e03(), 0); }
static void t3() { Ops::max_with_indices(e0(), 0, false); }

static void t_nonempty_still_correct() {
    float nan = std::numeric_limits<float>::quiet_NaN();
    auto a = Tensor::from_vector(std::vector<float>{1.0f, nan, -3.0f}, {3}, Device(DeviceType::CPU, 0));
    float got = Ops::max(a)->item();
    if (std::isnan(got)) { ++g_pass; std::cout << "PASS max_nonempty_nan\n"; }
    else { ++g_fail; std::cout << "FAIL max_nonempty_nan :: got " << got << "\n"; }
}

int main() {
    expect_throw("max_empty_throws_full", t1);
    expect_throw("max_empty_throws_dim", t2);
    expect_throw("max_with_indices_empty_throws", t3);
    t_nonempty_still_correct();
    std::cout << "----\nRESULT be_max_empty_throw_test: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
