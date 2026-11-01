#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <cstdio>
#include <vector>

using namespace litetorch;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); ++failures; } } while (0)

static bool shape_eq(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    return a == b;
}

static bool throws_fn(std::shared_ptr<Tensor> (*f)(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>),
                      std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    try {
        f(a, b);
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

int main() {
    Device cpu(DeviceType::CPU, 0);
    auto z = [&](std::vector<int64_t> s, DataType dt = DataType::FP32) {
        return Tensor::zeros(s, cpu, false, dt);
    };

    CHECK(shape_eq(Ops::add(z({0}), z({0, 1, 0}))->shape, {0, 1, 0}));
    CHECK(shape_eq(Ops::add(z({0, 3, 1}), z({4}))->shape, {0, 3, 4}));
    CHECK(shape_eq(Ops::add(z({0}), z({0, 0}))->shape, {0, 0}));
    CHECK(shape_eq(Ops::add(z({1}), z({0}))->shape, {0}));
    CHECK(shape_eq(Ops::add(z({0}), z({1}))->shape, {0}));
    CHECK(shape_eq(Ops::add(z({0}), z({0}))->shape, {0}));
    CHECK(shape_eq(Ops::add(z({2, 1}), z({0}))->shape, {2, 0}));
    CHECK(shape_eq(Ops::add(z({0}), z({2, 1}))->shape, {2, 0}));
    CHECK(shape_eq(Ops::add(z({1}), z({5}))->shape, {5}));
    CHECK(shape_eq(Ops::add(z({5}), z({1}))->shape, {5}));
    CHECK(shape_eq(Ops::add(z({3}), z({3}))->shape, {3}));
    CHECK(shape_eq(Ops::add(z({2, 3}), z({3}))->shape, {2, 3}));
    CHECK(shape_eq(Ops::sub(z({0}), z({0, 1, 0}))->shape, {0, 1, 0}));
    CHECK(shape_eq(Ops::mul(z({0, 3, 1}), z({4}))->shape, {0, 3, 4}));
    CHECK(shape_eq(Ops::div(z({0}), z({0, 0}))->shape, {0, 0}));
    CHECK(shape_eq(Ops::add(z({0}, DataType::INT32), z({0, 1, 0}, DataType::INT32))->shape, {0, 1, 0}));
    CHECK(shape_eq(Ops::add(z({0, 3, 1}, DataType::INT32), z({4}, DataType::INT32))->shape, {0, 3, 4}));
    CHECK(throws_fn(Ops::add, z({2}), z({3})));
    CHECK(throws_fn(Ops::add, z({2, 3}), z({4})));
    CHECK(Ops::allclose(z({0}), z({0, 1, 0}), 1e-5f, 1e-8f));
    CHECK(Ops::allclose(z({0, 3, 1}), z({4}), 1e-5f, 1e-8f));
    CHECK(shape_eq(Ops::isclose(z({0}), z({0, 1, 0}), 1e-5f, 1e-8f)->shape, {0, 1, 0}));
    CHECK(shape_eq(Ops::isclose(z({0, 3, 1}), z({4}), 1e-5f, 1e-8f)->shape, {0, 3, 4}));

    if (failures == 0) std::printf("ALL CORE_W55 TESTS PASSED\n");
    return failures == 0 ? 0 : 1;
}
