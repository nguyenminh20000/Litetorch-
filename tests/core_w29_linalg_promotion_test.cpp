#include "litetorch/tensor.h"
#include "litetorch/linalg.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <functional>
#include <tuple>
#include <vector>

using namespace litetorch;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static Device cpu(DeviceType::CPU, 0);

template <typename F>
static bool throws_runtime(F&& f) {
    try {
        f();
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

static std::shared_ptr<Tensor> mk_i64(const std::vector<int64_t>& v) {
    auto t = Tensor::create({(int64_t)v.size()}, cpu, false, DataType::INT64);
    int64_t* p = reinterpret_cast<int64_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
static std::shared_ptr<Tensor> mk_i64(const std::vector<int64_t>& v, const std::vector<int64_t>& shape) {
    auto t = Tensor::create(shape, cpu, false, DataType::INT64);
    int64_t* p = reinterpret_cast<int64_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
static std::shared_ptr<Tensor> mk_i32(const std::vector<int32_t>& v) {
    auto t = Tensor::create({(int64_t)v.size()}, cpu, false, DataType::INT32);
    int32_t* p = reinterpret_cast<int32_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
static std::shared_ptr<Tensor> mk_u8(const std::vector<uint8_t>& v) {
    auto t = Tensor::create({(int64_t)v.size()}, cpu, false, DataType::UINT8);
    uint8_t* p = reinterpret_cast<uint8_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
static std::shared_ptr<Tensor> mk_bool(const std::vector<uint8_t>& v) {
    auto t = Tensor::create({(int64_t)v.size()}, cpu, false, DataType::BOOL);
    uint8_t* p = reinterpret_cast<uint8_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i] ? 1 : 0;
    return t;
}
static std::shared_ptr<Tensor> mk_f32(const std::vector<float>& v) {
    auto t = Tensor::create({(int64_t)v.size()}, cpu, false, DataType::FP32);
    float* p = reinterpret_cast<float*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
static std::shared_ptr<Tensor> mk_f32(const std::vector<float>& v, const std::vector<int64_t>& shape) {
    auto t = Tensor::create(shape, cpu, false, DataType::FP32);
    float* p = reinterpret_cast<float*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
static std::shared_ptr<Tensor> mk_f64(const std::vector<double>& v) {
    auto t = Tensor::create({(int64_t)v.size()}, cpu, false, DataType::FP64);
    double* p = reinterpret_cast<double*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
static std::shared_ptr<Tensor> mk_f64(const std::vector<double>& v, const std::vector<int64_t>& shape) {
    auto t = Tensor::create(shape, cpu, false, DataType::FP64);
    double* p = reinterpret_cast<double*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
static std::shared_ptr<Tensor> mk_f16(const std::vector<uint16_t>& v) {
    auto t = Tensor::create({(int64_t)v.size()}, cpu, false, DataType::FP16);
    uint16_t* p = reinterpret_cast<uint16_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
static std::vector<int64_t> rd_i64(std::shared_ptr<Tensor> t) {
    const int64_t* p = reinterpret_cast<const int64_t*>(t->raw_data_ptr());
    size_t n = (size_t)t->numel();
    return std::vector<int64_t>(p, p + n);
}
static std::vector<int32_t> rd_i32(std::shared_ptr<Tensor> t) {
    const int32_t* p = reinterpret_cast<const int32_t*>(t->raw_data_ptr());
    size_t n = (size_t)t->numel();
    return std::vector<int32_t>(p, p + n);
}
static std::vector<uint8_t> rd_u8(std::shared_ptr<Tensor> t) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(t->raw_data_ptr());
    size_t n = (size_t)t->numel();
    return std::vector<uint8_t>(p, p + n);
}
static bool close_d(const std::vector<double>& a, const std::vector<double>& b, double tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

int main() {
    const int64_t B = (int64_t)((1LL << 30) + 1);
    const int64_t B2 = (int64_t)1152921506754330625LL;

    {
        auto r = linalg::outer(mk_i64({B, 2}), mk_i64({B, 3}));
        CHECK(r->dtype == DataType::INT64);
        CHECK(r->shape == std::vector<int64_t>({2, 2}));
        CHECK(rd_i64(r) == std::vector<int64_t>({B2, 3221225475LL, 2147483650LL, 6}));
    }
    {
        auto r = linalg::outer(mk_i64({1, 2}), mk_i32({3, 4}));
        CHECK(r->dtype == DataType::INT64);
        CHECK(rd_i64(r) == std::vector<int64_t>({3, 4, 6, 8}));
    }
    {
        auto r = linalg::outer(mk_u8({1, 2}), mk_i64({3, 4}));
        CHECK(r->dtype == DataType::INT64);
        CHECK(rd_i64(r) == std::vector<int64_t>({3, 4, 6, 8}));
    }
    {
        auto r = linalg::outer(mk_i64({1, 2}), mk_f32({3.0f, 4.0f}));
        CHECK(r->dtype == DataType::FP32);
        CHECK(close_d(r->flat_doubles(), {3.0, 4.0, 6.0, 8.0}, 1e-6));
    }
    {
        auto r = linalg::outer(mk_f32({1.0f, 2.0f}), mk_f64({3.0, 4.0}));
        CHECK(r->dtype == DataType::FP64);
        CHECK(close_d(r->flat_doubles(), {3.0, 4.0, 6.0, 8.0}, 1e-12));
    }
    {
        auto r = linalg::outer(mk_bool({1, 0}), mk_bool({1, 1}));
        CHECK(r->dtype == DataType::BOOL);
        CHECK(rd_u8(r) == std::vector<uint8_t>({1, 1, 0, 0}));
    }
    {
        auto r = linalg::outer(mk_f16({0x3c00, 0x4000}), mk_f16({0x4200, 0x4400}));
        CHECK(r->dtype == DataType::FP16);
        CHECK(close_d(r->flat_doubles(), {3.0, 4.0, 6.0, 8.0}, 1e-6));
    }
    {
        auto r = linalg::inner(mk_i64({B, 1}), mk_i64({B, 1}));
        CHECK(r->dtype == DataType::INT64);
        CHECK(rd_i64(r) == std::vector<int64_t>({B2 + 1}));
    }
    {
        auto r = linalg::inner(mk_i64({9007199254740993LL}), mk_i64({1}));
        CHECK(r->dtype == DataType::INT64);
        CHECK(rd_i64(r) == std::vector<int64_t>({9007199254740993LL}));
    }
    {
        auto r = linalg::inner(mk_i32({1, 2, 3}), mk_i32({4, 5, 6}));
        CHECK(r->dtype == DataType::INT32);
        CHECK(rd_i32(r) == std::vector<int32_t>({32}));
    }
    {
        auto r = linalg::inner(mk_u8({1, 2, 3}), mk_u8({4, 5, 6}));
        CHECK(r->dtype == DataType::UINT8);
        CHECK(rd_u8(r) == std::vector<uint8_t>({32}));
    }
    {
        auto r = linalg::inner(mk_f32({1.0f, 2.0f, 3.0f}), mk_f32({4.0f, 5.0f, 6.0f}));
        CHECK(r->dtype == DataType::FP32);
        CHECK(close_d(r->flat_doubles(), {32.0}, 1e-6));
    }
    CHECK(throws_runtime([&] { linalg::inner(mk_i64({1}), mk_i32({1})); }));
    CHECK(throws_runtime([&] { linalg::inner(mk_bool({1}), mk_bool({1})); }));
    {
        auto r = linalg::vdot(mk_i64({B, 2}), mk_i64({B, 3}));
        CHECK(r->dtype == DataType::INT64);
        CHECK(rd_i64(r) == std::vector<int64_t>({B2 + 6}));
    }
    {
        auto r = linalg::vdot(mk_f32({1.0f, 2.0f, 3.0f}), mk_f32({4.0f, 5.0f, 6.0f}));
        CHECK(r->dtype == DataType::FP32);
        CHECK(close_d(r->flat_doubles(), {32.0}, 1e-6));
    }
    CHECK(throws_runtime([&] { linalg::vdot(mk_i64({1}), mk_i32({1})); }));
    CHECK(throws_runtime([&] { linalg::vdot(mk_bool({1}), mk_bool({1})); }));
    {
        auto r = linalg::kron(mk_i64({B, 2}), mk_i64({3, 2}));
        CHECK(r->dtype == DataType::INT64);
        CHECK(rd_i64(r) == std::vector<int64_t>({3221225475LL, 2147483650LL, 6, 4}));
    }
    {
        auto r = linalg::kron(mk_f64({1.0, 2.0}), mk_f64({3.0, 4.0}));
        CHECK(r->dtype == DataType::FP64);
        CHECK(close_d(r->flat_doubles(), {3.0, 4.0, 6.0, 8.0}, 1e-12));
    }
    {
        auto r = linalg::cross(mk_i64({B, 0, 0}), mk_i64({0, B, 0}));
        CHECK(r->dtype == DataType::INT64);
        CHECK(rd_i64(r) == std::vector<int64_t>({0, 0, B2}));
    }
    {
        auto r = linalg::cross(mk_f32({1.0f, 0.0f, 0.0f}), mk_f32({0.0f, 1.0f, 0.0f}));
        CHECK(r->dtype == DataType::FP32);
        CHECK(close_d(r->flat_doubles(), {0.0, 0.0, 1.0}, 1e-6));
    }
    CHECK(throws_runtime([&] { linalg::cross(mk_i64({1, 2, 3}), mk_i32({1, 2, 3})); }));
    CHECK(throws_runtime([&] { linalg::cross(mk_bool({1, 0, 0}), mk_bool({0, 1, 0})); }));
    {
        auto r = linalg::vecdot(mk_f32({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), mk_f32({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}));
        CHECK(r->dtype == DataType::FP32);
        CHECK(close_d(r->flat_doubles(), {17.0, 53.0}, 1e-6));
    }
    {
        auto r = linalg::vecdot(mk_f16({0x3c00, 0x4000, 0x4200}), mk_f16({0x4400, 0x4500, 0x4600}));
        CHECK(r->dtype == DataType::FP16);
        CHECK(close_d(r->flat_doubles(), {32.0}, 1e-6));
    }
    CHECK(throws_runtime([&] { linalg::vecdot(mk_i64({1, 2, 3}), mk_i64({4, 5, 6})); }));
    CHECK(throws_runtime([&] { linalg::vecdot(mk_f32({1.0f, 2.0f}), mk_f64({1.0, 2.0})); }));
    {
        auto a = mk_f32({2.0f, 1.0f, 1.0f, 3.0f}, {2, 2});
        auto b = mk_f32({4.0f, 5.0f});
        auto r = linalg::solve(a, b);
        CHECK(r->dtype == DataType::FP32);
        CHECK(close_d(r->flat_doubles(), {1.4, 1.2}, 1e-6));
    }
    {
        auto a = mk_f64({2.0, 1.0, 1.0, 3.0}, {2, 2});
        auto b = mk_f64({4.0, 5.0});
        auto r = linalg::solve(a, b);
        CHECK(r->dtype == DataType::FP64);
        CHECK(close_d(r->flat_doubles(), {1.4, 1.2}, 1e-12));
    }
    CHECK(throws_runtime([&] {
        linalg::solve(mk_i64({2, 1, 1, 3}, {2, 2}), mk_i64({4, 5}));
    }));
    CHECK(throws_runtime([&] {
        linalg::solve(mk_f32({2.0f, 1.0f, 1.0f, 3.0f}, {2, 2}), mk_f64({4.0, 5.0}));
    }));
    {
        auto a = mk_f32({2.0f, 1.0f, 1.0f, 3.0f}, {2, 2});
        auto b = mk_f32({4.0f, 5.0f});
        auto res = linalg::lstsq(a, b);
        auto sol = std::get<0>(res);
        auto rank = std::get<2>(res);
        auto sv = std::get<3>(res);
        CHECK(sol->dtype == DataType::FP32);
        CHECK(close_d(sol->flat_doubles(), {1.4, 1.2}, 1e-6));
        CHECK(rank->dtype == DataType::INT64);
        CHECK(rd_i64(rank) == std::vector<int64_t>({2}));
        CHECK(sv->dtype == DataType::FP32);
    }
    CHECK(throws_runtime([&] {
        linalg::lstsq(mk_i64({2, 1, 1, 3}, {2, 2}), mk_i64({4, 5}));
    }));
    CHECK(throws_runtime([&] {
        linalg::lstsq(mk_f32({2.0f, 1.0f, 1.0f, 3.0f}, {2, 2}), mk_f64({4.0, 5.0}));
    }));

    if (fails == 0) std::printf("ALL PASS\n");
    else std::printf("%d FAILURES\n", fails);
    return fails == 0 ? 0 : 1;
}
