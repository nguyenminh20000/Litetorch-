#include "litetorch/tensor.h"
#include "litetorch/autograd.h"
#include "litetorch/ops.h"
#include <cstdio>
#include <vector>
#include <string>
#include <cmath>

using namespace litetorch;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static Device cpu(DeviceType::CPU, 0);

static std::shared_ptr<Tensor> mk_i64(const std::vector<int64_t>& v) {
    auto t = Tensor::create({(int64_t)v.size()}, cpu, false, DataType::INT64);
    int64_t* p = reinterpret_cast<int64_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
static std::shared_ptr<Tensor> mk_u8(const std::vector<uint8_t>& v) {
    auto t = Tensor::create({(int64_t)v.size()}, cpu, false, DataType::UINT8);
    uint8_t* p = reinterpret_cast<uint8_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
static std::shared_ptr<Tensor> mk_i8(const std::vector<int8_t>& v) {
    auto t = Tensor::create({(int64_t)v.size()}, cpu, false, DataType::INT8);
    int8_t* p = reinterpret_cast<int8_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
static std::shared_ptr<Tensor> mk_i16(const std::vector<int16_t>& v) {
    auto t = Tensor::create({(int64_t)v.size()}, cpu, false, DataType::INT16);
    int16_t* p = reinterpret_cast<int16_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
static std::shared_ptr<Tensor> mk_i32(const std::vector<int32_t>& v) {
    auto t = Tensor::create({(int64_t)v.size()}, cpu, false, DataType::INT32);
    int32_t* p = reinterpret_cast<int32_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
static std::shared_ptr<Tensor> mk_bool(const std::vector<uint8_t>& v) {
    auto t = Tensor::create({(int64_t)v.size()}, cpu, false, DataType::BOOL);
    uint8_t* p = reinterpret_cast<uint8_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i] ? 1 : 0;
    return t;
}
static std::vector<int64_t> rd_i64(std::shared_ptr<Tensor> t) {
    const int64_t* p = reinterpret_cast<const int64_t*>(t->raw_data_ptr());
    size_t n = (size_t)t->numel();
    return std::vector<int64_t>(p, p + n);
}
static std::vector<int16_t> rd_i16(std::shared_ptr<Tensor> t) {
    const int16_t* p = reinterpret_cast<const int16_t*>(t->raw_data_ptr());
    size_t n = (size_t)t->numel();
    return std::vector<int16_t>(p, p + n);
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
static bool near_f(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

static void test_promote_table_full() {
    using DT = DataType;
    struct Pair { DT a, b, exp; };
    const Pair cases[] = {
        {DT::BOOL, DT::BOOL, DT::BOOL}, {DT::BOOL, DT::UINT8, DT::UINT8},
        {DT::BOOL, DT::INT8, DT::INT8}, {DT::BOOL, DT::INT16, DT::INT16},
        {DT::BOOL, DT::INT32, DT::INT32}, {DT::BOOL, DT::INT64, DT::INT64},
        {DT::BOOL, DT::FP16, DT::FP16}, {DT::BOOL, DT::BF16, DT::BF16},
        {DT::BOOL, DT::FP32, DT::FP32}, {DT::BOOL, DT::FP64, DT::FP64},
        {DT::UINT8, DT::UINT8, DT::UINT8}, {DT::UINT8, DT::INT8, DT::INT16},
        {DT::UINT8, DT::INT16, DT::INT16}, {DT::UINT8, DT::INT32, DT::INT32},
        {DT::UINT8, DT::INT64, DT::INT64}, {DT::UINT8, DT::FP16, DT::FP16},
        {DT::UINT8, DT::BF16, DT::BF16}, {DT::UINT8, DT::FP32, DT::FP32},
        {DT::UINT8, DT::FP64, DT::FP64},
        {DT::INT8, DT::INT8, DT::INT8}, {DT::INT8, DT::INT16, DT::INT16},
        {DT::INT8, DT::INT32, DT::INT32}, {DT::INT8, DT::INT64, DT::INT64},
        {DT::INT8, DT::FP16, DT::FP16}, {DT::INT8, DT::BF16, DT::BF16},
        {DT::INT8, DT::FP32, DT::FP32}, {DT::INT8, DT::FP64, DT::FP64},
        {DT::INT16, DT::INT16, DT::INT16}, {DT::INT16, DT::INT32, DT::INT32},
        {DT::INT16, DT::INT64, DT::INT64}, {DT::INT16, DT::FP16, DT::FP16},
        {DT::INT16, DT::BF16, DT::BF16}, {DT::INT16, DT::FP32, DT::FP32},
        {DT::INT16, DT::FP64, DT::FP64},
        {DT::INT32, DT::INT32, DT::INT32}, {DT::INT32, DT::INT64, DT::INT64},
        {DT::INT32, DT::FP16, DT::FP16}, {DT::INT32, DT::BF16, DT::BF16},
        {DT::INT32, DT::FP32, DT::FP32}, {DT::INT32, DT::FP64, DT::FP64},
        {DT::INT64, DT::INT64, DT::INT64}, {DT::INT64, DT::FP16, DT::FP16},
        {DT::INT64, DT::BF16, DT::BF16}, {DT::INT64, DT::FP32, DT::FP32},
        {DT::INT64, DT::FP64, DT::FP64},
        {DT::FP16, DT::FP16, DT::FP16}, {DT::FP16, DT::BF16, DT::FP32},
        {DT::FP16, DT::FP32, DT::FP32}, {DT::FP16, DT::FP64, DT::FP64},
        {DT::BF16, DT::BF16, DT::BF16}, {DT::BF16, DT::FP32, DT::FP32},
        {DT::BF16, DT::FP64, DT::FP64},
        {DT::FP32, DT::FP32, DT::FP32}, {DT::FP32, DT::FP64, DT::FP64},
        {DT::FP64, DT::FP64, DT::FP64},
    };
    for (auto& c : cases) {
        CHECK(promote_types(c.a, c.b) == c.exp);
        CHECK(promote_types(c.b, c.a) == c.exp);
    }
    std::printf("PASS test_promote_table_full\n");
}

static void test_add_uint8_int8() {
    auto a = mk_u8({200, 250});
    auto b = mk_i8({100, 100});
    auto c = Ops::add(a, b);
    CHECK(c->dtype == DataType::INT16);
    auto v = rd_i16(c);
    CHECK(v[0] == 300 && v[1] == 350);
    auto d = Ops::add(b, a);
    CHECK(d->dtype == DataType::INT16);
    auto w = rd_i16(d);
    CHECK(w[0] == 300 && w[1] == 350);
    std::printf("PASS test_add_uint8_int8\n");
}

static void test_add_uint8_int16() {
    auto a = mk_u8({200});
    auto b = mk_i16({30000});
    auto c = Ops::add(a, b);
    CHECK(c->dtype == DataType::INT16);
    CHECK(rd_i16(c)[0] == 30200);
    std::printf("PASS test_add_uint8_int16\n");
}

static void test_add_uint8_int32() {
    auto a = mk_u8({200});
    auto b = mk_i32({2000000000});
    auto c = Ops::add(a, b);
    CHECK(c->dtype == DataType::INT32);
    CHECK(rd_i32(c)[0] == 2000000200);
    std::printf("PASS test_add_uint8_int32\n");
}

static void test_add_uint8_int64() {
    auto a = mk_u8({200, 1});
    auto b = mk_i64({1073741824LL, 5});
    auto c = Ops::add(a, b);
    CHECK(c->dtype == DataType::INT64);
    auto v = rd_i64(c);
    CHECK(v[0] == 1073742024LL);
    CHECK(v[1] == 6LL);
    std::printf("PASS test_add_uint8_int64\n");
}

static void test_sub_mul_mixed_sign() {
    auto a = mk_u8({200});
    auto b = mk_i8({100});
    auto s = Ops::sub(a, b);
    CHECK(s->dtype == DataType::INT16);
    CHECK(rd_i16(s)[0] == 100);
    auto m = Ops::mul(a, b);
    CHECK(m->dtype == DataType::INT16);
    CHECK(rd_i16(m)[0] == 20000);
    std::printf("PASS test_sub_mul_mixed_sign\n");
}

static void test_add_int64_large_exact() {
    auto a = mk_i64({1073741825LL});
    auto b = mk_i64({1073741825LL});
    auto c = Ops::add(a, b);
    CHECK(c->dtype == DataType::INT64);
    auto v = rd_i64(c);
    CHECK(v[0] == 2147483650LL);
    std::printf("PASS test_add_int64_large_exact\n");
}

static void test_sub_mul_int() {
    auto a = mk_i64({10, -3});
    auto b = mk_i64({4, 7});
    auto s = Ops::sub(a, b);
    CHECK(s->dtype == DataType::INT64);
    auto vs = rd_i64(s);
    CHECK(vs[0] == 6 && vs[1] == -10);
    auto m = Ops::mul(a, b);
    CHECK(m->dtype == DataType::INT64);
    auto vm = rd_i64(m);
    CHECK(vm[0] == 40 && vm[1] == -21);
    auto c = mk_i32({1000000});
    auto d = mk_i64({2000000});
    auto e = Ops::mul(c, d);
    CHECK(e->dtype == DataType::INT64);
    CHECK(rd_i64(e)[0] == 2000000000000LL);
    std::printf("PASS test_sub_mul_int\n");
}

static void test_add_broadcast_int() {
    auto a = mk_i64({1, 2, 3, 4});
    auto b = mk_u8({10});
    auto c = Ops::add(a, b);
    CHECK(c->dtype == DataType::INT64);
    auto v = rd_i64(c);
    CHECK(v[0] == 11 && v[1] == 12 && v[2] == 13 && v[3] == 14);
    std::printf("PASS test_add_broadcast_int\n");
}

static void test_bool_add() {
    auto a = mk_bool({1, 0});
    auto b = mk_bool({1, 1});
    auto c = Ops::add(a, b);
    CHECK(c->dtype == DataType::BOOL);
    auto v = rd_u8(c);
    CHECK(v[0] == 1 && v[1] == 1);
    std::printf("PASS test_bool_add\n");
}

static void test_div_true_division() {
    auto a = mk_i64({7, -7, 1});
    auto b = mk_i64({2, 2, 3});
    auto c = Ops::div(a, b);
    CHECK(c->dtype == DataType::FP32);
    auto v = c->to_vector();
    CHECK(near_f(v[0], 3.5f));
    CHECK(near_f(v[1], -3.5f));
    CHECK(near_f(v[2], 1.0f / 3.0f, 1e-6f));
    std::printf("PASS test_div_true_division\n");
}

static void test_div_mixed_int() {
    auto a = mk_u8({7});
    auto b = mk_i64({2});
    auto c = Ops::div(a, b);
    CHECK(c->dtype == DataType::FP32);
    CHECK(near_f(c->to_vector()[0], 3.5f));
    std::printf("PASS test_div_mixed_int\n");
}

static void test_div_backward_flows() {
    auto x = Tensor::from_vector({7.0f, 8.0f}, {2}, cpu, true);
    auto y = Tensor::from_vector({2.0f, 4.0f}, {2}, cpu, true);
    auto loss = Ops::sum(Ops::div(x, y));
    loss->backward();
    CHECK(x->grad != nullptr && y->grad != nullptr);
    auto gx = x->grad->to_vector();
    auto gy = y->grad->to_vector();
    CHECK(near_f(gx[0], 0.5f) && near_f(gx[1], 0.25f));
    CHECK(near_f(gy[0], -1.75f) && near_f(gy[1], -0.5f));
    std::printf("PASS test_div_backward_flows\n");
}

static void test_cat_mixed() {
    auto a = mk_u8({1, 2});
    auto b = mk_i64({1073741824LL, 5});
    auto c = Ops::cat({a, b}, 0);
    CHECK(c->dtype == DataType::INT64);
    auto v = rd_i64(c);
    CHECK(v[0] == 1 && v[1] == 2 && v[2] == 1073741824LL && v[3] == 5);
    std::printf("PASS test_cat_mixed\n");
}

static void test_where_mixed() {
    auto cond = mk_bool({1, 0});
    auto a = mk_u8({200, 7});
    auto b = mk_i64({1073741824LL, 9});
    auto c = Ops::where(cond, a, b);
    CHECK(c->dtype == DataType::INT64);
    auto v = rd_i64(c);
    CHECK(v[0] == 200 && v[1] == 9);
    auto b2 = mk_i64({1073741825LL, 1073741825LL});
    auto c2 = Ops::where(cond, a, b2);
    auto v2 = rd_i64(c2);
    CHECK(v2[0] == 200 && v2[1] == 1073741825LL);
    std::printf("PASS test_where_mixed\n");
}

static void test_int_fp16_promote() {
    auto a = mk_i64({4});
    auto b = Tensor::from_vector({0.5f}, {1}, cpu, false, DataType::FP16);
    auto c = Ops::add(a, b);
    CHECK(c->dtype == DataType::FP16);
    std::printf("PASS test_int_fp16_promote\n");
}

static void test_cast_int_pairs() {
    auto a = mk_u8({200});
    auto b = a->cast(DataType::INT64);
    CHECK(b->dtype == DataType::INT64 && rd_i64(b)[0] == 200);
    auto c = mk_i64({-5});
    auto d = c->cast(DataType::UINT8);
    CHECK(d->dtype == DataType::UINT8 && rd_u8(d)[0] == 251);
    auto e = mk_i64({1073741824LL});
    auto f = e->cast(DataType::INT32);
    const int32_t* pf = reinterpret_cast<const int32_t*>(f->raw_data_ptr());
    CHECK(pf[0] == 1073741824);
    std::printf("PASS test_cast_int_pairs\n");
}

int main() {
    test_promote_table_full();
    test_add_uint8_int8();
    test_add_uint8_int16();
    test_add_uint8_int32();
    test_add_uint8_int64();
    test_sub_mul_mixed_sign();
    test_add_int64_large_exact();
    test_sub_mul_int();
    test_add_broadcast_int();
    test_bool_add();
    test_div_true_division();
    test_div_mixed_int();
    test_div_backward_flows();
    test_cat_mixed();
    test_where_mixed();
    test_int_fp16_promote();
    test_cast_int_pairs();
    if (fails == 0) std::printf("ALL CORE_W18 TESTS PASSED\n");
    else std::printf("%d FAILURES\n", fails);
    return fails ? 1 : 0;
}
