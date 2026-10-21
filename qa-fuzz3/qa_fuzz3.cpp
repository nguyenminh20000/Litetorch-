#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <limits>

using namespace litetorch;

static int g_pass = 0, g_fail = 0;

static void check(const std::string& name, bool ok, const std::string& msg = "") {
    if (ok) { ++g_pass; std::cout << "PASS " << name << "\n"; }
    else { ++g_fail; std::cout << "FAIL " << name << " :: " << msg << "\n"; }
}

static std::shared_ptr<Tensor> fv(const std::vector<float>& d, const std::vector<int64_t>& s, DataType dt = DataType::FP32) {
    return Tensor::from_vector(d, s, Device(DeviceType::CPU, 0), false, dt);
}

static void t_uint8_add_wrap() {
    auto a = fv({200.0f, 255.0f}, {2}, DataType::UINT8);
    auto b = fv({100.0f, 1.0f}, {2}, DataType::UINT8);
    auto c = Ops::add(a, b);
    const uint8_t* p = static_cast<const uint8_t*>(c->raw_data_ptr());
    check("uint8_add_wrap", p[0] == 44 && p[1] == 0, std::to_string(p[0]) + "," + std::to_string(p[1]));
}

static void t_uint8_sub_mul() {
    auto a = fv({0.0f, 16.0f}, {2}, DataType::UINT8);
    auto b = fv({1.0f, 16.0f}, {2}, DataType::UINT8);
    auto s = Ops::sub(a, b);
    auto m = Ops::mul(a, b);
    const uint8_t* ps = static_cast<const uint8_t*>(s->raw_data_ptr());
    const uint8_t* pm = static_cast<const uint8_t*>(m->raw_data_ptr());
    check("uint8_sub_wrap", ps[0] == 255, std::to_string(ps[0]));
    check("uint8_mul_wrap", pm[1] == 0, std::to_string(pm[1]));
}

static void t_int64_add_matmul() {
    auto a = fv({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, DataType::INT64);
    auto b = fv({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, DataType::INT64);
    auto c = Ops::add(a, b);
    const int64_t* pc = static_cast<const int64_t*>(c->raw_data_ptr());
    check("int64_add", pc[0] == 6 && pc[3] == 12, std::to_string(pc[0]) + "," + std::to_string(pc[3]));
    auto d = Ops::matmul(a, b);
    const int64_t* pd = static_cast<const int64_t*>(d->raw_data_ptr());
    bool ok = pd[0] == 19 && pd[1] == 22 && pd[2] == 43 && pd[3] == 50;
    check("int64_matmul", ok, std::to_string(pd[0]) + "," + std::to_string(pd[1]) + "," + std::to_string(pd[2]) + "," + std::to_string(pd[3]));
}

static void t_int64_max_with_indices() {
    auto a = fv({3.0f, 1.0f, 2.0f}, {3}, DataType::INT64);
    auto pr = Ops::max_with_indices(a, 0, false);
    const int64_t* pv = static_cast<const int64_t*>(pr.first->raw_data_ptr());
    const int64_t* pi = static_cast<const int64_t*>(pr.second->raw_data_ptr());
    check("int64_max_with_indices", pv[0] == 3 && pi[0] == 0, std::to_string(pv[0]) + "," + std::to_string(pi[0]));
}

static void t_max_nan_inf() {
    float nan = std::numeric_limits<float>::quiet_NaN();
    float inf = std::numeric_limits<float>::infinity();
    auto a = fv({1.0f, nan, -inf}, {3});
    float got = Ops::max(a)->item();
    check("max_nan_inf", std::isnan(got), std::to_string(got));
    auto b = fv({1.0f, 1e30f, -1e30f}, {3});
    check("max_extreme", Ops::max(b)->item() == 1e30f, std::to_string(Ops::max(b)->item()));
}

static void t_dataptr_throws() {
    bool t1 = false, t2 = false, t3 = false, r1 = false;
    try { fv({1.0f}, {1}, DataType::INT64)->data_ptr(); } catch (...) { t1 = true; }
    try { fv({1.0f}, {1}, DataType::UINT8)->data_ptr(); } catch (...) { t2 = true; }
    try { fv({1.0f}, {1}, DataType::BOOL)->data_ptr(); } catch (...) { t3 = true; }
    try { auto t = fv({1.0f}, {1}, DataType::INT64); (void)t->raw_data_ptr(); r1 = true; } catch (...) {}
    check("data_ptr_throws_int64", t1, "no throw");
    check("data_ptr_throws_uint8", t2, "no throw");
    check("data_ptr_throws_bool", t3, "no throw");
    check("raw_data_ptr_ok_int64", r1, "threw");
}

static void t_migrated_bytecopy() {
    auto src = fv({1.5f, 2.5f, 3.5f}, {3});
    auto dst = Tensor::from_vector(std::vector<float>{0.0f, 0.0f, 0.0f}, {3}, Device(DeviceType::CPU, 0));
    std::memcpy(dst->raw_data_ptr(), src->raw_data_ptr(), 3 * sizeof(float));
    check("raw_bytecopy", std::memcmp(dst->raw_data_ptr(), src->raw_data_ptr(), 3 * sizeof(float)) == 0, "mismatch");
}

static void t_inplace_guard_preserved() {
    bool threw = false;
    try {
        auto x = Tensor::from_vector(std::vector<float>{2.0f, 4.0f}, {2}, Device(DeviceType::CPU, 0), true);
        auto y = Ops::mul(x, x);
        auto z = Ops::sum(y);
        x->add_(fv({1.0f, 1.0f}, {2}));
        z->backward();
    } catch (const std::exception&) { threw = true; }
    check("inplace_version_guard_throws", threw, "no throw after merge");
}

static void t_empty_tensors() {
    bool ok = false;
    try {
        auto a = fv({}, {0});
        auto b = fv({}, {0});
        auto c = Ops::add(a, b);
        ok = (c->numel() == 0);
    } catch (...) {}
    check("empty_add", ok, "crashed or wrong numel");
}

int main() {
    t_uint8_add_wrap();
    t_uint8_sub_mul();
    t_int64_add_matmul();
    t_int64_max_with_indices();
    t_max_nan_inf();
    t_dataptr_throws();
    t_migrated_bytecopy();
    t_inplace_guard_preserved();
    t_empty_tensors();
    std::cout << "----\nRESULT qa_fuzz3: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
