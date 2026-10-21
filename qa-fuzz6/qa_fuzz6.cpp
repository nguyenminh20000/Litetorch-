#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/onnx.h"
#include "litetorch/allocator.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <functional>
#include <string>
#include <vector>
#include <unistd.h>

using namespace litetorch;
using namespace litetorch::nn;

static int g_pass = 0, g_fail = 0;
static void check(const std::string& name, bool cond) {
    if (cond) { ++g_pass; std::printf("PASS %s\n", name.c_str()); }
    else { ++g_fail; std::printf("FAIL %s\n", name.c_str()); }
}
static bool threw_with(const std::string& needle, const std::function<void()>& f) {
    try { f(); } catch (const std::exception& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}
static long rss_kb() {
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return -1;
    long size = 0, resident = 0;
    std::fscanf(f, "%ld %ld", &size, &resident);
    std::fclose(f);
    return resident * sysconf(_SC_PAGESIZE) / 1024;
}
static Device cpu() { return Device(DeviceType::CPU, 0); }

static std::shared_ptr<Tensor> i64t(const std::vector<int64_t>& d, const std::vector<int64_t>& shape) {
    auto t = Tensor::create(shape, cpu(), false, DataType::INT64);
    std::memcpy(t->raw_data_ptr(), d.data(), d.size() * sizeof(int64_t));
    return t;
}
static std::vector<int64_t> i64v(std::shared_ptr<Tensor> t) {
    std::vector<int64_t> v(t->numel());
    std::memcpy(v.data(), t->raw_data_ptr(), v.size() * sizeof(int64_t));
    return v;
}
static std::shared_ptr<Tensor> u8t(const std::vector<uint8_t>& d, const std::vector<int64_t>& shape) {
    auto t = Tensor::create(shape, cpu(), false, DataType::UINT8);
    std::memcpy(t->raw_data_ptr(), d.data(), d.size());
    return t;
}
static std::vector<uint8_t> u8v(std::shared_ptr<Tensor> t) {
    std::vector<uint8_t> v(t->numel());
    std::memcpy(v.data(), t->raw_data_ptr(), v.size());
    return v;
}
static std::shared_ptr<Tensor> bt(const std::vector<uint8_t>& d, const std::vector<int64_t>& shape) {
    auto t = Tensor::create(shape, cpu(), false, DataType::BOOL);
    std::memcpy(t->raw_data_ptr(), d.data(), d.size());
    return t;
}
static std::vector<uint8_t> bv(std::shared_ptr<Tensor> t) {
    std::vector<uint8_t> v(t->numel());
    std::memcpy(v.data(), t->raw_data_ptr(), v.size());
    return v;
}

static void t_evict_stress() {
    auto& a = CachingAllocator::get();
    a.set_max_cpu_cache_size(4 * 1024 * 1024);
    auto guard = Tensor::create({256 * 1024}, cpu(), false, DataType::FP32);
    std::memset(guard->raw_data_ptr(), 0xAB, guard->numel() * 4);
    long r0 = rss_kb();
    bool guard_ok = true;
    for (int i = 0; i < 200; ++i) {
        void* p = a.allocate_cpu(10 * 1024 * 1024);
        std::memset(p, i & 0xFF, 10 * 1024 * 1024);
        a.free_cpu(p);
        if (i % 25 == 0) {
            unsigned char* g = (unsigned char*)guard->raw_data_ptr();
            for (size_t k = 0; k < guard->numel() * 4; k += 4096)
                if (g[k] != 0xAB) guard_ok = false;
        }
    }
    a.empty_cache();
    check("evict_live_guard_intact", guard_ok);
    check("evict_cache_empty", a.get_cached_cpu_bytes() == 0);
    long r1 = rss_kb();
    check("evict_no_leak", r0 >= 0 && r1 >= 0 && r1 < r0 + 32 * 1024);
}

static void t_version_stress() {
    int leaf_threw = 0, saved_threw = 0;
    int ok_grad = 0;
    for (int i = 0; i < 1000; ++i) {
        auto x = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, cpu(), true);
        auto y = Ops::sum(Ops::mul(x, x));
        try {
            x->add_(Tensor::ones({4}, cpu()));
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("in-place") != std::string::npos) ++leaf_threw;
        }
        auto xa = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu(), true);
        auto z = Ops::mul(xa, xa);
        auto z2 = Tensor::from_vector({3.0f, 4.0f}, {2}, cpu(), true);
        auto yb = Ops::sum(Ops::mul(z, z2));
        try {
            z->add_(Tensor::ones({2}, cpu()));
            yb->backward();
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("inplace operation") != std::string::npos) ++saved_threw;
        }
        if (i % 100 == 0) {
            auto x2 = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, cpu(), true);
            auto y2 = Ops::sum(Ops::mul(x2, x2));
            y2->backward();
            float* g = x2->grad->data_ptr();
            if (g[0] == 2.0f && g[1] == 4.0f && g[2] == 6.0f && g[3] == 8.0f) ++ok_grad;
        }
    }
    check("version_1000_leaf_inplace_throws", leaf_threw == 1000);
    check("version_1000_saved_inplace_throws", saved_threw == 1000);
    check("version_grad_still_correct", ok_grad == 10);
}

static void t_dtype_i64() {
    auto a = i64t({100, -200, 0, 7}, {4});
    auto b = i64t({50, 60, -5, 8}, {4});
    check("i64_add", i64v(Ops::add(a, b)) == std::vector<int64_t>({150, -140, -5, 15}));
    check("i64_mul", i64v(Ops::mul(i64t({3, -4}, {2}), i64t({5, 6}, {2}))) == std::vector<int64_t>({15, -24}));
    check("i64_sub", i64v(Ops::sub(b, a)) == std::vector<int64_t>({-50, 260, -5, 1}));
    auto m1 = i64t({1, 2, 3, 4}, {2, 2});
    auto m2 = i64t({5, 6, 7, 8}, {2, 2});
    check("i64_matmul", i64v(Ops::matmul(m1, m2)) == std::vector<int64_t>({19, 22, 43, 50}));
    check("i64_max", i64t({0}, {1})->numel() == 1 && i64v(Ops::max(i64t({3, 1, 2}, {3})))[0] == 3);
    auto pr = Ops::max_with_indices(i64t({1, 5, 4, 2}, {2, 2}), 1, false);
    check("i64_max_with_indices", i64v(pr.first) == std::vector<int64_t>({5, 4}) && i64v(pr.second) == std::vector<int64_t>({1, 0}));
    auto t = i64t({10, 20, 30}, {3});
    check("i64_index_select", i64v(t->index_select(0, i64t({2, 0}, {2}))) == std::vector<int64_t>({30, 10}));
    auto s = i64t({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {10});
    auto sl = s->slice(0, 2, 8, 2);
    check("i64_slice_strided", sl->numel() == 3 && Ops::sum(sl)->item() == 12.0f);
    auto f = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu());
    auto mask = bt({1, 0}, {2});
    auto mf = f->masked_fill(mask, 9.0f);
    float* mp = mf->data_ptr();
    check("i64_masked_fill", mp[0] == 9.0f && mp[1] == 2.0f);
    auto e1 = i64t({}, {0});
    auto e2 = i64t({}, {0});
    check("i64_empty_add", Ops::add(e1, e2)->numel() == 0);
    check("i64_empty_max_throws", threw_with("non-empty", [&]() { Ops::max(i64t({}, {0})); }));
    {
        auto r = Ops::add(i64t({1, 2}, {2}), Tensor::from_vector({1.0f, 2.0f}, {2}, cpu()));
        float* rp = r->data_ptr();
        check("i64_fp32_promotes_fp32", r->dtype == DataType::FP32 && rp[0] == 2.0f && rp[1] == 4.0f);
    }
    {
        auto r = Ops::add(u8t({7}, {1}), i64t({5}, {1}));
        check("u8_i64_mixed_fp32_values", r->dtype == DataType::FP32 && r->data_ptr()[0] == 12.0f);
    }
    check("i64_extreme_add", i64v(Ops::add(i64t({9000000000000000000LL, -9000000000000000000LL}, {2}), i64t({5, -5}, {2}))) == std::vector<int64_t>({9000000000000000005LL, -9000000000000000005LL}));
}

static void t_dtype_u8() {
    check("u8_add_wrap", u8v(Ops::add(u8t({200, 255}, {2}), u8t({100, 1}, {2}))) == std::vector<uint8_t>({44, 0}));
    check("u8_sub_wrap", u8v(Ops::sub(u8t({5, 0}, {2}), u8t({3, 1}, {2}))) == std::vector<uint8_t>({2, 255}));
    check("u8_mul_wrap", u8v(Ops::mul(u8t({16, 200}, {2}), u8t({16, 2}, {2}))) == std::vector<uint8_t>({0, 144}));
    auto e1 = u8t({}, {0});
    check("u8_empty_add", Ops::add(e1, e1)->numel() == 0);
    {
        auto r = Ops::mul(u8t({7}, {1}), i64t({5}, {1}));
        check("u8_i64_mul_values", r->data_ptr()[0] == 35.0f);
    }
}

static void t_dtype_bool() {
    check("bool_add_or", bv(Ops::add(bt({1, 0}, {2}), bt({0, 0}, {2}))) == std::vector<uint8_t>({1, 0}));
    check("bool_mul_and", bv(Ops::mul(bt({1, 0}, {2}), bt({1, 0}, {2}))) == std::vector<uint8_t>({1, 0}));
    auto e1 = bt({}, {0});
    check("bool_empty_add", Ops::add(e1, e1)->numel() == 0);
    {
        auto r = Ops::add(bt({1}, {1}), Tensor::from_vector({2.0f}, {1}, cpu()));
        check("bool_fp32_promotes_fp32", r->dtype == DataType::FP32 && r->data_ptr()[0] == 3.0f);
    }
}

static void t_odd_shapes() {
    auto m = Ops::matmul(Tensor::from_vector({3.0f}, {1, 1}, cpu()), Tensor::from_vector({4.0f}, {1, 1}, cpu()));
    check("shape_1x1x1x1_matmul", m->item() == 12.0f);
    auto a = Tensor::from_vector(std::vector<float>(1000, 1.0f), {1, 1000}, cpu(), true);
    auto y = Ops::sum(a);
    y->backward();
    float* g = a->grad->data_ptr();
    bool ok = true;
    for (int i = 0; i < 1000; ++i) if (g[i] != 1.0f) ok = false;
    check("shape_1x1000_bwd", ok);
    auto b = Tensor::from_vector(std::vector<float>(1000, 2.0f), {1000, 1}, cpu());
    check("shape_1000x1_sum", Ops::sum(b)->item() == 2000.0f);
    auto c = Tensor::from_vector({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, {8}, cpu(), true);
    auto sl = c->slice(0, 0, 8, 2);
    auto ys = Ops::sum(sl);
    ys->backward();
    float* gs = c->grad->data_ptr();
    bool sok = true;
    for (int i = 0; i < 8; ++i) {
        float exp = (i % 2 == 0) ? 1.0f : 0.0f;
        if (gs[i] != exp) sok = false;
    }
    check("shape_strided_slice_bwd", sok);
    auto sc = Tensor::create({}, cpu(), true);
    sc->raw_data_ptr();
    float* sp = (float*)sc->raw_data_ptr();
    sp[0] = 3.0f;
    auto ys2 = Ops::mul(sc, sc);
    ys2->backward();
    check("shape_0dim_scalar_bwd", sc->grad->item() == 6.0f);
}

static void t_nan_inf() {
    float nan = std::numeric_limits<float>::quiet_NaN();
    float inf = std::numeric_limits<float>::infinity();
    auto t = Tensor::from_vector({nan, -inf, 1.0f}, {3}, cpu());
    check("nan_aware_max", std::isnan(Ops::max(t)->item()));
    auto an = Tensor::from_vector({nan, nan, nan, nan}, {2, 2}, cpu());
    check("allnan_max", std::isnan(Ops::max(an)->item()));
    auto sm = Ops::softmax(Tensor::from_vector({1000.0f, 1001.0f, 1002.0f}, {3}, cpu()));
    float* sp = sm->data_ptr();
    float ssum = sp[0] + sp[1] + sp[2];
    check("softmax_extreme_finite", std::isfinite(ssum) && ssum > 0.99f && ssum < 1.01f);
}

static void t_onnx_stress() {
    auto m1 = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{
        std::make_shared<Linear>(4, 8), std::make_shared<ReLU>(), std::make_shared<Linear>(8, 2)});
    auto g = onnx::trace_module(m1, Tensor::zeros({1, 4}, cpu()));
    std::string b1 = onnx::export_onnx_binary(g);
    std::string b2 = onnx::export_onnx_binary(g);
    check("onnx_deterministic", !b1.empty() && b1 == b2);
    auto m2 = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{
        std::make_shared<Linear>(1, 3), std::make_shared<ReLU>(), std::make_shared<Linear>(3, 2)});
    auto g2 = onnx::trace_module(m2, Tensor::zeros({1, 1}, cpu()));
    std::string b3 = onnx::export_onnx_binary(g2);
    check("onnx_weird_1x1_shape", !b3.empty());
}

int main() {
    std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"evict_stress", t_evict_stress},
        {"version_stress", t_version_stress},
        {"dtype_i64", t_dtype_i64},
        {"dtype_u8", t_dtype_u8},
        {"dtype_bool", t_dtype_bool},
        {"odd_shapes", t_odd_shapes},
        {"nan_inf", t_nan_inf},
        {"onnx_stress", t_onnx_stress},
    };
    for (auto& kv : tests) {
        try { kv.second(); }
        catch (const std::exception& e) {
            ++g_fail;
            std::printf("FATAL %s :: %s\n", kv.first.c_str(), e.what());
        }
    }
    std::printf("----\nRESULT qa_fuzz6: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
