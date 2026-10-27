#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/autograd.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

using namespace litetorch;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static Device cpu(DeviceType::CPU, 0);

static bool near(float a, double b, double tol) { return std::fabs((double)a - b) <= tol; }
static bool near_vec(const std::vector<float>& v, const std::vector<double>& e, double tol) {
    if (v.size() != e.size()) return false;
    for (size_t i = 0; i < v.size(); ++i) if (!near(v[i], e[i], tol)) return false;
    return true;
}

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

static std::shared_ptr<Tensor> mk_f32(const std::vector<float>& v, const std::vector<int64_t>& shape, bool rg = false) {
    auto t = Tensor::create(shape, cpu, rg, DataType::FP32);
    float* p = reinterpret_cast<float*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}

static std::shared_ptr<Tensor> mk_i64(const std::vector<int64_t>& v, const std::vector<int64_t>& shape) {
    auto t = Tensor::create(shape, cpu, false, DataType::INT64);
    int64_t* p = reinterpret_cast<int64_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}

static std::shared_ptr<Tensor> mk_i32(const std::vector<int32_t>& v, const std::vector<int64_t>& shape) {
    auto t = Tensor::create(shape, cpu, false, DataType::INT32);
    int32_t* p = reinterpret_cast<int32_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}

static std::shared_ptr<Tensor> mk_i16(const std::vector<int16_t>& v, const std::vector<int64_t>& shape) {
    auto t = Tensor::create(shape, cpu, false, DataType::INT16);
    int16_t* p = reinterpret_cast<int16_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}

static std::shared_ptr<Tensor> mk_i8(const std::vector<int8_t>& v, const std::vector<int64_t>& shape) {
    auto t = Tensor::create(shape, cpu, false, DataType::INT8);
    int8_t* p = reinterpret_cast<int8_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}

static std::shared_ptr<Tensor> mk_log_softmax(const std::vector<float>& logits) {
    std::vector<float> lp(8);
    for (int r = 0; r < 2; ++r) {
        float mx = logits[r * 4];
        for (int c = 1; c < 4; ++c) mx = std::max(mx, logits[r * 4 + c]);
        float s = 0.0f;
        for (int c = 0; c < 4; ++c) s += std::exp(logits[r * 4 + c] - mx);
        float lse = mx + std::log(s);
        for (int c = 0; c < 4; ++c) lp[r * 4 + c] = logits[r * 4 + c] - lse;
    }
    return mk_f32(lp, {2, 4});
}

int main() {
    std::vector<float> lgv = {2.0f, 0.5f, -1.0f, 0.3f, 0.1f, 1.5f, -0.5f, 2.0f};
    auto logits = mk_f32(lgv, {2, 4});
    auto t64 = mk_i64({0, 2}, {2});
    auto w = mk_f32({1.0f, 0.5f, 2.0f, 1.0f}, {4});

    auto ce_mean = Ops::cross_entropy_loss(logits, t64);
    CHECK(near(ce_mean->to_vector()[0], 1.7420984506607056, 1e-5));
    auto ce_sum = Ops::cross_entropy_loss(logits, t64, nullptr, -100, "sum");
    CHECK(near(ce_sum->to_vector()[0], 3.484196901321411, 1e-5));
    auto ce_none = Ops::cross_entropy_loss(logits, t64, nullptr, -100, "none");
    CHECK(near_vec(ce_none->to_vector(), {0.37541869282722473, 3.108778238296509}, 1e-5));
    auto ce_w = Ops::cross_entropy_loss(logits, t64, w);
    CHECK(near(ce_w->to_vector()[0], 2.1976583003997803, 1e-5));
    auto t_ign = mk_i64({0, -100}, {2});
    auto ce_ign = Ops::cross_entropy_loss(logits, t_ign, nullptr, -100, "mean");
    CHECK(near(ce_ign->to_vector()[0], 0.37541869282722473, 1e-5));
    auto ce_smooth = Ops::cross_entropy_loss(logits, t64, nullptr, -100, "mean", 0.1f);
    CHECK(near(ce_smooth->to_vector()[0], 1.7558484077453613, 1e-5));

    auto lp = mk_log_softmax(lgv);
    auto nll_mean = Ops::nll_loss(lp, t64);
    CHECK(near(nll_mean->to_vector()[0], 1.7420984506607056, 1e-5));
    auto nll_none = Ops::nll_loss(lp, t64, nullptr, -100, "none");
    CHECK(near_vec(nll_none->to_vector(), {0.37541869282722473, 3.108778238296509}, 1e-5));

    auto x = mk_f32({0.5f, -1.0f, 2.0f, 1.0f, 0.25f, -0.5f}, {2, 3});
    auto W = mk_f32({0.3f, -0.7f, 1.1f, 1.0f, 0.2f, -0.4f, -0.5f, 0.9f, 0.6f, 0.8f, -0.2f, 0.3f}, {4, 3});
    auto b = mk_f32({0.1f, -0.2f, 0.3f, 0.0f}, {4});
    auto fused = Ops::fused_linear_cross_entropy(x, W, t64, b);
    CHECK(near(fused->to_vector()[0], 1.1479008197784424, 1e-5));

    nn::CrossEntropyLoss cel;
    auto mod_out = cel.forward(logits, t64);
    CHECK(near(mod_out->to_vector()[0], 1.7420984506607056, 1e-5));

    auto lg = mk_f32(lgv, {2, 4}, true);
    auto ce_loss = Ops::cross_entropy_loss(lg, t64);
    ce_loss->backward();
    CHECK(lg->grad != nullptr);
    CHECK(near_vec(lg->grad->to_vector(),
        {-0.1564992070198059, 0.0766453891992569, 0.017101896926760674, 0.06275193393230438,
         0.04068379104137421, 0.1649809181690216, -0.4776722490787506, 0.272007554769516}, 1e-5));

    auto xg = mk_f32({0.5f, -1.0f, 2.0f, 1.0f, 0.25f, -0.5f}, {2, 3}, true);
    auto Wg = mk_f32({0.3f, -0.7f, 1.1f, 1.0f, 0.2f, -0.4f, -0.5f, 0.9f, 0.6f, 0.8f, -0.2f, 0.3f}, {4, 3}, true);
    auto bg = mk_f32({0.1f, -0.2f, 0.3f, 0.0f}, {4}, true);
    auto fused_loss = Ops::fused_linear_cross_entropy(xg, Wg, t64, bg);
    fused_loss->backward();
    CHECK(xg->grad != nullptr);
    CHECK(near_vec(xg->grad->to_vector(),
        {0.015267033129930496, 0.07660502940416336, -0.07193279266357422,
         0.5869365930557251, -0.41876333951950073, -0.24692371487617493}, 1e-5));
    CHECK(Wg->grad != nullptr);
    CHECK(near_vec(Wg->grad->to_vector(),
        {0.01281658560037613, 0.10628187656402588, -0.21256375312805176,
         0.236226886510849, 0.04928033798933029, -0.09856067597866058,
         -0.42594847083091736, -0.1344246119260788, 0.2688492238521576,
         0.17690499126911163, -0.021137621253728867, 0.04227524250745773}, 1e-5));
    CHECK(bg->grad != nullptr);
    CHECK(near_vec(bg->grad->to_vector(),
        {-0.032995738089084625, 0.24057194590568542, -0.4135318100452423, 0.2059556096792221}, 1e-5));

    auto t32 = mk_i32({0, 2}, {2});
    auto ce_i32 = Ops::cross_entropy_loss(logits, t32);
    CHECK(near(ce_i32->to_vector()[0], 1.7420984506607056, 1e-5));
    auto t16 = mk_i16({0, 2}, {2});
    auto ce_i16 = Ops::cross_entropy_loss(logits, t16);
    CHECK(near(ce_i16->to_vector()[0], 1.7420984506607056, 1e-5));
    auto t8 = mk_i8({0, 2}, {2});
    auto ce_i8 = Ops::cross_entropy_loss(logits, t8);
    CHECK(near(ce_i8->to_vector()[0], 1.7420984506607056, 1e-5));
    auto tf32 = mk_f32({0.0f, 2.0f}, {2});
    auto ce_f32 = Ops::cross_entropy_loss(logits, tf32);
    CHECK(near(ce_f32->to_vector()[0], 1.7420984506607056, 1e-5));
    auto nll_i32 = Ops::nll_loss(lp, t32);
    CHECK(near(nll_i32->to_vector()[0], 1.7420984506607056, 1e-5));
    auto fused_i32 = Ops::fused_linear_cross_entropy(x, W, t32, b);
    CHECK(near(fused_i32->to_vector()[0], 1.1479008197784424, 1e-5));

    auto t_huge = mk_i64({0, 1099511627776LL}, {2});
    CHECK(throws_runtime([&] { Ops::cross_entropy_loss(logits, t_huge); }));
    CHECK(throws_runtime([&] { Ops::nll_loss(lp, t_huge); }));
    CHECK(throws_runtime([&] { Ops::fused_linear_cross_entropy(x, W, t_huge, b); }));
    auto t_oob = mk_i64({0, 4}, {2});
    CHECK(throws_runtime([&] { Ops::cross_entropy_loss(logits, t_oob); }));
    auto t_neg = mk_i64({0, -5}, {2});
    CHECK(throws_runtime([&] { Ops::cross_entropy_loss(logits, t_neg); }));

    if (fails == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d FAILURES\n", fails);
    return 1;
}
