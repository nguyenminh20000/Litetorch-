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

static std::shared_ptr<Tensor> mk_f32(const std::vector<float>& v, const std::vector<int64_t>& shape, bool rg = false) {
    auto t = Tensor::create(shape, cpu, rg, DataType::FP32);
    float* p = reinterpret_cast<float*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}

static std::shared_ptr<Tensor> mk_i32(const std::vector<int32_t>& v, const std::vector<int64_t>& shape) {
    auto t = Tensor::create(shape, cpu, false, DataType::INT32);
    int32_t* p = reinterpret_cast<int32_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}

static std::shared_ptr<Tensor> mk_i64(const std::vector<int64_t>& v, const std::vector<int64_t>& shape) {
    auto t = Tensor::create(shape, cpu, false, DataType::INT64);
    int64_t* p = reinterpret_cast<int64_t*>(t->raw_data_ptr());
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}

template <typename F>
static bool throws_runtime(F&& f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

static float scalar(std::shared_ptr<Tensor> t) { return t->to_vector()[0]; }

static void test_d26() {
    auto x = mk_f32({0.5f, 1.5f, 2.0f, 0.2f}, {4});
    auto t = mk_f32({1.0f, 1.0f, -1.0f, -1.0f}, {4});
    nn::HingeEmbeddingLoss m;
    CHECK(near(scalar(m.forward(x, t)), 0.7, 1e-5));
    nn::HingeEmbeddingLoss ms(1.0f, "sum");
    CHECK(near(scalar(ms.forward(x, t)), 2.8, 1e-5));
    nn::HingeEmbeddingLoss mn(1.0f, "none");
    CHECK(near_vec(mn.forward(x, t)->to_vector(), {0.5, 1.5, 0.0, 0.8}, 1e-5));
    nn::HingeEmbeddingLoss m2(2.0f, "mean");
    CHECK(near(scalar(m2.forward(x, t)), 0.95, 1e-5));
    nn::HingeEmbeddingLoss m05(0.5f, "mean");
    CHECK(near(scalar(m05.forward(x, t)), 0.575, 1e-5));
    auto x2 = mk_f32({0.0f, -1.0f, 3.0f}, {3});
    auto t2 = mk_f32({1.0f, -1.0f, -1.0f}, {3});
    CHECK(near(scalar(m.forward(x2, t2)), 0.6666667, 1e-5));
}

static void test_d27() {
    auto logits = mk_f32({0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f}, {2, 4});
    auto tgt = mk_i32({1, 0}, {2});
    nn::MultiMarginLoss m;
    CHECK(near(scalar(m.forward(logits, tgt)), 0.6, 1e-5));
    nn::MultiMarginLoss m2(2.0, 1.0f, "mean");
    CHECK(near(scalar(m2.forward(logits, tgt)), 0.7525, 1e-5));
    nn::MultiMarginLoss ms(1.0, 1.0f, "sum");
    CHECK(near(scalar(ms.forward(logits, tgt)), 1.2, 1e-5));
    nn::MultiMarginLoss mn(1.0, 1.0f, "none");
    CHECK(near_vec(mn.forward(logits, tgt)->to_vector(), {0.9, 0.3}, 1e-5));
    nn::MultiMarginLoss mn2(2.0, 1.0f, "none");
    CHECK(near_vec(mn2.forward(logits, tgt)->to_vector(), {1.145, 0.36}, 1e-5));
    auto z = mk_f32({0.0f, 0.0f, 0.0f, 0.0f}, {1, 4});
    auto zt = mk_i32({2}, {1});
    CHECK(near(scalar(m.forward(z, zt)), 0.75, 1e-5));
    CHECK(near(scalar(m2.forward(z, zt)), 0.75, 1e-5));
    auto logits3 = mk_f32({0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.0f, 0.0f, 0.0f, 0.0f}, {3, 4});
    auto tgt3 = mk_i32({1, 0, 3}, {3});
    CHECK(near(scalar(m.forward(logits3, tgt3)), 0.65, 1e-5));
    nn::MultiMarginLoss mn3(1.0, 1.0f, "none");
    CHECK(near_vec(mn3.forward(logits3, tgt3)->to_vector(), {0.9, 0.3, 0.75}, 1e-5));
    CHECK(near(scalar(m2.forward(logits3, tgt3)), 0.7516667, 1e-5));
    auto single = mk_f32({0.5f, -0.6f, 0.7f, -0.8f}, {1, 4});
    auto single_t = mk_i32({0}, {1});
    CHECK(near(scalar(m.forward(single, single_t)), 0.3, 1e-5));
    CHECK(throws_runtime([&]() { nn::MultiMarginLoss bad(3.0, 1.0f, "mean"); }));
    CHECK(throws_runtime([&]() { nn::MultiMarginLoss bad(1.5, 1.0f, "mean"); }));
    nn::MultiMarginLoss m3;
    m3.p = 3.0;
    CHECK(throws_runtime([&]() { m3.forward(logits, tgt); }));
}

static std::vector<float> ctc_logprobs() {
    const float A[4] = {-1.2722160f, -1.5722159f, -1.0722159f, -1.7722160f};
    const float B[4] = {-1.0391474f, -2.1391473f, -0.8391474f, -2.3391473f};
    std::vector<float> v;
    for (int t = 0; t < 5; ++t) {
        for (int k = 0; k < 4; ++k) v.push_back(A[k]);
        for (int k = 0; k < 4; ++k) v.push_back(B[k]);
    }
    return v;
}

static void test_d28() {
    auto lp = mk_f32(ctc_logprobs(), {5, 2, 4});
    auto tg = mk_i64({1, 2, 3, 1, 2, 3}, {6});
    auto il = mk_i64({5, 5}, {2});
    auto tl = mk_i64({3, 3}, {2});
    nn::CTCLoss m;
    CHECK(near(scalar(m.forward(lp, tg, il, tl)), 1.3600729, 1e-4));
    nn::CTCLoss ms(0, "sum", false);
    CHECK(near(scalar(ms.forward(lp, tg, il, tl)), 8.1604366, 1e-4));
    nn::CTCLoss mn(0, "none", false);
    CHECK(near_vec(mn.forward(lp, tg, il, tl)->to_vector(), {3.7511029, 4.4093337}, 1e-4));
    auto tg2 = mk_i64({1, 2, 3, 1}, {4});
    auto tl2 = mk_i64({3, 1}, {2});
    CHECK(near(scalar(m.forward(lp, tg2, il, tl2)), 2.8182855, 1e-4));
    CHECK(near_vec(mn.forward(lp, tg2, il, tl2)->to_vector(), {3.7511029, 4.3862033}, 1e-4));
    auto tg0 = mk_i64({1, 2, 3}, {3});
    auto tl0 = mk_i64({3, 0}, {2});
    CHECK(near(scalar(m.forward(lp, tg0, il, tl0)), 3.2230523, 1e-4));
    CHECK(near_vec(mn.forward(lp, tg0, il, tl0)->to_vector(), {3.7511029, 5.1957369}, 1e-4));
    auto lp_rg = mk_f32(ctc_logprobs(), {5, 2, 4}, true);
    auto loss = m.forward(lp_rg, tg, il, tl);
    loss->backward();
    std::vector<float> g = lp_rg->grad->to_vector();
    CHECK(near_vec({g[0], g[1], g[2], g[3]}, {0.00251853, -0.08788608, 0.05704155, 0.02832600}, 1e-4));
    CHECK(near_vec({g[4], g[5], g[6], g[7]}, {0.00994753, -0.09802899, 0.07201312, 0.01606830}, 1e-4));
    auto lp_rg2 = mk_f32(ctc_logprobs(), {5, 2, 4}, true);
    auto loss2 = ms.forward(lp_rg2, tg, il, tl);
    loss2->backward();
    std::vector<float> g2 = lp_rg2->grad->to_vector();
    CHECK(near_vec({g2[0], g2[1], g2[2], g2[3]}, {0.01511121, -0.52731645, 0.34224927, 0.16995595}, 1e-4));
}

int main() {
    test_d26();
    test_d27();
    test_d28();
    if (fails == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d CHECKS FAILED\n", fails);
    return 1;
}
