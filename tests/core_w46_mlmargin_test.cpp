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

static float scalar(std::shared_ptr<Tensor> t) { return t->to_vector()[0]; }

static void test_mlmargin() {
    auto x = mk_f32({0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f}, {2, 4});
    auto t32 = mk_i32({1, 0, -1, -1, 2, 1, -1, -1}, {2, 4});
    auto t64 = mk_i64({1, 0, -1, -1, 2, 1, -1, -1}, {2, 4});
    nn::MultiLabelMarginLoss mmean("mean");
    nn::MultiLabelMarginLoss msum("sum");
    nn::MultiLabelMarginLoss mnone("none");
    CHECK(near(scalar(mmean.forward(x, t32)), 0.9625, 1e-4));
    CHECK(near(scalar(msum.forward(x, t32)), 1.925, 1e-4));
    auto none32 = mnone.forward(x, t32);
    CHECK(none32->shape == std::vector<int64_t>({2}));
    CHECK(near_vec(none32->to_vector(), {1.0, 0.925}, 1e-4));
    CHECK(near(scalar(mmean.forward(x, t64)), 0.9625, 1e-4));
    CHECK(near_vec(mnone.forward(x, t64)->to_vector(), {1.0, 0.925}, 1e-4));

    auto x1 = mk_f32({0.1f, -0.2f, 0.3f, -0.4f}, {1, 4});
    CHECK(near(scalar(mnone.forward(x1, mk_i32({1, -1, 0, -1}, {1, 4}))), 0.9, 1e-4));
    CHECK(near(scalar(mnone.forward(x1, mk_i32({1, 0, -1, -1}, {1, 4}))), 1.0, 1e-4));
    CHECK(near(scalar(mnone.forward(x1, mk_i32({1, -1, -1, -1}, {1, 4}))), 0.9, 1e-4));
    CHECK(near(scalar(mnone.forward(x1, mk_i32({-1, -1, -1, -1}, {1, 4}))), 0.0, 1e-4));
    CHECK(near(scalar(mnone.forward(x1, mk_i32({0, 1, 2, 3}, {1, 4}))), 0.0, 1e-4));
    CHECK(near(scalar(mnone.forward(x1, mk_i32({2, -1, -1, -1}, {1, 4}))), 0.4, 1e-4));
    CHECK(near(scalar(mnone.forward(x1, mk_i32({1, 1, -1, -1}, {1, 4}))), 1.8, 1e-4));
    CHECK(near(scalar(mnone.forward(x1, mk_i32({1, 1, 1, -1}, {1, 4}))), 2.7, 1e-4));

    auto xc3 = mk_f32({0.2f, -0.1f, 0.5f, 0.3f, 0.4f, -0.2f}, {2, 3});
    auto tc3 = mk_i32({1, -1, -1, 0, 2, -1}, {2, 3});
    CHECK(near(scalar(mmean.forward(xc3, tc3)), 0.9333333, 1e-4));
    CHECK(near(scalar(msum.forward(xc3, tc3)), 1.8666667, 1e-4));
    CHECK(near_vec(mnone.forward(xc3, tc3)->to_vector(), {0.9666667, 0.9}, 1e-4));

    auto xg1 = mk_f32({0.1f, -0.2f, 0.3f, -0.4f}, {1, 4}, true);
    auto loss1 = mmean.forward(xg1, mk_i32({1, 0, -1, -1}, {1, 4}));
    loss1->backward();
    CHECK(near_vec(xg1->grad->to_vector(), {-0.5, -0.5, 0.5, 0.5}, 1e-4));
    auto xg = mk_f32({0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f}, {2, 4}, true);
    auto lossm = mmean.forward(xg, t32);
    lossm->backward();
    CHECK(near_vec(xg->grad->to_vector(), {-0.25, -0.25, 0.25, 0.25, 0.25, -0.25, -0.125, 0.125}, 1e-4));
    auto xg2 = mk_f32({0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f}, {2, 4}, true);
    auto losss = msum.forward(xg2, t32);
    losss->backward();
    CHECK(near_vec(xg2->grad->to_vector(), {-0.5, -0.5, 0.5, 0.5, 0.5, -0.5, -0.25, 0.25}, 1e-4));
}

int main() {
    test_mlmargin();
    if (fails == 0) std::printf("ALL PASS\n");
    else std::printf("%d FAILURES\n", fails);
    return fails == 0 ? 0 : 1;
}
