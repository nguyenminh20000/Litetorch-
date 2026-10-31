#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/optim.h"
#include "litetorch/autograd.h"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace litetorch;

static int fails = 0;
static int checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::printf("FAIL %d: %s\n", __LINE__, #c); ++fails; } } while (0)

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

int main() {
    {
        auto x = mk_f32({1.0f, 2.0f}, {2}, true);
        auto y = Ops::sum(Ops::mul(x, x));
        y->backward();
        CHECK(x->grad != nullptr);
        CHECK(near_vec(x->grad->to_vector(), {2.0, 4.0}, 1e-6));
        auto y2 = Ops::sum(Ops::mul(x, x));
        y2->backward();
        CHECK(near_vec(x->grad->to_vector(), {4.0, 8.0}, 1e-6));
        auto g0 = x->grad;
        void* ptr0 = g0->raw_data_ptr();
        x->zero_grad();
        CHECK(x->grad != nullptr);
        CHECK(x->grad == g0);
        CHECK(x->grad && x->grad->raw_data_ptr() == ptr0);
        CHECK(x->grad && near_vec(x->grad->to_vector(), {0.0, 0.0}, 1e-9));
        auto y3 = Ops::sum(Ops::mul(x, x));
        y3->backward();
        CHECK(near_vec(x->grad->to_vector(), {2.0, 4.0}, 1e-6));
    }
    {
        auto z = mk_f32({5.0f}, {1}, true);
        CHECK(z->grad == nullptr);
        z->zero_grad();
        CHECK(z->grad == nullptr);
    }
    {
        auto w = mk_f32({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);
        auto y = Ops::sum(Ops::mul(w, w));
        y->backward();
        CHECK(near_vec(w->grad->to_vector(), {2.0, 4.0, 6.0, 8.0}, 1e-6));
        optim::SGD sgd({w}, 0.01);
        sgd.zero_grad();
        CHECK(w->grad != nullptr);
        CHECK(w->grad && near_vec(w->grad->to_vector(), {0.0, 0.0, 0.0, 0.0}, 1e-9));
        auto y2 = Ops::sum(Ops::mul(w, w));
        y2->backward();
        CHECK(near_vec(w->grad->to_vector(), {2.0, 4.0, 6.0, 8.0}, 1e-6));
        sgd.zero_grad();
        sgd.zero_grad();
        CHECK(w->grad != nullptr);
        CHECK(w->grad && near_vec(w->grad->to_vector(), {0.0, 0.0, 0.0, 0.0}, 1e-9));
    }
    {
        nn::Linear lin(2, 2);
        auto inp = mk_f32({1.0f, 2.0f}, {1, 2}, false);
        auto out = lin.forward(inp);
        auto loss = Ops::sum(Ops::mul(out, out));
        loss->backward();
        CHECK(lin.weight->grad != nullptr);
        lin.zero_grad();
        CHECK(lin.weight->grad != nullptr);
        CHECK(near_vec(lin.weight->grad->to_vector(), {0.0, 0.0, 0.0, 0.0}, 1e-9));
        CHECK(lin.bias->grad != nullptr);
        CHECK(near_vec(lin.bias->grad->to_vector(), {0.0, 0.0}, 1e-9));
    }
    if (fails == 0) {
        std::printf("ALL PASS (%d checks)\n", checks);
        return 0;
    }
    std::printf("%d FAILURES (%d checks)\n", fails, checks);
    return 1;
}
