#include "litetorch/tensor.h"
#include "litetorch/optim.h"
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace litetorch;
using namespace litetorch::optim;

static int fails = 0;
static int checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::printf("FAIL %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static Device cpu(DeviceType::CPU, 0);

static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

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

static std::shared_ptr<Tensor> mk_param() { return Tensor::zeros({1}, cpu); }

static void check_seq(const std::vector<double>& got, const std::vector<double>& exp, double tol) {
    CHECK(got.size() == exp.size());
    if (got.size() != exp.size()) return;
    for (size_t i = 0; i < exp.size(); ++i) {
        if (!near(got[i], exp[i], tol)) std::printf("MISMATCH %zu: got %.17g exp %.17g tol %.1g\n", i, got[i], exp[i], tol);
        CHECK(near(got[i], exp[i], tol));
    }
}

static std::vector<double> run_onecycle(double max_lr, int total_steps, float pct_start,
                                        const std::string& anneal, int calls,
                                        int epochs = -1, int steps_per_epoch = -1) {
    auto p = mk_param();
    SGD opt({p}, 0.004);
    OneCycleLR sched(&opt, max_lr, total_steps, epochs, steps_per_epoch, pct_start, anneal);
    std::vector<double> out;
    out.push_back(opt.get_lr());
    for (int i = 0; i < calls; ++i) {
        sched.step();
        out.push_back(opt.get_lr());
    }
    return out;
}

int main() {
    std::vector<double> cos10 = {0.004, 0.052, 0.1, 0.09504846320134738, 0.0811745653949763,
                                 0.06112620219362893, 0.03887419780637107, 0.0188258346050237,
                                 0.004951936798652629, 4e-07, 0.004951936798652624};
    check_seq(run_onecycle(0.1, 10, 0.3f, "cos", 10), cos10, 1e-8);

    std::vector<double> lin10 = {0.004, 0.052, 0.1, 0.08571434285714286, 0.07142868571428572,
                                 0.05714302857142858, 0.042857371428571434, 0.028571714285714284,
                                 0.014286057142857148, 3.999999999976245e-07, -0.014285257142857138};
    check_seq(run_onecycle(0.1, 10, 0.3f, "linear", 10), lin10, 1e-8);

    std::vector<double> cos8 = {0.008, 0.056, 0.152, 0.2, 0.17071079527594227, 0.1000004,
                               0.02929000472405773, 8e-07, 0.029290004724057708};
    check_seq(run_onecycle(0.2, 8, 0.5f, "cos", 8), cos8, 1e-8);

    check_seq(run_onecycle(0.1, -1, 0.3f, "cos", 10, 2, 5), cos10, 1e-8);

    {
        auto p = mk_param();
        SGD opt({p}, 0.004);
        OneCycleLR sched(&opt, 0.1, 10, -1, -1, 0.3f, "cos");
        for (int i = 0; i < 10; ++i) sched.step();
        CHECK(near(opt.get_lr(), 0.004951936798652624, 1e-8));
        CHECK(throws_runtime([&] { sched.step(); }));
        CHECK(near(opt.get_lr(), 0.004951936798652624, 1e-8));
    }

    {
        auto p = mk_param();
        SGD opt({p}, 0.1);
        LambdaLR sched(&opt, [](int e) { return std::pow(0.9, e); });
        std::vector<double> got;
        got.push_back(opt.get_lr());
        for (int i = 0; i < 3; ++i) {
            sched.step();
            got.push_back(opt.get_lr());
        }
        check_seq(got, {0.1, 0.09, 0.081, 0.0729}, 1e-9);
    }

    {
        auto p = mk_param();
        SGD opt({p}, 0.1);
        LinearLR sched(&opt, 0.5f, 0.1f, 5);
        std::vector<double> got;
        got.push_back(opt.get_lr());
        for (int i = 0; i < 6; ++i) {
            sched.step();
            got.push_back(opt.get_lr());
        }
        check_seq(got, {0.1, 0.042, 0.034, 0.026, 0.018, 0.01, 0.01}, 1e-8);
    }

    {
        auto p = mk_param();
        SGD opt({p}, 0.1);
        PolynomialLR sched(&opt, 5, 2.0f);
        std::vector<double> got;
        got.push_back(opt.get_lr());
        for (int i = 0; i < 6; ++i) {
            sched.step();
            got.push_back(opt.get_lr());
        }
        check_seq(got, {0.1, 0.064, 0.036, 0.016, 0.004, 0.0, 0.0}, 1e-9);
    }

    {
        auto p = mk_param();
        SGD opt({p}, 0.1);
        CosineAnnealingWarmRestarts sched(&opt, 5, 1, 0.01f);
        std::vector<double> got;
        got.push_back(opt.get_lr());
        for (int i = 0; i < 7; ++i) {
            sched.step();
            got.push_back(opt.get_lr());
        }
        check_seq(got, {0.1, 0.09140576, 0.06890576, 0.04109424, 0.01859424, 0.1, 0.09140576, 0.06890576}, 1e-7);
    }

    if (fails == 0) {
        std::printf("ALL PASS (%d checks)\n", checks);
        return 0;
    }
    std::printf("%d FAILURES (%d checks)\n", fails, checks);
    return 1;
}
