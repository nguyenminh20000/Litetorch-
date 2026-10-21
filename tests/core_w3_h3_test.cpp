#include "litetorch/tensor.h"
#include "litetorch/autograd.h"
#include "litetorch/ops.h"
#include <cstdio>
#include <string>
#include <vector>

using namespace litetorch;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); ++failures; } } while (0)
static bool near(float a, float b, float tol = 1e-4f) { float d = a > b ? a - b : b - a; return d < tol; }

static std::vector<int64_t> read_i64(const std::shared_ptr<Tensor>& t) {
    CHECK(t->dtype == DataType::INT64);
    const int64_t* p = reinterpret_cast<const int64_t*>(t->storage->get_cpu_ptr()) + t->offset;
    return std::vector<int64_t>(p, p + (size_t)t->numel());
}

int main() {
    Device cpu(DeviceType::CPU);

    {
        auto a = Tensor::from_vector({3.0f, 1.0f, 2.0f}, {3}, cpu, false);
        auto r = Ops::topk(a, 2, -1, true, true);
        CHECK(r.first->dtype == DataType::FP32);
        CHECK(r.second->dtype == DataType::INT64);
        auto vals = r.first->to_vector();
        auto idx = read_i64(r.second);
        CHECK(vals.size() == 2 && near(vals[0], 3.0f) && near(vals[1], 2.0f));
        CHECK(idx.size() == 2 && idx[0] == 0 && idx[1] == 2);
    }

    {
        auto a = Tensor::from_vector({3.0f, 1.0f, 2.0f}, {3}, cpu, false);
        auto r = Ops::sort(a, -1, false);
        CHECK(r.second->dtype == DataType::INT64);
        auto vals = r.first->to_vector();
        auto idx = read_i64(r.second);
        CHECK(vals.size() == 3 && near(vals[0], 1.0f) && near(vals[1], 2.0f) && near(vals[2], 3.0f));
        CHECK(idx.size() == 3 && idx[0] == 1 && idx[1] == 2 && idx[2] == 0);
    }

    {
        auto a = Tensor::from_vector({3.0f, 1.0f, 2.0f}, {3}, cpu, false);
        auto idx_t = Ops::argsort(a, -1, false);
        CHECK(idx_t->dtype == DataType::INT64);
        auto idx = read_i64(idx_t);
        CHECK(idx.size() == 3 && idx[0] == 1 && idx[1] == 2 && idx[2] == 0);
    }

    {
        auto a = Tensor::from_vector({3.0f, 1.0f, 2.0f}, {3}, cpu, false);
        auto m = Ops::argmax(a, -1);
        CHECK(m->dtype == DataType::INT64);
        auto idx = read_i64(m);
        CHECK(idx.size() == 1 && idx[0] == 0);
        auto mn = Ops::argmin(a, -1);
        CHECK(mn->dtype == DataType::INT64);
        auto idx2 = read_i64(mn);
        CHECK(idx2.size() == 1 && idx2[0] == 1);
    }

    {
        auto x = Tensor::from_vector({3.0f, 1.0f, 2.0f}, {3}, cpu, true);
        auto r = Ops::sort(x, -1, false);
        auto y = Ops::sum(r.first);
        Autograd::backward(y);
        CHECK(x->grad != nullptr);
        auto g = x->grad->to_vector();
        CHECK(g.size() == 3 && near(g[0], 1.0f) && near(g[1], 1.0f) && near(g[2], 1.0f));
    }

    if (failures == 0) std::printf("core_w3_h3_test: ALL PASS\n");
    return failures;
}
