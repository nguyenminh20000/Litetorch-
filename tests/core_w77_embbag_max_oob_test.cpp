#include "litetorch/nn.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace litetorch;
using namespace litetorch::nn;

static int passed = 0;
static int failed = 0;

static void check(const char* name, bool ok) {
    if (ok) {
        ++passed;
        printf("  PASS %s\n", name);
    } else {
        ++failed;
        printf("  FAIL %s\n", name);
    }
}

static bool vec_close(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

static std::shared_ptr<Tensor> fvec(const std::vector<float>& d, const std::vector<int64_t>& s, int dev_idx) {
    return Tensor::from_vector(d, s, Device(DeviceType::CPU, dev_idx), false);
}

static bool forward_throws(std::shared_ptr<EmbeddingBag> bag, std::shared_ptr<Tensor> idx, std::shared_ptr<Tensor> off) {
    try {
        bag->forward(idx, off);
        return false;
    } catch (...) {
        return true;
    }
}

static void set_weight(std::shared_ptr<EmbeddingBag> bag, const std::vector<float>& w) {
    float* p = bag->weight->data_ptr();
    for (size_t i = 0; i < w.size(); ++i) p[i] = w[i];
}

static std::vector<float> known_weight() {
    return std::vector<float>({1.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 2.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 3.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 4.0f,
                               0.5f, 0.5f, 0.5f, 0.5f});
}

int main() {
    printf("embeddingbag max-branch oob, CPU:0\n");

    auto bagx = std::make_shared<EmbeddingBag>(5, 4, "max");
    set_weight(bagx, known_weight());
    auto idx = fvec({0.0f, 1.0f, 2.0f}, {3}, 0);
    auto off0 = fvec({0.0f}, {1}, 0);
    auto outx = bagx->forward(idx, off0);
    check("max valid forward values exact", vec_close(outx->to_vector(), std::vector<float>({1.0f, 2.0f, 3.0f, 0.0f}), 1e-6f));

    auto off2 = fvec({0.0f, 2.0f}, {2}, 0);
    auto out2 = bagx->forward(idx, off2);
    check("max two-bag forward values", vec_close(out2->to_vector(), std::vector<float>({1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f}), 1e-6f));

    auto idxe = fvec({0.0f, 1.0f}, {2}, 0);
    auto offe = fvec({0.0f, 2.0f, 2.0f}, {3}, 0);
    auto oute = bagx->forward(idxe, offe);
    check("max empty-bag forward zeros", vec_close(oute->to_vector(), std::vector<float>({1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}), 1e-6f));

    auto out1 = bagx->forward(idx);
    check("max forward(input) overload values", vec_close(out1->to_vector(), std::vector<float>({1.0f, 2.0f, 3.0f, 0.0f}), 1e-6f));

    auto bags = std::make_shared<EmbeddingBag>(5, 4, "sum");
    set_weight(bags, known_weight());
    auto outs = bags->forward(idx, off0);
    check("sum valid forward values exact", vec_close(outs->to_vector(), std::vector<float>({1.0f, 2.0f, 3.0f, 0.0f}), 1e-6f));

    auto idxoob = fvec({0.0f, 5.0f}, {2}, 0);
    auto idxoob6 = fvec({0.0f, 6.0f}, {2}, 0);
    auto idxneg = fvec({0.0f, -1.0f}, {2}, 0);
    check("sum oob index 5 throws", forward_throws(bags, idxoob, off0));
    check("sum negative index throws", forward_throws(bags, idxneg, off0));

    auto bagm = std::make_shared<EmbeddingBag>(5, 4, "mean");
    set_weight(bagm, known_weight());
    check("mean oob index 5 throws", forward_throws(bagm, idxoob, off0));
    check("mean negative index throws", forward_throws(bagm, idxneg, off0));

    check("max oob index 5 throws", forward_throws(bagx, idxoob, off0));
    check("max oob index 6 throws", forward_throws(bagx, idxoob6, off0));
    check("max negative index throws", forward_throws(bagx, idxneg, off0));

    printf("TOTAL %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
