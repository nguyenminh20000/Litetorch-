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

static bool try_backward(std::shared_ptr<Tensor> loss) {
    try {
        loss->backward();
        return true;
    } catch (...) {
        return false;
    }
}

static void set_weight(std::shared_ptr<EmbeddingBag> bag, const std::vector<float>& w) {
    float* p = bag->weight->data_ptr();
    for (size_t i = 0; i < w.size(); ++i) p[i] = w[i];
}

int main() {
    Device cpu0(DeviceType::CPU, 0);
    Device cpu1(DeviceType::CPU, 1);
    printf("embeddingbag autograd, CPU:0\n");

    auto bag = std::make_shared<EmbeddingBag>(5, 4, "sum");
    auto pre = bag->parameters();
    check("embbag parameters 1", pre.size() == 1);
    check("embbag baseline weight is member", pre.size() == 1 && pre[0].get() == bag->weight.get());
    check("embbag baseline leaf", pre[0]->creator == nullptr);
    check("embbag baseline grad null", pre[0]->grad == nullptr);
    check("embbag baseline device CPU:0", pre[0]->device == cpu0);
    std::vector<float> w = pre[0]->to_vector();

    auto idx = fvec({0.0f, 1.0f, 2.0f, 0.0f}, {4}, 0);
    auto out = bag->forward(idx);
    std::vector<float> want(4);
    for (int j = 0; j < 4; ++j) want[(size_t)j] = w[0 * 4 + j] + w[1 * 4 + j] + w[2 * 4 + j] + w[0 * 4 + j];
    check("embbag forward values exact row sums", vec_close(out->to_vector(), want, 1e-6f));
    check("embbag forward output device", out->device == cpu0);
    check("embbag forward output requires grad", out->requires_grad);
    check("embbag forward output has creator", out->creator != nullptr);
    auto loss = Ops::sum(out);
    check("embbag sum backward succeeds", try_backward(loss));
    check("embbag sum grad populated", pre[0]->grad != nullptr);
    std::vector<float> gwant(20, 0.0f);
    gwant[0 * 4 + 0] = 2.0f; gwant[0 * 4 + 1] = 2.0f; gwant[0 * 4 + 2] = 2.0f; gwant[0 * 4 + 3] = 2.0f;
    gwant[1 * 4 + 0] = 1.0f; gwant[1 * 4 + 1] = 1.0f; gwant[1 * 4 + 2] = 1.0f; gwant[1 * 4 + 3] = 1.0f;
    gwant[2 * 4 + 0] = 1.0f; gwant[2 * 4 + 1] = 1.0f; gwant[2 * 4 + 2] = 1.0f; gwant[2 * 4 + 3] = 1.0f;
    check("embbag sum grad values are per-row counts", pre[0]->grad && vec_close(pre[0]->grad->to_vector(), gwant, 1e-6f));
    check("embbag sum grad device", pre[0]->grad && pre[0]->grad->device == cpu0);

    auto bagm = std::make_shared<EmbeddingBag>(5, 4, "mean");
    auto pm = bagm->parameters();
    auto outm = bagm->forward(idx);
    auto lossm = Ops::sum(outm);
    check("embbag mean backward succeeds", try_backward(lossm));
    std::vector<float> gmwant(20, 0.0f);
    for (int j = 0; j < 4; ++j) {
        gmwant[0 * 4 + j] = 0.5f;
        gmwant[1 * 4 + j] = 0.25f;
        gmwant[2 * 4 + j] = 0.25f;
    }
    check("embbag mean grad values counts over bag size", pm[0]->grad && vec_close(pm[0]->grad->to_vector(), gmwant, 1e-6f));

    auto bagx = std::make_shared<EmbeddingBag>(5, 4, "max");
    auto px = bagx->parameters();
    std::vector<float> wx = px[0]->to_vector();
    auto outx = bagx->forward(idx);
    std::vector<float> xwant(4);
    for (int j = 0; j < 4; ++j) {
        float m = wx[0 * 4 + j];
        if (wx[1 * 4 + j] > m) m = wx[1 * 4 + j];
        if (wx[2 * 4 + j] > m) m = wx[2 * 4 + j];
        xwant[(size_t)j] = m;
    }
    check("embbag max forward values", vec_close(outx->to_vector(), xwant, 1e-6f));
    auto lossx = Ops::sum(outx);
    check("embbag max backward succeeds", try_backward(lossx));
    std::vector<float> gxwant(20, 0.0f);
    for (int j = 0; j < 4; ++j) {
        int best = 0;
        if (wx[1 * 4 + j] > wx[best * 4 + j]) best = 1;
        if (wx[2 * 4 + j] > wx[best * 4 + j]) best = 2;
        gxwant[(size_t)(best * 4 + j)] = 1.0f;
    }
    check("embbag max grad only to argmax rows", px[0]->grad && vec_close(px[0]->grad->to_vector(), gxwant, 1e-6f));

    auto bagt = std::make_shared<EmbeddingBag>(5, 4, "max");
    std::vector<float> wt(20, 0.0f);
    wt[1 * 4 + 0] = 2.0f; wt[1 * 4 + 1] = 2.0f; wt[1 * 4 + 2] = 2.0f; wt[1 * 4 + 3] = 2.0f;
    wt[3 * 4 + 0] = 2.0f; wt[3 * 4 + 1] = 2.0f; wt[3 * 4 + 2] = 2.0f; wt[3 * 4 + 3] = 2.0f;
    wt[0 * 4 + 0] = 1.0f; wt[0 * 4 + 1] = 0.0f; wt[0 * 4 + 2] = 0.0f; wt[0 * 4 + 3] = 0.0f;
    set_weight(bagt, wt);
    auto pt = bagt->parameters();
    auto idxt = fvec({1.0f, 3.0f}, {2}, 0);
    auto outt = bagt->forward(idxt);
    check("embbag max tie forward values", vec_close(outt->to_vector(), std::vector<float>({2.0f, 2.0f, 2.0f, 2.0f}), 1e-6f));
    auto losst = Ops::sum(outt);
    check("embbag max tie backward succeeds", try_backward(losst));
    std::vector<float> gtwant(20, 0.0f);
    gtwant[1 * 4 + 0] = 1.0f; gtwant[1 * 4 + 1] = 1.0f; gtwant[1 * 4 + 2] = 1.0f; gtwant[1 * 4 + 3] = 1.0f;
    check("embbag max tie grad first occurrence only", pt[0]->grad && vec_close(pt[0]->grad->to_vector(), gtwant, 1e-6f));

    auto bag2 = std::make_shared<EmbeddingBag>(5, 4, "mean");
    auto p2 = bag2->parameters();
    auto off2 = fvec({0.0f, 2.0f}, {2}, 0);
    auto out2 = bag2->forward(idx, off2);
    std::vector<float> w2 = p2[0]->to_vector();
    std::vector<float> want2(8);
    for (int j = 0; j < 4; ++j) {
        want2[(size_t)j] = (w2[0 * 4 + j] + w2[1 * 4 + j]) / 2.0f;
        want2[4 + (size_t)j] = (w2[2 * 4 + j] + w2[0 * 4 + j]) / 2.0f;
    }
    check("embbag two-bag mean forward values", vec_close(out2->to_vector(), want2, 1e-6f));
    auto loss2 = Ops::sum(out2);
    check("embbag two-bag backward succeeds", try_backward(loss2));
    std::vector<float> g2want(20, 0.0f);
    for (int j = 0; j < 4; ++j) {
        g2want[0 * 4 + j] = 1.0f;
        g2want[1 * 4 + j] = 0.5f;
        g2want[2 * 4 + j] = 0.5f;
    }
    check("embbag two-bag mean grad values", p2[0]->grad && vec_close(p2[0]->grad->to_vector(), g2want, 1e-6f));

    auto bage = std::make_shared<EmbeddingBag>(5, 4, "sum");
    auto pe = bage->parameters();
    auto idxe = fvec({0.0f, 1.0f}, {2}, 0);
    auto offe = fvec({0.0f, 2.0f, 2.0f}, {3}, 0);
    auto oute = bage->forward(idxe, offe);
    std::vector<float> we = pe[0]->to_vector();
    std::vector<float> wante(12);
    for (int j = 0; j < 4; ++j) {
        wante[(size_t)j] = we[0 * 4 + j] + we[1 * 4 + j];
        wante[4 + (size_t)j] = 0.0f;
        wante[8 + (size_t)j] = 0.0f;
    }
    check("embbag empty-bag forward values zero", vec_close(oute->to_vector(), wante, 1e-6f));
    auto losse = Ops::sum(oute);
    check("embbag empty-bag backward succeeds", try_backward(losse));
    std::vector<float> gewant(20, 0.0f);
    for (int j = 0; j < 4; ++j) {
        gewant[0 * 4 + j] = 1.0f;
        gewant[1 * 4 + j] = 1.0f;
    }
    check("embbag empty-bag grad values", pe[0]->grad && vec_close(pe[0]->grad->to_vector(), gewant, 1e-6f));

    auto bag1 = std::make_shared<EmbeddingBag>(5, 4, "sum");
    bag1->to(cpu1);
    auto idx1 = fvec({0.0f, 1.0f, 2.0f, 0.0f}, {4}, 1);
    auto out1 = bag1->forward(idx1);
    check("embbag cpu1 forward output device", out1->device == cpu1);
    auto loss1 = Ops::sum(out1);
    check("embbag cpu1 backward succeeds", try_backward(loss1));
    auto p1 = bag1->parameters();
    check("embbag cpu1 leaf grad populated", p1[0]->grad != nullptr);
    check("embbag cpu1 grad device follows weight", p1[0]->grad && p1[0]->grad->device == cpu1);
    check("embbag cpu1 grad values are per-row counts", p1[0]->grad && vec_close(p1[0]->grad->to_vector(), gwant, 1e-6f));

    printf("TOTAL %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
