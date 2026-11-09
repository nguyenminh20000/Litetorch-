#include "litetorch/nn.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/quant_scheme.h"
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

static bool all_close(const std::vector<float>& v, float want, float tol) {
    for (float x : v) {
        if (std::fabs(x - want) > tol) return false;
    }
    return true;
}

static std::shared_ptr<Tensor> fvec(const std::vector<float>& d, const std::vector<int64_t>& s, int dev_idx) {
    return Tensor::from_vector(d, s, Device(DeviceType::CPU, dev_idx), false);
}

int main() {
    Device cpu0(DeviceType::CPU, 0);
    Device cpu1(DeviceType::CPU, 1);
    printf("module-to grad-contract, CPU:0 -> CPU:1\n");

    auto lin = std::make_shared<Linear>(4, 4);
    auto pre = lin->parameters();
    check("baseline parameters 2", pre.size() == 2);
    check("baseline leaves creator null", pre[0]->creator == nullptr && pre[1]->creator == nullptr);
    check("baseline grad null", pre[0]->grad == nullptr && pre[1]->grad == nullptr);
    check("baseline device CPU:0", pre[0]->device == cpu0 && pre[1]->device == cpu0);
    std::vector<float> w_before = pre[0]->to_vector();
    std::vector<float> b_before = pre[1]->to_vector();

    auto t2 = pre[0]->to(cpu1);
    check("Tensor::to returns new tensor", t2.get() != pre[0].get());
    check("Tensor::to result non-leaf", t2->creator != nullptr);
    check("Tensor::to values equal", t2->to_vector() == w_before);
    check("Tensor::to leaves source untouched", pre[0]->device == cpu0 && pre[0]->creator == nullptr);

    lin->to(cpu1);
    auto post = lin->parameters();
    check("Linear::to identity preserved", post.size() == 2 && post[0].get() == pre[0].get() && post[1].get() == pre[1].get());
    check("Linear::to device moved", post[0]->device == cpu1 && post[1]->device == cpu1);
    check("Linear::to leaves stay leaves", post[0]->creator == nullptr && post[1]->creator == nullptr);
    check("Linear::to values preserved", post[0]->to_vector() == w_before && post[1]->to_vector() == b_before);

    auto seq = std::make_shared<Sequential>();
    seq->add(std::make_shared<Linear>(4, 4));
    auto spre = seq->parameters();
    seq->to(cpu1);
    auto spost = seq->parameters();
    check("Sequential::to identity preserved", spost.size() == 2 && spost[0].get() == spre[0].get() && spost[1].get() == spre[1].get());
    check("Sequential::to leaves stay leaves", spost[0]->creator == nullptr && spost[1]->creator == nullptr);

    auto x = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, 1);
    auto y = seq->forward(x);
    auto loss = Ops::sum(y);
    loss->backward();
    check("grad populated on post-to weight", spost[0]->grad != nullptr);
    check("grad populated on post-to bias", spost[1]->grad != nullptr);
    bool gw = spost[0]->grad && all_close(spost[0]->grad->to_vector(), 0.5f, 1e-6f);
    bool gb = spost[1]->grad && all_close(spost[1]->grad->to_vector(), 1.0f, 1e-6f);
    check("weight grad all 0.5 torch parity", gw);
    check("bias grad all 1.0 torch parity", gb);
    check("grad device moved", spost[0]->grad && spost[0]->grad->device == cpu1);

    auto cell = std::make_shared<RNNCell>(4, 4);
    auto rpre = cell->parameters();
    check("registry params 4", rpre.size() == 4);
    std::vector<float> rw_before = rpre[0]->to_vector();
    cell->to(cpu1);
    auto rparams = cell->parameters();
    check("registry params 4 after move", rparams.size() == 4);
    bool rident = rparams.size() == 4;
    for (size_t i = 0; i < 4 && rident; ++i) rident = rparams[i].get() == rpre[i].get();
    check("registry identity preserved", rident);
    bool rdev = true;
    bool rleaf = true;
    for (auto& p : rparams) { rdev = rdev && p->device == cpu1; rleaf = rleaf && p->creator == nullptr; }
    check("registry device moved", rdev);
    check("registry leaves stay leaves", rleaf);
    check("registry values preserved", rparams[0]->to_vector() == rw_before);

    auto lin2 = std::make_shared<Linear>(4, 4);
    auto x0 = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, 0);
    auto loss0 = Ops::sum(lin2->forward(x0));
    loss0->backward();
    auto p2 = lin2->parameters();
    check("pre-move grad populated", p2[0]->grad != nullptr && p2[1]->grad != nullptr);
    lin2->to(cpu1);
    auto p2post = lin2->parameters();
    check("existing grad survives move", p2post[0]->grad != nullptr && p2post[1]->grad != nullptr);
    check("existing grad values kept", p2post[0]->grad && all_close(p2post[0]->grad->to_vector(), 0.5f, 1e-6f) && p2post[1]->grad && all_close(p2post[1]->grad->to_vector(), 1.0f, 1e-6f));
    check("existing grad device moved", p2post[0]->grad && p2post[0]->grad->device == cpu1);

    auto bag = std::make_shared<EmbeddingBag>(5, 4, "sum");
    auto epre = bag->parameters();
    check("embbag parameters 1", epre.size() == 1);
    check("embbag baseline weight is member", epre.size() == 1 && epre[0].get() == bag->weight.get());
    check("embbag baseline leaf", epre[0]->creator == nullptr);
    check("embbag baseline grad null", epre[0]->grad == nullptr);
    check("embbag baseline device CPU:0", epre[0]->device == cpu0);
    std::vector<float> ew_before = epre[0]->to_vector();
    bag->to(cpu1);
    auto epost = bag->parameters();
    check("embbag parameters()[0] is member weight", epost.size() == 1 && epost[0].get() == bag->weight.get());
    check("embbag device moved", epost[0]->device == cpu1);
    check("embbag leaf", epost[0]->creator == nullptr);
    check("embbag values preserved", epost[0]->to_vector() == ew_before);
    auto eidx = fvec({0.0f, 1.0f, 2.0f}, {3}, 1);
    auto eout = bag->forward(eidx);
    std::vector<float> ewant(4);
    for (int j = 0; j < 4; ++j) ewant[j] = ew_before[0 * 4 + j] + ew_before[1 * 4 + j] + ew_before[2 * 4 + j];
    std::vector<float> eoutv = eout->to_vector();
    bool evalues = eoutv.size() == 4;
    for (int j = 0; j < 4 && evalues; ++j) evalues = std::fabs(eoutv[j] - ewant[j]) <= 1e-6f;
    check("embbag forward values exact row sums", evalues);
    check("embbag forward output device", eout->device == cpu1);
    auto eloss = Ops::sum(epost[0]);
    eloss->backward();
    check("embbag grad populated", epost[0]->grad != nullptr);
    check("embbag grad all 1.0", epost[0]->grad && all_close(epost[0]->grad->to_vector(), 1.0f, 1e-6f));
    check("embbag grad device moved", epost[0]->grad && epost[0]->grad->device == cpu1);

    auto qlin = std::make_shared<Linear>(4, 4);
    auto qat = std::make_shared<quantization::SchemeQATLinear>(qlin, "fake-quant");
    auto qpre = qat->parameters();
    check("qat parameters 2", qpre.size() == 2);
    check("qat baseline weight is member", qpre.size() == 2 && qpre[0].get() == qat->weight.get());
    check("qat baseline bias is member", qpre.size() == 2 && qpre[1].get() == qat->bias.get());
    check("qat baseline leaves", qpre[0]->creator == nullptr && qpre[1]->creator == nullptr);
    check("qat baseline grad null", qpre[0]->grad == nullptr && qpre[1]->grad == nullptr);
    check("qat baseline device CPU:0", qpre[0]->device == cpu0 && qpre[1]->device == cpu0);
    std::vector<float> qw_before = qpre[0]->to_vector();
    std::vector<float> qb_before = qpre[1]->to_vector();
    qat->to(cpu1);
    auto qpost = qat->parameters();
    check("qat identity preserved", qpost.size() == 2 && qpost[0].get() == qpre[0].get() && qpost[1].get() == qpre[1].get());
    check("qat members are post params", qpost[0].get() == qat->weight.get() && qpost[1].get() == qat->bias.get());
    check("qat device moved", qpost[0]->device == cpu1 && qpost[1]->device == cpu1);
    check("qat leaves stay leaves", qpost[0]->creator == nullptr && qpost[1]->creator == nullptr);
    check("qat values preserved", qpost[0]->to_vector() == qw_before && qpost[1]->to_vector() == qb_before);
    qat->qat_enabled = false;
    auto qx = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, 1);
    auto qy = qat->forward(qx);
    std::vector<float> qwant(4);
    for (int j = 0; j < 4; ++j) qwant[j] = 0.5f * (qw_before[j * 4 + 0] + qw_before[j * 4 + 1] + qw_before[j * 4 + 2] + qw_before[j * 4 + 3]) + qb_before[j];
    std::vector<float> qyv = qy->to_vector();
    bool qvalues = qyv.size() == 4;
    for (int j = 0; j < 4 && qvalues; ++j) qvalues = std::fabs(qyv[j] - qwant[j]) <= 1e-6f;
    check("qat forward values", qvalues);
    auto qloss = Ops::sum(qpost[0]);
    qloss->backward();
    check("qat grad populated on weight", qpost[0]->grad != nullptr);
    check("qat grad all 1.0", qpost[0]->grad && all_close(qpost[0]->grad->to_vector(), 1.0f, 1e-6f));
    check("qat grad device moved", qpost[0]->grad && qpost[0]->grad->device == cpu1);
    auto qlossb = Ops::sum(qpost[1]);
    qlossb->backward();
    check("qat grad populated on bias", qpost[1]->grad != nullptr);
    check("qat grad all 1.0 bias", qpost[1]->grad && all_close(qpost[1]->grad->to_vector(), 1.0f, 1e-6f));
    check("qat grad device moved bias", qpost[1]->grad && qpost[1]->grad->device == cpu1);
    auto qatf = std::make_shared<quantization::SchemeQATLinear>(std::make_shared<Linear>(4, 4), "fake-quant");
    qatf->to(cpu1);
    qatf->qat_enabled = false;
    auto qfx = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, 1);
    auto qfy = qatf->forward(qfx);
    auto qfloss = Ops::sum(qfy);
    qfloss->backward();
    auto qfp = qatf->parameters();
    check("qat forward grad weight all 0.5", qfp[0]->grad && all_close(qfp[0]->grad->to_vector(), 0.5f, 1e-6f));
    check("qat forward grad bias all 1.0", qfp[1]->grad && all_close(qfp[1]->grad->to_vector(), 1.0f, 1e-6f));

    printf("TOTAL %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
