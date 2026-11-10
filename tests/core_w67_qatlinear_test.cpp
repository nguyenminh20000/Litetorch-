#include "litetorch/nn.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/quantization.h"
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
    printf("qatlinear-to grad-contract, CPU:0 -> CPU:1\n");

    auto lin = std::make_shared<Linear>(4, 4);
    auto qat = std::make_shared<quantization::QATLinear>(lin);
    auto pre = qat->parameters();
    check("baseline parameters 2", pre.size() == 2);
    check("baseline weight is member", pre.size() == 2 && pre[0].get() == qat->weight.get() && pre[1].get() == qat->bias.get());
    check("baseline weight shared with source Linear", pre[0].get() == lin->weight.get() && pre[1].get() == lin->bias.get());
    check("baseline leaves creator null", pre[0]->creator == nullptr && pre[1]->creator == nullptr);
    check("baseline grad null", pre[0]->grad == nullptr && pre[1]->grad == nullptr);
    check("baseline device CPU:0", pre[0]->device == cpu0 && pre[1]->device == cpu0);
    std::vector<float> w_before = pre[0]->to_vector();
    std::vector<float> b_before = pre[1]->to_vector();

    qat->to(cpu1);
    auto post = qat->parameters();
    check("qatlinear::to identity preserved", post.size() == 2 && post[0].get() == pre[0].get() && post[1].get() == pre[1].get());
    check("qatlinear::to device moved", post[0]->device == cpu1 && post[1]->device == cpu1);
    check("qatlinear::to leaves stay leaves", post[0]->creator == nullptr && post[1]->creator == nullptr);
    check("qatlinear::to values preserved", post[0]->to_vector() == w_before && post[1]->to_vector() == b_before);
    check("qatlinear::to source Linear still shares weight", lin->weight.get() == qat->weight.get() && lin->bias.get() == qat->bias.get());
    check("qatlinear::to source Linear device moved", lin->weight->device == cpu1);

    qat->qat_enabled = false;
    auto x = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, 1);
    auto y = qat->forward(x);
    std::vector<float> ywant(4);
    for (int j = 0; j < 4; ++j) {
        float s = b_before[j];
        for (int k = 0; k < 4; ++k) s += 0.5f * w_before[j * 4 + k];
        ywant[j] = s;
    }
    std::vector<float> yv = y->to_vector();
    bool yvalues = yv.size() == 4;
    for (int j = 0; j < 4 && yvalues; ++j) yvalues = std::fabs(yv[j] - ywant[j]) <= 1e-6f;
    check("plain forward values exact", yvalues);
    check("plain forward output device", y->device == cpu1);
    auto loss = Ops::sum(y);
    loss->backward();
    check("plain grad populated on weight", post[0]->grad != nullptr);
    check("plain grad populated on bias", post[1]->grad != nullptr);
    check("plain weight grad all 0.5", post[0]->grad && all_close(post[0]->grad->to_vector(), 0.5f, 1e-6f));
    check("plain bias grad all 1.0", post[1]->grad && all_close(post[1]->grad->to_vector(), 1.0f, 1e-6f));
    check("plain grad device moved", post[0]->grad && post[0]->grad->device == cpu1);

    auto lin2 = std::make_shared<Linear>(4, 4);
    auto qat2 = std::make_shared<quantization::QATLinear>(lin2);
    qat2->to(cpu1);
    auto p2 = qat2->parameters();
    auto x2 = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, 1);
    auto loss2 = Ops::sum(qat2->forward(x2));
    loss2->backward();
    check("qat grad populated on weight", p2[0]->grad != nullptr);
    check("qat grad populated on bias", p2[1]->grad != nullptr);
    check("qat weight grad all 0.5 (STE)", p2[0]->grad && all_close(p2[0]->grad->to_vector(), 0.5f, 1e-6f));
    check("qat bias grad all 1.0", p2[1]->grad && all_close(p2[1]->grad->to_vector(), 1.0f, 1e-6f));

    printf("TOTAL %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
