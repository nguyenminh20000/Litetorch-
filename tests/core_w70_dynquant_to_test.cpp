#include "litetorch/nn.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/quantization.h"
#include "litetorch/optim.h"
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

int main() {
    Device cpu0(DeviceType::CPU, 0);
    Device cpu1(DeviceType::CPU, 1);
    printf("dynamic-quant-linear to() grad-contract, CPU:0 -> CPU:1\n");

    auto lin = std::make_shared<Linear>(4, 4);
    auto dq = quantization::quantize_dynamic(lin);
    auto pre = dq->parameters();
    check("baseline parameters 1", pre.size() == 1);
    check("baseline bias leaf", pre[0]->creator == nullptr);
    check("baseline bias grad null", pre[0]->grad == nullptr);
    check("baseline bias device CPU:0", pre[0]->device == cpu0);
    std::vector<float> b_before = pre[0]->to_vector();

    auto x0 = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, 0);
    auto out0 = dq->forward(x0);
    std::vector<float> out_before = out0->to_vector();

    dq->to(cpu1);
    auto post = dq->parameters();
    check("bias identity preserved", post.size() == 1 && post[0].get() == pre[0].get());
    check("bias device moved", post[0]->device == cpu1);
    check("bias stays leaf", post[0]->creator == nullptr);
    check("bias values preserved", post[0]->to_vector() == b_before);

    auto x1 = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, 1);
    auto out1 = dq->forward(x1);
    check("forward values unchanged by move", vec_close(out1->to_vector(), out_before, 1e-6f));
    check("forward output device moved", out1->device == cpu1);

    auto loss = Ops::sum(out1);
    loss->backward();
    check("bias grad populated", post[0]->grad != nullptr);
    check("bias grad all 1.0", post[0]->grad && all_close(post[0]->grad->to_vector(), 1.0f, 1e-6f));
    check("bias grad device moved", post[0]->grad && post[0]->grad->device == cpu1);

    auto dq2 = quantization::quantize_dynamic(std::make_shared<Linear>(4, 4));
    optim::SGD opt(dq2->parameters(), 0.01);
    dq2->to(cpu1);
    auto x2 = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, 1);
    auto loss2 = Ops::sum(dq2->forward(x2));
    loss2->backward();
    auto p2 = dq2->parameters();
    std::vector<float> b2_before = p2[0]->to_vector();
    opt.step();
    std::vector<float> b2_after = p2[0]->to_vector();
    bool stepped = b2_after.size() == b2_before.size();
    for (size_t i = 0; i < b2_before.size() && stepped; ++i)
        stepped = std::fabs((b2_after[i] - b2_before[i]) - (-0.01f)) <= 1e-6f;
    check("optimizer-before-to steps bias", stepped);

    auto dq3 = quantization::quantize_dynamic(std::make_shared<Linear>(4, 4));
    dq3->to(cpu1);
    optim::SGD opt3(dq3->parameters(), 0.01);
    auto x3 = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, 1);
    auto loss3 = Ops::sum(dq3->forward(x3));
    loss3->backward();
    auto p3 = dq3->parameters();
    std::vector<float> b3_before = p3[0]->to_vector();
    opt3.step();
    std::vector<float> b3_after = p3[0]->to_vector();
    bool stepped3 = b3_after.size() == b3_before.size();
    for (size_t i = 0; i < b3_before.size() && stepped3; ++i)
        stepped3 = std::fabs((b3_after[i] - b3_before[i]) - (-0.01f)) <= 1e-6f;
    check("optimizer-after-to steps bias", stepped3);

    printf("TOTAL %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
