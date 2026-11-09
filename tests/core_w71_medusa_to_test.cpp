#include "litetorch/nn.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/llm_serving.h"
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
    printf("medusa-head to grad-contract, CPU:0 -> CPU:1\n");

    auto head = std::make_shared<MedusaHead>(4, 4);
    auto pre = head->parameters();
    check("baseline parameters 2", pre.size() == 2);
    check("baseline weight is member", pre.size() == 2 && pre[0].get() == head->weight.get() && pre[1].get() == head->bias.get());
    check("baseline leaves creator null", pre[0]->creator == nullptr && pre[1]->creator == nullptr);
    check("baseline grad null", pre[0]->grad == nullptr && pre[1]->grad == nullptr);
    check("baseline device CPU:0", pre[0]->device == cpu0 && pre[1]->device == cpu0);
    std::vector<float> w_before = pre[0]->to_vector();
    std::vector<float> b_before = pre[1]->to_vector();

    head->to(cpu1);
    auto post = head->parameters();
    check("MedusaHead::to identity preserved", post.size() == 2 && post[0].get() == pre[0].get() && post[1].get() == pre[1].get());
    check("MedusaHead::to parameters are members", post.size() == 2 && post[0].get() == head->weight.get() && post[1].get() == head->bias.get());
    check("MedusaHead::to device moved", post[0]->device == cpu1 && post[1]->device == cpu1);
    check("MedusaHead::to leaves stay leaves", post[0]->creator == nullptr && post[1]->creator == nullptr);
    check("MedusaHead::to values preserved", post[0]->to_vector() == w_before && post[1]->to_vector() == b_before);

    auto x = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, 1);
    auto y = head->forward(x);
    auto loss = Ops::sum(y);
    loss->backward();
    check("grad populated on post-to weight", post[0]->grad != nullptr);
    check("grad populated on post-to bias", post[1]->grad != nullptr);
    bool gw = post[0]->grad && all_close(post[0]->grad->to_vector(), 0.5f, 1e-6f);
    bool gb = post[1]->grad && all_close(post[1]->grad->to_vector(), 1.0f, 1e-6f);
    check("weight grad all 0.5 torch parity", gw);
    check("bias grad all 1.0 torch parity", gb);
    check("grad device moved", post[0]->grad && post[0]->grad->device == cpu1);

    printf("TOTAL %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
