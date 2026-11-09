#include "litetorch/nn.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
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

static std::shared_ptr<Tensor> fvec(const std::vector<float>& d, const std::vector<int64_t>& s, int dev_idx) {
    return Tensor::from_vector(d, s, Device(DeviceType::CPU, dev_idx), false);
}

int main() {
    Device cpu0(DeviceType::CPU, 0);
    Device cpu1(DeviceType::CPU, 1);
    printf("GPT2B pipeline_to stale-optimizer repro, CPU:0 -> CPU:1\n");

    auto model = std::make_shared<GPT2B>(8, 4, 2, 2, 1, 8, 500000.0f);
    auto pre = model->parameters();
    check("baseline tied lm_head is tok_emb weight", model->lm_head->weight.get() == model->tok_emb->weight.get());
    check("baseline shared weight is leaf", model->tok_emb->weight->creator == nullptr);
    check("baseline shared grad null", model->tok_emb->weight->grad == nullptr);
    check("baseline shared device CPU:0", model->tok_emb->weight->device == cpu0);
    std::vector<float> w_before = model->lm_head->weight->to_vector();
    std::vector<float> tok_before = model->tok_emb->weight->to_vector();
    auto stale_leaf = model->tok_emb->weight;

    optim::SGD opt_before(pre, 0.01);

    model->pipeline_to(cpu0, cpu1, -1);
    auto post = model->parameters();
    check("pipeline unties lm_head", model->lm_head->weight.get() != model->tok_emb->weight.get());
    check("lm_head device moved CPU:1", model->lm_head->weight->device == cpu1);
    check("tok_emb device moved CPU:0", model->tok_emb->weight->device == cpu0);
    check("lm_head values preserved", model->lm_head->weight->to_vector() == w_before);
    check("lm_head is leaf after pipeline_to", model->lm_head->weight->creator == nullptr);
    check("tok_emb identity preserved", model->tok_emb->weight.get() == stale_leaf.get());
    check("lm_head in post parameters", post.size() == pre.size() + 1);

    auto x = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, 1);
    auto y = model->lm_head->forward(x);
    auto loss = Ops::sum(y);
    loss->backward();
    check("lm_head grad populated", model->lm_head->weight->grad != nullptr);
    check("lm_head grad all 0.5", model->lm_head->weight->grad && all_close(model->lm_head->weight->grad->to_vector(), 0.5f, 1e-6f));
    check("lm_head grad device moved", model->lm_head->weight->grad && model->lm_head->weight->grad->device == cpu1);

    std::vector<float> lm_before_step = model->lm_head->weight->to_vector();
    optim::SGD opt_after(post, 0.01);
    opt_after.step();
    std::vector<float> lm_after_step = model->lm_head->weight->to_vector();
    bool lm_stepped = true;
    for (size_t i = 0; i < lm_before_step.size() && lm_stepped; ++i) {
        lm_stepped = std::fabs((lm_after_step[i] - lm_before_step[i]) - (-0.005f)) <= 1e-6f;
    }
    check("optimizer-after steps lm_head -0.005", lm_stepped);

    check("tok_emb grad null after lm_head-only forward", model->tok_emb->weight->grad == nullptr);

    std::vector<float> tok_model_before = model->tok_emb->weight->to_vector();
    opt_before.step();
    std::vector<float> tok_model_after = model->tok_emb->weight->to_vector();
    check("optimizer-before leaves tok_emb unchanged", tok_model_after == tok_model_before);

    auto tok_loss = Ops::sum(model->tok_emb->weight);
    tok_loss->backward();
    check("tok_emb grad populated via direct-leaf backward", model->tok_emb->weight->grad != nullptr);
    check("tok_emb grad all 1.0", model->tok_emb->weight->grad && all_close(model->tok_emb->weight->grad->to_vector(), 1.0f, 1e-6f));

    std::vector<float> tok_before2 = model->tok_emb->weight->to_vector();
    opt_before.step();
    std::vector<float> tok_after2 = model->tok_emb->weight->to_vector();
    bool tok_stepped2 = true;
    for (size_t i = 0; i < tok_before2.size() && tok_stepped2; ++i) {
        tok_stepped2 = std::fabs((tok_after2[i] - tok_before2[i]) - (-0.01f)) <= 1e-6f;
    }
    check("optimizer-before steps tok_emb -0.01", tok_stepped2);

    auto model2 = std::make_shared<GPT2B>(8, 4, 2, 2, 1, 8, 500000.0f);
    model2->lm_head->weight->requires_grad = false;
    model2->pipeline_to(cpu0, cpu1, -1);
    check("frozen weight stays frozen after pipeline_to", model2->lm_head->weight->requires_grad == false);
    auto x2 = Tensor::from_vector({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, Device(DeviceType::CPU, 1), true);
    auto y2 = model2->lm_head->forward(x2);
    auto loss2 = Ops::sum(y2);
    loss2->backward();
    check("frozen weight grad stays null", model2->lm_head->weight->grad == nullptr);

    printf("TOTAL %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
