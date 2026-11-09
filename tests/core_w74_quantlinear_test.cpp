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

namespace litetorch {
namespace quantization {
class QuantizedLinear : public nn::Module {
public:
    std::shared_ptr<Tensor> q_weight;
    std::shared_ptr<Tensor> scales;
    std::shared_ptr<Tensor> bias;
    int bits = 8;
    QuantizedLinear(std::shared_ptr<Tensor> qw, std::shared_ptr<Tensor> sc,
                    std::shared_ptr<Tensor> b, int bits_ = 8);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& dev) override;
};
}
}

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
    printf("quantizedlinear-to bias grad-contract, CPU:0 -> CPU:1\n");

    auto lin = std::make_shared<Linear>(2, 4);
    lin->weight = fvec({0.5f, 1.0f, 1.5f, 2.0f, 0.25f, 0.75f, 1.25f, 1.75f}, {2, 4}, 0);
    lin->bias = Tensor::from_vector({0.1f, -0.2f}, {2}, cpu0, true, DataType::FP32);
    auto orig_bias = lin->bias;
    auto seq = std::make_shared<Sequential>();
    seq->add(lin);
    auto prepared = quantization::prepare(seq, quantization::PTQConfig());
    auto converted = quantization::convert(prepared);
    auto children = converted->named_children();
    check("converted child count 1", children.size() == 1);
    auto* ql = static_cast<quantization::QuantizedLinear*>(children[0].second.get());
    auto pre = converted->parameters();
    check("baseline parameters 1", pre.size() == 1);
    check("baseline bias is original", pre.size() == 1 && pre[0].get() == orig_bias.get());
    check("baseline leaf", pre[0]->creator == nullptr);
    check("baseline grad null", pre[0]->grad == nullptr);
    check("baseline device CPU:0", pre[0]->device == cpu0);
    std::vector<float> b_before = pre[0]->to_vector();

    converted->to(cpu1);
    auto post = converted->parameters();
    check("post-to identity pre-post", post.size() == 1 && post[0].get() == pre[0].get());
    check("post-to bias is original bias", post.size() == 1 && post[0].get() == orig_bias.get());
    check("post-to device moved", post[0]->device == cpu1);
    check("post-to leaf", post[0]->creator == nullptr);
    check("post-to values preserved", post[0]->to_vector() == b_before);
    check("post-to record data device moved", ql->q_weight->device == cpu1 && ql->scales->device == cpu1);

    std::vector<float> qv = ql->q_weight->to_vector();
    std::vector<float> sv = ql->scales->to_vector();
    printf("Q %f %f %f %f %f %f %f %f\n", qv[0], qv[1], qv[2], qv[3], qv[4], qv[5], qv[6], qv[7]);
    printf("S %f %f\n", sv[0], sv[1]);
    auto x = fvec({1.0f, 1.0f, 1.0f, 1.0f}, {1, 4}, 1);
    auto y = converted->forward(x);
    std::vector<float> yv = y->to_vector();
    printf("Y %f %f\n", yv[0], yv[1]);
    bool yvalues = yv.size() == 2;
    for (int i = 0; i < 2 && yvalues; ++i) {
        float want = b_before[i];
        for (int j = 0; j < 4; ++j) want += qv[i * 4 + j] * sv[i] * 1.0f;
        yvalues = std::fabs(yv[i] - want) <= 1e-5f;
    }
    check("forward values exact dequant", yvalues);
    check("forward output device", y->device == cpu1);
    auto loss = Ops::sum(y);
    loss->backward();
    if (post[0]->grad) printf("GRAD %f %f\n", post[0]->grad->to_vector()[0], post[0]->grad->to_vector()[1]);
    else printf("GRAD None\n");
    check("grad populated forward route", post[0]->grad != nullptr);
    check("grad all 1.0 forward route", post[0]->grad && all_close(post[0]->grad->to_vector(), 1.0f, 1e-6f));
    check("grad device moved", post[0]->grad && post[0]->grad->device == cpu1);

    auto lin2 = std::make_shared<Linear>(2, 4);
    lin2->weight = fvec({0.5f, 1.0f, 1.5f, 2.0f, 0.25f, 0.75f, 1.25f, 1.75f}, {2, 4}, 0);
    lin2->bias = Tensor::from_vector({0.1f, -0.2f}, {2}, cpu0, true, DataType::FP32);
    auto seq2 = std::make_shared<Sequential>();
    seq2->add(lin2);
    auto converted2 = quantization::convert(quantization::prepare(seq2, quantization::PTQConfig()));
    converted2->to(cpu1);
    auto p2 = converted2->parameters();
    auto loss2 = Ops::sum(p2[0]);
    loss2->backward();
    check("direct-leaf grad populated", p2[0]->grad != nullptr);
    check("direct-leaf grad all 1.0", p2[0]->grad && all_close(p2[0]->grad->to_vector(), 1.0f, 1e-6f));

    auto lin3 = std::make_shared<Linear>(2, 4);
    lin3->weight = fvec({0.5f, 1.0f, 1.5f, 2.0f, 0.25f, 0.75f, 1.25f, 1.75f}, {2, 4}, 0);
    lin3->bias = Tensor::from_vector({0.1f, -0.2f}, {2}, cpu0, true, DataType::FP32);
    auto seq3 = std::make_shared<Sequential>();
    seq3->add(lin3);
    auto converted3 = quantization::convert(quantization::prepare(seq3, quantization::PTQConfig()));
    optim::SGD opt(converted3->parameters(), 0.01);
    converted3->to(cpu1);
    auto x3 = fvec({1.0f, 1.0f, 1.0f, 1.0f}, {1, 4}, 1);
    auto loss3 = Ops::sum(converted3->forward(x3));
    loss3->backward();
    auto p3 = converted3->parameters();
    std::vector<float> w3_before = p3[0]->to_vector();
    opt.step();
    std::vector<float> w3_after = p3[0]->to_vector();
    bool ostep = w3_after.size() == 2;
    for (size_t i = 0; i < 2 && ostep; ++i) ostep = std::fabs((w3_after[i] - w3_before[i]) - (-0.01f)) < 1e-6f;
    check("optimizer-before-to steps bias -0.01", ostep);

    auto seq4 = std::make_shared<Sequential>();
    seq4->add(std::make_shared<Linear>(4, 4));
    auto orig4 = seq4->parameters();
    check("ptq4 pre-convert parameters 2", orig4.size() == 2);
    auto converted4 = quantization::convert(quantization::prepare(seq4, quantization::PTQConfig()));
    auto c4 = converted4->parameters();
    check("ptq4 post-convert parameters 1", c4.size() == 1);
    check("ptq4 converted bias is original bias", c4.size() == 1 && c4[0].get() == orig4[1].get());
    converted4->to(cpu1);
    auto c4post = converted4->parameters();
    check("ptq4 post-to identity pre-post", c4post.size() == 1 && c4post[0].get() == c4[0].get());
    check("ptq4 post-to device moved", c4post[0]->device == cpu1);
    check("ptq4 post-to leaf", c4post[0]->creator == nullptr);
    auto cx4 = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, 1);
    auto closs4 = Ops::sum(converted4->forward(cx4));
    closs4->backward();
    check("ptq4 grad populated", c4post[0]->grad != nullptr);
    check("ptq4 grad all 1.0", c4post[0]->grad && all_close(c4post[0]->grad->to_vector(), 1.0f, 1e-6f));

    auto seq5 = std::make_shared<Sequential>();
    seq5->add(std::make_shared<Linear>(4, 4));
    auto converted5 = quantization::convert(quantization::prepare(seq5, quantization::PTQConfig()));
    optim::SGD opt5(converted5->parameters(), 0.01);
    converted5->to(cpu1);
    auto cx5 = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {1, 4}, 1);
    auto closs5 = Ops::sum(converted5->forward(cx5));
    closs5->backward();
    auto cp5 = converted5->parameters();
    std::vector<float> cb_before = cp5[0]->to_vector();
    opt5.step();
    std::vector<float> cb_after = cp5[0]->to_vector();
    bool cstep = cb_after.size() == 4;
    for (size_t i = 0; i < 4 && cstep; ++i) cstep = std::fabs((cb_after[i] - cb_before[i]) - (-0.01f)) < 1e-6f;
    check("ptq5 optimizer-before-to steps bias -0.01", cstep);

    printf("TOTAL %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
