#include "litetorch/jit.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/inference.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>

using namespace litetorch;

static int failures = 0;
static void check(bool ok, const char* name) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++failures;
}

static bool bits_equal(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if (a->shape != b->shape) return false;
    auto va = a->to_vector();
    auto vb = b->to_vector();
    if (va.size() != vb.size()) return false;
    for (size_t i = 0; i < va.size(); ++i) {
        uint32_t x, y;
        std::memcpy(&x, &va[i], 4);
        std::memcpy(&y, &vb[i], 4);
        if (x != y) return false;
    }
    return true;
}

int main() {
    Device cpu(DeviceType::CPU, 0);

    auto x = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
    float c1 = 0.1f;
    float c2 = 1.0f / 3.0f;
    float c3 = 123.456789f;
    float c4 = 1e-7f;
    auto expr = x * c1 + c2 * x - c3 + c4;
    auto fn = std::make_shared<JITFunction>("rt", expr, std::vector<std::shared_ptr<JITVar>>{x});

    const std::string path = "/tmp/cmpw4_rt.txt";
    fn->save(path);
    auto fn2 = JITFunction::load(path);

    auto in = Tensor::from_vector({0.5f, -2.0f, 3.25f, 100.0f}, {4}, cpu, false);
    auto out1 = (*fn)({in});
    auto out2 = (*fn2)({in});
    check(bits_equal(out1, out2), "save/load round-trip bit-exact eval");

    std::ifstream ifs(path);
    std::string line, expr_line;
    for (int i = 0; i < 3 && std::getline(ifs, line); ++i) {}
    std::getline(ifs, expr_line);
    check(expr_line.find("0.100000001") != std::string::npos, "const 0.1f serialized at full precision");
    check(expr_line.find("123.45678") != std::string::npos, "const 123.456789f not truncated to 6 digits");

    auto xs = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, cpu, true);
    auto wgrad = Tensor::from_vector({2.0f}, {1}, cpu, true);
    bool threw = false;
    try {
        std::shared_ptr<Tensor> holder;
        auto tfn = Tracer::trace({xs}, [&](const std::vector<std::shared_ptr<Tensor>>& ins) {
            holder = Ops::mul(ins[0], wgrad);
            return Ops::add(holder, xs);
        }, "grad_scalar");
        (void)tfn;
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("requires_grad") != std::string::npos;
    }
    check(threw, "tracer throws loudly on requires_grad scalar leaf");

    auto wplain = Tensor::from_vector({2.0f}, {1}, cpu, false);
    bool threw_expired = false;
    try {
        auto tfn3 = Tracer::trace({xs}, [&](const std::vector<std::shared_ptr<Tensor>>& ins) {
            auto m = Ops::mul(ins[0], wplain);
            return Ops::add(m, xs);
        }, "expired_intermediate");
        (void)tfn3;
    } catch (const std::runtime_error& e) {
        threw_expired = std::string(e.what()).find("expired") != std::string::npos;
    }
    check(threw_expired, "tracer throws loudly on expired intermediate (no silent corrupt graph)");
    bool ok_bake = false;
    try {
        auto tfn2 = Tracer::trace({xs}, [&](const std::vector<std::shared_ptr<Tensor>>& ins) {
            return Ops::mul(ins[0], wplain);
        }, "plain_scalar");
        auto r = (*tfn2)({xs});
        auto v = r->to_vector();
        ok_bake = (v.size() == 4 && v[0] == 2.0f && v[3] == 8.0f);
    } catch (...) {
        ok_bake = false;
    }
    check(ok_bake, "non-grad scalar leaf still bakes as const and evals");

    auto seq = std::make_shared<nn::Sequential>();
    seq->add(std::make_shared<nn::Linear>(16, 16));
    seq->add(std::make_shared<nn::LayerNorm>(std::vector<int64_t>{16}));
    seq->add(std::make_shared<nn::LayerNorm>(std::vector<int64_t>{16}));
    auto rep_new = litetorch::inference::detect_layernorm_candidates(seq);
    auto rep_old = litetorch::inference::detect_layernorm_fusion(seq);
    check(rep_new.layernorm_detected == 2, "detect_layernorm_candidates counts 2");
    check(rep_old.layernorm_detected == 2, "deprecated alias detect_layernorm_fusion still works");
    check(rep_new.notes == rep_old.notes, "alias returns identical report");

    std::printf(failures == 0 ? "ALL PASS\n" : "FAILURES: %d\n", failures);
    return failures == 0 ? 0 : 1;
}
