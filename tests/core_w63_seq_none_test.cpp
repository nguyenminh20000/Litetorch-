#include "litetorch/nn.h"
#include "litetorch/tensor.h"
#include <cstdio>
#include <exception>
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

static std::shared_ptr<Tensor> fvec(const std::vector<float>& d, const std::vector<int64_t>& s) {
    return Tensor::from_vector(d, s, Device(DeviceType::CPU, 0), false);
}

static std::shared_ptr<Sequential> make_seq() {
    auto seq = std::make_shared<Sequential>();
    seq->add(std::make_shared<Linear>(4, 4));
    seq->add(std::make_shared<Linear>(4, 4));
    seq->add(std::make_shared<Linear>(4, 4));
    return seq;
}

int main() {
    auto x = fvec({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}, {2, 4});
    auto seq = make_seq();
    check("baseline modules size 3", seq->modules.size() == 3);
    check("baseline children size 3", seq->children().size() == 3);
    auto nc = seq->named_children();
    check("baseline named_children names 0,1,2", nc.size() == 3 && nc[0].first == "0" && nc[1].first == "1" && nc[2].first == "2");
    check("baseline parameters 6", seq->parameters().size() == 6);
    bool fwd_ok = false;
    try {
        auto y = seq->forward(x);
        fwd_ok = (y && y->shape.size() == 2 && y->shape[0] == 2 && y->shape[1] == 4);
    } catch (...) {
        fwd_ok = false;
    }
    check("baseline forward ok", fwd_ok);
    seq->modules[1] = nullptr;
    check("null slot kept at index 1", seq->modules[1] == nullptr);
    check("null children size still 3", seq->children().size() == 3);
    check("null parameters skip to 4", seq->parameters().size() == 4);
    bool to_ok = true;
    try {
        seq->to(Device(DeviceType::CPU, 0));
    } catch (...) {
        to_ok = false;
    }
    check("null to() skips null", to_ok);
    bool tr_ok = true;
    try {
        seq->train();
        seq->eval();
    } catch (...) {
        tr_ok = false;
    }
    check("null train/eval skip null", tr_ok);
    bool threw = false;
    try {
        auto y = seq->forward(x);
    } catch (const std::exception&) {
        threw = true;
    }
    check("null forward throws loud", threw);
    auto seq2 = make_seq();
    auto repl = std::make_shared<Linear>(4, 4);
    seq2->add_module("1", repl);
    check("replace keeps size 3", seq2->modules.size() == 3);
    check("replace writes slot 1", seq2->modules[1] == repl);
    printf("core_w63: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
