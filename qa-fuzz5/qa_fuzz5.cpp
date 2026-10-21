#include "litetorch/jit.h"
#include "litetorch/jit_detail.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/inference.h"
#include "litetorch/onnx.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace litetorch;
using namespace litetorch::nn;

static int g_pass = 0, g_fail = 0;
static void check(const std::string& name, bool cond) {
    if (cond) { ++g_pass; std::printf("PASS %s\n", name.c_str()); }
    else { ++g_fail; std::printf("FAIL %s\n", name.c_str()); }
}
static bool threw_with(const std::string& needle, const std::function<void()>& f) {
    try { f(); } catch (const std::exception& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

static void t_fastpath_guards() {
    auto x = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
    auto y = std::make_shared<JITVar>(JITVar::OpType::INPUT, "y");
    auto pure = x - y;
    std::vector<std::shared_ptr<JITVar>> in_xy = {x, y};
    std::vector<std::shared_ptr<JITVar>> in_yx = {y, x};
    check("fastpath pure x-y in-order", jit_detail::fastpath_binary_ok(pure, in_xy));
    check("fastpath y-x arg-order rejected", !jit_detail::fastpath_binary_ok(pure, in_yx));
    auto comp = (x + y) - std::make_shared<JITVar>(JITVar::OpType::INPUT, "c");
    check("fastpath composite (a+b)-c rejected", !jit_detail::fastpath_binary_ok(comp, in_xy));
    auto rx = JIT::relu(x);
    std::vector<std::shared_ptr<JITVar>> in_x = {x};
    check("fastpath unary relu(x)", jit_detail::fastpath_unary_ok(rx, in_x));
    check("fastpath unary relu(x*2) rejected", !jit_detail::fastpath_unary_ok(JIT::relu(x * 2.0f), in_x));
    check("fastpath null expr rejected", !jit_detail::fastpath_binary_ok(nullptr, in_xy));
}

class ResidualBlock : public Module {
public:
    std::shared_ptr<Conv2d> conv;
    std::shared_ptr<ReLU> relu;
    ResidualBlock()
        : conv(std::make_shared<Conv2d>(4, 4, 3, 1, 1)),
          relu(std::make_shared<ReLU>()) {}
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        return Ops::add(relu->forward(conv->forward(input)), input);
    }
    std::vector<std::shared_ptr<Tensor>> parameters() override { return conv->parameters(); }
    void to(const Device& d) override { conv->to(d); relu->to(d); }
    std::vector<std::shared_ptr<Module>> children() override { return {conv, relu}; }
};

static void t_fold_conv_bn() {
    Device cpu(DeviceType::CPU, 0);
    check("fold null conv throws", threw_with("null conv", []() {
        inference::fold_conv_bn(nullptr, std::make_shared<BatchNorm2d>(4));
    }));
    auto conv = std::make_shared<Conv2d>(3, 8, 3, 2, 1);
    auto bn_train = std::make_shared<BatchNorm2d>(8);
    check("fold training-mode BN throws", threw_with("eval mode", [conv, bn_train]() {
        inference::fold_conv_bn(conv, bn_train);
    }));
    auto bn = std::make_shared<BatchNorm2d>(8);
    bn->eval();
    auto folded = inference::fold_conv_bn(conv, bn);
    auto inp = Tensor::from_vector(std::vector<float>(2*3*8*8, 0.5f), {2, 3, 8, 8}, cpu);
    auto ref = bn->forward(conv->forward(inp));
    auto got = folded->forward(inp);
    auto rv = ref->to_vector(); auto gv = got->to_vector();
    bool geo = (ref->shape == got->shape);
    float worst = 0;
    for (size_t i = 0; i < rv.size(); ++i) worst = std::max(worst, std::fabs(rv[i] - gv[i]));
    check("fold geometry stride2/pad1 preserved", geo && ref->shape[2] == 4 && ref->shape[3] == 4);
    check("fold numerics match (worst diff)", worst < 1e-4f);
    std::printf("  worst fold diff: %g\n", worst);
    auto seq = std::make_shared<Sequential>();
    auto sc = std::make_shared<Conv2d>(3, 8, 3, 1, 1);
    auto sb = std::make_shared<BatchNorm2d>(8);
    sb->eval();
    seq->add(sc); seq->add(sc); seq->add(sb);
    check("fuse shared Conv2d refused loudly", threw_with("shared", [seq]() {
        inference::fuse_conv_bn(seq);
    }));
}

static void t_onnx_residual_throw() {
    Device cpu(DeviceType::CPU, 0);
    auto nested = std::make_shared<Sequential>(
        std::vector<std::shared_ptr<Module>>{std::make_shared<ResidualBlock>()});
    auto x = Tensor::zeros({1, 4, 8, 8}, cpu);
    check("onnx trace nested residual throws loudly", threw_with("refusing", [nested, x]() {
        onnx::trace_module(nested, x);
    }));
}

static bool pb_varint(const uint8_t*& p, const uint8_t* end, uint64_t& v) {
    v = 0; int shift = 0;
    while (p < end) {
        uint8_t b = *p++;
        v |= (uint64_t)(b & 0x7f) << shift;
        if (!(b & 0x80)) return true;
        shift += 7;
        if (shift > 63) return false;
    }
    return false;
}

static void t_onnx_binary_structure() {
    Device cpu(DeviceType::CPU, 0);
    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Linear>(4, 4));
    model->add(std::make_shared<ReLU>());
    auto x = Tensor::zeros({1, 4}, cpu);
    auto graph = onnx::trace_module(model, x);
    std::string bin = onnx::export_onnx_binary(graph);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(bin.data());
    const uint8_t* end = p + bin.size();
    bool ok = bin.size() > 4;
    uint64_t v = 0;
    ok = ok && *p++ == 0x08 && pb_varint(p, end, v) && v == 8;
    check("onnx ir_version=8", ok);
    ok = *p++ == 0x12 && pb_varint(p, end, v) && v == 9 &&
         std::string(reinterpret_cast<const char*>(p), 9) == "litetorch";
    check("onnx producer_name=litetorch", ok);
    p += 9;
    bool saw_graph = false, saw_opset = false, opset17 = false;
    while (p < end) {
        uint8_t tag = *p++;
        uint64_t len = 0;
        if ((tag & 0x07) != 2 || !pb_varint(p, end, len)) break;
        uint32_t field = tag >> 3;
        if (field == 7) saw_graph = true;
        if (field == 8) {
            saw_opset = true;
            const uint8_t* q = p; const uint8_t* qend = p + len;
            while (q < qend) {
                uint8_t t2 = *q++;
                if (t2 == 0x10) { uint64_t vv = 0; if (pb_varint(q, qend, vv) && vv == 17) opset17 = true; }
                else break;
            }
        }
        p += len;
    }
    check("onnx graph field present", saw_graph);
    check("onnx opset field present", saw_opset);
    check("onnx opset version=17", opset17);
    bool has_node = false;
    for (const auto& n : graph.nodes) if (n.op_type == "Gemm" || n.op_type == "Relu") has_node = true;
    check("onnx graph has Gemm/Relu nodes", has_node && graph.nodes.size() >= 2);
}

int main() {
    t_fastpath_guards();
    t_fold_conv_bn();
    t_onnx_residual_throw();
    t_onnx_binary_structure();
    std::printf("----\nRESULT qa_fuzz5: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
