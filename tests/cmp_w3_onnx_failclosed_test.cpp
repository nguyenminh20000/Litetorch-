#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/onnx.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace litetorch;
using namespace litetorch::nn;

static int g_pass = 0, g_fail = 0;

static void check(const std::string& name, bool cond) {
    if (cond) { ++g_pass; std::cout << "PASS " << name << "\n"; }
    else { ++g_fail; std::cout << "FAIL " << name << "\n"; }
}

template <typename F>
static void expect_throw(const std::string& name, F fn) {
    bool threw = false;
    std::string what;
    try { fn(); } catch (const std::exception& e) { threw = true; what = e.what(); }
    check(name + "_throws", threw);
    check(name + "_msg_loud", threw && what.find("onnx") != std::string::npos &&
                              what.find("refusing") != std::string::npos);
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

class CustomLeaf : public Module {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        return Ops::mul(input, Ops::sigmoid(input));
    }
};

struct PbCursor {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    bool read_varint(uint64_t& v) {
        v = 0;
        int shift = 0;
        while (p < end && shift < 64) {
            uint8_t b = *p++;
            v |= (uint64_t)(b & 0x7f) << shift;
            if (!(b & 0x80)) return true;
            shift += 7;
        }
        ok = false;
        return false;
    }
    bool read_tag(uint32_t& field, uint32_t& wire) {
        uint64_t t;
        if (!read_varint(t)) return false;
        field = (uint32_t)(t >> 3);
        wire = (uint32_t)(t & 7);
        return true;
    }
    bool read_len(const uint8_t*& d, size_t& n) {
        uint64_t len;
        if (!read_varint(len)) return false;
        if ((uint64_t)(end - p) < len) { ok = false; return false; }
        d = p;
        n = (size_t)len;
        p += n;
        return true;
    }
    bool read_str(std::string& s) {
        const uint8_t* d;
        size_t n;
        if (!read_len(d, n)) return false;
        s.assign((const char*)d, n);
        return true;
    }
    bool skip(uint32_t wire) {
        if (wire == 0) { uint64_t v; return read_varint(v); }
        if (wire == 1) { if (end - p < 8) { ok = false; return false; } p += 8; return true; }
        if (wire == 2) { const uint8_t* d; size_t n; return read_len(d, n); }
        if (wire == 5) { if (end - p < 4) { ok = false; return false; } p += 4; return true; }
        ok = false;
        return false;
    }
};

static uint32_t f32bits(float x) {
    uint32_t b;
    std::memcpy(&b, &x, 4);
    return b;
}

static std::shared_ptr<Tensor> cpu_zeros(const std::vector<int64_t>& shape) {
    return Tensor::zeros(shape, Device(DeviceType::CPU, 0));
}

static void t_linear_traces() {
    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{
        std::make_shared<Linear>(4, 3), std::make_shared<ReLU>()});
    auto g = onnx::trace_module(model, cpu_zeros({2, 4}));
    std::vector<std::string> ops = {"Gemm", "Relu", "Identity"};
    bool ok = g.nodes.size() == ops.size();
    for (size_t i = 0; i < ops.size() && ok; ++i)
        ok = g.nodes[i].op_type == ops[i] && g.nodes[i].domain.empty();
    check("linear_traces_op_types", ok);
    std::string txt = onnx::export_onnx_text(g);
    check("textproto_regression", txt.find("ir_version: 8") != std::string::npos &&
                                  txt.find("op_type: \"Gemm\"") != std::string::npos);
}

static void t_identity_child_traces() {
    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{
        std::make_shared<Linear>(4, 3), std::make_shared<Identity>(), std::make_shared<ReLU>()});
    auto g = onnx::trace_module(model, cpu_zeros({2, 4}));
    bool found = false;
    for (auto& n : g.nodes)
        if (n.op_type == "Identity" && n.domain.empty()) found = true;
    check("identity_child_traces_as_real_op", found);
}

static void t_residual_throws() {
    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{
        std::make_shared<ResidualBlock>()});
    auto x = cpu_zeros({1, 4, 8, 8});
    expect_throw("residual_block", [&]() { onnx::trace_module(model, x); });
}

static void t_nested_sequential_throws() {
    auto inner = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{
        std::make_shared<Linear>(4, 3)});
    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{
        inner, std::make_shared<ReLU>()});
    expect_throw("nested_sequential", [&]() { onnx::trace_module(model, cpu_zeros({2, 4})); });
}

static void t_custom_leaf_throws() {
    auto model = std::make_shared<CustomLeaf>();
    expect_throw("custom_leaf", [&]() { onnx::trace_module(model, cpu_zeros({2, 4})); });
}

static void t_empty_sequential_throws() {
    auto model = std::make_shared<Sequential>();
    expect_throw("empty_sequential", [&]() { onnx::trace_module(model, cpu_zeros({2, 4})); });
}

static void t_branch_depth2_throws() {
    auto inner = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{
        std::make_shared<ResidualBlock>()});
    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{
        std::make_shared<Conv2d>(4, 4, 3, 1, 1), inner});
    expect_throw("branch_depth2", [&]() { onnx::trace_module(model, cpu_zeros({1, 4, 8, 8})); });
}

static void t_binary_framing() {
    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{
        std::make_shared<Linear>(4, 3), std::make_shared<ReLU>()});
    auto g = onnx::trace_module(model, cpu_zeros({2, 4}));
    std::string bin = onnx::export_onnx_binary(g);
    check("binary_nonempty", !bin.empty());
    check("binary_ir_version_bytes",
          bin.size() >= 2 && (uint8_t)bin[0] == 0x08 && (uint8_t)bin[1] == 0x08);
    PbCursor c{(const uint8_t*)bin.data(), (const uint8_t*)bin.data() + bin.size()};
    bool saw_ir = false, saw_producer = false, saw_graph = false, saw_opset = false;
    std::string graph_bytes, opset_bytes;
    while (c.p < c.end && c.ok) {
        uint32_t f, w;
        c.read_tag(f, w);
        if (f == 1 && w == 0) { uint64_t v; c.read_varint(v); saw_ir = (v == 8); }
        else if (f == 2 && w == 2) { std::string s; c.read_str(s); saw_producer = (s == "litetorch"); }
        else if (f == 7 && w == 2) { std::string s; c.read_str(s); graph_bytes = s; saw_graph = true; }
        else if (f == 8 && w == 2) { std::string s; c.read_str(s); opset_bytes = s; saw_opset = true; }
        else c.skip(w);
    }
    check("binary_parse_clean", c.ok && c.p == c.end);
    check("binary_ir_version", saw_ir);
    check("binary_producer", saw_producer);
    check("binary_has_graph", saw_graph);
    check("binary_has_opset", saw_opset);
    PbCursor oc{(const uint8_t*)opset_bytes.data(),
                (const uint8_t*)opset_bytes.data() + opset_bytes.size()};
    bool opset17 = false;
    while (oc.p < oc.end && oc.ok) {
        uint32_t f, w;
        oc.read_tag(f, w);
        if (f == 2 && w == 0) { uint64_t v; oc.read_varint(v); opset17 = (v == 17); }
        else oc.skip(w);
    }
    check("binary_opset17", oc.ok && opset17);
    bool found_gemm_transb = false;
    PbCursor gc{(const uint8_t*)graph_bytes.data(),
                (const uint8_t*)graph_bytes.data() + graph_bytes.size()};
    while (gc.p < gc.end && gc.ok) {
        uint32_t f, w;
        gc.read_tag(f, w);
        if (f == 1 && w == 2) {
            std::string nb;
            gc.read_str(nb);
            PbCursor nc{(const uint8_t*)nb.data(), (const uint8_t*)nb.data() + nb.size()};
            std::string op;
            while (nc.p < nc.end && nc.ok) {
                uint32_t nf, nw;
                nc.read_tag(nf, nw);
                if (nf == 4 && nw == 2) nc.read_str(op);
                else if (nf == 5 && nw == 2) {
                    std::string ab;
                    nc.read_str(ab);
                    PbCursor ac{(const uint8_t*)ab.data(),
                                (const uint8_t*)ab.data() + ab.size()};
                    std::string aname;
                    uint64_t atype = 0, aval = 0;
                    bool aval_ok = false;
                    while (ac.p < ac.end && ac.ok) {
                        uint32_t af, aw;
                        ac.read_tag(af, aw);
                        if (af == 1 && aw == 2) ac.read_str(aname);
                        else if (af == 20 && aw == 0) ac.read_varint(atype);
                        else if (af == 3 && aw == 0) { ac.read_varint(aval); aval_ok = true; }
                        else ac.skip(aw);
                    }
                    if (op == "Gemm" && aname == "transB" && atype == 2 && aval_ok && aval == 1)
                        found_gemm_transb = true;
                } else nc.skip(nw);
            }
        } else gc.skip(w);
    }
    check("binary_graph_parse_clean", gc.ok && gc.p == gc.end);
    check("binary_gemm_transb_attr", found_gemm_transb);
}

static void t_binary_float_attr_exact() {
    auto bn = std::make_shared<BatchNorm2d>(4);
    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{bn});
    auto g = onnx::trace_module(model, cpu_zeros({2, 4, 4, 4}));
    std::string bin = onnx::export_onnx_binary(g);
    PbCursor c{(const uint8_t*)bin.data(), (const uint8_t*)bin.data() + bin.size()};
    std::string graph_bytes;
    while (c.p < c.end && c.ok) {
        uint32_t f, w;
        c.read_tag(f, w);
        if (f == 7 && w == 2) c.read_str(graph_bytes);
        else c.skip(w);
    }
    bool eps_exact = false;
    PbCursor gc{(const uint8_t*)graph_bytes.data(),
                (const uint8_t*)graph_bytes.data() + graph_bytes.size()};
    while (gc.p < gc.end && gc.ok) {
        uint32_t f, w;
        gc.read_tag(f, w);
        if (f == 1 && w == 2) {
            std::string nb;
            gc.read_str(nb);
            PbCursor nc{(const uint8_t*)nb.data(), (const uint8_t*)nb.data() + nb.size()};
            std::string op;
            while (nc.p < nc.end && nc.ok) {
                uint32_t nf, nw;
                nc.read_tag(nf, nw);
                if (nf == 4 && nw == 2) nc.read_str(op);
                else if (nf == 5 && nw == 2) {
                    std::string ab;
                    nc.read_str(ab);
                    PbCursor ac{(const uint8_t*)ab.data(),
                                (const uint8_t*)ab.data() + ab.size()};
                    std::string aname;
                    uint64_t atype = 0;
                    uint32_t bits = 0;
                    bool got = false;
                    while (ac.p < ac.end && ac.ok) {
                        uint32_t af, aw;
                        ac.read_tag(af, aw);
                        if (af == 1 && aw == 2) ac.read_str(aname);
                        else if (af == 20 && aw == 0) ac.read_varint(atype);
                        else if (af == 2 && aw == 5) {
                            if (ac.end - ac.p < 4) { ac.ok = false; break; }
                            std::memcpy(&bits, ac.p, 4);
                            ac.p += 4;
                            got = true;
                        } else ac.skip(aw);
                    }
                    if (op == "BatchNormalization" && aname == "epsilon" && atype == 1 && got &&
                        bits == f32bits(bn->eps))
                        eps_exact = true;
                } else nc.skip(nw);
            }
        } else gc.skip(w);
    }
    check("binary_epsilon_bitexact", gc.ok && eps_exact);
}

static void t_binary_initializer_exact() {
    auto lin = std::make_shared<Linear>(4, 3);
    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{lin});
    auto g = onnx::trace_module(model, cpu_zeros({2, 4}));
    std::string bin = onnx::export_onnx_binary(g);
    PbCursor c{(const uint8_t*)bin.data(), (const uint8_t*)bin.data() + bin.size()};
    std::string graph_bytes;
    while (c.p < c.end && c.ok) {
        uint32_t f, w;
        c.read_tag(f, w);
        if (f == 7 && w == 2) c.read_str(graph_bytes);
        else c.skip(w);
    }
    bool weight_exact = false;
    PbCursor gc{(const uint8_t*)graph_bytes.data(),
                (const uint8_t*)graph_bytes.data() + graph_bytes.size()};
    while (gc.p < gc.end && gc.ok) {
        uint32_t f, w;
        gc.read_tag(f, w);
        if (f == 5 && w == 2) {
            std::string tb;
            gc.read_str(tb);
            PbCursor tc{(const uint8_t*)tb.data(), (const uint8_t*)tb.data() + tb.size()};
            std::string tname;
            std::vector<uint32_t> fbits;
            std::vector<int64_t> dims;
            while (tc.p < tc.end && tc.ok) {
                uint32_t tf, tw;
                tc.read_tag(tf, tw);
                if (tf == 8 && tw == 2) tc.read_str(tname);
                else if (tf == 1 && tw == 0) { uint64_t d; tc.read_varint(d); dims.push_back((int64_t)d); }
                else if (tf == 4 && tw == 2) {
                    const uint8_t* d;
                    size_t n;
                    tc.read_len(d, n);
                    for (size_t i = 0; i + 4 <= n; i += 4) {
                        uint32_t b;
                        std::memcpy(&b, d + i, 4);
                        fbits.push_back(b);
                    }
                } else tc.skip(tw);
            }
            if (tname == "n0_weight" && dims == std::vector<int64_t>({3, 4})) {
                const float* src = lin->weight->data_ptr();
                bool eq = fbits.size() == 12;
                for (size_t i = 0; i < fbits.size() && eq; ++i)
                    eq = fbits[i] == f32bits(src[i]);
                weight_exact = eq;
            }
        } else gc.skip(w);
    }
    check("binary_initializer_bitexact", gc.ok && weight_exact);
}

static void t_binary_dim_param() {
    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{
        std::make_shared<Linear>(4, 3)});
    std::map<std::string, std::map<int64_t, std::string>> dax;
    dax["input"][0] = "batch";
    auto g = onnx::trace_module_dynamic(model, cpu_zeros({2, 4}), dax);
    std::string bin = onnx::export_onnx_binary(g);
    PbCursor c{(const uint8_t*)bin.data(), (const uint8_t*)bin.data() + bin.size()};
    std::string graph_bytes;
    while (c.p < c.end && c.ok) {
        uint32_t f, w;
        c.read_tag(f, w);
        if (f == 7 && w == 2) c.read_str(graph_bytes);
        else c.skip(w);
    }
    bool dim_param_ok = false;
    PbCursor gc{(const uint8_t*)graph_bytes.data(),
                (const uint8_t*)graph_bytes.data() + graph_bytes.size()};
    while (gc.p < gc.end && gc.ok) {
        uint32_t f, w;
        gc.read_tag(f, w);
        if (f == 11 && w == 2) {
            std::string vb;
            gc.read_str(vb);
            PbCursor vc{(const uint8_t*)vb.data(), (const uint8_t*)vb.data() + vb.size()};
            while (vc.p < vc.end && vc.ok) {
                uint32_t vf, vw;
                vc.read_tag(vf, vw);
                if (vf == 2 && vw == 2) {
                    std::string tp;
                    vc.read_str(tp);
                    PbCursor tc{(const uint8_t*)tp.data(),
                                (const uint8_t*)tp.data() + tp.size()};
                    while (tc.p < tc.end && tc.ok) {
                        uint32_t tf, tw;
                        tc.read_tag(tf, tw);
                        if (tf == 1 && tw == 2) {
                            std::string tt;
                            tc.read_str(tt);
                            PbCursor tt2{(const uint8_t*)tt.data(),
                                         (const uint8_t*)tt.data() + tt.size()};
                            while (tt2.p < tt2.end && tt2.ok) {
                                uint32_t sf, sw;
                                tt2.read_tag(sf, sw);
                                if (sf == 2 && sw == 2) {
                                    std::string sh;
                                    tt2.read_str(sh);
                                    PbCursor sc{(const uint8_t*)sh.data(),
                                                (const uint8_t*)sh.data() + sh.size()};
                                    while (sc.p < sc.end && sc.ok) {
                                        uint32_t df, dw;
                                        sc.read_tag(df, dw);
                                        if (df == 1 && dw == 2) {
                                            std::string db;
                                            sc.read_str(db);
                                            PbCursor dc{(const uint8_t*)db.data(),
                                                        (const uint8_t*)db.data() + db.size()};
                                            while (dc.p < dc.end && dc.ok) {
                                                uint32_t ef, ew;
                                                dc.read_tag(ef, ew);
                                                if (ef == 2 && ew == 2) {
                                                    std::string dp;
                                                    dc.read_str(dp);
                                                    if (dp == "batch") dim_param_ok = true;
                                                } else dc.skip(ew);
                                            }
                                        } else sc.skip(dw);
                                    }
                                } else tt2.skip(sw);
                            }
                        } else tc.skip(tw);
                    }
                } else vc.skip(vw);
            }
        } else gc.skip(w);
    }
    check("binary_dim_param", gc.ok && dim_param_ok);
}

static void t_export_onnx_writes_binary() {
    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{
        std::make_shared<Linear>(4, 3)});
    auto g = onnx::trace_module(model, cpu_zeros({2, 4}));
    std::string bin = onnx::export_onnx_binary(g);
    check("binary_deterministic", onnx::export_onnx_binary(g) == bin);
    std::string path = "/tmp/cmp_w3_test.onnx";
    onnx::export_onnx(g, path);
    std::ifstream f(path, std::ios::binary);
    std::string filebin((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    check("export_onnx_writes_binary", filebin == bin);
    check("export_onnx_not_textproto",
          onnx::export_onnx_text(g) != bin && filebin != onnx::export_onnx_text(g));
    std::remove(path.c_str());
}

int main() {
    t_linear_traces();
    t_identity_child_traces();
    t_residual_throws();
    t_nested_sequential_throws();
    t_custom_leaf_throws();
    t_empty_sequential_throws();
    t_branch_depth2_throws();
    t_binary_framing();
    t_binary_float_attr_exact();
    t_binary_initializer_exact();
    t_binary_dim_param();
    t_export_onnx_writes_binary();
    std::cout << "----\nRESULT cmp_w3_onnx_failclosed_test: " << g_pass << " passed, " << g_fail
              << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
