#include "litetorch/onnx.h"
#include "litetorch/ops.h"
#include <cstring>
#include <cxxabi.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

namespace litetorch {
namespace onnx {

void Graph::add_input(const std::string& name, const std::vector<int64_t>& shape, DataType dtype) {
    inputs.push_back({name, shape, dtype, {}});
}

void Graph::add_output(const std::string& name, const std::vector<int64_t>& shape, DataType dtype) {
    outputs.push_back({name, shape, dtype, {}});
}

void Graph::add_dynamic_input(const std::string& name, const std::vector<int64_t>& shape,
                              const std::vector<std::string>& dim_params, DataType dtype) {
    ValueInfo v;
    v.name = name;
    v.shape = shape;
    v.dtype = dtype;
    v.dim_params = dim_params;
    if (v.dim_params.size() < v.shape.size()) v.dim_params.resize(v.shape.size());
    inputs.push_back(std::move(v));
}

void Graph::add_dynamic_output(const std::string& name, const std::vector<int64_t>& shape,
                               const std::vector<std::string>& dim_params, DataType dtype) {
    ValueInfo v;
    v.name = name;
    v.shape = shape;
    v.dtype = dtype;
    v.dim_params = dim_params;
    if (v.dim_params.size() < v.shape.size()) v.dim_params.resize(v.shape.size());
    outputs.push_back(std::move(v));
}

void Graph::add_node(const std::string& op_type, const std::vector<std::string>& inputs,
                     const std::vector<std::string>& outputs,
                     const std::map<std::string, std::string>& attrs, const std::string& domain) {
    Node n;
    n.op_type = op_type;
    n.name = op_type + "_" + std::to_string(nodes.size());
    n.inputs = inputs;
    n.outputs = outputs;
    n.attrs = attrs;
    n.domain = domain;
    nodes.push_back(std::move(n));
}

void Graph::add_node_typed(const std::string& op_type, const std::vector<std::string>& inputs,
                           const std::vector<std::string>& outputs,
                           const std::map<std::string, int64_t>& int_attrs,
                           const std::map<std::string, float>& float_attrs,
                           const std::map<std::string, std::vector<int64_t>>& ints_attrs,
                           const std::map<std::string, std::string>& attrs,
                           const std::string& domain) {
    Node n;
    n.op_type = op_type;
    n.name = op_type + "_" + std::to_string(nodes.size());
    n.inputs = inputs;
    n.outputs = outputs;
    n.attrs = attrs;
    n.int_attrs = int_attrs;
    n.float_attrs = float_attrs;
    n.ints_attrs = ints_attrs;
    n.domain = domain;
    nodes.push_back(std::move(n));
}

void Graph::add_initializer(const std::string& name, std::shared_ptr<Tensor> tensor) {
    if (!tensor) throw std::runtime_error("[litetorch Error] onnx: null initializer tensor");
    initializers[name] = tensor;
}

int onnx_elem_type(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return 1;
        case DataType::FP64: return 11;
        case DataType::FP16: return 10;
        case DataType::BF16: return 16;
        case DataType::INT32: return 6;
        case DataType::INT16: return 5;
        case DataType::INT8: return 3;
        default: return 1;
    }
}

namespace {

std::string text_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c >= 32 && c < 127) out += c;
        else { char b[8]; snprintf(b, sizeof(b), "\\%03o", (unsigned char)c); out += b; }
    }
    return out;
}

void write_value_info(std::ostringstream& os, const ValueInfo& v, const char* kind) {
    os << "  " << kind << " {\n";
    os << "    name: \"" << text_escape(v.name) << "\"\n";
    os << "    type {\n";
    os << "      tensor_type {\n";
    os << "        elem_type: " << onnx_elem_type(v.dtype) << "\n";
    os << "        shape {\n";
    for (size_t i = 0; i < v.shape.size(); ++i) {
        if (i < v.dim_params.size() && !v.dim_params[i].empty())
            os << "          dim { dim_param: \"" << text_escape(v.dim_params[i]) << "\" }\n";
        else
            os << "          dim { dim_value: " << v.shape[i] << " }\n";
    }
    os << "        }\n";
    os << "      }\n";
    os << "    }\n";
    os << "  }\n";
}

bool tensor_to_floats(const Tensor& t, std::vector<float>& out) {
    if (t.dtype != DataType::FP32 || t.device.type != DeviceType::CPU) return false;
    if (!t.storage) return false;
    float* p = t.storage->get_cpu_ptr();
    if (!p) return false;
    size_t n = t.numel();
    out.assign(p, p + n);
    return true;
}

void write_attr(std::ostringstream& os, const std::string& k, const std::string& v) {
    os << "    attribute { name: \"" << text_escape(k)
       << "\" s: \"" << text_escape(v) << "\" type: STRING }\n";
}

void write_attr(std::ostringstream& os, const std::string& k, int64_t v) {
    os << "    attribute { name: \"" << text_escape(k) << "\" i: " << v << " type: INT }\n";
}

void write_attr(std::ostringstream& os, const std::string& k, float v) {
    os << "    attribute { name: \"" << text_escape(k) << "\" f: " << v << " type: FLOAT }\n";
}

void write_attr(std::ostringstream& os, const std::string& k, const std::vector<int64_t>& v) {
    os << "    attribute { name: \"" << text_escape(k) << "\" ints: [";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ", ";
        os << v[i];
    }
    os << "] type: INTS }\n";
}

}

std::string export_onnx_text(const Graph& graph) {
    std::ostringstream os;
    os << "ir_version: 8\n";
    os << "producer_name: \"litetorch\"\n";
    os << "graph {\n";
    os << "  name: \"" << text_escape(graph.name) << "\"\n";
    for (const auto& n : graph.nodes) {
        os << "  node {\n";
        for (const auto& i : n.inputs) os << "    input: \"" << text_escape(i) << "\"\n";
        for (const auto& o : n.outputs) os << "    output: \"" << text_escape(o) << "\"\n";
        os << "    name: \"" << text_escape(n.name) << "\"\n";
        os << "    op_type: \"" << text_escape(n.op_type) << "\"\n";
        if (!n.domain.empty()) os << "    domain: \"" << text_escape(n.domain) << "\"\n";
        for (const auto& kv : n.attrs) write_attr(os, kv.first, kv.second);
        for (const auto& kv : n.int_attrs) write_attr(os, kv.first, kv.second);
        for (const auto& kv : n.float_attrs) write_attr(os, kv.first, kv.second);
        for (const auto& kv : n.ints_attrs) write_attr(os, kv.first, kv.second);
        os << "  }\n";
    }
    for (const auto& v : graph.inputs) write_value_info(os, v, "input");
    for (const auto& v : graph.outputs) write_value_info(os, v, "output");
    for (const auto& kv : graph.initializers) {
        const Tensor& t = *kv.second;
        os << "  initializer {\n";
        os << "    name: \"" << text_escape(kv.first) << "\"\n";
        os << "    data_type: " << onnx_elem_type(t.dtype) << "\n";
        for (int64_t d : t.shape) os << "    dims: " << d << "\n";
        std::vector<float> vals;
        if (tensor_to_floats(t, vals)) {
            os << "    float_data: [";
            for (size_t i = 0; i < vals.size(); ++i) {
                if (i) os << ", ";
                os << vals[i];
            }
            os << "]\n";
        }
        os << "  }\n";
    }
    os << "}\n";
    os << "opset_import { version: 17 }\n";
    return os.str();
}

void export_onnx(const Graph& graph, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("[litetorch Error] onnx: cannot write " + path);
    std::string bin = export_onnx_binary(graph);
    f.write(bin.data(), static_cast<std::streamsize>(bin.size()));
    if (!f) throw std::runtime_error("[litetorch Error] onnx: write failed for " + path);
}

namespace {

void pb_varint(std::string& out, uint64_t v) {
    while (v >= 0x80) {
        out.push_back(static_cast<char>(0x80 | (v & 0x7f)));
        v >>= 7;
    }
    out.push_back(static_cast<char>(v));
}

void pb_tag(std::string& out, uint32_t field, uint32_t wire) {
    pb_varint(out, (static_cast<uint64_t>(field) << 3) | wire);
}

void pb_varint_field(std::string& out, uint32_t field, uint64_t v) {
    pb_tag(out, field, 0);
    pb_varint(out, v);
}

void pb_fixed32_field(std::string& out, uint32_t field, float v) {
    pb_tag(out, field, 5);
    uint32_t b;
    std::memcpy(&b, &v, 4);
    out.push_back(static_cast<char>(b & 0xff));
    out.push_back(static_cast<char>((b >> 8) & 0xff));
    out.push_back(static_cast<char>((b >> 16) & 0xff));
    out.push_back(static_cast<char>((b >> 24) & 0xff));
}

void pb_bytes_field(std::string& out, uint32_t field, const std::string& s) {
    pb_tag(out, field, 2);
    pb_varint(out, s.size());
    out.append(s);
}

void pb_msg_field(std::string& out, uint32_t field, const std::string& msg) {
    pb_bytes_field(out, field, msg);
}

std::string pb_dim(const ValueInfo& v, size_t i) {
    std::string m;
    if (i < v.dim_params.size() && !v.dim_params[i].empty())
        pb_bytes_field(m, 2, v.dim_params[i]);
    else
        pb_varint_field(m, 1, static_cast<uint64_t>(v.shape[i]));
    return m;
}

std::string pb_value_info(const ValueInfo& v) {
    std::string tt;
    pb_varint_field(tt, 1, static_cast<uint64_t>(onnx_elem_type(v.dtype)));
    std::string shape;
    for (size_t i = 0; i < v.shape.size(); ++i) pb_msg_field(shape, 1, pb_dim(v, i));
    pb_msg_field(tt, 2, shape);
    std::string tp;
    pb_msg_field(tp, 1, tt);
    std::string m;
    pb_bytes_field(m, 1, v.name);
    pb_msg_field(m, 2, tp);
    return m;
}

std::string pb_attr_string(const std::string& k, const std::string& val) {
    std::string m;
    pb_bytes_field(m, 1, k);
    pb_bytes_field(m, 4, val);
    pb_varint_field(m, 20, 3);
    return m;
}

std::string pb_attr_int(const std::string& k, int64_t val) {
    std::string m;
    pb_bytes_field(m, 1, k);
    pb_varint_field(m, 3, static_cast<uint64_t>(val));
    pb_varint_field(m, 20, 2);
    return m;
}

std::string pb_attr_float(const std::string& k, float val) {
    std::string m;
    pb_bytes_field(m, 1, k);
    pb_fixed32_field(m, 2, val);
    pb_varint_field(m, 20, 1);
    return m;
}

std::string pb_attr_ints(const std::string& k, const std::vector<int64_t>& vals) {
    std::string m;
    pb_bytes_field(m, 1, k);
    std::string packed;
    for (int64_t x : vals) pb_varint(packed, static_cast<uint64_t>(x));
    pb_bytes_field(m, 8, packed);
    pb_varint_field(m, 20, 7);
    return m;
}

std::string pb_tensor_proto(const std::string& name, const Tensor& t) {
    std::string m;
    for (int64_t d : t.shape) pb_varint_field(m, 1, static_cast<uint64_t>(d));
    pb_varint_field(m, 2, static_cast<uint64_t>(onnx_elem_type(t.dtype)));
    std::vector<float> vals;
    if (tensor_to_floats(t, vals)) {
        std::string packed;
        packed.reserve(vals.size() * 4);
        for (float x : vals) {
            uint32_t b;
            std::memcpy(&b, &x, 4);
            packed.push_back(static_cast<char>(b & 0xff));
            packed.push_back(static_cast<char>((b >> 8) & 0xff));
            packed.push_back(static_cast<char>((b >> 16) & 0xff));
            packed.push_back(static_cast<char>((b >> 24) & 0xff));
        }
        pb_bytes_field(m, 4, packed);
    }
    pb_bytes_field(m, 8, name);
    return m;
}

std::string pb_node(const Node& n) {
    std::string m;
    for (const auto& i : n.inputs) pb_bytes_field(m, 1, i);
    for (const auto& o : n.outputs) pb_bytes_field(m, 2, o);
    pb_bytes_field(m, 3, n.name);
    pb_bytes_field(m, 4, n.op_type);
    for (const auto& kv : n.attrs) pb_msg_field(m, 5, pb_attr_string(kv.first, kv.second));
    for (const auto& kv : n.int_attrs) pb_msg_field(m, 5, pb_attr_int(kv.first, kv.second));
    for (const auto& kv : n.float_attrs) pb_msg_field(m, 5, pb_attr_float(kv.first, kv.second));
    for (const auto& kv : n.ints_attrs) pb_msg_field(m, 5, pb_attr_ints(kv.first, kv.second));
    if (!n.domain.empty()) pb_bytes_field(m, 7, n.domain);
    return m;
}

}

std::string export_onnx_binary(const Graph& graph) {
    std::string g;
    for (const auto& n : graph.nodes) pb_msg_field(g, 1, pb_node(n));
    pb_bytes_field(g, 2, graph.name);
    for (const auto& kv : graph.initializers) pb_msg_field(g, 5, pb_tensor_proto(kv.first, *kv.second));
    for (const auto& v : graph.inputs) pb_msg_field(g, 11, pb_value_info(v));
    for (const auto& v : graph.outputs) pb_msg_field(g, 12, pb_value_info(v));
    std::string opset;
    pb_varint_field(opset, 2, 17);
    std::string m;
    pb_varint_field(m, 1, 8);
    pb_bytes_field(m, 2, "litetorch");
    pb_msg_field(m, 7, g);
    pb_msg_field(m, 8, opset);
    return m;
}

namespace {

std::string demangle(const char* name) {
    int status = 0;
    char* d = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    std::string out = (status == 0 && d) ? d : name;
    free(d);
    return out;
}

std::string module_class_name(const nn::Module& m) {
    std::string full = demangle(typeid(m).name());
    size_t p = full.rfind("::");
    std::string base = (p == std::string::npos) ? full : full.substr(p + 2);
    size_t t = base.find('<');
    if (t != std::string::npos) base = base.substr(0, t);
    return base;
}

std::pair<std::string, std::string> map_op(const std::string& cls) {
    static const std::map<std::string, std::string> table = {
        {"Linear", "Gemm"}, {"ColumnParallelLinear", "Gemm"}, {"RowParallelLinear", "Gemm"},
        {"Bilinear", "Gemm"},
        {"ReLU", "Relu"}, {"ReLU6", "Relu"}, {"Sigmoid", "Sigmoid"}, {"Tanh", "Tanh"},
        {"GELU", "Gelu"}, {"SiLU", "Silu"}, {"Softmax", "Softmax"}, {"LogSoftmax", "LogSoftmax"},
        {"LeakyReLU", "LeakyRelu"}, {"RReLU", "RRelu"}, {"ELU", "Elu"}, {"SELU", "Selu"},
        {"CELU", "Celu"}, {"Mish", "Mish"}, {"Hardswish", "HardSwish"},
        {"Hardsigmoid", "HardSigmoid"}, {"Softplus", "Softplus"}, {"Softsign", "Softsign"},
        {"Softshrink", "Softshrink"}, {"Hardshrink", "Hardshrink"}, {"Tanhshrink", "Tanhshrink"},
        {"Softmin", "Softmin"}, {"PReLU", "PRelu"}, {"Threshold", "ThresholdedRelu"},
        {"Hardtanh", "Clip"},
        {"Conv1d", "Conv"}, {"Conv2d", "Conv"}, {"Conv3d", "Conv"},
        {"ConvTranspose1d", "ConvTranspose"}, {"ConvTranspose2d", "ConvTranspose"},
        {"ConvTranspose3d", "ConvTranspose"},
        {"BatchNorm1d", "BatchNormalization"}, {"BatchNorm2d", "BatchNormalization"},
        {"BatchNorm3d", "BatchNormalization"}, {"SyncBatchNorm", "BatchNormalization"},
        {"InstanceNorm1d", "InstanceNormalization"}, {"InstanceNorm2d", "InstanceNormalization"},
        {"InstanceNorm3d", "InstanceNormalization"},
        {"LayerNorm", "LayerNormalization"},
        {"Dropout", "Dropout"}, {"Embedding", "Gather"}, {"EmbeddingBag", "Gather"},
        {"Flatten", "Flatten"}, {"Unflatten", "Unflatten"},
        {"MaxPool2d", "MaxPool"}, {"MaxPool3d", "MaxPool"},
        {"AvgPool2d", "AveragePool"},
        {"AdaptiveAvgPool2d", "GlobalAveragePool"},
        {"PixelShuffle", "DepthToSpace"}, {"PixelUnshuffle", "SpaceToDepth"},
        {"Fold", "Fold"}, {"Unfold", "Unfold"},
        {"RNN", "RNN"}, {"LSTM", "LSTM"}, {"GRU", "GRU"},
        {"Identity", "Identity"},
    };
    auto it = table.find(cls);
    if (it != table.end()) return {it->second, ""};
    return {cls, "litetorch"};
}

struct TypedAttrs {
    std::map<std::string, int64_t> i;
    std::map<std::string, float> f;
    std::map<std::string, std::vector<int64_t>> is;
};

std::vector<int64_t> symmetric_pads(const std::vector<int>& padding) {
    std::vector<int64_t> pads;
    for (int p : padding) pads.push_back(p);
    for (int p : padding) pads.push_back(p);
    return pads;
}

std::vector<int64_t> to_i64(const std::vector<int>& v) {
    std::vector<int64_t> out;
    for (int x : v) out.push_back(x);
    return out;
}

TypedAttrs extract_attrs(const nn::Module& m) {
    TypedAttrs a;
    if (auto* l = dynamic_cast<const nn::Linear*>(&m)) {
        (void)l;
        a.i["transB"] = 1;
        a.f["alpha"] = 1.0f;
        a.f["beta"] = 1.0f;
    } else if (auto* c = dynamic_cast<const nn::Conv2d*>(&m)) {
        a.is["kernel_shape"] = to_i64(c->kernel_size);
        a.is["strides"] = to_i64(c->stride_vec);
        a.is["pads"] = symmetric_pads(c->padding_vec);
        a.is["dilations"] = std::vector<int64_t>(c->kernel_size.size(), 1);
        a.i["group"] = c->groups;
    } else if (auto* c = dynamic_cast<const nn::Conv1d*>(&m)) {
        a.is["kernel_shape"] = {(int64_t)c->kernel_size};
        a.is["strides"] = {(int64_t)c->stride};
        a.is["pads"] = {(int64_t)c->padding, (int64_t)c->padding};
        a.is["dilations"] = {1};
        a.i["group"] = 1;
    } else if (auto* c = dynamic_cast<const nn::Conv3d*>(&m)) {
        if (c->weight && c->weight->shape.size() == 5) {
            int64_t k = c->weight->shape[2];
            a.is["kernel_shape"] = {k, k, k};
        }
        a.is["strides"] = {(int64_t)c->stride, (int64_t)c->stride, (int64_t)c->stride};
        a.is["pads"] = std::vector<int64_t>(6, c->padding);
        a.is["dilations"] = {1, 1, 1};
        a.i["group"] = c->groups;
    } else if (auto* p = dynamic_cast<const nn::MaxPool2d*>(&m)) {
        int s = (p->stride < 0) ? p->kernel_size : p->stride;
        a.is["kernel_shape"] = {(int64_t)p->kernel_size, (int64_t)p->kernel_size};
        a.is["strides"] = {(int64_t)s, (int64_t)s};
        a.is["pads"] = {(int64_t)p->padding, (int64_t)p->padding,
                        (int64_t)p->padding, (int64_t)p->padding};
    } else if (auto* p = dynamic_cast<const nn::MaxPool3d*>(&m)) {
        int s = (p->stride < 0) ? p->kernel_size : p->stride;
        a.is["kernel_shape"] = {(int64_t)p->kernel_size, (int64_t)p->kernel_size,
                                (int64_t)p->kernel_size};
        a.is["strides"] = {(int64_t)s, (int64_t)s, (int64_t)s};
        a.is["pads"] = std::vector<int64_t>(6, p->padding);
    } else if (auto* p = dynamic_cast<const nn::AvgPool2d*>(&m)) {
        int s = (p->stride < 0) ? p->kernel_size : p->stride;
        a.is["kernel_shape"] = {(int64_t)p->kernel_size, (int64_t)p->kernel_size};
        a.is["strides"] = {(int64_t)s, (int64_t)s};
        a.is["pads"] = {(int64_t)p->padding, (int64_t)p->padding,
                        (int64_t)p->padding, (int64_t)p->padding};
    } else if (auto* b = dynamic_cast<const nn::BatchNorm2d*>(&m)) {
        a.f["epsilon"] = b->eps;
        a.f["momentum"] = b->momentum;
    } else if (auto* b = dynamic_cast<const nn::BatchNorm1d*>(&m)) {
        a.f["epsilon"] = b->eps;
        a.f["momentum"] = b->momentum;
    } else if (auto* b = dynamic_cast<const nn::BatchNorm3d*>(&m)) {
        a.f["epsilon"] = b->eps;
        a.f["momentum"] = b->momentum;
    } else if (auto* l = dynamic_cast<const nn::LayerNorm*>(&m)) {
        a.i["axis"] = -1;
        a.f["epsilon"] = l->eps;
    } else if (auto* g = dynamic_cast<const nn::GroupNorm*>(&m)) {
        a.i["num_groups"] = g->num_groups;
        a.f["epsilon"] = g->eps;
    } else if (auto* d = dynamic_cast<const nn::Dropout*>(&m)) {
        a.f["ratio"] = d->p;
    } else if (auto* s = dynamic_cast<const nn::Softmax*>(&m)) {
        a.i["axis"] = s->dim;
    } else if (auto* l = dynamic_cast<const nn::LeakyReLU*>(&m)) {
        a.f["alpha"] = l->negative_slope;
    } else if (auto* e = dynamic_cast<const nn::ELU*>(&m)) {
        a.f["alpha"] = e->alpha;
    } else if (dynamic_cast<const nn::Flatten*>(&m)) {
        a.i["axis"] = 1;
    } else if (auto* r = dynamic_cast<const nn::LSTM*>(&m)) {
        a.i["hidden_size"] = r->hidden_size;
        a.i["direction"] = 0;
    } else if (auto* r = dynamic_cast<const nn::GRU*>(&m)) {
        a.i["hidden_size"] = r->hidden_size;
        a.i["direction"] = 0;
    } else if (auto* r = dynamic_cast<const nn::RNN*>(&m)) {
        a.i["hidden_size"] = r->hidden_size;
        a.i["direction"] = 0;
    }
    return a;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string find_param(const std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>& params,
                       const std::string& prefix, const std::string& suffix) {
    for (auto& kv : params) {
        if (kv.first.rfind(prefix, 0) == 0 && ends_with(kv.first, suffix)) return kv.first;
    }
    return "";
}

}

std::shared_ptr<Tensor> fold_prune_mask(nn::Module& m, const std::string& name, std::shared_ptr<Tensor> w) {
    for (auto& kv : m.named_buffers()) {
        if (kv.first != name + "_mask" || !kv.second || kv.second->numel() != w->numel()) continue;
        auto mf = kv.second;
        if (mf->dtype != w->dtype || mf->device != w->device) mf = mf->to(w->device, w->dtype);
        return Ops::mul(w, mf);
    }
    return w;
}

Graph trace_module_impl(std::shared_ptr<nn::Module> model, std::shared_ptr<Tensor> example_input,
                        const std::map<std::string, std::map<int64_t, std::string>>& dynamic_axes) {
    if (!model || !example_input)
        throw std::runtime_error("[litetorch Error] onnx: trace_module got null model or input");
    model->eval();
    Graph g;
    g.name = "litetorch_traced";

    auto dax = dynamic_axes.find("input");
    if (dax != dynamic_axes.end()) {
        std::vector<std::string> dp(example_input->shape.size());
        for (auto& kv : dax->second)
            if (kv.first >= 0 && (size_t)kv.first < dp.size()) dp[kv.first] = kv.second;
        g.add_dynamic_input("input", example_input->shape, dp, example_input->dtype);
    } else {
        g.add_input("input", example_input->shape, example_input->dtype);
    }

    std::string cur = "input";
    auto kids = model->children();
    if (kids.empty()) {
        auto out = model->forward(example_input);
        auto daxo = dynamic_axes.find("output");
        if (daxo != dynamic_axes.end()) {
            std::vector<std::string> dp(out->shape.size());
            for (auto& kv : daxo->second)
                if (kv.first >= 0 && (size_t)kv.first < dp.size()) dp[kv.first] = kv.second;
            g.add_dynamic_output("output", out->shape, dp, out->dtype);
        } else {
            g.add_output("output", out->shape, out->dtype);
        }
        auto mapped = map_op(module_class_name(*model));
        if (!mapped.second.empty())
            throw std::runtime_error("[litetorch Error] onnx: trace_module: module '" +
                module_class_name(*model) + "' has no ONNX operator mapping; refusing to emit " +
                "opaque '" + mapped.first + "' node (branched or custom topologies cannot be traced)");
        std::vector<std::string> node_inputs = {cur};
        for (auto& kv : model->named_parameters()) g.add_initializer("param_" + kv.first, fold_prune_mask(*model, kv.first, kv.second));
        auto params = model->named_parameters();
        std::string w = find_param(params, "param_", "weight");
        std::string b = find_param(params, "param_", "bias");
        if (mapped.first == "Gemm") {
            if (!w.empty()) node_inputs.push_back(w);
            if (!b.empty()) node_inputs.push_back(b);
        }
        TypedAttrs ta = extract_attrs(*model);
        std::map<std::string, std::string> sattrs;
        g.add_node_typed(mapped.first, node_inputs, {"output"}, ta.i, ta.f, ta.is, sattrs,
                         mapped.second);
        return g;
    }
    size_t idx = 0;
    for (auto& child : kids) {
        if (!child) continue;
        std::string out_name = "t" + std::to_string(idx);
        auto out = child->forward(example_input);
        std::string cls = module_class_name(*child);
        auto mapped = map_op(cls);
        if (!mapped.second.empty())
            throw std::runtime_error("[litetorch Error] onnx: trace_module: child #" +
                std::to_string(idx) + " ('" + cls + "') has no ONNX operator mapping; refusing " +
                "to flatten it into an opaque '" + mapped.first + "' node (branched, nested, or " +
                "custom module topologies cannot be traced)");
        std::map<std::string, std::string> sattrs;
        sattrs["module"] = cls;
        TypedAttrs ta = extract_attrs(*child);
        std::vector<std::string> node_inputs = {cur};
        std::string pfx = "n" + std::to_string(idx) + "_";
        auto params = child->named_parameters();
        for (auto& kv : params) g.add_initializer(pfx + kv.first, fold_prune_mask(*child, kv.first, kv.second));
        if (mapped.first == "Gemm" || mapped.first == "Conv" || mapped.first == "ConvTranspose") {
            std::string w = find_param(params, "", "weight");
            std::string b = find_param(params, "", "bias");
            if (!w.empty()) node_inputs.push_back(pfx + w);
            if (!b.empty()) node_inputs.push_back(b);
        } else if (mapped.first == "BatchNormalization") {
            std::string sc = find_param(params, "", "weight");
            std::string bi = find_param(params, "", "bias");
            std::string mn = find_param(params, "", "running_mean");
            std::string vr = find_param(params, "", "running_var");
            if (!sc.empty()) node_inputs.push_back(pfx + sc);
            if (!bi.empty()) node_inputs.push_back(pfx + bi);
            if (!mn.empty()) node_inputs.push_back(pfx + mn);
            if (!vr.empty()) node_inputs.push_back(pfx + vr);
        } else if (mapped.first == "LayerNormalization" || mapped.first == "InstanceNormalization") {
            std::string sc = find_param(params, "", "weight");
            std::string bi = find_param(params, "", "bias");
            if (!sc.empty()) node_inputs.push_back(pfx + sc);
            if (!bi.empty()) node_inputs.push_back(pfx + bi);
        } else if (mapped.first == "Gather" || mapped.first == "PRelu") {
            std::string w = find_param(params, "", "weight");
            if (!w.empty()) node_inputs.push_back(pfx + w);
        }
        g.add_node_typed(mapped.first, node_inputs, {out_name}, ta.i, ta.f, ta.is, sattrs,
                         mapped.second);
        cur = out_name;
        example_input = out;
        ++idx;
    }
    auto daxo = dynamic_axes.find("output");
    if (daxo != dynamic_axes.end()) {
        std::vector<std::string> dp(example_input->shape.size());
        for (auto& kv : daxo->second)
            if (kv.first >= 0 && (size_t)kv.first < dp.size()) dp[kv.first] = kv.second;
        g.add_dynamic_output("output", example_input->shape, dp, example_input->dtype);
    } else {
        g.add_output("output", example_input->shape, example_input->dtype);
    }
    if (cur != "output") g.add_node("Identity", {cur}, {"output"});
    return g;
}

Graph trace_module(std::shared_ptr<nn::Module> model, std::shared_ptr<Tensor> example_input) {
    return trace_module_impl(model, example_input, {});
}

Graph trace_module_dynamic(std::shared_ptr<nn::Module> model, std::shared_ptr<Tensor> example_input,
                           const std::map<std::string, std::map<int64_t, std::string>>& dynamic_axes) {
    return trace_module_impl(model, example_input, dynamic_axes);
}

void export_module(std::shared_ptr<nn::Module> model, std::shared_ptr<Tensor> example_input,
                   const std::string& path) {
    export_onnx(trace_module(model, example_input), path);
}

void export_module_dynamic(std::shared_ptr<nn::Module> model, std::shared_ptr<Tensor> example_input,
                           const std::string& path,
                           const std::map<std::string, std::map<int64_t, std::string>>& dynamic_axes) {
    export_onnx(trace_module_dynamic(model, example_input, dynamic_axes), path);
}

}
}
