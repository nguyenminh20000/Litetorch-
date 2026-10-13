#include "litetorch/onnx.h"
#include <cxxabi.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

namespace litetorch {
namespace onnx {

void Graph::add_input(const std::string& name, const std::vector<int64_t>& shape, DataType dtype) {
    inputs.push_back({name, shape, dtype});
}

void Graph::add_output(const std::string& name, const std::vector<int64_t>& shape, DataType dtype) {
    outputs.push_back({name, shape, dtype});
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
    for (int64_t d : v.shape) os << "          dim { dim_value: " << d << " }\n";
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
        for (const auto& kv : n.attrs)
            os << "    attribute { name: \"" << text_escape(kv.first)
               << "\" s: \"" << text_escape(kv.second) << "\" type: STRING }\n";
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
    f << export_onnx_text(graph);
    if (!f) throw std::runtime_error("[litetorch Error] onnx: write failed for " + path);
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
        {"Linear", "Gemm"}, {"ReLU", "Relu"}, {"ReLU6", "Relu"}, {"Sigmoid", "Sigmoid"},
        {"Tanh", "Tanh"}, {"GELU", "Gelu"}, {"SiLU", "Silu"}, {"Softmax", "Softmax"},
        {"LogSoftmax", "LogSoftmax"}, {"Conv1d", "Conv"}, {"Conv2d", "Conv"}, {"Conv3d", "Conv"},
        {"ConvTranspose2d", "ConvTranspose"}, {"BatchNorm1d", "BatchNormalization"},
        {"BatchNorm2d", "BatchNormalization"}, {"BatchNorm3d", "BatchNormalization"},
        {"LayerNorm", "LayerNormalization"}, {"Dropout", "Dropout"}, {"Embedding", "Gather"},
        {"Flatten", "Flatten"}, {"MaxPool2d", "MaxPool"}, {"AvgPool2d", "AveragePool"},
        {"AdaptiveAvgPool2d", "GlobalAveragePool"},
    };
    auto it = table.find(cls);
    if (it != table.end()) return {it->second, ""};
    return {cls, "litetorch"};
}

}

Graph trace_module(std::shared_ptr<nn::Module> model, std::shared_ptr<Tensor> example_input) {
    if (!model || !example_input)
        throw std::runtime_error("[litetorch Error] onnx: trace_module got null model or input");
    model->eval();
    Graph g;
    g.name = "litetorch_traced";
    g.add_input("input", example_input->shape, example_input->dtype);

    std::string cur = "input";
    auto kids = model->children();
    if (kids.empty()) {
        auto out = model->forward(example_input);
        g.add_output("output", out->shape, out->dtype);
        auto mapped = map_op(module_class_name(*model));
        g.add_node(mapped.first, {cur}, {"output"}, {}, mapped.second);
        for (auto& kv : model->named_parameters()) g.add_initializer("param_" + kv.first, kv.second);
        return g;
    }
    size_t idx = 0;
    for (auto& child : kids) {
        if (!child) continue;
        std::string out_name = "t" + std::to_string(idx);
        auto out = child->forward(example_input);
        std::string cls = module_class_name(*child);
        auto mapped = map_op(cls);
        std::map<std::string, std::string> attrs;
        attrs["module"] = cls;
        g.add_node(mapped.first, {cur}, {out_name}, attrs, mapped.second);
        for (auto& kv : child->named_parameters())
            g.add_initializer("n" + std::to_string(idx) + "_" + kv.first, kv.second);
        cur = out_name;
        example_input = out;
        ++idx;
    }
    g.add_output("output", example_input->shape, example_input->dtype);
    if (cur != "output") g.add_node("Identity", {cur}, {"output"});
    return g;
}

void export_module(std::shared_ptr<nn::Module> model, std::shared_ptr<Tensor> example_input,
                   const std::string& path) {
    export_onnx(trace_module(model, example_input), path);
}

}
}
