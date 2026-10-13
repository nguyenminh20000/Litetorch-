#ifndef LITETORCH_ONNX_H
#define LITETORCH_ONNX_H

#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace litetorch {
namespace onnx {

struct ValueInfo {
    std::string name;
    std::vector<int64_t> shape;
    DataType dtype = DataType::FP32;
};

struct Node {
    std::string op_type;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::map<std::string, std::string> attrs;
    std::string domain;
};

struct Graph {
    std::string name = "litetorch";
    std::vector<ValueInfo> inputs;
    std::vector<ValueInfo> outputs;
    std::vector<Node> nodes;
    std::map<std::string, std::shared_ptr<Tensor>> initializers;

    void add_input(const std::string& name, const std::vector<int64_t>& shape,
                   DataType dtype = DataType::FP32);
    void add_output(const std::string& name, const std::vector<int64_t>& shape,
                    DataType dtype = DataType::FP32);
    void add_node(const std::string& op_type, const std::vector<std::string>& inputs,
                  const std::vector<std::string>& outputs,
                  const std::map<std::string, std::string>& attrs = {},
                  const std::string& domain = "");
    void add_initializer(const std::string& name, std::shared_ptr<Tensor> tensor);
};

int onnx_elem_type(DataType dtype);
std::string export_onnx_text(const Graph& graph);
void export_onnx(const Graph& graph, const std::string& path);
Graph trace_module(std::shared_ptr<nn::Module> model, std::shared_ptr<Tensor> example_input);
void export_module(std::shared_ptr<nn::Module> model, std::shared_ptr<Tensor> example_input,
                   const std::string& path);

}
}

#endif
