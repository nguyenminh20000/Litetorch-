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
    std::vector<std::string> dim_params;
};

struct Node {
    std::string op_type;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::map<std::string, std::string> attrs;
    std::map<std::string, int64_t> int_attrs;
    std::map<std::string, float> float_attrs;
    std::map<std::string, std::vector<int64_t>> ints_attrs;
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
    void add_dynamic_input(const std::string& name, const std::vector<int64_t>& shape,
                           const std::vector<std::string>& dim_params,
                           DataType dtype = DataType::FP32);
    void add_dynamic_output(const std::string& name, const std::vector<int64_t>& shape,
                            const std::vector<std::string>& dim_params,
                            DataType dtype = DataType::FP32);
    void add_node(const std::string& op_type, const std::vector<std::string>& inputs,
                  const std::vector<std::string>& outputs,
                  const std::map<std::string, std::string>& attrs = {},
                  const std::string& domain = "");
    void add_node_typed(const std::string& op_type, const std::vector<std::string>& inputs,
                        const std::vector<std::string>& outputs,
                        const std::map<std::string, int64_t>& int_attrs,
                        const std::map<std::string, float>& float_attrs,
                        const std::map<std::string, std::vector<int64_t>>& ints_attrs,
                        const std::map<std::string, std::string>& attrs = {},
                        const std::string& domain = "");
    void add_initializer(const std::string& name, std::shared_ptr<Tensor> tensor);
};

int onnx_elem_type(DataType dtype);
std::string export_onnx_text(const Graph& graph);
std::string export_onnx_binary(const Graph& graph);
void export_onnx(const Graph& graph, const std::string& path);
Graph trace_module(std::shared_ptr<nn::Module> model, std::shared_ptr<Tensor> example_input);
Graph trace_module_dynamic(std::shared_ptr<nn::Module> model, std::shared_ptr<Tensor> example_input,
                           const std::map<std::string, std::map<int64_t, std::string>>& dynamic_axes);
void export_module(std::shared_ptr<nn::Module> model, std::shared_ptr<Tensor> example_input,
                   const std::string& path);
void export_module_dynamic(std::shared_ptr<nn::Module> model, std::shared_ptr<Tensor> example_input,
                           const std::string& path,
                           const std::map<std::string, std::map<int64_t, std::string>>& dynamic_axes);

}
}

#endif
