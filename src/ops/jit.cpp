#include "litetorch/jit.h"
#include "litetorch/autograd.h"
#include "litetorch/ops.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
#include "litetorch/thread_pool.h"
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace litetorch {

static std::string to_opencl_expr(std::shared_ptr<JITVar> var) {
    if (var->op == JITVar::OpType::INPUT) return var->name + "[id + " + var->name + "_off]";
    if (var->op == JITVar::OpType::CONST) return std::to_string(var->val) + "f";

    std::string left_expr = to_opencl_expr(var->left);
    std::string right_expr = var->right ? to_opencl_expr(var->right) : "";

    switch (var->op) {
        case JITVar::OpType::ADD: return "(" + left_expr + " + " + right_expr + ")";
        case JITVar::OpType::SUB: return "(" + left_expr + " - " + right_expr + ")";
        case JITVar::OpType::MUL: return "(" + left_expr + " * " + right_expr + ")";
        case JITVar::OpType::DIV: return "(" + left_expr + " / " + right_expr + ")";
        case JITVar::OpType::NEG: return "(-" + left_expr + ")";
        case JITVar::OpType::RELU: return "max(0.0f, " + left_expr + ")";
        case JITVar::OpType::SIGMOID: return "(1.0f / (1.0f + exp(-" + left_expr + ")))";
        case JITVar::OpType::TANH: return "tanh(" + left_expr + ")";
        case JITVar::OpType::GELU: return "gelu_func(" + left_expr + ")";
        case JITVar::OpType::SQRT: return "sqrt(" + left_expr + ")";
        case JITVar::OpType::EXP: return "exp(" + left_expr + ")";
        case JITVar::OpType::LOG: return "log(" + left_expr + ")";
        case JITVar::OpType::ABS: return "fabs(" + left_expr + ")";
        case JITVar::OpType::RELU_GRAD: return "(" + left_expr + " > 0.0f ? " + right_expr + " : 0.0f)";
        case JITVar::OpType::GELU_GRAD: return "gelu_grad(" + left_expr + ", " + right_expr + ")";
        case JITVar::OpType::ABS_GRAD: return "(" + left_expr + " > 0.0f ? " + right_expr + " : (" + left_expr + " < 0.0f ? -" + right_expr + " : 0.0f))";
        default: return "0.0f";
    }
}

static float evaluate_cpu(std::shared_ptr<JITVar> var, const std::unordered_map<std::string, float>& env) {
    if (var->op == JITVar::OpType::INPUT) return env.at(var->name);
    if (var->op == JITVar::OpType::CONST) return var->val;

    float left_val = evaluate_cpu(var->left, env);
    float right_val = var->right ? evaluate_cpu(var->right, env) : 0.0f;

    switch (var->op) {
        case JITVar::OpType::ADD: return left_val + right_val;
        case JITVar::OpType::SUB: return left_val - right_val;
        case JITVar::OpType::MUL: return left_val * right_val;
        case JITVar::OpType::DIV: return left_val / right_val;
        case JITVar::OpType::NEG: return -left_val;
        case JITVar::OpType::RELU: return std::max(0.0f, left_val);
        case JITVar::OpType::SIGMOID: return 1.0f / (1.0f + std::exp(-left_val));
        case JITVar::OpType::TANH: return std::tanh(left_val);
        case JITVar::OpType::GELU: {
            return left_val * 0.5f * (1.0f + std::tanh(0.7978845608f * (left_val + 0.044715f * left_val * left_val * left_val)));
        }
        case JITVar::OpType::SQRT: return std::sqrt(left_val);
        case JITVar::OpType::EXP: return std::exp(left_val);
        case JITVar::OpType::LOG: return std::log(left_val);
        case JITVar::OpType::ABS: return std::abs(left_val);
        case JITVar::OpType::RELU_GRAD: return left_val > 0.0f ? right_val : 0.0f;
        case JITVar::OpType::GELU_GRAD: {
            float tanh_val = std::tanh(0.7978845608f * (left_val + 0.044715f * left_val * left_val * left_val));
            float term1 = 0.5f * (1.0f + tanh_val);
            float term2 = left_val * 0.5f * (1.0f - tanh_val * tanh_val) * 0.7978845608f * (1.0f + 0.134145f * left_val * left_val);
            return right_val * (term1 + term2);
        }
        case JITVar::OpType::ABS_GRAD: return left_val > 0.0f ? right_val : (left_val < 0.0f ? -right_val : 0.0f);
        default: return 0.0f;
    }
}

static std::string shape_to_key(const std::vector<int64_t>& shape) {
    std::stringstream ss;
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i] << (i == shape.size() - 1 ? "" : "_");
    }
    return ss.str();
}

JITFunction::JITFunction(const std::string& name, std::shared_ptr<JITVar> expr, const std::vector<std::shared_ptr<JITVar>>& inputs)
    : name_(name), expr_(expr), inputs_(inputs) {}

static std::shared_ptr<Tensor> evaluate_gpu(std::shared_ptr<JITVar> var, const std::unordered_map<std::string, std::shared_ptr<Tensor>>& env, const Device& device, const std::vector<int64_t>& shape) {
    if (!var) return nullptr;
    if (var->op == JITVar::OpType::INPUT) {
        auto it = env.find(var->name);
        if (it != env.end()) return it->second;
        return Tensor::zeros(shape, device, false);
    }
    if (var->op == JITVar::OpType::CONST) {
        size_t n = 1;
        for (auto s : shape) n *= s;
        std::vector<float> cdata(n, var->val);
        return Tensor::from_vector(cdata, shape, device, false);
    }

    auto left_tensor = evaluate_gpu(var->left, env, device, shape);
    auto right_tensor = var->right ? evaluate_gpu(var->right, env, device, shape) : nullptr;

    switch (var->op) {
        case JITVar::OpType::ADD: return Ops::add(left_tensor, right_tensor);
        case JITVar::OpType::SUB: return Ops::sub(left_tensor, right_tensor);
        case JITVar::OpType::MUL: return Ops::mul(left_tensor, right_tensor);
        case JITVar::OpType::DIV: return Ops::div(left_tensor, right_tensor);
        case JITVar::OpType::NEG: return Ops::neg(left_tensor);
        case JITVar::OpType::RELU: return Ops::relu(left_tensor);
        case JITVar::OpType::GELU: return Ops::gelu(left_tensor);
        case JITVar::OpType::SIGMOID: return Ops::sigmoid(left_tensor);
        case JITVar::OpType::TANH: return Ops::tanh(left_tensor);
        case JITVar::OpType::SQRT: return Ops::sqrt(left_tensor);
        case JITVar::OpType::EXP: return Ops::exp(left_tensor);
        case JITVar::OpType::LOG: return Ops::log(left_tensor);
        case JITVar::OpType::ABS: return Ops::abs(left_tensor);
        case JITVar::OpType::RELU_GRAD: {
            if (device.type == DeviceType::GPU) {
                auto relu_fwd = Ops::relu(left_tensor);
                if (relu_fwd->creator) {
                    return relu_fwd->creator->backward(right_tensor)[0];
                }
            }
            auto out = Tensor::create(shape, device);
            float* in_ptr = left_tensor->data_ptr();
            float* gout_ptr = right_tensor->data_ptr();
            float* out_ptr = out->data_ptr();
            size_t size = out->numel();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                out_ptr[i] = in_ptr[i] > 0.0f ? gout_ptr[i] : 0.0f;
            });
            return out;
        }
        case JITVar::OpType::GELU_GRAD: {
            if (device.type == DeviceType::GPU) {
                auto gelu_fwd = Ops::gelu(left_tensor);
                if (gelu_fwd->creator) {
                    return gelu_fwd->creator->backward(right_tensor)[0];
                }
            }
            auto out = Tensor::create(shape, device);
            float* in_ptr = left_tensor->data_ptr();
            float* gout_ptr = right_tensor->data_ptr();
            float* out_ptr = out->data_ptr();
            size_t size = out->numel();
            ThreadPool::get().parallel_for(0, size, [&](int64_t idx) {
                float x = in_ptr[idx];
                float tanh_val = std::tanh(0.7978845608f * (x + 0.044715f * x * x * x));
                float term1 = 0.5f * (1.0f + tanh_val);
                float term2 = x * 0.5f * (1.0f - tanh_val * tanh_val) * 0.7978845608f * (1.0f + 0.134145f * x * x);
                out_ptr[idx] = gout_ptr[idx] * (term1 + term2);
            });
            return out;
        }
        case JITVar::OpType::ABS_GRAD: {
            if (device.type == DeviceType::GPU) {
                auto abs_fwd = Ops::abs(left_tensor);
                if (abs_fwd->creator) {
                    return abs_fwd->creator->backward(right_tensor)[0];
                }
            }
            auto out = Tensor::create(shape, device);
            float* in_ptr = left_tensor->data_ptr();
            float* gout_ptr = right_tensor->data_ptr();
            float* out_ptr = out->data_ptr();
            size_t size = out->numel();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                out_ptr[i] = in_ptr[i] > 0.0f ? gout_ptr[i] : (in_ptr[i] < 0.0f ? -gout_ptr[i] : 0.0f);
            });
            return out;
        }
        default: return left_tensor;
    }
}

class JITNode : public Node {
public:
    std::shared_ptr<JITVar> expr_;
    std::vector<std::shared_ptr<JITVar>> input_vars_;
    std::vector<std::shared_ptr<Tensor>> saved_inputs_;
    std::vector<bool> inputs_need_grad_;

    JITNode(
        std::shared_ptr<JITVar> expr,
        const std::vector<std::shared_ptr<JITVar>>& input_vars,
        const std::vector<std::shared_ptr<Tensor>>& saved_inputs,
        const std::vector<bool>& inputs_need_grad
    ) : Node("JITNode"), expr_(expr), input_vars_(input_vars), saved_inputs_(saved_inputs), inputs_need_grad_(inputs_need_grad) {}

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::vector<std::shared_ptr<Tensor>> grads;
        grads.reserve(saved_inputs_.size());

        auto grad_var = std::make_shared<JITVar>(JITVar::OpType::INPUT, "__grad_output__");

        for (size_t i = 0; i < saved_inputs_.size(); ++i) {
            if (!inputs_need_grad_[i]) {
                grads.push_back(nullptr);
                continue;
            }

            auto deriv_expr = Tracer::derivative(expr_, input_vars_[i]);
            auto bwd_expr = deriv_expr * grad_var;

            std::vector<std::shared_ptr<JITVar>> bwd_inputs = input_vars_;
            bwd_inputs.push_back(grad_var);

            JITFunction bwd_fn("jit_bwd_" + std::to_string(i), bwd_expr, bwd_inputs);

            std::vector<std::shared_ptr<Tensor>> bwd_args = saved_inputs_;
            bwd_args.push_back(grad_output);

            auto in_grad = bwd_fn(bwd_args);
            grads.push_back(in_grad);
        }
        return grads;
    }
};

void JITFunction::compile_for_shape(const std::string& shape_key) {
    if (kernels_map_.find(shape_key) != kernels_map_.end()) return;
    auto active_backend = BackendDispatcher::get().get_backend();
    bool is_opencl = (active_backend == nullptr);

    if (is_opencl && !CLBackend::get().is_available()) return;
    if (!is_opencl && !active_backend->is_available()) return;

    std::stringstream ss;
    std::string k_name = name_ + "_" + shape_key;

    if (is_opencl) {
        ss << "inline float gelu_func(float x) {\n";
        ss << "    return x * 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));\n";
        ss << "}\n\n";

        ss << "inline float gelu_grad(float x, float grad_out) {\n";
        ss << "    float tanh_val = tanh(0.7978845608f * (x + 0.044715f * x * x * x));\n";
        ss << "    float term1 = 0.5f * (1.0f + tanh_val);\n";
        ss << "    float term2 = x * 0.5f * (1.0f - tanh_val * tanh_val) * 0.7978845608f * (1.0f + 0.134145f * x * x);\n";
        ss << "    return grad_out * (term1 + term2);\n";
        ss << "}\n\n";

        ss << "__kernel void " << k_name << "(\n";
        for (size_t i = 0; i < inputs_.size(); ++i) {
            ss << "    __global const float* " << inputs_[i]->name << ",\n";
            ss << "    int " << inputs_[i]->name << "_off,\n";
        }
        ss << "    __global float* out,\n";
        ss << "    int out_off,\n";
        ss << "    int size\n";
        ss << ") {\n";
        ss << "    int idx = get_global_id(0);\n";
        ss << "    if (idx < size) {\n";
        for (size_t i = 0; i < inputs_.size(); ++i) {
            ss << "        float " << inputs_[i]->name << "_val = " << inputs_[i]->name << "[idx + " << inputs_[i]->name << "_off];\n";
        }
        ss << "        out[idx + out_off] = " << to_opencl_expr(expr_) << ";\n";
        ss << "    }\n";
        ss << "}\n";
    }

    std::string source = ss.str();
    void* kernel = nullptr;
    if (is_opencl && !source.empty()) {
        kernel = CLBackend::get().get_kernel(k_name + "_program", source, k_name);
    }
    kernels_map_[shape_key] = kernel;
}

std::shared_ptr<Tensor> JITFunction::operator()(const std::vector<std::shared_ptr<Tensor>>& args) {
    if (args.empty()) {
        throw std::runtime_error("[litetorch Error] JITFunction requires at least one input");
    }

    if (args.size() != inputs_.size()) {
        throw std::runtime_error("[litetorch Error] JITFunction input count mismatch");
    }

    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i]->shape != args[0]->shape) {
            throw std::runtime_error("[litetorch Error] JITFunction requires all inputs to have identical shapes");
        }
    }

    std::shared_ptr<Tensor> out = nullptr;
    if (args[0]->device.type == DeviceType::GPU) {
        std::unordered_map<std::string, std::shared_ptr<Tensor>> env;
        for (size_t i = 0; i < args.size(); ++i) {
            env[inputs_[i]->name] = args[i];
        }
        out = evaluate_gpu(expr_, env, args[0]->device, args[0]->shape);
    } else {
        int size = args[0]->numel();
        out = Tensor::create(args[0]->shape, args[0]->device);

        std::vector<const float*> input_ptrs(args.size());
        for (size_t i = 0; i < args.size(); ++i) {
            input_ptrs[i] = args[i]->data_ptr();
        }

        std::vector<float> cpu_out(size);
        ThreadPool::get().parallel_for(0, size, [&](int64_t id) {
            std::unordered_map<std::string, float> env;
            for (size_t i = 0; i < args.size(); ++i) {
                env[inputs_[i]->name] = input_ptrs[i][id];
            }
            cpu_out[id] = evaluate_cpu(expr_, env);
        });

        std::memcpy(out->data_ptr(), cpu_out.data(), size * sizeof(float));
    }

    bool any_requires_grad = false;
    std::vector<bool> inputs_need_grad(args.size(), false);
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i]->requires_grad) {
            any_requires_grad = true;
            inputs_need_grad[i] = true;
        }
    }

    if (any_requires_grad) {
        auto node = std::make_shared<JITNode>(expr_, inputs_, args, inputs_need_grad);
        for (size_t i = 0; i < args.size(); ++i) {
            node->inputs.push_back({ args[i], inputs_need_grad[i] });
            node->next_nodes.push_back(args[i]->creator);
        }
        node->saved_tensors = args;
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }

    return out;
}

static std::shared_ptr<JITVar> trace_tensor(
    std::shared_ptr<Tensor> t,
    const std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<JITVar>>& input_vars,
    std::unordered_map<Tensor*, std::shared_ptr<JITVar>>& tensor_cache
) {
    if (!t) return nullptr;
    auto it = tensor_cache.find(t.get());
    if (it != tensor_cache.end()) return it->second;

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].get() == t.get()) {
            auto var = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x_" + std::to_string(i));
            input_vars.push_back(var);
            tensor_cache[t.get()] = var;
            return var;
        }
    }

    std::shared_ptr<Node> creator = t->creator;
    if (!creator) {
        if (t->numel() == 1) {
            float val = t->to_vector()[0];
            auto var = std::make_shared<JITVar>(val);
            tensor_cache[t.get()] = var;
            return var;
        }
        throw std::runtime_error("[litetorch Error] Traced graph reached a leaf tensor that is not in the input list.");
    }

    std::shared_ptr<JITVar> res = nullptr;
    if (creator->name == "Add") {
        auto a = creator->inputs[0].tensor.lock();
        auto b = creator->inputs[1].tensor.lock();
        res = trace_tensor(a, inputs, input_vars, tensor_cache) + trace_tensor(b, inputs, input_vars, tensor_cache);
    } else if (creator->name == "Sub") {
        auto a = creator->inputs[0].tensor.lock();
        auto b = creator->inputs[1].tensor.lock();
        res = trace_tensor(a, inputs, input_vars, tensor_cache) - trace_tensor(b, inputs, input_vars, tensor_cache);
    } else if (creator->name == "Mul") {
        auto a = creator->inputs[0].tensor.lock();
        auto b = creator->inputs[1].tensor.lock();
        res = trace_tensor(a, inputs, input_vars, tensor_cache) * trace_tensor(b, inputs, input_vars, tensor_cache);
    } else if (creator->name == "Div") {
        auto a = creator->inputs[0].tensor.lock();
        auto b = creator->inputs[1].tensor.lock();
        res = trace_tensor(a, inputs, input_vars, tensor_cache) / trace_tensor(b, inputs, input_vars, tensor_cache);
    } else if (creator->name == "ReLU" || creator->name == "Relu") {
        auto a = creator->inputs[0].tensor.lock();
        res = JIT::relu(trace_tensor(a, inputs, input_vars, tensor_cache));
    } else if (creator->name == "GELU" || creator->name == "Gelu") {
        auto a = creator->inputs[0].tensor.lock();
        res = JIT::gelu(trace_tensor(a, inputs, input_vars, tensor_cache));
    } else if (creator->name == "Sigmoid") {
        auto a = creator->inputs[0].tensor.lock();
        res = JIT::sigmoid(trace_tensor(a, inputs, input_vars, tensor_cache));
    } else if (creator->name == "Tanh") {
        auto a = creator->inputs[0].tensor.lock();
        res = JIT::tanh(trace_tensor(a, inputs, input_vars, tensor_cache));
    } else if (creator->name == "Sqrt") {
        auto a = creator->inputs[0].tensor.lock();
        res = JIT::sqrt(trace_tensor(a, inputs, input_vars, tensor_cache));
    } else if (creator->name == "Exp") {
        auto a = creator->inputs[0].tensor.lock();
        res = JIT::exp(trace_tensor(a, inputs, input_vars, tensor_cache));
    } else if (creator->name == "Log") {
        auto a = creator->inputs[0].tensor.lock();
        res = JIT::log(trace_tensor(a, inputs, input_vars, tensor_cache));
    } else if (creator->name == "Abs") {
        auto a = creator->inputs[0].tensor.lock();
        res = JIT::abs(trace_tensor(a, inputs, input_vars, tensor_cache));
    } else if (creator->name == "Neg") {
        auto a = creator->inputs[0].tensor.lock();
        res = -trace_tensor(a, inputs, input_vars, tensor_cache);
    } else {
        throw std::runtime_error("[litetorch Error] Unsupported operator name in JIT compilation: " + creator->name);
    }

    tensor_cache[t.get()] = res;
    return res;
}

std::shared_ptr<JITFunction> Tracer::trace(
    const std::vector<std::shared_ptr<Tensor>>& inputs,
    std::function<std::shared_ptr<Tensor>(const std::vector<std::shared_ptr<Tensor>>&)> func,
    const std::string& name
) {
    std::vector<bool> original_requires_grad;
    for (auto& inp : inputs) {
        original_requires_grad.push_back(inp->requires_grad);
        inp->requires_grad = true;
    }

    auto output = func(inputs);

    std::vector<std::shared_ptr<JITVar>> input_vars;
    std::unordered_map<Tensor*, std::shared_ptr<JITVar>> tensor_cache;
    auto expr = trace_tensor(output, inputs, input_vars, tensor_cache);

    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i]->requires_grad = original_requires_grad[i];
    }

    std::vector<std::shared_ptr<JITVar>> ordered_input_vars;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto it = tensor_cache.find(inputs[i].get());
        if (it != tensor_cache.end()) {
            ordered_input_vars.push_back(it->second);
        }
    }

    return std::make_shared<JITFunction>(name, expr, ordered_input_vars);
}

std::shared_ptr<JITVar> Tracer::derivative(std::shared_ptr<JITVar> var, std::shared_ptr<JITVar> wrt) {
    if (var->op == JITVar::OpType::CONST) {
        return std::make_shared<JITVar>(0.0f);
    }
    if (var->op == JITVar::OpType::INPUT) {
        if (var->name == wrt->name) {
            return std::make_shared<JITVar>(1.0f);
        } else {
            return std::make_shared<JITVar>(0.0f);
        }
    }

    auto d_left = derivative(var->left, wrt);
    auto d_right = var->right ? derivative(var->right, wrt) : nullptr;

    switch (var->op) {
        case JITVar::OpType::ADD:
            return d_left + d_right;
        case JITVar::OpType::SUB:
            return d_left - d_right;
        case JITVar::OpType::MUL:
            return d_left * var->right + var->left * d_right;
        case JITVar::OpType::DIV: {
            auto num = d_left * var->right - var->left * d_right;
            auto den = var->right * var->right;
            return num / den;
        }
        case JITVar::OpType::NEG:
            return -d_left;
        case JITVar::OpType::RELU: {
            auto grad_node = std::make_shared<JITVar>(JITVar::OpType::RELU_GRAD);
            grad_node->left = var->left;
            grad_node->right = d_left;
            return grad_node;
        }
        case JITVar::OpType::SIGMOID: {
            auto sig = std::make_shared<JITVar>(JITVar::OpType::SIGMOID);
            sig->left = var->left;
            return d_left * sig * (1.0f - sig);
        }
        case JITVar::OpType::TANH: {
            auto t = std::make_shared<JITVar>(JITVar::OpType::TANH);
            t->left = var->left;
            return d_left * (1.0f - t * t);
        }
        case JITVar::OpType::GELU: {
            auto grad_node = std::make_shared<JITVar>(JITVar::OpType::GELU_GRAD);
            grad_node->left = var->left;
            grad_node->right = d_left;
            return grad_node;
        }
        case JITVar::OpType::SQRT: {
            auto s = std::make_shared<JITVar>(JITVar::OpType::SQRT);
            s->left = var->left;
            return d_left / (2.0f * s);
        }
        case JITVar::OpType::EXP: {
            auto e = std::make_shared<JITVar>(JITVar::OpType::EXP);
            e->left = var->left;
            return d_left * e;
        }
        case JITVar::OpType::LOG:
            return d_left / var->left;
        case JITVar::OpType::ABS: {
            auto grad_node = std::make_shared<JITVar>(JITVar::OpType::ABS_GRAD);
            grad_node->left = var->left;
            grad_node->right = d_left;
            return grad_node;
        }
    }
    return std::make_shared<JITVar>(0.0f);
}

void JITFunction::save(const std::string& filepath) {
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        throw std::runtime_error("[litetorch Error] Failed to open JITFunction save path: " + filepath);
    }
    ofs << name_ << "\n";
    ofs << inputs_.size() << "\n";
    for (auto& inp : inputs_) {
        ofs << inp->name << "\n";
    }
    ofs << Tracer::serialize_var(expr_) << "\n";
}

std::shared_ptr<JITFunction> JITFunction::load(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw std::runtime_error("[litetorch Error] Failed to open JITFunction load path: " + filepath);
    }
    std::string name;
    std::getline(ifs, name);

    std::string count_str;
    std::getline(ifs, count_str);
    size_t input_count = std::stoull(count_str);

    std::vector<std::shared_ptr<JITVar>> inputs;
    for (size_t i = 0; i < input_count; ++i) {
        std::string inp_name;
        std::getline(ifs, inp_name);
        inputs.push_back(std::make_shared<JITVar>(JITVar::OpType::INPUT, inp_name));
    }

    std::string expr_str;
    std::getline(ifs, expr_str);

    size_t pos = 0;
    auto expr = Tracer::deserialize_var(expr_str, pos);

    return std::make_shared<JITFunction>(name, expr, inputs);
}

std::string Tracer::serialize_var(std::shared_ptr<JITVar> var) {
    if (!var) return "N";
    std::stringstream ss;
    ss << static_cast<int>(var->op) << " " << (var->name.empty() ? "_" : var->name) << " " << var->val << " ";
    ss << serialize_var(var->left) << " " << serialize_var(var->right);
    return ss.str();
}

static std::shared_ptr<JITVar> deserialize_helper(std::istringstream& iss) {
    std::string token;
    if (!(iss >> token)) return nullptr;
    if (token == "N") return nullptr;
    int op_val = std::stoi(token);
    std::string name;
    iss >> name;
    if (name == "_") name = "";
    float val = 0.0f;
    iss >> val;
    auto var = std::make_shared<JITVar>(static_cast<JITVar::OpType>(op_val), name);
    var->val = val;
    var->left = deserialize_helper(iss);
    var->right = deserialize_helper(iss);
    return var;
}

std::shared_ptr<JITVar> Tracer::deserialize_var(const std::string& str, size_t& pos) {
    std::istringstream iss(str);
    return deserialize_helper(iss);
}

}
