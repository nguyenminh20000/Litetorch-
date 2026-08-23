#include "litetorch/jit.h"
#include "litetorch/autograd.h"
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

        ss << "inline float gelu_grad(float x, float dy) {\n";
        ss << "    float tanh_val = tanh(0.7978845608f * (x + 0.044715f * x * x * x));\n";
        ss << "    float term1 = 0.5f * (1.0f + tanh_val);\n";
        ss << "    float term2 = x * 0.5f * (1.0f - tanh_val * tanh_val) * 0.7978845608f * (1.0f + 0.134145f * x * x);\n";
        ss << "    return dy * (term1 + term2);\n";
        ss << "}\n\n";

        ss << "__kernel void " << k_name << "(\n";
        for (auto& inp : inputs_) {
            ss << "    __global const float* " << inp->name << ", int " << inp->name << "_off,\n";
        }
        ss << "    __global float* out, int out_off, int size) {\n";
        ss << "    int id = get_global_id(0);\n";
        ss << "    if (id < size) {\n";
        ss << "        out[id + out_off] = " << to_opencl_expr(expr_) << ";\n";
        ss << "    }\n";
        ss << "}\n";
    } else {
        ss << "#define max(a, b) fmaxf(a, b)\n";
        ss << "#define exp(x) expf(x)\n";
        ss << "#define tanh(x) tanhf(x)\n";
        ss << "#define sqrt(x) sqrtf(x)\n";
        ss << "#define log(x) logf(x)\n";
        ss << "#define fabs(x) fabsf(x)\n\n";

        ss << "__device__ inline float gelu_func(float x) {\n";
        ss << "    return x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));\n";
        ss << "}\n\n";

        ss << "__device__ inline float gelu_grad(float x, float dy) {\n";
        ss << "    float tanh_val = tanhf(0.7978845608f * (x + 0.044715f * x * x * x));\n";
        ss << "    float term1 = 0.5f * (1.0f + tanh_val);\n";
        ss << "    float term2 = x * 0.5f * (1.0f - tanh_val * tanh_val) * 0.7978845608f * (1.0f + 0.134145f * x * x);\n";
        ss << "    return dy * (term1 + term2);\n";
        ss << "}\n\n";

        ss << "extern \"C\" __global__ void " << k_name << "(\n";
        for (auto& inp : inputs_) {
            ss << "    const float* " << inp->name << ", int " << inp->name << "_off,\n";
        }
        ss << "    float* out, int out_off, int size) {\n";
        ss << "    int id = blockIdx.x * blockDim.x + threadIdx.x;\n";
        ss << "    if (id < size) {\n";
        ss << "        out[id + out_off] = " << to_opencl_expr(expr_) << ";\n";
        ss << "    }\n";
        ss << "}\n";
    }

    std::string source = ss.str();
    void* kernel = nullptr;
    if (is_opencl) {
        kernel = CLBackend::get().get_kernel(k_name + "_program", source, k_name);
    } else {
        kernel = active_backend->get_kernel(k_name + "_program", source, k_name);
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

    auto out = Tensor::create(args[0]->shape, args[0]->device);
    std::string shape_key = shape_to_key(args[0]->shape);
    auto active_backend = BackendDispatcher::get().get_backend();
    bool is_gpu = (args[0]->device.type == DeviceType::GPU);
    int size = out->numel();

    if (is_gpu) {
        compile_for_shape(shape_key);
        void* kernel = kernels_map_[shape_key];
        if (kernel) {
            if (active_backend) {
                std::vector<void*> gpu_mems;
                std::vector<int> offsets;
                for (auto& t : args) {
                    gpu_mems.push_back(t->gpu_data());
                    offsets.push_back(t->offset);
                }
                void* out_mem = out->gpu_data();
                int out_off = out->offset;

                std::vector<void*> arg_ptrs;
                std::vector<size_t> arg_sizes;

                for (size_t i = 0; i < args.size(); ++i) {
                    arg_ptrs.push_back(&gpu_mems[i]);
                    arg_sizes.push_back(sizeof(void*));
                    arg_ptrs.push_back(&offsets[i]);
                    arg_sizes.push_back(sizeof(int));
                }
                arg_ptrs.push_back(&out_mem);
                arg_sizes.push_back(sizeof(void*));
                arg_ptrs.push_back(&out_off);
                arg_sizes.push_back(sizeof(int));
                arg_ptrs.push_back(&size);
                arg_sizes.push_back(sizeof(int));

                active_backend->launch(kernel, {static_cast<size_t>(size)}, {}, arg_ptrs, arg_sizes);
                return out;
            } else if (CLBackend::get().is_available()) {
                std::vector<cl_mem> gpu_mems;
                std::vector<int> offsets;
                for (auto& t : args) {
                    gpu_mems.push_back(t->gpu_data());
                    offsets.push_back(t->offset);
                }
                cl_mem out_mem = out->gpu_data();
                int out_off = out->offset;
                int size = out->numel();

                std::vector<void*> arg_ptrs;
                std::vector<size_t> arg_sizes;

                for (size_t i = 0; i < args.size(); ++i) {
                    arg_ptrs.push_back(&gpu_mems[i]);
                    arg_sizes.push_back(sizeof(cl_mem));
                    arg_ptrs.push_back(&offsets[i]);
                    arg_sizes.push_back(sizeof(int));
                }
                arg_ptrs.push_back(&out_mem);
                arg_sizes.push_back(sizeof(cl_mem));
                arg_ptrs.push_back(&out_off);
                arg_sizes.push_back(sizeof(int));
                arg_ptrs.push_back(&size);
                arg_sizes.push_back(sizeof(int));

                CLBackend::get().launch(static_cast<cl_kernel>(kernel), {static_cast<size_t>(size)}, {}, arg_ptrs, arg_sizes);
                return out;
            }
        }
    }

    std::vector<std::vector<float>> cpu_inputs(args.size());
    std::vector<const float*> input_ptrs(args.size());

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i]->device.type == DeviceType::GPU) {
            cpu_inputs[i].resize(size);
            CLBackend::get().read(args[i]->gpu_data(), size * sizeof(float), cpu_inputs[i].data(), args[i]->offset * sizeof(float));
            input_ptrs[i] = cpu_inputs[i].data();
        } else {
            input_ptrs[i] = args[i]->data_ptr();
        }
    }

    std::vector<float> cpu_out(size);
    ThreadPool::get().parallel_for(0, size, [&](int64_t id) {
        std::unordered_map<std::string, float> env;
        for (size_t i = 0; i < args.size(); ++i) {
            env[inputs_[i]->name] = input_ptrs[i][id];
        }
        cpu_out[id] = evaluate_cpu(expr_, env);
    });

    if (out->device.type == DeviceType::GPU) {
        CLBackend::get().write(out->gpu_data(), size * sizeof(float), cpu_out.data(), out->offset * sizeof(float));
        auto native = BackendDispatcher::get().get_backend();
        if (native && native->is_available()) {
            native->finish();
        }
        float* out_cpu = out->data_ptr();
        if (out_cpu) {
            std::memcpy(out_cpu, cpu_out.data(), size * sizeof(float));
        }
    } else {
        std::memcpy(out->data_ptr(), cpu_out.data(), size * sizeof(float));
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
