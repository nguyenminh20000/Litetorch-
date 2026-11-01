#ifndef LITETORCH_JIT_H
#define LITETORCH_JIT_H

#include "litetorch/tensor.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <cstdint>

namespace litetorch {

class JITVar : public std::enable_shared_from_this<JITVar> {
public:
    enum class OpType {
        INPUT,
        CONST,
        ADD,
        SUB,
        MUL,
        DIV,
        RELU,
        GELU,
        SIGMOID,
        TANH,
        SQRT,
        EXP,
        LOG,
        ABS,
        NEG,
        RELU_GRAD,
        GELU_GRAD,
        ABS_GRAD
    };

    OpType op;
    std::string name;
    float val = 0.0f;
    std::shared_ptr<JITVar> left = nullptr;
    std::shared_ptr<JITVar> right = nullptr;

    JITVar(OpType op, const std::string& name = "") : op(op), name(name) {}
    JITVar(float val) : op(OpType::CONST), val(val) {}
};

inline std::shared_ptr<JITVar> operator+(std::shared_ptr<JITVar> a, std::shared_ptr<JITVar> b) {
    auto res = std::make_shared<JITVar>(JITVar::OpType::ADD);
    res->left = a;
    res->right = b;
    return res;
}
inline std::shared_ptr<JITVar> operator+(std::shared_ptr<JITVar> a, float b) {
    return a + std::make_shared<JITVar>(b);
}
inline std::shared_ptr<JITVar> operator+(float a, std::shared_ptr<JITVar> b) {
    return std::make_shared<JITVar>(a) + b;
}

inline std::shared_ptr<JITVar> operator-(std::shared_ptr<JITVar> a, std::shared_ptr<JITVar> b) {
    auto res = std::make_shared<JITVar>(JITVar::OpType::SUB);
    res->left = a;
    res->right = b;
    return res;
}
inline std::shared_ptr<JITVar> operator-(std::shared_ptr<JITVar> a, float b) {
    return a - std::make_shared<JITVar>(b);
}
inline std::shared_ptr<JITVar> operator-(float a, std::shared_ptr<JITVar> b) {
    return std::make_shared<JITVar>(a) - b;
}

inline std::shared_ptr<JITVar> operator*(std::shared_ptr<JITVar> a, std::shared_ptr<JITVar> b) {
    auto res = std::make_shared<JITVar>(JITVar::OpType::MUL);
    res->left = a;
    res->right = b;
    return res;
}
inline std::shared_ptr<JITVar> operator*(std::shared_ptr<JITVar> a, float b) {
    return a * std::make_shared<JITVar>(b);
}
inline std::shared_ptr<JITVar> operator*(float a, std::shared_ptr<JITVar> b) {
    return std::make_shared<JITVar>(a) * b;
}

inline std::shared_ptr<JITVar> operator/(std::shared_ptr<JITVar> a, std::shared_ptr<JITVar> b) {
    auto res = std::make_shared<JITVar>(JITVar::OpType::DIV);
    res->left = a;
    res->right = b;
    return res;
}
inline std::shared_ptr<JITVar> operator/(std::shared_ptr<JITVar> a, float b) {
    return a / std::make_shared<JITVar>(b);
}
inline std::shared_ptr<JITVar> operator/(float a, std::shared_ptr<JITVar> b) {
    return std::make_shared<JITVar>(a) / b;
}

inline std::shared_ptr<JITVar> operator-(std::shared_ptr<JITVar> a) {
    auto res = std::make_shared<JITVar>(JITVar::OpType::NEG);
    res->left = a;
    return res;
}

namespace JIT {
inline std::shared_ptr<JITVar> relu(std::shared_ptr<JITVar> a) {
    auto res = std::make_shared<JITVar>(JITVar::OpType::RELU);
    res->left = a;
    return res;
}
inline std::shared_ptr<JITVar> gelu(std::shared_ptr<JITVar> a) {
    auto res = std::make_shared<JITVar>(JITVar::OpType::GELU);
    res->left = a;
    return res;
}
inline std::shared_ptr<JITVar> sigmoid(std::shared_ptr<JITVar> a) {
    auto res = std::make_shared<JITVar>(JITVar::OpType::SIGMOID);
    res->left = a;
    return res;
}
inline std::shared_ptr<JITVar> tanh(std::shared_ptr<JITVar> a) {
    auto res = std::make_shared<JITVar>(JITVar::OpType::TANH);
    res->left = a;
    return res;
}
inline std::shared_ptr<JITVar> sqrt(std::shared_ptr<JITVar> a) {
    auto res = std::make_shared<JITVar>(JITVar::OpType::SQRT);
    res->left = a;
    return res;
}
inline std::shared_ptr<JITVar> exp(std::shared_ptr<JITVar> a) {
    auto res = std::make_shared<JITVar>(JITVar::OpType::EXP);
    res->left = a;
    return res;
}
inline std::shared_ptr<JITVar> log(std::shared_ptr<JITVar> a) {
    auto res = std::make_shared<JITVar>(JITVar::OpType::LOG);
    res->left = a;
    return res;
}
inline std::shared_ptr<JITVar> abs(std::shared_ptr<JITVar> a) {
    auto res = std::make_shared<JITVar>(JITVar::OpType::ABS);
    res->left = a;
    return res;
}
}


class JITShapeCacheEntry {
public:
    std::vector<int64_t> out_shape;
    std::vector<std::vector<int64_t>> input_strides;
};

class JITFunction {
private:
    std::string name_;
    std::shared_ptr<JITVar> expr_;
    std::vector<std::shared_ptr<JITVar>> inputs_;
    std::unordered_map<std::string, JITShapeCacheEntry> shape_cache_;
    uint64_t shape_cache_hits_ = 0;

public:
    JITFunction(const std::string& name, std::shared_ptr<JITVar> expr, const std::vector<std::shared_ptr<JITVar>>& inputs);

    std::shared_ptr<Tensor> operator()(const std::vector<std::shared_ptr<Tensor>>& args);

    size_t shape_cache_size() const { return shape_cache_.size(); }
    uint64_t shape_cache_hits() const { return shape_cache_hits_; }

    void save(const std::string& filepath);
    static std::shared_ptr<JITFunction> load(const std::string& filepath);
};

class Tracer {
public:
    static bool tracing_active();

    static std::shared_ptr<JITFunction> trace(
        const std::vector<std::shared_ptr<Tensor>>& inputs,
        std::function<std::shared_ptr<Tensor>(const std::vector<std::shared_ptr<Tensor>>&)> func,
        const std::string& name = "traced_fn"
    );

    static std::shared_ptr<JITVar> derivative(std::shared_ptr<JITVar> var, std::shared_ptr<JITVar> wrt);

    static std::string serialize_var(std::shared_ptr<JITVar> var);
    static std::shared_ptr<JITVar> deserialize_var(const std::string& str, size_t& pos);
};

}

#endif
