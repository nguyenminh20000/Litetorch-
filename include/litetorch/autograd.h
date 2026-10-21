#ifndef LITETORCH_AUTOGRAD_H
#define LITETORCH_AUTOGRAD_H

#include "litetorch/tensor.h"
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <functional>

namespace litetorch {

class Tensor;

struct SavedTensor {
    std::weak_ptr<Tensor> ptr;
    SavedTensor() = default;
    explicit SavedTensor(std::shared_ptr<Tensor> p);
    SavedTensor& operator=(const std::shared_ptr<Tensor>& p);
    std::shared_ptr<Tensor> lock() const { return ptr.lock(); }
    operator std::shared_ptr<Tensor>() const;
    std::shared_ptr<Tensor> operator->() const;
    bool operator==(std::nullptr_t) const;
    bool operator!=(std::nullptr_t) const;
    bool operator!() const;
    explicit operator bool() const;
};

struct SavedTensorsVector : public std::vector<std::shared_ptr<Tensor>> {
    using std::vector<std::shared_ptr<Tensor>>::vector;
    using std::vector<std::shared_ptr<Tensor>>::operator=;
    SavedTensorsVector() = default;
    SavedTensorsVector(const std::vector<std::shared_ptr<Tensor>>& v) : std::vector<std::shared_ptr<Tensor>>(v) {}
    SavedTensorsVector& operator=(const std::vector<std::shared_ptr<Tensor>>& v) {
        std::vector<std::shared_ptr<Tensor>>::operator=(v);
        return *this;
    }
    const std::shared_ptr<Tensor>& operator[](size_t idx) const {
        return at(idx);
    }
    std::shared_ptr<Tensor>& operator[](size_t idx) {
        return at(idx);
    }
};


struct NodeInput {
    std::weak_ptr<Tensor> tensor;
    bool requires_grad;
};

class Node : public std::enable_shared_from_this<Node> {
public:
    std::string name;
    std::vector<std::shared_ptr<Node>> next_nodes;
    std::vector<NodeInput> inputs;
    SavedTensor output;
    SavedTensorsVector saved_tensors;
    std::vector<uint64_t> saved_versions;
    bool consumed = false;
    bool had_saved_data = false;

    Node(const std::string& name = "Node") : name(name) {}
    virtual ~Node() = default;

    virtual std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) = 0;
};

class ActiveTensorsGuard {
public:
    ActiveTensorsGuard() = default;
    ~ActiveTensorsGuard();
    ActiveTensorsGuard(const ActiveTensorsGuard&) = delete;
    ActiveTensorsGuard& operator=(const ActiveTensorsGuard&) = delete;
};

class NoGradGuard {
private:
    bool prev_state_;
public:
    NoGradGuard();
    ~NoGradGuard();
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
};

class Autograd {
public:
    static thread_local std::vector<std::weak_ptr<Tensor>> active_tensors;
    static thread_local bool is_create_graph_;
    static thread_local bool is_grad_enabled_;
    static bool is_grad_enabled() { return is_grad_enabled_; }
    static void set_grad_enabled(bool enabled) { is_grad_enabled_ = enabled; }
    static bool has_active_tensors();
    static void backward(std::shared_ptr<Tensor> root_tensor, bool create_graph = false, bool retain_graph = false);
    static std::vector<std::shared_ptr<Tensor>> grad(
        const std::vector<std::shared_ptr<Tensor>>& outputs,
        const std::vector<std::shared_ptr<Tensor>>& inputs,
        const std::vector<std::shared_ptr<Tensor>>& grad_outputs = {},
        bool retain_graph = false,
        bool create_graph = false,
        bool allow_unused = false);
    static void retain_grad(std::shared_ptr<Tensor> t);
    static bool retains_grad(const std::shared_ptr<Tensor>& t);
};

class NoGradContext {
private:
    bool prev_state_ = true;
    bool active_ = false;
public:
    NoGradContext() = default;
    void enter() {
        if (!active_) {
            prev_state_ = Autograd::is_grad_enabled();
            Autograd::set_grad_enabled(false);
            active_ = true;
        }
    }
    void exit() {
        if (active_) {
            Autograd::set_grad_enabled(prev_state_);
            active_ = false;
        }
    }
};

class FunctionContext {
public:
    FunctionContext() = default;
    virtual ~FunctionContext() = default;
    void save_for_backward(const std::vector<std::shared_ptr<Tensor>>& tensors);
    void save_for_backward(std::shared_ptr<Tensor> t);
    const SavedTensorsVector& get_saved_tensors() const;
    std::vector<bool> needs_input_grad() const;
    void set_needs_input_grad(const std::vector<bool>& v);
    std::shared_ptr<void> py_state;
private:
    SavedTensorsVector saved_tensors_;
    std::vector<bool> needs_input_grad_;
};

class Function {
public:
    Function() = default;
    virtual ~Function() = default;
    virtual std::vector<std::shared_ptr<Tensor>> forward(FunctionContext* ctx,
        const std::vector<std::shared_ptr<Tensor>>& inputs) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> backward(FunctionContext* ctx,
        const std::vector<std::shared_ptr<Tensor>>& grad_outputs) = 0;
    static std::vector<std::shared_ptr<Tensor>> apply(std::shared_ptr<Function> fn,
        const std::vector<std::shared_ptr<Tensor>>& inputs);
};

using AutogradFunc = std::function<std::vector<std::shared_ptr<Tensor>>(
    const std::vector<std::shared_ptr<Tensor>>&)>;
bool autograd_gradcheck(const AutogradFunc& func,
    const std::vector<std::shared_ptr<Tensor>>& inputs,
    double eps, double atol, double rtol, bool raise_exception, std::string& error_out);
bool autograd_gradgradcheck(const AutogradFunc& func,
    const std::vector<std::shared_ptr<Tensor>>& inputs,
    double eps, double atol, double rtol, bool raise_exception, std::string& error_out);

}

#endif