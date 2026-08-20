#ifndef LITETORCH_AUTOGRAD_H
#define LITETORCH_AUTOGRAD_H

#include "litetorch/tensor.h"
#include <vector>
#include <memory>
#include <string>

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
    static thread_local std::vector<std::shared_ptr<Tensor>> active_tensors;
    static thread_local bool is_create_graph_;
    static thread_local bool is_grad_enabled_;
    static bool is_grad_enabled() { return is_grad_enabled_; }
    static void set_grad_enabled(bool enabled) { is_grad_enabled_ = enabled; }
    static void backward(std::shared_ptr<Tensor> root_tensor, bool create_graph = false);
};

}

#endif