#include "litetorch/autograd.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <unordered_set>
#include <unordered_map>

namespace litetorch {

thread_local std::vector<std::shared_ptr<Tensor>> Autograd::active_tensors;
thread_local bool Autograd::is_create_graph_ = false;
thread_local bool Autograd::is_grad_enabled_ = true;

ActiveTensorsGuard::~ActiveTensorsGuard() {
    Autograd::active_tensors.clear();
}

NoGradGuard::NoGradGuard() {
    prev_state_ = Autograd::is_grad_enabled_;
    Autograd::is_grad_enabled_ = false;
}

NoGradGuard::~NoGradGuard() {
    Autograd::is_grad_enabled_ = prev_state_;
}

namespace {

struct TraversalState {
    std::vector<Node*> order;
    std::vector<std::pair<Node*, size_t>> stack;
    std::unordered_set<Node*> visited;
    std::unordered_set<Node*> visiting;
    std::unordered_map<Node*, std::shared_ptr<Tensor>> grads;
    bool in_use = false;
};

thread_local TraversalState tl_state;

struct StateLease {
    TraversalState local;
    TraversalState* s;
    StateLease() {
        if (tl_state.in_use) {
            s = &local;
        } else {
            s = &tl_state;
            s->in_use = true;
        }
    }
    ~StateLease() {
        if (s == &tl_state) {
            s->grads.clear();
            s->in_use = false;
        }
    }
    StateLease(const StateLease&) = delete;
    StateLease& operator=(const StateLease&) = delete;
};

void topological_sort(Node* root_node, TraversalState& st) {
    if (!root_node) return;
    std::vector<Node*>& order = st.order;
    std::vector<std::pair<Node*, size_t>>& stack = st.stack;
    std::unordered_set<Node*>& visited = st.visited;
    std::unordered_set<Node*>& visiting = st.visiting;
    order.clear();
    stack.clear();
    visited.clear();
    visiting.clear();

    stack.emplace_back(root_node, 0);
    visiting.insert(root_node);

    while (!stack.empty()) {
        auto& top = stack.back();
        Node* node = top.first;
        size_t& child_idx = top.second;
        const std::vector<std::shared_ptr<Node>>& next_nodes = node->next_nodes;

        if (child_idx < next_nodes.size()) {
            Node* next = next_nodes[child_idx++].get();
            if (next && visited.find(next) == visited.end() && visiting.find(next) == visiting.end()) {
                visiting.insert(next);
                stack.emplace_back(next, 0);
            }
        } else {
            visiting.erase(node);
            visited.insert(node);
            order.push_back(node);
            stack.pop_back();
        }
    }
}

}

void Autograd::backward(std::shared_ptr<Tensor> root_tensor, bool create_graph) {
    if (!root_tensor || !root_tensor->creator) return;
    ActiveTensorsGuard guard;

    bool old_create_graph = is_create_graph_;
    is_create_graph_ = create_graph;

    StateLease lease;
    TraversalState& st = *lease.s;

    Node* root_node = root_tensor->creator.ptr.get();
    topological_sort(root_node, st);
    std::vector<Node*>& order = st.order;
    std::unordered_map<Node*, std::shared_ptr<Tensor>>& grads = st.grads;
    grads.clear();

    if (create_graph && root_tensor->grad) {
        root_tensor->grad->requires_grad = true;
    }
    grads[root_node] = root_tensor->grad;

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* node = *it;
        auto git = grads.find(node);
        if (git == grads.end()) continue;
        std::shared_ptr<Tensor> grad_output = git->second;
        if (!grad_output) continue;

        std::vector<std::shared_ptr<Tensor>> input_grads = node->backward(grad_output);
        const std::vector<NodeInput>& inputs = node->inputs;
        const std::vector<std::shared_ptr<Node>>& next_nodes = node->next_nodes;

        for (size_t i = 0; i < inputs.size(); ++i) {
            if (i >= input_grads.size()) continue;
            std::shared_ptr<Tensor> grad = input_grads[i];
            if (!grad) continue;

            const NodeInput& in = inputs[i];
            if (in.requires_grad) {
                std::shared_ptr<Tensor> input_t = in.tensor.lock();
                if (input_t) {
                    for (auto& hook : input_t->backward_hooks) {
                        std::shared_ptr<Tensor> new_grad = hook(grad);
                        if (new_grad) {
                            grad = new_grad;
                        }
                    }
                    std::lock_guard<std::mutex> lock(input_t->grad_mutex);
                    if (!input_t->grad) {
                        input_t->grad = grad;
                    } else if (create_graph) {
                        input_t->grad = Ops::add(input_t->grad, grad);
                    } else {
                        input_t->grad->add_(grad);
                    }
                }
            }

            if (i < next_nodes.size()) {
                Node* next_node = next_nodes[i].get();
                if (next_node) {
                    auto nit = grads.find(next_node);
                    if (nit == grads.end()) {
                        grads.emplace(next_node, grad->clone());
                    } else if (create_graph) {
                        nit->second = Ops::add(nit->second, grad);
                    } else {
                        nit->second->add_(grad);
                    }
                }
            }
        }
        if (!create_graph) {
            node->saved_tensors.clear();
            node->output = SavedTensor();
        }
    }
    is_create_graph_ = old_create_graph;
}

}
