#include "litetorch/autograd.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

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
void topological_sort(std::shared_ptr<Node> root_node, std::vector<std::shared_ptr<Node>>& order) {
    if (!root_node) return;
    std::unordered_set<std::shared_ptr<Node>> visited;
    std::unordered_set<std::shared_ptr<Node>> visiting;
    std::vector<std::pair<std::shared_ptr<Node>, size_t>> stack;
    
    stack.push_back({root_node, 0});
    visiting.insert(root_node);
    
    while (!stack.empty()) {
        auto& top = stack.back();
        auto node = top.first;
        size_t& child_idx = top.second;
        
        if (child_idx < node->next_nodes.size()) {
            auto next = node->next_nodes[child_idx];
            child_idx++;
            if (next && visited.find(next) == visited.end() && visiting.find(next) == visiting.end()) {
                visiting.insert(next);
                stack.push_back({next, 0});
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

    std::vector<std::shared_ptr<Node>> order;
    topological_sort(root_tensor->creator, order);
    std::reverse(order.begin(), order.end());

    std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Tensor>> grads;
    if (create_graph && root_tensor->grad) {
        root_tensor->grad->requires_grad = true;
    }
    grads[root_tensor->creator] = root_tensor->grad;

    for (auto& node : order) {
        auto grad_output = grads[node];
        if (!grad_output) continue;

        std::vector<std::shared_ptr<Tensor>> input_grads = node->backward(grad_output);

        for (size_t i = 0; i < node->inputs.size(); ++i) {
            if (i >= input_grads.size()) continue;
            auto grad = input_grads[i];
            if (!grad) continue;

            if (node->inputs[i].requires_grad) {
                auto input_t = node->inputs[i].tensor.lock();
                if (input_t) {
                    for (auto& hook : input_t->backward_hooks) {
                        auto new_grad = hook(grad);
                        if (new_grad) {
                            grad = new_grad;
                        }
                    }
                    std::lock_guard<std::mutex> lock(input_t->grad_mutex);
                    if (!input_t->grad) {
                        input_t->grad = grad;
                    } else {
                        if (create_graph) {
                            input_t->grad = Ops::add(input_t->grad, grad);
                        } else {
                            input_t->grad->add_(grad);
                        }
                    }
                }
            }

            if (i < node->next_nodes.size() && node->next_nodes[i]) {
                auto next_node = node->next_nodes[i];
                if (grads.find(next_node) == grads.end()) {
                    grads[next_node] = grad;
                } else {
                    if (create_graph) {
                        grads[next_node] = Ops::add(grads[next_node], grad);
                    } else {
                        grads[next_node]->add_(grad);
                    }
                }
            }
        }
        if (!create_graph) {
            node->saved_tensors.clear();
            node->output = SavedTensor();
        }
    }
    grads.clear();
    is_create_graph_ = old_create_graph;
}

}
