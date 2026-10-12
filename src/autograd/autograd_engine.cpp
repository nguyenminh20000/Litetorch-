#include "litetorch/autograd.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cmath>
#include <functional>
#include <string>
#include <sstream>
#include <stdexcept>

namespace litetorch {

thread_local std::vector<std::weak_ptr<Tensor>> Autograd::active_tensors;
thread_local bool Autograd::is_create_graph_ = false;
thread_local bool Autograd::is_grad_enabled_ = true;

ActiveTensorsGuard::~ActiveTensorsGuard() {
    Autograd::active_tensors.clear();
}

bool Autograd::has_active_tensors() {
    auto& v = active_tensors;
    v.erase(std::remove_if(v.begin(), v.end(),
        [](const std::weak_ptr<Tensor>& w) { return w.expired(); }),
        v.end());
    return !v.empty();
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
    if (!root_tensor->grad) {
        if (root_tensor->numel() != 1) {
            throw std::runtime_error("[litetorch Error] Autograd::backward: grad must be specified for non-scalar tensors");
        }
        root_tensor->grad = Tensor::from_vector({1.0f}, root_tensor->shape, root_tensor->device, false, root_tensor->dtype);
    }
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
                        input_t->grad = grad->clone();
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

// === W2-T4 autograd: grad / retain_grad / Function / gradcheck ===

namespace litetorch {

void FunctionContext::save_for_backward(const std::vector<std::shared_ptr<Tensor>>& tensors) {
    saved_tensors_ = SavedTensorsVector(tensors);
}

void FunctionContext::save_for_backward(std::shared_ptr<Tensor> t) {
    saved_tensors_.push_back(t);
}

const SavedTensorsVector& FunctionContext::get_saved_tensors() const {
    return saved_tensors_;
}

std::vector<bool> FunctionContext::needs_input_grad() const {
    return needs_input_grad_;
}

void FunctionContext::set_needs_input_grad(const std::vector<bool>& v) {
    needs_input_grad_ = v;
}

namespace {
std::mutex g_retain_mutex;
std::vector<std::weak_ptr<Tensor>> g_retained;

void accumulate_grad_w2t4(std::shared_ptr<Tensor>& acc, std::shared_ptr<Tensor> g, bool create_graph) {
    if (!g) return;
    if (!acc) {
        acc = g->clone();
        return;
    }
    if (create_graph) acc = Ops::add(acc, g);
    else acc->add_(g);
}
}

void Autograd::retain_grad(std::shared_ptr<Tensor> t) {
    if (!t) return;
    std::lock_guard<std::mutex> lk(g_retain_mutex);
    for (auto& w : g_retained) {
        if (auto s = w.lock()) {
            if (s.get() == t.get()) return;
        }
    }
    g_retained.push_back(t);
}

bool Autograd::retains_grad(const std::shared_ptr<Tensor>& t) {
    if (!t) return false;
    std::lock_guard<std::mutex> lk(g_retain_mutex);
    for (auto& w : g_retained) {
        if (auto s = w.lock()) {
            if (s.get() == t.get()) return true;
        }
    }
    return false;
}

std::vector<std::shared_ptr<Tensor>> Autograd::grad(
    const std::vector<std::shared_ptr<Tensor>>& outputs,
    const std::vector<std::shared_ptr<Tensor>>& inputs,
    const std::vector<std::shared_ptr<Tensor>>& grad_outputs,
    bool retain_graph, bool create_graph, bool allow_unused) {
    if (outputs.empty())
        throw std::runtime_error("[litetorch Error] autograd.grad: outputs must be non-empty");
    if (inputs.empty())
        throw std::runtime_error("[litetorch Error] autograd.grad: inputs must be non-empty");
    if (!grad_outputs.empty() && grad_outputs.size() != outputs.size())
        throw std::runtime_error("[litetorch Error] autograd.grad: grad_outputs length must match outputs");

    std::unique_ptr<NoGradGuard> ng;
    bool old_enabled = is_grad_enabled_;
    bool old_create_graph = is_create_graph_;
    if (create_graph) {
        is_grad_enabled_ = true;
        is_create_graph_ = true;
    } else {
        ng = std::make_unique<NoGradGuard>();
    }

    std::vector<std::shared_ptr<Tensor>> result;
    try {
        StateLease lease;
        TraversalState& st = *lease.s;
        std::unordered_map<Node*, std::shared_ptr<Tensor>>& grads = st.grads;
        grads.clear();
        std::vector<Node*> order;
        std::unordered_set<Node*> seen;

        for (size_t oi = 0; oi < outputs.size(); ++oi) {
            auto& out = outputs[oi];
            if (!out) throw std::runtime_error("[litetorch Error] autograd.grad: output is null");
            bool is_identity = false;
            for (auto& in_t : inputs) {
                if (in_t.get() == out.get()) { is_identity = true; break; }
            }
            if (is_identity) continue;
            std::shared_ptr<Tensor> seed;
            if (!grad_outputs.empty()) {
                seed = grad_outputs[oi];
                if (!seed) throw std::runtime_error("[litetorch Error] autograd.grad: grad_output is null");
                if (seed->shape != out->shape)
                    throw std::runtime_error("[litetorch Error] autograd.grad: grad_output shape mismatch");
                if (seed->device != out->device)
                    throw std::runtime_error("[litetorch Error] autograd.grad: grad_output device mismatch");
                seed = seed->clone();
            } else {
                if (out->numel() != 1)
                    throw std::runtime_error("[litetorch Error] autograd.grad: grad_outputs must be specified for non-scalar outputs");
                seed = Tensor::ones(out->shape, out->device, false, out->dtype);
            }
            Node* root = out->creator.ptr.get();
            if (!root) continue;
            auto nit = grads.find(root);
            if (nit == grads.end()) grads.emplace(root, seed);
            else accumulate_grad_w2t4(nit->second, seed, create_graph);
            TraversalState tmp;
            topological_sort(root, tmp);
            for (Node* n : tmp.order) {
                if (seen.insert(n).second) order.push_back(n);
            }
        }

        std::unordered_map<Tensor*, std::shared_ptr<Tensor>> input_grads;
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            Node* node = *it;
            auto git = grads.find(node);
            if (git == grads.end() || !git->second) continue;
            std::vector<std::shared_ptr<Tensor>> in_grads = node->backward(git->second);
            const std::vector<NodeInput>& node_inputs = node->inputs;
            const std::vector<std::shared_ptr<Node>>& next_nodes = node->next_nodes;
            for (size_t i = 0; i < node_inputs.size() && i < in_grads.size(); ++i) {
                std::shared_ptr<Tensor> g = in_grads[i];
                if (!g) continue;
                const NodeInput& in = node_inputs[i];
                if (in.requires_grad) {
                    if (auto t = in.tensor.lock()) {
                        for (auto& hook : t->backward_hooks) {
                            std::shared_ptr<Tensor> replaced = hook(g);
                            if (replaced) g = replaced;
                        }
                        accumulate_grad_w2t4(input_grads[t.get()], g, create_graph);
                    }
                }
                if (i < next_nodes.size()) {
                    Node* nn = next_nodes[i].get();
                    if (nn) accumulate_grad_w2t4(grads[nn], g, create_graph);
                }
            }
        }
        if (!retain_graph && !create_graph) {
            for (Node* n : order) {
                n->saved_tensors.clear();
                n->output = SavedTensor();
            }
        }
        result.reserve(inputs.size());
        for (auto& in_t : inputs) {
            if (!in_t) throw std::runtime_error("[litetorch Error] autograd.grad: input is null");
            auto it = input_grads.find(in_t.get());
            if (it == input_grads.end() || !it->second) {
                bool is_output = false;
                for (auto& o : outputs) {
                    if (o.get() == in_t.get()) { is_output = true; break; }
                }
                if (is_output) {
                    result.push_back(Tensor::ones(in_t->shape, in_t->device, false, in_t->dtype));
                } else if (!allow_unused) {
                    throw std::runtime_error("[litetorch Error] autograd.grad: one of the differentiated Tensors was not used in the graph");
                } else {
                    result.push_back(nullptr);
                }
            } else {
                result.push_back(it->second);
            }
        }
    } catch (...) {
        is_create_graph_ = old_create_graph;
        is_grad_enabled_ = old_enabled;
        throw;
    }
    is_create_graph_ = old_create_graph;
    is_grad_enabled_ = old_enabled;
    return result;
}

namespace {
struct FunctionShared {
    std::vector<std::shared_ptr<Tensor>> grad_slots;
    std::atomic<size_t> arrived{0};
};

struct FunctionNode : public Node {
    std::shared_ptr<Function> fn;
    std::shared_ptr<FunctionContext> ctx;
    std::shared_ptr<FunctionShared> shared;
    size_t output_index;
    size_t num_outputs;
    FunctionNode(std::shared_ptr<Function> fn, std::shared_ptr<FunctionContext> ctx,
                 std::shared_ptr<FunctionShared> shared, size_t output_index, size_t num_outputs)
        : Node("Function"), fn(fn), ctx(ctx), shared(shared),
          output_index(output_index), num_outputs(num_outputs) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        shared->grad_slots[output_index] = grad_output;
        size_t n = shared->arrived.fetch_add(1) + 1;
        std::vector<std::shared_ptr<Tensor>> empty(inputs.size(), nullptr);
        if (n < num_outputs) return empty;
        std::vector<std::shared_ptr<Tensor>> go = shared->grad_slots;
        std::vector<std::shared_ptr<Tensor>> out = fn->backward(ctx.get(), go);
        if (out.size() != inputs.size())
            throw std::runtime_error("[litetorch Error] autograd.Function.backward: expected " +
                std::to_string(inputs.size()) + " grads, got " + std::to_string(out.size()));
        return out;
    }
};
}

std::vector<std::shared_ptr<Tensor>> Function::apply(std::shared_ptr<Function> fn,
    const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (!fn) throw std::runtime_error("[litetorch Error] Function.apply: fn is null");
    auto ctx = std::make_shared<FunctionContext>();
    std::vector<bool> needs;
    needs.reserve(inputs.size());
    bool any_requires = false;
    for (auto& t : inputs) {
        if (!t) throw std::runtime_error("[litetorch Error] Function.apply: input is null");
        needs.push_back(t->requires_grad);
        any_requires = any_requires || t->requires_grad;
    }
    ctx->set_needs_input_grad(needs);
    std::vector<std::shared_ptr<Tensor>> outputs = fn->forward(ctx.get(), inputs);
    if (outputs.empty())
        throw std::runtime_error("[litetorch Error] Function.apply: forward() must return a Tensor or a tuple of Tensors");
    for (auto& o : outputs) {
        if (!o) throw std::runtime_error("[litetorch Error] Function.apply: forward() returned null");
    }
    if (Autograd::is_grad_enabled() && any_requires) {
        auto shared = std::make_shared<FunctionShared>();
        shared->grad_slots.assign(outputs.size(), nullptr);
        for (size_t k = 0; k < outputs.size(); ++k) {
            auto node = std::make_shared<FunctionNode>(fn, ctx, shared, k, outputs.size());
            for (auto& t : inputs) {
                node->inputs.push_back({t, t->requires_grad});
                node->next_nodes.push_back(t->creator);
            }
            outputs[k]->creator = node;
            outputs[k]->requires_grad = true;
        }
    }
    return outputs;
}

namespace {
double scalar_sum_w2t4(const std::vector<std::shared_ptr<Tensor>>& outs) {
    double s = 0.0;
    for (auto& o : outs) {
        if (!o) continue;
        if (o->device.type != DeviceType::CPU)
            throw std::runtime_error("[litetorch Error] gradcheck: func must return CPU tensors");
        for (double v : o->flat_doubles()) s += v;
    }
    return s;
}
}

bool autograd_gradcheck(const AutogradFunc& func,
    const std::vector<std::shared_ptr<Tensor>>& inputs,
    double eps, double atol, double rtol, bool raise_exception, std::string& error_out) {
    if (inputs.empty())
        throw std::runtime_error("[litetorch Error] gradcheck: inputs must be non-empty");
    if (!Autograd::is_grad_enabled())
        throw std::runtime_error("[litetorch Error] gradcheck: requires grad enabled");
    for (auto& t : inputs) {
        if (!t) throw std::runtime_error("[litetorch Error] gradcheck: input is null");
        if (t->device.type != DeviceType::CPU)
            throw std::runtime_error("[litetorch Error] gradcheck: only CPU tensors are supported");
        if (t->dtype != DataType::FP32 && t->dtype != DataType::FP64)
            throw std::runtime_error("[litetorch Error] gradcheck: only floating-point dtypes are supported");
        if (!t->requires_grad)
            throw std::runtime_error("[litetorch Error] gradcheck: inputs must require grad");
        if (!t->is_contiguous())
            throw std::runtime_error("[litetorch Error] gradcheck: inputs must be contiguous");
    }
    std::vector<std::shared_ptr<Tensor>> outs0 = func(inputs);
    std::vector<std::shared_ptr<Tensor>> analytical = Autograd::grad(outs0, inputs, {}, false, false, true);

    std::ostringstream failures;
    size_t nfail = 0;
    size_t ntotal = 0;
    for (size_t ti = 0; ti < inputs.size(); ++ti) {
        auto& t = inputs[ti];
        size_t n = t->numel();
        std::vector<double> avec(n, 0.0);
        if (analytical[ti]) {
            std::vector<double> g = analytical[ti]->flat_doubles();
            if (g.size() == n) avec = g;
        }
        bool is_fp64 = (t->dtype == DataType::FP64);
        double* dptr = is_fp64 ? reinterpret_cast<double*>(t->data_ptr()) : nullptr;
        float* fptr = is_fp64 ? nullptr : t->data_ptr();
        for (size_t j = 0; j < n; ++j) {
            double orig = is_fp64 ? dptr[j] : (double)fptr[j];
            if (is_fp64) dptr[j] = orig + eps; else fptr[j] = (float)(orig + eps);
            double fp = scalar_sum_w2t4(func(inputs));
            if (is_fp64) dptr[j] = orig - eps; else fptr[j] = (float)(orig - eps);
            double fm;
            try {
                fm = scalar_sum_w2t4(func(inputs));
            } catch (...) {
                if (is_fp64) dptr[j] = orig; else fptr[j] = (float)orig;
                throw;
            }
            if (is_fp64) dptr[j] = orig; else fptr[j] = (float)orig;
            double num = (fp - fm) / (2.0 * eps);
            double a = avec[j];
            double tol = atol + rtol * std::fabs(num);
            ++ntotal;
            if (!(std::fabs(a - num) <= tol)) {
                if (nfail < 8)
                    failures << "input " << ti << " elem " << j << ": analytical " << a << " vs numerical " << num << "; ";
                ++nfail;
            }
        }
    }
    if (nfail > 0) {
        std::ostringstream msg;
        msg << "gradcheck failed for " << nfail << "/" << ntotal << " elements. " << failures.str();
        error_out = msg.str();
        if (raise_exception) throw std::runtime_error("[litetorch Error] gradcheck: " + error_out);
        return false;
    }
    return true;
}

bool autograd_gradgradcheck(const AutogradFunc& func,
    const std::vector<std::shared_ptr<Tensor>>& inputs,
    double eps, double atol, double rtol, bool raise_exception, std::string& error_out) {
    if (inputs.empty())
        throw std::runtime_error("[litetorch Error] gradgradcheck: inputs must be non-empty");
    AutogradFunc g = [&](const std::vector<std::shared_ptr<Tensor>>& xs)
        -> std::vector<std::shared_ptr<Tensor>> {
        std::vector<std::shared_ptr<Tensor>> outs = func(xs);
        std::vector<std::shared_ptr<Tensor>> gs = Autograd::grad(outs, xs, {}, false, true, true);
        std::shared_ptr<Tensor> total;
        for (auto& gr : gs) {
            if (!gr) continue;
            std::shared_ptr<Tensor> s = Ops::sum(gr);
            total = total ? Ops::add(total, s) : s;
        }
        if (!total) total = Tensor::zeros({1}, inputs[0]->device, false, inputs[0]->dtype);
        return {total};
    };
    return autograd_gradcheck(g, inputs, eps, atol, rtol, raise_exception, error_out);
}

}
