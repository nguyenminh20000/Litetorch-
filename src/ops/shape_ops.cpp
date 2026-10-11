#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include "litetorch/backend.h"
#include "litetorch/backend_route.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace litetorch {
namespace Ops {

namespace {

int64_t norm_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::runtime_error("[litetorch Error] Dimension out of range");
    return dim;
}

void write_vec(const std::shared_ptr<Tensor>& out, const std::vector<float>& data) {
    size_t bytes = data.size() * sizeof(float);
    if (out->device.type == DeviceType::GPU) {
        rt_gpu_write(out->gpu_data(), bytes, data.data(), 0);
    } else {
        std::copy(data.begin(), data.end(), out->storage->get_cpu_ptr());
    }
}

std::vector<int64_t> bcast2(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t n = std::max(na, nb);
    std::vector<int64_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t da = (i < n - na) ? 1 : a[i - (n - na)];
        int64_t db = (i < n - nb) ? 1 : b[i - (n - nb)];
        if (da != db && da != 1 && db != 1) throw std::runtime_error("[litetorch Error] Incompatible shapes for broadcasting");
        out[i] = std::max(da, db);
    }
    return out;
}

int64_t bcast_index(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
                    int64_t ndim, const int64_t* coords) {
    int64_t idx = 0;
    int64_t diff = ndim - (int64_t)shape.size();
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t sd = d - diff;
        if (sd < 0) continue;
        int64_t c = (shape[(size_t)sd] == 1) ? 0 : coords[d];
        idx += c * strides[(size_t)sd];
    }
    return idx;
}

void unify_device(std::shared_ptr<Tensor>& a, std::shared_ptr<Tensor>& b) {
    if (a->device != b->device) {
        if (a->device.type == DeviceType::GPU) b = b->to(a->device);
        else a = a->to(b->device);
    }
}

std::shared_ptr<Tensor> vec_to_tensor(const std::vector<float>& data, const std::vector<int64_t>& shape, const Device& dev) {
    auto out = Tensor::create(shape, dev);
    write_vec(out, data);
    return out;
}

}

class PermuteNode : public Node {
public:
    std::vector<int64_t> inv;
    PermuteNode(const std::vector<int64_t>& inv) : Node("Permute"), inv(inv) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        return { permute(grad_output, inv) };
    }
};

std::shared_ptr<Tensor> permute(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims) {
    int64_t ndim = (int64_t)a->shape.size();
    if ((int64_t)dims.size() != ndim) throw std::runtime_error("[litetorch Error] permute: dims size must match tensor ndim");
    std::vector<int64_t> d = dims;
    for (auto& x : d) x = x < 0 ? x + ndim : x;
    std::vector<int64_t> chk = d;
    std::sort(chk.begin(), chk.end());
    for (int64_t i = 0; i < ndim; ++i)
        if (chk[(size_t)i] != i) throw std::runtime_error("[litetorch Error] permute: invalid permutation");
    std::vector<int64_t> new_shape((size_t)ndim), new_strides((size_t)ndim), inv((size_t)ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        new_shape[(size_t)i] = a->shape[(size_t)d[(size_t)i]];
        new_strides[(size_t)i] = a->strides[(size_t)d[(size_t)i]];
        inv[(size_t)d[(size_t)i]] = i;
    }
    auto out = std::make_shared<Tensor>(a->storage, new_shape, new_strides, a->offset, a->device, a->requires_grad);
    if (Autograd::is_grad_enabled() && a->requires_grad) {
        auto node = std::make_shared<PermuteNode>(inv);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class ExpandNode : public Node {
public:
    std::vector<int64_t> orig_shape;
    ExpandNode(const std::vector<int64_t>& s) : Node("Expand"), orig_shape(s) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        return { reduce_broadcast(grad_output, orig_shape) };
    }
};

std::shared_ptr<Tensor> expand(std::shared_ptr<Tensor> a, const std::vector<int64_t>& shape) {
    int64_t ndim = (int64_t)shape.size();
    int64_t adim = (int64_t)a->shape.size();
    if (ndim < adim) throw std::runtime_error("[litetorch Error] expand: target ndim smaller than input ndim");
    std::vector<int64_t> new_strides((size_t)ndim);
    int64_t diff = ndim - adim;
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t ai = i - diff;
        int64_t asz = ai < 0 ? 1 : a->shape[(size_t)ai];
        int64_t astr = ai < 0 ? 0 : a->strides[(size_t)ai];
        int64_t ssz = shape[(size_t)i];
        if (ssz != asz && asz != 1) throw std::runtime_error("[litetorch Error] expand: incompatible size");
        new_strides[(size_t)i] = (ssz == asz) ? astr : 0;
    }
    auto out = std::make_shared<Tensor>(a->storage, shape, new_strides, a->offset, a->device, a->requires_grad);
    if (Autograd::is_grad_enabled() && a->requires_grad) {
        auto node = std::make_shared<ExpandNode>(a->shape);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> stack(const std::vector<std::shared_ptr<Tensor>>& tensors, int64_t dim) {
    if (tensors.empty()) throw std::runtime_error("[litetorch Error] stack: empty tensor list");
    int64_t ndim = (int64_t)tensors[0]->shape.size();
    int64_t d = dim < 0 ? dim + ndim + 1 : dim;
    if (d < 0 || d > ndim) throw std::runtime_error("[litetorch Error] stack: dimension out of range");
    std::vector<std::shared_ptr<Tensor>> us;
    us.reserve(tensors.size());
    for (auto& t : tensors) us.push_back(unsqueeze(t, d));
    return cat(us, d);
}

class GatherNode : public Node {
public:
    std::vector<int64_t> input_shape;
    int64_t dim;
    GatherNode(int64_t dim, const std::vector<int64_t>& s) : Node("Gather"), input_shape(s), dim(dim) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto index = saved_tensors[0];
        int64_t ndim = (int64_t)input_shape.size();
        auto in_strides = default_strides(input_shape);
        std::vector<float> g = grad_output->to_vector();
        std::vector<float> idx = index->to_vector();
        std::vector<int64_t> out_shape = grad_output->shape;
        size_t gi_n = 1;
        for (auto s : input_shape) gi_n *= (size_t)s;
        std::vector<float> gi(gi_n, 0.0f);
        size_t n = g.size();
        int64_t dim_size = input_shape[(size_t)dim];
        for (size_t i = 0; i < n; ++i) {
            size_t rem = i;
            int64_t pos = 0;
            for (int64_t dd = ndim - 1; dd >= 0; --dd) {
                int64_t c = (int64_t)(rem % (size_t)out_shape[(size_t)dd]);
                rem /= (size_t)out_shape[(size_t)dd];
                int64_t sc = c;
                if (dd == dim) {
                    int64_t ii = (int64_t)idx[i];
                    if (ii < 0) ii += dim_size;
                    sc = ii;
                }
                pos += sc * in_strides[(size_t)dd];
            }
            gi[(size_t)pos] += g[i];
        }
        auto grad_in_f32 = Tensor::create(input_shape, grad_output->device);
        write_vec(grad_in_f32, gi);
        if (grad_output->dtype == DataType::FP32) {
            return { grad_in_f32 };
        }
        auto grad_in = grad_in_f32->cast(grad_output->dtype);
        return { grad_in };
    }
};

std::shared_ptr<Tensor> gather(std::shared_ptr<Tensor> input, int64_t dim, std::shared_ptr<Tensor> index) {
    int64_t ndim = (int64_t)input->shape.size();
    int64_t d = norm_dim(dim, ndim);
    if ((int64_t)index->shape.size() != ndim) throw std::runtime_error("[litetorch Error] gather: index must have same ndim as input");
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != d && index->shape[(size_t)i] > input->shape[(size_t)i])
            throw std::runtime_error("[litetorch Error] gather: index size out of bounds");
    }
    std::vector<float> in_data = input->to_vector();
    std::vector<float> idx_data = index->to_vector();
    std::vector<int64_t> in_strides = default_strides(input->shape);
    std::vector<int64_t> out_shape = index->shape;
    size_t n = idx_data.size();
    int64_t dim_size = input->shape[(size_t)d];
    for (size_t i = 0; i < n; ++i) {
        int64_t ii = (int64_t)idx_data[i];
        if (ii < 0) ii += dim_size;
        if (ii < 0 || ii >= dim_size) throw std::runtime_error("[litetorch Error] gather: index out of bounds");
        idx_data[i] = (float)ii;
    }
    std::vector<float> out_data(n);
    ThreadPool::get().parallel_for(0, (int64_t)n, [&](int64_t i) {
        size_t rem = (size_t)i;
        int64_t pos = 0;
        for (int64_t dd = ndim - 1; dd >= 0; --dd) {
            int64_t c = (int64_t)(rem % (size_t)out_shape[(size_t)dd]);
            rem /= (size_t)out_shape[(size_t)dd];
            int64_t sc = (dd == d) ? (int64_t)idx_data[(size_t)i] : c;
            pos += sc * in_strides[(size_t)dd];
        }
        out_data[(size_t)i] = in_data[(size_t)pos];
    });
    auto out = Tensor::create(out_shape, input->device);
    write_vec(out, out_data);
    if (Autograd::is_grad_enabled() && input->requires_grad) {
        auto node = std::make_shared<GatherNode>(d, input->shape);
        node->inputs = { {input, true} };
        node->next_nodes = { input->creator };
        node->saved_tensors = { index };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

static void argmax_impl(const std::vector<float>& data, const std::vector<int64_t>& shape,
                        int64_t dim, bool keepdim, bool want_min,
                        std::vector<float>& out_vals, std::vector<float>& out_idx,
                        std::vector<int64_t>& out_shape) {
    int64_t ndim = (int64_t)shape.size();
    auto in_strides = default_strides(shape);
    std::vector<int64_t> rdims;
    for (int64_t i = 0; i < ndim; ++i) if (i != dim) rdims.push_back(i);
    out_shape.clear();
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(shape[(size_t)i]);
    }
    size_t n = 1;
    for (auto s : out_shape) n *= (size_t)s;
    out_vals.assign(n, 0.0f);
    out_idx.assign(n, 0.0f);
    int64_t dim_size = shape[(size_t)dim];
    int64_t m = (int64_t)rdims.size();
    ThreadPool::get().parallel_for(0, (int64_t)n, [&](int64_t o) {
        int64_t rem = o;
        int64_t base = 0;
        for (int64_t j = m - 1; j >= 0; --j) {
            int64_t dd = rdims[(size_t)j];
            int64_t c = rem % shape[(size_t)dd];
            rem /= shape[(size_t)dd];
            base += c * in_strides[(size_t)dd];
        }
        float best = data[(size_t)base];
        int64_t bi = 0;
        for (int64_t k = 1; k < dim_size; ++k) {
            float v = data[(size_t)(base + k * in_strides[(size_t)dim])];
            if ((want_min && v < best) || (!want_min && v > best)) { best = v; bi = k; }
        }
        out_vals[(size_t)o] = best;
        out_idx[(size_t)o] = (float)bi;
    });
}

std::shared_ptr<Tensor> argmax(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim) {
    int64_t d = norm_dim(dim, (int64_t)a->shape.size());
    std::vector<float> data = a->to_vector();
    std::vector<float> vals, idx;
    std::vector<int64_t> out_shape;
    argmax_impl(data, a->shape, d, keepdim, false, vals, idx, out_shape);
    auto out = Tensor::create(out_shape, a->device);
    write_vec(out, idx);
    return out;
}

std::shared_ptr<Tensor> argmin(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim) {
    int64_t d = norm_dim(dim, (int64_t)a->shape.size());
    std::vector<float> data = a->to_vector();
    std::vector<float> vals, idx;
    std::vector<int64_t> out_shape;
    argmax_impl(data, a->shape, d, keepdim, true, vals, idx, out_shape);
    auto out = Tensor::create(out_shape, a->device);
    write_vec(out, idx);
    return out;
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> topk(std::shared_ptr<Tensor> a, int64_t k, int64_t dim, bool largest, bool sorted) {
    int64_t ndim = (int64_t)a->shape.size();
    int64_t d = norm_dim(dim, ndim);
    int64_t dim_size = a->shape[(size_t)d];
    if (k <= 0 || k > dim_size) throw std::runtime_error("[litetorch Error] topk: k out of range");
    std::vector<float> data = a->to_vector();
    auto in_strides = default_strides(a->shape);
    std::vector<int64_t> rdims;
    for (int64_t i = 0; i < ndim; ++i) if (i != d) rdims.push_back(i);
    std::vector<int64_t> out_shape = a->shape;
    out_shape[(size_t)d] = k;
    size_t n = 1;
    for (auto s : out_shape) n *= (size_t)s;
    std::vector<float> out_vals(n), out_idx(n);
    int64_t m = (int64_t)rdims.size();
    for (int64_t o = 0; o < (int64_t)n / k; ++o) {
        int64_t rem = o;
        int64_t base = 0;
        for (int64_t j = m - 1; j >= 0; --j) {
            int64_t dd = rdims[(size_t)j];
            int64_t c = rem % a->shape[(size_t)dd];
            rem /= a->shape[(size_t)dd];
            base += c * in_strides[(size_t)dd];
        }
        std::vector<int64_t> order((size_t)dim_size);
        std::iota(order.begin(), order.end(), 0);
        auto cmp = [&](int64_t x, int64_t y) {
            float vx = data[(size_t)(base + x * in_strides[(size_t)d])];
            float vy = data[(size_t)(base + y * in_strides[(size_t)d])];
            return largest ? (vx > vy) : (vx < vy);
        };
        if (k < dim_size) std::nth_element(order.begin(), order.begin() + k, order.end(), cmp);
        if (sorted) std::sort(order.begin(), order.begin() + k, cmp);
        for (int64_t j = 0; j < k; ++j) {
            int64_t kk = order[(size_t)j];
            out_vals[(size_t)(o * k + j)] = data[(size_t)(base + kk * in_strides[(size_t)d])];
            out_idx[(size_t)(o * k + j)] = (float)kk;
        }
    }
    auto vals = Tensor::create(out_shape, a->device);
    auto idx = Tensor::create(out_shape, a->device);
    write_vec(vals, out_vals);
    write_vec(idx, out_idx);
    return { vals, idx };
}

static std::shared_ptr<Tensor> cmp_op(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, int op) {
    unify_device(a, b);
    auto ac = a->is_contiguous() ? a : a->contiguous();
    auto bc = b->is_contiguous() ? b : b->contiguous();
    std::vector<int64_t> out_shape = bcast2(ac->shape, bc->shape);
    int64_t ndim = (int64_t)out_shape.size();
    std::vector<int64_t> out_strides = default_strides(out_shape);
    size_t n = 1;
    for (auto s : out_shape) n *= (size_t)s;
    float* ap = ac->data_ptr();
    float* bp = bc->data_ptr();
    std::vector<float> out_data(n);
    std::vector<int64_t> coords((size_t)ndim);
    for (size_t i = 0; i < n; ++i) {
        size_t rem = i;
        for (int64_t dd = ndim - 1; dd >= 0; --dd) {
            coords[(size_t)dd] = (int64_t)(rem % (size_t)out_shape[(size_t)dd]);
            rem /= (size_t)out_shape[(size_t)dd];
        }
        float x = ap[bcast_index(ac->shape, ac->strides, ndim, coords.data())];
        float y = bp[bcast_index(bc->shape, bc->strides, ndim, coords.data())];
        bool r = false;
        switch (op) {
            case 0: r = (x == y); break;
            case 1: r = (x != y); break;
            case 2: r = (x < y); break;
            case 3: r = (x > y); break;
            case 4: r = (x <= y); break;
            default: r = (x >= y); break;
        }
        out_data[i] = r ? 1.0f : 0.0f;
    }
    auto out = Tensor::create(out_shape, ac->device);
    if (out->device.type == DeviceType::GPU) {
        rt_gpu_write(out->gpu_data(), n * sizeof(float), out_data.data(), 0);
    } else {
        std::copy(out_data.begin(), out_data.end(), out->storage->get_cpu_ptr());
    }
    return out;
}

std::shared_ptr<Tensor> eq(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return cmp_op(a, b, 0); }
std::shared_ptr<Tensor> ne(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return cmp_op(a, b, 1); }
std::shared_ptr<Tensor> lt(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return cmp_op(a, b, 2); }
std::shared_ptr<Tensor> gt(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return cmp_op(a, b, 3); }
std::shared_ptr<Tensor> le(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return cmp_op(a, b, 4); }
std::shared_ptr<Tensor> ge(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return cmp_op(a, b, 5); }

class WhereNode : public Node {
public:
    std::vector<int64_t> shape_a;
    std::vector<int64_t> shape_b;
    std::vector<int64_t> out_shape;
    WhereNode(const std::vector<int64_t>& sa, const std::vector<int64_t>& sb, const std::vector<int64_t>& so)
        : Node("Where"), shape_a(sa), shape_b(sb), out_shape(so) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto cond = saved_tensors[0];
        std::vector<float> g = grad_output->to_vector();
        std::vector<float> c = cond->to_vector();
        int64_t ndim = (int64_t)out_shape.size();
        auto c_strides = default_strides(cond->shape);
        size_t n = g.size();
        std::vector<float> ga(n, 0.0f), gb(n, 0.0f);
        std::vector<int64_t> coords((size_t)ndim);
        for (size_t i = 0; i < n; ++i) {
            size_t rem = i;
            for (int64_t dd = ndim - 1; dd >= 0; --dd) {
                coords[(size_t)dd] = (int64_t)(rem % (size_t)out_shape[(size_t)dd]);
                rem /= (size_t)out_shape[(size_t)dd];
            }
            float cv = c[(size_t)bcast_index(cond->shape, c_strides, ndim, coords.data())];
            if (cv != 0.0f) ga[i] = g[i];
            else gb[i] = g[i];
        }
        auto ga_t = vec_to_tensor(ga, out_shape, grad_output->device);
        auto gb_t = vec_to_tensor(gb, out_shape, grad_output->device);
        return { reduce_broadcast(ga_t, shape_a), reduce_broadcast(gb_t, shape_b) };
    }
};

std::shared_ptr<Tensor> where(std::shared_ptr<Tensor> cond, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    unify_device(a, b);
    if (cond->device != a->device) cond = cond->to(a->device);
    auto cc = cond->is_contiguous() ? cond : cond->contiguous();
    auto ac = a->is_contiguous() ? a : a->contiguous();
    auto bc = b->is_contiguous() ? b : b->contiguous();
    std::vector<int64_t> out_shape = bcast2(bcast2(cc->shape, ac->shape), bc->shape);
    int64_t ndim = (int64_t)out_shape.size();
    size_t n = 1;
    for (auto s : out_shape) n *= (size_t)s;
    float* cp = cc->data_ptr();
    float* ap = ac->data_ptr();
    float* bp = bc->data_ptr();
    std::vector<float> out_data(n);
    std::vector<int64_t> coords((size_t)ndim);
    for (size_t i = 0; i < n; ++i) {
        size_t rem = i;
        for (int64_t dd = ndim - 1; dd >= 0; --dd) {
            coords[(size_t)dd] = (int64_t)(rem % (size_t)out_shape[(size_t)dd]);
            rem /= (size_t)out_shape[(size_t)dd];
        }
        float cv = cp[bcast_index(cc->shape, cc->strides, ndim, coords.data())];
        out_data[i] = (cv != 0.0f)
            ? ap[bcast_index(ac->shape, ac->strides, ndim, coords.data())]
            : bp[bcast_index(bc->shape, bc->strides, ndim, coords.data())];
    }
    auto out = Tensor::create(out_shape, ac->device);
    write_vec(out, out_data);
    if (Autograd::is_grad_enabled() && (a->requires_grad || b->requires_grad)) {
        auto node = std::make_shared<WhereNode>(a->shape, b->shape, out_shape);
        node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
        node->next_nodes = { a->creator, b->creator };
        node->saved_tensors = { cc };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class LogSoftmaxNode : public Node {
public:
    int64_t dim;
    LogSoftmaxNode(int64_t dim) : Node("LogSoftmax"), dim(dim) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto out = saved_tensors[0];
        auto s = sum(grad_output, dim, true);
        return { sub(grad_output, mul(exp(out), s)) };
    }
};

std::shared_ptr<Tensor> log_softmax(std::shared_ptr<Tensor> a, int64_t dim) {
    int64_t ndim = (int64_t)a->shape.size();
    int64_t d = norm_dim(dim, ndim);
    std::vector<float> data = a->to_vector();
    auto in_strides = default_strides(a->shape);
    std::vector<int64_t> rdims;
    for (int64_t i = 0; i < ndim; ++i) if (i != d) rdims.push_back(i);
    size_t n = data.size();
    std::vector<float> out_data(n);
    int64_t dim_size = a->shape[(size_t)d];
    int64_t m = (int64_t)rdims.size();
    int64_t outer = 1;
    for (auto r : rdims) outer *= a->shape[(size_t)r];
    ThreadPool::get().parallel_for(0, outer, [&](int64_t o) {
        int64_t rem = o;
        int64_t base = 0;
        for (int64_t j = m - 1; j >= 0; --j) {
            int64_t dd = rdims[(size_t)j];
            int64_t c = rem % a->shape[(size_t)dd];
            rem /= a->shape[(size_t)dd];
            base += c * in_strides[(size_t)dd];
        }
        float mx = data[(size_t)base];
        for (int64_t k = 1; k < dim_size; ++k) {
            float v = data[(size_t)(base + k * in_strides[(size_t)d])];
            if (v > mx) mx = v;
        }
        double se = 0.0;
        for (int64_t k = 0; k < dim_size; ++k)
            se += std::exp((double)data[(size_t)(base + k * in_strides[(size_t)d])] - mx);
        float lse = (float)std::log(se);
        for (int64_t k = 0; k < dim_size; ++k)
            out_data[(size_t)(base + k * in_strides[(size_t)d])] = data[(size_t)(base + k * in_strides[(size_t)d])] - mx - lse;
    });
    auto out = Tensor::create(a->shape, a->device);
    write_vec(out, out_data);
    if (Autograd::is_grad_enabled() && a->requires_grad) {
        auto node = std::make_shared<LogSoftmaxNode>(d);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { out };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class SiluNode : public Node {
public:
    SiluNode() : Node("Silu") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto s = saved_tensors[0];
        auto x = saved_tensors[1];
        auto one = Tensor::ones(s->shape, s->device);
        auto ds = mul(s, add(one, mul(x, sub(one, s))));
        return { mul(grad_output, ds) };
    }
};

std::shared_ptr<Tensor> silu(std::shared_ptr<Tensor> a) {
    auto ac = a->is_contiguous() ? a : a->contiguous();
    std::vector<float> data = ac->to_vector();
    size_t n = data.size();
    std::vector<float> out_data(n);
    std::vector<float> sig_data(n);
    ThreadPool::get().parallel_for(0, (int64_t)n, [&](int64_t i) {
        float s = 1.0f / (1.0f + std::exp(-data[(size_t)i]));
        sig_data[(size_t)i] = s;
        out_data[(size_t)i] = data[(size_t)i] * s;
    });
    auto out = Tensor::create(ac->shape, ac->device);
    write_vec(out, out_data);
    if (Autograd::is_grad_enabled() && a->requires_grad) {
        auto s = Tensor::create(ac->shape, ac->device);
        write_vec(s, sig_data);
        auto node = std::make_shared<SiluNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { s, ac };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

}
}
