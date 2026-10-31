#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include "litetorch/backend.h"
#include "litetorch/backend_route.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace litetorch {

std::mt19937& get_generator();

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

void write_vec_i64(const std::shared_ptr<Tensor>& out, const std::vector<int64_t>& data) {
    size_t bytes = data.size() * sizeof(int64_t);
    if (out->device.type == DeviceType::GPU) {
        rt_gpu_write(out->gpu_data(), bytes, data.data(), 0);
    } else {
        int64_t* dst = reinterpret_cast<int64_t*>(out->storage->get_cpu_ptr()) + out->offset;
        std::copy(data.begin(), data.end(), dst);
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

std::vector<int64_t> read_index_int64(std::shared_ptr<Tensor> index);

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
        std::vector<int64_t> idx = read_index_int64(index);
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
                    int64_t ii = idx[i];
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

std::vector<int64_t> read_index_int64(std::shared_ptr<Tensor> index) {
    auto idx_c = index->is_contiguous() ? index : index->contiguous();
    size_t n = (size_t)idx_c->numel();
    std::vector<int64_t> out(n);
    if (idx_c->dtype == DataType::INT64) {
        const int64_t* p = reinterpret_cast<const int64_t*>(idx_c->raw_data_ptr());
        for (size_t i = 0; i < n; ++i) out[i] = p[i];
    } else if (idx_c->dtype == DataType::INT32) {
        const int32_t* p = reinterpret_cast<const int32_t*>(idx_c->raw_data_ptr());
        for (size_t i = 0; i < n; ++i) out[i] = (int64_t)p[i];
    } else {
        std::vector<float> v = idx_c->to_vector();
        for (size_t i = 0; i < n; ++i) out[i] = (int64_t)v[i];
    }
    return out;
}

std::shared_ptr<Tensor> gather(std::shared_ptr<Tensor> input, int64_t dim, std::shared_ptr<Tensor> index) {
    int64_t ndim = (int64_t)input->shape.size();
    int64_t d = norm_dim(dim, ndim);
    if ((int64_t)index->shape.size() != ndim) throw std::runtime_error("[litetorch Error] gather: index must have same ndim as input");
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != d && index->shape[(size_t)i] > input->shape[(size_t)i])
            throw std::runtime_error("[litetorch Error] gather: index size out of bounds");
    }
    std::vector<float> in_data = input->to_vector();
    std::vector<int64_t> idx_data = read_index_int64(index);
    std::vector<int64_t> in_strides = default_strides(input->shape);
    std::vector<int64_t> out_shape = index->shape;
    size_t n = idx_data.size();
    int64_t dim_size = input->shape[(size_t)d];
    for (size_t i = 0; i < n; ++i) {
        int64_t ii = idx_data[i];
        if (ii < 0) ii += dim_size;
        if (ii < 0 || ii >= dim_size) throw std::runtime_error("[litetorch Error] gather: index out of bounds");
        idx_data[i] = ii;
    }
    std::vector<float> out_data(n);
    ThreadPool::get().parallel_for(0, (int64_t)n, [&](int64_t i) {
        size_t rem = (size_t)i;
        int64_t pos = 0;
        for (int64_t dd = ndim - 1; dd >= 0; --dd) {
            int64_t c = (int64_t)(rem % (size_t)out_shape[(size_t)dd]);
            rem /= (size_t)out_shape[(size_t)dd];
            int64_t sc = (dd == d) ? idx_data[(size_t)i] : c;
            pos += sc * in_strides[(size_t)dd];
        }
        out_data[(size_t)i] = in_data[(size_t)pos];
    });
    auto out_fp32 = Tensor::create(out_shape, input->device);
    write_vec(out_fp32, out_data);
    auto out = (input->dtype == DataType::FP32) ? out_fp32 : out_fp32->cast(input->dtype);
    if (Autograd::is_grad_enabled() && input->requires_grad) {
        auto node = std::make_shared<GatherNode>(d, input->shape);
        node->inputs = { {input, true} };
        node->next_nodes = { input->creator };
        node->saved_tensors = { index };
        node->saved_versions = {(index ? index->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

static void argmax_impl(const std::vector<float>& data, const std::vector<int64_t>& shape,
                        int64_t dim, bool keepdim, bool want_min,
                        std::vector<float>& out_vals, std::vector<int64_t>& out_idx,
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
    out_idx.assign(n, 0);
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
        out_idx[(size_t)o] = bi;
    });
}

std::shared_ptr<Tensor> argmax(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim) {
    int64_t d = norm_dim(dim, (int64_t)a->shape.size());
    std::vector<float> data = a->to_vector();
    std::vector<float> vals;
    std::vector<int64_t> idx;
    std::vector<int64_t> out_shape;
    argmax_impl(data, a->shape, d, keepdim, false, vals, idx, out_shape);
    auto out = Tensor::create(out_shape, a->device, false, DataType::INT64);
    write_vec_i64(out, idx);
    return out;
}

std::shared_ptr<Tensor> argmin(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim) {
    int64_t d = norm_dim(dim, (int64_t)a->shape.size());
    std::vector<float> data = a->to_vector();
    std::vector<float> vals;
    std::vector<int64_t> idx;
    std::vector<int64_t> out_shape;
    argmax_impl(data, a->shape, d, keepdim, true, vals, idx, out_shape);
    auto out = Tensor::create(out_shape, a->device, false, DataType::INT64);
    write_vec_i64(out, idx);
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
    std::vector<float> out_vals(n);
    std::vector<int64_t> out_idx(n);
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
            out_idx[(size_t)(o * k + j)] = kk;
        }
    }
    auto vals_fp32 = Tensor::create(out_shape, a->device);
    auto idx = Tensor::create(out_shape, a->device, false, DataType::INT64);
    write_vec(vals_fp32, out_vals);
    write_vec_i64(idx, out_idx);
    auto vals = (a->dtype == DataType::FP32) ? vals_fp32 : vals_fp32->cast(a->dtype);
    return { vals, idx };
}

static std::shared_ptr<Tensor> cmp_op(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, int op) {
    unify_device(a, b);
    std::vector<float> av = a->to_vector();
    std::vector<float> bv = b->to_vector();
    std::vector<int64_t> a_strides = default_strides(a->shape);
    std::vector<int64_t> b_strides = default_strides(b->shape);
    std::vector<int64_t> out_shape = bcast2(a->shape, b->shape);
    int64_t ndim = (int64_t)out_shape.size();
    std::vector<int64_t> out_strides = default_strides(out_shape);
    size_t n = 1;
    for (auto s : out_shape) n *= (size_t)s;
    std::vector<float> out_data(n);
    std::vector<int64_t> coords((size_t)ndim);
    for (size_t i = 0; i < n; ++i) {
        size_t rem = i;
        for (int64_t dd = ndim - 1; dd >= 0; --dd) {
            coords[(size_t)dd] = (int64_t)(rem % (size_t)out_shape[(size_t)dd]);
            rem /= (size_t)out_shape[(size_t)dd];
        }
        float x = av[(size_t)bcast_index(a->shape, a_strides, ndim, coords.data())];
        float y = bv[(size_t)bcast_index(b->shape, b_strides, ndim, coords.data())];
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
    auto out = Tensor::create(out_shape, a->device);
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
    DataType dtype_a;
    DataType dtype_b;
    WhereNode(const std::vector<int64_t>& sa, const std::vector<int64_t>& sb, const std::vector<int64_t>& so,
              DataType da, DataType db)
        : Node("Where"), shape_a(sa), shape_b(sb), out_shape(so), dtype_a(da), dtype_b(db) {}
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
        auto ra = reduce_broadcast(ga_t, shape_a);
        auto rb = reduce_broadcast(gb_t, shape_b);
        if (ra->dtype != dtype_a) ra = ra->cast(dtype_a);
        if (rb->dtype != dtype_b) rb = rb->cast(dtype_b);
        return { ra, rb };
    }
};

namespace {

std::shared_ptr<Tensor> densify_int64(std::shared_ptr<Tensor> t) {
    auto out = Tensor::create(t->shape, Device(DeviceType::CPU, 0), false, DataType::INT64);
    const size_t esz = 8;
    t->storage->ensure_cpu();
    const char* src_base = reinterpret_cast<const char*>(t->storage->get_cpu_ptr()) + t->offset * esz;
    char* dst_base = reinterpret_cast<char*>(out->storage->get_cpu_ptr());
    int64_t nd = (int64_t)t->shape.size();
    int64_t n = t->numel();
    for (int64_t lin = 0; lin < n; ++lin) {
        int64_t rem = lin;
        int64_t off = 0;
        for (int64_t d = nd - 1; d >= 0; --d) {
            int64_t c = rem % t->shape[(size_t)d];
            rem /= t->shape[(size_t)d];
            off += c * t->strides[(size_t)d];
        }
        std::memcpy(dst_base + lin * esz, src_base + off * esz, esz);
    }
    return out;
}

}

std::shared_ptr<Tensor> where(std::shared_ptr<Tensor> cond, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    unify_device(a, b);
    if (cond->device != a->device) cond = cond->to(a->device);
    DataType out_dtype = promote_types(a->dtype, b->dtype);
    if (is_integer_dtype(out_dtype)) {
        auto ai = cast(a, DataType::INT64);
        auto bi = cast(b, DataType::INT64);
        if (!ai->is_contiguous() || ai->offset != 0) ai = densify_int64(ai);
        if (!bi->is_contiguous() || bi->offset != 0) bi = densify_int64(bi);
        std::vector<float> cv = cond->to_vector();
        const int64_t* av = reinterpret_cast<const int64_t*>(ai->raw_data_ptr());
        const int64_t* bv = reinterpret_cast<const int64_t*>(bi->raw_data_ptr());
        auto c_strides = default_strides(cond->shape);
        auto a_strides = default_strides(ai->shape);
        auto b_strides = default_strides(bi->shape);
        std::vector<int64_t> out_shape = bcast2(bcast2(cond->shape, ai->shape), bi->shape);
        int64_t ndim = (int64_t)out_shape.size();
        size_t n = 1;
        for (auto s : out_shape) n *= (size_t)s;
        std::vector<int64_t> out_data(n);
        std::vector<int64_t> coords((size_t)ndim);
        for (size_t i = 0; i < n; ++i) {
            size_t rem = i;
            for (int64_t dd = ndim - 1; dd >= 0; --dd) {
                coords[(size_t)dd] = (int64_t)(rem % (size_t)out_shape[(size_t)dd]);
                rem /= (size_t)out_shape[(size_t)dd];
            }
            float c = cv[(size_t)bcast_index(cond->shape, c_strides, ndim, coords.data())];
            out_data[i] = (c != 0.0f)
                ? av[(size_t)bcast_index(ai->shape, a_strides, ndim, coords.data())]
                : bv[(size_t)bcast_index(bi->shape, b_strides, ndim, coords.data())];
        }
        auto out_i64 = Tensor::create(out_shape, a->device, false, DataType::INT64);
        write_vec_i64(out_i64, out_data);
        auto out = cast(out_i64, out_dtype);
        if (Autograd::is_grad_enabled() && (a->requires_grad || b->requires_grad)) {
            auto node = std::make_shared<WhereNode>(ai->shape, bi->shape, out_shape, ai->dtype, bi->dtype);
            node->inputs = { {ai, ai->requires_grad}, {bi, bi->requires_grad} };
            node->next_nodes = { ai->creator, bi->creator };
            node->saved_tensors = { cond };
            node->saved_versions = {(cond ? cond->_version() : 0)};
            node->output = out;
            out->creator = node;
            out->requires_grad = true;
        }
        return out;
    }
    std::vector<float> cv = cond->to_vector();
    std::vector<float> av = a->to_vector();
    std::vector<float> bv = b->to_vector();
    auto c_strides = default_strides(cond->shape);
    auto a_strides = default_strides(a->shape);
    auto b_strides = default_strides(b->shape);
    std::vector<int64_t> out_shape = bcast2(bcast2(cond->shape, a->shape), b->shape);
    int64_t ndim = (int64_t)out_shape.size();
    size_t n = 1;
    for (auto s : out_shape) n *= (size_t)s;
    std::vector<float> out_data(n);
    std::vector<int64_t> coords((size_t)ndim);
    for (size_t i = 0; i < n; ++i) {
        size_t rem = i;
        for (int64_t dd = ndim - 1; dd >= 0; --dd) {
            coords[(size_t)dd] = (int64_t)(rem % (size_t)out_shape[(size_t)dd]);
            rem /= (size_t)out_shape[(size_t)dd];
        }
        float c = cv[(size_t)bcast_index(cond->shape, c_strides, ndim, coords.data())];
        out_data[i] = (c != 0.0f)
            ? av[(size_t)bcast_index(a->shape, a_strides, ndim, coords.data())]
            : bv[(size_t)bcast_index(b->shape, b_strides, ndim, coords.data())];
    }
    auto out_fp32 = Tensor::create(out_shape, a->device);
    write_vec(out_fp32, out_data);
    auto out = (out_dtype == DataType::FP32) ? out_fp32 : out_fp32->cast(out_dtype);
    if (Autograd::is_grad_enabled() && (a->requires_grad || b->requires_grad)) {
        auto node = std::make_shared<WhereNode>(a->shape, b->shape, out_shape, a->dtype, b->dtype);
        node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
        node->next_nodes = { a->creator, b->creator };
        node->saved_tensors = { cond };
        node->saved_versions = {(cond ? cond->_version() : 0)};
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
    auto out_fp32 = Tensor::create(a->shape, a->device);
    write_vec(out_fp32, out_data);
    auto out = (a->dtype == DataType::FP32) ? out_fp32 : out_fp32->cast(a->dtype);
    if (Autograd::is_grad_enabled() && a->requires_grad) {
        auto node = std::make_shared<LogSoftmaxNode>(d);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { out };
        node->saved_versions = {(out ? out->_version() : 0)};
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
        auto one = Tensor::ones(s->shape, s->device, false, s->dtype);
        auto ds = mul(s, add(one, mul(x, sub(one, s))));
        auto go = (grad_output->dtype == s->dtype) ? grad_output : grad_output->cast(s->dtype);
        auto g = mul(go, ds);
        if (g->dtype != grad_output->dtype) g = g->cast(grad_output->dtype);
        return { g };
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
    auto out_fp32 = Tensor::create(ac->shape, ac->device);
    write_vec(out_fp32, out_data);
    auto out = (a->dtype == DataType::FP32) ? out_fp32 : out_fp32->cast(a->dtype);
    if (Autograd::is_grad_enabled() && a->requires_grad) {
        auto s = Tensor::create(ac->shape, ac->device);
        write_vec(s, sig_data);
        auto xf = Tensor::create(ac->shape, ac->device);
        write_vec(xf, data);
        auto node = std::make_shared<SiluNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { s, xf };
        node->saved_versions = {(s ? s->_version() : 0), (xf ? xf->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

// === W1-T1 core-ops ===

namespace {

void gpu_fail_shape(const char* op) {
    throw std::runtime_error(std::string("[litetorch Error] ") + op + ": GPU kernel not implemented (fail-loud)");
}

size_t shape_numel(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (auto s : shape) n *= (size_t)s;
    return n;
}

int64_t norm_dim_shape(int64_t dim, int64_t ndim, const char* op) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::runtime_error(std::string("[litetorch Error] ") + op + ": dimension out of range");
    return dim;
}

}

std::shared_ptr<Tensor> from_data(const std::vector<float>& data, const std::vector<int64_t>& shape,
        const Device& device, bool requires_grad, DataType dtype) {
    if (data.size() != shape_numel(shape))
        throw std::runtime_error("[litetorch Error] tensor(): data size does not match shape");
    return Tensor::from_vector(data, shape, device, requires_grad, dtype);
}

std::shared_ptr<Tensor> empty(const std::vector<int64_t>& shape,
        const Device& device, bool requires_grad, DataType dtype) {
    return Tensor::create(shape, device, requires_grad, dtype);
}

std::shared_ptr<Tensor> full(const std::vector<int64_t>& shape, float fill_value,
        const Device& device, bool requires_grad, DataType dtype) {
    return Tensor::from_vector(std::vector<float>(shape_numel(shape), fill_value), shape, device, requires_grad, dtype);
}

std::shared_ptr<Tensor> linspace(float start, float end, int64_t steps,
        const Device& device, DataType dtype) {
    if (steps <= 0) throw std::runtime_error("[litetorch Error] linspace: steps must be positive");
    std::vector<float> data((size_t)steps);
    if (steps == 1) {
        data[0] = start;
    } else {
        for (int64_t i = 0; i < steps; ++i)
            data[(size_t)i] = start + (end - start) * (float)i / (float)(steps - 1);
    }
    return Tensor::from_vector(data, {steps}, device, false, dtype);
}

std::shared_ptr<Tensor> logspace(float start, float end, int64_t steps, float base,
        const Device& device, DataType dtype) {
    if (steps <= 0) throw std::runtime_error("[litetorch Error] logspace: steps must be positive");
    std::vector<float> data((size_t)steps);
    if (steps == 1) {
        data[0] = std::pow(base, start);
    } else {
        for (int64_t i = 0; i < steps; ++i)
            data[(size_t)i] = std::pow(base, start + (end - start) * (float)i / (float)(steps - 1));
    }
    return Tensor::from_vector(data, {steps}, device, false, dtype);
}

std::shared_ptr<Tensor> eye(int64_t n, int64_t m, const Device& device, DataType dtype) {
    if (n <= 0) throw std::runtime_error("[litetorch Error] eye: n must be positive");
    if (m < 0) m = n;
    std::vector<float> data((size_t)(n * m), 0.0f);
    int64_t k = std::min(n, m);
    for (int64_t i = 0; i < k; ++i) data[(size_t)(i * m + i)] = 1.0f;
    return Tensor::from_vector(data, {n, m}, device, false, dtype);
}

std::shared_ptr<Tensor> rand(const std::vector<int64_t>& shape,
        const Device& device, bool requires_grad, DataType dtype) {
    std::vector<float> data(shape_numel(shape));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    auto& gen = get_generator();
    for (auto& v : data) v = dist(gen);
    return Tensor::from_vector(data, shape, device, requires_grad, dtype);
}

std::shared_ptr<Tensor> randint(int64_t low, int64_t high, const std::vector<int64_t>& shape,
        const Device& device, DataType dtype) {
    if (high <= low) throw std::runtime_error("[litetorch Error] randint: high must be greater than low");
    std::vector<float> data(shape_numel(shape));
    std::uniform_int_distribution<int64_t> dist(low, high - 1);
    auto& gen = get_generator();
    for (auto& v : data) v = (float)dist(gen);
    return Tensor::from_vector(data, shape, device, false, dtype);
}

std::shared_ptr<Tensor> randperm(int64_t n, const Device& device, DataType dtype) {
    if (n < 0) throw std::runtime_error("[litetorch Error] randperm: n must be non-negative");
    std::vector<float> data((size_t)n);
    std::iota(data.begin(), data.end(), 0.0f);
    std::shuffle(data.begin(), data.end(), get_generator());
    return Tensor::from_vector(data, {n}, device, false, dtype);
}

std::shared_ptr<Tensor> zeros_like(std::shared_ptr<Tensor> a, const Device& device, DataType dtype) {
    return Tensor::zeros(a->shape, device, a->requires_grad, dtype);
}

std::shared_ptr<Tensor> ones_like(std::shared_ptr<Tensor> a, const Device& device, DataType dtype) {
    return Tensor::ones(a->shape, device, a->requires_grad, dtype);
}

std::shared_ptr<Tensor> empty_like(std::shared_ptr<Tensor> a, const Device& device, DataType dtype) {
    return Tensor::create(a->shape, device, a->requires_grad, dtype);
}

std::shared_ptr<Tensor> full_like(std::shared_ptr<Tensor> a, float fill_value, const Device& device, DataType dtype) {
    return full(a->shape, fill_value, device, a->requires_grad, dtype);
}

std::shared_ptr<Tensor> rand_like(std::shared_ptr<Tensor> a, const Device& device, DataType dtype) {
    return rand(a->shape, device, a->requires_grad, dtype);
}

std::shared_ptr<Tensor> randn_like(std::shared_ptr<Tensor> a, const Device& device, DataType dtype) {
    return Tensor::randn(a->shape, device, a->requires_grad, dtype);
}

class TriuNode : public Node {
public:
    int64_t diagonal;
    TriuNode(int64_t d) : Node("Triu"), diagonal(d) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        return { triu(grad_output, diagonal) };
    }
};

class TrilNode : public Node {
public:
    int64_t diagonal;
    TrilNode(int64_t d) : Node("Tril"), diagonal(d) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        return { tril(grad_output, diagonal) };
    }
};

namespace {

std::shared_ptr<Tensor> triu_tril_impl(std::shared_ptr<Tensor> a, int64_t diagonal, bool upper, const char* op) {
    if (a->device.type == DeviceType::GPU) gpu_fail_shape(op);
    int64_t ndim = (int64_t)a->shape.size();
    if (ndim < 2) throw std::runtime_error(std::string("[litetorch Error] ") + op + ": expected at least 2D tensor");
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = upper ? triu(a_fp32, diagonal) : tril(a_fp32, diagonal);
        return cast(out_fp32, a->dtype);
    }
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    std::vector<float> data = a_c->to_vector();
    int64_t rows = a_c->shape[(size_t)(ndim - 2)];
    int64_t cols = a_c->shape[(size_t)(ndim - 1)];
    int64_t plane = rows * cols;
    int64_t batch = (int64_t)a_c->numel() / plane;
    std::vector<float> result(data.size());
    ThreadPool::get().parallel_for(0, batch, [&](int64_t b) {
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                int64_t k = b * plane + i * cols + j;
                bool keep = upper ? ((j - i) >= diagonal) : ((j - i) <= diagonal);
                result[(size_t)k] = keep ? data[(size_t)k] : 0.0f;
            }
        }
    });
    write_vec(out, result);
    if (Autograd::is_grad_enabled() && a->requires_grad) {
        std::shared_ptr<Node> node = upper ? std::static_pointer_cast<Node>(std::make_shared<TriuNode>(diagonal))
                                           : std::static_pointer_cast<Node>(std::make_shared<TrilNode>(diagonal));
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

}

std::shared_ptr<Tensor> triu(std::shared_ptr<Tensor> a, int64_t diagonal) {
    return triu_tril_impl(a, diagonal, true, "triu");
}

std::shared_ptr<Tensor> tril(std::shared_ptr<Tensor> a, int64_t diagonal) {
    return triu_tril_impl(a, diagonal, false, "tril");
}

class SortNode : public Node {
public:
    int64_t dim;
    std::shared_ptr<Tensor> indices;
    std::vector<int64_t> input_shape;
    SortNode(int64_t d, std::shared_ptr<Tensor> idx, const std::vector<int64_t>& s)
        : Node("Sort"), dim(d), indices(idx), input_shape(s) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto zeros = Tensor::zeros(input_shape, grad_output->device, false, DataType::FP32);
        return { scatter_add(zeros, dim, indices, grad_output) };
    }
};

namespace {

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> sort_impl(
        std::shared_ptr<Tensor> a, int64_t dim, bool descending, const char* op) {
    if (a->device.type == DeviceType::GPU) gpu_fail_shape(op);
    int64_t ndim = (int64_t)a->shape.size();
    if (ndim < 1) throw std::runtime_error(std::string("[litetorch Error] ") + op + ": expected at least 1D tensor");
    int64_t d = norm_dim_shape(dim, ndim, op);
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto r = sort_impl(a_fp32, dim, descending, op);
        return { cast(r.first, a->dtype), r.second };
    }
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    std::vector<float> data = a_c->to_vector();
    auto in_strides = default_strides(a_c->shape);
    int64_t dim_size = a_c->shape[(size_t)d];
    size_t n = data.size();
    std::vector<float> out_vals(n);
    std::vector<int64_t> out_idx(n);
    int64_t slices = (int64_t)n / dim_size;
    std::vector<int64_t> rdims;
    for (int64_t i = 0; i < ndim; ++i) if (i != d) rdims.push_back(i);
    ThreadPool::get().parallel_for(0, slices, [&](int64_t o) {
        int64_t rem = o;
        int64_t base = 0;
        for (int64_t j = (int64_t)rdims.size() - 1; j >= 0; --j) {
            int64_t dd = rdims[(size_t)j];
            int64_t c = rem % a_c->shape[(size_t)dd];
            rem /= a_c->shape[(size_t)dd];
            base += c * in_strides[(size_t)dd];
        }
        std::vector<int64_t> order((size_t)dim_size);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int64_t x, int64_t y) {
            float vx = data[(size_t)(base + x * in_strides[(size_t)d])];
            float vy = data[(size_t)(base + y * in_strides[(size_t)d])];
            if (vx != vy) return descending ? (vx > vy) : (vx < vy);
            return x < y;
        });
        int64_t dstride = in_strides[(size_t)d];
        for (int64_t k = 0; k < dim_size; ++k) {
            out_vals[(size_t)(base + k * dstride)] = data[(size_t)(base + order[(size_t)k] * dstride)];
            out_idx[(size_t)(base + k * dstride)] = order[(size_t)k];
        }
    });
    auto vals = Tensor::create(a_c->shape, a_c->device);
    auto idx = Tensor::create(a_c->shape, a_c->device, false, DataType::INT64);
    write_vec(vals, out_vals);
    write_vec_i64(idx, out_idx);
    if (Autograd::is_grad_enabled() && a->requires_grad) {
        auto node = std::make_shared<SortNode>(d, idx, a->shape);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { idx };
        node->saved_versions = {(idx ? idx->_version() : 0)};
        node->output = vals;
        vals->creator = node;
        vals->requires_grad = true;
    }
    return { vals, idx };
}

}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> sort(std::shared_ptr<Tensor> a, int64_t dim, bool descending) {
    return sort_impl(a, dim, descending, "sort");
}

std::shared_ptr<Tensor> argsort(std::shared_ptr<Tensor> a, int64_t dim, bool descending) {
    return sort_impl(a, dim, descending, "argsort").second;
}

class CumsumNode : public Node {
public:
    int64_t dim;
    CumsumNode(int64_t d) : Node("Cumsum"), dim(d) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto g_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        std::vector<float> gd = g_c->to_vector();
        int64_t ndim = (int64_t)g_c->shape.size();
        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < dim; ++i) outer *= g_c->shape[(size_t)i];
        int64_t dim_size = g_c->shape[(size_t)dim];
        for (int64_t i = dim + 1; i < ndim; ++i) inner *= g_c->shape[(size_t)i];
        std::vector<float> gi(gd.size());
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t in = 0; in < inner; ++in) {
                float acc = 0.0f;
                for (int64_t k = dim_size - 1; k >= 0; --k) {
                    acc += gd[(size_t)((o * dim_size + k) * inner + in)];
                    gi[(size_t)((o * dim_size + k) * inner + in)] = acc;
                }
            }
        }
        auto out = Tensor::create(g_c->shape, g_c->device);
        write_vec(out, gi);
        return { out };
    }
};

std::shared_ptr<Tensor> cumsum(std::shared_ptr<Tensor> a, int64_t dim) {
    if (a->device.type == DeviceType::GPU) gpu_fail_shape("cumsum");
    int64_t ndim = (int64_t)a->shape.size();
    if (ndim < 1) throw std::runtime_error("[litetorch Error] cumsum: expected at least 1D tensor");
    int64_t d = norm_dim_shape(dim, ndim, "cumsum");
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        return cast(cumsum(a_fp32, dim), a->dtype);
    }
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    std::vector<float> data = a_c->to_vector();
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < d; ++i) outer *= a_c->shape[(size_t)i];
    int64_t dim_size = a_c->shape[(size_t)d];
    for (int64_t i = d + 1; i < ndim; ++i) inner *= a_c->shape[(size_t)i];
    std::vector<float> result(data.size());
    ThreadPool::get().parallel_for(0, outer, [&](int64_t o) {
        for (int64_t in = 0; in < inner; ++in) {
            float acc = 0.0f;
            for (int64_t k = 0; k < dim_size; ++k) {
                acc += data[(size_t)((o * dim_size + k) * inner + in)];
                result[(size_t)((o * dim_size + k) * inner + in)] = acc;
            }
        }
    });
    auto out = Tensor::create(a_c->shape, a_c->device);
    write_vec(out, result);
    if (Autograd::is_grad_enabled() && a->requires_grad) {
        auto node = std::make_shared<CumsumNode>(d);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class ScatterNode : public Node {
public:
    int64_t dim;
    std::shared_ptr<Tensor> index;
    std::vector<int64_t> input_shape;
    ScatterNode(int64_t d, std::shared_ptr<Tensor> idx, const std::vector<int64_t>& s)
        : Node("Scatter"), dim(d), index(idx), input_shape(s) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto g_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        std::vector<float> gd = g_c->to_vector();
        std::vector<float> idxd = index->to_vector();
        int64_t ndim = (int64_t)input_shape.size();
        auto in_strides = default_strides(input_shape);
        std::vector<float> gi = gd;
        size_t nidx = idxd.size();
        for (size_t i = 0; i < nidx; ++i) {
            size_t rem = i;
            int64_t pos = 0;
            for (int64_t dd = ndim - 1; dd >= 0; --dd) {
                int64_t c = (int64_t)(rem % (size_t)index->shape[(size_t)dd]);
                rem /= (size_t)index->shape[(size_t)dd];
                int64_t sc = (dd == dim) ? (int64_t)idxd[i] : c;
                pos += sc * in_strides[(size_t)dd];
            }
            gi[(size_t)pos] = 0.0f;
        }
        auto grad_input = Tensor::create(input_shape, grad_output->device);
        write_vec(grad_input, gi);
        auto grad_src = gather(grad_output, dim, index);
        return { grad_input, grad_src };
    }
};

class ScatterAddNode : public Node {
public:
    int64_t dim;
    std::shared_ptr<Tensor> index;
    std::vector<int64_t> input_shape;
    ScatterAddNode(int64_t d, std::shared_ptr<Tensor> idx, const std::vector<int64_t>& s)
        : Node("ScatterAdd"), dim(d), index(idx), input_shape(s) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto grad_src = gather(grad_output, dim, index);
        return { grad_output->clone(), grad_src };
    }
};

namespace {

std::shared_ptr<Tensor> scatter_impl(std::shared_ptr<Tensor> input, int64_t dim,
        std::shared_ptr<Tensor> index, std::shared_ptr<Tensor> src, bool add, const char* op) {
    if (input->device.type == DeviceType::GPU || index->device.type == DeviceType::GPU ||
            src->device.type == DeviceType::GPU) gpu_fail_shape(op);
    int64_t ndim = (int64_t)input->shape.size();
    if (ndim < 1) throw std::runtime_error(std::string("[litetorch Error] ") + op + ": expected at least 1D input");
    int64_t d = norm_dim_shape(dim, ndim, op);
    if ((int64_t)index->shape.size() != ndim)
        throw std::runtime_error(std::string("[litetorch Error] ") + op + ": index ndim must match input ndim");
    if (index->shape != src->shape)
        throw std::runtime_error(std::string("[litetorch Error] ") + op + ": index and src shapes must match");
    if (input->dtype != DataType::FP32 || src->dtype != DataType::FP32) {
        auto in_fp32 = cast(input, DataType::FP32);
        auto src_fp32 = cast(src, DataType::FP32);
        auto out_fp32 = scatter_impl(in_fp32, dim, index, src_fp32, add, op);
        return (input->dtype == DataType::FP32) ? out_fp32 : cast(out_fp32, input->dtype);
    }
    if (input->device != src->device)
        throw std::runtime_error(std::string("[litetorch Error] ") + op + ": input and src must be on the same device");
    auto in_c = input->is_contiguous() ? input : input->contiguous();
    auto src_c = src->is_contiguous() ? src : src->contiguous();
    auto out = in_c->clone();
    std::vector<float> out_data = out->to_vector();
    std::vector<float> idx_data = index->to_vector();
    std::vector<float> src_data = src_c->to_vector();
    auto in_strides = default_strides(in_c->shape);
    auto idx_strides = default_strides(index->shape);
    int64_t dim_size = in_c->shape[(size_t)d];
    size_t n = idx_data.size();
    for (size_t i = 0; i < n; ++i) {
        int64_t ii = (int64_t)idx_data[i];
        if (ii < 0) ii += dim_size;
        if (ii < 0 || ii >= dim_size)
            throw std::runtime_error(std::string("[litetorch Error] ") + op + ": index out of bounds");
        size_t rem = i;
        int64_t pos = 0;
        for (int64_t dd = ndim - 1; dd >= 0; --dd) {
            int64_t c = (int64_t)(rem % (size_t)index->shape[(size_t)dd]);
            rem /= (size_t)index->shape[(size_t)dd];
            int64_t sc = (dd == d) ? ii : c;
            pos += sc * in_strides[(size_t)dd];
        }
        if (add) out_data[(size_t)pos] += src_data[i];
        else out_data[(size_t)pos] = src_data[i];
    }
    write_vec(out, out_data);
    if (Autograd::is_grad_enabled() && (input->requires_grad || src->requires_grad)) {
        std::shared_ptr<Node> node = add
            ? std::static_pointer_cast<Node>(std::make_shared<ScatterAddNode>(d, index, in_c->shape))
            : std::static_pointer_cast<Node>(std::make_shared<ScatterNode>(d, index, in_c->shape));
        node->inputs = { {input, input->requires_grad}, {src, src->requires_grad} };
        node->next_nodes = { input->creator, src->creator };
        node->saved_tensors = { index };
        node->saved_versions = {(index ? index->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

}

std::shared_ptr<Tensor> scatter(std::shared_ptr<Tensor> input, int64_t dim,
        std::shared_ptr<Tensor> index, std::shared_ptr<Tensor> src) {
    return scatter_impl(input, dim, index, src, false, "scatter");
}

std::shared_ptr<Tensor> scatter_add(std::shared_ptr<Tensor> input, int64_t dim,
        std::shared_ptr<Tensor> index, std::shared_ptr<Tensor> src) {
    return scatter_impl(input, dim, index, src, true, "scatter_add");
}

class FlattenNode : public Node {
public:
    std::vector<int64_t> input_shape;
    FlattenNode(const std::vector<int64_t>& s) : Node("Flatten"), input_shape(s) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        return { grad_output->reshape(input_shape) };
    }
};

std::shared_ptr<Tensor> flatten(std::shared_ptr<Tensor> a, int64_t start_dim, int64_t end_dim) {
    if (a->device.type == DeviceType::GPU) gpu_fail_shape("flatten");
    int64_t ndim = (int64_t)a->shape.size();
    int64_t s = norm_dim_shape(start_dim, ndim, "flatten");
    int64_t e = norm_dim_shape(end_dim, ndim, "flatten");
    if (s > e) throw std::runtime_error("[litetorch Error] flatten: start_dim must not exceed end_dim");
    std::vector<int64_t> new_shape;
    for (int64_t i = 0; i < s; ++i) new_shape.push_back(a->shape[(size_t)i]);
    int64_t flat = 1;
    for (int64_t i = s; i <= e; ++i) flat *= a->shape[(size_t)i];
    new_shape.push_back(flat);
    for (int64_t i = e + 1; i < ndim; ++i) new_shape.push_back(a->shape[(size_t)i]);
    auto out = a->reshape(new_shape);
    if (Autograd::is_grad_enabled() && a->requires_grad) {
        auto node = std::make_shared<FlattenNode>(a->shape);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class RepeatNode : public Node {
public:
    std::vector<int64_t> in_shape;
    std::vector<int64_t> repeats;
    RepeatNode(const std::vector<int64_t>& s, const std::vector<int64_t>& r)
        : Node("Repeat"), in_shape(s), repeats(r) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto g_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        std::vector<float> gd = g_c->to_vector();
        size_t n_in = shape_numel(in_shape);
        std::vector<float> gi(n_in, 0.0f);
        auto in_strides = default_strides(in_shape);
        auto out_strides = default_strides(g_c->shape);
        int64_t ndim = (int64_t)in_shape.size();
        for (size_t i = 0; i < gd.size(); ++i) {
            size_t rem = i;
            int64_t pos = 0;
            for (int64_t dd = ndim - 1; dd >= 0; --dd) {
                int64_t c = (int64_t)(rem % (size_t)g_c->shape[(size_t)dd]);
                rem /= (size_t)g_c->shape[(size_t)dd];
                pos += (c % in_shape[(size_t)dd]) * in_strides[(size_t)dd];
            }
            gi[(size_t)pos] += gd[i];
        }
        auto out = Tensor::create(in_shape, g_c->device);
        write_vec(out, gi);
        return { out };
    }
};

std::shared_ptr<Tensor> repeat(std::shared_ptr<Tensor> a, const std::vector<int64_t>& repeats) {
    if (a->device.type == DeviceType::GPU) gpu_fail_shape("repeat");
    std::vector<int64_t> sh = a->shape;
    std::vector<int64_t> rep = repeats;
    if (rep.size() < sh.size())
        throw std::runtime_error("[litetorch Error] repeat: repeats must have at least as many dims as input");
    while (sh.size() < rep.size()) sh.insert(sh.begin(), 1);
    for (auto r : rep) if (r <= 0) throw std::runtime_error("[litetorch Error] repeat: repeats must be positive");
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        return cast(repeat(a_fp32, repeats), a->dtype);
    }
    int64_t ndim = (int64_t)sh.size();
    std::vector<int64_t> out_shape((size_t)ndim);
    for (int64_t i = 0; i < ndim; ++i) out_shape[(size_t)i] = sh[(size_t)i] * rep[(size_t)i];
    std::vector<float> data = a->to_vector();
    size_t n_out = shape_numel(out_shape);
    std::vector<float> result(n_out);
    auto in_strides = default_strides(sh);
    auto out_strides = default_strides(out_shape);
    ThreadPool::get().parallel_for(0, (int64_t)n_out, [&](int64_t i) {
        size_t rem = (size_t)i;
        int64_t pos = 0;
        for (int64_t dd = ndim - 1; dd >= 0; --dd) {
            int64_t c = (int64_t)(rem % (size_t)out_shape[(size_t)dd]);
            rem /= (size_t)out_shape[(size_t)dd];
            pos += (c % sh[(size_t)dd]) * in_strides[(size_t)dd];
        }
        result[(size_t)i] = data[(size_t)pos];
    });
    auto out = Tensor::create(out_shape, a->device);
    write_vec(out, result);
    if (Autograd::is_grad_enabled() && a->requires_grad) {
        auto node = std::make_shared<RepeatNode>(sh, rep);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class NarrowNode : public Node {
public:
    int64_t dim, start, length;
    std::vector<int64_t> input_shape;
    NarrowNode(int64_t d, int64_t s, int64_t l, const std::vector<int64_t>& ish)
        : Node("Narrow"), dim(d), start(s), length(l), input_shape(ish) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto g_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        std::vector<float> gd = g_c->to_vector();
        std::vector<float> gi(shape_numel(input_shape), 0.0f);
        int64_t ndim = (int64_t)input_shape.size();
        auto in_strides = default_strides(input_shape);
        auto g_strides = default_strides(g_c->shape);
        for (size_t i = 0; i < gd.size(); ++i) {
            size_t rem = i;
            int64_t pos = 0;
            for (int64_t dd = ndim - 1; dd >= 0; --dd) {
                int64_t c = (int64_t)(rem % (size_t)g_c->shape[(size_t)dd]);
                rem /= (size_t)g_c->shape[(size_t)dd];
                int64_t sc = (dd == dim) ? (c + start) : c;
                pos += sc * in_strides[(size_t)dd];
            }
            gi[(size_t)pos] = gd[i];
        }
        auto out = Tensor::create(input_shape, g_c->device);
        write_vec(out, gi);
        return { out };
    }
};

std::shared_ptr<Tensor> narrow(std::shared_ptr<Tensor> a, int64_t dim, int64_t start, int64_t length) {
    if (a->device.type == DeviceType::GPU) gpu_fail_shape("narrow");
    int64_t ndim = (int64_t)a->shape.size();
    int64_t d = norm_dim_shape(dim, ndim, "narrow");
    if (start < 0 || length < 0 || start + length > a->shape[(size_t)d])
        throw std::runtime_error("[litetorch Error] narrow: invalid start/length");
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        return cast(narrow(a_fp32, dim, start, length), a->dtype);
    }
    auto out = a->slice(d, start, start + length);
    if (Autograd::is_grad_enabled() && a->requires_grad) {
        auto node = std::make_shared<NarrowNode>(d, start, length, a->shape);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> t(std::shared_ptr<Tensor> a) {
    int64_t ndim = (int64_t)a->shape.size();
    if (ndim > 2) throw std::runtime_error("[litetorch Error] t: expected 0D, 1D or 2D tensor");
    if (ndim < 2) return a;
    return permute(a, {1, 0});
}

// === end W1-T1 core-ops ===

}
}
