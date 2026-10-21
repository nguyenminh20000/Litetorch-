#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
#include "litetorch/backend_route.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <limits>
#include <string>
#include <unordered_map>
#include <unistd.h>
#ifdef __linux__
#include <dlfcn.h>
#endif

#include "litetorch/storage_guard.h"
#include <cstdint>
namespace litetorch {

extern const std::string litetorch_kernels_src;

namespace {

inline uint16_t ew_float_to_half(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t exp = (x >> 23) & 0xff;
    uint32_t mant = x & 0x7fffff;
    if (exp == 0) return static_cast<uint16_t>(sign);
    if (exp == 255) return static_cast<uint16_t>(sign | 0x7c00 | (mant ? 0x200 : 0));
    int nexp = static_cast<int>(exp) - 127 + 15;
    if (nexp >= 31) return static_cast<uint16_t>(sign | 0x7c00);
    if (nexp <= 0) return static_cast<uint16_t>(sign);
    return static_cast<uint16_t>(sign | (nexp << 10) | (mant >> 13));
}

inline float ew_half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        f = sign << 31;
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    return *reinterpret_cast<float*>(&f);
}

struct IntBufferPool {
    static cl_mem get_and_fill(const std::vector<int64_t>& vec) {
        thread_local std::vector<cl_mem> pool;
        thread_local size_t index = 0;
        for (int64_t v : vec) {
            if (v > (int64_t)std::numeric_limits<int>::max() || v < (int64_t)std::numeric_limits<int>::min())
                throw std::runtime_error("[litetorch Error] IntBufferPool: shape/stride exceeds INT32 range");
        }
        std::vector<int> int_vec(vec.begin(), vec.end());
        size_t size_bytes = int_vec.size() * sizeof(int);
        if (pool.size() < 32) {
            cl_mem buf = rt_gpu_allocate(64 * sizeof(int));
            if (buf) {
                pool.push_back(buf);
                rt_gpu_write(buf, size_bytes, int_vec.data());
                return buf;
            }
        }
        if (!pool.empty()) {
            cl_mem buf = pool[index % pool.size()];
            index++;
            rt_gpu_write(buf, size_bytes, int_vec.data());
            return buf;
        }
        cl_mem buf = rt_gpu_allocate(size_bytes);
        if (buf) {
            rt_gpu_write(buf, size_bytes, int_vec.data());
        }
        return buf;
    }
};

struct TemporaryGPUBuffers {
    cl_mem add(const std::vector<int64_t>& vec) {
        return IntBufferPool::get_and_fill(vec);
    }
    ~TemporaryGPUBuffers() = default;
};

std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& shape_a, const std::vector<int64_t>& shape_b) {
    std::vector<int64_t> result;
    int ndims_a = shape_a.size();
    int ndims_b = shape_b.size();
    int max_ndims = std::max(ndims_a, ndims_b);
    result.resize(max_ndims);
    for (int i = 0; i < max_ndims; ++i) {
        int idx_a = ndims_a - 1 - i;
        int idx_b = ndims_b - 1 - i;
        int64_t dim_a = (idx_a >= 0) ? shape_a[idx_a] : 1;
        int64_t dim_b = (idx_b >= 0) ? shape_b[idx_b] : 1;
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::runtime_error("[litetorch Error] Incompatible shapes for broadcasting");
        }
        result[max_ndims - 1 - i] = std::max(dim_a, dim_b);
    }
    return result;
}
}

#include "fp16_ew_bridge.h"

namespace Ops {

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> pow(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> sqrt(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> exp(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> log(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> abs(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> neg(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim);
std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim);
std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim);
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> max_with_indices(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim);
std::shared_ptr<Tensor> cat(const std::vector<std::shared_ptr<Tensor>>& tensors, int64_t dim);
std::vector<std::shared_ptr<Tensor>> split(std::shared_ptr<Tensor> input, const std::vector<int64_t>& sizes, int64_t dim);
std::vector<std::shared_ptr<Tensor>> chunk(std::shared_ptr<Tensor> input, int64_t chunks, int64_t dim);
std::shared_ptr<Tensor> squeeze(std::shared_ptr<Tensor> a, int64_t dim);
std::shared_ptr<Tensor> unsqueeze(std::shared_ptr<Tensor> a, int64_t dim);
std::shared_ptr<Tensor> clamp(std::shared_ptr<Tensor> a, float min_val, float max_val);
std::shared_ptr<Tensor> sin(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> cos(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> log2(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> log10(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> rsqrt(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> tan(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> reciprocal(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sign(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> floor(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> ceil(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> round(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> trunc(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> clamp_min(std::shared_ptr<Tensor> a, float min_val);
std::shared_ptr<Tensor> clamp_max(std::shared_ptr<Tensor> a, float max_val);
std::shared_ptr<Tensor> min(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> amax(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim);
std::shared_ptr<Tensor> amin(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim);
std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim);
std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim);
std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim);
bool allclose(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, float rtol, float atol, bool equal_nan);
std::shared_ptr<Tensor> isclose(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, float rtol, float atol, bool equal_nan);

std::shared_ptr<Tensor> reduce_broadcast(std::shared_ptr<Tensor> grad, const std::vector<int64_t>& orig_shape);

std::shared_ptr<Tensor> reduce_broadcast(std::shared_ptr<Tensor> grad, const std::vector<int64_t>& orig_shape) {
    if (!grad->is_contiguous()) grad = grad->contiguous();
    // Must return a distinct tensor object: callers (e.g. Add/Sub/Mul/Div
    // backward) request one grad per input from the same grad_output. Returning
    // grad itself would alias the results, and the in-place grad accumulation
    // in Autograd::backward would then corrupt them.
    if (grad->shape == orig_shape) return grad->clone();

    DataType dt = grad->dtype;
    if (dt != DataType::FP32) {
        auto g32 = grad->cast(DataType::FP32);
        auto r32 = reduce_broadcast(g32, orig_shape);
        return r32->cast(dt);
    }

    std::shared_ptr<Tensor> current = grad;
    int diff = grad->shape.size() - orig_shape.size();

    if (diff > 0) {
        int64_t prod_prepended = 1;
        for (int i = 0; i < diff; ++i) prod_prepended *= current->shape[i];
        int64_t remaining = 1;
        for (size_t i = diff; i < current->shape.size(); ++i) remaining *= current->shape[i];

        std::vector<int64_t> final_shape;
        for (size_t i = diff; i < grad->shape.size(); ++i) final_shape.push_back(grad->shape[i]);

        bool run_gpu = false;
        if (grad->device.type == DeviceType::GPU) {
            auto native = BackendDispatcher::get().get_backend();
            if (native && native->is_available()) {
                typedef void (*ReducePrependedFn)(void*, int64_t, void*, int64_t, int64_t, int64_t);
                auto fn = reinterpret_cast<ReducePrependedFn>(native->get_kernel("", "", "gpu_reduce_broadcast_prepended"));
                if (fn) {
                    run_gpu = true;
                    auto reshaped = current->view({prod_prepended, remaining});
                    auto summed = Tensor::create({remaining}, grad->device);
                    StorageUseGuard guard({reshaped->storage, summed->storage});
                    fn(reshaped->gpu_data(), reshaped->offset, summed->gpu_data(), summed->offset,
                       (int)prod_prepended, (int)remaining);
                    current = summed->view(final_shape);
                }
            }
            if (!run_gpu) {
                auto kernel = rt_gpu_get_kernel_by_id(KernelID::ReduceBroadcastPrepended);
                if (kernel) {
                    run_gpu = true;
                    auto reshaped = current->view({prod_prepended, remaining});
                    auto summed = Tensor::create({remaining}, grad->device);
                    StorageUseGuard guard({reshaped->storage, summed->storage});
                    
                    cl_mem in_mem = reshaped->gpu_data();
                    int64_t in_off = reshaped->offset;
                    cl_mem out_mem = summed->gpu_data();
                    int64_t out_off = summed->offset;
                    int64_t prod_val = prod_prepended;
                    int64_t rem_val = remaining;

                    rt_gpu_launch(kernel, {static_cast<size_t>(remaining)}, {},
                        {&in_mem, &in_off, &out_mem, &out_off, &prod_val, &rem_val},
                        {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t), sizeof(int64_t)});

                    current = summed->view(final_shape);
                }
            }
        }
        if (!run_gpu) {
            auto reshaped = current->view({prod_prepended, remaining});
            auto summed = Tensor::create({remaining}, grad->device);
            float* res_ptr = summed->data_ptr();
            float* reshaped_ptr = reshaped->data_ptr();
            
            ThreadPool::get().parallel_for(0, remaining, [&](int64_t j) {
                float s = 0.0f;
                for (int64_t i = 0; i < prod_prepended; ++i) {
                    s += reshaped_ptr[i * remaining + j];
                }
                res_ptr[j] = s;
            });
            if (grad->device.type == DeviceType::GPU) {
                auto gpu_summed = Tensor::create({remaining}, grad->device);
                rt_gpu_write(gpu_summed->gpu_data(), remaining * sizeof(float), res_ptr);
                current = gpu_summed->view(final_shape);
            } else if (grad->device.type == DeviceType::TPU) {
                auto tpu = BackendDispatcher::get().get_tpu_backend();
                if (tpu) tpu->write(summed->gpu_data(), remaining * sizeof(float), res_ptr);
                current = summed->view(final_shape);
            } else {
                current = summed->view(final_shape);
            }
        }
    }

    for (size_t i = 0; i < orig_shape.size(); ++i) {
        if (orig_shape[i] == 1 && current->shape[i] > 1) {
            int64_t dim_size = current->shape[i];
            int64_t outer_size = 1;
            for (size_t j = 0; j < i; ++j) outer_size *= current->shape[j];
            int64_t inner_size = 1;
            for (size_t j = i + 1; j < current->shape.size(); ++j) inner_size *= current->shape[j];

            std::vector<int64_t> new_shape = current->shape;
            new_shape[i] = 1;

            bool run_gpu = false;
            if (grad->device.type == DeviceType::GPU) {
                auto native = BackendDispatcher::get().get_backend();
                if (native && native->is_available()) {
                    typedef void (*ReduceDimFn)(void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t);
                    auto fn = reinterpret_cast<ReduceDimFn>(native->get_kernel("", "", "gpu_reduce_broadcast_dim"));
                    if (fn) {
                        run_gpu = true;
                        auto summed = Tensor::create(new_shape, grad->device);
                        StorageUseGuard guard({current->storage, summed->storage});
                        fn(current->gpu_data(), current->offset, summed->gpu_data(), summed->offset,
                           outer_size, dim_size, inner_size);
                        current = summed;
                    }
                }
                if (!run_gpu) {
                    auto kernel = rt_gpu_get_kernel_by_id(KernelID::ReduceBroadcastDim);
                    if (kernel) {
                        run_gpu = true;
                        auto summed = Tensor::create(new_shape, grad->device);
                        StorageUseGuard guard({current->storage, summed->storage});

                        cl_mem in_mem = current->gpu_data();
                        int64_t in_off = current->offset;
                        cl_mem out_mem = summed->gpu_data();
                        int64_t out_off = summed->offset;
                        int64_t out_sz = outer_size;
                        int64_t d_sz = dim_size;
                        int64_t in_sz = inner_size;

                        int64_t total_threads = outer_size * inner_size;
                        rt_gpu_launch(kernel, {static_cast<size_t>(total_threads)}, {},
                            {&in_mem, &in_off, &out_mem, &out_off, &out_sz, &d_sz, &in_sz},
                            {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t), sizeof(int64_t), sizeof(int64_t)});

                        current = summed;
                    }
                }
            }
            if (!run_gpu) {
                auto summed = Tensor::create(new_shape, grad->device);
                float* src = current->data_ptr();
                float* dst = summed->data_ptr();

                ThreadPool::get().parallel_for(0, outer_size, [&](int64_t o) {
                    for (int64_t in = 0; in < inner_size; ++in) {
                        float s = 0.0f;
                        for (int64_t d = 0; d < dim_size; ++d) {
                            s += src[(o * dim_size + d) * inner_size + in];
                        }
                        dst[o * inner_size + in] = s;
                    }
                });
                if (grad->device.type == DeviceType::GPU) {
                    auto gpu_summed = Tensor::create(new_shape, grad->device);
                    rt_gpu_write(gpu_summed->gpu_data(), gpu_summed->numel() * sizeof(float), dst);
                    current = gpu_summed;
                } else if (grad->device.type == DeviceType::TPU) {
                    auto tpu = BackendDispatcher::get().get_tpu_backend();
                    if (tpu) tpu->write(summed->gpu_data(), summed->numel() * sizeof(float), dst);
                    current = summed;
                } else {
                    current = summed;
                }
            }
        }
    }
    return current;
}

class AddNode : public Node {
public:
    std::vector<int64_t> shape_a;
    std::vector<int64_t> shape_b;
    AddNode(const std::vector<int64_t>& shape_a, const std::vector<int64_t>& shape_b)
        : Node("Add"), shape_a(shape_a), shape_b(shape_b) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto grad_a = reduce_broadcast(grad_output, shape_a);
        auto grad_b = reduce_broadcast(grad_output, shape_b);
        return { grad_a, grad_b };
    }
};

class SubNode : public Node {
public:
    std::vector<int64_t> shape_a;
    std::vector<int64_t> shape_b;
    SubNode(const std::vector<int64_t>& shape_a, const std::vector<int64_t>& shape_b)
        : Node("Sub"), shape_a(shape_a), shape_b(shape_b) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto grad_a = reduce_broadcast(grad_output, shape_a);
        auto grad_b = reduce_broadcast(neg(grad_output), shape_b);
        return { grad_a, grad_b };
    }
};

class MulNode : public Node {
public:
    MulNode() : Node("Mul") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto a = saved_tensors[0];
        auto b = saved_tensors[1];
        auto grad_a = reduce_broadcast(mul(grad_output, b), a->shape);
        auto grad_b = reduce_broadcast(mul(grad_output, a), b->shape);
        return { grad_a, grad_b };
    }
};

class DivNode : public Node {
public:
    DivNode() : Node("Div") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto a = saved_tensors[0];
        auto b = saved_tensors[1];
        auto grad_a = reduce_broadcast(div(grad_output, b), a->shape);
        auto b2 = mul(b, b);
        auto neg_a = mul(a, Tensor::from_vector({-1.0f}, {1}, a->device, false, a->dtype));
        auto grad_b = reduce_broadcast(mul(grad_output, div(neg_a, b2)), b->shape);
        return { grad_a, grad_b };
    }
};


template <typename T, typename F>
std::shared_ptr<Tensor> ew_binary_int(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
                                      DataType dt, F f,
                                      std::function<std::shared_ptr<Node>()> make_node,
                                      bool save_inputs) {
    auto ai = (a->dtype == dt) ? a : Ops::cast(a, dt);
    auto bi = (b->dtype == dt) ? b : Ops::cast(b, dt);
    std::vector<int64_t> out_shape = broadcast_shapes(ai->shape, bi->shape);
    auto out = Tensor::create(out_shape, ai->device, false, dt);
    StorageUseGuard guard({ai->storage, bi->storage, out->storage});
    const T* ap = reinterpret_cast<const T*>(ai->raw_data_ptr());
    const T* bp = reinterpret_cast<const T*>(bi->raw_data_ptr());
    T* op = reinterpret_cast<T*>(out->raw_data_ptr());
    int64_t n = static_cast<int64_t>(out->numel());
    if (ai->shape == bi->shape && ai->is_contiguous() && bi->is_contiguous()) {
        ThreadPool::get().parallel_for(0, n, [&](int64_t i) { op[i] = f(ap[i], bp[i]); });
    } else {
        int ndims = static_cast<int>(out_shape.size());
        std::vector<int64_t> out_strides((size_t)ndims);
        int64_t current = 1;
        for (int d = ndims - 1; d >= 0; --d) {
            out_strides[(size_t)d] = current;
            current *= out_shape[(size_t)d];
        }
        int a_diff = ndims - static_cast<int>(ai->shape.size());
        int b_diff = ndims - static_cast<int>(bi->shape.size());
        ThreadPool::get().parallel_for(0, n, [&](int64_t i) {
            int64_t temp = i;
            int64_t a_idx = 0;
            int64_t b_idx = 0;
            for (int d = 0; d < ndims; ++d) {
                int64_t coord = temp / out_strides[(size_t)d];
                temp %= out_strides[(size_t)d];
                if (d >= a_diff) {
                    int a_dim_idx = d - a_diff;
                    if (ai->shape[(size_t)a_dim_idx] != 1) {
                        a_idx += coord * ai->strides[(size_t)a_dim_idx];
                    }
                }
                if (d >= b_diff) {
                    int b_dim_idx = d - b_diff;
                    if (bi->shape[(size_t)b_dim_idx] != 1) {
                        b_idx += coord * bi->strides[(size_t)b_dim_idx];
                    }
                }
            }
            op[i] = f(ap[a_idx], bp[b_idx]);
        });
    }
    if (Autograd::is_grad_enabled() && (a->requires_grad || b->requires_grad)) {
        auto node = make_node();
        node->inputs = { {ai, ai->requires_grad}, {bi, bi->requires_grad} };
        node->next_nodes = { ai->creator, bi->creator };
        if (save_inputs) {
            node->saved_tensors = { ai, bi };
            node->saved_versions = { (ai ? ai->_version() : 0), (bi ? bi->_version() : 0) };
        } else {
            node->saved_tensors = {};
            node->saved_versions = {};
        }
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> ew_add_int(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, DataType dt) {
    switch (dt) {
        case DataType::BOOL:
            return ew_binary_int<uint8_t>(a, b, dt,
                [](uint8_t x, uint8_t y) -> uint8_t { return (x + y) ? 1 : 0; },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<AddNode>(a->shape, b->shape); }, false);
        case DataType::UINT8:
            return ew_binary_int<uint8_t>(a, b, dt,
                [](uint8_t x, uint8_t y) -> uint8_t { return static_cast<uint8_t>(x + y); },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<AddNode>(a->shape, b->shape); }, false);
        case DataType::INT8:
        case DataType::INT4:
            return ew_binary_int<int8_t>(a, b, dt,
                [](int8_t x, int8_t y) -> int8_t { return static_cast<int8_t>(x + y); },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<AddNode>(a->shape, b->shape); }, false);
        case DataType::INT16:
            return ew_binary_int<int16_t>(a, b, dt,
                [](int16_t x, int16_t y) -> int16_t { return static_cast<int16_t>(x + y); },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<AddNode>(a->shape, b->shape); }, false);
        case DataType::INT32:
            return ew_binary_int<int32_t>(a, b, dt,
                [](int32_t x, int32_t y) -> int32_t { return x + y; },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<AddNode>(a->shape, b->shape); }, false);
        case DataType::INT64:
            return ew_binary_int<int64_t>(a, b, dt,
                [](int64_t x, int64_t y) -> int64_t { return x + y; },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<AddNode>(a->shape, b->shape); }, false);
        default:
            throw std::runtime_error("[litetorch Error] ew_add_int: unsupported integer dtype");
    }
}

std::shared_ptr<Tensor> ew_sub_int(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, DataType dt) {
    switch (dt) {
        case DataType::BOOL:
            return ew_binary_int<uint8_t>(a, b, dt,
                [](uint8_t x, uint8_t y) -> uint8_t { return (x - y) ? 1 : 0; },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<SubNode>(a->shape, b->shape); }, false);
        case DataType::UINT8:
            return ew_binary_int<uint8_t>(a, b, dt,
                [](uint8_t x, uint8_t y) -> uint8_t { return static_cast<uint8_t>(x - y); },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<SubNode>(a->shape, b->shape); }, false);
        case DataType::INT8:
        case DataType::INT4:
            return ew_binary_int<int8_t>(a, b, dt,
                [](int8_t x, int8_t y) -> int8_t { return static_cast<int8_t>(x - y); },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<SubNode>(a->shape, b->shape); }, false);
        case DataType::INT16:
            return ew_binary_int<int16_t>(a, b, dt,
                [](int16_t x, int16_t y) -> int16_t { return static_cast<int16_t>(x - y); },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<SubNode>(a->shape, b->shape); }, false);
        case DataType::INT32:
            return ew_binary_int<int32_t>(a, b, dt,
                [](int32_t x, int32_t y) -> int32_t { return x - y; },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<SubNode>(a->shape, b->shape); }, false);
        case DataType::INT64:
            return ew_binary_int<int64_t>(a, b, dt,
                [](int64_t x, int64_t y) -> int64_t { return x - y; },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<SubNode>(a->shape, b->shape); }, false);
        default:
            throw std::runtime_error("[litetorch Error] ew_sub_int: unsupported integer dtype");
    }
}

std::shared_ptr<Tensor> ew_mul_int(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, DataType dt) {
    switch (dt) {
        case DataType::BOOL:
            return ew_binary_int<uint8_t>(a, b, dt,
                [](uint8_t x, uint8_t y) -> uint8_t { return (x * y) ? 1 : 0; },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<MulNode>(); }, true);
        case DataType::UINT8:
            return ew_binary_int<uint8_t>(a, b, dt,
                [](uint8_t x, uint8_t y) -> uint8_t { return static_cast<uint8_t>(x * y); },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<MulNode>(); }, true);
        case DataType::INT8:
        case DataType::INT4:
            return ew_binary_int<int8_t>(a, b, dt,
                [](int8_t x, int8_t y) -> int8_t { return static_cast<int8_t>(x * y); },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<MulNode>(); }, true);
        case DataType::INT16:
            return ew_binary_int<int16_t>(a, b, dt,
                [](int16_t x, int16_t y) -> int16_t { return static_cast<int16_t>(x * y); },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<MulNode>(); }, true);
        case DataType::INT32:
            return ew_binary_int<int32_t>(a, b, dt,
                [](int32_t x, int32_t y) -> int32_t { return x * y; },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<MulNode>(); }, true);
        case DataType::INT64:
            return ew_binary_int<int64_t>(a, b, dt,
                [](int64_t x, int64_t y) -> int64_t { return x * y; },
                [&]() -> std::shared_ptr<Node> { return std::make_shared<MulNode>(); }, true);
        default:
            throw std::runtime_error("[litetorch Error] ew_mul_int: unsupported integer dtype");
    }
}

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (auto out = try_binary_fp16(a, b, "elementwise_add_fp16",
            [&]() -> std::shared_ptr<Node> { return std::make_shared<AddNode>(a->shape, b->shape); },
            false)) return out;
    {
        DataType dt = promote_types(a->dtype, b->dtype);
        if (is_integer_dtype(dt) && a->device.type == DeviceType::CPU && b->device.type == DeviceType::CPU
            && a->device == b->device) {
            return ew_add_int(a, b, dt);
        }
        if (a->dtype != DataType::FP32 || b->dtype != DataType::FP32) {
            auto a_fp32 = cast(a, DataType::FP32);
            auto b_fp32 = cast(b, DataType::FP32);
            auto out_fp32 = add(a_fp32, b_fp32);
            return cast(out_fp32, dt);
        }
    }
    if (a->device != b->device) {
        if (a->device.type == DeviceType::GPU) b = b->to(a->device);
        else a = a->to(b->device);
    }
    std::vector<int64_t> out_shape = broadcast_shapes(a->shape, b->shape);
    auto out = Tensor::create(out_shape, a->device);
    StorageUseGuard guard({a->storage, b->storage, out->storage});

    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        if (a->shape == b->shape && a->is_contiguous() && b->is_contiguous()) {
            auto kernel = rt_gpu_get_kernel_by_id(KernelID::Add);
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int64_t a_off = a->offset;
                int64_t b_off = b->offset;
                int64_t c_off = out->offset;
                int64_t size = out->numel();
                rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &size},
                    {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t)});
            }
        } else {
            auto kernel = rt_gpu_get_kernel("litetorch_kernels", litetorch_kernels_src, "elementwise_broadcast_add");
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int64_t a_off = a->offset;
                int64_t b_off = b->offset;
                int64_t c_off = out->offset;

                TemporaryGPUBuffers temp_bufs;
                cl_mem out_shape_mem = temp_bufs.add(out_shape);
                cl_mem a_shape_mem = temp_bufs.add(a->shape);
                cl_mem a_strides_mem = temp_bufs.add(a->strides);
                cl_mem b_shape_mem = temp_bufs.add(b->shape);
                cl_mem b_strides_mem = temp_bufs.add(b->strides);

                int ndims = out_shape.size();
                int a_ndims = a->shape.size();
                int b_ndims = b->shape.size();
                int64_t size = out->numel();

                rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &ndims, &out_shape_mem,
                     &a_ndims, &a_shape_mem, &a_strides_mem,
                     &b_ndims, &b_shape_mem, &b_strides_mem, &size},
                    {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t),
                     sizeof(int), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem), sizeof(int64_t)});
            }
        }
    }
    if (!run_gpu) {
        float* a_ptr = a->data_ptr();
        float* b_ptr = b->data_ptr();
        float* out_ptr = out->data_ptr();
        if (a->shape == b->shape && a->is_contiguous() && b->is_contiguous()) {
            int64_t size = out->numel();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                out_ptr[i] = a_ptr[i] + b_ptr[i];
            });
        } else {
            int64_t size = out->numel();
            int ndims = out_shape.size();
            std::vector<int64_t> out_strides(ndims);
            int64_t current = 1;
            for (int i = ndims - 1; i >= 0; --i) {
                out_strides[i] = current;
                current *= out_shape[i];
            }
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                int64_t temp = i;
                int64_t a_idx = 0;
                int64_t b_idx = 0;
                int64_t a_diff = (int64_t)ndims - (int64_t)a->shape.size();
                int64_t b_diff = (int64_t)ndims - (int64_t)b->shape.size();
                for (int d = 0; d < ndims; ++d) {
                    int64_t coord = temp / out_strides[d];
                    temp %= out_strides[d];
                    if (d >= a_diff) {
                        int64_t a_dim_idx = d - a_diff;
                        if (a->shape[a_dim_idx] != 1) {
                            a_idx += coord * a->strides[a_dim_idx];
                        }
                    }
                    if (d >= b_diff) {
                        int64_t b_dim_idx = d - b_diff;
                        if (b->shape[b_dim_idx] != 1) {
                            b_idx += coord * b->strides[b_dim_idx];
                        }
                    }
                }
                out_ptr[i] = a_ptr[a_idx] + b_ptr[b_idx];
            });
        }
        if (out->device.type == DeviceType::GPU) {
            rt_gpu_write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        }
    }

    if (Autograd::is_grad_enabled() && (a->requires_grad || b->requires_grad)) {
        auto node = std::make_shared<AddNode>(a->shape, b->shape);
        node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
        node->next_nodes = { a->creator, b->creator };
        node->saved_tensors = {};
        node->saved_versions = {};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (auto out = try_binary_fp16(a, b, "elementwise_sub_fp16",
            [&]() -> std::shared_ptr<Node> { return std::make_shared<SubNode>(a->shape, b->shape); },
            false)) return out;
    {
        DataType dt = promote_types(a->dtype, b->dtype);
        if (is_integer_dtype(dt) && a->device.type == DeviceType::CPU && b->device.type == DeviceType::CPU
            && a->device == b->device) {
            return ew_sub_int(a, b, dt);
        }
        if (a->dtype != DataType::FP32 || b->dtype != DataType::FP32) {
            auto a_fp32 = cast(a, DataType::FP32);
            auto b_fp32 = cast(b, DataType::FP32);
            auto out_fp32 = sub(a_fp32, b_fp32);
            return cast(out_fp32, dt);
        }
    }
    if (a->device != b->device) {
        if (a->device.type == DeviceType::GPU) b = b->to(a->device);
        else a = a->to(b->device);
    }
    std::vector<int64_t> out_shape = broadcast_shapes(a->shape, b->shape);
    auto out = Tensor::create(out_shape, a->device);
    StorageUseGuard guard({a->storage, b->storage, out->storage});

    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        if (a->shape == b->shape && a->is_contiguous() && b->is_contiguous()) {
            auto kernel = rt_gpu_get_kernel_by_id(KernelID::Sub);
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int64_t a_off = a->offset;
                int64_t b_off = b->offset;
                int64_t c_off = out->offset;
                int64_t size = out->numel();
                rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &size},
                    {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t)});
            }
        } else {
            auto kernel = rt_gpu_get_kernel("litetorch_kernels", litetorch_kernels_src, "elementwise_broadcast_sub");
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int64_t a_off = a->offset;
                int64_t b_off = b->offset;
                int64_t c_off = out->offset;

                TemporaryGPUBuffers temp_bufs;
                cl_mem out_shape_mem = temp_bufs.add(out_shape);
                cl_mem a_shape_mem = temp_bufs.add(a->shape);
                cl_mem a_strides_mem = temp_bufs.add(a->strides);
                cl_mem b_shape_mem = temp_bufs.add(b->shape);
                cl_mem b_strides_mem = temp_bufs.add(b->strides);

                int ndims = out_shape.size();
                int a_ndims = a->shape.size();
                int b_ndims = b->shape.size();
                int64_t size = out->numel();

                rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &ndims, &out_shape_mem,
                     &a_ndims, &a_shape_mem, &a_strides_mem,
                     &b_ndims, &b_shape_mem, &b_strides_mem, &size},
                    {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t),
                     sizeof(int), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem), sizeof(int64_t)});
            }
        }
    }
    if (!run_gpu) {
        float* a_ptr = a->data_ptr();
        float* b_ptr = b->data_ptr();
        float* out_ptr = out->data_ptr();
        if (a->shape == b->shape && a->is_contiguous() && b->is_contiguous()) {
            int64_t size = out->numel();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                out_ptr[i] = a_ptr[i] - b_ptr[i];
            });
        } else {
            int64_t size = out->numel();
            int ndims = out_shape.size();
            std::vector<int64_t> out_strides(ndims);
            int64_t current = 1;
            for (int i = ndims - 1; i >= 0; --i) {
                out_strides[i] = current;
                current *= out_shape[i];
            }
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                int64_t temp = i;
                int64_t a_idx = 0;
                int64_t b_idx = 0;
                int64_t a_diff = (int64_t)ndims - (int64_t)a->shape.size();
                int64_t b_diff = (int64_t)ndims - (int64_t)b->shape.size();
                for (int d = 0; d < ndims; ++d) {
                    int64_t coord = temp / out_strides[d];
                    temp %= out_strides[d];
                    if (d >= a_diff) {
                        int64_t a_dim_idx = d - a_diff;
                        if (a->shape[a_dim_idx] != 1) {
                            a_idx += coord * a->strides[a_dim_idx];
                        }
                    }
                    if (d >= b_diff) {
                        int64_t b_dim_idx = d - b_diff;
                        if (b->shape[b_dim_idx] != 1) {
                            b_idx += coord * b->strides[b_dim_idx];
                        }
                    }
                }
                out_ptr[i] = a_ptr[a_idx] - b_ptr[b_idx];
            });
        }
        if (out->device.type == DeviceType::GPU) {
            rt_gpu_write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        }
    }

    if (Autograd::is_grad_enabled() && (a->requires_grad || b->requires_grad)) {
        auto node = std::make_shared<SubNode>(a->shape, b->shape);
        node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
        node->next_nodes = { a->creator, b->creator };
        node->saved_tensors = {};
        node->saved_versions = {};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (auto out = try_binary_fp16(a, b, "elementwise_mul_fp16",
            [&]() -> std::shared_ptr<Node> { return std::make_shared<MulNode>(); },
            true)) return out;
    if (a->dtype == DataType::FP16 && b->dtype == DataType::FP16 &&
        a->device.type == DeviceType::GPU && a->device == b->device &&
        a->shape != b->shape &&
        (!Autograd::is_grad_enabled() || (!a->requires_grad && !b->requires_grad))) {
        std::vector<int64_t> bc_shape = broadcast_shapes(a->shape, b->shape);
        auto a_exp = (a->shape == bc_shape) ? a : a->expand(bc_shape);
        auto b_exp = (b->shape == bc_shape) ? b : b->expand(bc_shape);
        auto a_c = a_exp->is_contiguous() ? a_exp : a_exp->contiguous();
        auto b_c = b_exp->is_contiguous() ? b_exp : b_exp->contiguous();
        if (auto out = try_binary_fp16(a_c, b_c, "elementwise_mul_fp16",
                [&]() -> std::shared_ptr<Node> { return std::make_shared<MulNode>(); },
                true)) return out;
    }
    if (a->dtype != DataType::FP32 || b->dtype != DataType::FP32) {
        DataType dt = promote_types(a->dtype, b->dtype);
        if (is_integer_dtype(dt) && a->device.type == DeviceType::CPU && b->device.type == DeviceType::CPU
            && a->device == b->device) {
            return ew_mul_int(a, b, dt);
        }
        auto a_fp32 = cast(a, DataType::FP32);
        auto b_fp32 = cast(b, DataType::FP32);
        auto out_fp32 = mul(a_fp32, b_fp32);
        return cast(out_fp32, dt);
    }
    if (a->device != b->device) {
        if (a->device.type == DeviceType::GPU) b = b->to(a->device);
        else a = a->to(b->device);
    }
    std::vector<int64_t> out_shape = broadcast_shapes(a->shape, b->shape);
    auto out = Tensor::create(out_shape, a->device);
    StorageUseGuard guard({a->storage, b->storage, out->storage});

    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        if (a->shape == b->shape && a->is_contiguous() && b->is_contiguous()) {
            auto kernel = rt_gpu_get_kernel_by_id(KernelID::Mul);
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int64_t a_off = a->offset;
                int64_t b_off = b->offset;
                int64_t c_off = out->offset;
                int64_t size = out->numel();
                rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &size},
                    {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t)});
            }
        } else {
            auto kernel = rt_gpu_get_kernel("litetorch_kernels", litetorch_kernels_src, "elementwise_broadcast_mul");
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int64_t a_off = a->offset;
                int64_t b_off = b->offset;
                int64_t c_off = out->offset;

                TemporaryGPUBuffers temp_bufs;
                cl_mem out_shape_mem = temp_bufs.add(out_shape);
                cl_mem a_shape_mem = temp_bufs.add(a->shape);
                cl_mem a_strides_mem = temp_bufs.add(a->strides);
                cl_mem b_shape_mem = temp_bufs.add(b->shape);
                cl_mem b_strides_mem = temp_bufs.add(b->strides);

                int ndims = out_shape.size();
                int a_ndims = a->shape.size();
                int b_ndims = b->shape.size();
                int64_t size = out->numel();

                rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &ndims, &out_shape_mem,
                     &a_ndims, &a_shape_mem, &a_strides_mem,
                     &b_ndims, &b_shape_mem, &b_strides_mem, &size},
                    {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t),
                     sizeof(int), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem), sizeof(int64_t)});
            }
        }
    }
    if (!run_gpu) {
        float* a_ptr = a->data_ptr();
        float* b_ptr = b->data_ptr();
        float* out_ptr = out->data_ptr();
        if (a->shape == b->shape && a->is_contiguous() && b->is_contiguous()) {
            int64_t size = out->numel();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                out_ptr[i] = a_ptr[i] * b_ptr[i];
            });
        } else {
            int64_t size = out->numel();
            int ndims = out_shape.size();
            std::vector<int64_t> out_strides(ndims);
            int64_t current = 1;
            for (int i = ndims - 1; i >= 0; --i) {
                out_strides[i] = current;
                current *= out_shape[i];
            }
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                int64_t temp = i;
                int64_t a_idx = 0;
                int64_t b_idx = 0;
                int64_t a_diff = (int64_t)ndims - (int64_t)a->shape.size();
                int64_t b_diff = (int64_t)ndims - (int64_t)b->shape.size();
                for (int d = 0; d < ndims; ++d) {
                    int64_t coord = temp / out_strides[d];
                    temp %= out_strides[d];
                    if (d >= a_diff) {
                        int64_t a_dim_idx = d - a_diff;
                        if (a->shape[a_dim_idx] != 1) {
                            a_idx += coord * a->strides[a_dim_idx];
                        }
                    }
                    if (d >= b_diff) {
                        int64_t b_dim_idx = d - b_diff;
                        if (b->shape[b_dim_idx] != 1) {
                            b_idx += coord * b->strides[b_dim_idx];
                        }
                    }
                }
                out_ptr[i] = a_ptr[a_idx] * b_ptr[b_idx];
            });
        }
        if (out->device.type == DeviceType::GPU) {
            rt_gpu_write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        }
    }

    if (Autograd::is_grad_enabled() && (a->requires_grad || b->requires_grad)) {
        auto node = std::make_shared<MulNode>();
        node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
        node->next_nodes = { a->creator, b->creator };
        node->saved_tensors = { a, b };
        node->saved_versions = {(a ? a->_version() : 0), (b ? b->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (auto out = try_binary_fp16(a, b, "elementwise_div_fp16",
            [&]() -> std::shared_ptr<Node> { return std::make_shared<DivNode>(); },
            true)) return out;
    {
        DataType dt = promote_types(a->dtype, b->dtype);
        if (is_integer_dtype(dt)) {
            auto a_fp32 = cast(a, DataType::FP32);
            auto b_fp32 = cast(b, DataType::FP32);
            return div(a_fp32, b_fp32);
        }
        if (a->dtype != DataType::FP32 || b->dtype != DataType::FP32) {
            auto a_fp32 = cast(a, DataType::FP32);
            auto b_fp32 = cast(b, DataType::FP32);
            auto out_fp32 = div(a_fp32, b_fp32);
            return cast(out_fp32, dt);
        }
    }
    if (a->device != b->device) {
        if (a->device.type == DeviceType::GPU) b = b->to(a->device);
        else a = a->to(b->device);
    }
    std::vector<int64_t> out_shape = broadcast_shapes(a->shape, b->shape);
    auto out = Tensor::create(out_shape, a->device);
    StorageUseGuard guard({a->storage, b->storage, out->storage});

    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        if (a->shape == b->shape && a->is_contiguous() && b->is_contiguous()) {
            auto kernel = rt_gpu_get_kernel_by_id(KernelID::Div);
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int64_t a_off = a->offset;
                int64_t b_off = b->offset;
                int64_t c_off = out->offset;
                int64_t size = out->numel();
                rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &size},
                    {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t)});
            }
        } else {
            auto kernel = rt_gpu_get_kernel("litetorch_kernels", litetorch_kernels_src, "elementwise_broadcast_div");
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int64_t a_off = a->offset;
                int64_t b_off = b->offset;
                int64_t c_off = out->offset;

                TemporaryGPUBuffers temp_bufs;
                cl_mem out_shape_mem = temp_bufs.add(out_shape);
                cl_mem a_shape_mem = temp_bufs.add(a->shape);
                cl_mem a_strides_mem = temp_bufs.add(a->strides);
                cl_mem b_shape_mem = temp_bufs.add(b->shape);
                cl_mem b_strides_mem = temp_bufs.add(b->strides);

                int ndims = out_shape.size();
                int a_ndims = a->shape.size();
                int b_ndims = b->shape.size();
                int64_t size = out->numel();

                rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &ndims, &out_shape_mem,
                     &a_ndims, &a_shape_mem, &a_strides_mem,
                     &b_ndims, &b_shape_mem, &b_strides_mem, &size},
                    {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t),
                     sizeof(int), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem), sizeof(int64_t)});
            }
        }
    }
    if (!run_gpu) {
        float* a_ptr = a->data_ptr();
        float* b_ptr = b->data_ptr();
        float* out_ptr = out->data_ptr();
        if (a->shape == b->shape && a->is_contiguous() && b->is_contiguous()) {
            int64_t size = out->numel();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                out_ptr[i] = a_ptr[i] / b_ptr[i];
            });
        } else {
            int64_t size = out->numel();
            int ndims = out_shape.size();
            std::vector<int64_t> out_strides(ndims);
            int64_t current = 1;
            for (int i = ndims - 1; i >= 0; --i) {
                out_strides[i] = current;
                current *= out_shape[i];
            }
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                int64_t temp = i;
                int64_t a_idx = 0;
                int64_t b_idx = 0;
                int64_t a_diff = (int64_t)ndims - (int64_t)a->shape.size();
                int64_t b_diff = (int64_t)ndims - (int64_t)b->shape.size();
                for (int d = 0; d < ndims; ++d) {
                    int64_t coord = temp / out_strides[d];
                    temp %= out_strides[d];
                    if (d >= a_diff) {
                        int64_t a_dim_idx = d - a_diff;
                        if (a->shape[a_dim_idx] != 1) {
                            a_idx += coord * a->strides[a_dim_idx];
                        }
                    }
                    if (d >= b_diff) {
                        int64_t b_dim_idx = d - b_diff;
                        if (b->shape[b_dim_idx] != 1) {
                            b_idx += coord * b->strides[b_dim_idx];
                        }
                    }
                }
                out_ptr[i] = a_ptr[a_idx] / b_ptr[b_idx];
            });
        }
        if (out->device.type == DeviceType::GPU) {
            rt_gpu_write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        }
    }

    if (Autograd::is_grad_enabled() && (a->requires_grad || b->requires_grad)) {
        auto node = std::make_shared<DivNode>();
        node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
        node->next_nodes = { a->creator, b->creator };
        node->saved_tensors = { a, b };
        node->saved_versions = {(a ? a->_version() : 0), (b ? b->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class PowNode : public Node {
public:
    float exponent;
    PowNode(float exponent) : Node("Pow"), exponent(exponent) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto exponent_t = Tensor::from_vector({exponent}, {1}, input->device);
        auto exponent_minus_one_t = Tensor::from_vector({exponent - 1.0f}, {1}, input->device);
        auto grad_input = mul(grad_output, mul(exponent_t, pow(input, exponent_minus_one_t->item())));
        return { grad_input };
    }
};

std::shared_ptr<Tensor> pow(std::shared_ptr<Tensor> a, float exponent) {
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = pow(a_fp32, exponent);
        return cast(out_fp32, a->dtype);
    }
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    StorageUseGuard guard({a_c->storage, out->storage});
    int64_t size = a_c->numel();
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto kernel = rt_gpu_get_kernel_by_id(KernelID::Pow);
        if (kernel) {
            run_gpu = true;
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int64_t a_off = a_c->offset;
            int64_t b_off = out->offset;
            rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                {&a_mem, &a_off, &exponent, &b_mem, &b_off, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(cl_mem), sizeof(int), sizeof(int)});
        }
    }
    if (!run_gpu) {
        float* a_ptr = a_c->data_ptr();
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            out_ptr[i] = std::pow(a_ptr[i], exponent);
        });
        if (out->device.type == DeviceType::GPU) {
            rt_gpu_write(out->gpu_data(), size * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), out_ptr);
        }
    }

    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<PowNode>(exponent);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
        node->saved_versions = {(a ? a->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class SqrtNode : public Node {
public:
    SqrtNode() : Node("Sqrt") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        if (saved_tensors.size() <= 0 || !saved_tensors[0]) throw std::runtime_error("[litetorch Error] Sqrt backward: saved output tensor missing");
        auto out = saved_tensors[0];
        auto two_t = Tensor::from_vector({2.0f}, {1}, out->device, false, out->dtype);
        auto grad_input = div(grad_output, mul(two_t, out));
        return { grad_input };
    }
};

std::shared_ptr<Tensor> sqrt(std::shared_ptr<Tensor> a) {
    if (auto out = try_unary_fp16(a, "sqrt_forward_fp16",
            [&]() -> std::shared_ptr<Node> { return std::make_shared<SqrtNode>(); },
            true)) return out;
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = sqrt(a_fp32);
        return cast(out_fp32, a->dtype);
    }
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    StorageUseGuard guard({a_c->storage, out->storage});
    int64_t size = a_c->numel();
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto kernel = rt_gpu_get_kernel_by_id(KernelID::Sqrt);
        if (kernel) {
            run_gpu = true;
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int64_t a_off = a_c->offset;
            int64_t b_off = out->offset;
            rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                {&a_mem, &a_off, &b_mem, &b_off, &size},
                {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t)});
        }
    }
    if (!run_gpu) {
        float* a_ptr = a_c->data_ptr();
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            out_ptr[i] = std::sqrt(a_ptr[i]);
        });
        if (out->device.type == DeviceType::GPU) {
            rt_gpu_write(out->gpu_data(), size * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), out_ptr);
        }
    }

    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<SqrtNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = {out};
        node->saved_versions = {(out ? out->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class ExpNode : public Node {
public:
    ExpNode() : Node("Exp") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        if (saved_tensors.size() <= 0 || !saved_tensors[0]) throw std::runtime_error("[litetorch Error] Exp backward: saved output tensor missing");
        auto out = saved_tensors[0];
        auto grad_input = mul(grad_output, out);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> exp(std::shared_ptr<Tensor> a) {
    if (auto out = try_unary_fp16(a, "exp_forward_fp16",
            [&]() -> std::shared_ptr<Node> { return std::make_shared<ExpNode>(); },
            true)) return out;
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = exp(a_fp32);
        return cast(out_fp32, a->dtype);
    }
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    StorageUseGuard guard({a_c->storage, out->storage});
    int64_t size = a_c->numel();
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto kernel = rt_gpu_get_kernel_by_id(KernelID::Exp);
        if (kernel) {
            run_gpu = true;
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int64_t a_off = a_c->offset;
            int64_t b_off = out->offset;
            rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                {&a_mem, &a_off, &b_mem, &b_off, &size},
                {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t)});
        }
    }
    if (!run_gpu) {
        float* a_ptr = a_c->data_ptr();
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            out_ptr[i] = std::exp(a_ptr[i]);
        });
        if (out->device.type == DeviceType::GPU) {
            rt_gpu_write(out->gpu_data(), size * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), out_ptr);
        }
    }

    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<ExpNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = {out};
        node->saved_versions = {(out ? out->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class LogNode : public Node {
public:
    LogNode() : Node("Log") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto grad_input = div(grad_output, input);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> log(std::shared_ptr<Tensor> a) {
    if (auto out = try_unary_fp16(a, "log_forward_fp16",
            [&]() -> std::shared_ptr<Node> { return std::make_shared<LogNode>(); },
            false)) return out;
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = log(a_fp32);
        return cast(out_fp32, a->dtype);
    }
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    StorageUseGuard guard({a_c->storage, out->storage});
    int64_t size = a_c->numel();
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto kernel = rt_gpu_get_kernel_by_id(KernelID::Log);
        if (kernel) {
            run_gpu = true;
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int64_t a_off = a_c->offset;
            int64_t b_off = out->offset;
            rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                {&a_mem, &a_off, &b_mem, &b_off, &size},
                {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t)});
        }
    }
    if (!run_gpu) {
        float* a_ptr = a_c->data_ptr();
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            out_ptr[i] = std::log(a_ptr[i]);
        });
        if (out->device.type == DeviceType::GPU) {
            rt_gpu_write(out->gpu_data(), size * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), out_ptr);
        }
    }

    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<LogNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
        node->saved_versions = {(a ? a->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class AbsNode : public Node {
public:
    AbsNode() : Node("Abs") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        std::shared_ptr<Tensor> cpu_in;
        if (input_c->dtype == DataType::FP32) {
            cpu_in = input_c;
        } else {
            cpu_in = input_c->cast(DataType::FP32)->to(Device(DeviceType::CPU, 0));
            if (!cpu_in->is_contiguous()) cpu_in = cpu_in->contiguous();
        }
        auto signs = Tensor::create(cpu_in->shape, Device(DeviceType::CPU, 0));
        float* in_ptr = cpu_in->data_ptr();
        float* s_ptr = signs->data_ptr();
        for (size_t i = 0; i < cpu_in->numel(); ++i) {
            s_ptr[i] = (in_ptr[i] > 0.0f) ? 1.0f : ((in_ptr[i] < 0.0f) ? -1.0f : 0.0f);
        }
        if (input->dtype != DataType::FP32) signs = signs->cast(input->dtype);
        if (input->device.type != DeviceType::CPU) {
            signs = signs->to(input->device);
        }
        auto grad_input = mul(grad_output, signs);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> abs(std::shared_ptr<Tensor> a) {
    if (auto out = try_unary_fp16(a, "abs_forward_fp16",
            [&]() -> std::shared_ptr<Node> { return std::make_shared<AbsNode>(); },
            false)) return out;
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = abs(a_fp32);
        return cast(out_fp32, a->dtype);
    }
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    StorageUseGuard guard({a_c->storage, out->storage});
    int64_t size = a_c->numel();
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto kernel = rt_gpu_get_kernel_by_id(KernelID::Abs);
        if (kernel) {
            run_gpu = true;
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int64_t a_off = a_c->offset;
            int64_t b_off = out->offset;
            rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                {&a_mem, &a_off, &b_mem, &b_off, &size},
                {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t)});
        }
    }
    if (!run_gpu) {
        float* a_ptr = a_c->data_ptr();
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            out_ptr[i] = std::fabs(a_ptr[i]);
        });
        if (out->device.type == DeviceType::GPU) {
            rt_gpu_write(out->gpu_data(), size * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), out_ptr);
        }
    }

    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<AbsNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
        node->saved_versions = {(a ? a->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class NegNode : public Node {
public:
    NegNode() : Node("Neg") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto grad_input = neg(grad_output);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> neg(std::shared_ptr<Tensor> a) {
    if (auto out = try_unary_fp16(a, "neg_forward_fp16",
            [&]() -> std::shared_ptr<Node> { return std::make_shared<NegNode>(); },
            false)) return out;
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = neg(a_fp32);
        return cast(out_fp32, a->dtype);
    }
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    StorageUseGuard guard({a_c->storage, out->storage});
    int64_t size = a_c->numel();
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto kernel = rt_gpu_get_kernel_by_id(KernelID::Neg);
        if (kernel) {
            run_gpu = true;
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int64_t a_off = a_c->offset;
            int64_t b_off = out->offset;
            rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                {&a_mem, &a_off, &b_mem, &b_off, &size},
                {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t)});
        }
    }
    if (!run_gpu) {
        float* a_ptr = a_c->data_ptr();
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            out_ptr[i] = -a_ptr[i];
        });
        if (out->device.type == DeviceType::GPU) {
            rt_gpu_write(out->gpu_data(), size * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), out_ptr);
        }
    }

    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<NegNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = {};
        node->saved_versions = {};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class SumNode : public Node {
public:
    std::vector<int64_t> input_shape;
    SumNode(const std::vector<int64_t>& input_shape) : Node("Sum"), input_shape(input_shape) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto grad_input = Tensor::create(input_shape, grad_output->device, false, grad_output->dtype);
        StorageUseGuard guard({grad_input->storage, grad_output->storage});
        bool run_gpu = false;
        if (grad_output->device.type == DeviceType::GPU) {
            void* kernel = nullptr;
            if (grad_output->dtype == DataType::FP16) {
                kernel = rt_gpu_get_kernel("", "", "sum_backward_fp16");
            } else {
                kernel = rt_gpu_get_kernel_by_id(KernelID::SumBackward);
            }
            if (kernel) {
                run_gpu = true;
                cl_mem gout_mem = grad_output->gpu_data();
                int64_t gout_off = grad_output->offset;
                cl_mem gin_mem = grad_input->gpu_data();
                int64_t gin_off = grad_input->offset;
                int64_t size = grad_input->numel();
                rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                    {&gin_mem, &gin_off, &gout_mem, &gout_off, &size},
                    {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t)});
            }
        }
        if (!run_gpu) {
            float val = grad_output->data_ptr()[0];
            float* ptr = grad_input->data_ptr();
            size_t size = grad_input->numel();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                ptr[i] = val;
            });
            if (grad_input->device.type == DeviceType::GPU) {
                rt_gpu_write(grad_input->gpu_data(), size * sizeof(float), ptr);
            } else if (grad_input->device.type == DeviceType::TPU) {
                auto tpu = BackendDispatcher::get().get_tpu_backend();
                if (tpu) tpu->write(grad_input->gpu_data(), size * sizeof(float), ptr);
            }
        }
        return { grad_input };
    }
};

std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a) {
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = sum(a_fp32);
        return cast(out_fp32, a->dtype);
    }
    auto out = Tensor::create({1}, a->device);
    StorageUseGuard guard({a->storage, out->storage});
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto backend = BackendDispatcher::get().get_backend();
        if (backend && backend->is_available()) {
            run_gpu = true;
            backend->sum(a->gpu_data(), a->offset, out->gpu_data(), out->offset, a->numel());
        } else {
            auto kernel = rt_gpu_get_kernel_by_id(KernelID::SumForward);
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                int a_off = a->offset;
                cl_mem out_mem = out->gpu_data();
                int out_off = out->offset;
                int size = a->numel();
                float zero = 0.0f;
                rt_gpu_write(out_mem, sizeof(float), &zero, out_off);
                size_t blocks = std::min(static_cast<size_t>(32), (static_cast<size_t>(size) + 255) / 256);
                if (blocks < 1) blocks = 1;
                rt_gpu_launch(kernel, {blocks * 256}, {256},
                    {&a_mem, &a_off, &out_mem, &out_off, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
            }
        }
    } else if (a->device.type == DeviceType::TPU) {
        auto tpu = BackendDispatcher::get().get_tpu_backend();
        if (tpu && tpu->is_available()) {
            run_gpu = true;
            tpu->sum(a->gpu_data(), a->offset, out->gpu_data(), out->offset, a->numel());
        }
    }
    if (!run_gpu) {
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        float total = 0.0f;
        float* ptr = a_c->data_ptr();
        size_t size = a_c->numel();
        for (size_t i = 0; i < size; ++i) {
            total += ptr[i];
        }
        float* out_ptr = out->data_ptr();
        out_ptr[0] = total;
        if (out->device.type == DeviceType::GPU) {
            rt_gpu_write(out->gpu_data(), sizeof(float), out_ptr, static_cast<size_t>(out->offset) * sizeof(float));
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), sizeof(float), out_ptr, out->offset);
        }
    }
    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<SumNode>(a->shape);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class MeanNode : public Node {
public:
    std::vector<int64_t> input_shape;
    MeanNode(const std::vector<int64_t>& input_shape) : Node("Mean"), input_shape(input_shape) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto grad_input = Tensor::create(input_shape, grad_output->device, false, grad_output->dtype);
        StorageUseGuard guard({grad_input->storage, grad_output->storage});
        float num_el = static_cast<float>(grad_input->numel());
        bool run_gpu = false;
        if (grad_output->device.type == DeviceType::GPU) {
            void* kernel = nullptr;
            if (grad_output->dtype == DataType::FP16) {
                kernel = rt_gpu_get_kernel("", "", "sum_backward_fp16");
            } else {
                kernel = rt_gpu_get_kernel_by_id(KernelID::SumBackward);
            }
            if (kernel) {
                run_gpu = true;
                cl_mem in_mem = grad_output->gpu_data();
                int64_t in_off = grad_output->offset;
                cl_mem out_mem = grad_input->gpu_data();
                int64_t out_off = grad_input->offset;
                int64_t size = grad_input->numel();
                rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                    {&out_mem, &out_off, &in_mem, &in_off, &size},
                    {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t)});
                
                auto scale_t = Tensor::from_vector({1.0f / num_el}, {1}, grad_output->device);
                grad_input = mul(grad_input, scale_t);
            }
        }
        if (!run_gpu) {
            float val = grad_output->data_ptr()[0] / num_el;
            float* ptr = grad_input->data_ptr();
            size_t size = grad_input->numel();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                ptr[i] = val;
            });
            if (grad_input->device.type == DeviceType::GPU) {
                rt_gpu_write(grad_input->gpu_data(), size * sizeof(float), ptr);
            }
        }
        return { grad_input };
    }
};

std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a) {
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = mean(a_fp32);
        return cast(out_fp32, a->dtype);
    }
    float num_el = static_cast<float>(a->numel());
    auto out = sum(a);
    auto scale = Tensor::from_vector({1.0f / num_el}, {1}, a->device);
    out = mul(out, scale);
    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<MeanNode>(a->shape);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class MaxNode : public Node {
public:
    MaxNode() : Node("Max") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto grad_input = Tensor::zeros(input->shape, grad_output->device, false, grad_output->dtype);
        StorageUseGuard guard({input->storage, grad_input->storage, grad_output->storage});
        
        auto cpu_input = input->to(Device(DeviceType::CPU, 0));
        float* in_ptr = cpu_input->data_ptr();
        float max_val = in_ptr[0];
        size_t max_idx = 0;
        size_t size = input->numel();
        for (size_t i = 1; i < size; ++i) {
            if (std::isnan(in_ptr[i]) || in_ptr[i] > max_val) {
                max_val = in_ptr[i];
                max_idx = i;
            }
        }
        
        if (grad_output->device.type == DeviceType::GPU) {
            auto cpu_gout = grad_output->to(Device(DeviceType::CPU, 0));
            float val = cpu_gout->data_ptr()[0];
            std::vector<float> cpu_gin(size, 0.0f);
            cpu_gin[max_idx] = val;
            rt_gpu_write(grad_input->gpu_data(), size * sizeof(float), cpu_gin.data());
        } else {
            float val = grad_output->data_ptr()[0];
            grad_input->data_ptr()[max_idx] = val;
        }
        return { grad_input };
    }
};

std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a) {
    if (a->numel() == 0) throw std::runtime_error("[litetorch Error] max(): expected non-empty input");
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = max(a_fp32);
        return cast(out_fp32, a->dtype);
    }
    auto out = Tensor::create({1}, a->device);
    StorageUseGuard guard({a->storage, out->storage});

    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto backend = BackendDispatcher::get().get_backend();
        if (backend && backend->is_available()) {
            run_gpu = true;
            backend->max(a->gpu_data(), a->offset, out->gpu_data(), out->offset, a->numel());
        }
    }

    if (!run_gpu) {
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        float* ptr = a_c->data_ptr();
        size_t size = a_c->numel();
        float max_val = (size > 0) ? ptr[0] : -std::numeric_limits<float>::infinity();
        for (size_t i = 1; i < size; ++i) {
            if (std::isnan(ptr[i]) || ptr[i] > max_val) max_val = ptr[i];
        }
        float* out_ptr = out->data_ptr();
        out_ptr[0] = max_val;
        if (out->device.type == DeviceType::GPU) {
            rt_gpu_write(out->gpu_data(), sizeof(float), out_ptr, static_cast<size_t>(out->offset) * sizeof(float));
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), sizeof(float), out_ptr, out->offset);
        }
    }

    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<MaxNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
        node->saved_versions = {(a ? a->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

static std::shared_ptr<Tensor> sum_dim_forward(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim, float scale) {
    int64_t ndim = static_cast<int64_t>(a->shape.size());
    if (ndim == 0) throw std::runtime_error("[litetorch Error] sum over dim of a 0-dim tensor");
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::runtime_error("[litetorch Error] Dimension out of range for sum");
    int64_t outer_size = 1, inner_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= a->shape[i];
    int64_t dim_size = a->shape[dim];
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= a->shape[i];
    std::vector<int64_t> out_shape;
    out_shape.reserve(keepdim ? static_cast<size_t>(ndim) : static_cast<size_t>(ndim - 1));
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(a->shape[i]);
    }
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(out_shape, a->device);
    StorageUseGuard guard({a_c->storage, out->storage});
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto native = BackendDispatcher::get().get_backend();
        if (native && native->is_available()) {
            typedef void (*ReduceDimFn)(void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t);
            auto fn = reinterpret_cast<ReduceDimFn>(native->get_kernel("", "", "gpu_reduce_broadcast_dim"));
            if (fn) {
                run_gpu = true;
                fn(a_c->gpu_data(), a_c->offset, out->gpu_data(), out->offset,
                   outer_size, dim_size, inner_size);
                if (scale != 1.0f) {
                    auto scale_t = Tensor::from_vector({scale}, {1}, a->device);
                    out = mul(out, scale_t);
                }
            }
        }
        if (!run_gpu) {
            auto kernel = rt_gpu_get_kernel_by_id(KernelID::ReduceBroadcastDim);
            if (kernel) {
                run_gpu = true;
                cl_mem in_mem = a_c->gpu_data();
                int64_t in_off = a_c->offset;
                cl_mem out_mem = out->gpu_data();
                int64_t out_off = out->offset;
                int64_t out_sz = outer_size;
                int64_t d_sz = dim_size;
                int64_t in_sz = inner_size;
                int64_t total_threads = outer_size * inner_size;
                rt_gpu_launch(kernel, {static_cast<size_t>(total_threads)}, {},
                    {&in_mem, &in_off, &out_mem, &out_off, &out_sz, &d_sz, &in_sz},
                    {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t), sizeof(int64_t), sizeof(int64_t)});
                if (scale != 1.0f) {
                    auto scale_t = Tensor::from_vector({scale}, {1}, a->device);
                    out = mul(out, scale_t);
                }
            }
        }
    }
    if (!run_gpu) {
        float* src = a_c->data_ptr();
        float* dst = out->data_ptr();
        ThreadPool::get().parallel_for(0, outer_size, [&](int64_t o) {
            for (int64_t in = 0; in < inner_size; ++in) {
                float s = 0.0f;
                for (int64_t d = 0; d < dim_size; ++d) {
                    s += src[(o * dim_size + d) * inner_size + in];
                }
                dst[o * inner_size + in] = s * scale;
            }
        });
        if (a->device.type == DeviceType::GPU) {
            rt_gpu_write(out->gpu_data(), out->numel() * sizeof(float), dst);
        } else if (a->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), out->numel() * sizeof(float), dst);
        }
    }
    return out;
}

class SumDimNode : public Node {
public:
    int64_t dim;
    bool keepdim;
    std::vector<int64_t> input_shape;
    float divisor;
    SumDimNode(int64_t dim, bool keepdim, const std::vector<int64_t>& input_shape, float divisor)
        : Node("SumDim"), dim(dim), keepdim(keepdim), input_shape(input_shape), divisor(divisor) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto cpu_gout = grad_output;
        if (grad_output->device.type != DeviceType::CPU) {
            cpu_gout = grad_output->to(Device(DeviceType::CPU, 0));
        }
        if (!cpu_gout->is_contiguous()) cpu_gout = cpu_gout->contiguous();
        DataType dt = grad_output->dtype;
        auto grad_input = Tensor::create(input_shape, grad_output->device, false, dt);
        StorageUseGuard guard({grad_input->storage, grad_output->storage});
        int64_t ndim = static_cast<int64_t>(input_shape.size());
        int64_t outer_size = 1, inner_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= input_shape[i];
        int64_t dim_size = input_shape[dim];
        for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= input_shape[i];
        float scale = 1.0f / divisor;
        size_t esz = grad_input->storage->element_size();
        if (dt == DataType::FP32) {
            float* g = reinterpret_cast<float*>(grad_input->data_ptr());
            float* gout = reinterpret_cast<float*>(cpu_gout->data_ptr());
            ThreadPool::get().parallel_for(0, outer_size, [&](int64_t o) {
                for (int64_t i = 0; i < inner_size; ++i) {
                    float v = gout[o * inner_size + i] * scale;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        g[(o * dim_size + d) * inner_size + i] = v;
                    }
                }
            });
        } else if (dt == DataType::FP16) {
            uint16_t* g = reinterpret_cast<uint16_t*>(grad_input->data_ptr());
            uint16_t* gout = reinterpret_cast<uint16_t*>(cpu_gout->data_ptr());
            ThreadPool::get().parallel_for(0, outer_size, [&](int64_t o) {
                for (int64_t i = 0; i < inner_size; ++i) {
                    float v = ew_half_to_float(gout[o * inner_size + i]) * scale;
                    uint16_t hv = ew_float_to_half(v);
                    for (int64_t d = 0; d < dim_size; ++d) {
                        g[(o * dim_size + d) * inner_size + i] = hv;
                    }
                }
            });
        } else {
            auto gout_f32 = cpu_gout->dtype == DataType::FP32 ? cpu_gout : cpu_gout->cast(DataType::FP32);
            if (!gout_f32->is_contiguous()) gout_f32 = gout_f32->contiguous();
            auto tmp = Tensor::create(input_shape, grad_output->device);
            float* g = reinterpret_cast<float*>(tmp->data_ptr());
            float* gout = reinterpret_cast<float*>(gout_f32->data_ptr());
            ThreadPool::get().parallel_for(0, outer_size, [&](int64_t o) {
                for (int64_t i = 0; i < inner_size; ++i) {
                    float v = gout[o * inner_size + i] * scale;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        g[(o * dim_size + d) * inner_size + i] = v;
                    }
                }
            });
            auto casted = tmp->cast(dt);
            if (grad_output->device.type == DeviceType::GPU) {
                rt_gpu_write(grad_input->gpu_data(), grad_input->numel() * esz,
                             casted->storage->get_cpu_ptr());
            } else if (grad_output->device.type == DeviceType::TPU) {
                auto tpu = BackendDispatcher::get().get_tpu_backend();
                if (tpu) tpu->write(grad_input->gpu_data(), grad_input->numel() * esz,
                                   casted->storage->get_cpu_ptr());
            } else {
                std::memcpy(grad_input->storage->get_cpu_ptr(), casted->storage->get_cpu_ptr(),
                            grad_input->numel() * esz);
            }
            return { grad_input };
        }
        if (grad_output->device.type == DeviceType::GPU) {
            rt_gpu_write(grad_input->gpu_data(), grad_input->numel() * esz, grad_input->storage->get_cpu_ptr());
        } else if (grad_output->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(grad_input->gpu_data(), grad_input->numel() * esz, grad_input->storage->get_cpu_ptr());
        }
        return { grad_input };
    }
};

std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim) {
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = sum(a_fp32, dim, keepdim);
        return cast(out_fp32, a->dtype);
    }
    auto out = sum_dim_forward(a, dim, keepdim, 1.0f);
    int64_t ndim = static_cast<int64_t>(a->shape.size());
    int64_t d = dim < 0 ? dim + ndim : dim;
    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<SumDimNode>(d, keepdim, a->shape, 1.0f);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim) {
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = mean(a_fp32, dim, keepdim);
        return cast(out_fp32, a->dtype);
    }
    int64_t ndim = static_cast<int64_t>(a->shape.size());
    int64_t d = dim < 0 ? dim + ndim : dim;
    if (d < 0 || d >= ndim) throw std::runtime_error("[litetorch Error] Dimension out of range for mean");
    float dim_size = static_cast<float>(a->shape[d]);
    auto out = sum_dim_forward(a, dim, keepdim, 1.0f / dim_size);
    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<SumDimNode>(d, keepdim, a->shape, dim_size);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class MaxDimNode : public Node {
public:
    int64_t dim;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> argmax;
    MaxDimNode(int64_t dim, const std::vector<int64_t>& input_shape, const std::vector<int64_t>& argmax)
        : Node("MaxDim"), dim(dim), input_shape(input_shape), argmax(argmax) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto cpu_gout = grad_output;
        if (grad_output->device.type != DeviceType::CPU) {
            cpu_gout = grad_output->to(Device(DeviceType::CPU, 0));
        }
        if (!cpu_gout->is_contiguous()) cpu_gout = cpu_gout->contiguous();
        auto grad_input = Tensor::zeros(input_shape, grad_output->device, false, grad_output->dtype);
        StorageUseGuard guard({grad_input->storage, grad_output->storage});
        int64_t ndim = static_cast<int64_t>(input_shape.size());
        int64_t outer_size = 1, inner_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= input_shape[i];
        int64_t dim_size = input_shape[dim];
        for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= input_shape[i];
        float* g = grad_input->data_ptr();
        float* gout = cpu_gout->data_ptr();
        ThreadPool::get().parallel_for(0, outer_size, [&](int64_t o) {
            for (int64_t i = 0; i < inner_size; ++i) {
                int64_t k = o * inner_size + i;
                g[(o * dim_size + argmax[static_cast<size_t>(k)]) * inner_size + i] += gout[k];
            }
        });
        if (grad_output->device.type == DeviceType::GPU) {
            rt_gpu_write(grad_input->gpu_data(), grad_input->numel() * sizeof(float), g);
        } else if (grad_output->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(grad_input->gpu_data(), grad_input->numel() * sizeof(float), g);
        }
        return { grad_input };
    }
};

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> max_with_indices(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim) {
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto res_fp32 = max_with_indices(a_fp32, dim, keepdim);
        return {cast(res_fp32.first, a->dtype), res_fp32.second};
    }
    int64_t ndim = static_cast<int64_t>(a->shape.size());
    if (ndim == 0) throw std::runtime_error("[litetorch Error] max over dim of a 0-dim tensor");
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::runtime_error("[litetorch Error] Dimension out of range for max");
    if (a->numel() == 0) throw std::runtime_error("[litetorch Error] max(): expected non-empty input");
    int64_t outer_size = 1, inner_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= a->shape[i];
    int64_t dim_size = a->shape[dim];
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= a->shape[i];
    std::vector<int64_t> out_shape;
    out_shape.reserve(keepdim ? static_cast<size_t>(ndim) : static_cast<size_t>(ndim - 1));
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(a->shape[i]);
    }
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto cpu_in = a_c;
    if (a->device.type != DeviceType::CPU) {
        cpu_in = a_c->to(Device(DeviceType::CPU, 0));
    }
    if (!cpu_in->is_contiguous()) cpu_in = cpu_in->contiguous();
    auto out = Tensor::create(out_shape, a->device);
    auto indices = Tensor::create(out_shape, a->device, false, DataType::INT64);
    StorageUseGuard guard({a_c->storage, out->storage, indices->storage});
    float* src = cpu_in->data_ptr();
    float* dst = out->data_ptr();
    int64_t* idst = reinterpret_cast<int64_t*>(indices->raw_data_ptr());
    std::vector<int64_t> argmax(static_cast<size_t>(outer_size * inner_size));
    ThreadPool::get().parallel_for(0, outer_size, [&](int64_t o) {
        for (int64_t i = 0; i < inner_size; ++i) {
            float best = src[(o * dim_size) * inner_size + i];
            int64_t best_d = 0;
            for (int64_t d = 1; d < dim_size; ++d) {
                float v = src[(o * dim_size + d) * inner_size + i];
                if (std::isnan(v) || v > best) { best = v; best_d = d; }
            }
            int64_t k = o * inner_size + i;
            dst[k] = best;
            idst[k] = static_cast<int64_t>(best_d);
            argmax[static_cast<size_t>(k)] = best_d;
        }
    });
    if (a->device.type == DeviceType::GPU) {
        rt_gpu_write(out->gpu_data(), out->numel() * sizeof(float), dst);
        rt_gpu_write(indices->gpu_data(), indices->numel() * sizeof(int64_t), idst);
    } else if (a->device.type == DeviceType::TPU) {
        auto tpu = BackendDispatcher::get().get_tpu_backend();
        if (tpu) {
            tpu->write(out->gpu_data(), out->numel() * sizeof(float), dst);
            tpu->write(indices->gpu_data(), indices->numel() * sizeof(int64_t), idst);
        }
    }
    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<MaxDimNode>(dim, a->shape, argmax);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return { out, indices };
}

std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim) {
    return max_with_indices(a, dim, keepdim).first;
}

class CatNode : public Node {
public:
    int64_t dim;
    std::vector<std::vector<int64_t>> input_shapes;
    CatNode(int64_t dim, const std::vector<std::vector<int64_t>>& shapes)
        : Node("Cat"), dim(dim), input_shapes(shapes) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        std::vector<std::shared_ptr<Tensor>> grads;
        int64_t offset = 0;
        int64_t ndim = grad_output->shape.size();
        int64_t concat_dim_size = grad_output->shape[dim];
        DataType dt = grad_output->dtype;
        size_t esz = grad_output->storage->element_size();
        bool run_gpu = false;
        if (grad_output->device.type == DeviceType::GPU && dt == DataType::FP32) {
            auto backend = BackendDispatcher::get().get_backend();
            if (backend && backend->is_available()) {
                run_gpu = true;
                for (size_t i = 0; i < input_shapes.size(); ++i) {
                    auto& shape = input_shapes[i];
                    auto grad_input = Tensor::create(shape, grad_output->device, false, dt);
                    int64_t outer_size = 1;
                    for (int64_t d = 0; d < dim; ++d) outer_size *= shape[d];
                    int64_t inner_size = 1;
                    for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= shape[d];
                    int64_t dim_size = shape[dim];

                    backend->cat_backward(grad_output->gpu_data(), grad_output->offset, grad_input->gpu_data(), grad_input->offset, outer_size, inner_size, dim_size, concat_dim_size, offset);
                    grads.push_back(grad_input);
                    offset += dim_size;
                }
            }
        }

        if (!run_gpu) {
            auto cpu_gout = grad_output->to(Device(DeviceType::CPU, 0));
            if (!cpu_gout->is_contiguous()) cpu_gout = cpu_gout->contiguous();
            const char* src_base = reinterpret_cast<const char*>(cpu_gout->storage->get_cpu_ptr())
                + cpu_gout->offset * esz;

            for (size_t i = 0; i < input_shapes.size(); ++i) {
                auto& shape = input_shapes[i];
                auto grad_input = Tensor::create(shape, grad_output->device, false, dt);
                auto cpu_gin = Tensor::create(shape, Device(DeviceType::CPU, 0), false, dt);
                char* dst_base = reinterpret_cast<char*>(cpu_gin->storage->get_cpu_ptr());

                int64_t outer_size = 1;
                for (int64_t d = 0; d < dim; ++d) outer_size *= shape[d];
                int64_t inner_size = 1;
                for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= shape[d];
                int64_t dim_size = shape[dim];

                ThreadPool::get().parallel_for(0, outer_size, [&](int64_t o) {
                    for (int64_t d = 0; d < dim_size; ++d) {
                        int64_t src_idx = (o * concat_dim_size + (offset + d)) * inner_size;
                        int64_t dst_idx = (o * dim_size + d) * inner_size;
                        std::memcpy(dst_base + dst_idx * esz, src_base + src_idx * esz,
                                    static_cast<size_t>(inner_size) * esz);
                    }
                });

                grad_input->copy_(cpu_gin);
                grads.push_back(grad_input);
                offset += dim_size;
            }
        }
        return grads;
    }
};

std::shared_ptr<Tensor> cat(const std::vector<std::shared_ptr<Tensor>>& tensors, int64_t dim) {
    if (!tensors.empty()) {
        DataType out_dtype = tensors[0]->dtype;
        bool all_same = true;
        for (auto& t : tensors) {
            out_dtype = promote_types(out_dtype, t->dtype);
            if (t->dtype != tensors[0]->dtype) all_same = false;
        }
        if (!all_same) {
            std::vector<std::shared_ptr<Tensor>> casted;
            casted.reserve(tensors.size());
            for (auto& t : tensors) casted.push_back(cast(t, out_dtype));
            return cat(casted, dim);
        }
    }
    if (tensors.empty()) {
        throw std::runtime_error("[litetorch Error] cat requires at least one tensor");
    }

    int64_t ndim = tensors[0]->shape.size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("[litetorch Error] Dimension out of range for cat");
    }

    std::vector<int64_t> out_shape = tensors[0]->shape;
    int64_t concat_dim_size = 0;
    bool any_requires_grad = false;
    std::vector<std::shared_ptr<StorageImpl>> storages;
    std::vector<std::vector<int64_t>> shapes;

    for (auto& t : tensors) {
        if (t->shape.size() != static_cast<size_t>(ndim)) {
            throw std::runtime_error("[litetorch Error] All tensors must have the same number of dimensions");
        }
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != dim && t->shape[i] != out_shape[i]) {
                throw std::runtime_error("[litetorch Error] Tensor shapes must match except in the concatenating dimension");
            }
        }
        concat_dim_size += t->shape[dim];
        any_requires_grad = any_requires_grad || t->requires_grad;
        storages.push_back(t->storage);
        shapes.push_back(t->shape);
    }
    out_shape[dim] = concat_dim_size;

    auto out = Tensor::create(out_shape, tensors[0]->device, false, tensors[0]->dtype);
    storages.push_back(out->storage);
    std::vector<std::shared_ptr<Tensor>> inputs;
    inputs.reserve(tensors.size());
    for (auto& t : tensors) {
        auto tc = t->is_contiguous() ? t : t->contiguous();
        inputs.push_back(tc);
        storages.push_back(tc->storage);
    }
    StorageUseGuard guard(storages);

    bool run_gpu = false;
    int64_t offset = 0;
    if (out->device.type == DeviceType::GPU) {
        auto backend = BackendDispatcher::get().get_backend();
        if (backend && backend->is_available()) {
            run_gpu = true;
            for (auto& t : inputs) {
                int64_t outer_size = 1;
                for (int64_t d = 0; d < dim; ++d) outer_size *= t->shape[d];
                int64_t inner_size = 1;
                for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= t->shape[d];
                int64_t dim_size = t->shape[dim];
                backend->cat_forward(t->gpu_data(), t->offset, out->gpu_data(), out->offset, outer_size, inner_size, dim_size, concat_dim_size, offset);
                offset += dim_size;
            }
        }
    }

    if (!run_gpu) {
        auto cpu_out = out->to(Device(DeviceType::CPU, 0));
        size_t esz = cpu_out->storage->element_size();
        char* dst_ptr = reinterpret_cast<char*>(cpu_out->raw_data_ptr());
        offset = 0;

        for (auto& t : inputs) {
            auto cpu_t = t->to(Device(DeviceType::CPU, 0));
            const char* src_ptr = reinterpret_cast<const char*>(cpu_t->raw_data_ptr());
            int64_t outer_size = 1;
            for (int64_t d = 0; d < dim; ++d) outer_size *= t->shape[d];
            int64_t inner_size = 1;
            for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= t->shape[d];
            int64_t dim_size = t->shape[dim];

            ThreadPool::get().parallel_for(0, outer_size, [&](int64_t o) {
                for (int64_t d = 0; d < dim_size; ++d) {
                    for (int64_t in = 0; in < inner_size; ++in) {
                        int64_t dst_idx = (o * concat_dim_size + (offset + d)) * inner_size + in;
                        int64_t src_idx = (o * dim_size + d) * inner_size + in;
                        std::memcpy(dst_ptr + dst_idx * esz, src_ptr + src_idx * esz, esz);
                    }
                }
            });
            offset += dim_size;
        }
        out->copy_(cpu_out);
    }

    if (Autograd::is_grad_enabled() && (any_requires_grad)) {
        auto node = std::make_shared<CatNode>(dim, shapes);
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
        for (auto& t : tensors) {
            node->inputs.push_back({t, t->requires_grad});
            node->next_nodes.push_back(t->creator);
        }
    }
    return out;
}

class NarrowNode : public Node {
public:
    int64_t dim;
    int64_t start;
    int64_t length;
    std::vector<int64_t> input_shape;
    NarrowNode(int64_t dim, int64_t start, int64_t length, const std::vector<int64_t>& input_shape)
        : Node("Narrow"), dim(dim), start(start), length(length), input_shape(input_shape) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto grad_input = Tensor::zeros(input_shape, grad_output->device, false, grad_output->dtype);
        auto region = grad_input->slice(dim, start, start + length, 1);
        region->copy_(grad_output);
        return { grad_input };
    }
};

std::vector<std::shared_ptr<Tensor>> split(std::shared_ptr<Tensor> input, const std::vector<int64_t>& sizes, int64_t dim) {
    int64_t ndim = static_cast<int64_t>(input->shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("[litetorch Error] Dimension out of range for split");
    }
    int64_t total = 0;
    for (auto s : sizes) {
        if (s <= 0) throw std::runtime_error("[litetorch Error] split sizes must be positive");
        total += s;
    }
    if (total != input->shape[dim]) {
        throw std::runtime_error("[litetorch Error] split sizes must sum to dimension size");
    }
    std::vector<std::shared_ptr<Tensor>> outputs;
    outputs.reserve(sizes.size());
    int64_t start = 0;
    bool grad_enabled = Autograd::is_grad_enabled() && input->requires_grad;
    for (auto s : sizes) {
        auto out = input->slice(dim, start, start + s, 1);
        if (grad_enabled) {
            auto node = std::make_shared<NarrowNode>(dim, start, s, input->shape);
            node->inputs.push_back({input, true});
            node->next_nodes.push_back(input->creator);
            node->output = out;
            out->creator = node;
            out->requires_grad = true;
        }
        outputs.push_back(out);
        start += s;
    }
    return outputs;
}

std::vector<std::shared_ptr<Tensor>> chunk(std::shared_ptr<Tensor> input, int64_t chunks, int64_t dim) {
    int64_t ndim = static_cast<int64_t>(input->shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("[litetorch Error] Dimension out of range for chunk");
    }
    if (chunks <= 0) {
        throw std::runtime_error("[litetorch Error] chunk expects at least a 1-dimensional tensor and chunks > 0");
    }
    int64_t dim_size = input->shape[dim];
    if (dim_size == 0) {
        auto out = input->slice(dim, 0, 0, 1);
        if (Autograd::is_grad_enabled() && input->requires_grad) {
            auto node = std::make_shared<NarrowNode>(dim, 0, 0, input->shape);
            node->inputs.push_back({input, true});
            node->next_nodes.push_back(input->creator);
            node->output = out;
            out->creator = node;
            out->requires_grad = true;
        }
        return {out};
    }
    int64_t chunk_size = (dim_size + chunks - 1) / chunks;
    std::vector<int64_t> sizes;
    sizes.reserve(static_cast<size_t>(chunks));
    for (int64_t start = 0; start < dim_size; start += chunk_size) {
        sizes.push_back(std::min(chunk_size, dim_size - start));
    }
    return split(input, sizes, dim);
}

class SqueezeNode : public Node {
public:
    std::vector<int64_t> input_shape;
    std::vector<int64_t> input_strides;
    SqueezeNode(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& input_strides) : Node("Squeeze"), input_shape(input_shape), input_strides(input_strides) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto grad_input = std::make_shared<Tensor>(grad_output->storage, input_shape, input_strides, grad_output->offset, grad_output->device, false);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> squeeze(std::shared_ptr<Tensor> a, int64_t dim) {
    std::vector<int64_t> new_shape;
    std::vector<int64_t> new_strides;
    int64_t ndim = static_cast<int64_t>(a->shape.size());
    if (dim == SQUEEZE_ALL_DIMS) {
        for (int64_t i = 0; i < ndim; ++i) {
            if (a->shape[i] != 1) {
                new_shape.push_back(a->shape[i]);
                new_strides.push_back(a->strides[i]);
            }
        }
    } else {
        if (dim < 0) dim += ndim;
        if (dim < 0 || dim >= ndim) throw std::runtime_error("[litetorch Error] Dimension out of range for squeeze");
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != dim || a->shape[i] != 1) {
                new_shape.push_back(a->shape[i]);
                new_strides.push_back(a->strides[i]);
            }
        }
    }

    auto out = std::make_shared<Tensor>(a->storage, new_shape, new_strides, a->offset, a->device, a->requires_grad);
    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<SqueezeNode>(a->shape, a->strides);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->output = out;
        out->creator = node;
    }
    return out;
}

class UnsqueezeNode : public Node {
public:
    std::vector<int64_t> input_shape;
    std::vector<int64_t> input_strides;
    UnsqueezeNode(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& input_strides) : Node("Unsqueeze"), input_shape(input_shape), input_strides(input_strides) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto grad_input = std::make_shared<Tensor>(grad_output->storage, input_shape, input_strides, grad_output->offset, grad_output->device, false);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> unsqueeze(std::shared_ptr<Tensor> a, int64_t dim) {
    int64_t ndim = a->shape.size();
    if (dim < 0) dim += ndim + 1;
    if (dim < 0 || dim > ndim) throw std::runtime_error("[litetorch Error] Dimension out of range for unsqueeze");

    std::vector<int64_t> new_shape;
    std::vector<int64_t> new_strides;
    for (int64_t i = 0; i <= ndim; ++i) {
        if (i == dim) {
            new_shape.push_back(1);
            new_strides.push_back(i < ndim ? a->strides[i] : 1);
        }
        if (i < ndim) {
            new_shape.push_back(a->shape[i]);
            new_strides.push_back(a->strides[i]);
        }
    }

    auto out = std::make_shared<Tensor>(a->storage, new_shape, new_strides, a->offset, a->device, a->requires_grad);
    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<UnsqueezeNode>(a->shape, a->strides);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->output = out;
        out->creator = node;
    }
    return out;
}

class ClampNode : public Node {
public:
    float min_val;
    float max_val;
    ClampNode(float min_val, float max_val) : Node("Clamp"), min_val(min_val), max_val(max_val) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto mask = Tensor::create(input_c->shape, Device(DeviceType::CPU, 0));
        StorageUseGuard guard({input_c->storage, mask->storage});
        float* in_ptr = input_c->data_ptr();
        float* m_ptr = mask->data_ptr();
        for (size_t i = 0; i < input_c->numel(); ++i) {
            m_ptr[i] = (in_ptr[i] >= min_val && in_ptr[i] <= max_val) ? 1.0f : 0.0f;
        }
        if (input->device.type != DeviceType::CPU) {
            mask = mask->to(input->device);
        }
        auto grad_input = mul(grad_output, mask);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> clamp(std::shared_ptr<Tensor> a, float min_val, float max_val) {
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = clamp(a_fp32, min_val, max_val);
        return cast(out_fp32, a->dtype);
    }
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    StorageUseGuard guard({a_c->storage, out->storage});
    float* src = a_c->data_ptr();
    float* dst = out->data_ptr();
    size_t size = out->numel();
    ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
        float x = src[i];
        dst[i] = x < min_val ? min_val : (x > max_val ? max_val : x);
    });
    if (out->device.type == DeviceType::GPU) {
        rt_gpu_write(out->gpu_data(), size * sizeof(float), dst);
    } else if (out->device.type == DeviceType::TPU) {
        auto tpu = BackendDispatcher::get().get_tpu_backend();
        if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), dst);
    }
    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<ClampNode>(min_val, max_val);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
        node->saved_versions = {(a ? a->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class SinNode : public Node {
public:
    SinNode() : Node("Sin") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto grad_input = mul(grad_output, cos(input));
        return { grad_input };
    }
};

std::shared_ptr<Tensor> sin(std::shared_ptr<Tensor> a) {
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = sin(a_fp32);
        return cast(out_fp32, a->dtype);
    }
    auto out = Tensor::create(a->shape, a->device);
    StorageUseGuard guard({a->storage, out->storage});
    float* src = a->data_ptr();
    float* dst = out->data_ptr();
    size_t size = out->numel();
    ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
        dst[i] = std::sin(src[i]);
    });
    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<SinNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
        node->saved_versions = {(a ? a->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class CosNode : public Node {
public:
    CosNode() : Node("Cos") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto neg_sin = mul(sin(input), Tensor::from_vector({-1.0f}, {1}, input->device));
        auto grad_input = mul(grad_output, neg_sin);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> cos(std::shared_ptr<Tensor> a) {
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = cos(a_fp32);
        return cast(out_fp32, a->dtype);
    }
    auto out = Tensor::create(a->shape, a->device);
    StorageUseGuard guard({a->storage, out->storage});
    float* src = a->data_ptr();
    float* dst = out->data_ptr();
    size_t size = out->numel();
    ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
        dst[i] = std::cos(src[i]);
    });
    if (Autograd::is_grad_enabled() && (a->requires_grad)) {
        auto node = std::make_shared<CosNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
        node->saved_versions = {(a ? a->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

// === W1-T1 core-ops ===

namespace {

void gpu_fail(const char* op) {
    throw std::runtime_error(std::string("[litetorch Error] ") + op + ": GPU kernel not implemented (fail-loud)");
}

std::shared_ptr<Tensor> unary_fp32_cpu(std::shared_ptr<Tensor> a, const char* op_name,
        const std::function<float(float)>& fn,
        const std::function<std::shared_ptr<Node>()>& make_node,
        bool differentiable) {
    if (a->device.type == DeviceType::GPU) gpu_fail(op_name);
    if (a->dtype != DataType::FP32) {
        auto a_fp32 = cast(a, DataType::FP32);
        auto out_fp32 = unary_fp32_cpu(a_fp32, op_name, fn, make_node, differentiable);
        return cast(out_fp32, a->dtype);
    }
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    StorageUseGuard guard({a_c->storage, out->storage});
    float* src = a_c->data_ptr();
    float* dst = out->data_ptr();
    size_t size = out->numel();
    ThreadPool::get().parallel_for(0, size, [&](int64_t i) { dst[i] = fn(src[i]); });
    if (out->device.type == DeviceType::TPU) {
        auto tpu = BackendDispatcher::get().get_tpu_backend();
        if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), dst);
    }
    if (differentiable && Autograd::is_grad_enabled() && a->requires_grad && make_node) {
        auto node = make_node();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a, out };
        node->saved_versions = {(a ? a->_version() : 0), (out ? out->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> scalar_tensor(float v, const Device& device) {
    return Tensor::from_vector({v}, {1}, device);
}

std::vector<int64_t> norm_dims_asc(const std::vector<int64_t>& dims, int64_t ndim, const char* op) {
    std::vector<int64_t> ds;
    for (auto d : dims) {
        int64_t nd = d < 0 ? d + ndim : d;
        if (nd < 0 || nd >= ndim) throw std::runtime_error(std::string("[litetorch Error] ") + op + ": dimension out of range");
        ds.push_back(nd);
    }
    std::sort(ds.begin(), ds.end());
    ds.erase(std::unique(ds.begin(), ds.end()), ds.end());
    return ds;
}

std::vector<int64_t> bcast_strides_w1(const std::vector<int64_t>& tshape, const std::vector<int64_t>& out_shape) {
    auto tstr = default_strides(tshape);
    int64_t diff = (int64_t)out_shape.size() - (int64_t)tshape.size();
    std::vector<int64_t> s(out_shape.size(), 0);
    for (size_t i = 0; i < out_shape.size(); ++i) {
        int64_t td = (int64_t)i - diff;
        if (td < 0) continue;
        s[i] = (tshape[(size_t)td] == 1) ? 0 : tstr[(size_t)td];
    }
    return s;
}

}

class Log2Node : public Node {
public:
    Log2Node() : Node("Log2") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto denom = mul(input, scalar_tensor(0.6931471805599453f, input->device));
        return { div(grad_output, denom) };
    }
};

class Log10Node : public Node {
public:
    Log10Node() : Node("Log10") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto denom = mul(input, scalar_tensor(2.302585092994046f, input->device));
        return { div(grad_output, denom) };
    }
};

class RsqrtNode : public Node {
public:
    RsqrtNode() : Node("Rsqrt") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        if (saved_tensors.size() <= 1 || !saved_tensors[1]) throw std::runtime_error("[litetorch Error] Rsqrt backward: saved output tensor missing");
        auto out = saved_tensors[1];
        auto y3 = mul(mul(out, out), out);
        return { mul(grad_output, mul(scalar_tensor(-0.5f, out->device), y3)) };
    }
};

class TanNode : public Node {
public:
    TanNode() : Node("Tan") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        if (saved_tensors.size() <= 1 || !saved_tensors[1]) throw std::runtime_error("[litetorch Error] Tan backward: saved output tensor missing");
        auto out = saved_tensors[1];
        auto sec2 = add(scalar_tensor(1.0f, out->device), mul(out, out));
        return { mul(grad_output, sec2) };
    }
};

class ReciprocalNode : public Node {
public:
    ReciprocalNode() : Node("Reciprocal") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        if (saved_tensors.size() <= 1 || !saved_tensors[1]) throw std::runtime_error("[litetorch Error] Reciprocal backward: saved output tensor missing");
        auto out = saved_tensors[1];
        return { mul(grad_output, neg(mul(out, out))) };
    }
};

class ClampMinNode : public Node {
public:
    float min_val;
    ClampMinNode(float v) : Node("ClampMin"), min_val(v) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto mask = Tensor::create(input_c->shape, input->device);
        float* in_ptr = input_c->data_ptr();
        float* m_ptr = mask->data_ptr();
        for (size_t i = 0; i < input_c->numel(); ++i) m_ptr[i] = (in_ptr[i] >= min_val) ? 1.0f : 0.0f;
        return { mul(grad_output, mask) };
    }
};

class ClampMaxNode : public Node {
public:
    float max_val;
    ClampMaxNode(float v) : Node("ClampMax"), max_val(v) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto mask = Tensor::create(input_c->shape, input->device);
        float* in_ptr = input_c->data_ptr();
        float* m_ptr = mask->data_ptr();
        for (size_t i = 0; i < input_c->numel(); ++i) m_ptr[i] = (in_ptr[i] <= max_val) ? 1.0f : 0.0f;
        return { mul(grad_output, mask) };
    }
};

std::shared_ptr<Tensor> log2(std::shared_ptr<Tensor> a) {
    return unary_fp32_cpu(a, "log2", [](float x) { return std::log2(x); },
        []() -> std::shared_ptr<Node> { return std::make_shared<Log2Node>(); }, true);
}

std::shared_ptr<Tensor> log10(std::shared_ptr<Tensor> a) {
    return unary_fp32_cpu(a, "log10", [](float x) { return std::log10(x); },
        []() -> std::shared_ptr<Node> { return std::make_shared<Log10Node>(); }, true);
}

std::shared_ptr<Tensor> rsqrt(std::shared_ptr<Tensor> a) {
    return unary_fp32_cpu(a, "rsqrt", [](float x) { return 1.0f / std::sqrt(x); },
        []() -> std::shared_ptr<Node> { return std::make_shared<RsqrtNode>(); }, true);
}

std::shared_ptr<Tensor> tan(std::shared_ptr<Tensor> a) {
    return unary_fp32_cpu(a, "tan", [](float x) { return std::tan(x); },
        []() -> std::shared_ptr<Node> { return std::make_shared<TanNode>(); }, true);
}

std::shared_ptr<Tensor> reciprocal(std::shared_ptr<Tensor> a) {
    return unary_fp32_cpu(a, "reciprocal", [](float x) { return 1.0f / x; },
        []() -> std::shared_ptr<Node> { return std::make_shared<ReciprocalNode>(); }, true);
}

std::shared_ptr<Tensor> sign(std::shared_ptr<Tensor> a) {
    return unary_fp32_cpu(a, "sign", [](float x) { return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f); },
        nullptr, false);
}

std::shared_ptr<Tensor> floor(std::shared_ptr<Tensor> a) {
    return unary_fp32_cpu(a, "floor", [](float x) { return std::floor(x); }, nullptr, false);
}

std::shared_ptr<Tensor> ceil(std::shared_ptr<Tensor> a) {
    return unary_fp32_cpu(a, "ceil", [](float x) { return std::ceil(x); }, nullptr, false);
}

std::shared_ptr<Tensor> round(std::shared_ptr<Tensor> a) {
    return unary_fp32_cpu(a, "round", [](float x) { return std::nearbyint(x); }, nullptr, false);
}

std::shared_ptr<Tensor> trunc(std::shared_ptr<Tensor> a) {
    return unary_fp32_cpu(a, "trunc", [](float x) { return std::trunc(x); }, nullptr, false);
}

std::shared_ptr<Tensor> clamp_min(std::shared_ptr<Tensor> a, float min_val) {
    return unary_fp32_cpu(a, "clamp_min", [min_val](float x) { return x < min_val ? min_val : x; },
        [min_val]() -> std::shared_ptr<Node> { return std::make_shared<ClampMinNode>(min_val); }, true);
}

std::shared_ptr<Tensor> clamp_max(std::shared_ptr<Tensor> a, float max_val) {
    return unary_fp32_cpu(a, "clamp_max", [max_val](float x) { return x > max_val ? max_val : x; },
        [max_val]() -> std::shared_ptr<Node> { return std::make_shared<ClampMaxNode>(max_val); }, true);
}

std::shared_ptr<Tensor> min(std::shared_ptr<Tensor> a) {
    return neg(max(neg(a)));
}

std::shared_ptr<Tensor> amax(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim) {
    auto ds = norm_dims_asc(dims, (int64_t)a->shape.size(), "amax");
    auto out = a;
    for (auto d : ds) out = max(out, d, true);
    if (!keepdim) for (auto it = ds.rbegin(); it != ds.rend(); ++it) out = squeeze(out, *it);
    return out;
}

std::shared_ptr<Tensor> amin(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim) {
    return neg(amax(neg(a), dims, keepdim));
}

std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim) {
    auto ds = norm_dims_asc(dims, (int64_t)a->shape.size(), "sum");
    if (ds.empty()) return a;
    auto out = a;
    for (auto d : ds) out = sum(out, d, true);
    if (!keepdim) for (auto it = ds.rbegin(); it != ds.rend(); ++it) out = squeeze(out, *it);
    return out;
}

std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim) {
    auto ds = norm_dims_asc(dims, (int64_t)a->shape.size(), "mean");
    if (ds.empty()) return a;
    auto out = a;
    for (auto d : ds) out = mean(out, d, true);
    if (!keepdim) for (auto it = ds.rbegin(); it != ds.rend(); ++it) out = squeeze(out, *it);
    return out;
}

std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim) {
    auto ds = norm_dims_asc(dims, (int64_t)a->shape.size(), "max");
    if (ds.empty()) return a;
    auto out = a;
    for (auto d : ds) out = max(out, d, true);
    if (!keepdim) for (auto it = ds.rbegin(); it != ds.rend(); ++it) out = squeeze(out, *it);
    return out;
}

namespace {

bool close_enough(float x, float y, float rtol, float atol, bool equal_nan) {
    if (std::isnan(x) || std::isnan(y)) return equal_nan && std::isnan(x) && std::isnan(y);
    if (std::isinf(x) || std::isinf(y)) return x == y;
    return std::fabs(x - y) <= atol + rtol * std::fabs(y);
}

}

bool allclose(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, float rtol, float atol, bool equal_nan) {
    if (a->device.type == DeviceType::GPU || b->device.type == DeviceType::GPU) gpu_fail("allclose");
    if (a->device != b->device) b = b->to(a->device);
    auto shape = broadcast_shapes(a->shape, b->shape);
    auto sa = bcast_strides_w1(a->shape, shape);
    auto sb = bcast_strides_w1(b->shape, shape);
    std::vector<float> da = a->to_vector();
    std::vector<float> db = b->to_vector();
    size_t n = 1;
    for (auto s : shape) n *= (size_t)s;
    std::vector<int64_t> coord(shape.size(), 0);
    for (size_t i = 0; i < n; ++i) {
        size_t rem = i;
        int64_t ia = 0, ib = 0;
        for (int64_t d = (int64_t)shape.size() - 1; d >= 0; --d) {
            int64_t c = (int64_t)(rem % (size_t)shape[(size_t)d]);
            rem /= (size_t)shape[(size_t)d];
            ia += c * sa[(size_t)d];
            ib += c * sb[(size_t)d];
        }
        if (!close_enough(da[(size_t)ia], db[(size_t)ib], rtol, atol, equal_nan)) return false;
    }
    return true;
}

std::shared_ptr<Tensor> isclose(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, float rtol, float atol, bool equal_nan) {
    if (a->device.type == DeviceType::GPU || b->device.type == DeviceType::GPU) gpu_fail("isclose");
    if (a->device != b->device) b = b->to(a->device);
    auto shape = broadcast_shapes(a->shape, b->shape);
    auto sa = bcast_strides_w1(a->shape, shape);
    auto sb = bcast_strides_w1(b->shape, shape);
    std::vector<float> da = a->to_vector();
    std::vector<float> db = b->to_vector();
    size_t n = 1;
    for (auto s : shape) n *= (size_t)s;
    auto out = Tensor::create(shape, a->device);
    float* dst = out->data_ptr();
    for (size_t i = 0; i < n; ++i) {
        size_t rem = i;
        int64_t ia = 0, ib = 0;
        for (int64_t d = (int64_t)shape.size() - 1; d >= 0; --d) {
            int64_t c = (int64_t)(rem % (size_t)shape[(size_t)d]);
            rem /= (size_t)shape[(size_t)d];
            ia += c * sa[(size_t)d];
            ib += c * sb[(size_t)d];
        }
        dst[i] = close_enough(da[(size_t)ia], db[(size_t)ib], rtol, atol, equal_nan) ? 1.0f : 0.0f;
    }
    return out;
}

// === end W1-T1 core-ops ===

}
}
