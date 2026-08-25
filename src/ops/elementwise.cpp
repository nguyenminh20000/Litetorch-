#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace litetorch {

extern const std::string litetorch_kernels_src;

namespace {
struct StorageUseGuard {
    std::vector<std::shared_ptr<StorageImpl>> storages;
    StorageUseGuard(const std::vector<std::shared_ptr<StorageImpl>>& list) : storages(list) {
        for (auto& s : storages) {
            if (s) s->in_use = true;
        }
    }
    ~StorageUseGuard() {
        for (auto& s : storages) {
            if (s) s->in_use = false;
        }
    }
};

cl_mem create_gpu_int_buffer(const std::vector<int64_t>& vec) {
    std::vector<int> int_vec(vec.begin(), vec.end());
    size_t size_bytes = int_vec.size() * sizeof(int);
    cl_mem buf = CLBackend::get().allocate(size_bytes);
    CLBackend::get().write(buf, size_bytes, int_vec.data());
    return buf;
}

struct TemporaryGPUBuffers {
    std::vector<cl_mem> buffers;
    cl_mem add(const std::vector<int64_t>& vec) {
        cl_mem buf = create_gpu_int_buffer(vec);
        buffers.push_back(buf);
        return buf;
    }
    ~TemporaryGPUBuffers() {
        for (auto buf : buffers) {
            if (buf) {
                CLBackend::get().free(buf);
            }
        }
    }
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
std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> cat(const std::vector<std::shared_ptr<Tensor>>& tensors, int64_t dim);
std::shared_ptr<Tensor> squeeze(std::shared_ptr<Tensor> a, int64_t dim);
std::shared_ptr<Tensor> unsqueeze(std::shared_ptr<Tensor> a, int64_t dim);
std::shared_ptr<Tensor> clamp(std::shared_ptr<Tensor> a, float min_val, float max_val);
std::shared_ptr<Tensor> sin(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> cos(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> reduce_broadcast(std::shared_ptr<Tensor> grad, const std::vector<int64_t>& orig_shape);

std::shared_ptr<Tensor> reduce_broadcast(std::shared_ptr<Tensor> grad, const std::vector<int64_t>& orig_shape) {
    if (grad->shape == orig_shape) return grad;

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
            auto kernel = CLBackend::get().get_kernel(KernelID::ReduceBroadcastPrepended);
            if (kernel) {
                run_gpu = true;
                auto reshaped = current->view({prod_prepended, remaining});
                auto summed = Tensor::create({remaining}, grad->device);
                StorageUseGuard guard({reshaped->storage, summed->storage});
                
                cl_mem in_mem = reshaped->gpu_data();
                int in_off = reshaped->offset;
                cl_mem out_mem = summed->gpu_data();
                int out_off = summed->offset;
                int prod_val = prod_prepended;
                int rem_val = remaining;

                CLBackend::get().launch(kernel, {static_cast<size_t>(remaining)}, {},
                    {&in_mem, &in_off, &out_mem, &out_off, &prod_val, &rem_val},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int)});

                current = summed->view(final_shape);
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
                CLBackend::get().write(gpu_summed->gpu_data(), remaining * sizeof(float), res_ptr);
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
                auto kernel = CLBackend::get().get_kernel(KernelID::ReduceBroadcastDim);
                if (kernel) {
                    run_gpu = true;
                    auto summed = Tensor::create(new_shape, grad->device);
                    StorageUseGuard guard({current->storage, summed->storage});

                    cl_mem in_mem = current->gpu_data();
                    int in_off = current->offset;
                    cl_mem out_mem = summed->gpu_data();
                    int out_off = summed->offset;
                    int out_sz = outer_size;
                    int d_sz = dim_size;
                    int in_sz = inner_size;

                    int total_threads = outer_size * inner_size;
                    CLBackend::get().launch(kernel, {static_cast<size_t>(total_threads)}, {},
                        {&in_mem, &in_off, &out_mem, &out_off, &out_sz, &d_sz, &in_sz},
                        {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});

                    current = summed;
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
                    CLBackend::get().write(gpu_summed->gpu_data(), gpu_summed->numel() * sizeof(float), dst);
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
        auto grad_a = reduce_broadcast(grad_output, shape_a);
        auto grad_b = reduce_broadcast(neg(grad_output), shape_b);
        return { grad_a, grad_b };
    }
};

class MulNode : public Node {
public:
    MulNode() : Node("Mul") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
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
        auto a = saved_tensors[0];
        auto b = saved_tensors[1];
        auto grad_a = reduce_broadcast(div(grad_output, b), a->shape);
        auto b2 = mul(b, b);
        auto neg_a = mul(a, Tensor::from_vector({-1.0f}, {1}, a->device));
        auto grad_b = reduce_broadcast(mul(grad_output, div(neg_a, b2)), b->shape);
        return { grad_a, grad_b };
    }
};

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
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
            auto kernel = CLBackend::get().get_kernel(KernelID::Add);
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int a_off = a->offset;
                int b_off = b->offset;
                int c_off = out->offset;
                int size = out->numel();
                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
            }
        } else {
            auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "elementwise_broadcast_add");
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int a_off = a->offset;
                int b_off = b->offset;
                int c_off = out->offset;

                TemporaryGPUBuffers temp_bufs;
                cl_mem out_shape_mem = temp_bufs.add(out_shape);
                cl_mem a_shape_mem = temp_bufs.add(a->shape);
                cl_mem a_strides_mem = temp_bufs.add(a->strides);
                cl_mem b_shape_mem = temp_bufs.add(b->shape);
                cl_mem b_strides_mem = temp_bufs.add(b->strides);

                int ndims = out_shape.size();
                int a_ndims = a->shape.size();
                int b_ndims = b->shape.size();
                int size = out->numel();

                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &ndims, &out_shape_mem,
                     &a_ndims, &a_shape_mem, &a_strides_mem,
                     &b_ndims, &b_shape_mem, &b_strides_mem, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int),
                     sizeof(int), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem), sizeof(int)});
            }
        }
    }
    if (!run_gpu) {
        float* a_ptr = a->data_ptr();
        float* b_ptr = b->data_ptr();
        float* out_ptr = out->data_ptr();
        if (a->shape == b->shape && a->is_contiguous() && b->is_contiguous()) {
            int size = out->numel();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                out_ptr[i] = a_ptr[i] + b_ptr[i];
            });
        } else {
            int size = out->numel();
            int ndims = out_shape.size();
            std::vector<int64_t> out_strides(ndims);
            int64_t current = 1;
            for (int i = ndims - 1; i >= 0; --i) {
                out_strides[i] = current;
                current *= out_shape[i];
            }
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                int temp = i;
                int a_idx = 0;
                int b_idx = 0;
                int a_diff = ndims - a->shape.size();
                int b_diff = ndims - b->shape.size();
                for (int d = 0; d < ndims; ++d) {
                    int coord = temp / out_strides[d];
                    temp %= out_strides[d];
                    if (d >= a_diff) {
                        int a_dim_idx = d - a_diff;
                        if (a->shape[a_dim_idx] != 1) {
                            a_idx += coord * a->strides[a_dim_idx];
                        }
                    }
                    if (d >= b_diff) {
                        int b_dim_idx = d - b_diff;
                        if (b->shape[b_dim_idx] != 1) {
                            b_idx += coord * b->strides[b_dim_idx];
                        }
                    }
                }
                out_ptr[i] = a_ptr[a_idx] + b_ptr[b_idx];
            });
        }
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        }
    }

    if (a->requires_grad || b->requires_grad) {
        auto node = std::make_shared<AddNode>(a->shape, b->shape);
        node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
        node->next_nodes = { a->creator, b->creator };
        node->saved_tensors = {};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
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
            auto kernel = CLBackend::get().get_kernel(KernelID::Sub);
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int a_off = a->offset;
                int b_off = b->offset;
                int c_off = out->offset;
                int size = out->numel();
                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
            }
        } else {
            auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "elementwise_broadcast_sub");
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int a_off = a->offset;
                int b_off = b->offset;
                int c_off = out->offset;

                TemporaryGPUBuffers temp_bufs;
                cl_mem out_shape_mem = temp_bufs.add(out_shape);
                cl_mem a_shape_mem = temp_bufs.add(a->shape);
                cl_mem a_strides_mem = temp_bufs.add(a->strides);
                cl_mem b_shape_mem = temp_bufs.add(b->shape);
                cl_mem b_strides_mem = temp_bufs.add(b->strides);

                int ndims = out_shape.size();
                int a_ndims = a->shape.size();
                int b_ndims = b->shape.size();
                int size = out->numel();

                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &ndims, &out_shape_mem,
                     &a_ndims, &a_shape_mem, &a_strides_mem,
                     &b_ndims, &b_shape_mem, &b_strides_mem, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int),
                     sizeof(int), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem), sizeof(int)});
            }
        }
    }
    if (!run_gpu) {
        float* a_ptr = a->data_ptr();
        float* b_ptr = b->data_ptr();
        float* out_ptr = out->data_ptr();
        if (a->shape == b->shape && a->is_contiguous() && b->is_contiguous()) {
            int size = out->numel();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                out_ptr[i] = a_ptr[i] - b_ptr[i];
            });
        } else {
            int size = out->numel();
            int ndims = out_shape.size();
            std::vector<int64_t> out_strides(ndims);
            int64_t current = 1;
            for (int i = ndims - 1; i >= 0; --i) {
                out_strides[i] = current;
                current *= out_shape[i];
            }
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                int temp = i;
                int a_idx = 0;
                int b_idx = 0;
                int a_diff = ndims - a->shape.size();
                int b_diff = ndims - b->shape.size();
                for (int d = 0; d < ndims; ++d) {
                    int coord = temp / out_strides[d];
                    temp %= out_strides[d];
                    if (d >= a_diff) {
                        int a_dim_idx = d - a_diff;
                        if (a->shape[a_dim_idx] != 1) {
                            a_idx += coord * a->strides[a_dim_idx];
                        }
                    }
                    if (d >= b_diff) {
                        int b_dim_idx = d - b_diff;
                        if (b->shape[b_dim_idx] != 1) {
                            b_idx += coord * b->strides[b_dim_idx];
                        }
                    }
                }
                out_ptr[i] = a_ptr[a_idx] - b_ptr[b_idx];
            });
        }
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        }
    }

    if (a->requires_grad || b->requires_grad) {
        auto node = std::make_shared<SubNode>(a->shape, b->shape);
        node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
        node->next_nodes = { a->creator, b->creator };
        node->saved_tensors = {};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
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
            auto kernel = CLBackend::get().get_kernel(KernelID::Mul);
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int a_off = a->offset;
                int b_off = b->offset;
                int c_off = out->offset;
                int size = out->numel();
                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
            }
        } else {
            auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "elementwise_broadcast_mul");
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int a_off = a->offset;
                int b_off = b->offset;
                int c_off = out->offset;

                TemporaryGPUBuffers temp_bufs;
                cl_mem out_shape_mem = temp_bufs.add(out_shape);
                cl_mem a_shape_mem = temp_bufs.add(a->shape);
                cl_mem a_strides_mem = temp_bufs.add(a->strides);
                cl_mem b_shape_mem = temp_bufs.add(b->shape);
                cl_mem b_strides_mem = temp_bufs.add(b->strides);

                int ndims = out_shape.size();
                int a_ndims = a->shape.size();
                int b_ndims = b->shape.size();
                int size = out->numel();

                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &ndims, &out_shape_mem,
                     &a_ndims, &a_shape_mem, &a_strides_mem,
                     &b_ndims, &b_shape_mem, &b_strides_mem, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int),
                     sizeof(int), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem), sizeof(int)});
            }
        }
    }
    if (!run_gpu) {
        float* a_ptr = a->data_ptr();
        float* b_ptr = b->data_ptr();
        float* out_ptr = out->data_ptr();
        if (a->shape == b->shape && a->is_contiguous() && b->is_contiguous()) {
            int size = out->numel();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                out_ptr[i] = a_ptr[i] * b_ptr[i];
            });
        } else {
            int size = out->numel();
            int ndims = out_shape.size();
            std::vector<int64_t> out_strides(ndims);
            int64_t current = 1;
            for (int i = ndims - 1; i >= 0; --i) {
                out_strides[i] = current;
                current *= out_shape[i];
            }
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                int temp = i;
                int a_idx = 0;
                int b_idx = 0;
                int a_diff = ndims - a->shape.size();
                int b_diff = ndims - b->shape.size();
                for (int d = 0; d < ndims; ++d) {
                    int coord = temp / out_strides[d];
                    temp %= out_strides[d];
                    if (d >= a_diff) {
                        int a_dim_idx = d - a_diff;
                        if (a->shape[a_dim_idx] != 1) {
                            a_idx += coord * a->strides[a_dim_idx];
                        }
                    }
                    if (d >= b_diff) {
                        int b_dim_idx = d - b_diff;
                        if (b->shape[b_dim_idx] != 1) {
                            b_idx += coord * b->strides[b_dim_idx];
                        }
                    }
                }
                out_ptr[i] = a_ptr[a_idx] * b_ptr[b_idx];
            });
        }
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        }
    }

    if (a->requires_grad || b->requires_grad) {
        auto node = std::make_shared<MulNode>();
        node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
        node->next_nodes = { a->creator, b->creator };
        node->saved_tensors = { a, b };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
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
            auto kernel = CLBackend::get().get_kernel(KernelID::Div);
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int a_off = a->offset;
                int b_off = b->offset;
                int c_off = out->offset;
                int size = out->numel();
                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
            }
        } else {
            auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "elementwise_broadcast_div");
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                cl_mem b_mem = b->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int a_off = a->offset;
                int b_off = b->offset;
                int c_off = out->offset;

                TemporaryGPUBuffers temp_bufs;
                cl_mem out_shape_mem = temp_bufs.add(out_shape);
                cl_mem a_shape_mem = temp_bufs.add(a->shape);
                cl_mem a_strides_mem = temp_bufs.add(a->strides);
                cl_mem b_shape_mem = temp_bufs.add(b->shape);
                cl_mem b_strides_mem = temp_bufs.add(b->strides);

                int ndims = out_shape.size();
                int a_ndims = a->shape.size();
                int b_ndims = b->shape.size();
                int size = out->numel();

                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &ndims, &out_shape_mem,
                     &a_ndims, &a_shape_mem, &a_strides_mem,
                     &b_ndims, &b_shape_mem, &b_strides_mem, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int),
                     sizeof(int), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem),
                     sizeof(int), sizeof(cl_mem), sizeof(cl_mem), sizeof(int)});
            }
        }
    }
    if (!run_gpu) {
        float* a_ptr = a->data_ptr();
        float* b_ptr = b->data_ptr();
        float* out_ptr = out->data_ptr();
        if (a->shape == b->shape && a->is_contiguous() && b->is_contiguous()) {
            int size = out->numel();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                out_ptr[i] = a_ptr[i] / b_ptr[i];
            });
        } else {
            int size = out->numel();
            int ndims = out_shape.size();
            std::vector<int64_t> out_strides(ndims);
            int64_t current = 1;
            for (int i = ndims - 1; i >= 0; --i) {
                out_strides[i] = current;
                current *= out_shape[i];
            }
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                int temp = i;
                int a_idx = 0;
                int b_idx = 0;
                int a_diff = ndims - a->shape.size();
                int b_diff = ndims - b->shape.size();
                for (int d = 0; d < ndims; ++d) {
                    int coord = temp / out_strides[d];
                    temp %= out_strides[d];
                    if (d >= a_diff) {
                        int a_dim_idx = d - a_diff;
                        if (a->shape[a_dim_idx] != 1) {
                            a_idx += coord * a->strides[a_dim_idx];
                        }
                    }
                    if (d >= b_diff) {
                        int b_dim_idx = d - b_diff;
                        if (b->shape[b_dim_idx] != 1) {
                            b_idx += coord * b->strides[b_dim_idx];
                        }
                    }
                }
                out_ptr[i] = a_ptr[a_idx] / b_ptr[b_idx];
            });
        }
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        }
    }

    if (a->requires_grad || b->requires_grad) {
        auto node = std::make_shared<DivNode>();
        node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
        node->next_nodes = { a->creator, b->creator };
        node->saved_tensors = { a, b };
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
        auto input = saved_tensors[0];
        auto exponent_t = Tensor::from_vector({exponent}, {1}, input->device);
        auto exponent_minus_one_t = Tensor::from_vector({exponent - 1.0f}, {1}, input->device);
        auto grad_input = mul(grad_output, mul(exponent_t, pow(input, exponent_minus_one_t->item())));
        return { grad_input };
    }
};

std::shared_ptr<Tensor> pow(std::shared_ptr<Tensor> a, float exponent) {
    auto out = Tensor::create(a->shape, a->device);
    StorageUseGuard guard({a->storage, out->storage});
    int size = a->numel();
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto kernel = CLBackend::get().get_kernel(KernelID::Pow);
        if (kernel) {
            run_gpu = true;
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int a_off = a_c->offset;
            int b_off = out->offset;
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                {&a_mem, &a_off, &exponent, &b_mem, &b_off, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(cl_mem), sizeof(int), sizeof(int)});
        }
    }
    if (!run_gpu) {
        float* a_ptr = a->data_ptr();
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            out_ptr[i] = std::pow(a_ptr[i], exponent);
        });
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), size * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), out_ptr);
        }
    }

    if (a->requires_grad) {
        auto node = std::make_shared<PowNode>(exponent);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
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
        auto out = output.lock();
        if (!out) return { nullptr };
        auto two_t = Tensor::from_vector({2.0f}, {1}, out->device);
        auto grad_input = div(grad_output, mul(two_t, out));
        return { grad_input };
    }
};

std::shared_ptr<Tensor> sqrt(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(a->shape, a->device);
    StorageUseGuard guard({a->storage, out->storage});
    int size = a->numel();
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto kernel = CLBackend::get().get_kernel(KernelID::Sqrt);
        if (kernel) {
            run_gpu = true;
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int a_off = a_c->offset;
            int b_off = out->offset;
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                {&a_mem, &a_off, &b_mem, &b_off, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
        }
    }
    if (!run_gpu) {
        float* a_ptr = a->data_ptr();
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            out_ptr[i] = std::sqrt(a_ptr[i]);
        });
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), size * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), out_ptr);
        }
    }

    if (a->requires_grad) {
        auto node = std::make_shared<SqrtNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = {};
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
        auto out = output.lock();
        if (!out) return { nullptr };
        auto grad_input = mul(grad_output, out);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> exp(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(a->shape, a->device);
    StorageUseGuard guard({a->storage, out->storage});
    int size = a->numel();
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto kernel = CLBackend::get().get_kernel(KernelID::Exp);
        if (kernel) {
            run_gpu = true;
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int a_off = a_c->offset;
            int b_off = out->offset;
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                {&a_mem, &a_off, &b_mem, &b_off, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
        }
    }
    if (!run_gpu) {
        float* a_ptr = a->data_ptr();
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            out_ptr[i] = std::exp(a_ptr[i]);
        });
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), size * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), out_ptr);
        }
    }

    if (a->requires_grad) {
        auto node = std::make_shared<ExpNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = {};
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
        auto input = saved_tensors[0];
        auto grad_input = div(grad_output, input);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> log(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(a->shape, a->device);
    StorageUseGuard guard({a->storage, out->storage});
    int size = a->numel();
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto kernel = CLBackend::get().get_kernel(KernelID::Log);
        if (kernel) {
            run_gpu = true;
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int a_off = a_c->offset;
            int b_off = out->offset;
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                {&a_mem, &a_off, &b_mem, &b_off, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
        }
    }
    if (!run_gpu) {
        float* a_ptr = a->data_ptr();
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            out_ptr[i] = std::log(a_ptr[i]);
        });
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), size * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), out_ptr);
        }
    }

    if (a->requires_grad) {
        auto node = std::make_shared<LogNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
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
        auto input = saved_tensors[0];
        auto signs = Tensor::create(input->shape, input->device);
        float* in_ptr = input->data_ptr();
        float* s_ptr = signs->data_ptr();
        for (size_t i = 0; i < input->numel(); ++i) {
            s_ptr[i] = (in_ptr[i] > 0.0f) ? 1.0f : ((in_ptr[i] < 0.0f) ? -1.0f : 0.0f);
        }
        if (input->device.type == DeviceType::GPU) {
            signs->to(input->device);
        }
        auto grad_input = mul(grad_output, signs);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> abs(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(a->shape, a->device);
    StorageUseGuard guard({a->storage, out->storage});
    int size = a->numel();
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto kernel = CLBackend::get().get_kernel(KernelID::Abs);
        if (kernel) {
            run_gpu = true;
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int a_off = a_c->offset;
            int b_off = out->offset;
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                {&a_mem, &a_off, &b_mem, &b_off, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
        }
    }
    if (!run_gpu) {
        float* a_ptr = a->data_ptr();
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            out_ptr[i] = std::fabs(a_ptr[i]);
        });
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), size * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), out_ptr);
        }
    }

    if (a->requires_grad) {
        auto node = std::make_shared<AbsNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
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
        auto grad_input = neg(grad_output);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> neg(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(a->shape, a->device);
    StorageUseGuard guard({a->storage, out->storage});
    int size = a->numel();
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto kernel = CLBackend::get().get_kernel(KernelID::Neg);
        if (kernel) {
            run_gpu = true;
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int a_off = a_c->offset;
            int b_off = out->offset;
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                {&a_mem, &a_off, &b_mem, &b_off, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
        }
    }
    if (!run_gpu) {
        float* a_ptr = a->data_ptr();
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            out_ptr[i] = -a_ptr[i];
        });
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), size * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), out_ptr);
        }
    }

    if (a->requires_grad) {
        auto node = std::make_shared<NegNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = {};
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
        auto grad_input = Tensor::create(input_shape, grad_output->device);
        StorageUseGuard guard({grad_input->storage, grad_output->storage});
        bool run_gpu = false;
        if (grad_output->device.type == DeviceType::GPU) {
            auto kernel = CLBackend::get().get_kernel(KernelID::SumBackward);
            if (kernel) {
                run_gpu = true;
                cl_mem gout_mem = grad_output->gpu_data();
                int gout_off = grad_output->offset;
                cl_mem gin_mem = grad_input->gpu_data();
                int gin_off = grad_input->offset;
                int size = grad_input->numel();
                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&gin_mem, &gin_off, &gout_mem, &gout_off, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
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
                CLBackend::get().write(grad_input->gpu_data(), size * sizeof(float), ptr);
            } else if (grad_input->device.type == DeviceType::TPU) {
                auto tpu = BackendDispatcher::get().get_tpu_backend();
                if (tpu) tpu->write(grad_input->gpu_data(), size * sizeof(float), ptr);
            }
        }
        return { grad_input };
    }
};

std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create({1}, a->device);
    StorageUseGuard guard({a->storage, out->storage});
    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto backend = BackendDispatcher::get().get_backend();
        if (backend && backend->is_available()) {
            run_gpu = true;
            backend->sum(a->gpu_data(), a->offset, out->gpu_data(), out->offset, a->numel());
        } else {
            auto kernel = CLBackend::get().get_kernel(KernelID::SumForward);
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a->gpu_data();
                int a_off = a->offset;
                cl_mem out_mem = out->gpu_data();
                int out_off = out->offset;
                int size = a->numel();
                float zero = 0.0f;
                CLBackend::get().write(out_mem, sizeof(float), &zero, out_off);
                size_t blocks = std::min(static_cast<size_t>(32), (static_cast<size_t>(size) + 255) / 256);
                if (blocks < 1) blocks = 1;
                CLBackend::get().launch(kernel, {blocks * 256}, {256},
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
        float total = 0.0f;
        float* ptr = a->data_ptr();
        size_t size = a->numel();
        for (size_t i = 0; i < size; ++i) {
            total += ptr[i];
        }
        out->data_ptr()[0] = total;
    }
    if (a->requires_grad) {
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
        auto grad_input = Tensor::create(input_shape, grad_output->device);
        StorageUseGuard guard({grad_input->storage, grad_output->storage});
        float num_el = static_cast<float>(grad_input->numel());
        bool run_gpu = false;
        if (grad_output->device.type == DeviceType::GPU) {
            auto kernel = CLBackend::get().get_kernel(KernelID::SumBackward);
            if (kernel) {
                run_gpu = true;
                cl_mem in_mem = grad_output->gpu_data();
                int in_off = grad_output->offset;
                cl_mem out_mem = grad_input->gpu_data();
                int out_off = grad_input->offset;
                int size = grad_input->numel();
                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&in_mem, &in_off, &out_mem, &out_off, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
                
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
                CLBackend::get().write(grad_input->gpu_data(), size * sizeof(float), ptr);
            }
        }
        return { grad_input };
    }
};

std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a) {
    float num_el = static_cast<float>(a->numel());
    auto out = sum(a);
    auto scale = Tensor::from_vector({1.0f / num_el}, {1}, a->device);
    out = mul(out, scale);
    if (a->requires_grad) {
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
        auto input = saved_tensors[0];
        auto grad_input = Tensor::zeros(input->shape, grad_output->device);
        StorageUseGuard guard({input->storage, grad_input->storage, grad_output->storage});
        
        auto cpu_input = input->to(Device(DeviceType::CPU, 0));
        float* in_ptr = cpu_input->data_ptr();
        float max_val = in_ptr[0];
        size_t max_idx = 0;
        size_t size = input->numel();
        for (size_t i = 1; i < size; ++i) {
            if (in_ptr[i] > max_val) {
                max_val = in_ptr[i];
                max_idx = i;
            }
        }
        
        if (grad_output->device.type == DeviceType::GPU) {
            auto cpu_gout = grad_output->to(Device(DeviceType::CPU, 0));
            float val = cpu_gout->data_ptr()[0];
            std::vector<float> cpu_gin(size, 0.0f);
            cpu_gin[max_idx] = val;
            CLBackend::get().write(grad_input->gpu_data(), size * sizeof(float), cpu_gin.data());
        } else {
            float val = grad_output->data_ptr()[0];
            grad_input->data_ptr()[max_idx] = val;
        }
        return { grad_input };
    }
};

std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a) {
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
        float* ptr = a->data_ptr();
        float max_val = ptr[0];
        size_t size = a->numel();
        for (size_t i = 1; i < size; ++i) {
            if (ptr[i] > max_val) max_val = ptr[i];
        }
        out->data_ptr()[0] = max_val;
    }

    if (a->requires_grad) {
        auto node = std::make_shared<MaxNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

class CatNode : public Node {
public:
    int64_t dim;
    std::vector<std::vector<int64_t>> input_shapes;
    CatNode(int64_t dim, const std::vector<std::vector<int64_t>>& shapes)
        : Node("Cat"), dim(dim), input_shapes(shapes) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::vector<std::shared_ptr<Tensor>> grads;
        int64_t offset = 0;
        int64_t ndim = grad_output->shape.size();
        int64_t concat_dim_size = grad_output->shape[dim];
        bool run_gpu = false;
        if (grad_output->device.type == DeviceType::GPU) {
            auto backend = BackendDispatcher::get().get_backend();
            if (backend && backend->is_available()) {
                run_gpu = true;
                for (size_t i = 0; i < input_shapes.size(); ++i) {
                    auto& shape = input_shapes[i];
                    auto grad_input = Tensor::create(shape, grad_output->device);
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
            float* src_ptr = cpu_gout->data_ptr();
            
            for (size_t i = 0; i < input_shapes.size(); ++i) {
                auto& shape = input_shapes[i];
                auto grad_input = Tensor::create(shape, grad_output->device);
                auto cpu_gin = grad_input->to(Device(DeviceType::CPU, 0));
                float* dst_ptr = cpu_gin->data_ptr();
                
                int64_t outer_size = 1;
                for (int64_t d = 0; d < dim; ++d) outer_size *= shape[d];
                int64_t inner_size = 1;
                for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= shape[d];
                int64_t dim_size = shape[dim];
                
                ThreadPool::get().parallel_for(0, outer_size, [&](int64_t o) {
                    for (int64_t d = 0; d < dim_size; ++d) {
                        for (int64_t in = 0; in < inner_size; ++in) {
                            int64_t src_idx = (o * concat_dim_size + (offset + d)) * inner_size + in;
                            int64_t dst_idx = (o * dim_size + d) * inner_size + in;
                            dst_ptr[dst_idx] = src_ptr[src_idx];
                        }
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
    if (tensors.empty()) {
        throw std::runtime_error("[litetorch Error] cat requires at least one tensor");
    }

    int64_t ndim = tensors[0]->shape.size();
    if (dim < 0) dim += ndim;

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

    auto out = Tensor::create(out_shape, tensors[0]->device);
    storages.push_back(out->storage);
    StorageUseGuard guard(storages);

    bool run_gpu = false;
    int64_t offset = 0;
    if (out->device.type == DeviceType::GPU) {
        auto backend = BackendDispatcher::get().get_backend();
        if (backend && backend->is_available()) {
            run_gpu = true;
            for (auto& t : tensors) {
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
        float* dst_ptr = cpu_out->data_ptr();
        offset = 0;

        for (auto& t : tensors) {
            auto cpu_t = t->to(Device(DeviceType::CPU, 0));
            float* src_ptr = cpu_t->data_ptr();
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
                        dst_ptr[dst_idx] = src_ptr[src_idx];
                    }
                }
            });
            offset += dim_size;
        }
        out->copy_(cpu_out);
    }

    if (any_requires_grad) {
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

class SqueezeNode : public Node {
public:
    std::vector<int64_t> input_shape;
    SqueezeNode(const std::vector<int64_t>& input_shape) : Node("Squeeze"), input_shape(input_shape) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto grad_input = std::make_shared<Tensor>(grad_output->storage, input_shape, default_strides(input_shape), grad_output->offset, grad_output->device, false);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> squeeze(std::shared_ptr<Tensor> a, int64_t dim) {
    std::vector<int64_t> new_shape;
    int64_t ndim = a->shape.size();
    if (dim < 0) {
        for (int64_t i = 0; i < ndim; ++i) {
            if (a->shape[i] != 1) {
                new_shape.push_back(a->shape[i]);
            }
        }
    } else {
        if (dim >= ndim) throw std::runtime_error("[litetorch Error] Dimension out of range for squeeze");
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != dim || a->shape[i] != 1) {
                new_shape.push_back(a->shape[i]);
            }
        }
    }

    auto out = std::make_shared<Tensor>(a->storage, new_shape, default_strides(new_shape), a->offset, a->device, a->requires_grad);
    if (a->requires_grad) {
        auto node = std::make_shared<SqueezeNode>(a->shape);
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
    UnsqueezeNode(const std::vector<int64_t>& input_shape) : Node("Unsqueeze"), input_shape(input_shape) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto grad_input = std::make_shared<Tensor>(grad_output->storage, input_shape, default_strides(input_shape), grad_output->offset, grad_output->device, false);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> unsqueeze(std::shared_ptr<Tensor> a, int64_t dim) {
    int64_t ndim = a->shape.size();
    if (dim < 0) dim += ndim + 1;
    if (dim < 0 || dim > ndim) throw std::runtime_error("[litetorch Error] Dimension out of range for unsqueeze");

    std::vector<int64_t> new_shape;
    for (int64_t i = 0; i <= ndim; ++i) {
        if (i == dim) {
            new_shape.push_back(1);
        }
        if (i < ndim) {
            new_shape.push_back(a->shape[i]);
        }
    }

    auto out = std::make_shared<Tensor>(a->storage, new_shape, default_strides(new_shape), a->offset, a->device, a->requires_grad);
    if (a->requires_grad) {
        auto node = std::make_shared<UnsqueezeNode>(a->shape);
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
        auto input = saved_tensors[0];
        auto mask = Tensor::create(input->shape, input->device);
        StorageUseGuard guard({input->storage, mask->storage});
        float* in_ptr = input->data_ptr();
        float* m_ptr = mask->data_ptr();
        for (size_t i = 0; i < input->numel(); ++i) {
            m_ptr[i] = (in_ptr[i] >= min_val && in_ptr[i] <= max_val) ? 1.0f : 0.0f;
        }
        if (input->device.type == DeviceType::GPU) {
            mask->to(input->device);
        }
        auto grad_input = mul(grad_output, mask);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> clamp(std::shared_ptr<Tensor> a, float min_val, float max_val) {
    auto out = Tensor::create(a->shape, a->device);
    StorageUseGuard guard({a->storage, out->storage});
    float* src = a->data_ptr();
    float* dst = out->data_ptr();
    size_t size = out->numel();
    ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
        float x = src[i];
        dst[i] = x < min_val ? min_val : (x > max_val ? max_val : x);
    });
    if (a->requires_grad) {
        auto node = std::make_shared<ClampNode>(min_val, max_val);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
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
        auto input = saved_tensors[0];
        auto grad_input = mul(grad_output, cos(input));
        return { grad_input };
    }
};

std::shared_ptr<Tensor> sin(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(a->shape, a->device);
    StorageUseGuard guard({a->storage, out->storage});
    float* src = a->data_ptr();
    float* dst = out->data_ptr();
    size_t size = out->numel();
    ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
        dst[i] = std::sin(src[i]);
    });
    if (a->requires_grad) {
        auto node = std::make_shared<SinNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
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
        auto input = saved_tensors[0];
        auto neg_sin = mul(sin(input), Tensor::from_vector({-1.0f}, {1}, input->device));
        auto grad_input = mul(grad_output, neg_sin);
        return { grad_input };
    }
};

std::shared_ptr<Tensor> cos(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(a->shape, a->device);
    StorageUseGuard guard({a->storage, out->storage});
    float* src = a->data_ptr();
    float* dst = out->data_ptr();
    size_t size = out->numel();
    ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
        dst[i] = std::cos(src[i]);
    });
    if (a->requires_grad) {
        auto node = std::make_shared<CosNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

}
}
