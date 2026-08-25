#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
#include <algorithm>
#include <cmath>
#include <limits>
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
}

class ReluNode : public Node {
public:
    ReluNode() : Node("ReLU") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto input = saved_tensors[0];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        StorageUseGuard guard({input_c->storage, gout_c->storage, grad_input->storage});
        int size = input_c->numel();
        bool run_gpu = false;
        if (input_c->device.type == DeviceType::GPU) {
            auto kernel = CLBackend::get().get_kernel(KernelID::ReluBackward);
            if (kernel) {
                run_gpu = true;
                cl_mem in_mem = input_c->gpu_data();
                int in_off = input_c->offset;
                cl_mem gout_mem = gout_c->gpu_data();
                int gout_off = gout_c->offset;
                cl_mem gin_mem = grad_input->gpu_data();
                int gin_off = grad_input->offset;
                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&in_mem, &in_off, &gout_mem, &gout_off, &gin_mem, &gin_off, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
            }
        }
        if (!run_gpu) {
            float* in_ptr = input_c->data_ptr();
            float* g_in = grad_input->data_ptr();
            float* g_out = gout_c->data_ptr();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                g_in[i] = in_ptr[i] > 0.0f ? g_out[i] : 0.0f;
            });
            if (grad_input->device.type == DeviceType::GPU) {
                CLBackend::get().write(grad_input->gpu_data(), size * sizeof(float), g_in);
            }
        }
        return { grad_input };
    }
};

class SigmoidNode : public Node {
public:
    SigmoidNode() : Node("Sigmoid") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto out = output.lock();
        if (!out) return { nullptr };
        auto out_c = out->is_contiguous() ? out : out->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto grad_input = Tensor::create(out_c->shape, out_c->device);
        StorageUseGuard guard({out_c->storage, gout_c->storage, grad_input->storage});
        int size = out_c->numel();
        bool run_gpu = false;
        if (out_c->device.type == DeviceType::GPU) {
            auto kernel = CLBackend::get().get_kernel(KernelID::SigmoidBackward);
            if (kernel) {
                run_gpu = true;
                cl_mem out_mem = out_c->gpu_data();
                int out_off = out_c->offset;
                cl_mem gout_mem = gout_c->gpu_data();
                int gout_off = gout_c->offset;
                cl_mem gin_mem = grad_input->gpu_data();
                int gin_off = grad_input->offset;
                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&out_mem, &out_off, &gout_mem, &gout_off, &gin_mem, &gin_off, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
            }
        }
        if (!run_gpu) {
            float* out_ptr = out_c->data_ptr();
            float* g_in = grad_input->data_ptr();
            float* g_out = gout_c->data_ptr();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                g_in[i] = g_out[i] * out_ptr[i] * (1.0f - out_ptr[i]);
            });
            if (grad_input->device.type == DeviceType::GPU) {
                CLBackend::get().write(grad_input->gpu_data(), size * sizeof(float), g_in);
            }
        }
        return { grad_input };
    }
};

class TanhNode : public Node {
public:
    TanhNode() : Node("Tanh") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto out = output.lock();
        if (!out) return { nullptr };
        auto out_c = out->is_contiguous() ? out : out->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto grad_input = Tensor::create(out_c->shape, out_c->device);
        StorageUseGuard guard({out_c->storage, gout_c->storage, grad_input->storage});
        int size = out_c->numel();
        bool run_gpu = false;
        if (out_c->device.type == DeviceType::GPU) {
            auto kernel = CLBackend::get().get_kernel(KernelID::TanhBackward);
            if (kernel) {
                run_gpu = true;
                cl_mem out_mem = out_c->gpu_data();
                int out_off = out_c->offset;
                cl_mem gout_mem = gout_c->gpu_data();
                int gout_off = gout_c->offset;
                cl_mem gin_mem = grad_input->gpu_data();
                int gin_off = grad_input->offset;
                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&out_mem, &out_off, &gout_mem, &gout_off, &gin_mem, &gin_off, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
            }
        }
        if (!run_gpu) {
            float* out_ptr = out_c->data_ptr();
            float* g_in = grad_input->data_ptr();
            float* g_out = gout_c->data_ptr();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                g_in[i] = g_out[i] * (1.0f - out_ptr[i] * out_ptr[i]);
            });
            if (grad_input->device.type == DeviceType::GPU) {
                CLBackend::get().write(grad_input->gpu_data(), size * sizeof(float), g_in);
            }
        }
        return { grad_input };
    }
};

class LeakyReluNode : public Node {
public:
    LeakyReluNode(float negative_slope) : Node("LeakyReLU"), negative_slope(negative_slope) {}
    float negative_slope;
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto input = saved_tensors[0];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        StorageUseGuard guard({input_c->storage, gout_c->storage, grad_input->storage});
        int size = input_c->numel();
        bool run_gpu = false;
        if (input_c->device.type == DeviceType::GPU) {
            auto kernel = CLBackend::get().get_kernel(KernelID::LeakyReluBackward);
            if (kernel) {
                run_gpu = true;
                cl_mem in_mem = input_c->gpu_data();
                int in_off = input_c->offset;
                cl_mem gout_mem = grad_output->is_contiguous() ? grad_output->gpu_data() : gout_c->gpu_data();
                int gout_off = grad_output->is_contiguous() ? grad_output->offset : gout_c->offset;
                cl_mem gin_mem = grad_input->gpu_data();
                int gin_off = grad_input->offset;
                float ns = negative_slope;
                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&in_mem, &in_off, &gout_mem, &gout_off, &gin_mem, &gin_off, &size, &ns},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(float)});
            }
        }
        if (!run_gpu) {
            float* in_ptr = input_c->data_ptr();
            float* g_in = grad_input->data_ptr();
            float* g_out = gout_c->data_ptr();
            float ns = negative_slope;
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                g_in[i] = in_ptr[i] > 0.0f ? g_out[i] : g_out[i] * ns;
            });
            if (grad_input->device.type == DeviceType::GPU) {
                CLBackend::get().write(grad_input->gpu_data(), size * sizeof(float), g_in);
            }
        }
        return { grad_input };
    }
};

class SoftmaxNode : public Node {
public:
    int64_t dim;
    SoftmaxNode(int64_t dim) : Node("Softmax"), dim(dim) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto out = output.lock();
        if (!out) return { nullptr };
        auto out_c = out->is_contiguous() ? out : out->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto grad_input = Tensor::create(out_c->shape, out_c->device);
        StorageUseGuard guard({out_c->storage, gout_c->storage, grad_input->storage});
        
        int64_t d_dim = dim;
        if (d_dim < 0) d_dim += out_c->shape.size();
        int64_t dim_size = out_c->shape[d_dim];
        int64_t inner_size = 1;
        for (size_t i = d_dim + 1; i < out_c->shape.size(); ++i) inner_size *= out_c->shape[i];
        int64_t outer_size = out_c->numel() / (dim_size * inner_size);
        
        bool run_gpu = false;
        if (out_c->device.type == DeviceType::GPU) {
            auto kernel = CLBackend::get().get_kernel(KernelID::SoftmaxBackward);
            if (kernel) {
                run_gpu = true;
                cl_mem out_mem = out_c->gpu_data();
                int out_off = out_c->offset;
                cl_mem gout_mem = gout_c->gpu_data();
                int gout_off = gout_c->offset;
                cl_mem gin_mem = grad_input->gpu_data();
                int gin_off = grad_input->offset;
                int d_size = dim_size;
                int i_size = inner_size;
                int o_size = outer_size;
                int total = o_size * i_size;
                CLBackend::get().launch(kernel, {static_cast<size_t>(total)}, {},
                    {&out_mem, &out_off, &gout_mem, &gout_off, &gin_mem, &gin_off, &d_size, &i_size, &o_size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
            }
        }
        if (!run_gpu) {
            float* out_ptr = out_c->data_ptr();
            float* grad_out_ptr = gout_c->data_ptr();
            float* grad_in_ptr = grad_input->data_ptr();
            
            ThreadPool::get().parallel_for(0, outer_size, [&](int64_t o) {
                for (int64_t i = 0; i < inner_size; ++i) {
                    float sum_grad_out = 0.0f;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                        sum_grad_out += grad_out_ptr[idx] * out_ptr[idx];
                    }
                    for (int64_t d = 0; d < dim_size; ++d) {
                        int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                        grad_in_ptr[idx] = out_ptr[idx] * (grad_out_ptr[idx] - sum_grad_out);
                    }
                }
            });
            if (grad_input->device.type == DeviceType::GPU) {
                CLBackend::get().write(grad_input->gpu_data(), grad_input->numel() * sizeof(float), grad_in_ptr);
            } else if (grad_input->device.type == DeviceType::TPU) {
                auto tpu = BackendDispatcher::get().get_tpu_backend();
                if (tpu) tpu->write(grad_input->gpu_data(), grad_input->numel() * sizeof(float), grad_in_ptr);
            }
        }
        
        return { grad_input };
    }
};

class GeluNode : public Node {
public:
    GeluNode() : Node("GELU") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto a = saved_tensors[0];
        auto save_tanh = saved_tensors[1];
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto st_c = save_tanh->is_contiguous() ? save_tanh : save_tanh->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto grad_input = Tensor::create(a_c->shape, a_c->device);
        StorageUseGuard guard({a_c->storage, st_c->storage, gout_c->storage, grad_input->storage});
        int size = a_c->numel();
        
        bool run_gpu = false;
        if (a_c->device.type == DeviceType::GPU) {
            auto kernel = CLBackend::get().get_kernel(KernelID::GeluBackward);
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a_c->gpu_data();
                int a_off = a_c->offset;
                cl_mem st_mem = st_c->gpu_data();
                int st_off = st_c->offset;
                cl_mem gout_mem = gout_c->gpu_data();
                int gout_off = gout_c->offset;
                cl_mem gin_mem = grad_input->gpu_data();
                int gin_off = grad_input->offset;
                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&a_mem, &a_off, &st_mem, &st_off, &gout_mem, &gout_off, &gin_mem, &gin_off, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
            }
        }
        if (!run_gpu) {
            float* a_ptr = a_c->data_ptr();
            float* st_ptr = st_c->data_ptr();
            float* grad_out_ptr = gout_c->data_ptr();
            float* grad_in_ptr = grad_input->data_ptr();
            
            float C = 0.79788456f;
            ThreadPool::get().parallel_for(0, size, [&](int64_t idx) {
                float x = a_ptr[idx];
                float tanh_u = st_ptr[idx];
                float d_gelu = 0.5f * (1.0f + tanh_u) + 0.5f * x * (1.0f - tanh_u * tanh_u) * C * (1.0f + 0.134145f * x * x);
                grad_in_ptr[idx] = grad_out_ptr[idx] * d_gelu;
            });
            if (grad_input->device.type == DeviceType::GPU) {
                CLBackend::get().write(grad_input->gpu_data(), size * sizeof(float), grad_in_ptr);
            } else if (grad_input->device.type == DeviceType::TPU) {
                auto tpu = BackendDispatcher::get().get_tpu_backend();
                if (tpu) tpu->write(grad_input->gpu_data(), size * sizeof(float), grad_in_ptr);
            }
        }
        
        return { grad_input };
    }
};

namespace Ops {

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a) {
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    StorageUseGuard guard({a_c->storage, out->storage});

    bool run_gpu = false;
    if (a_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::ReLU);
        if (kernel) {
            run_gpu = true;
            int size = out->numel();
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int a_off = a_c->offset;
            int b_off = out->offset;
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {}, {&a_mem, &a_off, &b_mem, &b_off, &size}, {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
        }
    }
    if (!run_gpu) {
        float* src = a_c->data_ptr();
        float* dst = out->data_ptr();
        size_t size = out->numel();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            dst[i] = std::max(0.0f, src[i]);
        });
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), size * sizeof(float), dst);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), dst);
        }
    }

    if (a->requires_grad) {
        auto node = std::make_shared<ReluNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> a) {
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    StorageUseGuard guard({a_c->storage, out->storage});

    bool run_gpu = false;
    if (a_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::Sigmoid);
        if (kernel) {
            run_gpu = true;
            int size = out->numel();
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int a_off = a_c->offset;
            int b_off = out->offset;
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {}, {&a_mem, &a_off, &b_mem, &b_off, &size}, {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
        }
    }
    if (!run_gpu) {
        float* src = a_c->data_ptr();
        float* dst = out->data_ptr();
        size_t size = out->numel();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
        });
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), size * sizeof(float), dst);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), dst);
        }
    }

    if (a->requires_grad) {
        auto node = std::make_shared<SigmoidNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = {};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> a) {
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    StorageUseGuard guard({a_c->storage, out->storage});

    bool run_gpu = false;
    if (a_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::Tanh);
        if (kernel) {
            run_gpu = true;
            int size = out->numel();
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int a_off = a_c->offset;
            int b_off = out->offset;
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {}, {&a_mem, &a_off, &b_mem, &b_off, &size}, {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
        }
    }
    if (!run_gpu) {
        float* src = a_c->data_ptr();
        float* dst = out->data_ptr();
        size_t size = out->numel();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            dst[i] = std::tanh(src[i]);
        });
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), size * sizeof(float), dst);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), dst);
        }
    }

    if (a->requires_grad) {
        auto node = std::make_shared<TanhNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = {};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> leaky_relu(std::shared_ptr<Tensor> a, float negative_slope) {
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    StorageUseGuard guard({a_c->storage, out->storage});

    bool run_gpu = false;
    if (a_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::LeakyReluForward);
        if (kernel) {
            run_gpu = true;
            int size = out->numel();
            cl_mem a_mem = a_c->gpu_data();
            cl_mem b_mem = out->gpu_data();
            int a_off = a_c->offset;
            int b_off = out->offset;
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {}, {&a_mem, &a_off, &negative_slope, &b_mem, &b_off, &size}, {sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(cl_mem), sizeof(int), sizeof(int)});
        }
    }
    if (!run_gpu) {
        float* src = a_c->data_ptr();
        float* dst = out->data_ptr();
        size_t size = out->numel();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            dst[i] = src[i] > 0.0f ? src[i] : src[i] * negative_slope;
        });
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), size * sizeof(float), dst);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), size * sizeof(float), dst);
        }
    }

    if (a->requires_grad) {
        auto node = std::make_shared<LeakyReluNode>(negative_slope);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> softmax(std::shared_ptr<Tensor> a, int64_t dim) {
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    StorageUseGuard guard({a_c->storage, out->storage});
    
    int64_t d_dim = dim;
    if (d_dim < 0) d_dim += a_c->shape.size();
    int64_t dim_size = a_c->shape[d_dim];
    int64_t inner_size = 1;
    for (size_t i = d_dim + 1; i < a_c->shape.size(); ++i) inner_size *= a_c->shape[i];
    int64_t outer_size = a_c->numel() / (dim_size * inner_size);
    
    bool run_gpu = false;
    if (a_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::SoftmaxForward);
        if (kernel) {
            run_gpu = true;
            cl_mem a_mem = a_c->gpu_data();
            int a_off = a_c->offset;
            cl_mem b_mem = out->gpu_data();
            int b_off = out->offset;
            int d_size = dim_size;
            int i_size = inner_size;
            int o_size = outer_size;
            int total = o_size * i_size;
            CLBackend::get().launch(kernel, {static_cast<size_t>(total)}, {},
                {&a_mem, &a_off, &b_mem, &b_off, &d_size, &i_size, &o_size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
        }
    }
    if (!run_gpu) {
        float* a_ptr = a_c->data_ptr();
        float* out_ptr = out->data_ptr();
        
        ThreadPool::get().parallel_for(0, outer_size, [&](int64_t o) {
            for (int64_t i = 0; i < inner_size; ++i) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t d = 0; d < dim_size; ++d) {
                    float val = a_ptr[o * dim_size * inner_size + d * inner_size + i];
                    if (val > max_val) max_val = val;
                }
                float sum = 0.0f;
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                    out_ptr[idx] = std::exp(a_ptr[idx] - max_val);
                    sum += out_ptr[idx];
                }
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                    out_ptr[idx] /= sum;
                }
            }
        });
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        }
    }
    
    if (a->requires_grad) {
        auto node = std::make_shared<SoftmaxNode>(dim);
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = {};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> gelu(std::shared_ptr<Tensor> a) {
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto out = Tensor::create(a_c->shape, a_c->device);
    auto save_tanh = Tensor::create(a_c->shape, a_c->device);
    StorageUseGuard guard({a_c->storage, out->storage, save_tanh->storage});
    int size = a_c->numel();
    
    bool run_gpu = false;
    if (a_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::GELU);
        if (kernel) {
            run_gpu = true;
            cl_mem a_mem = a_c->gpu_data();
            int a_off = a_c->offset;
            cl_mem b_mem = out->gpu_data();
            int b_off = out->offset;
            cl_mem st_mem = save_tanh->gpu_data();
            int st_off = save_tanh->offset;
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                {&a_mem, &a_off, &b_mem, &b_off, &st_mem, &st_off, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
        }
    }
    if (!run_gpu) {
        float* a_ptr = a_c->data_ptr();
        float* out_ptr = out->data_ptr();
        float* st_ptr = save_tanh->data_ptr();
        
        float C = 0.79788456f;
        ThreadPool::get().parallel_for(0, size, [&](int64_t idx) {
            float x = a_ptr[idx];
            float u = C * (x + 0.044715f * x * x * x);
            float tanh_u = std::tanh(u);
            st_ptr[idx] = tanh_u;
            out_ptr[idx] = 0.5f * x * (1.0f + tanh_u);
        });
        if (out->device.type == DeviceType::GPU) {
            CLBackend::get().write(out->gpu_data(), size * sizeof(float), out_ptr);
        } else if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) {
                tpu->write(out->gpu_data(), size * sizeof(float), out_ptr);
                tpu->write(save_tanh->gpu_data(), size * sizeof(float), st_ptr);
            }
        }
    }
    
    if (a->requires_grad) {
        auto node = std::make_shared<GeluNode>();
        node->inputs = { {a, true} };
        node->next_nodes = { a->creator };
        node->saved_tensors = { a, save_tanh };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

}
}
