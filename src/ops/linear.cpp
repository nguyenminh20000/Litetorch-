#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
#include "litetorch/amp.h"
#include <algorithm>
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

class MatMulNode : public Node {
public:
    MatMulNode() : Node("MatMul") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto a = saved_tensors[0];
        auto b = saved_tensors[1];
        std::shared_ptr<Tensor> grad_a = nullptr;
        std::shared_ptr<Tensor> grad_b = nullptr;
        if (inputs.size() > 0 && inputs[0].requires_grad) {
            auto b_t = b->transpose(0, 1);
            grad_a = Ops::matmul(grad_output, b_t);
        }
        if (inputs.size() > 1 && inputs[1].requires_grad) {
            auto a_t = a->transpose(0, 1);
            grad_b = Ops::matmul(a_t, grad_output);
        }
        return { grad_a, grad_b };
    }
};

class BmmNode : public Node {
public:
    BmmNode() : Node("Bmm") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto a = saved_tensors[0];
        auto b = saved_tensors[1];
        std::shared_ptr<Tensor> grad_a = nullptr;
        std::shared_ptr<Tensor> grad_b = nullptr;
        if (inputs.size() > 0 && inputs[0].requires_grad) {
            auto b_t = b->transpose(1, 2);
            grad_a = Ops::bmm(grad_output, b_t);
        }
        if (inputs.size() > 1 && inputs[1].requires_grad) {
            auto a_t = a->transpose(1, 2);
            grad_b = Ops::bmm(a_t, grad_output);
        }
        return { grad_a, grad_b };
    }
};

namespace Ops {

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (amp::AutocastGuard::is_enabled()) {
        auto cast_dtype = amp::AutocastGuard::get_dtype();
        if (a->dtype != cast_dtype) {
            a = Ops::cast(a, cast_dtype);
        }
        if (b->dtype != cast_dtype) {
            b = Ops::cast(b, cast_dtype);
        }
    }

    if (a->device.type == DeviceType::CPU) {
        if (a->dtype != DataType::FP32 || b->dtype != DataType::FP32) {
            amp::AutocastGuard disable_guard(false);
            auto a_fp32 = a->dtype == DataType::FP32 ? a : a->cast(DataType::FP32);
            auto b_fp32 = b->dtype == DataType::FP32 ? b : b->cast(DataType::FP32);
            auto out_fp32 = matmul(a_fp32, b_fp32);
            auto out = out_fp32->cast(a->dtype);
            if (a->requires_grad || b->requires_grad) {
                auto node = std::make_shared<MatMulNode>();
                node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
                node->next_nodes = { a->creator, b->creator };
                node->saved_tensors = { a, b };
                node->output = out;
                out->creator = node;
                out->requires_grad = true;
            }
            return out;
        }
    }

    if (a->device != b->device) {
        if (a->device.type == DeviceType::GPU) b = b->to(a->device);
        else a = a->to(b->device);
    }

    size_t ndim_a = a->shape.size();
    size_t ndim_b = b->shape.size();

    if (ndim_a >= 3 && ndim_b >= 3) {
        for (size_t i = 0; i < ndim_a - 2; ++i) {
            if (a->shape[i] != b->shape[i]) {
                throw std::runtime_error("[litetorch Error] Batch dimensions must match in high-dimensional matmul");
            }
        }
        int64_t batch_size = 1;
        std::vector<int64_t> outer_shape;
        for (size_t i = 0; i < ndim_a - 2; ++i) {
            batch_size *= a->shape[i];
            outer_shape.push_back(a->shape[i]);
        }
        int64_t M = a->shape[ndim_a - 2];
        int64_t K = a->shape[ndim_a - 1];
        int64_t N = b->shape[ndim_b - 1];

        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto b_c = b->is_contiguous() ? b : b->contiguous();
        auto a_flat = a_c->view({batch_size, M, K});
        auto b_flat = b_c->view({batch_size, K, N});
        auto out_flat = bmm(a_flat, b_flat);

        std::vector<int64_t> out_shape = outer_shape;
        out_shape.push_back(M);
        out_shape.push_back(N);
        return out_flat->is_contiguous() ? out_flat->view(out_shape) : out_flat->contiguous()->view(out_shape);
    }
    else if (ndim_a > 2 && ndim_b == 2) {
        int64_t in_features = a->shape.back();
        if (in_features != b->shape[0]) {
            throw std::runtime_error("[litetorch Error] Dimension mismatch in matmul");
        }
        int64_t batch_elements = a->numel() / in_features;
        auto a_c = a->is_contiguous() ? a : a->contiguous();
        auto b_c = b->is_contiguous() ? b : b->contiguous();
        auto a_flat = a_c->view({batch_elements, in_features});
        auto out_flat = matmul(a_flat, b_c);
        
        std::vector<int64_t> out_shape = a->shape;
        out_shape.back() = b->shape[1];
        return out_flat->is_contiguous() ? out_flat->view(out_shape) : out_flat->contiguous()->view(out_shape);
    }
    else if (ndim_a == 3 && ndim_b == 3) {
        return bmm(a, b);
    }

    if (ndim_a != 2 || ndim_b != 2) {
        throw std::runtime_error("[litetorch Error] MatMul requires 2D tensors");
    }
    if (a->shape[1] != b->shape[0]) {
        throw std::runtime_error("[litetorch Error] Matrix inner dimensions must match");
    }

    bool a_trans = (a->shape.size() == 2 && a->strides[0] == 1 && a->strides[1] == a->shape[0]);
    bool b_trans = (b->shape.size() == 2 && b->strides[0] == 1 && b->strides[1] == b->shape[0]);

    auto a_c = a_trans ? a : (a->is_contiguous() ? a : a->contiguous());
    auto b_c = b_trans ? b : (b->is_contiguous() ? b : b->contiguous());
    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];
    auto out = Tensor::create({M, N}, a->device, false, a->dtype);
    StorageUseGuard guard({a_c->storage, b_c->storage, out->storage});

    bool run_gpu = false;
    if (a->device.type == DeviceType::GPU) {
        auto native = BackendDispatcher::get().get_backend();
        if (native && native->is_available()) {
            run_gpu = true;
            int64_t lda = a_trans ? a->shape[0] : a_c->shape[1];
            int64_t ldb = b_trans ? b->shape[0] : b_c->shape[1];
            if (a->dtype == DataType::FP16) {
                if (a_trans || b_trans) {
                    auto a_cont = a->is_contiguous() ? a : a->contiguous();
                    auto b_cont = b->is_contiguous() ? b : b->contiguous();
                    native->matmul_half(a_cont->gpu_data(), a_cont->offset, b_cont->gpu_data(), b_cont->offset, out->gpu_data(), out->offset, M, N, K);
                } else {
                    native->matmul_half(a_c->gpu_data(), a_c->offset, b_c->gpu_data(), b_c->offset, out->gpu_data(), out->offset, M, N, K);
                }
            } else {
                native->matmul_ex(a_c->gpu_data(), a_c->offset, a_trans, lda, b_c->gpu_data(), b_c->offset, b_trans, ldb, out->gpu_data(), out->offset, M, N, K);
            }
        } else {
            void* kernel = nullptr;
            if (a_c->dtype == DataType::FP16) {
                kernel = CLBackend::get().get_kernel("", "", "matmul_half_kernel");
            } else {
                kernel = CLBackend::get().get_kernel(KernelID::MatMul);
            }
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a_c->gpu_data();
                cl_mem b_mem = b_c->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int a_off = a_c->offset;
                int b_off = b_c->offset;
                int c_off = out->offset;
                
                size_t local_sz[2] = {16, 16};
                size_t global_sz[2] = {
                    static_cast<size_t>((M + 15) / 16 * 16),
                    static_cast<size_t>((N + 15) / 16 * 16)
                };
                
                CLBackend::get().launch(kernel, {global_sz[0], global_sz[1]}, {local_sz[0], local_sz[1]},
                                        {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &M, &K, &N},
                                        {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
            }
        }
    }
    if (!run_gpu) {
        if (a_c->dtype == DataType::FP16 || b_c->dtype == DataType::FP16) {
            auto a_fp32 = a_c->cast(DataType::FP32);
            auto b_fp32 = b_c->cast(DataType::FP32);
            auto out_fp32 = matmul(a_fp32, b_fp32);
            auto out_fp16 = out_fp32->cast(DataType::FP16);
            out->copy_(out_fp16);
        } else {
            float* A = a_c->data_ptr();
            float* B = b_c->data_ptr();
            float* C = out->data_ptr();
            std::fill(C, C + M * N, 0.0f);

            ThreadPool::get().parallel_for(0, M, [&](int64_t i) {
                float* C_row = C + i * N;
                for (int k = 0; k < K; ++k) {
                    float aval = A[i * K + k];
                    float* B_row = B + k * N;
                    for (int j = 0; j < N; ++j) {
                        C_row[j] += aval * B_row[j];
                    }
                }
            });
            if (out->device.type == DeviceType::GPU) {
                CLBackend::get().write(out->gpu_data(), out->numel() * sizeof(float), C);
            }
        }
    }

    if (a->requires_grad || b->requires_grad) {
        auto node = std::make_shared<MatMulNode>();
        node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
        node->next_nodes = { a->creator, b->creator };
        node->saved_tensors = { a, b };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> bmm(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->device.type == DeviceType::CPU) {
        if (a->dtype != DataType::FP32 || b->dtype != DataType::FP32) {
            auto a_fp32 = a->dtype == DataType::FP32 ? a : a->cast(DataType::FP32);
            auto b_fp32 = b->dtype == DataType::FP32 ? b : b->cast(DataType::FP32);
            auto out_fp32 = bmm(a_fp32, b_fp32);
            auto out = out_fp32->cast(a->dtype);
            if (a->requires_grad || b->requires_grad) {
                auto node = std::make_shared<BmmNode>();
                node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
                node->next_nodes = { a->creator, b->creator };
                node->saved_tensors = { a, b };
                node->output = out;
                out->creator = node;
                out->requires_grad = true;
            }
            return out;
        }
    }

    if (a->device != b->device) {
        if (a->device.type == DeviceType::GPU) b = b->to(a->device);
        else a = a->to(b->device);
    }
    if (a->shape.size() != 3 || b->shape.size() != 3) {
        throw std::runtime_error("[litetorch Error] bmm requires 3D tensors");
    }
    if (a->shape[0] != b->shape[0]) {
        throw std::runtime_error("[litetorch Error] Batch size mismatch in bmm");
    }
    if (a->shape[2] != b->shape[1]) {
        throw std::runtime_error("[litetorch Error] Dimension mismatch in bmm");
    }
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    auto b_c = b->is_contiguous() ? b : b->contiguous();
    int B = a_c->shape[0];
    int M = a_c->shape[1];
    int K = a_c->shape[2];
    int N = b_c->shape[2];
    auto out = Tensor::create({B, M, N}, a_c->device, false, a_c->dtype);
    StorageUseGuard guard({a_c->storage, b_c->storage, out->storage});
    bool run_gpu = false;
    if (a_c->device.type == DeviceType::GPU) {
        auto native = BackendDispatcher::get().get_backend();
        if (native && native->is_available()) {
            run_gpu = true;
            if (a_c->dtype == DataType::FP16) {
                native->bmm_half(a_c->gpu_data(), a_c->offset, b_c->gpu_data(), b_c->offset, out->gpu_data(), out->offset, B, M, N, K);
            } else {
                native->bmm(a_c->gpu_data(), a_c->offset, b_c->gpu_data(), b_c->offset, out->gpu_data(), out->offset, B, M, N, K);
            }
        } else {
            void* kernel = nullptr;
            if (a_c->dtype == DataType::FP16) {
                kernel = CLBackend::get().get_kernel("", "", "bmm_half_kernel");
            } else {
                kernel = CLBackend::get().get_kernel(KernelID::BMM);
            }
            if (kernel) {
                run_gpu = true;
                cl_mem a_mem = a_c->gpu_data();
                cl_mem b_mem = b_c->gpu_data();
                cl_mem c_mem = out->gpu_data();
                int a_off = a_c->offset;
                int b_off = b_c->offset;
                int c_off = out->offset;
                size_t local_sz[3] = {16, 16, 1};
                size_t global_sz[3] = {
                    static_cast<size_t>((M + 15) / 16 * 16),
                    static_cast<size_t>((N + 15) / 16 * 16),
                    static_cast<size_t>(B)
                };
                CLBackend::get().launch(kernel, {global_sz[0], global_sz[1], global_sz[2]}, {local_sz[0], local_sz[1], local_sz[2]},
                                        {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &M, &K, &N, &B},
                                        {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
            }
        }
    }
    if (!run_gpu) {
        if (a_c->dtype == DataType::FP16 || b_c->dtype == DataType::FP16) {
            auto a_fp32 = a_c->cast(DataType::FP32);
            auto b_fp32 = b_c->cast(DataType::FP32);
            auto out_fp32 = bmm(a_fp32, b_fp32);
            auto out_fp16 = out_fp32->cast(DataType::FP16);
            out->copy_(out_fp16);
        } else {
            float* A = a_c->data_ptr();
            float* B_ptr = b_c->data_ptr();
            float* C = out->data_ptr();
            std::fill(C, C + B * M * N, 0.0f);
            ThreadPool::get().parallel_for(0, B, [&](int64_t b_idx) {
                float* A_b = A + b_idx * M * K;
                float* B_b = B_ptr + b_idx * K * N;
                float* C_b = C + b_idx * M * N;
                for (int i = 0; i < M; ++i) {
                    for (int k = 0; k < K; ++k) {
                        float a_val = A_b[i * K + k];
                        for (int j = 0; j < N; ++j) {
                            C_b[i * N + j] += a_val * B_b[k * N + j];
                        }
                    }
                }
            });
            if (out->device.type == DeviceType::GPU) {
                CLBackend::get().write(out->gpu_data(), out->numel() * sizeof(float), C);
            }
        }
    }
    if (a->requires_grad || b->requires_grad) {
        auto node = std::make_shared<BmmNode>();
        node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
        node->next_nodes = { a->creator, b->creator };
        node->saved_tensors = { a, b };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

}
}
