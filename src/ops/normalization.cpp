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
}

class LayerNormNode : public Node {
public:
    std::vector<int64_t> normalized_shape;
    float eps;
    LayerNormNode(const std::vector<int64_t>& normalized_shape, float eps)
        : Node("LayerNorm"), normalized_shape(normalized_shape), eps(eps) {}
        
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto input = saved_tensors[0];
        auto weight = saved_tensors[1];
        auto bias = saved_tensors[2];
        auto save_mean = saved_tensors[3];
        auto save_var = saved_tensors[4];
        
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto weight_c = weight ? (weight->is_contiguous() ? weight : weight->contiguous()) : nullptr;
        auto bias_c = bias ? (bias->is_contiguous() ? bias : bias->contiguous()) : nullptr;
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto save_mean_c = save_mean->is_contiguous() ? save_mean : save_mean->contiguous();
        auto save_var_c = save_var->is_contiguous() ? save_var : save_var->contiguous();

        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        std::shared_ptr<Tensor> grad_weight = nullptr;
        std::shared_ptr<Tensor> grad_bias = nullptr;
        if (weight_c) grad_weight = Tensor::create(weight_c->shape, weight_c->device);
        if (bias_c) grad_bias = Tensor::create(bias_c->shape, bias_c->device);
        
        int64_t M = 1;
        for (auto s : normalized_shape) M *= s;
        int64_t N = input_c->numel() / M;
        
        StorageUseGuard guard({input_c->storage, weight_c ? weight_c->storage : nullptr, bias_c ? bias_c->storage : nullptr,
                              gout_c->storage, grad_input->storage,
                              grad_weight ? grad_weight->storage : nullptr,
                              grad_bias ? grad_bias->storage : nullptr,
                              save_mean_c->storage, save_var_c->storage});
                              
        if (input_c->device.type == DeviceType::GPU) {
            {
                auto kernel = CLBackend::get().get_kernel(KernelID::LayerNormBackwardDx);
                cl_mem in_mem = input_c->gpu_data();
                int in_off = input_c->offset;
                cl_mem gout_mem = gout_c->gpu_data();
                int gout_off = gout_c->offset;
                cl_mem w_mem = weight_c ? weight_c->gpu_data() : cl_mem();
                int w_off = weight_c ? weight_c->offset : 0;
                int has_weight = weight_c ? 1 : 0;
                cl_mem gin_mem = grad_input->gpu_data();
                int gin_off = grad_input->offset;
                cl_mem sm_mem = save_mean_c->gpu_data();
                int sm_off = save_mean_c->offset;
                cl_mem sv_mem = save_var_c->gpu_data();
                int sv_off = save_var_c->offset;
                int n_val = N;
                int m_val = M;
                float eps_val = eps;
                CLBackend::get().launch(kernel, {static_cast<size_t>(N)}, {},
                    {&in_mem, &in_off, &gout_mem, &gout_off, &w_mem, &w_off, &has_weight, &gin_mem, &gin_off, &sm_mem, &sm_off, &sv_mem, &sv_off, &n_val, &m_val, &eps_val},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(float)});
            }
            if (grad_weight) {
                auto kernel = CLBackend::get().get_kernel(KernelID::LayerNormBackwardDw);
                cl_mem in_mem = input_c->gpu_data();
                int in_off = input_c->offset;
                cl_mem gout_mem = gout_c->gpu_data();
                int gout_off = gout_c->offset;
                cl_mem gw_mem = grad_weight->gpu_data();
                int gw_off = grad_weight->offset;
                cl_mem sm_mem = save_mean_c->gpu_data();
                int sm_off = save_mean_c->offset;
                cl_mem sv_mem = save_var_c->gpu_data();
                int sv_off = save_var_c->offset;
                int n_val = N;
                int m_val = M;
                float eps_val = eps;
                CLBackend::get().launch(kernel, {static_cast<size_t>(M)}, {},
                    {&in_mem, &in_off, &gout_mem, &gout_off, &gw_mem, &gw_off, &sm_mem, &sm_off, &sv_mem, &sv_off, &n_val, &m_val, &eps_val},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(float)});
            }
            if (grad_bias) {
                auto kernel = CLBackend::get().get_kernel(KernelID::LayerNormBackwardDb);
                cl_mem gout_mem = gout_c->gpu_data();
                int gout_off = gout_c->offset;
                cl_mem gb_mem = grad_bias->gpu_data();
                int gb_off = grad_bias->offset;
                int n_val = N;
                int m_val = M;
                CLBackend::get().launch(kernel, {static_cast<size_t>(M)}, {},
                    {&gout_mem, &gout_off, &gb_mem, &gb_off, &n_val, &m_val},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int)});
            }
        } else {
            float* in_ptr = input_c->data_ptr();
            float* dy_ptr = gout_c->data_ptr();
            float* dx_ptr = grad_input->data_ptr();
            float* weight_ptr = weight_c ? weight_c->data_ptr() : nullptr;
            float* dw_ptr = grad_weight ? grad_weight->data_ptr() : nullptr;
            float* db_ptr = grad_bias ? grad_bias->data_ptr() : nullptr;
            
            float* means = save_mean_c->data_ptr();
            float* inv_stds = save_var_c->data_ptr();

            ThreadPool::get().parallel_for(0, N, [&](int64_t r) {
                float mean = means[r];
                float inv_std = inv_stds[r];

                float sum_dy = 0.0f;
                float sum_dy_xhat = 0.0f;
                for (int64_t c = 0; c < M; ++c) {
                    int64_t idx = r * M + c;
                    float dy = dy_ptr[idx];
                    float x_hat = (in_ptr[idx] - mean) * inv_std;
                    float w = weight_ptr ? weight_ptr[c] : 1.0f;
                    sum_dy += dy * w;
                    sum_dy_xhat += dy * w * x_hat;
                }

                for (int64_t c = 0; c < M; ++c) {
                    int64_t idx = r * M + c;
                    float x_hat = (in_ptr[idx] - mean) * inv_std;
                    float dy = dy_ptr[idx];
                    float w = weight_ptr ? weight_ptr[c] : 1.0f;
                    dx_ptr[idx] = inv_std * (dy * w - (sum_dy + x_hat * sum_dy_xhat) / M);
                }
            });

            if (dw_ptr) {
                ThreadPool::get().parallel_for(0, M, [&](int64_t c) {
                    float sum_dw = 0.0f;
                    for (int r = 0; r < N; ++r) {
                        int64_t idx = r * M + c;
                        float x_hat = (in_ptr[idx] - means[r]) * inv_stds[r];
                        sum_dw += dy_ptr[idx] * x_hat;
                    }
                    dw_ptr[c] = sum_dw;
                });
            }

            if (db_ptr) {
                ThreadPool::get().parallel_for(0, M, [&](int64_t c) {
                    float sum_db = 0.0f;
                    for (int r = 0; r < N; ++r) {
                        sum_db += dy_ptr[r * M + c];
                    }
                    db_ptr[c] = sum_db;
                });
            }

            if (grad_input->device.type == DeviceType::TPU) {
                auto tpu = BackendDispatcher::get().get_tpu_backend();
                if (tpu) {
                    tpu->write(grad_input->gpu_data(), grad_input->numel() * sizeof(float), dx_ptr);
                    if (grad_weight && dw_ptr) tpu->write(grad_weight->gpu_data(), grad_weight->numel() * sizeof(float), dw_ptr);
                    if (grad_bias && db_ptr) tpu->write(grad_bias->gpu_data(), grad_bias->numel() * sizeof(float), db_ptr);
                }
            }
        }
        
        return { grad_input, grad_weight, grad_bias };
    }
};

class FusedAddLayerNormNode : public Node {
public:
    std::vector<int64_t> normalized_shape;
    float eps;
    FusedAddLayerNormNode(const std::vector<int64_t>& normalized_shape, float eps)
        : Node("FusedAddLayerNorm"), normalized_shape(normalized_shape), eps(eps) {}
        
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto input = saved_tensors[0];
        auto residual = saved_tensors[1];
        auto weight = saved_tensors[2];
        auto bias = saved_tensors[3];
        auto save_mean = saved_tensors[4];
        auto save_var = saved_tensors[5];
        
        auto sum_input = Ops::add(input, residual);
        auto ln_node = std::make_shared<LayerNormNode>(normalized_shape, eps);
        ln_node->saved_tensors = { sum_input, weight, bias, save_mean, save_var };
        auto grads = ln_node->backward(grad_output);
        
        return { grads[0], grads[0], grads[1], grads[2] };
    }
};

class BatchNorm2dNode : public Node {
public:
    BatchNorm2dNode(float eps) : Node("BatchNorm2d"), eps(eps) {}
    float eps;
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto input = saved_tensors[0];
        auto weight = saved_tensors[1];
        auto mean = saved_tensors[2];
        auto var = saved_tensors[3];
        
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto weight_c = weight->is_contiguous() ? weight : weight->contiguous();
        auto mean_c = mean->is_contiguous() ? mean : mean->contiguous();
        auto var_c = var->is_contiguous() ? var : var->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();

        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        auto grad_weight = Tensor::create(weight_c->shape, weight_c->device);
        auto grad_bias = Tensor::create(weight_c->shape, weight_c->device);
        
        int N = input_c->shape[0];
        int C = input_c->shape[1];
        int H = input_c->shape[2];
        int W = input_c->shape[3];
        int M = N * H * W;
        
        StorageUseGuard guard({input_c->storage, weight_c->storage, mean_c->storage, var_c->storage,
                               gout_c->storage, grad_input->storage, grad_weight->storage, grad_bias->storage});
                               
        if (input_c->device.type == DeviceType::GPU) {
            {
                auto kernel = CLBackend::get().get_kernel(KernelID::BatchNorm2dBackwardStats);
                cl_mem in_mem = input_c->gpu_data();
                int in_off = input_c->offset;
                cl_mem gout_mem = gout_c->gpu_data();
                int gout_off = gout_c->offset;
                cl_mem sm_mem = mean_c->gpu_data();
                int sm_off = mean_c->offset;
                cl_mem sv_mem = var_c->gpu_data();
                int sv_off = var_c->offset;
                cl_mem gw_mem = grad_weight->gpu_data();
                int gw_off = grad_weight->offset;
                cl_mem gb_mem = grad_bias->gpu_data();
                int gb_off = grad_bias->offset;
                int n_val = N;
                int c_val = C;
                int h_val = H;
                int w_val = W;
                float eps_val = eps;
                
                CLBackend::get().launch(kernel, {static_cast<size_t>(C)}, {},
                    {&in_mem, &in_off, &gout_mem, &gout_off, &sm_mem, &sm_off, &sv_mem, &sv_off, &gw_mem, &gw_off, &gb_mem, &gb_off, &n_val, &c_val, &h_val, &w_val, &eps_val},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(float)});
            }
            {
                auto kernel = CLBackend::get().get_kernel(KernelID::BatchNorm2dBackwardDx);
                cl_mem in_mem = input_c->gpu_data();
                int in_off = input_c->offset;
                cl_mem gout_mem = gout_c->gpu_data();
                int gout_off = gout_c->offset;
                cl_mem sm_mem = mean_c->gpu_data();
                int sm_off = mean_c->offset;
                cl_mem sv_mem = var_c->gpu_data();
                int sv_off = var_c->offset;
                cl_mem w_mem = weight_c->gpu_data();
                int w_off = weight_c->offset;
                cl_mem gw_mem = grad_weight->gpu_data();
                int gw_off = grad_weight->offset;
                cl_mem gb_mem = grad_bias->gpu_data();
                int gb_off = grad_bias->offset;
                cl_mem gin_mem = grad_input->gpu_data();
                int gin_off = grad_input->offset;
                int n_val = N;
                int c_val = C;
                int h_val = H;
                int w_val = W;
                float eps_val = eps;
                
                int total = N * C * H * W;
                CLBackend::get().launch(kernel, {static_cast<size_t>(total)}, {},
                    {&in_mem, &in_off, &gout_mem, &gout_off, &sm_mem, &sm_off, &sv_mem, &sv_off, &w_mem, &w_off, &gw_mem, &gw_off, &gb_mem, &gb_off, &gin_mem, &gin_off, &n_val, &c_val, &h_val, &w_val, &eps_val},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(float)});
            }
        } else {
            float* in_ptr = input_c->data_ptr();
            float* w_ptr = weight_c->data_ptr();
            float* mean_ptr = mean_c->data_ptr();
            float* var_ptr = var_c->data_ptr();
            float* gout_ptr = gout_c->data_ptr();
            float* gin_ptr = grad_input->data_ptr();
            float* gw_ptr = grad_weight->data_ptr();
            float* gb_ptr = grad_bias ? grad_bias->data_ptr() : nullptr;
            
            if (gw_ptr) std::fill(gw_ptr, gw_ptr + weight_c->numel(), 0.0f);
            if (gb_ptr) std::fill(gb_ptr, gb_ptr + weight_c->numel(), 0.0f);
            
            ThreadPool::get().parallel_for(0, C, [&](int64_t c) {
                float inv_std = 1.0f / std::sqrt(var_ptr[c] + eps);
                float dscale_sum = 0.0f;
                float dshift_sum = 0.0f;
                for (int b = 0; b < N; ++b) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            int idx = ((b * C + c) * H + h) * W + w;
                            float x_hat = (in_ptr[idx] - mean_ptr[c]) * inv_std;
                            dscale_sum += gout_ptr[idx] * x_hat;
                            dshift_sum += gout_ptr[idx];
                        }
                    }
                }
                gw_ptr[c] = dscale_sum;
                gb_ptr[c] = dshift_sum;
                for (int b = 0; b < N; ++b) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            int idx = ((b * C + c) * H + h) * W + w;
                            float x_hat = (in_ptr[idx] - mean_ptr[c]) * inv_std;
                            gin_ptr[idx] = w_ptr[c] * inv_std / M * (M * gout_ptr[idx] - dscale_sum * x_hat - dshift_sum);
                        }
                    }
                }
            });
        }
        return { grad_input, grad_weight, grad_bias };
    }
};

namespace Ops {

std::shared_ptr<Tensor> layer_norm(std::shared_ptr<Tensor> input, const std::vector<int64_t>& normalized_shape, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias, float eps) {
    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto weight_c = weight ? (weight->is_contiguous() ? weight : weight->contiguous()) : nullptr;
    auto bias_c = bias ? (bias->is_contiguous() ? bias : bias->contiguous()) : nullptr;
    auto out = Tensor::create(input_c->shape, input_c->device);
    StorageUseGuard guard({input_c->storage, weight_c ? weight_c->storage : nullptr, bias_c ? bias_c->storage : nullptr, out->storage});
    
    int64_t M = 1;
    for (auto s : normalized_shape) M *= s;
    int64_t N = input_c->numel() / M;
    
    auto save_mean = Tensor::create({N}, input_c->device, false, DataType::FP32);
    auto save_var = Tensor::create({N}, input_c->device, false, DataType::FP32);
    
    if (input_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::LayerNormForward);
        cl_mem in_mem = input_c->gpu_data();
        int in_off = input_c->offset;
        cl_mem w_mem = weight_c ? weight_c->gpu_data() : cl_mem();
        int w_off = weight_c ? weight_c->offset : 0;
        int has_weight = weight_c ? 1 : 0;
        cl_mem b_mem = bias_c ? bias_c->gpu_data() : cl_mem();
        int b_off = bias_c ? bias_c->offset : 0;
        int has_bias = bias_c ? 1 : 0;
        cl_mem out_mem = out->gpu_data();
        int out_off = out->offset;
        cl_mem sm_mem = save_mean->gpu_data();
        int sm_off = save_mean->offset;
        cl_mem sv_mem = save_var->gpu_data();
        int sv_off = save_var->offset;
        int n_val = N;
        int m_val = M;
        float eps_val = eps;
        
        CLBackend::get().launch(kernel, {static_cast<size_t>(N)}, {},
            {&in_mem, &in_off, &w_mem, &w_off, &has_weight, &b_mem, &b_off, &has_bias, &out_mem, &out_off, &sm_mem, &sm_off, &sv_mem, &sv_off, &n_val, &m_val, &eps_val},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(float)});
    } else {
        float* in_ptr = input_c->data_ptr();
        float* weight_ptr = weight_c ? weight_c->data_ptr() : nullptr;
        float* bias_ptr = bias_c ? bias_c->data_ptr() : nullptr;
        float* out_ptr = out->data_ptr();
        float* sm_ptr = save_mean->data_ptr();
        float* sv_ptr = save_var->data_ptr();
        
        ThreadPool::get().parallel_for(0, N, [&](int64_t r) {
            float mean = 0.0f;
            for (int64_t c = 0; c < M; ++c) mean += in_ptr[r * M + c];
            mean /= M;
            float var = 0.0f;
            for (int64_t c = 0; c < M; ++c) {
                float diff = in_ptr[r * M + c] - mean;
                var += diff * diff;
            }
            var /= M;
            float inv_std = 1.0f / std::sqrt(var + eps);
            sm_ptr[r] = mean;
            sv_ptr[r] = inv_std;
            
            for (int64_t c = 0; c < M; ++c) {
                int64_t idx = r * M + c;
                float x_hat = (in_ptr[idx] - mean) * inv_std;
                float w = weight_ptr ? weight_ptr[c] : 1.0f;
                float b = bias_ptr ? bias_ptr[c] : 0.0f;
                out_ptr[idx] = w * x_hat + b;
            }
        });
        if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) {
                tpu->write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
                tpu->write(save_mean->gpu_data(), save_mean->numel() * sizeof(float), sm_ptr);
                tpu->write(save_var->gpu_data(), save_var->numel() * sizeof(float), sv_ptr);
            }
        }
    }
    
    bool req_grad = input->requires_grad || (weight && weight->requires_grad) || (bias && bias->requires_grad);
    if (req_grad) {
        auto node = std::make_shared<LayerNormNode>(normalized_shape, eps);
        node->inputs = { {input, input->requires_grad}, {weight, weight && weight->requires_grad}, {bias, bias && bias->requires_grad} };
        node->next_nodes = { input->creator, weight ? weight->creator : nullptr, bias ? bias->creator : nullptr };
        node->saved_tensors = { input, weight, bias, save_mean, save_var };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> fused_add_layernorm(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> residual, const std::vector<int64_t>& normalized_shape, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias, float eps) {
    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto residual_c = residual->is_contiguous() ? residual : residual->contiguous();
    auto weight_c = weight ? (weight->is_contiguous() ? weight : weight->contiguous()) : nullptr;
    auto bias_c = bias ? (bias->is_contiguous() ? bias : bias->contiguous()) : nullptr;
    auto out = Tensor::create(input_c->shape, input_c->device);
    StorageUseGuard guard({input_c->storage, residual_c->storage, weight_c ? weight_c->storage : nullptr, bias_c ? bias_c->storage : nullptr, out->storage});
    
    int64_t M = 1;
    for (auto s : normalized_shape) M *= s;
    int64_t N = input_c->numel() / M;
    
    auto save_mean = Tensor::create({N}, input_c->device, false, DataType::FP32);
    auto save_var = Tensor::create({N}, input_c->device, false, DataType::FP32);
    
    if (input_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::FusedAddLayerNormForward);
        cl_mem in_mem = input_c->gpu_data();
        int in_off = input_c->offset;
        cl_mem res_mem = residual_c->gpu_data();
        int res_off = residual_c->offset;
        cl_mem w_mem = weight_c ? weight_c->gpu_data() : cl_mem();
        int w_off = weight_c ? weight_c->offset : 0;
        int has_weight = weight_c ? 1 : 0;
        cl_mem b_mem = bias_c ? bias_c->gpu_data() : cl_mem();
        int b_off = bias_c ? bias_c->offset : 0;
        int has_bias = bias_c ? 1 : 0;
        cl_mem out_mem = out->gpu_data();
        int out_off = out->offset;
        cl_mem sm_mem = save_mean->gpu_data();
        int sm_off = save_mean->offset;
        cl_mem sv_mem = save_var->gpu_data();
        int sv_off = save_var->offset;
        int n_val = N;
        int m_val = M;
        float eps_val = eps;
        
        CLBackend::get().launch(kernel, {static_cast<size_t>(N)}, {},
            {&in_mem, &in_off, &res_mem, &res_off, &w_mem, &w_off, &has_weight, &b_mem, &b_off, &has_bias, &out_mem, &out_off, &sm_mem, &sm_off, &sv_mem, &sv_off, &n_val, &m_val, &eps_val},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(float)});
    } else {
        float* in_ptr = input_c->data_ptr();
        float* res_ptr = residual_c->data_ptr();
        float* weight_ptr = weight_c ? weight_c->data_ptr() : nullptr;
        float* bias_ptr = bias_c ? bias_c->data_ptr() : nullptr;
        float* out_ptr = out->data_ptr();
        float* sm_ptr = save_mean->data_ptr();
        float* sv_ptr = save_var->data_ptr();
        
        ThreadPool::get().parallel_for(0, N, [&](int64_t r) {
            float mean = 0.0f;
            for (int64_t c = 0; c < M; ++c) {
                mean += in_ptr[r * M + c] + res_ptr[r * M + c];
            }
            mean /= M;
            float var = 0.0f;
            for (int64_t c = 0; c < M; ++c) {
                float val = in_ptr[r * M + c] + res_ptr[r * M + c];
                float diff = val - mean;
                var += diff * diff;
            }
            var /= M;
            float inv_std = 1.0f / std::sqrt(var + eps);
            sm_ptr[r] = mean;
            sv_ptr[r] = inv_std;
            
            for (int64_t c = 0; c < M; ++c) {
                int64_t idx = r * M + c;
                float val = in_ptr[idx] + res_ptr[idx];
                float x_hat = (val - mean) * inv_std;
                float w = weight_ptr ? weight_ptr[c] : 1.0f;
                float b = bias_ptr ? bias_ptr[c] : 0.0f;
                out_ptr[idx] = w * x_hat + b;
            }
        });
    }
    
    bool req_grad = input->requires_grad || residual->requires_grad || (weight && weight->requires_grad) || (bias && bias->requires_grad);
    if (req_grad) {
        auto node = std::make_shared<FusedAddLayerNormNode>(normalized_shape, eps);
        node->inputs = { {input, input->requires_grad}, {residual, residual->requires_grad} };
        node->next_nodes = { input->creator, residual->creator };
        if (weight) {
            node->inputs.push_back({weight, weight->requires_grad});
            node->next_nodes.push_back(weight->creator);
        } else {
            node->inputs.push_back(NodeInput{std::shared_ptr<Tensor>(), false});
            node->next_nodes.push_back(nullptr);
        }
        if (bias) {
            node->inputs.push_back({bias, bias->requires_grad});
            node->next_nodes.push_back(bias->creator);
        } else {
            node->inputs.push_back(NodeInput{std::shared_ptr<Tensor>(), false});
            node->next_nodes.push_back(nullptr);
        }
        node->saved_tensors = { input, residual, weight, bias, save_mean, save_var };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}


std::shared_ptr<Tensor> batch_norm2d(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> running_mean, std::shared_ptr<Tensor> running_var, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias, bool training, float momentum, float eps) {
    if (input->shape.size() != 4) {
        throw std::runtime_error("[litetorch Error] BatchNorm2d requires 4D input");
    }

    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto running_mean_c = running_mean->is_contiguous() ? running_mean : running_mean->contiguous();
    auto running_var_c = running_var->is_contiguous() ? running_var : running_var->contiguous();
    auto weight_c = weight->is_contiguous() ? weight : weight->contiguous();
    auto bias_c = bias->is_contiguous() ? bias : bias->contiguous();

    int N = input_c->shape[0];
    int C = input_c->shape[1];
    int H = input_c->shape[2];
    int W = input_c->shape[3];
    int M = N * H * W;
    auto out = Tensor::create(input_c->shape, input_c->device);
    StorageUseGuard guard({input_c->storage, running_mean_c->storage, running_var_c->storage, weight_c->storage, bias_c->storage, out->storage});

    auto save_mean = Tensor::create({C}, input_c->device);
    auto save_var = Tensor::create({C}, input_c->device);

    if (input_c->device.type == DeviceType::GPU) {
        {
            auto kernel = CLBackend::get().get_kernel(KernelID::BatchNorm2dForwardStats);
            cl_mem in_mem = input_c->gpu_data();
            int in_off = input_c->offset;
            cl_mem rm_mem = running_mean_c->gpu_data();
            int rm_off = running_mean_c->offset;
            cl_mem rv_mem = running_var_c->gpu_data();
            int rv_off = running_var_c->offset;
            cl_mem sm_mem = save_mean->gpu_data();
            int sm_off = save_mean->offset;
            cl_mem sv_mem = save_var->gpu_data();
            int sv_off = save_var->offset;
            int n_val = N;
            int c_val = C;
            int h_val = H;
            int w_val = W;
            int train_val = training ? 1 : 0;
            float mom_val = momentum;
            float eps_val = eps;

            CLBackend::get().launch(kernel, {static_cast<size_t>(C)}, {},
                {&in_mem, &in_off, &rm_mem, &rm_off, &rv_mem, &rv_off, &sm_mem, &sm_off, &sv_mem, &sv_off, &n_val, &c_val, &h_val, &w_val, &train_val, &mom_val, &eps_val},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(float), sizeof(float)});
        }
        {
            auto kernel = CLBackend::get().get_kernel(KernelID::BatchNorm2dForwardNorm);
            cl_mem in_mem = input_c->gpu_data();
            int in_off = input_c->offset;
            cl_mem sm_mem = save_mean->gpu_data();
            int sm_off = save_mean->offset;
            cl_mem sv_mem = save_var->gpu_data();
            int sv_off = save_var->offset;
            cl_mem w_mem = weight_c->gpu_data();
            int w_off = weight_c->offset;
            cl_mem b_mem = bias_c->gpu_data();
            int b_off = bias_c->offset;
            cl_mem out_mem = out->gpu_data();
            int out_off = out->offset;
            int n_val = N;
            int c_val = C;
            int h_val = H;
            int w_val = W;
            float eps_val = eps;

            int total = N * C * H * W;
            CLBackend::get().launch(kernel, {static_cast<size_t>(total)}, {},
                {&in_mem, &in_off, &sm_mem, &sm_off, &sv_mem, &sv_off, &w_mem, &w_off, &b_mem, &b_off, &out_mem, &out_off, &n_val, &c_val, &h_val, &w_val, &eps_val},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(float)});
        }
    } else {
        float* in_ptr = input_c->data_ptr();
        float* rmean_ptr = running_mean_c->data_ptr();
        float* rvar_ptr = running_var_c->data_ptr();
        float* w_ptr = weight_c->data_ptr();
        float* b_ptr = bias_c->data_ptr();
        float* out_ptr = out->data_ptr();
        float* smean_ptr = save_mean->data_ptr();
        float* svar_ptr = save_var->data_ptr();

        ThreadPool::get().parallel_for(0, C, [&](int64_t c) {
            float m_val = 0.0f;
            float v_val = 0.0f;
            if (training) {
                float sum_val = 0.0f;
                for (int b = 0; b < N; ++b) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            int idx = ((b * C + c) * H + h) * W + w;
                            sum_val += in_ptr[idx];
                        }
                    }
                }
                m_val = sum_val / M;
                smean_ptr[c] = m_val;
                float sum_sq_val = 0.0f;
                for (int b = 0; b < N; ++b) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            int idx = ((b * C + c) * H + h) * W + w;
                            float diff = in_ptr[idx] - m_val;
                            sum_sq_val += diff * diff;
                        }
                    }
                }
                v_val = sum_sq_val / M;
                svar_ptr[c] = v_val;
                rmean_ptr[c] = (1.0f - momentum) * rmean_ptr[c] + momentum * m_val;
                float unbiased_factor = M > 1 ? static_cast<float>(M) / (M - 1) : 1.0f;
                rvar_ptr[c] = (1.0f - momentum) * rvar_ptr[c] + momentum * v_val * unbiased_factor;
            } else {
                m_val = rmean_ptr[c];
                v_val = rvar_ptr[c];
                smean_ptr[c] = m_val;
                svar_ptr[c] = v_val;
            }
            float inv_std = 1.0f / std::sqrt(v_val + eps);
            for (int b = 0; b < N; ++b) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int idx = ((b * C + c) * H + h) * W + w;
                        float x_hat = (in_ptr[idx] - m_val) * inv_std;
                        out_ptr[idx] = w_ptr[c] * x_hat + b_ptr[c];
                    }
                }
            }
        });
    }

    if (input->requires_grad || weight->requires_grad || bias->requires_grad) {
        auto node = std::make_shared<BatchNorm2dNode>(eps);
        node->inputs = { {input, input->requires_grad}, {weight, weight->requires_grad}, {bias, bias->requires_grad} };
        node->next_nodes = { input->creator, weight->creator, bias->creator };
        node->saved_tensors = { input, weight, save_mean, save_var };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

}
}
