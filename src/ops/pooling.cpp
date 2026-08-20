#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
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

class MaxPool2dNode : public Node {
public:
    MaxPool2dNode(int kernel_size, int stride, int padding)
        : Node("MaxPool2d"), kernel_size(kernel_size), stride(stride), padding(padding) {}
    int kernel_size;
    int stride;
    int padding;
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto input = saved_tensors[0];
        auto save_indices = saved_tensors[1];
        auto out_shared = output.lock();
        if (!out_shared) return { nullptr };
        
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto ind_c = save_indices->is_contiguous() ? save_indices : save_indices->contiguous();
        
        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        int N = input_c->shape[0];
        int C = input_c->shape[1];
        int H_in = input_c->shape[2];
        int W_in = input_c->shape[3];
        int H_out = gout_c->shape[2];
        int W_out = gout_c->shape[3];

        StorageUseGuard guard({input_c->storage, ind_c->storage, gout_c->storage, grad_input->storage});

        if (input_c->device.type == DeviceType::GPU) {
            auto zero_kernel = CLBackend::get().get_kernel(KernelID::FillZero);
            int gin_size = grad_input->numel();
            cl_mem gin_mem = grad_input->gpu_data();
            CLBackend::get().launch(zero_kernel, {static_cast<size_t>(gin_size)}, {}, {&gin_mem, &gin_size}, {sizeof(cl_mem), sizeof(int)});

            auto kernel = CLBackend::get().get_kernel(KernelID::MaxPool2dBackward);
            cl_mem ind_mem = ind_c->gpu_data();
            cl_mem gout_mem = gout_c->gpu_data();
            int ind_off = ind_c->offset;
            int gout_off = gout_c->offset;
            int gin_off = grad_input->offset;

            int total_threads = N * C * H_out * W_out;
            CLBackend::get().launch(kernel, {static_cast<size_t>(total_threads)}, {},
                                    {&ind_mem, &ind_off, &gout_mem, &gout_off, &gin_mem, &gin_off,
                                     &N, &C, &H_in, &W_in, &H_out, &W_out},
                                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int),
                                     sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
        } else {
            float* ind_ptr = ind_c->data_ptr();
            float* gout_ptr = gout_c->data_ptr();
            float* gin_ptr = grad_input->data_ptr();
            std::fill(gin_ptr, gin_ptr + grad_input->numel(), 0.0f);
            ThreadPool::get().parallel_for(0, N * C, [&](int64_t thread_idx) {
                int b = thread_idx / C;
                int c = thread_idx % C;
                for (int ho = 0; ho < H_out; ++ho) {
                    for (int wo = 0; wo < W_out; ++wo) {
                        int out_idx = ((b * C + c) * H_out + ho) * W_out + wo;
                        int max_idx = (int)ind_ptr[out_idx];
                        if (max_idx >= 0) {
                            gin_ptr[max_idx] += gout_ptr[out_idx];
                        }
                    }
                }
            });
        }
        return { grad_input };
    }
};

class MaxPool3dNode : public Node {
public:
    MaxPool3dNode(int kernel_size, int stride, int padding)
        : Node("MaxPool3d"), kernel_size(kernel_size), stride(stride), padding(padding) {}
    int kernel_size;
    int stride;
    int padding;

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto input = saved_tensors[0];
        auto save_indices = saved_tensors[1];
        auto out_shared = output.lock();
        if (!out_shared) return { nullptr };
        
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto ind_c = save_indices->is_contiguous() ? save_indices : save_indices->contiguous();
        
        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        int N = input_c->shape[0];
        int C = input_c->shape[1];
        int D_in = input_c->shape[2];
        int H_in = input_c->shape[3];
        int W_in = input_c->shape[4];
        int D_out = gout_c->shape[2];
        int H_out = gout_c->shape[3];
        int W_out = gout_c->shape[4];

        StorageUseGuard guard({input_c->storage, ind_c->storage, gout_c->storage, grad_input->storage});

        if (input_c->device.type == DeviceType::GPU) {
            auto zero_kernel = CLBackend::get().get_kernel(KernelID::FillZero);
            int gin_size = grad_input->numel();
            cl_mem gin_mem = grad_input->gpu_data();
            CLBackend::get().launch(zero_kernel, {static_cast<size_t>(gin_size)}, {}, {&gin_mem, &gin_size}, {sizeof(cl_mem), sizeof(int)});

            auto kernel = CLBackend::get().get_kernel(KernelID::MaxPool3dBackward);
            cl_mem ind_mem = ind_c->gpu_data();
            cl_mem gout_mem = gout_c->gpu_data();
            int ind_off = ind_c->offset;
            int gout_off = gout_c->offset;
            int gin_off = grad_input->offset;

            int total_threads = N * C * D_out * H_out * W_out;
            CLBackend::get().launch(kernel, {static_cast<size_t>(total_threads)}, {},
                                    {&ind_mem, &ind_off, &gout_mem, &gout_off, &gin_mem, &gin_off,
                                     &N, &C, &D_in, &H_in, &W_in, &D_out, &H_out, &W_out},
                                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int),
                                     sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
        } else {
            float* ind_ptr = ind_c->data_ptr();
            float* gout_ptr = gout_c->data_ptr();
            float* gin_ptr = grad_input->data_ptr();
            std::fill(gin_ptr, gin_ptr + grad_input->numel(), 0.0f);
            ThreadPool::get().parallel_for(0, N * C, [&](int64_t thread_idx) {
                int b = thread_idx / C;
                int c = thread_idx % C;
                for (int do_ = 0; do_ < D_out; ++do_) {
                    for (int ho = 0; ho < H_out; ++ho) {
                        for (int wo = 0; wo < W_out; ++wo) {
                            int out_idx = (((b * C + c) * D_out + do_) * H_out + ho) * W_out + wo;
                            int max_idx = (int)ind_ptr[out_idx];
                            if (max_idx >= 0) {
                                gin_ptr[max_idx] += gout_ptr[out_idx];
                            }
                        }
                    }
                }
            });
        }
        return { grad_input };
    }
};

class AdaptiveAvgPool2dNode : public Node {
public:
    AdaptiveAvgPool2dNode() : Node("AdaptiveAvgPool2d") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto input = saved_tensors[0];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        StorageUseGuard guard({input_c->storage, gout_c->storage, grad_input->storage});

        int N = input_c->shape[0];
        int C = input_c->shape[1];
        int H = input_c->shape[2];
        int W = input_c->shape[3];
        int OH = gout_c->shape[2];
        int OW = gout_c->shape[3];

        if (input_c->device.type == DeviceType::GPU) {
            auto zero_kernel = CLBackend::get().get_kernel(KernelID::FillZero);
            int gin_size = grad_input->numel();
            cl_mem gin_mem = grad_input->gpu_data();
            CLBackend::get().launch(zero_kernel, {static_cast<size_t>(gin_size)}, {}, {&gin_mem, &gin_size}, {sizeof(cl_mem), sizeof(int)});

            auto kernel = CLBackend::get().get_kernel(KernelID::AdaptiveAvgPool2dBackward);
            cl_mem gout_mem = gout_c->gpu_data();
            int gout_off = gout_c->offset;
            int gin_off = grad_input->offset;

            int total_threads = N * C * H * W;
            CLBackend::get().launch(kernel, {static_cast<size_t>(total_threads)}, {},
                                    {&gout_mem, &gout_off, &gin_mem, &gin_off, &N, &C, &H, &W, &OH, &OW},
                                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
        } else {
            float* dy_ptr = gout_c->data_ptr();
            float* dx_ptr = grad_input->data_ptr();
            std::fill(dx_ptr, dx_ptr + grad_input->numel(), 0.0f);
            
            ThreadPool::get().parallel_for(0, N * C, [&](int64_t nc) {
                int64_t n = nc / C;
                int64_t c = nc % C;
                for (int oh = 0; oh < OH; ++oh) {
                    int64_t h_start = (oh * H) / OH;
                    int64_t h_end = ((oh + 1) * H + OH - 1) / OH;
                    h_end = std::min(h_end, (int64_t)H);
                    for (int ow = 0; ow < OW; ++ow) {
                        int64_t w_start = (ow * W) / OW;
                        int64_t w_end = ((ow + 1) * W + OW - 1) / OW;
                        w_end = std::min(w_end, (int64_t)W);
                        
                        int64_t count = (h_end - h_start) * (w_end - w_start);
                        float grad_val = dy_ptr[n * C * OH * OW + c * OH * OW + oh * OW + ow] / count;
                        for (int64_t h = h_start; h < h_end; ++h) {
                            for (int64_t w = w_start; w < w_end; ++w) {
                                dx_ptr[n * C * H * W + c * H * W + h * W + w] += grad_val;
                            }
                        }
                    }
                }
            });
        }
        
        return { grad_input };
    }
};

namespace Ops {

std::shared_ptr<Tensor> max_pool2d(std::shared_ptr<Tensor> input, int kernel_size, int stride, int padding) {
    if (stride == -1) stride = kernel_size;

    if (input->shape.size() != 4) {
        throw std::runtime_error("[litetorch Error] MaxPool2d requires 4D input");
    }

    auto input_c = input->is_contiguous() ? input : input->contiguous();
    int N = input_c->shape[0];
    int C = input_c->shape[1];
    int H_in = input_c->shape[2];
    int W_in = input_c->shape[3];
    int H_out = (H_in - kernel_size + 2 * padding) / stride + 1;
    int W_out = (W_in - kernel_size + 2 * padding) / stride + 1;
    auto out = Tensor::create({N, C, H_out, W_out}, input_c->device);
    auto save_indices = Tensor::create({N, C, H_out, W_out}, input_c->device);
    StorageUseGuard guard({input_c->storage, out->storage, save_indices->storage});

    if (input_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::MaxPool2dForward);
        cl_mem in_mem = input_c->gpu_data();
        cl_mem out_mem = out->gpu_data();
        cl_mem ind_mem = save_indices->gpu_data();
        int in_off = input_c->offset;
        int out_off = out->offset;
        int ind_off = save_indices->offset;
        int total_threads = N * C * H_out * W_out;
        CLBackend::get().launch(kernel, {static_cast<size_t>(total_threads)}, {},
                                {&in_mem, &in_off, &out_mem, &out_off, &ind_mem, &ind_off,
                                 &N, &C, &H_in, &W_in, &H_out, &W_out, &kernel_size, &stride, &padding},
                                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int),
                                 sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
    } else {
        float* in_ptr = input_c->data_ptr();
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, N * C, [&](int64_t thread_idx) {
            int b = thread_idx / C;
            int c = thread_idx % C;
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    float max_val = -1e37f;
                    int max_idx = -1;
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        int y = ho * stride - padding + ky;
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int x = wo * stride - padding + kx;
                            if (y >= 0 && y < H_in && x >= 0 && x < W_in) {
                                int input_idx = ((b * C + c) * H_in + y) * W_in + x;
                                if (in_ptr[input_idx] > max_val) {
                                    max_val = in_ptr[input_idx];
                                    max_idx = input_idx;
                                }
                            }
                        }
                    }
                    int out_idx = ((b * C + c) * H_out + ho) * W_out + wo;
                    out_ptr[out_idx] = max_val;
                    save_indices->data_ptr()[out_idx] = (float)max_idx;
                }
            }
        });
    }

    if (input->requires_grad) {
        auto node = std::make_shared<MaxPool2dNode>(kernel_size, stride, padding);
        node->inputs = { {input, true} };
        node->next_nodes = { input->creator };
        node->saved_tensors = { input, save_indices };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> max_pool3d(std::shared_ptr<Tensor> input, int kernel_size, int stride, int padding) {
    if (stride == -1) stride = kernel_size;

    if (input->shape.size() != 5) {
        throw std::runtime_error("[litetorch Error] MaxPool3d requires 5D input");
    }

    auto input_c = input->is_contiguous() ? input : input->contiguous();
    int N = input_c->shape[0];
    int C = input_c->shape[1];
    int D_in = input_c->shape[2];
    int H_in = input_c->shape[3];
    int W_in = input_c->shape[4];
    int D_out = (D_in - kernel_size + 2 * padding) / stride + 1;
    int H_out = (H_in - kernel_size + 2 * padding) / stride + 1;
    int W_out = (W_in - kernel_size + 2 * padding) / stride + 1;
    auto out = Tensor::create({N, C, D_out, H_out, W_out}, input_c->device);
    auto save_indices = Tensor::create({N, C, D_out, H_out, W_out}, input_c->device);
    StorageUseGuard guard({input_c->storage, out->storage, save_indices->storage});

    if (input_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::MaxPool3dForward);
        cl_mem in_mem = input_c->gpu_data();
        cl_mem out_mem = out->gpu_data();
        cl_mem ind_mem = save_indices->gpu_data();
        int in_off = input_c->offset;
        int out_off = out->offset;
        int ind_off = save_indices->offset;
        int total_threads = N * C * D_out * H_out * W_out;
        CLBackend::get().launch(kernel, {static_cast<size_t>(total_threads)}, {},
                                {&in_mem, &in_off, &out_mem, &out_off, &ind_mem, &ind_off,
                                 &N, &C, &D_in, &H_in, &W_in, &D_out, &H_out, &W_out, &kernel_size, &stride, &padding},
                                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int),
                                 sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
    } else {
        float* in_ptr = input_c->data_ptr();
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, N * C * D_out, [&](int64_t job) {
            int d_out = job % D_out;
            int c = (job / D_out) % C;
            int b = job / (D_out * C);

            float* out_slice = out_ptr + ((((b * C + c) * D_out + d_out) * H_out) * W_out);
            float* ind_slice = save_indices->data_ptr() + ((((b * C + c) * D_out + d_out) * H_out) * W_out);

            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    float max_val = -1e37f;
                    int max_idx = -1;
                    for (int kz = 0; kz < kernel_size; ++kz) {
                        int z = d_out * stride - padding + kz;
                        if (z >= 0 && z < D_in) {
                            for (int ky = 0; ky < kernel_size; ++ky) {
                                int y = ho * stride - padding + ky;
                                if (y >= 0 && y < H_in) {
                                    for (int kx = 0; kx < kernel_size; ++kx) {
                                        int x = wo * stride - padding + kx;
                                        if (x >= 0 && x < W_in) {
                                            int in_idx = (((b * C + c) * D_in + z) * H_in + y) * W_in + x;
                                            float val = in_ptr[in_idx];
                                            if (val > max_val) {
                                                max_val = val;
                                                max_idx = in_idx;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    out_slice[ho * W_out + wo] = max_val;
                    ind_slice[ho * W_out + wo] = (float)max_idx;
                }
            }
        });
    }

    if (input->requires_grad) {
        auto node = std::make_shared<MaxPool3dNode>(kernel_size, stride, padding);
        node->inputs = { {input, true} };
        node->next_nodes = { input->creator };
        node->saved_tensors = { input, save_indices };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> adaptive_avg_pool2d(std::shared_ptr<Tensor> input, int output_height, int output_width) {
    if (input->shape.size() != 4) {
        throw std::runtime_error("AdaptiveAvgPool2d requires 4D input (N, C, H, W)");
    }
    
    auto input_c = input->is_contiguous() ? input : input->contiguous();
    int N = input_c->shape[0];
    int C = input_c->shape[1];
    int H = input_c->shape[2];
    int W = input_c->shape[3];
    int OH = output_height;
    int OW = output_width;
    
    auto out = Tensor::create({N, C, OH, OW}, input_c->device);
    StorageUseGuard guard({input_c->storage, out->storage});
    
    if (input_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::AdaptiveAvgPool2dForward);
        cl_mem in_mem = input_c->gpu_data();
        cl_mem out_mem = out->gpu_data();
        int in_off = input_c->offset;
        int out_off = out->offset;
        int total_threads = N * C * OH * OW;
        CLBackend::get().launch(kernel, {static_cast<size_t>(total_threads)}, {},
                                {&in_mem, &in_off, &out_mem, &out_off, &N, &C, &H, &W, &OH, &OW},
                                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
    } else {
        float* in_ptr = input_c->data_ptr();
        float* out_ptr = out->data_ptr();
        
        ThreadPool::get().parallel_for(0, N * C, [&](int64_t nc) {
            int64_t n = nc / C;
            int64_t c = nc % C;
            for (int oh = 0; oh < OH; ++oh) {
                int64_t h_start = (oh * H) / OH;
                int64_t h_end = ((oh + 1) * H + OH - 1) / OH;
                h_end = std::min(h_end, (int64_t)H);
                for (int ow = 0; ow < OW; ++ow) {
                    int64_t w_start = (ow * W) / OW;
                    int64_t w_end = ((ow + 1) * W + OW - 1) / OW;
                    w_end = std::min(w_end, (int64_t)W);
                    
                    float sum = 0.0f;
                    int64_t count = (h_end - h_start) * (w_end - w_start);
                    for (int64_t h = h_start; h < h_end; ++h) {
                        for (int64_t w = w_start; w < w_end; ++w) {
                            sum += in_ptr[n * C * H * W + c * H * W + h * W + w];
                        }
                    }
                    out_ptr[n * C * OH * OW + c * OH * OW + oh * OW + ow] = sum / count;
                }
            }
        });
    }
    
    if (input->requires_grad) {
        auto node = std::make_shared<AdaptiveAvgPool2dNode>();
        node->inputs = { {input, true} };
        node->next_nodes = { input->creator };
        node->saved_tensors = { input };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

}
}
