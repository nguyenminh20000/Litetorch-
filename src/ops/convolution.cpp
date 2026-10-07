#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>
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

void im2col_cpu(const float* in_data, int C, int H, int W,
                int KH, int KW, int padding, int stride,
                int H_out, int W_out, float* col_data) {
    int channels_col = C * KH * KW;
    ThreadPool::get().parallel_for(0, channels_col, [&](int64_t c) {
        int w_offset = c % KW;
        int h_offset = (c / KW) % KH;
        int c_im = c / (KH * KW);
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                int im_row = h * stride - padding + h_offset;
                int im_col = w * stride - padding + w_offset;
                int col_idx = (c * H_out + h) * W_out + w;
                if (im_row >= 0 && im_row < H && im_col >= 0 && im_col < W) {
                    col_data[col_idx] = in_data[(c_im * H + im_row) * W + im_col];
                } else {
                    col_data[col_idx] = 0.0f;
                }
            }
        }
    });
}
}

class Conv2dNode : public Node {
public:
    Conv2dNode(int stride, int padding) : Node("Conv2d"), stride(stride), padding(padding) {}
    int stride;
    int padding;

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto input = saved_tensors[0];
        auto weight = saved_tensors[1];
        auto bias = saved_tensors.size() > 2 ? saved_tensors[2] : nullptr;

        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto weight_c = weight->is_contiguous() ? weight : weight->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();

        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        auto grad_weight = Tensor::create(weight_c->shape, weight_c->device);
        std::shared_ptr<Tensor> grad_bias = nullptr;
        if (bias) {
            grad_bias = Tensor::create(bias->shape, bias->device);
        }

        int N = input_c->shape[0];
        int C_in = input_c->shape[1];
        int H_in = input_c->shape[2];
        int W_in = input_c->shape[3];

        int C_out = weight_c->shape[0];
        int KH = weight_c->shape[2];
        int KW = weight_c->shape[3];

        int H_out = gout_c->shape[2];
        int W_out = gout_c->shape[3];

        StorageUseGuard guard({input_c->storage, weight_c->storage, bias ? bias->storage : nullptr,
                              gout_c->storage, grad_input->storage, grad_weight->storage, grad_bias ? grad_bias->storage : nullptr});

        if (input_c->device.type == DeviceType::GPU) {
            auto native = BackendDispatcher::get().get_backend();
            bool use_gemm = native && native->is_available();
            auto im2col_k = CLBackend::get().get_kernel(KernelID::Im2colFlat);
            auto transp_inv_k = CLBackend::get().get_kernel(KernelID::TransposeConvOutInv);
            auto col2im_k = CLBackend::get().get_kernel(KernelID::Col2Im);
            if (use_gemm && im2col_k && transp_inv_k && col2im_k) {
                int64_t K = (int64_t)C_in * KH * KW;
                int64_t HW_out = (int64_t)H_out * W_out;
                int64_t NHW = (int64_t)N * HW_out;
                cl_mem in_mem = input_c->gpu_data();
                int in_off = input_c->offset;
                cl_mem w_mem = weight_c->gpu_data();
                int w_off = weight_c->offset;
                cl_mem gout_mem = gout_c->gpu_data();
                int gout_off = gout_c->offset;
                auto gout_2d = Tensor::create({C_out, NHW}, input_c->device);
                {
                    cl_mem g2_mem = gout_2d->gpu_data();
                    int g2_off = gout_2d->offset;
                    int n_v = N, co_v = C_out, ho_v = H_out, wo_v = W_out;
                    size_t total = (size_t)C_out * (size_t)NHW;
                    CLBackend::get().launch(transp_inv_k, {total}, {},
                        {&gout_mem, &gout_off, &g2_mem, &g2_off, &n_v, &co_v, &ho_v, &wo_v},
                        {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
                }
                auto col = Tensor::create({K, NHW}, input_c->device);
                {
                    cl_mem col_mem = col->gpu_data();
                    int col_off = col->offset;
                    int n_v = N, ci_v = C_in, hi_v = H_in, wi_v = W_in;
                    int kh_v = KH, kw_v = KW, pad_v = padding, st_v = stride;
                    int ho_v = H_out, wo_v = W_out;
                    size_t total = (size_t)K * (size_t)NHW;
                    CLBackend::get().launch(im2col_k, {total}, {},
                        {&in_mem, &in_off, &n_v, &ci_v, &hi_v, &wi_v, &kh_v, &kw_v, &pad_v, &st_v, &ho_v, &wo_v, &col_mem, &col_off},
                        {sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(cl_mem), sizeof(int)});
                }
                auto gw_2d = Tensor::create({C_out, K}, input_c->device);
                gw_2d->storage = grad_weight->storage;
                gw_2d->offset = grad_weight->offset;
                gw_2d->requires_grad = false;
                native->matmul_ex(gout_2d->gpu_data(), gout_2d->offset, false, NHW,
                                  col->gpu_data(), col->offset, true, NHW,
                                  gw_2d->gpu_data(), gw_2d->offset, C_out, K, NHW);
                auto dcol = Tensor::create({K, NHW}, input_c->device);
                auto w_2d = Tensor::create({C_out, K}, weight_c->device);
                w_2d->storage = weight_c->storage;
                w_2d->offset = weight_c->offset;
                w_2d->requires_grad = false;
                native->matmul_ex(w_2d->gpu_data(), w_2d->offset, true, K,
                                  gout_2d->gpu_data(), gout_2d->offset, false, NHW,
                                  dcol->gpu_data(), dcol->offset, K, NHW, C_out);
                {
                    std::vector<float> zeros(grad_input->numel(), 0.0f);
                    CLBackend::get().write(grad_input->gpu_data(), zeros.size() * sizeof(float), zeros.data(), grad_input->offset);
                }
                {
                    cl_mem dc_mem = dcol->gpu_data();
                    int dc_off = dcol->offset;
                    cl_mem gin_mem = grad_input->gpu_data();
                    int gin_off = grad_input->offset;
                    int n_v = N, ci_v = C_in, hi_v = H_in, wi_v = W_in;
                    int kh_v = KH, kw_v = KW, pad_v = padding, st_v = stride;
                    int ho_v = H_out, wo_v = W_out;
                    size_t total = (size_t)K * (size_t)NHW;
                    CLBackend::get().launch(col2im_k, {total}, {},
                        {&dc_mem, &dc_off, &gin_mem, &gin_off, &n_v, &ci_v, &hi_v, &wi_v, &kh_v, &kw_v, &pad_v, &st_v, &ho_v, &wo_v},
                        {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
                }
                if (bias) {
                    auto kernel_gb = CLBackend::get().get_kernel(KernelID::Conv2dBackwardBias);
                    if (kernel_gb) {
                        cl_mem gb_mem = grad_bias->gpu_data();
                        int gb_off = grad_bias->offset;
                        CLBackend::get().launch(kernel_gb, {static_cast<size_t>(C_out)}, {},
                            {&gout_mem, &gout_off, &gb_mem, &gb_off, &N, &C_out, &H_out, &W_out},
                            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
                    }
                }
            } else {
                cl_mem in_mem = input_c->gpu_data();
                int in_off = input_c->offset;
                cl_mem w_mem = weight_c->gpu_data();
                int w_off = weight_c->offset;
                cl_mem gout_mem = gout_c->gpu_data();
                int gout_off = gout_c->offset;
                cl_mem gin_mem = grad_input->gpu_data();
                int gin_off = grad_input->offset;
                cl_mem gw_mem = grad_weight->gpu_data();
                int gw_off = grad_weight->offset;
                if (bias) {
                    auto kernel_gb = CLBackend::get().get_kernel(KernelID::Conv2dBackwardBias);
                    if (!kernel_gb) {
                        throw std::runtime_error("[litetorch Error] Conv2dBackwardBias kernel not available");
                    }
                    cl_mem gb_mem = grad_bias->gpu_data();
                    int gb_off = grad_bias->offset;
                    CLBackend::get().launch(kernel_gb, {static_cast<size_t>(C_out)}, {},
                        {&gout_mem, &gout_off, &gb_mem, &gb_off, &N, &C_out, &H_out, &W_out},
                        {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
                }
                {
                    auto kernel_gw = CLBackend::get().get_kernel(KernelID::Conv2dBackwardWeight);
                    if (!kernel_gw) {
                        throw std::runtime_error("[litetorch Error] Conv2dBackwardWeight kernel not available");
                    }
                    int total_gw = C_out * C_in * KH * KW;
                    CLBackend::get().launch(kernel_gw, {static_cast<size_t>(total_gw)}, {},
                        {&in_mem, &in_off, &gout_mem, &gout_off, &gw_mem, &gw_off, &N, &C_in, &H_in, &W_in, &C_out, &H_out, &W_out, &KH, &KW, &stride, &padding},
                        {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
                }
                {
                    auto kernel_gdx = CLBackend::get().get_kernel(KernelID::Conv2dBackwardInput);
                    if (!kernel_gdx) {
                        throw std::runtime_error("[litetorch Error] Conv2dBackwardInput kernel not available");
                    }
                    int total_gdx = N * C_in * H_in * W_in;
                    CLBackend::get().launch(kernel_gdx, {static_cast<size_t>(total_gdx)}, {},
                        {&gout_mem, &gout_off, &w_mem, &w_off, &gin_mem, &gin_off, &N, &C_in, &H_in, &W_in, &C_out, &H_out, &W_out, &KH, &KW, &stride, &padding},
                        {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
                }
            }
        } else {
            float* in_ptr = input_c->data_ptr();
            float* w_ptr = weight_c->data_ptr();
            float* gout_ptr = gout_c->data_ptr();
            float* gin_ptr = grad_input->data_ptr();
            float* gw_ptr = grad_weight->data_ptr();
            float* gb_ptr = grad_bias ? grad_bias->data_ptr() : nullptr;

            // Race-free: each thread writes exactly one output element (same
            // pattern as Conv3dNode::backward). The old fused loop had data
            // races on gb_ptr[co], gw_ptr[.] and gin_ptr[.] across threads.
            if (gb_ptr) {
                ThreadPool::get().parallel_for(0, C_out, [&](int64_t co) {
                    float sum = 0.0f;
                    for (int b = 0; b < N; ++b) {
                        for (int ho = 0; ho < H_out; ++ho) {
                            for (int wo = 0; wo < W_out; ++wo) {
                                sum += gout_ptr[((b * C_out + co) * H_out + ho) * W_out + wo];
                            }
                        }
                    }
                    gb_ptr[co] = sum;
                });
            }

            ThreadPool::get().parallel_for(0, (int64_t)C_out * C_in * KH * KW, [&](int64_t idx) {
                int kw = (int)(idx % KW);
                int kh = (int)((idx / KW) % KH);
                int ci = (int)((idx / (KW * KH)) % C_in);
                int co = (int)(idx / (KW * KH * C_in));

                float sum = 0.0f;
                for (int b = 0; b < N; ++b) {
                    for (int ho = 0; ho < H_out; ++ho) {
                        int y = ho * stride - padding + kh;
                        if (y >= 0 && y < H_in) {
                            for (int wo = 0; wo < W_out; ++wo) {
                                int x = wo * stride - padding + kw;
                                if (x >= 0 && x < W_in) {
                                    int gout_idx = ((b * C_out + co) * H_out + ho) * W_out + wo;
                                    int in_idx = ((b * C_in + ci) * H_in + y) * W_in + x;
                                    sum += gout_ptr[gout_idx] * in_ptr[in_idx];
                                }
                            }
                        }
                    }
                }
                gw_ptr[idx] = sum;
            });

            ThreadPool::get().parallel_for(0, (int64_t)N * C_in * H_in * W_in, [&](int64_t idx) {
                int x = (int)(idx % W_in);
                int y = (int)((idx / W_in) % H_in);
                int ci = (int)((idx / (W_in * H_in)) % C_in);
                int b = (int)(idx / (W_in * H_in * C_in));

                float sum = 0.0f;
                for (int co = 0; co < C_out; ++co) {
                    for (int kh = 0; kh < KH; ++kh) {
                        int ho_temp = y + padding - kh;
                        if (ho_temp % stride == 0) {
                            int ho = ho_temp / stride;
                            if (ho >= 0 && ho < H_out) {
                                for (int kw = 0; kw < KW; ++kw) {
                                    int wo_temp = x + padding - kw;
                                    if (wo_temp % stride == 0) {
                                        int wo = wo_temp / stride;
                                        if (wo >= 0 && wo < W_out) {
                                            int gout_idx = ((b * C_out + co) * H_out + ho) * W_out + wo;
                                            int w_idx = ((co * C_in + ci) * KH + kh) * KW + kw;
                                            sum += gout_ptr[gout_idx] * w_ptr[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                gin_ptr[idx] = sum;
            });
        }

        if (grad_bias) {
            return { grad_input, grad_weight, grad_bias };
        }
        return { grad_input, grad_weight };
    }
};

class Conv3dNode : public Node {
public:
    Conv3dNode(int stride, int padding) : Node("Conv3d"), stride(stride), padding(padding) {}
    int stride;
    int padding;

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto input = saved_tensors[0];
        auto weight = saved_tensors[1];
        auto bias = saved_tensors.size() > 2 ? saved_tensors[2] : nullptr;

        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto weight_c = weight->is_contiguous() ? weight : weight->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();

        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        auto grad_weight = Tensor::create(weight_c->shape, weight_c->device);
        std::shared_ptr<Tensor> grad_bias = nullptr;
        if (bias) {
            grad_bias = Tensor::create(bias->shape, bias->device);
        }

        int N = input_c->shape[0];
        int C_in = input_c->shape[1];
        int D_in = input_c->shape[2];
        int H_in = input_c->shape[3];
        int W_in = input_c->shape[4];

        int C_out = weight_c->shape[0];
        int KD = weight_c->shape[2];
        int KH = weight_c->shape[3];
        int KW = weight_c->shape[4];

        int D_out = gout_c->shape[2];
        int H_out = gout_c->shape[3];
        int W_out = gout_c->shape[4];

        StorageUseGuard guard({input_c->storage, weight_c->storage, bias ? bias->storage : nullptr,
                               gout_c->storage, grad_input->storage, grad_weight->storage, grad_bias ? grad_bias->storage : nullptr});

        if (input_c->device.type == DeviceType::GPU) {
            cl_mem in_mem = input_c->gpu_data();
            int in_off = input_c->offset;
            cl_mem w_mem = weight_c->gpu_data();
            int w_off = weight_c->offset;
            cl_mem gout_mem = gout_c->gpu_data();
            int gout_off = gout_c->offset;
            cl_mem gin_mem = grad_input->gpu_data();
            int gin_off = grad_input->offset;
            cl_mem gw_mem = grad_weight->gpu_data();
            int gw_off = grad_weight->offset;

            if (bias) {
                auto kernel_gb = CLBackend::get().get_kernel(KernelID::Conv3dBackwardBias);
                if (!kernel_gb) {
                    throw std::runtime_error("[litetorch Error] Conv3dBackwardBias kernel not available");
                }
                cl_mem gb_mem = grad_bias->gpu_data();
                int gb_off = grad_bias->offset;
                CLBackend::get().launch(kernel_gb, {static_cast<size_t>(C_out)}, {},
                    {&gout_mem, &gout_off, &gb_mem, &gb_off, &N, &C_out, &D_out, &H_out, &W_out},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
            }

            {
                auto kernel_gw = CLBackend::get().get_kernel(KernelID::Conv3dBackwardWeight);
                if (!kernel_gw) {
                    throw std::runtime_error("[litetorch Error] Conv3dBackwardWeight kernel not available");
                }
                int total_gw = C_out * C_in * KD * KH * KW;
                CLBackend::get().launch(kernel_gw, {static_cast<size_t>(total_gw)}, {},
                    {&in_mem, &in_off, &gout_mem, &gout_off, &gw_mem, &gw_off, &N, &C_in, &D_in, &H_in, &W_in, &C_out, &D_out, &H_out, &W_out, &KD, &KH, &KW, &stride, &padding},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
            }

            {
                auto kernel_gdx = CLBackend::get().get_kernel(KernelID::Conv3dBackwardInput);
                if (!kernel_gdx) {
                    throw std::runtime_error("[litetorch Error] Conv3dBackwardInput kernel not available");
                }
                int total_gdx = N * C_in * D_in * H_in * W_in;
                CLBackend::get().launch(kernel_gdx, {static_cast<size_t>(total_gdx)}, {},
                    {&gout_mem, &gout_off, &w_mem, &w_off, &gin_mem, &gin_off, &N, &C_in, &D_in, &H_in, &W_in, &C_out, &D_out, &H_out, &W_out, &KD, &KH, &KW, &stride, &padding},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
            }
        } else {
            float* in_ptr = input_c->data_ptr();
            float* w_ptr = weight_c->data_ptr();
            float* gout_ptr = gout_c->data_ptr();
            float* gin_ptr = grad_input->data_ptr();
            float* gw_ptr = grad_weight->data_ptr();
            float* gb_ptr = grad_bias ? grad_bias->data_ptr() : nullptr;

            std::fill(gin_ptr, gin_ptr + grad_input->numel(), 0.0f);
            std::fill(gw_ptr, gw_ptr + grad_weight->numel(), 0.0f);
            if (gb_ptr) {
                std::fill(gb_ptr, gb_ptr + grad_bias->numel(), 0.0f);
            }

            if (gb_ptr) {
                ThreadPool::get().parallel_for(0, C_out, [&](int64_t co) {
                    float sum = 0.0f;
                    for (int b = 0; b < N; ++b) {
                        for (int do_ = 0; do_ < D_out; ++do_) {
                            for (int ho = 0; ho < H_out; ++ho) {
                                for (int wo = 0; wo < W_out; ++wo) {
                                    sum += gout_ptr[(((b * C_out + co) * D_out + do_) * H_out + ho) * W_out + wo];
                                }
                            }
                        }
                    }
                    gb_ptr[co] = sum;
                });
            }

            ThreadPool::get().parallel_for(0, C_out * C_in * KD * KH * KW, [&](int64_t idx) {
                int kw = idx % KW;
                int kh = (idx / KW) % KH;
                int kd = (idx / (KW * KH)) % KD;
                int ci = (idx / (KW * KH * KD)) % C_in;
                int co = idx / (KW * KH * KD * C_in);

                float sum = 0.0f;
                for (int b = 0; b < N; ++b) {
                    for (int do_ = 0; do_ < D_out; ++do_) {
                        int z = do_ * stride - padding + kd;
                        if (z >= 0 && z < D_in) {
                            for (int ho = 0; ho < H_out; ++ho) {
                                int y = ho * stride - padding + kh;
                                if (y >= 0 && y < H_in) {
                                    for (int wo = 0; wo < W_out; ++wo) {
                                        int x = wo * stride - padding + kw;
                                        if (x >= 0 && x < W_in) {
                                            int gout_idx = (((b * C_out + co) * D_out + do_) * H_out + ho) * W_out + wo;
                                            int in_idx = (((b * C_in + ci) * D_in + z) * H_in + y) * W_in + x;
                                            sum += gout_ptr[gout_idx] * in_ptr[in_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                gw_ptr[idx] = sum;
            });

            ThreadPool::get().parallel_for(0, N * C_in * D_in * H_in * W_in, [&](int64_t idx) {
                int x = idx % W_in;
                int y = (idx / W_in) % H_in;
                int z = (idx / (W_in * H_in)) % D_in;
                int ci = (idx / (W_in * H_in * D_in)) % C_in;
                int b = idx / (W_in * H_in * D_in * C_in);

                float sum = 0.0f;
                for (int co = 0; co < C_out; ++co) {
                    for (int kd = 0; kd < KD; ++kd) {
                        int do_temp = z + padding - kd;
                        if (do_temp % stride == 0) {
                            int do_ = do_temp / stride;
                            if (do_ >= 0 && do_ < D_out) {
                                for (int kh = 0; kh < KH; ++kh) {
                                    int ho_temp = y + padding - kh;
                                    if (ho_temp % stride == 0) {
                                        int ho = ho_temp / stride;
                                        if (ho >= 0 && ho < H_out) {
                                            for (int kw = 0; kw < KW; ++kw) {
                                                int wo_temp = x + padding - kw;
                                                if (wo_temp % stride == 0) {
                                                    int wo = wo_temp / stride;
                                                    if (wo >= 0 && wo < W_out) {
                                                        int gout_idx = (((b * C_out + co) * D_out + do_) * H_out + ho) * W_out + wo;
                                                        int w_idx = ((((co * C_in + ci) * KD + kd) * KH + kh) * KW + kw);
                                                        sum += gout_ptr[gout_idx] * w_ptr[w_idx];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                gin_ptr[idx] = sum;
            });
        }

        if (grad_bias) {
            return { grad_input, grad_weight, grad_bias };
        }
        return { grad_input, grad_weight };
    }
};

namespace Ops {

std::shared_ptr<Tensor> conv2d(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias, int stride, int padding) {
    if (input->device != weight->device) {
        if (input->device.type == DeviceType::GPU) weight = weight->to(input->device);
        else input = input->to(weight->device);
    }
    if (bias && bias->device != input->device) {
        bias = bias->to(input->device);
    }

    if (input->shape.size() != 4 || weight->shape.size() != 4) {
        throw std::runtime_error("[litetorch Error] Conv2d requires 4D input and weight");
    }
    if (input->dtype != DataType::FP32 || weight->dtype != DataType::FP32 ||
        (bias && bias->dtype != DataType::FP32)) {
        throw std::runtime_error("[litetorch Error] Conv2d requires FP32 input, weight and bias");
    }

    int N = input->shape[0];
    int C_in = input->shape[1];
    int H_in = input->shape[2];
    int W_in = input->shape[3];
    int C_out = weight->shape[0];
    int KH = weight->shape[2];
    int KW = weight->shape[3];
    int H_out = (H_in - KH + 2 * padding) / stride + 1;
    int W_out = (W_in - KW + 2 * padding) / stride + 1;
    auto out = Tensor::create({N, C_out, H_out, W_out}, input->device);

    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto weight_c = weight->is_contiguous() ? weight : weight->contiguous();
    auto bias_c = (bias && bias->is_contiguous()) ? bias : (bias ? bias->contiguous() : nullptr);

    StorageUseGuard guard({input_c->storage, weight_c->storage, bias_c ? bias_c->storage : nullptr, out->storage});

    if (input->device.type == DeviceType::GPU) {
        bool done = false;
        auto native = BackendDispatcher::get().get_backend();
        typedef void (*Conv2dCudnnFn)(
            const float*, int, const float*, int, const float*, int, int,
            float*, int, int, int, int, int, int, int, int, int, int, int, int);
        Conv2dCudnnFn cudnn_conv2d = nullptr;
        if (native && native->is_available() && !std::getenv("LITETORCH_NO_CUDNN")) {
            cudnn_conv2d = reinterpret_cast<Conv2dCudnnFn>(
                native->get_kernel("", "", "conv2d_cudnn"));
        }
        if (cudnn_conv2d) {
            cudnn_conv2d(
                static_cast<const float*>(input_c->gpu_data()), input_c->offset,
                static_cast<const float*>(weight_c->gpu_data()), weight_c->offset,
                bias_c ? static_cast<const float*>(bias_c->gpu_data()) : nullptr,
                bias_c ? bias_c->offset : 0, bias_c ? 1 : 0,
                static_cast<float*>(out->gpu_data()), out->offset,
                N, C_in, H_in, W_in, C_out, H_out, W_out, KH, KW, stride, padding);
            done = true;
        }
        auto im2col_k = CLBackend::get().get_kernel(KernelID::Im2colFlat);
        void* epilogue_k = nullptr;
        if (native && native->is_available()) {
            epilogue_k = native->get_kernel("", "", "transpose_conv_out_bias_relu_kernel");
        }
        if (!done && native && native->is_available() && im2col_k && epilogue_k) {
            int64_t K = (int64_t)C_in * KH * KW;
            int64_t HW_out = (int64_t)H_out * W_out;
            int64_t NHW = (int64_t)N * HW_out;
            auto col = Tensor::create({K, NHW}, input_c->device);
            auto tmp = Tensor::create({C_out, NHW}, input_c->device);
            {
                cl_mem in_mem = input_c->gpu_data();
                int in_off = input_c->offset;
                int n_v = N, ci_v = C_in, hi_v = H_in, wi_v = W_in;
                int kh_v = KH, kw_v = KW, pad_v = padding, st_v = stride;
                int ho_v = H_out, wo_v = W_out;
                cl_mem col_mem = col->gpu_data();
                int col_off = col->offset;
                size_t total = (size_t)K * (size_t)NHW;
                CLBackend::get().launch(im2col_k, {total}, {},
                    {&in_mem, &in_off, &n_v, &ci_v, &hi_v, &wi_v, &kh_v, &kw_v, &pad_v, &st_v, &ho_v, &wo_v, &col_mem, &col_off},
                    {sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(cl_mem), sizeof(int)});
            }
            auto weight_2d = Tensor::create({C_out, K}, weight_c->device);
            weight_2d->storage = weight_c->storage;
            weight_2d->offset = weight_c->offset;
            weight_2d->requires_grad = false;
            native->matmul(weight_2d->gpu_data(), weight_2d->offset, col->gpu_data(), col->offset,
                           tmp->gpu_data(), tmp->offset, C_out, NHW, K);
            {
                cl_mem tmp_mem = tmp->gpu_data();
                int tmp_off = tmp->offset;
                cl_mem b_mem = bias_c ? bias_c->gpu_data() : nullptr;
                int b_off = bias_c ? bias_c->offset : 0;
                int has_bias = bias_c ? 1 : 0;
                int apply_relu = 0;
                cl_mem out_mem = out->gpu_data();
                int out_off = out->offset;
                int n_v = N, co_v = C_out, ho_v = H_out, wo_v = W_out;
                size_t total = (size_t)N * (size_t)C_out * (size_t)HW_out;
                CLBackend::get().launch(epilogue_k, {total}, {},
                    {&tmp_mem, &tmp_off, &b_mem, &b_off, &has_bias, &out_mem, &out_off, &apply_relu, &n_v, &co_v, &ho_v, &wo_v},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
            }
            done = true;
        }
        if (!done) {
            auto kernel_conv = CLBackend::get().get_kernel(KernelID::Conv2dForward);
            if (!kernel_conv) {
                throw std::runtime_error("[litetorch Error] Conv2dForward kernel not available");
            }
            {
                cl_mem in_mem = input_c->gpu_data();
                int in_off = input_c->offset;
                cl_mem w_mem = weight_c->gpu_data();
                int w_off = weight_c->offset;
                cl_mem b_mem = bias_c ? bias_c->gpu_data() : nullptr;
                int b_off = bias_c ? bias_c->offset : 0;
                int has_bias = bias_c ? 1 : 0;
                cl_mem out_mem = out->gpu_data();
                int out_off = out->offset;
                int total_threads = N * C_out * H_out * W_out;

                CLBackend::get().launch(kernel_conv, {static_cast<size_t>(total_threads)}, {},
                    {&in_mem, &in_off, &w_mem, &w_off, &b_mem, &b_off, &has_bias, &out_mem, &out_off,
                     &N, &C_in, &H_in, &W_in, &C_out, &H_out, &W_out, &KH, &KW, &stride, &padding},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(cl_mem), sizeof(int),
                     sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
            }
        }
    } else {
        const float* in_ptr = input_c->data_ptr();
        float* out_ptr = out->data_ptr();

        auto weight_flat = Tensor::create({C_out, C_in * KH * KW}, weight_c->device);
        weight_flat->storage = weight_c->storage;
        weight_flat->offset = weight_c->offset;
        weight_flat->requires_grad = false;

        auto col = Tensor::create({C_in * KH * KW, H_out * W_out}, input_c->device);
        col->requires_grad = false;
        float* col_ptr = col->data_ptr();

        for (int n = 0; n < N; ++n) {
            im2col_cpu(in_ptr + n * C_in * H_in * W_in, C_in, H_in, W_in, KH, KW, padding, stride, H_out, W_out, col_ptr);
            auto out_n_flat = matmul(weight_flat, col);
            std::memcpy(out_ptr + n * C_out * H_out * W_out, out_n_flat->data_ptr(), C_out * H_out * W_out * sizeof(float));
        }

        if (bias_c) {
            const float* b_ptr = bias_c->data_ptr();
            ThreadPool::get().parallel_for(0, N * C_out, [&](int64_t thread_idx) {
                int co = thread_idx % C_out;
                float b_val = b_ptr[co];
                float* slice = out_ptr + thread_idx * H_out * W_out;
                for (int i = 0; i < H_out * W_out; ++i) {
                    slice[i] += b_val;
                }
            });
        }
    }

    if (input->requires_grad || weight->requires_grad || (bias && bias->requires_grad)) {
        auto node = std::make_shared<Conv2dNode>(stride, padding);
        node->inputs = { {input, input->requires_grad}, {weight, weight->requires_grad} };
        node->next_nodes = { input->creator, weight->creator };
        node->saved_tensors = { input, weight };
        if (bias) {
            node->inputs.push_back({bias, bias->requires_grad});
            node->next_nodes.push_back(bias->creator);
            node->saved_tensors.push_back(bias);
        }
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> conv3d(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias, int stride, int padding) {
    if (input->device != weight->device) {
        if (input->device.type == DeviceType::GPU) weight = weight->to(input->device);
        else input = input->to(weight->device);
    }
    if (bias && bias->device != input->device) {
        bias = bias->to(input->device);
    }

    if (input->shape.size() != 5 || weight->shape.size() != 5) {
        throw std::runtime_error("[litetorch Error] Conv3d requires 5D input and weight");
    }
    if (input->dtype != DataType::FP32 || weight->dtype != DataType::FP32 ||
        (bias && bias->dtype != DataType::FP32)) {
        throw std::runtime_error("[litetorch Error] Conv3d requires FP32 input, weight and bias");
    }

    int N = input->shape[0];
    int C_in = input->shape[1];
    int D_in = input->shape[2];
    int H_in = input->shape[3];
    int W_in = input->shape[4];
    int C_out = weight->shape[0];
    int KD = weight->shape[2];
    int KH = weight->shape[3];
    int KW = weight->shape[4];
    int D_out = (D_in - KD + 2 * padding) / stride + 1;
    int H_out = (H_in - KH + 2 * padding) / stride + 1;
    int W_out = (W_in - KW + 2 * padding) / stride + 1;
    auto out = Tensor::create({N, C_out, D_out, H_out, W_out}, input->device);
    StorageUseGuard guard({input->storage, weight->storage, bias ? bias->storage : nullptr, out->storage});

    if (input->device.type == DeviceType::GPU) {
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto weight_c = weight->is_contiguous() ? weight : weight->contiguous();
        auto bias_c = (bias && bias->is_contiguous()) ? bias : (bias ? bias->contiguous() : nullptr);
        auto kernel = CLBackend::get().get_kernel(KernelID::Conv3dForward);
        if (!kernel) {
            throw std::runtime_error("[litetorch Error] Conv3dForward kernel not available");
        }
        cl_mem in_mem = input_c->gpu_data();
        cl_mem w_mem = weight_c->gpu_data();
        cl_mem b_mem = bias_c ? bias_c->gpu_data() : nullptr;
        cl_mem out_mem = out->gpu_data();
        int in_off = input_c->offset;
        int w_off = weight_c->offset;
        int b_off = bias_c ? bias_c->offset : 0;
        int out_off = out->offset;
        int has_bias = bias_c ? 1 : 0;
        int total_threads = N * C_out * D_out * H_out * W_out;
        CLBackend::get().launch(kernel, {static_cast<size_t>(total_threads)}, {},
                                {&in_mem, &in_off, &w_mem, &w_off, &b_mem, &b_off, &has_bias, &out_mem, &out_off,
                                 &N, &C_in, &D_in, &H_in, &W_in, &C_out, &D_out, &H_out, &W_out, &KD, &KH, &KW, &stride, &padding},
                                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(cl_mem), sizeof(int),
                                 sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
    } else {
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto weight_c = weight->is_contiguous() ? weight : weight->contiguous();
        auto bias_c = (bias && bias->is_contiguous()) ? bias : (bias ? bias->contiguous() : nullptr);
        float* in_ptr = input_c->data_ptr();
        float* w_ptr = weight_c->data_ptr();
        float* b_ptr = bias_c ? bias_c->data_ptr() : nullptr;
        float* out_ptr = out->data_ptr();
        ThreadPool::get().parallel_for(0, N * C_out * D_out, [&](int64_t job) {
            int d_out = job % D_out;
            int co = (job / D_out) % C_out;
            int b = job / (D_out * C_out);

            float* out_slice = out_ptr + ((((b * C_out + co) * D_out + d_out) * H_out) * W_out);

            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    float val = 0.0f;
                    for (int ci = 0; ci < C_in; ++ci) {
                        for (int kz = 0; kz < KD; ++kz) {
                            int z = d_out * stride - padding + kz;
                            if (z >= 0 && z < D_in) {
                                for (int ky = 0; ky < KH; ++ky) {
                                    int y = ho * stride - padding + ky;
                                    if (y >= 0 && y < H_in) {
                                        for (int kx = 0; kx < KW; ++kx) {
                                            int x = wo * stride - padding + kx;
                                            if (x >= 0 && x < W_in) {
                                                int input_idx = (((b * C_in + ci) * D_in + z) * H_in + y) * W_in + x;
                                                int weight_idx = (((co * C_in + ci) * KD + kz) * KH + ky) * KW + kx;
                                                val += in_ptr[input_idx] * w_ptr[weight_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (b_ptr) {
                        val += b_ptr[co];
                    }
                    out_slice[ho * W_out + wo] = val;
                }
            }
        });
    }

    if (input->requires_grad || weight->requires_grad || (bias && bias->requires_grad)) {
        auto node = std::make_shared<Conv3dNode>(stride, padding);
        node->inputs = { {input, input->requires_grad}, {weight, weight->requires_grad} };
        node->next_nodes = { input->creator, weight->creator };
        node->saved_tensors = { input, weight };
        if (bias) {
            node->inputs.push_back({bias, bias->requires_grad});
            node->next_nodes.push_back(bias->creator);
            node->saved_tensors.push_back(bias);
        }
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

}
}
