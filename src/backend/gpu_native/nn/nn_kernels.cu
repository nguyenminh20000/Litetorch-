#include "gpu_common.h"

#ifdef USE_CUDNN
extern "C" void gpu_conv2d_cudnn(
    const float* input, int in_off,
    const float* weight, int w_off,
    const float* bias, int b_off, int has_bias,
    float* output, int out_off,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kh, int kw, int stride, int padding) {
    cudnnHandle_t handle = get_cudnn_handle();
    cudnnTensorDescriptor_t xDesc, yDesc, bDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;

    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&bDesc);
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, H_in, W_in);
    cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out, C_in, kh, kw);
    cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, H_out, W_out);

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(handle, &alpha, xDesc, input + in_off, wDesc, weight + w_off, convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, yDesc, output + out_off);

    if (has_bias && bias) {
        cudnnSetTensor4dDescriptor(bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1);
        cudnnAddTensor(handle, &alpha, bDesc, bias + b_off, &alpha, yDesc, output + out_off);
    }

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(bDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
}

extern "C" void gpu_softmax_cudnn(const float* input, int in_off, float* output, int out_off, int N, int C, int H, int W) {
    cudnnHandle_t handle = get_cudnn_handle();
    cudnnTensorDescriptor_t srcDesc, dstDesc;
    cudnnCreateTensorDescriptor(&srcDesc);
    cudnnCreateTensorDescriptor(&dstDesc);
    cudnnSetTensor4dDescriptor(srcDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnSetTensor4dDescriptor(dstDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    float alpha = 1.0f, beta = 0.0f;
    cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, srcDesc, input + in_off, &beta, dstDesc, output + out_off);
    cudnnDestroyTensorDescriptor(srcDesc);
    cudnnDestroyTensorDescriptor(dstDesc);
}
#endif

#ifdef USE_MIOPEN
extern "C" void gpu_conv2d_miopen(
    const float* input, int in_off,
    const float* weight, int w_off,
    const float* bias, int b_off, int has_bias,
    float* output, int out_off,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kh, int kw, int stride, int padding) {
    miopenHandle_t handle = get_miopen_handle();
    miopenTensorDescriptor_t xDesc, yDesc, bDesc;
    miopenTensorDescriptor_t wDesc;
    miopenConvolutionDescriptor_t convDesc;

    miopenCreateTensorDescriptor(&xDesc);
    miopenCreateTensorDescriptor(&yDesc);
    miopenCreateTensorDescriptor(&bDesc);
    miopenCreateTensorDescriptor(&wDesc);
    miopenCreateConvolutionDescriptor(&convDesc);

    miopenSet4dTensorDescriptor(xDesc, miopenFloat, N, C_in, H_in, W_in);
    miopenSet4dTensorDescriptor(wDesc, miopenFloat, C_out, C_in, kh, kw);
    miopenInitConvolutionDescriptor(convDesc, miopenConvolution, padding, padding, stride, stride, 1, 1);
    miopenSet4dTensorDescriptor(yDesc, miopenFloat, N, C_out, H_out, W_out);

    float alpha = 1.0f, beta = 0.0f;
    miopenConvolutionForward(handle, &alpha, xDesc, input + in_off, wDesc, weight + w_off, convDesc, miopenConvolutionFwdAlgoGEMM, &beta, yDesc, output + out_off, nullptr, 0);

    if (has_bias && bias) {
        miopenSet4dTensorDescriptor(bDesc, miopenFloat, 1, C_out, 1, 1);
        miopenOpTensor(handle, miopenTensorOpAdd, &alpha, yDesc, output + out_off, &alpha, bDesc, bias + b_off, &beta, yDesc, output + out_off);
    }

    miopenDestroyTensorDescriptor(xDesc);
    miopenDestroyTensorDescriptor(yDesc);
    miopenDestroyTensorDescriptor(bDesc);
    miopenDestroyTensorDescriptor(wDesc);
    miopenDestroyConvolutionDescriptor(convDesc);
}

extern "C" void gpu_softmax_miopen(const float* input, int in_off, float* output, int out_off, int N, int C, int H, int W) {
    miopenHandle_t handle = get_miopen_handle();
    miopenTensorDescriptor_t srcDesc, dstDesc;
    miopenCreateTensorDescriptor(&srcDesc);
    miopenCreateTensorDescriptor(&dstDesc);
    miopenSet4dTensorDescriptor(srcDesc, miopenFloat, N, C, H, W);
    miopenSet4dTensorDescriptor(dstDesc, miopenFloat, N, C, H, W);
    float alpha = 1.0f, beta = 0.0f;
    miopenSoftmaxForward_V2(handle, &alpha, srcDesc, input + in_off, &beta, dstDesc, output + out_off, MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_CHANNEL);
    miopenDestroyTensorDescriptor(srcDesc);
    miopenDestroyTensorDescriptor(dstDesc);
}
#endif

extern "C" __global__ void conv2d_kernel(const float* input, int in_off,
                            const float* weight, int w_off,
                            const float* bias, int b_off, int has_bias,
                            float* output, int out_off,
                            int batch_size, int in_channels, int in_h, int in_w,
                            int out_channels, int out_h, int out_w,
                            int kh, int kw, int stride, int padding) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    int total_threads = batch_size * out_channels * out_h * out_w;
    if (idx >= total_threads) return;

    int w_out = idx % out_w;
    int h_out = (idx / out_w) % out_h;
    int c_out = (idx / (out_w * out_h)) % out_channels;
    int b = idx / (out_w * out_h * out_channels);

    float val = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int ky = 0; ky < kh; ++ky) {
            int y = h_out * stride - padding + ky;
            for (int kx = 0; kx < kw; ++kx) {
                int x = w_out * stride - padding + kx;
                if (y >= 0 && y < in_h && x >= 0 && x < in_w) {
                    int input_idx = in_off + ((b * in_channels + c_in) * in_h + y) * in_w + x;
                    int weight_idx = w_off + ((c_out * in_channels + c_in) * kh + ky) * kw + kx;
                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    if (has_bias) {
        val += bias[b_off + c_out];
    }
    output[out_off + idx] = val;
}

extern "C" __global__ void conv2d_backward_gb(
    const float* grad_output, int gout_off,
    float* grad_bias, int gb_off,
    int N, int C_out, int H_out, int W_out)
{
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    if (co >= C_out) return;

    float sum_val = 0.0f;
    for (int b = 0; b < N; ++b) {
        for (int ho = 0; ho < H_out; ++ho) {
            for (int wo = 0; wo < W_out; ++wo) {
                sum_val += grad_output[gout_off + ((b * C_out + co) * H_out + ho) * W_out + wo];
            }
        }
    }
    grad_bias[gb_off + co] = sum_val;
}

extern "C" __global__ void conv2d_backward_gw(
    const float* input, int in_off,
    const float* grad_output, int gout_off,
    float* grad_weight, int gw_off,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int KH, int KW, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * KH * KW;
    if (idx >= total) return;

    int kw = idx % KW;
    int kh = (idx / KW) % KH;
    int ci = (idx / (KW * KH)) % C_in;
    int co = idx / (KW * KH * C_in);

    float sum_val = 0.0f;
    for (int b = 0; b < N; ++b) {
        for (int ho = 0; ho < H_out; ++ho) {
            int y = ho * stride - padding + kh;
            if (y >= 0 && y < H_in) {
                for (int wo = 0; wo < W_out; ++wo) {
                    int x = wo * stride - padding + kw;
                    if (x >= 0 && x < W_in) {
                        int gout_idx = ((b * C_out + co) * H_out + ho) * W_out + wo;
                        int in_idx = ((b * C_in + ci) * H_in + y) * W_in + x;
                        sum_val += grad_output[gout_off + gout_idx] * input[in_off + in_idx];
                    }
                }
            }
        }
    }
    grad_weight[gw_off + idx] = sum_val;
}

extern "C" __global__ void conv2d_backward_gdx(
    const float* grad_output, int gout_off,
    const float* weight, int w_off,
    float* grad_input, int gin_off,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int KH, int KW, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * H_in * W_in;
    if (idx >= total) return;

    int x = idx % W_in;
    int y = (idx / W_in) % H_in;
    int ci = (idx / (W_in * H_in)) % C_in;
    int b = idx / (W_in * H_in * C_in);

    float sum_val = 0.0f;
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
                                sum_val += grad_output[gout_off + gout_idx] * weight[w_off + w_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    grad_input[gin_off + idx] = sum_val;
}

extern "C" __global__ void im2col_kernel(
    const float* im, int im_off,
    int C, int H, int W,
    int KH, int KW, int padding, int stride,
    int H_out, int W_out,
    float* col, int col_off)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * KH * KW * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int kw = (idx / (W_out * H_out)) % KW;
    int kh = (idx / (W_out * H_out * KW)) % KH;
    int c_im = idx / (W_out * H_out * KW * KH);

    int im_row = h_out * stride - padding + kh;
    int im_col = w_out * stride - padding + kw;

    int c_col = c_im * KH * KW + kh * KW + kw;
    int col_idx = (c_col * H_out + h_out) * W_out + w_out;

    if (im_row >= 0 && im_row < H && im_col >= 0 && im_col < W) {
        col[col_off + col_idx] = im[im_off + (c_im * H + im_row) * W + im_col];
    } else {
        col[col_off + col_idx] = 0.0f;
    }
}

extern "C" __global__ void add_bias_2d(
    float* out, int out_off,
    const float* bias, int b_off,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx < total) {
        int c = (idx / (H * W)) % C;
        out[out_off + idx] += bias[b_off + c];
    }
}

extern "C" __global__ void conv3d_kernel(
    const float* input, int in_off,
    const float* weight, int w_off,
    const float* bias, int b_off, int has_bias,
    float* output, int out_off,
    int batch_size, int in_channels, int in_d, int in_h, int in_w,
    int out_channels, int out_d, int out_h, int out_w,
    int kd, int kh, int kw, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * out_d * out_h * out_w;
    if (idx >= total_threads) return;

    int w_out = idx % out_w;
    int h_out = (idx / out_w) % out_h;
    int d_out = (idx / (out_w * out_h)) % out_d;
    int c_out = (idx / (out_w * out_h * out_d)) % out_channels;
    int b = idx / (out_w * out_h * out_d * out_channels);

    float val = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kz = 0; kz < kd; ++kz) {
            int z = d_out * stride - padding + kz;
            if (z >= 0 && z < in_d) {
                for (int ky = 0; ky < kh; ++ky) {
                    int y = h_out * stride - padding + ky;
                    if (y >= 0 && y < in_h) {
                        for (int kx = 0; kx < kw; ++kx) {
                            int x = w_out * stride - padding + kx;
                            if (x >= 0 && x < in_w) {
                                int input_idx = in_off + ((((b * in_channels + c_in) * in_d + z) * in_h + y) * in_w + x);
                                int weight_idx = w_off + ((((c_out * in_channels + c_in) * kd + kz) * kh + ky) * kw + kx);
                                val += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    if (has_bias && bias) {
        val += bias[b_off + c_out];
    }
    output[out_off + idx] = val;
}

extern "C" __global__ void conv3d_backward_gb(
    const float* grad_output, int gout_off,
    float* grad_bias, int gb_off,
    int N, int C_out, int D_out, int H_out, int W_out)
{
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    if (co >= C_out) return;

    float sum_val = 0.0f;
    for (int b = 0; b < N; ++b) {
        for (int do_ = 0; do_ < D_out; ++do_) {
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    int gout_idx = (((b * C_out + co) * D_out + do_) * H_out + ho) * W_out + wo;
                    sum_val += grad_output[gout_off + gout_idx];
                }
            }
        }
    }
    grad_bias[gb_off + co] = sum_val;
}

extern "C" __global__ void conv3d_backward_gw(
    const float* input, int in_off,
    const float* grad_output, int gout_off,
    float* grad_weight, int gw_off,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int KD, int KH, int KW, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * KD * KH * KW;
    if (idx >= total) return;

    int kw = idx % KW;
    int kh = (idx / KW) % KH;
    int kd = (idx / (KW * KH)) % KD;
    int ci = (idx / (KW * KH * KD)) % C_in;
    int co = idx / (KW * KH * KD * C_in);

    float sum_val = 0.0f;
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
                                sum_val += grad_output[gout_off + gout_idx] * input[in_off + in_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    grad_weight[gw_off + idx] = sum_val;
}

extern "C" __global__ void conv3d_backward_gdx(
    const float* grad_output, int gout_off,
    const float* weight, int w_off,
    float* grad_input, int gin_off,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int KD, int KH, int KW, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * D_in * H_in * W_in;
    if (idx >= total) return;

    int x = idx % W_in;
    int y = (idx / W_in) % H_in;
    int z = (idx / (W_in * H_in)) % D_in;
    int ci = (idx / (W_in * H_in * D_in)) % C_in;
    int b = idx / (W_in * H_in * D_in * C_in);

    float sum_val = 0.0f;
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
                                            sum_val += grad_output[gout_off + gout_idx] * weight[w_off + w_idx];
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
    grad_input[gin_off + idx] = sum_val;
}

extern "C" __global__ void maxpool2d_kernel(const float* input, int in_off,
                               float* output, int out_off,
                               float* indices, int ind_off,
                               int batch_size, int channels, int in_h, int in_w,
                               int out_h, int out_w, int kernel_size, int stride, int padding) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    int total_threads = batch_size * channels * out_h * out_w;
    if (idx >= total_threads) return;

    int w_out = idx % out_w;
    int h_out = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % channels;
    int b = idx / (out_w * out_h * channels);

    float max_val = -1e37f;
    int max_idx = -1;
    for (int ky = 0; ky < kernel_size; ++ky) {
        int y = h_out * stride - padding + ky;
        for (int kx = 0; kx < kernel_size; ++kx) {
            int x = w_out * stride - padding + kx;
            if (y >= 0 && y < in_h && x >= 0 && x < in_w) {
                int input_idx = ((b * channels + c) * in_h + y) * in_w + x;
                float val = input[in_off + input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = input_idx;
                }
            }
        }
    }
    output[out_off + idx] = max_val;
    indices[ind_off + idx] = (float)max_idx;
}

extern "C" __global__ void maxpool2d_backward_kernel(
    const float* indices, int ind_off,
    const float* grad_output, int gout_off,
    float* grad_input, int gin_off,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * channels * out_h * out_w;
    if (idx >= total_threads) return;

    int max_idx = static_cast<int>(indices[ind_off + idx]);
    if (max_idx >= 0) {
        atomic_add_float(&grad_input[gin_off + max_idx], grad_output[gout_off + idx]);
    }
}

extern "C" __global__ void adaptive_avg_pool2d_forward_kernel(
    const float* input, int in_off,
    float* output, int out_off,
    int N, int C, int H, int W, int OH, int OW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * OH * OW;
    if (idx >= total) return;

    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int c = (idx / (OW * OH)) % C;
    int n = idx / (OW * OH * C);

    int h_start = (oh * H) / OH;
    int h_end = ((oh + 1) * H + OH - 1) / OH;
    h_end = (h_end < H) ? h_end : H;

    int w_start = (ow * W) / OW;
    int w_end = ((ow + 1) * W + OW - 1) / OW;
    w_end = (w_end < W) ? w_end : W;

    int count = (h_end - h_start) * (w_end - w_start);
    float sum = 0.0f;
    int base_in = in_off + (n * C + c) * H * W;

    for (int h = h_start; h < h_end; ++h) {
        for (int w = w_start; w < w_end; ++w) {
            sum += input[base_in + h * W + w];
        }
    }
    output[out_off + idx] = count > 0 ? sum / count : 0.0f;
}

extern "C" __global__ void adaptive_avg_pool2d_backward_kernel(
    const float* grad_output, int gout_off,
    float* grad_input, int gin_off,
    int N, int C, int H, int W, int OH, int OW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    float sum_grad = 0.0f;
    for (int oh = 0; oh < OH; ++oh) {
        int h_start = (oh * H) / OH;
        int h_end = ((oh + 1) * H + OH - 1) / OH;
        h_end = (h_end < H) ? h_end : H;

        if (h >= h_start && h < h_end) {
            for (int ow = 0; ow < OW; ++ow) {
                int w_start = (ow * W) / OW;
                int w_end = ((ow + 1) * W + OW - 1) / OW;
                w_end = (w_end < W) ? w_end : W;

                if (w >= w_start && w < w_end) {
                    int count = (h_end - h_start) * (w_end - w_start);
                    int gout_idx = gout_off + ((n * C + c) * OH + oh) * OW + ow;
                    sum_grad += count > 0 ? grad_output[gout_idx] / count : 0.0f;
                }
            }
        }
    }
    grad_input[gin_off + idx] = sum_grad;
}

extern "C" __global__ void maxpool3d_kernel(
    const float* input, int in_off,
    float* output, int out_off,
    float* save_indices, int ind_off,
    int batch_size, int channels, int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w, int kernel_size, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * channels * out_d * out_h * out_w;
    if (idx >= total_threads) return;

    int w_out = idx % out_w;
    int h_out = (idx / out_w) % out_h;
    int d_out = (idx / (out_w * out_h)) % out_d;
    int c = (idx / (out_w * out_h * out_d)) % channels;
    int b = idx / (out_w * out_h * out_d * channels);

    float max_val = -1e37f;
    int max_idx = -1;
    for (int kz = 0; kz < kernel_size; ++kz) {
        int z = d_out * stride - padding + kz;
        if (z >= 0 && z < in_d) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                int y = h_out * stride - padding + ky;
                if (y >= 0 && y < in_h) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int x = w_out * stride - padding + kx;
                        if (x >= 0 && x < in_w) {
                            int input_idx = (((b * channels + c) * in_d + z) * in_h + y) * in_w + x;
                            float val = input[in_off + input_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = input_idx;
                            }
                        }
                    }
                }
            }
        }
    }
    output[out_off + idx] = max_val;
    save_indices[ind_off + idx] = static_cast<float>(max_idx);
}

extern "C" __global__ void maxpool3d_backward_kernel(
    const float* save_indices, int ind_off,
    const float* grad_output, int gout_off,
    float* grad_input, int gin_off,
    int batch_size, int channels, int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * channels * out_d * out_h * out_w;
    if (idx >= total_threads) return;

    int max_idx = static_cast<int>(save_indices[ind_off + idx]);
    if (max_idx >= 0) {
        atomic_add_float(&grad_input[gin_off + max_idx], grad_output[gout_off + idx]);
    }
}

extern "C" __global__ void softmax_forward_kernel(const float* A, int a_off,
                                     float* B, int b_off,
                                     int dim_size, int inner_size, int outer_size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    int total = outer_size * inner_size;
    if (idx >= total) return;
    int o = idx / inner_size;
    int i = idx % inner_size;
    float max_val = -1e37f;
    for (int d = 0; d < dim_size; ++d) {
        float val = A[a_off + o * dim_size * inner_size + d * inner_size + i];
        if (val > max_val) max_val = val;
    }
    float sum = 0.0f;
    for (int d = 0; d < dim_size; ++d) {
        float e = expf(A[a_off + o * dim_size * inner_size + d * inner_size + i] - max_val);
        sum += e;
    }
    for (int d = 0; d < dim_size; ++d) {
        int target_idx = o * dim_size * inner_size + d * inner_size + i;
        B[b_off + target_idx] = expf(A[a_off + target_idx] - max_val) / (sum + 1e-15f);
    }
}

extern "C" __global__ void softmax_backward_kernel(
    const float* out_data, int out_off,
    const float* grad_output, int gout_off,
    float* grad_input, int gin_off,
    int dim_size, int inner_size, int outer_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_size * inner_size;
    if (idx >= total) return;
    int o = idx / inner_size;
    int i = idx % inner_size;
    float sum_grad_out = 0.0f;
    for (int d = 0; d < dim_size; ++d) {
        int cur_idx = o * dim_size * inner_size + d * inner_size + i;
        sum_grad_out += grad_output[gout_off + cur_idx] * out_data[out_off + cur_idx];
    }
    for (int d = 0; d < dim_size; ++d) {
        int cur_idx = o * dim_size * inner_size + d * inner_size + i;
        grad_input[gin_off + cur_idx] = out_data[out_off + cur_idx] * (grad_output[gout_off + cur_idx] - sum_grad_out);
    }
}

extern "C" __global__ void layer_norm_forward_kernel(const float* input, int in_off,
                                        const float* weight, int w_off, int has_weight,
                                        const float* bias, int b_off, int has_bias,
                                        float* output, int out_off,
                                        float* save_mean, int sm_off,
                                        float* save_var, int sv_off,
                                        int N, int M, float eps) {
    int r = (blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= N) return;
    float mean = 0.0f;
    for (int c = 0; c < M; ++c) {
        mean += input[in_off + r * M + c];
    }
    mean /= M;
    float var = 0.0f;
    for (int c = 0; c < M; ++c) {
        float diff = input[in_off + r * M + c] - mean;
        var += diff * diff;
    }
    var /= M;
    float inv_std = 1.0f / sqrtf(var + eps);
    save_mean[sm_off + r] = mean;
    save_var[sv_off + r] = inv_std;
    for (int c = 0; c < M; ++c) {
        int idx = r * M + c;
        float x_hat = (input[in_off + idx] - mean) * inv_std;
        float w = (has_weight && weight) ? weight[w_off + c] : 1.0f;
        float b = (has_bias && bias) ? bias[b_off + c] : 0.0f;
        output[out_off + idx] = w * x_hat + b;
    }
}

extern "C" __global__ void fused_add_layer_norm_forward_kernel(
    const float* input, int in_off,
    const float* residual, int res_off,
    const float* weight, int w_off, int has_weight,
    const float* bias, int b_off, int has_bias,
    float* output, int out_off,
    float* save_mean, int sm_off,
    float* save_var, int sv_off,
    int N, int M, float eps)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    float mean = 0.0f;
    for (int c = 0; c < M; ++c) {
        mean += input[in_off + r * M + c] + residual[res_off + r * M + c];
    }
    mean /= M;
    float var = 0.0f;
    for (int c = 0; c < M; ++c) {
        float val = input[in_off + r * M + c] + residual[res_off + r * M + c];
        float diff = val - mean;
        var += diff * diff;
    }
    var /= M;
    float inv_std = 1.0f / sqrtf(var + eps);
    save_mean[sm_off + r] = mean;
    save_var[sv_off + r] = inv_std;
    for (int c = 0; c < M; ++c) {
        int idx = r * M + c;
        float val = input[in_off + idx] + residual[res_off + idx];
        float x_hat = (val - mean) * inv_std;
        float w = (has_weight && weight) ? weight[w_off + c] : 1.0f;
        float b = (has_bias && bias) ? bias[b_off + c] : 0.0f;
        output[out_off + idx] = w * x_hat + b;
    }
}

extern "C" __global__ void layer_norm_backward_dx_kernel(const float* input, int in_off,
                                            const float* grad_output, int gout_off,
                                            const float* weight, int w_off, int has_weight,
                                            float* grad_input, int gin_off,
                                            const float* save_mean, int sm_off,
                                            const float* save_var, int sv_off,
                                            int N, int M, float eps) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < N) {
        float mean = save_mean[sm_off + r];
        float inv_std = save_var[sv_off + r];
        float sum_dy = 0.0f;
        float sum_dy_xhat = 0.0f;
        for (int c = 0; c < M; ++c) {
            int idx = r * M + c;
            float dy = grad_output[gout_off + idx];
            float x_hat = (input[in_off + idx] - mean) * inv_std;
            float w = (has_weight && weight) ? weight[w_off + c] : 1.0f;
            sum_dy += dy * w;
            sum_dy_xhat += dy * w * x_hat;
        }
        for (int c = 0; c < M; ++c) {
            int idx = r * M + c;
            float x_hat = (input[in_off + idx] - mean) * inv_std;
            float dy = grad_output[gout_off + idx];
            float w = (has_weight && weight) ? weight[w_off + c] : 1.0f;
            grad_input[gin_off + idx] = inv_std * (dy * w - (sum_dy + x_hat * sum_dy_xhat) / M);
        }
    }
}

extern "C" __global__ void layer_norm_backward_dw_kernel(const float* input, int in_off,
                                            const float* grad_output, int gout_off,
                                            float* grad_weight, int gw_off,
                                            const float* save_mean, int sm_off,
                                            const float* save_var, int sv_off,
                                            int N, int M, float eps) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < M) {
        float sum_dw = 0.0f;
        for (int r = 0; r < N; ++r) {
            float mean = save_mean[sm_off + r];
            float inv_std = save_var[sv_off + r];
            int idx = r * M + c;
            float x_hat = (input[in_off + idx] - mean) * inv_std;
            sum_dw += grad_output[gout_off + idx] * x_hat;
        }
        grad_weight[gw_off + c] = sum_dw;
    }
}

extern "C" __global__ void layer_norm_backward_db_kernel(const float* grad_output, int gout_off,
                                            float* grad_bias, int gb_off,
                                            int N, int M) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < M) {
        float sum_db = 0.0f;
        for (int r = 0; r < N; ++r) {
            sum_db += grad_output[gout_off + r * M + c];
        }
        grad_bias[gb_off + c] = sum_db;
    }
}

extern "C" __global__ void batch_norm2d_forward_stats_kernel(
    const float* input, int in_off,
    float* running_mean, int rm_off,
    float* running_var, int rv_off,
    float* save_mean, int sm_off,
    float* save_var, int sv_off,
    int N, int C, int H, int W,
    int training, float momentum, float eps)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    int M = N * H * W;
    float m_val = 0.0f;
    float v_val = 0.0f;
    if (training) {
        float sum_val = 0.0f;
        for (int b = 0; b < N; ++b) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int idx = ((b * C + c) * H + h) * W + w;
                    sum_val += input[in_off + idx];
                }
            }
        }
        m_val = sum_val / M;
        save_mean[sm_off + c] = m_val;
        float sum_sq_val = 0.0f;
        for (int b = 0; b < N; ++b) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int idx = ((b * C + c) * H + h) * W + w;
                    float diff = input[in_off + idx] - m_val;
                    sum_sq_val += diff * diff;
                }
            }
        }
        v_val = sum_sq_val / M;
        save_var[sv_off + c] = v_val;
        running_mean[rm_off + c] = (1.0f - momentum) * running_mean[rm_off + c] + momentum * m_val;
        float unbiased_factor = M > 1 ? static_cast<float>(M) / (M - 1) : 1.0f;
        running_var[rv_off + c] = (1.0f - momentum) * running_var[rv_off + c] + momentum * v_val * unbiased_factor;
    } else {
        m_val = running_mean[rm_off + c];
        v_val = running_var[rv_off + c];
        save_mean[sm_off + c] = m_val;
        save_var[sv_off + c] = v_val;
    }
}

extern "C" __global__ void batch_norm2d_forward_norm_kernel(
    const float* input, int in_off,
    const float* save_mean, int sm_off,
    const float* save_var, int sv_off,
    const float* weight, int w_off,
    const float* bias, int b_off,
    float* output, int out_off,
    int N, int C, int H, int W, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    int c = (idx / (W * H)) % C;
    float m_val = save_mean[sm_off + c];
    float v_val = save_var[sv_off + c];
    float inv_std = 1.0f / sqrtf(v_val + eps);
    float x_hat = (input[in_off + idx] - m_val) * inv_std;
    float w = weight ? weight[w_off + c] : 1.0f;
    float b = bias ? bias[b_off + c] : 0.0f;
    output[out_off + idx] = w * x_hat + b;
}

extern "C" __global__ void batch_norm2d_backward_stats_kernel(
    const float* input, int in_off,
    const float* grad_output, int gout_off,
    const float* save_mean, int sm_off,
    const float* save_var, int sv_off,
    float* grad_weight, int gw_off,
    float* grad_bias, int gb_off,
    int N, int C, int H, int W, float eps)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float m_val = save_mean[sm_off + c];
    float v_val = save_var[sv_off + c];
    float inv_std = 1.0f / sqrtf(v_val + eps);
    float dscale_sum = 0.0f;
    float dshift_sum = 0.0f;
    for (int b = 0; b < N; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int idx = ((b * C + c) * H + h) * W + w;
                float x_hat = (input[in_off + idx] - m_val) * inv_std;
                dscale_sum += grad_output[gout_off + idx] * x_hat;
                dshift_sum += grad_output[gout_off + idx];
            }
        }
    }
    if (grad_weight) grad_weight[gw_off + c] = dscale_sum;
    if (grad_bias) grad_bias[gb_off + c] = dshift_sum;
}

extern "C" __global__ void batch_norm2d_backward_dx_kernel(
    const float* input, int in_off,
    const float* grad_output, int gout_off,
    const float* save_mean, int sm_off,
    const float* save_var, int sv_off,
    const float* weight, int w_off,
    const float* grad_weight, int gw_off,
    const float* grad_bias, int gb_off,
    float* grad_input, int gin_off,
    int N, int C, int H, int W, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    int c = (idx / (W * H)) % C;
    int M = N * H * W;
    float m_val = save_mean[sm_off + c];
    float v_val = save_var[sv_off + c];
    float inv_std = 1.0f / sqrtf(v_val + eps);
    float x_hat = (input[in_off + idx] - m_val) * inv_std;
    float dscale_sum = grad_weight ? grad_weight[gw_off + c] : 0.0f;
    float dshift_sum = grad_bias ? grad_bias[gb_off + c] : 0.0f;
    float w = weight ? weight[w_off + c] : 1.0f;
    grad_input[gin_off + idx] = w * inv_std / M * (M * grad_output[gout_off + idx] - dscale_sum * x_hat - dshift_sum);
}

extern "C" __global__ void mse_loss_forward(const float* input, int in_off,
                               const float* target, int tgt_off,
                               float* output, int out_off,
                               int size) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    while (idx < size) {
        float d = input[in_off + idx] - target[tgt_off + idx];
        local_sum += d * d;
        idx += gridDim.x * blockDim.x;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomic_add_float(&output[out_off], sdata[0] / (size > 0 ? size : 1));
    }
}

extern "C" __global__ void mse_loss_backward(
    const float* input, int in_off,
    const float* target, int tgt_off,
    const float* grad_output, int gout_off,
    float* grad_input, int gin_off,
    float scale, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        float diff = input[in_off + id] - target[tgt_off + id];
        float go = grad_output[gout_off];
        grad_input[gin_off + id] = diff * scale * go;
    }
}

extern "C" __global__ void l1_loss_forward(
    const float* input, int in_off,
    const float* target, int tgt_off,
    float* output, int out_off,
    int size)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    while (idx < size) {
        local_sum += fabsf(input[in_off + idx] - target[tgt_off + idx]);
        idx += gridDim.x * blockDim.x;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomic_add_float(&output[out_off], sdata[0] / (size > 0 ? size : 1));
    }
}

extern "C" __global__ void l1_loss_backward(
    const float* input, int in_off,
    const float* target, int tgt_off,
    const float* grad_output, int gout_off,
    float* grad_input, int gin_off,
    float scale, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        float diff = input[in_off + id] - target[tgt_off + id];
        float go = grad_output[gout_off];
        float val = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
        grad_input[gin_off + id] = val * scale * go;
    }
}

extern "C" __global__ void bce_loss_forward(
    const float* input, int in_off,
    const float* target, int tgt_off,
    float* output, int out_off,
    int size)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    while (idx < size) {
        float x = input[in_off + idx];
        float y = target[tgt_off + idx];
        if (x < 1e-7f) x = 1e-7f;
        if (x > 1.0f - 1e-7f) x = 1.0f - 1e-7f;
        local_sum -= (y * logf(x) + (1.0f - y) * logf(1.0f - x));
        idx += gridDim.x * blockDim.x;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomic_add_float(&output[out_off], sdata[0] / (size > 0 ? size : 1));
    }
}

extern "C" __global__ void bce_loss_backward(
    const float* input, int in_off,
    const float* target, int tgt_off,
    const float* grad_output, int gout_off,
    float* grad_input, int gin_off,
    float scale, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        float x = input[in_off + id];
        float y = target[tgt_off + id];
        float go = grad_output[gout_off];
        if (x < 1e-7f) x = 1e-7f;
        if (x > 1.0f - 1e-7f) x = 1.0f - 1e-7f;
        grad_input[gin_off + id] = scale * go * (x - y) / (x * (1.0f - x));
    }
}

extern "C" __global__ void cross_entropy_loss_forward(const float* input, int in_off,
                                         const float* target, int tgt_off,
                                         float* output, int out_off,
                                         int N, int C) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < N) {
        float max_val = input[in_off + i * C];
        for (int j = 1; j < C; ++j) {
            float val = input[in_off + i * C + j];
            if (val > max_val) max_val = val;
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < C; ++j) {
            sum_exp += expf(input[in_off + i * C + j] - max_val);
        }
        int target_idx = (int)target[tgt_off + i];
        float correct_logit = input[in_off + i * C + target_idx];
        float loss = -correct_logit + max_val + logf(sum_exp);
        atomic_add_float(&output[out_off], loss / (N > 0 ? N : 1));
    }
}

extern "C" __global__ void cross_entropy_loss_backward(const float* input, int in_off,
                                          const float* target, int tgt_off,
                                          const float* grad_output, int gout_off,
                                          float* grad_input, int gin_off,
                                          int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float max_val = input[in_off + i * C];
        for (int j = 1; j < C; ++j) {
            float val = input[in_off + i * C + j];
            if (val > max_val) max_val = val;
        }
        float sum = 0.0f;
        for (int j = 0; j < C; ++j) {
            sum += expf(input[in_off + i * C + j] - max_val);
        }
        int target_idx = (int)target[tgt_off + i];
        float go = grad_output[gout_off];
        for (int j = 0; j < C; ++j) {
            float prob = expf(input[in_off + i * C + j] - max_val) / sum;
            float indicator = (j == target_idx) ? 1.0f : 0.0f;
            grad_input[gin_off + i * C + j] = (prob - indicator) / N * go;
        }
    }
}

extern "C" __global__ void gelu_forward_kernel(const float* A, int a_off,
                                  float* B, int b_off,
                                  float* save_tanh, int st_off,
                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = A[a_off + idx];
        float C = 0.79788456f;
        float u = C * (x + 0.044715f * x * x * x);
        float tanh_u = tanhf(u);
        save_tanh[st_off + idx] = tanh_u;
        B[b_off + idx] = 0.5f * x * (1.0f + tanh_u);
    }
}

extern "C" __global__ void gelu_backward_kernel(const float* A, int a_off,
                                   const float* save_tanh, int st_off,
                                   const float* grad_output, int gout_off,
                                   float* grad_input, int gin_off,
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = A[a_off + idx];
        float tanh_u = save_tanh[st_off + idx];
        float C = 0.79788456f;
        float d_gelu = 0.5f * (1.0f + tanh_u) + 0.5f * x * (1.0f - tanh_u * tanh_u) * C * (1.0f + 0.134145f * x * x);
        grad_input[gin_off + idx] = grad_output[gout_off + idx] * d_gelu;
    }
}

extern "C" __global__ void embedding_forward(
    const float* input, int in_off,
    const float* weight, int w_off,
    float* output, int out_off,
    int num_indices, int num_embeddings, int embedding_dim)
{
    int idx_thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_thread >= num_indices * embedding_dim) return;

    int i = idx_thread / embedding_dim;
    int d = idx_thread % embedding_dim;

    int idx = static_cast<int>(input[in_off + i]);
    if (idx >= 0 && idx < num_embeddings) {
        output[out_off + i * embedding_dim + d] = weight[w_off + idx * embedding_dim + d];
    } else {
        output[out_off + i * embedding_dim + d] = 0.0f;
    }
}

extern "C" __global__ void embedding_backward(
    const float* input, int in_off,
    const float* grad_output, int gout_off,
    float* grad_weight, int gw_off,
    int num_indices, int num_embeddings, int embedding_dim)
{
    int idx_thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_thread >= num_embeddings * embedding_dim) return;

    int idx = idx_thread / embedding_dim;
    int d = idx_thread % embedding_dim;

    float sum = 0.0f;
    for (int i = 0; i < num_indices; ++i) {
        int input_val = static_cast<int>(input[in_off + i]);
        if (input_val == idx) {
            sum += grad_output[gout_off + i * embedding_dim + d];
        }
    }
    grad_weight[gw_off + idx_thread] = sum;
}

extern "C" __global__ void generate_dropout_mask(
    float* mask, int mask_off,
    float p, float scale, unsigned int seed, int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        unsigned int x = gid + seed;
        x = 1664525U * x + 1013904223U;
        x = 1664525U * x + 1013904223U;
        float r = static_cast<float>(x & 0xFFFFFFFF) / 4294967296.0f;
        mask[mask_off + gid] = (r >= p) ? scale : 0.0f;
    }
}

extern "C" __global__ void rope_forward(
    const float* X, int x_off,
    const float* cos_val, int cos_off,
    const float* sin_val, int sin_off,
    float* Y, int y_off,
    int B, int H, int T, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_d = D / 2;
    int total = B * H * T * half_d;
    if (idx >= total) return;

    int i = idx % half_d;
    int t = (idx / half_d) % T;
    int h = (idx / (half_d * T)) % H;
    int b = idx / (half_d * T * H);

    int64_t offset = (b * H * T + h * T + t) * D;

    float cos_v = cos_val[cos_off + t * half_d + i];
    float sin_v = sin_val[sin_off + t * half_d + i];

    Y[y_off + offset + 2 * i] = X[x_off + offset + 2 * i] * cos_v - X[x_off + offset + 2 * i + 1] * sin_v;
    Y[y_off + offset + 2 * i + 1] = X[x_off + offset + 2 * i + 1] * cos_v + X[x_off + offset + 2 * i] * sin_v;
}

extern "C" __global__ void rope_backward(
    const float* dy, int dy_off,
    const float* cos_val, int cos_off,
    const float* sin_val, int sin_off,
    float* dx, int dx_off,
    int B, int H, int T, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_d = D / 2;
    int total = B * H * T * half_d;
    if (idx >= total) return;

    int i = idx % half_d;
    int t = (idx / half_d) % T;
    int h = (idx / (half_d * T)) % H;
    int b = idx / (half_d * T * H);

    int64_t offset = (b * H * T + h * T + t) * D;

    float cos_v = cos_val[cos_off + t * half_d + i];
    float sin_v = sin_val[sin_off + t * half_d + i];

    dx[dx_off + offset + 2 * i] = dy[dy_off + offset + 2 * i] * cos_v + dy[dy_off + offset + 2 * i + 1] * sin_v;
    dx[dx_off + offset + 2 * i + 1] = dy[dy_off + offset + 2 * i + 1] * cos_v - dy[dy_off + offset + 2 * i] * sin_v;
}

extern "C" __global__ void paged_attention_forward(
    const float* q_ptr, int q_off,
    const float* k_ptr, int k_off,
    const float* v_ptr, int v_off,
    const float* bt_ptr, int bt_off,
    const float* cl_ptr, int cl_off,
    float* out_ptr, int out_off,
    int num_seqs, int num_heads, int num_kv_heads, int head_dim,
    int max_num_blocks, int block_size, float scale)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_seqs * num_heads;
    if (id >= total) return;

    int seq_idx = id / num_heads;
    int head_idx = id % num_heads;
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    int context_len = (int)cl_ptr[cl_off + seq_idx];
    if (context_len <= 0) return;

    float scores[2048];
    float max_val = -1e37f;

    for (int t = 0; t < context_len && t < 2048; ++t) {
        int block_idx = (int)bt_ptr[bt_off + seq_idx * max_num_blocks + t / block_size];
        int block_offset = t % block_size;
        int k_idx = block_idx * (num_kv_heads * block_size * head_dim) + kv_head_idx * (block_size * head_dim) + block_offset * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_ptr[q_off + seq_idx * (num_heads * head_dim) + head_idx * head_dim + d] * k_ptr[k_off + k_idx + d];
        }
        dot *= scale;
        scores[t] = dot;
        if (dot > max_val) max_val = dot;
    }

    float denominator = 0.0f;
    float acc[128] = {0.0f};

    for (int t = 0; t < context_len && t < 2048; ++t) {
        float exp_val = expf(scores[t] - max_val);
        denominator += exp_val;

        int block_idx = (int)bt_ptr[bt_off + seq_idx * max_num_blocks + t / block_size];
        int block_offset = t % block_size;
        int v_idx = block_idx * (num_kv_heads * block_size * head_dim) + kv_head_idx * (block_size * head_dim) + block_offset * head_dim;

        for (int d = 0; d < head_dim && d < 128; ++d) {
            acc[d] += exp_val * v_ptr[v_off + v_idx + d];
        }
    }

    for (int d = 0; d < head_dim && d < 128; ++d) {
        out_ptr[out_off + seq_idx * (num_heads * head_dim) + head_idx * head_dim + d] = acc[d] / (denominator + 1e-15f);
    }
}

extern "C" __global__ void w8a8_matmul_kernel(
    const float* X, int x_off,
    const float* W, int w_off,
    float* Y, int y_off,
    int M, int K, int N,
    float x_scale, float w_scale)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float xv = X[x_off + row * K + k];
            float wv = W[w_off + col * K + k];

            int8_t qx = (int8_t)clamp(rintf(xv / x_scale), -128.0f, 127.0f);
            int8_t qw = (int8_t)clamp(rintf(wv / w_scale), -128.0f, 127.0f);

            acc += (float)qx * (float)qw;
        }
        Y[y_off + row * N + col] = acc * x_scale * w_scale;
    }
}

__global__ void moe_gate_kernel(
    const float* logits, int l_off,
    float* probs, int p_off,
    float* indices, int idx_off,
    int N, int E, int top_k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float max_val = -1e9f;
    for (int e = 0; e < E; ++e) {
        float l = logits[l_off + i * E + e];
        if (l > max_val) max_val = l;
    }

    float sum_exp = 0.0f;
    for (int e = 0; e < E; ++e) {
        sum_exp += expf(logits[l_off + i * E + e] - max_val);
    }

    float top_p[8];
    int top_idx[8];
    for (int k = 0; k < top_k && k < 8; ++k) {
        top_p[k] = -1.0f;
        top_idx[k] = -1;
    }

    for (int e = 0; e < E; ++e) {
        float p = expf(logits[l_off + i * E + e] - max_val) / (sum_exp + 1e-9f);
        for (int k = 0; k < top_k && k < 8; ++k) {
            if (p > top_p[k]) {
                for (int s = (top_k < 8 ? top_k : 8) - 1; s > k; --s) {
                    top_p[s] = top_p[s - 1];
                    top_idx[s] = top_idx[s - 1];
                }
                top_p[k] = p;
                top_idx[k] = e;
                break;
            }
        }
    }

    float sum_top_k = 0.0f;
    for (int k = 0; k < top_k && k < 8; ++k) {
        sum_top_k += top_p[k];
    }

    for (int k = 0; k < top_k && k < 8; ++k) {
        probs[p_off + i * top_k + k] = top_p[k] / (sum_top_k + 1e-9f);
        indices[idx_off + i * top_k + k] = static_cast<float>(top_idx[k]);
    }
}

__global__ void moe_gate_backward_kernel(
    const float* grad_output, int gout_off,
    const float* input, int in_off,
    const float* gate_weight, int gw_off,
    const float* probs, int p_off,
    const float* indices, int idx_off,
    float* grad_input, int gin_off,
    float* grad_gate_weight, int ggw_off,
    int N, int D, int E, int top_k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sum_ds_s = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        sum_ds_s += grad_output[gout_off + i * top_k + k] * probs[p_off + i * top_k + k];
    }

    for (int k = 0; k < top_k; ++k) {
        int e = static_cast<int>(indices[idx_off + i * top_k + k]);
        if (e >= 0 && e < E) {
            float p = probs[p_off + i * top_k + k];
            float dh = p * (grad_output[gout_off + i * top_k + k] - sum_ds_s);

            for (int d = 0; d < D; ++d) {
                atomic_add_float(&grad_input[gin_off + i * D + d], dh * gate_weight[gw_off + e * D + d]);
                atomic_add_float(&grad_gate_weight[ggw_off + e * D + d], dh * input[in_off + i * D + d]);
            }
        }
    }
}

__global__ void moe_expert_forward_kernel(
    const float* input, int in_off,
    const float* expert_weight, int ew_off,
    const float* expert_bias, int eb_off,
    const float* probs, int p_off,
    const float* indices, int idx_off,
    float* output, int out_off,
    int N, int D, int out_features, int expert_idx, int top_k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    for (int k = 0; k < top_k; ++k) {
        int exp_idx = static_cast<int>(indices[idx_off + i * top_k + k]);
        if (exp_idx == expert_idx) {
            float p = probs[p_off + i * top_k + k];
            for (int o = 0; o < out_features; ++o) {
                float val = 0.0f;
                for (int d = 0; d < D; ++d) {
                    val += input[in_off + i * D + d] * expert_weight[ew_off + o * D + d];
                }
                if (expert_bias) val += expert_bias[eb_off + o];
                atomic_add_float(&output[out_off + i * out_features + o], p * val);
            }
        }
    }
}

__global__ void moe_expert_backward_kernel(
    const float* grad_output, int gout_off,
    const float* input, int in_off,
    const float* expert_weight, int ew_off,
    const float* expert_bias, int eb_off,
    const float* probs, int p_off,
    const float* indices, int idx_off,
    float* grad_input, int gin_off,
    float* grad_expert, int ge_off,
    float* grad_bias, int gb_off,
    float* grad_probs, int gp_off,
    int N, int D, int out_features, int expert_idx, int top_k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    for (int k = 0; k < top_k; ++k) {
        int exp_idx = static_cast<int>(indices[idx_off + i * top_k + k]);
        if (exp_idx == expert_idx) {
            float p = probs[p_off + i * top_k + k];
            for (int o = 0; o < out_features; ++o) {
                float val = 0.0f;
                for (int d = 0; d < D; ++d) {
                    val += input[in_off + i * D + d] * expert_weight[ew_off + o * D + d];
                }
                if (expert_bias) val += expert_bias[eb_off + o];
                atomic_add_float(&grad_probs[gp_off + i * top_k + k], grad_output[gout_off + i * out_features + o] * val);
            }

            for (int o = 0; o < out_features; ++o) {
                float go = grad_output[gout_off + i * out_features + o] * p;
                if (grad_bias) {
                    atomic_add_float(&grad_bias[gb_off + o], go);
                }
                for (int d = 0; d < D; ++d) {
                    atomic_add_float(&grad_expert[ge_off + o * D + d], go * input[in_off + i * D + d]);
                    atomic_add_float(&grad_input[gin_off + i * D + d], go * expert_weight[ew_off + o * D + d]);
                }
            }
        }
    }
}

extern "C" void gpu_moe_gate(void* logits, int l_off, void* probs, int p_off, void* indices, int idx_off, int N, int E, int top_k) {
    if (N <= 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
#ifndef __HIP_PLATFORM_AMD__
    moe_gate_kernel<<<blocks, threads, 0, g_compute_stream>>>((const float*)logits, l_off, (float*)probs, p_off, (float*)indices, idx_off, N, E, top_k);
#else
    hipLaunchKernelGGL(moe_gate_kernel, dim3(blocks), dim3(threads), 0, g_compute_stream, (const float*)logits, l_off, (float*)probs, p_off, (float*)indices, idx_off, N, E, top_k);
#endif
}

extern "C" void gpu_moe_gate_backward(void* grad_output, int gout_off, void* input, int in_off, void* gate_weight, int gw_off, void* probs, int p_off, void* indices, int idx_off, void* grad_input, int gin_off, void* grad_gate_weight, int ggw_off, int N, int D, int E, int top_k) {
    if (N <= 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
#ifndef __HIP_PLATFORM_AMD__
    moe_gate_backward_kernel<<<blocks, threads, 0, g_compute_stream>>>((const float*)grad_output, gout_off, (const float*)input, in_off, (const float*)gate_weight, gw_off, (const float*)probs, p_off, (const float*)indices, idx_off, (float*)grad_input, gin_off, (float*)grad_gate_weight, ggw_off, N, D, E, top_k);
#else
    hipLaunchKernelGGL(moe_gate_backward_kernel, dim3(blocks), dim3(threads), 0, g_compute_stream, (const float*)grad_output, gout_off, (const float*)input, in_off, (const float*)gate_weight, gw_off, (const float*)probs, p_off, (const float*)indices, idx_off, (float*)grad_input, gin_off, (float*)grad_gate_weight, ggw_off, N, D, E, top_k);
#endif
}

extern "C" void gpu_moe_expert_forward(void* input, int in_off, void* expert_weight, int ew_off, void* expert_bias, int eb_off, void* probs, int p_off, void* indices, int idx_off, void* output, int out_off, int N, int D, int out_features, int expert_idx, int top_k) {
    if (N <= 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
#ifndef __HIP_PLATFORM_AMD__
    moe_expert_forward_kernel<<<blocks, threads, 0, g_compute_stream>>>((const float*)input, in_off, (const float*)expert_weight, ew_off, (const float*)expert_bias, eb_off, (const float*)probs, p_off, (const float*)indices, idx_off, (float*)output, out_off, N, D, out_features, expert_idx, top_k);
#else
    hipLaunchKernelGGL(moe_expert_forward_kernel, dim3(blocks), dim3(threads), 0, g_compute_stream, (const float*)input, in_off, (const float*)expert_weight, ew_off, (const float*)expert_bias, eb_off, (const float*)probs, p_off, (const float*)indices, idx_off, (float*)output, out_off, N, D, out_features, expert_idx, top_k);
#endif
}

extern "C" void gpu_moe_expert_backward(void* grad_output, int gout_off, void* input, int in_off, void* expert_weight, int ew_off, void* expert_bias, int eb_off, void* probs, int p_off, void* indices, int idx_off, void* grad_input, int gin_off, void* grad_expert, int ge_off, void* grad_bias, int gb_off, void* grad_probs, int gp_off, int N, int D, int out_features, int expert_idx, int top_k) {
    if (N <= 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
#ifndef __HIP_PLATFORM_AMD__
    moe_expert_backward_kernel<<<blocks, threads, 0, g_compute_stream>>>((const float*)grad_output, gout_off, (const float*)input, in_off, (const float*)expert_weight, ew_off, (const float*)expert_bias, eb_off, (const float*)probs, p_off, (const float*)indices, idx_off, (float*)grad_input, gin_off, (float*)grad_expert, ge_off, (float*)grad_bias, gb_off, (float*)grad_probs, gp_off, N, D, out_features, expert_idx, top_k);
#else
    hipLaunchKernelGGL(moe_expert_backward_kernel, dim3(blocks), dim3(threads), 0, g_compute_stream, (const float*)grad_output, gout_off, (const float*)input, in_off, (const float*)expert_weight, ew_off, (const float*)expert_bias, eb_off, (const float*)probs, p_off, (const float*)indices, idx_off, (float*)grad_input, gin_off, (float*)grad_expert, ge_off, (float*)grad_bias, gb_off, (float*)grad_probs, gp_off, N, D, out_features, expert_idx, top_k);
#endif
}
