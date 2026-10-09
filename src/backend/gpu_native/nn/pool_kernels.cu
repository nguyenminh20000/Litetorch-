#include "gpu_common.h"

extern "C" __global__ void maxpool2d_kernel(const float* input, int in_off,
                               float* output, int out_off,
                               int* indices, int ind_off,
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
    indices[ind_off + idx] = max_idx;
}

extern "C" __global__ void maxpool2d_backward_kernel(
    const int* indices, int ind_off,
    const float* grad_output, int gout_off,
    float* grad_input, int gin_off,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * channels * out_h * out_w;
    if (idx >= total_threads) return;

    int max_idx = indices[ind_off + idx];
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
    int* save_indices, int ind_off,
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
    save_indices[ind_off + idx] = max_idx;
}

extern "C" __global__ void maxpool3d_backward_kernel(
    const int* save_indices, int ind_off,
    const float* grad_output, int gout_off,
    float* grad_input, int gin_off,
    int batch_size, int channels, int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * channels * out_d * out_h * out_w;
    if (idx >= total_threads) return;

    int max_idx = save_indices[ind_off + idx];
    if (max_idx >= 0) {
        atomic_add_float(&grad_input[gin_off + max_idx], grad_output[gout_off + idx]);
    }
}
