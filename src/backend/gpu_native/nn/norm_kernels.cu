#include "gpu_common.h"

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

extern "C" __global__ void layer_norm_fast_kernel(const float* input, int in_off,
                                        const float* weight, int w_off, int has_weight,
                                        const float* bias, int b_off, int has_bias,
                                        float* output, int out_off,
                                        float* save_mean, int sm_off,
                                        float* save_var, int sv_off,
                                        int N, int M, float eps) {
    int r = blockIdx.x;
    if (r >= N) return;
    int tid = threadIdx.x;
    const float* in_row = input + in_off + (int64_t)r * M;
    float* out_row = output + out_off + (int64_t)r * M;
    bool aligned = ((reinterpret_cast<uintptr_t>(in_row) & 15) == 0) &&
                   ((reinterpret_cast<uintptr_t>(out_row) & 15) == 0);
    __shared__ float sdata[256];

    float tsum = 0.0f;
    int vec_n = M >> 2;
    if (aligned) {
        const float4* in4 = reinterpret_cast<const float4*>(in_row);
        for (int i = tid; i < vec_n; i += 256) {
            float4 v = in4[i];
            tsum += v.x + v.y + v.z + v.w;
        }
    }
    for (int i = (aligned ? (vec_n << 2) : 0) + tid; i < M; i += 256) {
        tsum += in_row[i];
    }
    sdata[tid] = tsum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)M;

    float tvar = 0.0f;
    if (aligned) {
        const float4* in4 = reinterpret_cast<const float4*>(in_row);
        for (int i = tid; i < vec_n; i += 256) {
            float4 v = in4[i];
            float d0 = v.x - mean, d1 = v.y - mean, d2 = v.z - mean, d3 = v.w - mean;
            tvar += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }
    }
    for (int i = (aligned ? (vec_n << 2) : 0) + tid; i < M; i += 256) {
        float d = in_row[i] - mean;
        tvar += d * d;
    }
    sdata[tid] = tvar;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / (float)M + eps);
    if (tid == 0) {
        if (save_mean) save_mean[sm_off + r] = mean;
        if (save_var) save_var[sv_off + r] = inv_std;
    }

    for (int i = tid; i < M; i += 256) {
        float x_hat = (in_row[i] - mean) * inv_std;
        float w = (has_weight && weight) ? weight[w_off + i] : 1.0f;
        float b = (has_bias && bias) ? bias[b_off + i] : 0.0f;
        out_row[i] = w * x_hat + b;
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

extern "C" __global__ void layer_norm_backward_fused_kernel(const float* input, int in_off,
                                            const float* grad_output, int gout_off,
                                            const float* weight, int w_off, int has_weight,
                                            float* grad_input, int gin_off,
                                            float* grad_weight, int gw_off, int has_dw,
                                            float* grad_bias, int gb_off, int has_db,
                                            const float* save_mean, int sm_off,
                                            const float* save_var, int sv_off,
                                            int N, int M) {
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
            float dy = grad_output[gout_off + idx];
            float x_hat = (input[in_off + idx] - mean) * inv_std;
            float w = (has_weight && weight) ? weight[w_off + c] : 1.0f;
            grad_input[gin_off + idx] = inv_std * (dy * w - (sum_dy + x_hat * sum_dy_xhat) / M);
            if (has_dw) atomicAdd(&grad_weight[gw_off + c], dy * x_hat);
            if (has_db) atomicAdd(&grad_bias[gb_off + c], dy);
        }
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
