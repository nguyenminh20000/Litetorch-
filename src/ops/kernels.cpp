#include "litetorch/ops.h"
#include <string>

namespace litetorch {

extern const std::string litetorch_kernels_src = R"litetorch_raw_cl(
inline float atomic_add_float(__global float* addr, float val) {
    union {
        unsigned int u32;
        float        f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32     = expected.f32 + val;
        current.u32  = atomic_cmpxchg((__global volatile unsigned int*)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
    return current.f32;
}
__kernel void elementwise_add(__global const float* A, int a_off, __global const float* B, int b_off, __global float* C, int c_off, int size) {
    int id = get_global_id(0);
    if (id < size) C[c_off + id] = A[a_off + id] + B[b_off + id];
}
__kernel void elementwise_add_inplace(__global float* A, int a_off, __global const float* B, int b_off, int size) {
    int id = get_global_id(0);
    if (id < size) A[a_off + id] = A[a_off + id] + B[b_off + id];
}
__kernel void elementwise_sub(__global const float* A, int a_off, __global const float* B, int b_off, __global float* C, int c_off, int size) {
    int id = get_global_id(0);
    if (id < size) C[c_off + id] = A[a_off + id] - B[b_off + id];
}
__kernel void elementwise_mul(__global const float* A, int a_off, __global const float* B, int b_off, __global float* C, int c_off, int size) {
    int id = get_global_id(0);
    if (id < size) C[c_off + id] = A[a_off + id] * B[b_off + id];
}
__kernel void elementwise_div(__global const float* A, int a_off, __global const float* B, int b_off, __global float* C, int c_off, int size) {
    int id = get_global_id(0);
    if (id < size) C[c_off + id] = A[a_off + id] / B[b_off + id];
}
__kernel void relu_forward(__global const float* A, int a_off, __global float* B, int b_off, int size) {
    int id = get_global_id(0);
    if (id < size) B[b_off + id] = A[a_off + id] > 0.0f ? A[a_off + id] : 0.0f;
}
__kernel void relu_backward_kernel(__global const float* A, int a_off, __global const float* grad_output, int gout_off, __global float* grad_input, int gin_off, int size) {
    int id = get_global_id(0);
    if (id < size) grad_input[gin_off + id] = A[a_off + id] > 0.0f ? grad_output[gout_off + id] : 0.0f;
}
__kernel void sigmoid_forward(__global const float* A, int a_off, __global float* B, int b_off, int size) {
    int id = get_global_id(0);
    if (id < size) B[b_off + id] = 1.0f / (1.0f + exp(-A[a_off + id]));
}
__kernel void tanh_forward(__global const float* A, int a_off, __global float* B, int b_off, int size) {
    int id = get_global_id(0);
    if (id < size) B[b_off + id] = tanh(A[a_off + id]);
}
__kernel void pow_forward(__global const float* A, int a_off, float exponent, __global float* B, int b_off, int size) {
    int id = get_global_id(0);
    if (id < size) B[b_off + id] = pow(A[a_off + id], exponent);
}
__kernel void sqrt_forward(__global const float* A, int a_off, __global float* B, int b_off, int size) {
    int id = get_global_id(0);
    if (id < size) B[b_off + id] = sqrt(A[a_off + id]);
}
__kernel void exp_forward(__global const float* A, int a_off, __global float* B, int b_off, int size) {
    int id = get_global_id(0);
    if (id < size) B[b_off + id] = exp(A[a_off + id]);
}
__kernel void log_forward(__global const float* A, int a_off, __global float* B, int b_off, int size) {
    int id = get_global_id(0);
    if (id < size) B[b_off + id] = log(A[a_off + id]);
}
__kernel void abs_forward(__global const float* A, int a_off, __global float* B, int b_off, int size) {
    int id = get_global_id(0);
    if (id < size) B[b_off + id] = fabs(A[a_off + id]);
}
__kernel void neg_forward(__global const float* A, int a_off, __global float* B, int b_off, int size) {
    int id = get_global_id(0);
    if (id < size) B[b_off + id] = -A[a_off + id];
}
__kernel void leaky_relu_forward(__global const float* A, int a_off, float negative_slope, __global float* B, int b_off, int size) {
    int id = get_global_id(0);
    if (id < size) B[b_off + id] = A[a_off + id] > 0.0f ? A[a_off + id] : A[a_off + id] * negative_slope;
}
__kernel void sigmoid_backward_kernel(__global const float* out_data, int out_off, __global const float* grad_output, int gout_off, __global float* grad_input, int gin_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        float o = out_data[out_off + id];
        grad_input[gin_off + id] = grad_output[gout_off + id] * o * (1.0f - o);
    }
}
__kernel void tanh_backward_kernel(__global const float* out_data, int out_off, __global const float* grad_output, int gout_off, __global float* grad_input, int gin_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        float o = out_data[out_off + id];
        grad_input[gin_off + id] = grad_output[gout_off + id] * (1.0f - o * o);
    }
}
__kernel void leaky_relu_backward_kernel(__global const float* A, int a_off, __global const float* grad_output, int gout_off, __global float* grad_input, int gin_off, int size, float negative_slope) {
    int id = get_global_id(0);
    if (id < size) {
        float x = A[a_off + id];
        grad_input[gin_off + id] = x > 0.0f ? grad_output[gout_off + id] : grad_output[gout_off + id] * negative_slope;
    }
}
__kernel void matmul_kernel(__global const float* A, int a_off, __global const float* B, int b_off, __global float* C, int c_off, int M, int K, int N) {
    __local float As[16][16];
    __local float Bs[16][16];
    int row = get_global_id(0);
    int col = get_global_id(1);
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);
    float sum = 0.0f;
    int numTiles = (K + 15) / 16;
    for (int t = 0; t < numTiles; ++t) {
        int tiledCol = t * 16 + localCol;
        if (row < M && tiledCol < K) {
            As[localRow][localCol] = A[a_off + row * K + tiledCol];
        } else {
            As[localRow][localCol] = 0.0f;
        }
        int tiledRow = t * 16 + localRow;
        if (tiledRow < K && col < N) {
            Bs[localRow][localCol] = B[b_off + tiledRow * N + col];
        } else {
            Bs[localRow][localCol] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 16; ++k) {
            sum += As[localRow][k] * Bs[k][localCol];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) {
        C[c_off + row * N + col] = sum;
    }
}
__kernel void bmm_kernel(__global const float* A, int a_off, __global const float* B, int b_off, __global float* C, int c_off, int M, int K, int N, int batch_size) {
    __local float As[16][16];
    __local float Bs[16][16];
    int row = get_global_id(0);
    int col = get_global_id(1);
    int batch_idx = get_global_id(2);
    if (batch_idx >= batch_size) return;
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);
    int a_batch_offset = batch_idx * M * K;
    int b_batch_offset = batch_idx * K * N;
    int c_batch_offset = batch_idx * M * N;
    float sum = 0.0f;
    int numTiles = (K + 15) / 16;
    for (int t = 0; t < numTiles; ++t) {
        int tiledCol = t * 16 + localCol;
        if (row < M && tiledCol < K) {
            As[localRow][localCol] = A[a_off + a_batch_offset + row * K + tiledCol];
        } else {
            As[localRow][localCol] = 0.0f;
        }
        int tiledRow = t * 16 + localRow;
        if (tiledRow < K && col < N) {
            Bs[localRow][localCol] = B[b_off + b_batch_offset + tiledRow * N + col];
        } else {
            Bs[localRow][localCol] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 16; ++k) {
            sum += As[localRow][k] * Bs[k][localCol];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) {
        C[c_off + c_batch_offset + row * N + col] = sum;
    }
}
__kernel void conv2d_kernel(__global const float* input, int in_off,
                            __global const float* weight, int w_off,
                            __global const float* bias, int b_off, int has_bias,
                            __global float* output, int out_off,
                            int batch_size, int in_channels, int in_h, int in_w,
                            int out_channels, int out_h, int out_w,
                            int kh, int kw, int stride, int padding) {
    int idx = get_global_id(0);
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
__kernel void maxpool2d_kernel(__global const float* input, int in_off,
                               __global float* output, int out_off,
                               __global float* indices, int ind_off,
                               int batch_size, int channels, int in_h, int in_w,
                               int out_h, int out_w, int kernel_size, int stride, int padding) {
    int idx = get_global_id(0);
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
__kernel void softmax_forward_kernel(__global const float* A, int a_off,
                                     __global float* B, int b_off,
                                     int dim_size, int inner_size, int outer_size) {
    int idx = get_global_id(0);
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
        float e = exp(A[a_off + o * dim_size * inner_size + d * inner_size + i] - max_val);
        sum += e;
    }
    for (int d = 0; d < dim_size; ++d) {
        int target_idx = o * dim_size * inner_size + d * inner_size + i;
        B[b_off + target_idx] = exp(A[a_off + target_idx] - max_val) / sum;
    }
}
__kernel void softmax_backward_kernel(__global const float* out_data, int out_off,
                                      __global const float* grad_output, int gout_off,
                                      __global float* grad_input, int gin_off,
                                      int dim_size, int inner_size, int outer_size) {
    int idx = get_global_id(0);
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
__kernel void gelu_forward_kernel(__global const float* A, int a_off,
                                  __global float* B, int b_off,
                                  __global float* save_tanh, int st_off,
                                  int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        float x = A[a_off + idx];
        float C = 0.79788456f;
        float u = C * (x + 0.044715f * x * x * x);
        float tanh_u = tanh(u);
        save_tanh[st_off + idx] = tanh_u;
        B[b_off + idx] = 0.5f * x * (1.0f + tanh_u);
    }
}
__kernel void gelu_backward_kernel(__global const float* A, int a_off,
                                   __global const float* save_tanh, int st_off,
                                   __global const float* grad_output, int gout_off,
                                   __global float* grad_input, int gin_off,
                                   int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        float x = A[a_off + idx];
        float tanh_u = save_tanh[st_off + idx];
        float C = 0.79788456f;
        float d_gelu = 0.5f * (1.0f + tanh_u) + 0.5f * x * (1.0f - tanh_u * tanh_u) * C * (1.0f + 0.134145f * x * x);
        grad_input[gin_off + idx] = grad_output[gout_off + idx] * d_gelu;
    }
}
__kernel void layer_norm_forward_kernel(__global const float* input, int in_off,
                                        __global const float* weight, int w_off, int has_weight,
                                        __global const float* bias, int b_off, int has_bias,
                                        __global float* output, int out_off,
                                        __global float* save_mean, int sm_off,
                                        __global float* save_var, int sv_off,
                                        int N, int M, float eps) {
    int r = get_global_id(0);
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
    float inv_std = 1.0f / sqrt(var + eps);
    save_mean[sm_off + r] = mean;
    save_var[sv_off + r] = inv_std;
    for (int c = 0; c < M; ++c) {
        int idx = r * M + c;
        float x_hat = (input[in_off + idx] - mean) * inv_std;
        float w = has_weight ? weight[w_off + c] : 1.0f;
        float b = has_bias ? bias[b_off + c] : 0.0f;
        output[out_off + idx] = w * x_hat + b;
    }
}
__kernel void fused_add_layer_norm_forward_kernel(__global const float* input, int in_off,
                                                  __global const float* residual, int res_off,
                                                  __global const float* weight, int w_off, int has_weight,
                                                  __global const float* bias, int b_off, int has_bias,
                                                  __global float* output, int out_off,
                                                  __global float* save_mean, int sm_off,
                                                  __global float* save_var, int sv_off,
                                                  int N, int M, float eps) {
    int r = get_global_id(0);
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
    float inv_std = 1.0f / sqrt(var + eps);
    save_mean[sm_off + r] = mean;
    save_var[sv_off + r] = inv_std;
    for (int c = 0; c < M; ++c) {
        int idx = r * M + c;
        float val = input[in_off + idx] + residual[res_off + idx];
        float x_hat = (val - mean) * inv_std;
        float w = has_weight ? weight[w_off + c] : 1.0f;
        float b = has_bias ? bias[b_off + c] : 0.0f;
        output[out_off + idx] = w * x_hat + b;
    }
}
__kernel void layer_norm_backward_dx_kernel(__global const float* input, int in_off,
                                            __global const float* grad_output, int gout_off,
                                            __global const float* weight, int w_off, int has_weight,
                                            __global float* grad_input, int gin_off,
                                            __global const float* save_mean, int sm_off,
                                            __global const float* save_var, int sv_off,
                                            int N, int M, float eps) {
    int r = get_global_id(0);
    if (r >= N) return;
    float mean = save_mean[sm_off + r];
    float inv_std = save_var[sv_off + r];
    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (int c = 0; c < M; ++c) {
        int idx = r * M + c;
        float dy = grad_output[gout_off + idx];
        float x_hat = (input[in_off + idx] - mean) * inv_std;
        float w = has_weight ? weight[w_off + c] : 1.0f;
        sum_dy += dy * w;
        sum_dy_xhat += dy * w * x_hat;
    }
    for (int c = 0; c < M; ++c) {
        int idx = r * M + c;
        float x_hat = (input[in_off + idx] - mean) * inv_std;
        float dy = grad_output[gout_off + idx];
        float w = has_weight ? weight[w_off + c] : 1.0f;
        grad_input[gin_off + idx] = inv_std * (dy * w - (sum_dy + x_hat * sum_dy_xhat) / M);
    }
}
__kernel void layer_norm_backward_dw_kernel(__global const float* input, int in_off,
                                            __global const float* grad_output, int gout_off,
                                            __global float* grad_weight, int gw_off,
                                            __global const float* save_mean, int sm_off,
                                            __global const float* save_var, int sv_off,
                                            int N, int M, float eps) {
    int c = get_global_id(0);
    if (c >= M) return;
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
__kernel void layer_norm_backward_db_kernel(__global const float* grad_output, int gout_off,
                                            __global float* grad_bias, int gb_off,
                                            int N, int M) {
    int c = get_global_id(0);
    if (c >= M) return;
    float sum_db = 0.0f;
    for (int r = 0; r < N; ++r) {
        sum_db += grad_output[gout_off + r * M + c];
    }
    grad_bias[gb_off + c] = sum_db;
}
__kernel void batch_norm2d_forward_stats_kernel(__global const float* input, int in_off,
                                                __global float* running_mean, int rm_off,
                                                __global float* running_var, int rv_off,
                                                __global float* save_mean, int sm_off,
                                                __global float* save_var, int sv_off,
                                                int N, int C, int H, int W,
                                                int training, float momentum, float eps) {
    int c = get_global_id(0);
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
        float unbiased_factor = M > 1 ? (float)M / (M - 1) : 1.0f;
        running_var[rv_off + c] = (1.0f - momentum) * running_var[rv_off + c] + momentum * v_val * unbiased_factor;

    } else {
        m_val = running_mean[rm_off + c];
        v_val = running_var[rv_off + c];
        save_mean[sm_off + c] = m_val;
        save_var[sv_off + c] = v_val;
    }
}
__kernel void batch_norm2d_forward_norm_kernel(__global const float* input, int in_off,
                                               __global const float* save_mean, int sm_off,
                                               __global const float* save_var, int sv_off,
                                               __global const float* weight, int w_off,
                                               __global const float* bias, int b_off,
                                               __global float* output, int out_off,
                                               int N, int C, int H, int W, float eps) {
    int idx = get_global_id(0);
    int total = N * C * H * W;
    if (idx >= total) return;
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int b = idx / (W * H * C);
    float m_val = save_mean[sm_off + c];
    float v_val = save_var[sv_off + c];
    float inv_std = 1.0f / sqrt(v_val + eps);
    float x_hat = (input[in_off + idx] - m_val) * inv_std;
    output[out_off + idx] = weight[w_off + c] * x_hat + bias[b_off + c];
}
__kernel void batch_norm2d_backward_stats_kernel(__global const float* input, int in_off,
                                                 __global const float* grad_output, int gout_off,
                                                 __global const float* save_mean, int sm_off,
                                                 __global const float* save_var, int sv_off,
                                                 __global float* grad_weight, int gw_off,
                                                 __global float* grad_bias, int gb_off,
                                                 int N, int C, int H, int W, float eps) {
    int c = get_global_id(0);
    if (c >= C) return;
    int M = N * H * W;
    float m_val = save_mean[sm_off + c];
    float v_val = save_var[sv_off + c];
    float inv_std = 1.0f / sqrt(v_val + eps);
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
    grad_weight[gw_off + c] = dscale_sum;
    grad_bias[gb_off + c] = dshift_sum;
}
__kernel void batch_norm2d_backward_dx_kernel(__global const float* input, int in_off,
                                              __global const float* grad_output, int gout_off,
                                              __global const float* save_mean, int sm_off,
                                              __global const float* save_var, int sv_off,
                                              __global const float* weight, int w_off,
                                              __global const float* grad_weight, int gw_off,
                                              __global const float* grad_bias, int gb_off,
                                              __global float* grad_input, int gin_off,
                                              int N, int C, int H, int W, float eps) {
    int idx = get_global_id(0);
    int total = N * C * H * W;
    if (idx >= total) return;
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int b = idx / (W * H * C);
    int M = N * H * W;
    float m_val = save_mean[sm_off + c];
    float v_val = save_var[sv_off + c];
    float inv_std = 1.0f / sqrt(v_val + eps);
    float x_hat = (input[in_off + idx] - m_val) * inv_std;
    float dscale_sum = grad_weight[gw_off + c];
    float dshift_sum = grad_bias[gb_off + c];
    grad_input[gin_off + idx] = weight[w_off + c] * inv_std / M * (M * grad_output[gout_off + idx] - dscale_sum * x_hat - dshift_sum);
}
__kernel void mse_loss_forward(__global const float* input, int in_off,
                               __global const float* target, int tgt_off,
                               __global float* output, int out_off,
                               int size) {
    __local float sdata[256];
    int tid = get_local_id(0);
    int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
    float local_sum = 0.0f;
    while (idx < size) {
        float d = input[in_off + idx] - target[tgt_off + idx];
        local_sum += d * d;
        idx += get_num_groups(0) * get_local_size(0);
    }
    sdata[tid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        atomic_add_float(&output[out_off], sdata[0]);
    }
}
__kernel void mse_loss_backward(__global const float* input, int in_off,
                                __global const float* target, int tgt_off,
                                __global const float* grad_output, int gout_off,
                                __global float* grad_input, int gin_off,
                                float scale, int size) {
    int id = get_global_id(0);
    if (id < size) {
        float diff = input[in_off + id] - target[tgt_off + id];
        float go = grad_output[gout_off];
        grad_input[gin_off + id] = diff * scale * go;
    }
}
__kernel void l1_loss_forward(__global const float* input, int in_off,
                              __global const float* target, int tgt_off,
                              __global float* output, int out_off,
                              int size) {
    __local float sdata[256];
    int tid = get_local_id(0);
    int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
    float local_sum = 0.0f;
    while (idx < size) {
        local_sum += fabs(input[in_off + idx] - target[tgt_off + idx]);
        idx += get_num_groups(0) * get_local_size(0);
    }
    sdata[tid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        atomic_add_float(&output[out_off], sdata[0]);
    }
}
__kernel void l1_loss_backward(__global const float* input, int in_off,
                               __global const float* target, int tgt_off,
                               __global const float* grad_output, int gout_off,
                               __global float* grad_input, int gin_off,
                               float scale, int size) {
    int id = get_global_id(0);
    if (id < size) {
        float diff = input[in_off + id] - target[tgt_off + id];
        float go = grad_output[gout_off];
        float val = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
        grad_input[gin_off + id] = val * scale * go;
    }
}
__kernel void bce_loss_forward(__global const float* input, int in_off,
                               __global const float* target, int tgt_off,
                               __global float* output, int out_off,
                               int size) {
    __local float sdata[256];
    int tid = get_local_id(0);
    int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
    float local_sum = 0.0f;
    while (idx < size) {
        float x = input[in_off + idx];
        float y = target[tgt_off + idx];
        if (x < 1e-7f) x = 1e-7f;
        if (x > 1.0f - 1e-7f) x = 1.0f - 1e-7f;
        local_sum -= (y * log(x) + (1.0f - y) * log(1.0f - x));
        idx += get_num_groups(0) * get_local_size(0);
    }
    sdata[tid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        atomic_add_float(&output[out_off], sdata[0]);
    }
}
__kernel void bce_loss_backward(__global const float* input, int in_off,
                                __global const float* target, int tgt_off,
                                __global const float* grad_output, int gout_off,
                                __global float* grad_input, int gin_off,
                                float scale, int size) {
    int id = get_global_id(0);
    if (id < size) {
        float x = input[in_off + id];
        float y = target[tgt_off + id];
        float go = grad_output[gout_off];
        if (x < 1e-7f) x = 1e-7f;
        if (x > 1.0f - 1e-7f) x = 1.0f - 1e-7f;
        grad_input[gin_off + id] = scale * go * (x - y) / (x * (1.0f - x));
    }
}
__kernel void cross_entropy_loss_forward(__global const float* input, int in_off,
                                         __global const float* target, int tgt_off,
                                         __global float* output, int out_off,
                                         int N, int C) {
    int i = get_global_id(0);
    if (i >= N) return;
    float max_val = input[in_off + i * C];
    for (int j = 1; j < C; ++j) {
        float val = input[in_off + i * C + j];
        if (val > max_val) max_val = val;
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < C; ++j) {
        sum_exp += exp(input[in_off + i * C + j] - max_val);
    }
    int target_idx = (int)target[tgt_off + i];
    float correct_logit = input[in_off + i * C + target_idx];
    float loss = -correct_logit + max_val + log(sum_exp);
    atomic_add_float(&output[out_off], loss);
}
__kernel void cross_entropy_loss_backward(__global const float* input, int in_off,
                                          __global const float* target, int tgt_off,
                                          __global const float* grad_output, int gout_off,
                                          __global float* grad_input, int gin_off,
                                          int N, int C) {
    int i = get_global_id(0);
    if (i >= N) return;

    float max_val = input[in_off + i * C];
    for (int j = 1; j < C; ++j) {
        float val = input[in_off + i * C + j];
        if (val > max_val) max_val = val;
    }

    float sum = 0.0f;
    for (int j = 0; j < C; ++j) {
        sum += exp(input[in_off + i * C + j] - max_val);
    }

    int target_idx = (int)target[tgt_off + i];
    float go = grad_output[gout_off];
    for (int j = 0; j < C; ++j) {
        float prob = exp(input[in_off + i * C + j] - max_val) / sum;
        float indicator = (j == target_idx) ? 1.0f : 0.0f;
        grad_input[gin_off + i * C + j] = (prob - indicator) / N * go;
    }
}
__kernel void sum_forward(__global const float* A, int a_off,
                          __global float* B, int b_off,
                          int size) {
    __local float sdata[256];
    int tid = get_local_id(0);
    int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
    float local_sum = 0.0f;
    while (idx < size) {
        local_sum += A[a_off + idx];
        idx += get_num_groups(0) * get_local_size(0);
    }
    sdata[tid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        atomic_add_float(&B[b_off], sdata[0]);
    }
}
__kernel void conv2d_backward_gb(__global const float* grad_output, int gout_off,
                                 __global float* grad_bias, int gb_off,
                                 int N, int C_out, int H_out, int W_out) {
    int co = get_global_id(0);
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
__kernel void conv2d_backward_gw(__global const float* input, int in_off,
                                 __global const float* grad_output, int gout_off,
                                 __global float* grad_weight, int gw_off,
                                 int N, int C_in, int H_in, int W_in,
                                 int C_out, int H_out, int W_out,
                                 int KH, int KW, int stride, int padding) {
    int idx = get_global_id(0);
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
__kernel void conv2d_backward_gdx(__global const float* grad_output, int gout_off,
                                  __global const float* weight, int w_off,
                                  __global float* grad_input, int gin_off,
                                  int N, int C_in, int H_in, int W_in,
                                  int C_out, int H_out, int W_out,
                                  int KH, int KW, int stride, int padding) {
    int idx = get_global_id(0);
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
__kernel void embedding_forward(__global const float* input, int in_off,
                                __global const float* weight, int w_off,
                                __global float* output, int out_off,
                                int num_indices, int num_embeddings, int embedding_dim) {
    int idx_thread = get_global_id(0);
    if (idx_thread >= num_indices * embedding_dim) return;

    int i = idx_thread / embedding_dim;
    int d = idx_thread % embedding_dim;

    int idx = (int)input[in_off + i];
    if (idx >= 0 && idx < num_embeddings) {
        output[out_off + i * embedding_dim + d] = weight[w_off + idx * embedding_dim + d];
    } else {
        output[out_off + i * embedding_dim + d] = 0.0f;
    }
}
__kernel void embedding_backward(__global const float* input, int in_off,
                                 __global const float* grad_output, int gout_off,
                                 __global float* grad_weight, int gw_off,
                                 int num_indices, int num_embeddings, int embedding_dim) {
    int idx_thread = get_global_id(0);
    if (idx_thread >= num_embeddings * embedding_dim) return;

    int idx = idx_thread / embedding_dim;
    int d = idx_thread % embedding_dim;

    float sum = 0.0f;
    for (int i = 0; i < num_indices; ++i) {
        int input_val = (int)input[in_off + i];
        if (input_val == idx) {
            sum += grad_output[gout_off + i * embedding_dim + d];
        }
    }
    grad_weight[gw_off + idx_thread] = sum;
}
__kernel void sum_backward(__global float* grad_input, int gin_off,
                           __global const float* grad_output, int gout_off,
                           int size) {
    int id = get_global_id(0);
    if (id < size) {
        grad_input[gin_off + id] = grad_output[gout_off];
    }
}
__kernel void reduce_broadcast_prepended(__global const float* A, int a_off,
                                         __global float* B, int b_off,
                                         int prod_prepended, int remaining) {
    int j = get_global_id(0);
    if (j < remaining) {
        float s = 0.0f;
        for (int i = 0; i < prod_prepended; ++i) {
            s += A[a_off + i * remaining + j];
        }
        B[b_off + j] = s;
    }
}
__kernel void reduce_broadcast_dim(__global const float* A, int a_off,
                                   __global float* B, int b_off,
                                   int outer_size, int dim_size, int inner_size) {
    int idx = get_global_id(0);
    int total = outer_size * inner_size;
    if (idx < total) {
        int o = idx / inner_size;
        int in = idx % inner_size;
        float s = 0.0f;
        for (int d = 0; d < dim_size; ++d) {
            s += A[a_off + (o * dim_size + d) * inner_size + in];
        }
        B[b_off + o * inner_size + in] = s;
    }
}
__kernel void sgd_step_kernel(__global float* P, int p_off,
                              __global const float* G, int g_off,
                              __global float* V, int v_off,
                              int has_momentum, float momentum,
                              float lr, float weight_decay, int size) {
    int id = get_global_id(0);
    if (id < size) {
        float grad_val = G[g_off + id];
        if (weight_decay != 0.0f) {
            grad_val += weight_decay * P[p_off + id];
        }
        if (has_momentum) {
            V[v_off + id] = momentum * V[v_off + id] + grad_val;
            P[p_off + id] -= lr * V[v_off + id];
        } else {
            P[p_off + id] -= lr * grad_val;
        }
    }
}
__kernel void adam_step_kernel(__global float* P, int p_off,
                               __global const float* G, int g_off,
                               __global float* M, int m_off,
                               __global float* V, int v_off,
                               float beta1, float beta2,
                               float lr, float eps, float weight_decay,
                               float bias_correction1, float bias_correction2, int size) {
    int id = get_global_id(0);
    if (id < size) {
        float grad_val = G[g_off + id];
        if (weight_decay != 0.0f) {
            grad_val += weight_decay * P[p_off + id];
        }
        M[m_off + id] = beta1 * M[m_off + id] + (1.0f - beta1) * grad_val;
        V[v_off + id] = beta2 * V[v_off + id] + (1.0f - beta2) * grad_val * grad_val;

        float m_hat = M[m_off + id] / bias_correction1;
        float v_hat = V[v_off + id] / bias_correction2;
        P[p_off + id] -= lr * m_hat / (sqrt(v_hat) + eps);
    }
}
__kernel void adamw_step_kernel(__global float* P, int p_off,
                                __global const float* G, int g_off,
                                __global float* M, int m_off,
                                __global float* V, int v_off,
                                float beta1, float beta2,
                                float lr, float eps, float weight_decay,
                                float bias_correction1, float bias_correction2, int size) {
    int id = get_global_id(0);
    if (id < size) {
        float grad_val = G[g_off + id];
        M[m_off + id] = beta1 * M[m_off + id] + (1.0f - beta1) * grad_val;
        V[v_off + id] = beta2 * V[v_off + id] + (1.0f - beta2) * grad_val * grad_val;

        float m_hat = M[m_off + id] / bias_correction1;
        float v_hat = V[v_off + id] / bias_correction2;
        float update = m_hat / (sqrt(v_hat) + eps);
        if (weight_decay != 0.0f) {
            P[p_off + id] -= lr * (weight_decay * P[p_off + id] + update);
        } else {
            P[p_off + id] -= lr * update;
        }
    }
}
__kernel void rmsprop_step_kernel(__global float* P, int p_off,
                                  __global const float* G, int g_off,
                                  __global float* SQ, int sq_off,
                                  float alpha, float lr, float eps, float weight_decay, int size) {
    int id = get_global_id(0);
    if (id < size) {
        float grad_val = G[g_off + id];
        if (weight_decay != 0.0f) {
            grad_val += weight_decay * P[p_off + id];
        }
        SQ[sq_off + id] = alpha * SQ[sq_off + id] + (1.0f - alpha) * grad_val * grad_val;
        P[p_off + id] -= lr * grad_val / (sqrt(SQ[sq_off + id]) + eps);
    }
}
__kernel void make_contiguous_kernel(
    __global const float* src, int src_offset,
    __global float* dst, int dst_offset,
    int ndims,
    int8 shape_val,
    int8 strides_val,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid >= total_elements) return;

    int shape[8] = { shape_val.s0, shape_val.s1, shape_val.s2, shape_val.s3, shape_val.s4, shape_val.s5, shape_val.s6, shape_val.s7 };
    int strides[8] = { strides_val.s0, strides_val.s1, strides_val.s2, strides_val.s3, strides_val.s4, strides_val.s5, strides_val.s6, strides_val.s7 };

    int src_idx = src_offset;
    int temp = gid;
    for (int d = ndims - 1; d >= 0; --d) {
        int coord = temp % shape[d];
        temp /= shape[d];
        src_idx += coord * strides[d];
    }
    dst[dst_offset + gid] = src[src_idx];
}
__kernel void copy_to_strided_kernel(
    __global const float* src, int src_offset,
    __global float* dst, int dst_offset,
    int ndims,
    int8 shape_val,
    int8 strides_val,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid >= total_elements) return;

    int shape[8] = { shape_val.s0, shape_val.s1, shape_val.s2, shape_val.s3, shape_val.s4, shape_val.s5, shape_val.s6, shape_val.s7 };
    int strides[8] = { strides_val.s0, strides_val.s1, strides_val.s2, strides_val.s3, strides_val.s4, strides_val.s5, strides_val.s6, strides_val.s7 };

    int dst_idx = dst_offset;
    int temp = gid;
    for (int d = ndims - 1; d >= 0; --d) {
        int coord = temp % shape[d];
        temp /= shape[d];
        dst_idx += coord * strides[d];
    }
    dst[dst_idx] = src[src_offset + gid];
}
__kernel void elementwise_broadcast_add(
    __global const float* A, int a_off,
    __global const float* B, int b_off,
    __global float* C, int c_off,
    int ndims,
    __global const int* out_shape,
    int a_ndims, __global const int* a_shape, __global const int* a_strides,
    int b_ndims, __global const int* b_shape, __global const int* b_strides,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid >= total_elements) return;

    int temp = gid;
    int coords[16];
    for (int d = ndims - 1; d >= 0; --d) {
        coords[d] = temp % out_shape[d];
        temp /= out_shape[d];
    }

    int a_idx = a_off;
    int a_diff = ndims - a_ndims;
    for (int i = 0; i < a_ndims; ++i) {
        if (a_shape[i] != 1) {
            a_idx += coords[i + a_diff] * a_strides[i];
        }
    }

    int b_idx = b_off;
    int b_diff = ndims - b_ndims;
    for (int i = 0; i < b_ndims; ++i) {
        if (b_shape[i] != 1) {
            b_idx += coords[i + b_diff] * b_strides[i];
        }
    }

    C[c_off + gid] = A[a_idx] + B[b_idx];
}
__kernel void elementwise_broadcast_sub(
    __global const float* A, int a_off,
    __global const float* B, int b_off,
    __global float* C, int c_off,
    int ndims,
    __global const int* out_shape,
    int a_ndims, __global const int* a_shape, __global const int* a_strides,
    int b_ndims, __global const int* b_shape, __global const int* b_strides,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid >= total_elements) return;

    int temp = gid;
    int coords[16];
    for (int d = ndims - 1; d >= 0; --d) {
        coords[d] = temp % out_shape[d];
        temp /= out_shape[d];
    }

    int a_idx = a_off;
    int a_diff = ndims - a_ndims;
    for (int i = 0; i < a_ndims; ++i) {
        if (a_shape[i] != 1) {
            a_idx += coords[i + a_diff] * a_strides[i];
        }
    }

    int b_idx = b_off;
    int b_diff = ndims - b_ndims;
    for (int i = 0; i < b_ndims; ++i) {
        if (b_shape[i] != 1) {
            b_idx += coords[i + b_diff] * b_strides[i];
        }
    }

    C[c_off + gid] = A[a_idx] - B[b_idx];
}
__kernel void elementwise_broadcast_mul(
    __global const float* A, int a_off,
    __global const float* B, int b_off,
    __global float* C, int c_off,
    int ndims,
    __global const int* out_shape,
    int a_ndims, __global const int* a_shape, __global const int* a_strides,
    int b_ndims, __global const int* b_shape, __global const int* b_strides,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid >= total_elements) return;

    int temp = gid;
    int coords[16];
    for (int d = ndims - 1; d >= 0; --d) {
        coords[d] = temp % out_shape[d];
        temp /= out_shape[d];
    }

    int a_idx = a_off;
    int a_diff = ndims - a_ndims;
    for (int i = 0; i < a_ndims; ++i) {
        if (a_shape[i] != 1) {
            a_idx += coords[i + a_diff] * a_strides[i];
        }
    }

    int b_idx = b_off;
    int b_diff = ndims - b_ndims;
    for (int i = 0; i < b_ndims; ++i) {
        if (b_shape[i] != 1) {
            b_idx += coords[i + b_diff] * b_strides[i];
        }
    }

    C[c_off + gid] = A[a_idx] * B[b_idx];
}
__kernel void elementwise_broadcast_div(
    __global const float* A, int a_off,
    __global const float* B, int b_off,
    __global float* C, int c_off,
    int ndims,
    __global const int* out_shape,
    int a_ndims, __global const int* a_shape, __global const int* a_strides,
    int b_ndims, __global const int* b_shape, __global const int* b_strides,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid >= total_elements) return;

    int temp = gid;
    int coords[16];
    for (int d = ndims - 1; d >= 0; --d) {
        coords[d] = temp % out_shape[d];
        temp /= out_shape[d];
    }

    int a_idx = a_off;
    int a_diff = ndims - a_ndims;
    for (int i = 0; i < a_ndims; ++i) {
        if (a_shape[i] != 1) {
            a_idx += coords[i + a_diff] * a_strides[i];
        }
    }

    int b_idx = b_off;
    int b_diff = ndims - b_ndims;
    for (int i = 0; i < b_ndims; ++i) {
        if (b_shape[i] != 1) {
            b_idx += coords[i + b_diff] * b_strides[i];
        }
    }

    C[c_off + gid] = A[a_idx] / B[b_idx];
}
__kernel void fill_zero(__global float* data, int size) {
    int id = get_global_id(0);
    if (id < size) {
        data[id] = 0.0f;
    }
}
__kernel void maxpool2d_backward_kernel(
    __global const float* indices, int ind_off,
    __global const float* grad_output, int gout_off,
    __global float* grad_input, int gin_off,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w)
{
    int idx = get_global_id(0);
    int total_threads = batch_size * channels * out_h * out_w;
    if (idx >= total_threads) return;

    int max_idx = (int)indices[ind_off + idx];
    if (max_idx >= 0) {
        atomic_add_float(&grad_input[gin_off + max_idx], grad_output[gout_off + idx]);
    }
}
__kernel void adaptive_avg_pool2d_forward_kernel(
    __global const float* input, int in_off,
    __global float* output, int out_off,
    int N, int C, int H, int W, int OH, int OW)
{
    int idx = get_global_id(0);
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
    output[out_off + idx] = sum / count;
}
__kernel void adaptive_avg_pool2d_backward_kernel(
    __global const float* grad_output, int gout_off,
    __global float* grad_input, int gin_off,
    int N, int C, int H, int W, int OH, int OW)
{
    int idx = get_global_id(0);
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
                    sum_grad += grad_output[gout_idx] / count;
                }
            }
        }
    }
    grad_input[gin_off + idx] = sum_grad;
}
__kernel void conv3d_kernel(
    __global const float* input, int in_off,
    __global const float* weight, int w_off,
    __global const float* bias, int b_off, int has_bias,
    __global float* output, int out_off,
    int batch_size, int in_channels, int in_d, int in_h, int in_w,
    int out_channels, int out_d, int out_h, int out_w,
    int kd, int kh, int kw, int stride, int padding)
{
    int idx = get_global_id(0);
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
    if (has_bias) {
        val += bias[b_off + c_out];
    }
    output[out_off + idx] = val;
}
__kernel void conv3d_backward_gb(
    __global const float* grad_output, int gout_off,
    __global float* grad_bias, int gb_off,
    int N, int C_out, int D_out, int H_out, int W_out)
{
    int co = get_global_id(0);
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
__kernel void conv3d_backward_gw(
    __global const float* input, int in_off,
    __global const float* grad_output, int gout_off,
    __global float* grad_weight, int gw_off,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int KD, int KH, int KW, int stride, int padding)
{
    int idx = get_global_id(0);
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
__kernel void conv3d_backward_gdx(
    __global const float* grad_output, int gout_off,
    __global const float* weight, int w_off,
    __global float* grad_input, int gin_off,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int KD, int KH, int KW, int stride, int padding)
{
    int idx = get_global_id(0);
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
__kernel void maxpool3d_kernel(
    __global const float* input, int in_off,
    __global float* output, int out_off,
    __global float* save_indices, int ind_off,
    int batch_size, int channels, int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w, int kernel_size, int stride, int padding)
{
    int idx = get_global_id(0);
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
    save_indices[ind_off + idx] = (float)max_idx;
}
__kernel void maxpool3d_backward_kernel(
    __global const float* save_indices, int ind_off,
    __global const float* grad_output, int gout_off,
    __global float* grad_input, int gin_off,
    int batch_size, int channels, int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w)
{
    int idx = get_global_id(0);
    int total_threads = batch_size * channels * out_d * out_h * out_w;
    if (idx >= total_threads) return;

    int max_idx = (int)save_indices[ind_off + idx];
    if (max_idx >= 0) {
        atomic_add_float(&grad_input[gin_off + max_idx], grad_output[gout_off + idx]);
    }
}
__kernel void im2col_kernel(
    __global const float* im, int im_off,
    int C, int H, int W,
    int KH, int KW, int padding, int stride,
    int H_out, int W_out,
    __global float* col, int col_off)
{
    int idx = get_global_id(0);
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
__kernel void add_bias_2d(
    __global float* out, int out_off,
    __global const float* bias, int b_off,
    int N, int C, int H, int W)
{
    int idx = get_global_id(0);
    int total = N * C * H * W;
    if (idx < total) {
        int c = (idx / (H * W)) % C;
        out[out_off + idx] += bias[b_off + c];
    }
}
__kernel void generate_dropout_mask(
    __global float* mask, int mask_off,
    float p, float scale, unsigned int seed, int size)
{
    int gid = get_global_id(0);
    if (gid < size) {
        unsigned int x = gid + seed;
        x = 1664525U * x + 1013904223U;
        x = 1664525U * x + 1013904223U;
        float r = (float)(x & 0xFFFFFFFF) / 4294967296.0f;
        mask[mask_off + gid] = (r >= p) ? scale : 0.0f;
    }
}
__kernel void fake_quantize_forward(
    __global const float* A, int a_off,
    float scale, float zero_point, float clip_min, float clip_max,
    __global float* B, int b_off, int size)
{
    int id = get_global_id(0);
    if (id < size) {
        float val = A[a_off + id] / scale + zero_point;
        float rounded = rint(val);
        if (rounded < clip_min) rounded = clip_min;
        if (rounded > clip_max) rounded = clip_max;
        B[b_off + id] = (rounded - zero_point) * scale;
    }
}

__kernel void flash_attention_forward(
    __global const float* Q, int q_off,
    __global const float* K, int k_off,
    __global const float* V, int v_off,
    __global float* O, int o_off,
    int B, int H, int H_kv, int Tq, int Tk, int D, float scale)
{
    int id = get_global_id(0);
    int total_rows = B * H * Tq;
    if (id >= total_rows) return;

    int b_idx = id / (H * Tq);
    int rem = id % (H * Tq);
    int h_idx = rem / Tq;
    int i_idx = rem % Tq;

    int g = H / H_kv;
    int h_kv = h_idx / g;

    int q_start = q_off + (b_idx * H + h_idx) * Tq * D + i_idx * D;
    int k_start_base = k_off + (b_idx * H_kv + h_kv) * Tk * D;
    int v_start_base = v_off + (b_idx * H_kv + h_kv) * Tk * D;
    int o_start = o_off + (b_idx * H + h_idx) * Tq * D + i_idx * D;

    float scores[2048];
    float max_val = -1e38f;
    for (int j = 0; j < Tk; ++j) {
        float dot = 0.0f;
        for (int k = 0; k < D; ++k) {
            dot += Q[q_start + k] * K[k_start_base + j * D + k];
        }
        dot *= scale;
        if (j < 2048) {
            scores[j] = dot;
        }
        if (dot > max_val) max_val = dot;
    }

    float denom = 0.0f;
    for (int j = 0; j < Tk; ++j) {
        float dot = (j < 2048) ? scores[j] : 0.0f;
        if (j >= 2048) {
            for (int k = 0; k < D; ++k) {
                dot += Q[q_start + k] * K[k_start_base + j * D + k];
            }
            dot *= scale;
        }
        float exp_val = exp(dot - max_val);
        denom += exp_val;
        if (j < 2048) {
            scores[j] = exp_val;
        }
    }

    for (int k = 0; k < D; ++k) {
        float acc = 0.0f;
        for (int j = 0; j < Tk; ++j) {
            float exp_val;
            if (j < 2048) {
                exp_val = scores[j];
            } else {
                float dot = 0.0f;
                for (int k_d = 0; k_d < D; ++k_d) {
                    dot += Q[q_start + k_d] * K[k_start_base + j * D + k_d];
                }
                dot *= scale;
                exp_val = exp(dot - max_val);
            }
            float weight = exp_val / denom;
            acc += weight * V[v_start_base + j * D + k];
        }
        O[o_start + k] = acc;
    }
}

__kernel void rope_forward(__global const float* X, int x_off,
                           __global const float* COS, int cos_off,
                           __global const float* SIN, int sin_off,
                           __global float* Y, int y_off,
                           int B, int H, int T, int D) {
    int id = get_global_id(0);
    int total = B * H * T * D;
    if (id >= total) return;

    int b = id / (H * T * D);
    int rem = id % (H * T * D);
    int h = rem / (T * D);
    int rem2 = rem % (T * D);
    int t = rem2 / D;
    int d = rem2 % D;

    int half_d = D / 2;
    int i = d / 2;

    float cos_val = COS[cos_off + t * half_d + i];
    float sin_val = SIN[sin_off + t * half_d + i];

    if (d % 2 == 0) {
        Y[y_off + id] = X[x_off + id] * cos_val - X[x_off + id + 1] * sin_val;
    } else {
        Y[y_off + id] = X[x_off + id] * cos_val + X[x_off + id - 1] * sin_val;
    }
}

__kernel void rope_backward(__global const float* DY, int dy_off,
                            __global const float* COS, int cos_off,
                            __global const float* SIN, int sin_off,
                            __global float* DX, int dx_off,
                            int B, int H, int T, int D) {
    int id = get_global_id(0);
    int total = B * H * T * D;
    if (id >= total) return;

    int b = id / (H * T * D);
    int rem = id % (H * T * D);
    int h = rem / (T * D);
    int rem2 = rem % (T * D);
    int t = rem2 / D;
    int d = rem2 % D;

    int half_d = D / 2;
    int i = d / 2;

    float cos_val = COS[cos_off + t * half_d + i];
    float sin_val = SIN[sin_off + t * half_d + i];

    if (d % 2 == 0) {
        DX[dx_off + id] = DY[dy_off + id] * cos_val + DY[dy_off + id + 1] * sin_val;
    } else {
        DX[dx_off + id] = DY[dy_off + id] * cos_val - DY[dy_off + id - 1] * sin_val;
    }
}

__kernel void paged_attention_forward(
    __global const float* Q, int q_off,
    __global const float* K_cache, int k_cache_off,
    __global const float* V_cache, int v_cache_off,
    __global const int* block_tables, int block_tables_off,
    __global const int* context_lens, int context_lens_off,
    __global float* O, int o_off,
    int num_seqs, int num_heads, int num_kv_heads, int head_dim,
    int max_num_blocks_per_seq, int block_size, float scale)
{
    int id = get_global_id(0);
    int total = num_seqs * num_heads;
    if (id >= total) return;

    int seq_idx = id / num_heads;
    int head_idx = id % num_heads;
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    int context_len = context_lens[context_lens_off + seq_idx];
    if (context_len <= 0) return;

    float max_val = -1e38f;
    for (int t = 0; t < context_len; ++t) {
        int block_idx = block_tables[block_tables_off + seq_idx * max_num_blocks_per_seq + t / block_size];
        int block_offset = t % block_size;
        int k_idx = k_cache_off + block_idx * (num_kv_heads * block_size * head_dim) + kv_head_idx * (block_size * head_dim) + block_offset * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += Q[q_off + seq_idx * (num_heads * head_dim) + head_idx * head_dim + d] * K_cache[k_idx + d];
        }
        dot *= scale;
        if (dot > max_val) max_val = dot;
    }

    float denominator = 0.0f;
    float acc[128];
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }

    for (int t = 0; t < context_len; ++t) {
        int block_idx = block_tables[block_tables_off + seq_idx * max_num_blocks_per_seq + t / block_size];
        int block_offset = t % block_size;
        int k_idx = k_cache_off + block_idx * (num_kv_heads * block_size * head_dim) + kv_head_idx * (block_size * head_dim) + block_offset * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += Q[q_off + seq_idx * (num_heads * head_dim) + head_idx * head_dim + d] * K_cache[k_idx + d];
        }
        dot *= scale;

        float exp_val = exp(dot - max_val);
        denominator += exp_val;

        int v_idx = v_cache_off + block_idx * (num_kv_heads * block_size * head_dim) + kv_head_idx * (block_size * head_dim) + block_offset * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            acc[d] += exp_val * V_cache[v_idx + d];
        }
    }

    for (int d = 0; d < head_dim; ++d) {
        O[o_off + seq_idx * (num_heads * head_dim) + head_idx * head_dim + d] = acc[d] / denominator;
    }
}

__kernel void w8a8_matmul_kernel(
    __global const float* A, int a_off,
    __global const float* B, int b_off,
    __global float* C, int c_off,
    int M, int K, int N,
    float x_scale, float w_scale
) {
    __local float As[16][16];
    __local float Bs[16][16];
    int row = get_global_id(0);
    int col = get_global_id(1);
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);
    float sum = 0.0f;
    int numTiles = (K + 15) / 16;
    for (int t = 0; t < numTiles; ++t) {
        int tiledCol = t * 16 + localCol;
        if (row < M && tiledCol < K) {
            float val = A[a_off + row * K + tiledCol];
            float q = rint(val / x_scale);
            As[localRow][localCol] = clamp(q, -128.0f, 127.0f);
        } else {
            As[localRow][localCol] = 0.0f;
        }
        int tiledRow = t * 16 + localRow;
        if (tiledRow < K && col < N) {
            float val = B[b_off + col * K + tiledRow];
            float q = rint(val / w_scale);
            Bs[localRow][localCol] = clamp(q, -128.0f, 127.0f);
        } else {
            Bs[localRow][localCol] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 16; ++k) {
            sum += As[localRow][k] * Bs[k][localCol];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) {
        C[c_off + row * N + col] = sum * (x_scale * w_scale);
    }
}

inline ushort float_to_half(float f) {
    uint x = as_uint(f);
    uint sign = (x >> 16) & 0x8000;
    uint exponent = (x >> 23) & 0xff;
    uint mantissa = x & 0x7fffff;
    if (exponent == 0) {
        return (ushort)sign;
    } else if (exponent == 255) {
        return (ushort)(sign | 0x7c00 | (mantissa ? 0x200 : 0));
    } else {
        int new_exp = (int)exponent - 127 + 15;
        if (new_exp >= 31) {
            return (ushort)(sign | 0x7c00);
        } else if (new_exp <= 0) {
            return (ushort)sign;
        }
        return (ushort)(sign | (new_exp << 10) | (mantissa >> 13));
    }
}

inline float half_to_float(ushort h) {
    uint sign = (h & 0x8000) << 16;
    uint exponent = (h & 0x7c00) >> 10;
    uint mantissa = h & 0x03ff;
    uint val = 0;
    if (exponent == 0) {
        if (mantissa != 0) {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03ff;
            val = sign | (exponent << 23) | (mantissa << 13);
        } else {
            val = sign;
        }
    } else if (exponent == 31) {
        val = sign | (0xff << 23) | (mantissa << 13);
    } else {
        val = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    return as_float(val);
}

inline ushort float_to_bf16(float f) {
    uint val = as_uint(f);
    return (ushort)(val >> 16);
}

inline float bf16_to_float(ushort h) {
    uint val = ((uint)h) << 16;
    return as_float(val);
}

__constant float NF4_GRID[16] = {
    -1.0f, -0.6961917f, -0.525073f, -0.3930782f, -0.2753147f, -0.1651313f, -0.0596338f, 0.0f,
    0.0596338f, 0.1651313f, 0.2753147f, 0.3930782f, 0.525073f, 0.6961917f, 1.0f, 1.0f
};

inline uchar float_to_nf4(float val) {
    float min_dist = 1e9f;
    uchar best_idx = 0;
    for (uchar i = 0; i < 16; ++i) {
        float dist = fabs(NF4_GRID[i] - val);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

inline float nf4_to_float(uchar idx) {
    uchar i = idx;
    if (i >= 16) i = 15;
    return NF4_GRID[i];
}

inline uchar float_to_fp8_e4m3(float val) {
    if (isnan(val)) return 0x7F;
    uint ui = as_uint(val);
    uint sign = (ui >> 31) & 1;
    uint exp = (ui >> 23) & 0xFF;
    uint mant = ui & 0x7FFFFF;
    if (exp == 0) return (uchar)(sign << 7);
    if (exp == 0xFF) return (uchar)((sign << 7) | 0x7F);
    int new_exp = (int)exp - 127 + 7;
    if (new_exp <= 0) {
        int shift = 1 - new_exp;
        if (shift > 3) return (uchar)(sign << 7);
        uint m = (1 << 3) | (mant >> 20);
        m >>= shift;
        return (uchar)((sign << 7) | m);
    } else if (new_exp >= 15) {
        return (uchar)((sign << 7) | 0x7E);
    }
    uint m = mant >> 20;
    return (uchar)((sign << 7) | (new_exp << 3) | m);
}

inline float fp8_e4m3_to_float(uchar val) {
    uint sign = (val >> 7) & 1;
    uint exp = (val >> 3) & 0x0F;
    uint mant = val & 0x07;
    if (exp == 15) return NAN;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        return (sign ? -1.0f : 1.0f) * pow(2.0f, -6.0f) * ((float)mant / 8.0f);
    }
    return (sign ? -1.0f : 1.0f) * pow(2.0f, (float)exp - 7.0f) * (1.0f + (float)mant / 8.0f);
}

inline uchar float_to_fp8_e5m2(float val) {
    if (isnan(val)) return 0x7F;
    uint ui = as_uint(val);
    uint sign = (ui >> 31) & 1;
    uint exp = (ui >> 23) & 0xFF;
    uint mant = ui & 0x7FFFFF;
    if (exp == 0) return (uchar)(sign << 7);
    if (exp == 0xFF) return (uchar)((sign << 7) | 0x7F);
    int new_exp = (int)exp - 127 + 15;
    if (new_exp <= 0) {
        int shift = 1 - new_exp;
        if (shift > 2) return (uchar)(sign << 7);
        uint m = (1 << 2) | (mant >> 21);
        m >>= shift;
        return (uchar)((sign << 7) | m);
    } else if (new_exp >= 31) {
        return (uchar)((sign << 7) | 0x7E);
    }
    uint m = mant >> 21;
    return (uchar)((sign << 7) | (new_exp << 2) | m);
}

inline float fp8_e5m2_to_float(uchar val) {
    uint sign = (val >> 7) & 1;
    uint exp = (val >> 2) & 0x1F;
    uint mant = val & 0x03;
    if (exp == 31) return NAN;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        return (sign ? -1.0f : 1.0f) * pow(2.0f, -14.0f) * ((float)mant / 4.0f);
    }
    return (sign ? -1.0f : 1.0f) * pow(2.0f, (float)exp - 15.0f) * (1.0f + (float)mant / 4.0f);
}

__kernel void cast_fp32_to_fp16(__global const float* src, int src_off, __global ushort* dst, int dst_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        dst[dst_off + id] = float_to_half(src[src_off + id]);
    }
}

__kernel void cast_fp16_to_fp32(__global const ushort* src, int src_off, __global float* dst, int dst_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        dst[dst_off + id] = half_to_float(src[src_off + id]);
    }
}

__kernel void cast_fp32_to_bf16(__global const float* src, int src_off, __global ushort* dst, int dst_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        dst[dst_off + id] = float_to_bf16(src[src_off + id]);
    }
}

__kernel void cast_bf16_to_fp32(__global const ushort* src, int src_off, __global float* dst, int dst_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        dst[dst_off + id] = bf16_to_float(src[src_off + id]);
    }
}

__kernel void cast_fp32_to_nf4(__global const float* src, int src_off, __global uchar* dst, int dst_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        dst[dst_off + id] = float_to_nf4(src[src_off + id]);
    }
}

__kernel void cast_nf4_to_fp32(__global const uchar* src, int src_off, __global float* dst, int dst_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        dst[dst_off + id] = nf4_to_float(src[src_off + id]);
    }
}

__kernel void cast_fp32_to_int8(__global const float* src, int src_off, __global char* dst, int dst_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        dst[dst_off + id] = (char)clamp(src[src_off + id], -128.0f, 127.0f);
    }
}

__kernel void cast_int8_to_fp32(__global const char* src, int src_off, __global float* dst, int dst_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        dst[dst_off + id] = (float)src[src_off + id];
    }
}

__kernel void cast_fp32_to_int4(__global const float* src, int src_off, __global char* dst, int dst_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        dst[dst_off + id] = (char)clamp(src[src_off + id], -8.0f, 7.0f);
    }
}

__kernel void cast_int4_to_fp32(__global const char* src, int src_off, __global float* dst, int dst_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        dst[dst_off + id] = (float)src[src_off + id];
    }
}

__kernel void cast_fp32_to_fp8_e4m3(__global const float* src, int src_off, __global uchar* dst, int dst_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        dst[dst_off + id] = float_to_fp8_e4m3(src[src_off + id]);
    }
}

__kernel void cast_fp8_e4m3_to_fp32(__global const uchar* src, int src_off, __global float* dst, int dst_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        dst[dst_off + id] = fp8_e4m3_to_float(src[src_off + id]);
    }
}

__kernel void cast_fp32_to_fp8_e5m2(__global const float* src, int src_off, __global uchar* dst, int dst_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        dst[dst_off + id] = float_to_fp8_e5m2(src[src_off + id]);
    }
}

__kernel void cast_fp8_e5m2_to_fp32(__global const uchar* src, int src_off, __global float* dst, int dst_off, int size) {
    int id = get_global_id(0);
    if (id < size) {
        dst[dst_off + id] = fp8_e5m2_to_float(src[src_off + id]);
    }
}





)litetorch_raw_cl";

}
