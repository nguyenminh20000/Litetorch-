#include "gpu_common.h"

extern "C" __global__ void relu_inplace_kernel(float* data, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && data[idx] < 0.0f) data[idx] = 0.0f;
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_indices) return;
    int idx = static_cast<int>(input[in_off + i]);
    if (idx < 0 || idx >= num_embeddings) return;
    for (int d = 0; d < embedding_dim; ++d) {
        atomicAdd(&grad_weight[gw_off + idx * embedding_dim + d],
                  grad_output[gout_off + i * embedding_dim + d]);
    }
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
    const int* bt_ptr, int bt_off,
    const int* cl_ptr, int cl_off,
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
    int context_len = cl_ptr[cl_off + seq_idx];
    if (context_len <= 0) return;

    float scores[2048];
    float max_val = -1e37f;

    for (int t = 0; t < context_len && t < 2048; ++t) {
        int block_idx = bt_ptr[bt_off + seq_idx * max_num_blocks + t / block_size];
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

        int block_idx = bt_ptr[bt_off + seq_idx * max_num_blocks + t / block_size];
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
    auto_set_device(logits);
    if (N <= 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
#ifndef __HIP_PLATFORM_AMD__
    moe_gate_kernel<<<blocks, threads, 0, dev_stream(current_device())>>>((const float*)logits, l_off, (float*)probs, p_off, (float*)indices, idx_off, N, E, top_k);
#else
    hipLaunchKernelGGL(moe_gate_kernel, dim3(blocks), dim3(threads), 0, dev_stream(current_device()), (const float*)logits, l_off, (float*)probs, p_off, (float*)indices, idx_off, N, E, top_k);
#endif
}

extern "C" void gpu_moe_gate_backward(void* grad_output, int gout_off, void* input, int in_off, void* gate_weight, int gw_off, void* probs, int p_off, void* indices, int idx_off, void* grad_input, int gin_off, void* grad_gate_weight, int ggw_off, int N, int D, int E, int top_k) {
    auto_set_device(grad_output);
    if (N <= 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
#ifndef __HIP_PLATFORM_AMD__
    moe_gate_backward_kernel<<<blocks, threads, 0, dev_stream(current_device())>>>((const float*)grad_output, gout_off, (const float*)input, in_off, (const float*)gate_weight, gw_off, (const float*)probs, p_off, (const float*)indices, idx_off, (float*)grad_input, gin_off, (float*)grad_gate_weight, ggw_off, N, D, E, top_k);
#else
    hipLaunchKernelGGL(moe_gate_backward_kernel, dim3(blocks), dim3(threads), 0, dev_stream(current_device()), (const float*)grad_output, gout_off, (const float*)input, in_off, (const float*)gate_weight, gw_off, (const float*)probs, p_off, (const float*)indices, idx_off, (float*)grad_input, gin_off, (float*)grad_gate_weight, ggw_off, N, D, E, top_k);
#endif
}

extern "C" void gpu_moe_expert_forward(void* input, int in_off, void* expert_weight, int ew_off, void* expert_bias, int eb_off, void* probs, int p_off, void* indices, int idx_off, void* output, int out_off, int N, int D, int out_features, int expert_idx, int top_k) {
    auto_set_device(input);
    if (N <= 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
#ifndef __HIP_PLATFORM_AMD__
    moe_expert_forward_kernel<<<blocks, threads, 0, dev_stream(current_device())>>>((const float*)input, in_off, (const float*)expert_weight, ew_off, (const float*)expert_bias, eb_off, (const float*)probs, p_off, (const float*)indices, idx_off, (float*)output, out_off, N, D, out_features, expert_idx, top_k);
#else
    hipLaunchKernelGGL(moe_expert_forward_kernel, dim3(blocks), dim3(threads), 0, dev_stream(current_device()), (const float*)input, in_off, (const float*)expert_weight, ew_off, (const float*)expert_bias, eb_off, (const float*)probs, p_off, (const float*)indices, idx_off, (float*)output, out_off, N, D, out_features, expert_idx, top_k);
#endif
}

extern "C" void gpu_moe_expert_backward(void* grad_output, int gout_off, void* input, int in_off, void* expert_weight, int ew_off, void* expert_bias, int eb_off, void* probs, int p_off, void* indices, int idx_off, void* grad_input, int gin_off, void* grad_expert, int ge_off, void* grad_bias, int gb_off, void* grad_probs, int gp_off, int N, int D, int out_features, int expert_idx, int top_k) {
    auto_set_device(grad_output);
    if (N <= 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
#ifndef __HIP_PLATFORM_AMD__
    moe_expert_backward_kernel<<<blocks, threads, 0, dev_stream(current_device())>>>((const float*)grad_output, gout_off, (const float*)input, in_off, (const float*)expert_weight, ew_off, (const float*)expert_bias, eb_off, (const float*)probs, p_off, (const float*)indices, idx_off, (float*)grad_input, gin_off, (float*)grad_expert, ge_off, (float*)grad_bias, gb_off, (float*)grad_probs, gp_off, N, D, out_features, expert_idx, top_k);
#else
    hipLaunchKernelGGL(moe_expert_backward_kernel, dim3(blocks), dim3(threads), 0, dev_stream(current_device()), (const float*)grad_output, gout_off, (const float*)input, in_off, (const float*)expert_weight, ew_off, (const float*)expert_bias, eb_off, (const float*)probs, p_off, (const float*)indices, idx_off, (float*)grad_input, gin_off, (float*)grad_expert, ge_off, (float*)grad_bias, gb_off, (float*)grad_probs, gp_off, N, D, out_features, expert_idx, top_k);
#endif
}
