#include "gpu_common.h"

extern "C" __global__ void adamw_step_kernel_native(
    float* P, int p_off,
    const float* G, int g_off,
    float* M, int m_off,
    float* V, int v_off,
    int size, float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float p = P[p_off + idx];
        float g = G[g_off + idx];
        float m = beta1 * M[m_off + idx] + (1.0f - beta1) * g;
        float v = beta2 * V[v_off + idx] + (1.0f - beta2) * g * g;
        M[m_off + idx] = m;
        V[v_off + idx] = v;
        float m_hat = m / bias_correction1;
        float v_hat = v / bias_correction2;
        float update = m_hat / (sqrtf(v_hat) + eps);
        // Decoupled weight decay (AdamW), matching the CPU/TPU implementation.
        if (weight_decay != 0.0f) {
            P[p_off + idx] = p - lr * (weight_decay * p + update);
        } else {
            P[p_off + idx] = p - lr * update;
        }
    }
}

extern "C" __global__ void adam_step_kernel(
    float* P, int p_off,
    const float* G, int g_off,
    float* M, int m_off,
    float* V, int v_off,
    float beta1, float beta2,
    float lr, float eps, float weight_decay,
    float bias_correction1, float bias_correction2, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float p = P[p_off + idx];
        float g = G[g_off + idx];
        if (weight_decay != 0.0f) {
            g += weight_decay * p;
        }
        float m = beta1 * M[m_off + idx] + (1.0f - beta1) * g;
        float v = beta2 * V[v_off + idx] + (1.0f - beta2) * g * g;
        M[m_off + idx] = m;
        V[v_off + idx] = v;

        float m_hat = m / bias_correction1;
        float v_hat = v / bias_correction2;
        P[p_off + idx] = p - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

extern "C" __global__ void adamw_step_kernel(
    float* P, int p_off,
    const float* G, int g_off,
    float* M, int m_off,
    float* V, int v_off,
    float beta1, float beta2,
    float lr, float eps, float weight_decay,
    float bias_correction1, float bias_correction2, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float p = P[p_off + idx];
        float g = G[g_off + idx];
        float m = beta1 * M[m_off + idx] + (1.0f - beta1) * g;
        float v = beta2 * V[v_off + idx] + (1.0f - beta2) * g * g;
        M[m_off + idx] = m;
        V[v_off + idx] = v;

        float m_hat = m / bias_correction1;
        float v_hat = v / bias_correction2;
        float update = m_hat / (sqrtf(v_hat) + eps);
        if (weight_decay != 0.0f) {
            P[p_off + idx] = p - lr * (weight_decay * p + update);
        } else {
            P[p_off + idx] = p - lr * update;
        }
    }
}

extern "C" void gpu_adamw_step(void* P, int p_off, void* G, int g_off, void* M_state, int m_off, void* V, int v_off, int size, float lr, float beta1, float beta2, float eps, float weight_decay, float bias_correction1, float bias_correction2) {
    int blocks = (size + 255) / 256;
#ifndef __HIP_PLATFORM_AMD__
    adamw_step_kernel<<<blocks, 256, 0, g_compute_stream>>>((float*)P, p_off, (const float*)G, g_off, (float*)M_state, m_off, (float*)V, v_off, beta1, beta2, lr, eps, weight_decay, bias_correction1, bias_correction2, size);
#else
    hipLaunchKernelGGL(adamw_step_kernel, dim3(blocks), dim3(256), 0, g_compute_stream, (float*)P, p_off, (const float*)G, g_off, (float*)M_state, m_off, (float*)V, v_off, beta1, beta2, lr, eps, weight_decay, bias_correction1, bias_correction2, size);
#endif
}

extern "C" __global__ void sgd_step_kernel(
    float* P, int p_off,
    const float* G, int g_off,
    float* V, int v_off,
    int has_momentum, float momentum,
    float lr, float weight_decay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad_val = G[g_off + idx];
        if (weight_decay != 0.0f) {
            grad_val += weight_decay * P[p_off + idx];
        }
        if (has_momentum) {
            V[v_off + idx] = momentum * V[v_off + idx] + grad_val;
            P[p_off + idx] -= lr * V[v_off + idx];
        } else {
            P[p_off + idx] -= lr * grad_val;
        }
    }
}

extern "C" __global__ void rmsprop_step_kernel(
    float* P, int p_off,
    const float* G, int g_off,
    float* SQ, int sq_off,
    float alpha, float lr, float eps, float weight_decay, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad_val = G[g_off + idx];
        if (weight_decay != 0.0f) {
            grad_val += weight_decay * P[p_off + idx];
        }
        SQ[sq_off + idx] = alpha * SQ[sq_off + idx] + (1.0f - alpha) * grad_val * grad_val;
        P[p_off + idx] -= lr * grad_val / (sqrtf(SQ[sq_off + idx]) + eps);
    }
}

__global__ void adam_foreach_kernel(
    float** P_list, int* p_offs,
    float** G_list, int* g_offs,
    float** M_list, int* m_offs,
    float** V_list, int* v_offs,
    int* sizes, int n_tensors,
    float beta1, float beta2,
    float lr, float eps, float weight_decay,
    float bias_correction1, float bias_correction2)
{
    int t = blockIdx.y;
    if (t >= n_tensors) return;
    float* P = P_list[t];
    const float* G = G_list[t];
    float* M = M_list[t];
    float* V = V_list[t];
    int p_off = p_offs[t];
    int g_off = g_offs[t];
    int m_off = m_offs[t];
    int v_off = v_offs[t];
    int size = sizes[t];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float p = P[p_off + idx];
        float g = G[g_off + idx];
        if (weight_decay != 0.0f) {
            g += weight_decay * p;
        }
        float m = beta1 * M[m_off + idx] + (1.0f - beta1) * g;
        float v = beta2 * V[v_off + idx] + (1.0f - beta2) * g * g;
        M[m_off + idx] = m;
        V[v_off + idx] = v;
        float m_hat = m / bias_correction1;
        float v_hat = v / bias_correction2;
        P[p_off + idx] = p - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

static float** g_adam_P_list = nullptr;
static int* g_adam_p_offs = nullptr;
static float** g_adam_G_list = nullptr;
static int* g_adam_g_offs = nullptr;
static float** g_adam_M_list = nullptr;
static int* g_adam_m_offs = nullptr;
static float** g_adam_V_list = nullptr;
static int* g_adam_v_offs = nullptr;
static int* g_adam_sizes = nullptr;
static int g_adam_capacity = 0;

static void ensure_adam_foreach_buffers(int n) {
    if (n > g_adam_capacity) {
        if (g_adam_P_list) GPU_API(Free)(g_adam_P_list);
        if (g_adam_p_offs) GPU_API(Free)(g_adam_p_offs);
        if (g_adam_G_list) GPU_API(Free)(g_adam_G_list);
        if (g_adam_g_offs) GPU_API(Free)(g_adam_g_offs);
        if (g_adam_M_list) GPU_API(Free)(g_adam_M_list);
        if (g_adam_m_offs) GPU_API(Free)(g_adam_m_offs);
        if (g_adam_V_list) GPU_API(Free)(g_adam_V_list);
        if (g_adam_v_offs) GPU_API(Free)(g_adam_v_offs);
        if (g_adam_sizes) GPU_API(Free)(g_adam_sizes);
        int cap = n * 2;
        GPU_API(Malloc)((void**)&g_adam_P_list, cap * sizeof(float*));
        GPU_API(Malloc)((void**)&g_adam_p_offs, cap * sizeof(int));
        GPU_API(Malloc)((void**)&g_adam_G_list, cap * sizeof(float*));
        GPU_API(Malloc)((void**)&g_adam_g_offs, cap * sizeof(int));
        GPU_API(Malloc)((void**)&g_adam_M_list, cap * sizeof(float*));
        GPU_API(Malloc)((void**)&g_adam_m_offs, cap * sizeof(int));
        GPU_API(Malloc)((void**)&g_adam_V_list, cap * sizeof(float*));
        GPU_API(Malloc)((void**)&g_adam_v_offs, cap * sizeof(int));
        GPU_API(Malloc)((void**)&g_adam_sizes, cap * sizeof(int));
        g_adam_capacity = cap;
    }
}

extern "C" void gpu_adam_foreach(
    void** h_P_list, int* h_p_offs,
    void** h_G_list, int* h_g_offs,
    void** h_M_list, int* h_m_offs,
    void** h_V_list, int* h_v_offs,
    int* h_sizes, int n_tensors,
    float beta1, float beta2, float lr, float eps, float weight_decay,
    float bias_correction1, float bias_correction2, int max_size)
{
    if (n_tensors <= 0) return;
    ensure_adam_foreach_buffers(n_tensors);
    GPU_API(Memcpy)(g_adam_P_list, h_P_list, n_tensors * sizeof(float*), GPU_API(MemcpyHostToDevice));
    GPU_API(Memcpy)(g_adam_p_offs, h_p_offs, n_tensors * sizeof(int), GPU_API(MemcpyHostToDevice));
    GPU_API(Memcpy)(g_adam_G_list, h_G_list, n_tensors * sizeof(float*), GPU_API(MemcpyHostToDevice));
    GPU_API(Memcpy)(g_adam_g_offs, h_g_offs, n_tensors * sizeof(int), GPU_API(MemcpyHostToDevice));
    GPU_API(Memcpy)(g_adam_M_list, h_M_list, n_tensors * sizeof(float*), GPU_API(MemcpyHostToDevice));
    GPU_API(Memcpy)(g_adam_m_offs, h_m_offs, n_tensors * sizeof(int), GPU_API(MemcpyHostToDevice));
    GPU_API(Memcpy)(g_adam_V_list, h_V_list, n_tensors * sizeof(float*), GPU_API(MemcpyHostToDevice));
    GPU_API(Memcpy)(g_adam_v_offs, h_v_offs, n_tensors * sizeof(int), GPU_API(MemcpyHostToDevice));
    GPU_API(Memcpy)(g_adam_sizes, h_sizes, n_tensors * sizeof(int), GPU_API(MemcpyHostToDevice));
    dim3 block(256, 1, 1);
    dim3 grid((max_size + 255) / 256, n_tensors, 1);
#ifndef __HIP_PLATFORM_AMD__
    adam_foreach_kernel<<<grid, block, 0, g_compute_stream>>>(
        g_adam_P_list, g_adam_p_offs, g_adam_G_list, g_adam_g_offs,
        g_adam_M_list, g_adam_m_offs, g_adam_V_list, g_adam_v_offs,
        g_adam_sizes, n_tensors, beta1, beta2, lr, eps, weight_decay,
        bias_correction1, bias_correction2);
#else
    hipLaunchKernelGGL(adam_foreach_kernel, grid, block, 0, g_compute_stream,
        g_adam_P_list, g_adam_p_offs, g_adam_G_list, g_adam_g_offs,
        g_adam_M_list, g_adam_m_offs, g_adam_V_list, g_adam_v_offs,
        g_adam_sizes, n_tensors, beta1, beta2, lr, eps, weight_decay,
        bias_correction1, bias_correction2);
#endif
}
