#include "gpu_common.h"

extern "C" __global__ void adamw_step_kernel_native(
    float* P, int p_off,
    const float* G, int g_off,
    float* M, int m_off,
    float* V, int v_off,
    int size, float lr_t, float beta1, float beta2, float eps, float weight_decay)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float p = P[p_off + idx];
        float g = G[g_off + idx] + weight_decay * p;
        float m = beta1 * M[m_off + idx] + (1.0f - beta1) * g;
        float v = beta2 * V[v_off + idx] + (1.0f - beta2) * g * g;
        M[m_off + idx] = m;
        V[v_off + idx] = v;
        P[p_off + idx] = p - lr_t * m / (sqrtf(v) + eps);
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
