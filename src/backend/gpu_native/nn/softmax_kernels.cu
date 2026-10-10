#include "gpu_common.h"

#ifndef __HIP_PLATFORM_AMD__
extern "C" void gpu_softmax_cudnn(const float* input, int in_off, float* output, int out_off, int N, int C, int H, int W) {
    auto_set_device(input);
    lt_cudnnHandle_t handle = get_cudnn_handle(output);
    if (!handle) return;
    lt_cudnnTensorDescriptor_t srcDesc, dstDesc;
    g_cudnn.CreateTensorDescriptor(&srcDesc);
    g_cudnn.CreateTensorDescriptor(&dstDesc);
    g_cudnn.SetTensor4dDescriptor(srcDesc, LT_CUDNN_TENSOR_NCHW, LT_CUDNN_DATA_FLOAT, N, C, H, W);
    g_cudnn.SetTensor4dDescriptor(dstDesc, LT_CUDNN_TENSOR_NCHW, LT_CUDNN_DATA_FLOAT, N, C, H, W);
    float alpha = 1.0f, beta = 0.0f;
    g_cudnn.SoftmaxForward(handle, LT_CUDNN_SOFTMAX_ACCURATE, LT_CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, srcDesc, input + in_off, &beta, dstDesc, output + out_off);
    g_cudnn.DestroyTensorDescriptor(srcDesc);
    g_cudnn.DestroyTensorDescriptor(dstDesc);
}
#endif

#ifdef USE_MIOPEN
extern "C" void gpu_softmax_miopen(const float* input, int in_off, float* output, int out_off, int N, int C, int H, int W) {
    auto_set_device(input);
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

extern "C" __global__ void softmax_fast_kernel(const float* A, int a_off,
                                              float* B, int b_off,
                                              int dim_size, int outer_size) {
    int row = blockIdx.x;
    if (row >= outer_size) return;
    int tid = threadIdx.x;
    const float* a_row = A + a_off + (int64_t)row * dim_size;
    float* b_row = B + b_off + (int64_t)row * dim_size;
    bool aligned = ((reinterpret_cast<uintptr_t>(a_row) & 15) == 0) &&
                   ((reinterpret_cast<uintptr_t>(b_row) & 15) == 0);

    __shared__ float sdata[256];

    float tmax = -3.4028235e38f;
    int vec_n = dim_size >> 2;
    if (aligned) {
        const float4* a4 = reinterpret_cast<const float4*>(a_row);
        for (int i = tid; i < vec_n; i += 256) {
            float4 v = a4[i];
            float m = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
            tmax = fmaxf(tmax, m);
        }
    }
    for (int i = (aligned ? (vec_n << 2) : 0) + tid; i < dim_size; i += 256) {
        tmax = fmaxf(tmax, a_row[i]);
    }
    sdata[tid] = tmax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float tsum = 0.0f;
    if (aligned) {
        const float4* a4 = reinterpret_cast<const float4*>(a_row);
        float4* b4 = reinterpret_cast<float4*>(b_row);
        for (int i = tid; i < vec_n; i += 256) {
            float4 v = a4[i];
            float4 e;
            e.x = expf(v.x - row_max); e.y = expf(v.y - row_max);
            e.z = expf(v.z - row_max); e.w = expf(v.w - row_max);
            b4[i] = e;
            tsum += e.x + e.y + e.z + e.w;
        }
    }
    for (int i = (aligned ? (vec_n << 2) : 0) + tid; i < dim_size; i += 256) {
        float e = expf(a_row[i] - row_max);
        b_row[i] = e;
        tsum += e;
    }
    sdata[tid] = tsum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / (sdata[0] + 1e-20f);
    if (aligned) {
        float4* b4 = reinterpret_cast<float4*>(b_row);
        for (int i = tid; i < vec_n; i += 256) {
            float4 e = b4[i];
            e.x *= inv_sum; e.y *= inv_sum; e.z *= inv_sum; e.w *= inv_sum;
            b4[i] = e;
        }
    }
    for (int i = (aligned ? (vec_n << 2) : 0) + tid; i < dim_size; i += 256) {
        b_row[i] *= inv_sum;
    }
}
