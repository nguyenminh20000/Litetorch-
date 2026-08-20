#include "gpu_common.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#ifndef __HIP_PLATFORM_AMD__
__device__ __forceinline__ __half atomicAdd(__half *address, __half val) {
  unsigned int *address_as_ui =
      (unsigned int *)((char *)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  do {
    assumed = old;
    unsigned short old_val = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    __half old_h = __ushort_as_half(old_val);
    __half new_h = __float2half(__half2float(old_h) + __half2float(val));
    unsigned int new_val =
        (size_t)address & 2 ? (old & 0xffff) | (__half_as_ushort(new_h) << 16)
                            : (old & 0xffff0000) | __half_as_ushort(new_h);
    old = atomicCAS(address_as_ui, assumed, new_val);
  } while (assumed != old);
  return __ushort_as_half((size_t)address & 2 ? (old >> 16) : (old & 0xffff));
}
#endif
#endif

__global__ void fused_softmax_fwd_kernel(const float* S, float* P, int M, int Tk) {
    int row = blockIdx.x;
    if (row >= M) return;
    const float* s_row = S + row * Tk;
    float* p_row = P + row * Tk;
    int tid = threadIdx.x;

    float max_val = -1e37f;
    for (int i = tid; i < Tk; i += blockDim.x) {
        float val = s_row[i];
        if (val > max_val) max_val = val;
    }
#ifndef __HIP_PLATFORM_AMD__
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
#endif
    __shared__ float s_max[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) s_max[wid] = max_val;
    __syncthreads();
    if (wid == 0) {
        float b_max = (lane < (blockDim.x + 31) / 32) ? s_max[lane] : -1e37f;
#ifndef __HIP_PLATFORM_AMD__
        for (int offset = 16; offset > 0; offset /= 2) {
            b_max = fmaxf(b_max, __shfl_down_sync(0xffffffff, b_max, offset));
        }
#endif
        if (lane == 0) s_max[0] = b_max;
    }
    __syncthreads();
    float row_max = s_max[0];

    float sum_exp = 0.0f;
    for (int i = tid; i < Tk; i += blockDim.x) {
        float e = expf(s_row[i] - row_max);
        p_row[i] = e;
        sum_exp += e;
    }
#ifndef __HIP_PLATFORM_AMD__
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
#endif
    if (lane == 0) s_max[wid] = sum_exp;
    __syncthreads();
    if (wid == 0) {
        float b_sum = (lane < (blockDim.x + 31) / 32) ? s_max[lane] : 0.0f;
#ifndef __HIP_PLATFORM_AMD__
        for (int offset = 16; offset > 0; offset /= 2) {
            b_sum += __shfl_down_sync(0xffffffff, b_sum, offset);
        }
#endif
        if (lane == 0) s_max[0] = b_sum;
    }
    __syncthreads();
    float inv_sum = 1.0f / (s_max[0] + 1e-6f);

    for (int i = tid; i < Tk; i += blockDim.x) {
        p_row[i] *= inv_sum;
    }
}

__global__ void fused_softmax_bwd_kernel(const float* dP, const float* P, float* dS, int M, int Tk, float scale) {
    int row = blockIdx.x;
    if (row >= M) return;
    const float* dp_row = dP + row * Tk;
    const float* p_row = P + row * Tk;
    float* ds_row = dS + row * Tk;
    int tid = threadIdx.x;

    float dot = 0.0f;
    for (int i = tid; i < Tk; i += blockDim.x) {
        dot += dp_row[i] * p_row[i];
    }
#ifndef __HIP_PLATFORM_AMD__
    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    }
#endif
    __shared__ float s_dot[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) s_dot[wid] = dot;
    __syncthreads();
    if (wid == 0) {
        float b_dot = (lane < (blockDim.x + 31) / 32) ? s_dot[lane] : 0.0f;
#ifndef __HIP_PLATFORM_AMD__
        for (int offset = 16; offset > 0; offset /= 2) {
            b_dot += __shfl_down_sync(0xffffffff, b_dot, offset);
        }
#endif
        if (lane == 0) s_dot[0] = b_dot;
    }
    __syncthreads();
    float row_dot = s_dot[0];

    for (int i = tid; i < Tk; i += blockDim.x) {
        ds_row[i] = p_row[i] * (dp_row[i] - row_dot) * scale;
    }
}

__global__ void fused_softmax_fwd_half_kernel(const __half* S, __half* P, int M, int Tk) {
    int row = blockIdx.x;
    if (row >= M) return;
    const __half* s_row = S + row * Tk;
    __half* p_row = P + row * Tk;
    int tid = threadIdx.x;

    float max_val = -1e37f;
    for (int i = tid; i < Tk; i += blockDim.x) {
        float val = __half2float(s_row[i]);
        if (val > max_val) max_val = val;
    }
#ifndef __HIP_PLATFORM_AMD__
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
#endif
    __shared__ float s_max[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) s_max[wid] = max_val;
    __syncthreads();
    if (wid == 0) {
        float b_max = (lane < (blockDim.x + 31) / 32) ? s_max[lane] : -1e37f;
#ifndef __HIP_PLATFORM_AMD__
        for (int offset = 16; offset > 0; offset /= 2) {
            b_max = fmaxf(b_max, __shfl_down_sync(0xffffffff, b_max, offset));
        }
#endif
        if (lane == 0) s_max[0] = b_max;
    }
    __syncthreads();
    float row_max = s_max[0];

    float sum_exp = 0.0f;
    for (int i = tid; i < Tk; i += blockDim.x) {
        float e = expf(__half2float(s_row[i]) - row_max);
        p_row[i] = __float2half(e);
        sum_exp += e;
    }
#ifndef __HIP_PLATFORM_AMD__
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
#endif
    if (lane == 0) s_max[wid] = sum_exp;
    __syncthreads();
    if (wid == 0) {
        float b_sum = (lane < (blockDim.x + 31) / 32) ? s_max[lane] : 0.0f;
#ifndef __HIP_PLATFORM_AMD__
        for (int offset = 16; offset > 0; offset /= 2) {
            b_sum += __shfl_down_sync(0xffffffff, b_sum, offset);
        }
#endif
        if (lane == 0) s_max[0] = b_sum;
    }
    __syncthreads();
    float inv_sum = 1.0f / (s_max[0] + 1e-6f);

    for (int i = tid; i < Tk; i += blockDim.x) {
        p_row[i] = __float2half(__half2float(p_row[i]) * inv_sum);
    }
}

__global__ void fused_softmax_bwd_half_kernel(const __half* dP, const __half* P, __half* dS, int M, int Tk, float scale) {
    int row = blockIdx.x;
    if (row >= M) return;
    const __half* dp_row = dP + row * Tk;
    const __half* p_row = P + row * Tk;
    __half* ds_row = dS + row * Tk;
    int tid = threadIdx.x;

    float dot = 0.0f;
    for (int i = tid; i < Tk; i += blockDim.x) {
        dot += __half2float(dp_row[i]) * __half2float(p_row[i]);
    }
#ifndef __HIP_PLATFORM_AMD__
    for (int offset = 16; offset > 0; offset /= 2) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    }
#endif
    __shared__ float s_dot[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) s_dot[wid] = dot;
    __syncthreads();
    if (wid == 0) {
        float b_dot = (lane < (blockDim.x + 31) / 32) ? s_dot[lane] : 0.0f;
#ifndef __HIP_PLATFORM_AMD__
        for (int offset = 16; offset > 0; offset /= 2) {
            b_dot += __shfl_down_sync(0xffffffff, b_dot, offset);
        }
#endif
        if (lane == 0) s_dot[0] = b_dot;
    }
    __syncthreads();
    float row_dot = s_dot[0];

    for (int i = tid; i < Tk; i += blockDim.x) {
        float p_val = __half2float(p_row[i]);
        float dp_val = __half2float(dp_row[i]);
        ds_row[i] = __float2half(p_val * (dp_val - row_dot) * scale);
    }
}

static float* g_attn_scratch_s = nullptr;
static float* g_attn_scratch_p = nullptr;
static float* g_attn_scratch_dp = nullptr;
static float* g_attn_scratch_ds = nullptr;
static size_t g_attn_scratch_elements = 0;

static void ensure_attn_scratch(size_t elements) {
    if (elements > g_attn_scratch_elements) {
        if (g_attn_scratch_s) GPU_API(Free)(g_attn_scratch_s);
        if (g_attn_scratch_p) GPU_API(Free)(g_attn_scratch_p);
        if (g_attn_scratch_dp) GPU_API(Free)(g_attn_scratch_dp);
        if (g_attn_scratch_ds) GPU_API(Free)(g_attn_scratch_ds);

        GPU_API(Malloc)((void**)&g_attn_scratch_s, elements * sizeof(float));
        GPU_API(Malloc)((void**)&g_attn_scratch_p, elements * sizeof(float));
        GPU_API(Malloc)((void**)&g_attn_scratch_dp, elements * sizeof(float));
        GPU_API(Malloc)((void**)&g_attn_scratch_ds, elements * sizeof(float));
        g_attn_scratch_elements = elements;
    }
}

extern "C" void gpu_flash_attention(void *Q, int64_t q_off, void *K, int64_t k_off,
                                    void *V, int64_t v_off, void *O, int64_t o_off,
                                    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk,
                                    int64_t D, float scale) {
    if (g_fa3_fwd_fn) {
        g_fa3_fwd_fn((const char*)Q + q_off * sizeof(float), (const char*)K + k_off * sizeof(float),
                     (const char*)V + v_off * sizeof(float), (char*)O + o_off * sizeof(float),
                     B, H, H_kv, Tq, Tk, D, scale, (void*)g_compute_stream, 0);
        return;
    }

    size_t total_elements = (size_t)B * H * Tq * Tk;
    ensure_attn_scratch(total_elements);

    const float* q_ptr = (const float*)Q + q_off;
    const float* k_ptr = (const float*)K + k_off;
    const float* v_ptr = (const float*)V + v_off;
    float* o_ptr = (float*)O + o_off;

#ifndef __HIP_PLATFORM_AMD__
    cublasHandle_t handle = get_cublas_handle();
    float alpha = scale;
    float beta = 0.0f;
    long long strideK = Tk * D;
    long long strideQ = Tq * D;
    long long strideS = Tq * Tk;
    long long strideV = Tk * D;
    long long strideO = Tq * D;
    int batch_count = B * H;

    cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             Tk, Tq, D,
                             &alpha,
                             k_ptr, D, strideK,
                             q_ptr, D, strideQ,
                             &beta,
                             g_attn_scratch_s, Tk, strideS,
                             batch_count);

    int total_rows = B * H * Tq;
    int block_threads = 256;
    fused_softmax_fwd_kernel<<<total_rows, block_threads, 0, g_compute_stream>>>(
        g_attn_scratch_s, g_attn_scratch_p, total_rows, Tk);

    float one = 1.0f;
    float zero = 0.0f;
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             D, Tq, Tk,
                             &one,
                             v_ptr, D, strideV,
                             g_attn_scratch_p, Tk, strideS,
                             &zero,
                             o_ptr, D, strideO,
                             batch_count);
#else
    rocblas_handle handle = get_rocblas_handle();
    float alpha = scale;
    float beta = 0.0f;
    long long strideK = Tk * D;
    long long strideQ = Tq * D;
    long long strideS = Tq * Tk;
    long long strideV = Tk * D;
    long long strideO = Tq * D;
    int batch_count = B * H;

    rocblas_sgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none,
                                  Tk, Tq, D,
                                  &alpha,
                                  k_ptr, D, strideK,
                                  q_ptr, D, strideQ,
                                  &beta,
                                  g_attn_scratch_s, Tk, strideS,
                                  batch_count);

    int total_rows = B * H * Tq;
    int block_threads = 256;
    hipLaunchKernelGGL(fused_softmax_fwd_kernel, dim3(total_rows), dim3(block_threads), 0, g_compute_stream,
                       g_attn_scratch_s, g_attn_scratch_p, total_rows, Tk);

    float one = 1.0f;
    float zero = 0.0f;
    rocblas_sgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
                                  D, Tq, Tk,
                                  &one,
                                  v_ptr, D, strideV,
                                  g_attn_scratch_p, Tk, strideS,
                                  &zero,
                                  o_ptr, D, strideO,
                                  batch_count);
#endif
}

extern "C" void gpu_flash_attention_half(void *Q, int64_t q_off, void *K, int64_t k_off,
                                         void *V, int64_t v_off, void *O, int64_t o_off,
                                         int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk,
                                         int64_t D, float scale) {
    if (g_fa3_fwd_fn) {
        g_fa3_fwd_fn((const char*)Q + q_off * sizeof(__half), (const char*)K + k_off * sizeof(__half),
                     (const char*)V + v_off * sizeof(__half), (char*)O + o_off * sizeof(__half),
                     B, H, H_kv, Tq, Tk, D, scale, (void*)g_compute_stream, 0);
        return;
    }

    size_t total_elements = (size_t)B * H * Tq * Tk;
    ensure_attn_scratch(total_elements);

    const __half* q_ptr = (const __half*)Q + q_off;
    const __half* k_ptr = (const __half*)K + k_off;
    const __half* v_ptr = (const __half*)V + v_off;
    __half* o_ptr = (__half*)O + o_off;
    __half* s_half = (__half*)g_attn_scratch_s;
    __half* p_half = (__half*)g_attn_scratch_p;

#ifndef __HIP_PLATFORM_AMD__
    cublasHandle_t handle = get_cublas_handle();
    const __half alpha = __float2half(scale);
    const __half beta = __float2half(0.0f);
    long long strideK = Tk * D;
    long long strideQ = Tq * D;
    long long strideS = Tq * Tk;
    long long strideV = Tk * D;
    long long strideO = Tq * D;
    int batch_count = B * H;

    cublasHgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             Tk, Tq, D,
                             &alpha,
                             k_ptr, D, strideK,
                             q_ptr, D, strideQ,
                             &beta,
                             s_half, Tk, strideS,
                             batch_count);

    int total_rows = B * H * Tq;
    int block_threads = 256;
    fused_softmax_fwd_half_kernel<<<total_rows, block_threads, 0, g_compute_stream>>>(
        s_half, p_half, total_rows, Tk);

    const __half one = __float2half(1.0f);
    const __half zero = __float2half(0.0f);
    cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             D, Tq, Tk,
                             &one,
                             v_ptr, D, strideV,
                             p_half, Tk, strideS,
                             &zero,
                             o_ptr, D, strideO,
                             batch_count);
#else
    rocblas_handle handle = get_rocblas_handle();
    const __half alpha_h = __float2half(scale);
    const __half beta_h = __float2half(0.0f);
    long long strideK = Tk * D;
    long long strideQ = Tq * D;
    long long strideS = Tq * Tk;
    long long strideV = Tk * D;
    long long strideO = Tq * D;
    int batch_count = B * H;

    rocblas_hgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none,
                                  Tk, Tq, D,
                                  &alpha_h,
                                  k_ptr, D, strideK,
                                  q_ptr, D, strideQ,
                                  &beta_h,
                                  s_half, Tk, strideS,
                                  batch_count);

    int total_rows = B * H * Tq;
    int block_threads = 256;
    hipLaunchKernelGGL(fused_softmax_fwd_half_kernel, dim3(total_rows), dim3(block_threads), 0, g_compute_stream,
                       s_half, p_half, total_rows, Tk);

    const __half one = __float2half(1.0f);
    const __half zero = __float2half(0.0f);
    rocblas_hgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
                                  D, Tq, Tk,
                                  &one,
                                  v_ptr, D, strideV,
                                  p_half, Tk, strideS,
                                  &zero,
                                  o_ptr, D, strideO,
                                  batch_count);
#endif
}

extern "C" void gpu_flash_attention_backward(void *dQ, int64_t dq_off, void *dK, int64_t dk_off,
                                             void *dV, int64_t dv_off, void *O, int64_t o_off, void *dO,
                                             int64_t do_off, void *Q, int64_t q_off, void *K, int64_t k_off,
                                             void *V, int64_t v_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq,
                                             int64_t Tk, int64_t D, float scale) {
    if (g_fa3_bwd_fn) {
        g_fa3_bwd_fn((char*)dQ + dq_off * sizeof(float), (char*)dK + dk_off * sizeof(float),
                     (char*)dV + dv_off * sizeof(float), (const char*)O + o_off * sizeof(float),
                     (const char*)dO + do_off * sizeof(float), (const char*)Q + q_off * sizeof(float),
                     (const char*)K + k_off * sizeof(float), (const char*)V + v_off * sizeof(float),
                     B, H, H_kv, Tq, Tk, D, scale, (void*)g_compute_stream, 0);
        return;
    }

    size_t total_elements = (size_t)B * H * Tq * Tk;
    ensure_attn_scratch(total_elements);

    float* dq_ptr = (float*)dQ + dq_off;
    float* dk_ptr = (float*)dK + dk_off;
    float* dv_ptr = (float*)dV + dv_off;
    const float* do_ptr = (const float*)dO + do_off;
    const float* q_ptr = (const float*)Q + q_off;
    const float* k_ptr = (const float*)K + k_off;
    const float* v_ptr = (const float*)V + v_off;

#ifndef __HIP_PLATFORM_AMD__
    cublasHandle_t handle = get_cublas_handle();
    float alpha = scale;
    float beta = 0.0f;
    long long strideK = Tk * D;
    long long strideQ = Tq * D;
    long long strideS = Tq * Tk;
    long long strideV = Tk * D;
    long long strideO = Tq * D;
    int batch_count = B * H;

    cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             Tk, Tq, D,
                             &alpha,
                             k_ptr, D, strideK,
                             q_ptr, D, strideQ,
                             &beta,
                             g_attn_scratch_s, Tk, strideS,
                             batch_count);

    int total_rows = B * H * Tq;
    int block_threads = 256;
    fused_softmax_fwd_kernel<<<total_rows, block_threads, 0, g_compute_stream>>>(
        g_attn_scratch_s, g_attn_scratch_p, total_rows, Tk);

    float one = 1.0f;
    float zero = 0.0f;
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             D, Tk, Tq,
                             &one,
                             do_ptr, D, strideO,
                             g_attn_scratch_p, Tk, strideS,
                             &zero,
                             dv_ptr, D, strideV,
                             batch_count);

    cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             Tk, Tq, D,
                             &one,
                             v_ptr, D, strideV,
                             do_ptr, D, strideO,
                             &zero,
                             g_attn_scratch_dp, Tk, strideS,
                             batch_count);

    fused_softmax_bwd_kernel<<<total_rows, block_threads, 0, g_compute_stream>>>(
        g_attn_scratch_dp, g_attn_scratch_p, g_attn_scratch_ds, total_rows, Tk, scale);

    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             D, Tq, Tk,
                             &one,
                             k_ptr, D, strideK,
                             g_attn_scratch_ds, Tk, strideS,
                             &zero,
                             dq_ptr, D, strideQ,
                             batch_count);

    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             D, Tk, Tq,
                             &one,
                             q_ptr, D, strideQ,
                             g_attn_scratch_ds, Tk, strideS,
                             &zero,
                             dk_ptr, D, strideK,
                             batch_count);
#else
    rocblas_handle handle = get_rocblas_handle();
    float alpha = scale;
    float beta = 0.0f;
    long long strideK = Tk * D;
    long long strideQ = Tq * D;
    long long strideS = Tq * Tk;
    long long strideV = Tk * D;
    long long strideO = Tq * D;
    int batch_count = B * H;

    rocblas_sgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none,
                                  Tk, Tq, D,
                                  &alpha,
                                  k_ptr, D, strideK,
                                  q_ptr, D, strideQ,
                                  &beta,
                                  g_attn_scratch_s, Tk, strideS,
                                  batch_count);

    int total_rows = B * H * Tq;
    int block_threads = 256;
    hipLaunchKernelGGL(fused_softmax_fwd_kernel, dim3(total_rows), dim3(block_threads), 0, g_compute_stream,
                       g_attn_scratch_s, g_attn_scratch_p, total_rows, Tk);

    float one = 1.0f;
    float zero = 0.0f;
    rocblas_sgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_transpose,
                                  D, Tk, Tq,
                                  &one,
                                  do_ptr, D, strideO,
                                  g_attn_scratch_p, Tk, strideS,
                                  &zero,
                                  dv_ptr, D, strideV,
                                  batch_count);

    rocblas_sgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none,
                                  Tk, Tq, D,
                                  &one,
                                  v_ptr, D, strideV,
                                  do_ptr, D, strideO,
                                  &zero,
                                  g_attn_scratch_dp, Tk, strideS,
                                  batch_count);

    hipLaunchKernelGGL(fused_softmax_bwd_kernel, dim3(total_rows), dim3(block_threads), 0, g_compute_stream,
                       g_attn_scratch_dp, g_attn_scratch_p, g_attn_scratch_ds, total_rows, Tk, scale);

    rocblas_sgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
                                  D, Tq, Tk,
                                  &one,
                                  k_ptr, D, strideK,
                                  g_attn_scratch_ds, Tk, strideS,
                                  &zero,
                                  dq_ptr, D, strideQ,
                                  batch_count);

    rocblas_sgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_transpose,
                                  D, Tk, Tq,
                                  &one,
                                  q_ptr, D, strideQ,
                                  g_attn_scratch_ds, Tk, strideS,
                                  &zero,
                                  dk_ptr, D, strideK,
                                  batch_count);
#endif
}

extern "C" void gpu_flash_attention_backward_half(
    void *dQ, int64_t dq_off, void *dK, int64_t dk_off, void *dV, int64_t dv_off, void *O,
    int64_t o_off, void *dO, int64_t do_off, void *Q, int64_t q_off, void *K, int64_t k_off,
    void *V, int64_t v_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D,
    float scale) {
    if (g_fa3_bwd_fn) {
        g_fa3_bwd_fn((char*)dQ + dq_off * sizeof(__half), (char*)dK + dk_off * sizeof(__half),
                     (char*)dV + dv_off * sizeof(__half), (const char*)O + o_off * sizeof(__half),
                     (const char*)dO + do_off * sizeof(__half), (const char*)Q + q_off * sizeof(__half),
                     (const char*)K + k_off * sizeof(__half), (const char*)V + v_off * sizeof(__half),
                     B, H, H_kv, Tq, Tk, D, scale, (void*)g_compute_stream, 0);
        return;
    }

    size_t total_elements = (size_t)B * H * Tq * Tk;
    ensure_attn_scratch(total_elements);

    __half* dq_ptr = (__half*)dQ + dq_off;
    __half* dk_ptr = (__half*)dK + dk_off;
    __half* dv_ptr = (__half*)dV + dv_off;
    const __half* do_ptr = (const __half*)dO + do_off;
    const __half* q_ptr = (const __half*)Q + q_off;
    const __half* k_ptr = (const __half*)K + k_off;
    const __half* v_ptr = (const __half*)V + v_off;
    __half* s_half = (__half*)g_attn_scratch_s;
    __half* p_half = (__half*)g_attn_scratch_p;
    __half* dp_half = (__half*)g_attn_scratch_dp;
    __half* ds_half = (__half*)g_attn_scratch_ds;

#ifndef __HIP_PLATFORM_AMD__
    cublasHandle_t handle = get_cublas_handle();
    const __half alpha = __float2half(scale);
    const __half beta = __float2half(0.0f);
    long long strideK = Tk * D;
    long long strideQ = Tq * D;
    long long strideS = Tq * Tk;
    long long strideV = Tk * D;
    long long strideO = Tq * D;
    int batch_count = B * H;

    cublasHgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             Tk, Tq, D,
                             &alpha,
                             k_ptr, D, strideK,
                             q_ptr, D, strideQ,
                             &beta,
                             s_half, Tk, strideS,
                             batch_count);

    int total_rows = B * H * Tq;
    int block_threads = 256;
    fused_softmax_fwd_half_kernel<<<total_rows, block_threads, 0, g_compute_stream>>>(
        s_half, p_half, total_rows, Tk);

    const __half one = __float2half(1.0f);
    const __half zero = __float2half(0.0f);
    cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             D, Tk, Tq,
                             &one,
                             do_ptr, D, strideO,
                             p_half, Tk, strideS,
                             &zero,
                             dv_ptr, D, strideV,
                             batch_count);

    cublasHgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             Tk, Tq, D,
                             &one,
                             v_ptr, D, strideV,
                             do_ptr, D, strideO,
                             &zero,
                             dp_half, Tk, strideS,
                             batch_count);

    fused_softmax_bwd_half_kernel<<<total_rows, block_threads, 0, g_compute_stream>>>(
        dp_half, p_half, ds_half, total_rows, Tk, scale);

    cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             D, Tq, Tk,
                             &one,
                             k_ptr, D, strideK,
                             ds_half, Tk, strideS,
                             &zero,
                             dq_ptr, D, strideQ,
                             batch_count);

    cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             D, Tk, Tq,
                             &one,
                             q_ptr, D, strideQ,
                             ds_half, Tk, strideS,
                             &zero,
                             dk_ptr, D, strideK,
                             batch_count);
#else
    rocblas_handle handle = get_rocblas_handle();
    const __half alpha_h = __float2half(scale);
    const __half beta_h = __float2half(0.0f);
    long long strideK = Tk * D;
    long long strideQ = Tq * D;
    long long strideS = Tq * Tk;
    long long strideV = Tk * D;
    long long strideO = Tq * D;
    int batch_count = B * H;

    rocblas_hgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none,
                                  Tk, Tq, D,
                                  &alpha_h,
                                  k_ptr, D, strideK,
                                  q_ptr, D, strideQ,
                                  &beta_h,
                                  s_half, Tk, strideS,
                                  batch_count);

    int total_rows = B * H * Tq;
    int block_threads = 256;
    hipLaunchKernelGGL(fused_softmax_fwd_half_kernel, dim3(total_rows), dim3(block_threads), 0, g_compute_stream,
                       s_half, p_half, total_rows, Tk);

    const __half one = __float2half(1.0f);
    const __half zero = __float2half(0.0f);
    rocblas_hgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_transpose,
                                  D, Tk, Tq,
                                  &one,
                                  do_ptr, D, strideO,
                                  p_half, Tk, strideS,
                                  &zero,
                                  dv_ptr, D, strideV,
                                  batch_count);

    rocblas_hgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none,
                                  Tk, Tq, D,
                                  &one,
                                  v_ptr, D, strideV,
                                  do_ptr, D, strideO,
                                  &zero,
                                  dp_half, Tk, strideS,
                                  batch_count);

    hipLaunchKernelGGL(fused_softmax_bwd_half_kernel, dim3(total_rows), dim3(block_threads), 0, g_compute_stream,
                       dp_half, p_half, ds_half, total_rows, Tk, scale);

    rocblas_hgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
                                  D, Tq, Tk,
                                  &one,
                                  k_ptr, D, strideK,
                                  ds_half, Tk, strideS,
                                  &zero,
                                  dq_ptr, D, strideQ,
                                  batch_count);

    rocblas_hgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_transpose,
                                  D, Tk, Tq,
                                  &one,
                                  q_ptr, D, strideQ,
                                  ds_half, Tk, strideS,
                                  &zero,
                                  dk_ptr, D, strideK,
                                  batch_count);
#endif
}
