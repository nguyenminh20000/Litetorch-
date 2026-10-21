#include "gpu_common.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>

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

template <typename T> struct FaDtype;
template <> struct FaDtype<float> {
    __device__ static inline float to_float(float x) { return x; }
    __device__ static inline float from_float(float x) { return x; }
};
template <> struct FaDtype<__half> {
    __device__ static inline float to_float(__half x) { return __half2float(x); }
    __device__ static inline __half from_float(float x) { return __float2half(x); }
};

template <typename T, int DMAX, int BCMAX>
__global__ void fa_fwd_kernel(const T* Q, const T* K, const T* V, T* O, float* LSE_out,
                              int64_t B, int64_t H, int64_t H_kv,
                              int64_t Tq, int64_t Tk, int64_t D,
                              float scale, int causal) {
    const int Br = 8;
    const int DL = DMAX / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    __shared__ float KV[BCMAX][DMAX];
    __shared__ float PS[Br][BCMAX];

    int64_t tiles_q = (Tq + Br - 1) / Br;
    int64_t bh = (int64_t)blockIdx.x / tiles_q;
    int64_t tq = (int64_t)blockIdx.x % tiles_q;
    if (bh >= B * H) return;
    int64_t b = bh / H;
    int64_t h = bh % H;
    int64_t hk = h / (H / H_kv);
    int64_t qi = tq * Br + warp;

    int64_t q_base = (b * H + h) * Tq * D;
    int64_t kv_base = (b * H_kv + hk) * Tk * D;

    float q_reg[DL];
    float o_acc[DL];
#pragma unroll
    for (int t = 0; t < DL; t++) {
        q_reg[t] = 0.0f;
        o_acc[t] = 0.0f;
    }
    if (qi < Tq) {
        int64_t q_row = q_base + qi * D;
#pragma unroll
        for (int t = 0; t < DL; t++) {
            int d = (t << 5) + lane;
            if (d < D) q_reg[t] = FaDtype<T>::to_float(Q[q_row + d]);
        }
    }

    float m = -1e37f;
    float l = 0.0f;

    for (int64_t j0 = 0; j0 < Tk; j0 += BCMAX) {
        for (int t = threadIdx.x; t < BCMAX * DMAX; t += blockDim.x) {
            int jj = t / DMAX;
            int dd = t % DMAX;
            int64_t kj = j0 + jj;
            float v = 0.0f;
            if (kj < Tk && dd < D) v = FaDtype<T>::to_float(K[kv_base + kj * D + dd]);
            KV[jj][dd] = v;
        }
        __syncthreads();

        float s = -1e37f;
        if (qi < Tq) {
            float dot = 0.0f;
            int Di = (int)D;
            for (int d = 0; d < Di; ++d) {
#ifdef __HIP_PLATFORM_AMD__
                float qv = __shfl(q_reg[d >> 5], d & 31);
#else
                float qv = __shfl_sync(0xffffffff, q_reg[d >> 5], d & 31);
#endif
                float kv = (lane < BCMAX) ? KV[lane][d] : 0.0f;
                dot = fmaf(qv, kv, dot);
            }
            if (lane < BCMAX) {
                int64_t kj = j0 + lane;
                s = dot * scale;
                if (kj >= Tk || (causal && kj > qi)) s = -1e37f;
            }
        }
        float rowmax = warp_reduce_max(s);
        float m_new = fmaxf(m, rowmax);
        float alpha = expf(m - m_new);
        float p = expf(s - m_new);
        if (lane < BCMAX) PS[warp][lane] = p;
        float rowsum = warp_reduce_sum(p);
        float l_new = l * alpha + rowsum;

        for (int t = threadIdx.x; t < BCMAX * DMAX; t += blockDim.x) {
            int jj = t / DMAX;
            int dd = t % DMAX;
            int64_t kj = j0 + jj;
            float v = 0.0f;
            if (kj < Tk && dd < D) v = FaDtype<T>::to_float(V[kv_base + kj * D + dd]);
            KV[jj][dd] = v;
        }
        __syncthreads();

        if (qi < Tq) {
            float o_scale = (l * alpha) / l_new;
            float p_scale = 1.0f / l_new;
#pragma unroll
            for (int t = 0; t < DL; t++) {
                int d = (t << 5) + lane;
                float pv = 0.0f;
                if (d < D) {
                    for (int jj = 0; jj < BCMAX; jj++) pv = fmaf(PS[warp][jj], KV[jj][d], pv);
                }
                o_acc[t] = o_acc[t] * o_scale + pv * p_scale;
            }
            m = m_new;
            l = l_new;
        }
        __syncthreads();
    }

    float lse = 0.0f;
    if (qi < Tq) {
        lse = m + logf(l);
        int64_t o_row = q_base + qi * D;
#pragma unroll
        for (int t = 0; t < DL; t++) {
            int d = (t << 5) + lane;
            if (d < D) O[o_row + d] = FaDtype<T>::from_float(o_acc[t]);
        }
        if (LSE_out) LSE_out[(b * H + h) * Tq + qi] = lse;
    }
}

template <typename T>
static void launch_fa_fwd(const T* Q, const T* K, const T* V, T* O, float* LSE_out,
                          int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk,
                          int64_t D, float scale, int causal) {
    if (D <= 0 || D > 256) throw std::runtime_error("[litetorch] flash_attention: head dim D out of range (1..256)");
    if (H_kv <= 0 || H <= 0 || H % H_kv != 0) throw std::runtime_error("[litetorch] flash_attention: H must be divisible by H_kv");
    const int Br = 8;
    int64_t tiles = B * H * ((Tq + Br - 1) / Br);
    if (D <= 32) {
        fa_fwd_kernel<T, 32, 32><<<(unsigned)tiles, 256, 0, dev_stream(current_device())>>>(Q, K, V, O, LSE_out, B, H, H_kv, Tq, Tk, D, scale, causal);
    } else if (D <= 64) {
        fa_fwd_kernel<T, 64, 32><<<(unsigned)tiles, 256, 0, dev_stream(current_device())>>>(Q, K, V, O, LSE_out, B, H, H_kv, Tq, Tk, D, scale, causal);
    } else if (D <= 128) {
        fa_fwd_kernel<T, 128, 32><<<(unsigned)tiles, 256, 0, dev_stream(current_device())>>>(Q, K, V, O, LSE_out, B, H, H_kv, Tq, Tk, D, scale, causal);
    } else {
        fa_fwd_kernel<T, 256, 16><<<(unsigned)tiles, 256, 0, dev_stream(current_device())>>>(Q, K, V, O, LSE_out, B, H, H_kv, Tq, Tk, D, scale, causal);
    }
}

__global__ void fused_softmax_fwd_kernel(const float* S, float* P, int M, int Tk, int Tq, bool causal) {
    int row = blockIdx.x;
    if (row >= M) return;
    const float* s_row = S + row * Tk;
    float* p_row = P + row * Tk;
    int tid = threadIdx.x;
    int qi = causal ? (row % Tq) : Tk;

    float max_val = -1e37f;
    for (int i = tid; i < Tk; i += blockDim.x) {
        float val = (i > qi) ? -1e37f : s_row[i];
        if (val > max_val) max_val = val;
    }
    max_val = warp_reduce_max(max_val);
    __shared__ float s_max[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) s_max[wid] = max_val;
    __syncthreads();
    if (wid == 0) {
        float b_max = (lane < (blockDim.x + 31) / 32) ? s_max[lane] : -1e37f;
        b_max = warp_reduce_max(b_max);
        if (lane == 0) s_max[0] = b_max;
    }
    __syncthreads();
    float row_max = s_max[0];

    float sum_exp = 0.0f;
    for (int i = tid; i < Tk; i += blockDim.x) {
        float e = (i > qi) ? 0.0f : expf(s_row[i] - row_max);
        p_row[i] = e;
        sum_exp += e;
    }
    sum_exp = warp_reduce_sum(sum_exp);
    if (lane == 0) s_max[wid] = sum_exp;
    __syncthreads();
    if (wid == 0) {
        float b_sum = (lane < (blockDim.x + 31) / 32) ? s_max[lane] : 0.0f;
        b_sum = warp_reduce_sum(b_sum);
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
    dot = warp_reduce_sum(dot);
    __shared__ float s_dot[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) s_dot[wid] = dot;
    __syncthreads();
    if (wid == 0) {
        float b_dot = (lane < (blockDim.x + 31) / 32) ? s_dot[lane] : 0.0f;
        b_dot = warp_reduce_sum(b_dot);
        if (lane == 0) s_dot[0] = b_dot;
    }
    __syncthreads();
    float row_dot = s_dot[0];

    for (int i = tid; i < Tk; i += blockDim.x) {
        ds_row[i] = p_row[i] * (dp_row[i] - row_dot) * scale;
    }
}

__global__ void fused_softmax_fwd_half_kernel(const __half* S, __half* P, int M, int Tk, int Tq, bool causal) {
    int row = blockIdx.x;
    if (row >= M) return;
    const __half* s_row = S + row * Tk;
    __half* p_row = P + row * Tk;
    int tid = threadIdx.x;
    int qi = causal ? (row % Tq) : Tk;

    float max_val = -1e37f;
    for (int i = tid; i < Tk; i += blockDim.x) {
        float val = (i > qi) ? -1e37f : __half2float(s_row[i]);
        if (val > max_val) max_val = val;
    }
    max_val = warp_reduce_max(max_val);
    __shared__ float s_max[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) s_max[wid] = max_val;
    __syncthreads();
    if (wid == 0) {
        float b_max = (lane < (blockDim.x + 31) / 32) ? s_max[lane] : -1e37f;
        b_max = warp_reduce_max(b_max);
        if (lane == 0) s_max[0] = b_max;
    }
    __syncthreads();
    float row_max = s_max[0];

    float sum_exp = 0.0f;
    for (int i = tid; i < Tk; i += blockDim.x) {
        float e = (i > qi) ? 0.0f : expf(__half2float(s_row[i]) - row_max);
        p_row[i] = __float2half(e);
        sum_exp += e;
    }
    sum_exp = warp_reduce_sum(sum_exp);
    if (lane == 0) s_max[wid] = sum_exp;
    __syncthreads();
    if (wid == 0) {
        float b_sum = (lane < (blockDim.x + 31) / 32) ? s_max[lane] : 0.0f;
        b_sum = warp_reduce_sum(b_sum);
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
    dot = warp_reduce_sum(dot);
    __shared__ float s_dot[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) s_dot[wid] = dot;
    __syncthreads();
    if (wid == 0) {
        float b_dot = (lane < (blockDim.x + 31) / 32) ? s_dot[lane] : 0.0f;
        b_dot = warp_reduce_sum(b_dot);
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

__global__ void softmax_fwd_warp_kernel(const float* S, float* P, int M, int Tk, int Tq, bool causal) {
    int row = blockIdx.x;
    if (row >= M) return;
    const float* s_row = S + row * Tk;
    float* p_row = P + row * Tk;
    int lane = threadIdx.x;
    int qi = causal ? (row % Tq) : Tk;

    float max_val = -1e37f;
    for (int i = lane; i < Tk; i += 32) {
        float val = (i > qi) ? -1e37f : s_row[i];
        if (val > max_val) max_val = val;
    }
    max_val = warp_reduce_max(max_val);

    float sum_exp = 0.0f;
    for (int i = lane; i < Tk; i += 32) {
        float e = (i > qi) ? 0.0f : expf(s_row[i] - max_val);
        p_row[i] = e;
        sum_exp += e;
    }
    sum_exp = warp_reduce_sum(sum_exp);
    float inv_sum = 1.0f / (sum_exp + 1e-6f);

    for (int i = lane; i < Tk; i += 32) {
        p_row[i] *= inv_sum;
    }
}

__global__ void softmax_bwd_warp_kernel(const float* dP, const float* P, float* dS, int M, int Tk, float scale) {
    int row = blockIdx.x;
    if (row >= M) return;
    const float* dp_row = dP + row * Tk;
    const float* p_row = P + row * Tk;
    float* ds_row = dS + row * Tk;
    int lane = threadIdx.x;

    float dot = 0.0f;
    for (int i = lane; i < Tk; i += 32) {
        dot += dp_row[i] * p_row[i];
    }
    dot = warp_reduce_sum(dot);

    for (int i = lane; i < Tk; i += 32) {
        ds_row[i] = p_row[i] * (dp_row[i] - dot) * scale;
    }
}

__global__ void softmax_fwd_warp_half_kernel(const __half* S, __half* P, int M, int Tk, int Tq, bool causal) {
    int row = blockIdx.x;
    if (row >= M) return;
    const __half* s_row = S + row * Tk;
    __half* p_row = P + row * Tk;
    int lane = threadIdx.x;
    int qi = causal ? (row % Tq) : Tk;

    float max_val = -1e37f;
    for (int i = lane; i < Tk; i += 32) {
        float val = (i > qi) ? -1e37f : __half2float(s_row[i]);
        if (val > max_val) max_val = val;
    }
    max_val = warp_reduce_max(max_val);

    float sum_exp = 0.0f;
    for (int i = lane; i < Tk; i += 32) {
        float e = (i > qi) ? 0.0f : expf(__half2float(s_row[i]) - max_val);
        p_row[i] = __float2half(e);
        sum_exp += e;
    }
    sum_exp = warp_reduce_sum(sum_exp);
    float inv_sum = 1.0f / (sum_exp + 1e-6f);

    for (int i = lane; i < Tk; i += 32) {
        p_row[i] = __float2half(__half2float(p_row[i]) * inv_sum);
    }
}

__global__ void softmax_bwd_warp_half_kernel(const __half* dP, const __half* P, __half* dS, int M, int Tk, float scale) {
    int row = blockIdx.x;
    if (row >= M) return;
    const __half* dp_row = dP + row * Tk;
    const __half* p_row = P + row * Tk;
    __half* ds_row = dS + row * Tk;
    int lane = threadIdx.x;

    float dot = 0.0f;
    for (int i = lane; i < Tk; i += 32) {
        dot += __half2float(dp_row[i]) * __half2float(p_row[i]);
    }
    dot = warp_reduce_sum(dot);

    for (int i = lane; i < Tk; i += 32) {
        float p_val = __half2float(p_row[i]);
        float dp_val = __half2float(dp_row[i]);
        ds_row[i] = __float2half(p_val * (dp_val - dot) * scale);
    }
}

struct AttnScratch {
    float* s = nullptr;
    float* p = nullptr;
    float* dp = nullptr;
    float* ds = nullptr;
    size_t elements = 0;
};
static std::unordered_map<int, AttnScratch> g_attn_scratch_map;
static float* g_attn_scratch_s = nullptr;
static float* g_attn_scratch_p = nullptr;
static float* g_attn_scratch_dp = nullptr;
static float* g_attn_scratch_ds = nullptr;

static void ensure_attn_scratch(size_t elements) {
    int dev = current_device();
    AttnScratch& sc = g_attn_scratch_map[dev];
    if (elements > sc.elements) {
        GPU_API(StreamSynchronize)(dev_stream(dev));
        int cur = current_device();
        if (cur != dev) GPU_API(SetDevice)(dev);
        if (sc.s) GPU_API(Free)(sc.s);
        if (sc.p) GPU_API(Free)(sc.p);
        if (sc.dp) GPU_API(Free)(sc.dp);
        if (sc.ds) GPU_API(Free)(sc.ds);
        GPU_API(Malloc)((void**)&sc.s, elements * sizeof(float));
        GPU_API(Malloc)((void**)&sc.p, elements * sizeof(float));
        GPU_API(Malloc)((void**)&sc.dp, elements * sizeof(float));
        GPU_API(Malloc)((void**)&sc.ds, elements * sizeof(float));
        sc.elements = elements;
        if (cur != dev) GPU_API(SetDevice)(cur);
    }
    g_attn_scratch_s = sc.s;
    g_attn_scratch_p = sc.p;
    g_attn_scratch_dp = sc.dp;
    g_attn_scratch_ds = sc.ds;
}

static void launch_softmax_fwd(const float* S, float* P, int total_rows, int Tk, int Tq, bool causal) {
    if (Tk <= 32) {
        softmax_fwd_warp_kernel<<<total_rows, 32, 0, dev_stream(current_device())>>>(S, P, total_rows, Tk, Tq, causal);
    } else {
        fused_softmax_fwd_kernel<<<total_rows, 256, 0, dev_stream(current_device())>>>(S, P, total_rows, Tk, Tq, causal);
    }
}

static void launch_softmax_bwd(const float* dP, const float* P, float* dS, int total_rows, int Tk, float scale) {
    if (Tk <= 32) {
        softmax_bwd_warp_kernel<<<total_rows, 32, 0, dev_stream(current_device())>>>(dP, P, dS, total_rows, Tk, scale);
    } else {
        fused_softmax_bwd_kernel<<<total_rows, 256, 0, dev_stream(current_device())>>>(dP, P, dS, total_rows, Tk, scale);
    }
}

static void launch_softmax_fwd_half(const __half* S, __half* P, int total_rows, int Tk, int Tq, bool causal) {
    if (Tk <= 32) {
        softmax_fwd_warp_half_kernel<<<total_rows, 32, 0, dev_stream(current_device())>>>(S, P, total_rows, Tk, Tq, causal);
    } else {
        fused_softmax_fwd_half_kernel<<<total_rows, 256, 0, dev_stream(current_device())>>>(S, P, total_rows, Tk, Tq, causal);
    }
}

static void launch_softmax_bwd_half(const __half* dP, const __half* P, __half* dS, int total_rows, int Tk, float scale) {
    if (Tk <= 32) {
        softmax_bwd_warp_half_kernel<<<total_rows, 32, 0, dev_stream(current_device())>>>(dP, P, dS, total_rows, Tk, scale);
    } else {
        fused_softmax_bwd_half_kernel<<<total_rows, 256, 0, dev_stream(current_device())>>>(dP, P, dS, total_rows, Tk, scale);
    }
}

#ifdef __HIP_PLATFORM_AMD__
static void launch_softmax_fwd_hip(const float* S, float* P, int total_rows, int Tk, int Tq, bool causal) {
    if (Tk <= 32) {
        hipLaunchKernelGGL(softmax_fwd_warp_kernel, dim3(total_rows), dim3(32), 0, dev_stream(current_device()), S, P, total_rows, Tk, Tq, causal);
    } else {
        hipLaunchKernelGGL(fused_softmax_fwd_kernel, dim3(total_rows), dim3(256), 0, dev_stream(current_device()), S, P, total_rows, Tk, Tq, causal);
    }
}

static void launch_softmax_bwd_hip(const float* dP, const float* P, float* dS, int total_rows, int Tk, float scale) {
    if (Tk <= 32) {
        hipLaunchKernelGGL(softmax_bwd_warp_kernel, dim3(total_rows), dim3(32), 0, dev_stream(current_device()), dP, P, dS, total_rows, Tk, scale);
    } else {
        hipLaunchKernelGGL(fused_softmax_bwd_kernel, dim3(total_rows), dim3(256), 0, dev_stream(current_device()), dP, P, dS, total_rows, Tk, scale);
    }
}

static void launch_softmax_fwd_half_hip(const __half* S, __half* P, int total_rows, int Tk, int Tq, bool causal) {
    if (Tk <= 32) {
        hipLaunchKernelGGL(softmax_fwd_warp_half_kernel, dim3(total_rows), dim3(32), 0, dev_stream(current_device()), S, P, total_rows, Tk, Tq, causal);
    } else {
        hipLaunchKernelGGL(fused_softmax_fwd_half_kernel, dim3(total_rows), dim3(256), 0, dev_stream(current_device()), S, P, total_rows, Tk, Tq, causal);
    }
}

static void launch_softmax_bwd_half_hip(const __half* dP, const __half* P, __half* dS, int total_rows, int Tk, float scale) {
    if (Tk <= 32) {
        hipLaunchKernelGGL(softmax_bwd_warp_half_kernel, dim3(total_rows), dim3(32), 0, dev_stream(current_device()), dP, P, dS, total_rows, Tk, scale);
    } else {
        hipLaunchKernelGGL(fused_softmax_bwd_half_kernel, dim3(total_rows), dim3(256), 0, dev_stream(current_device()), dP, P, dS, total_rows, Tk, scale);
    }
}
#endif

__global__ void p_from_lse_kernel(const float* S, const float* LSE, float* P,
                                  int64_t total, int Tk, int Tq, int causal) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t row = idx / Tk;
    int64_t j = idx % Tk;
    int64_t qi = row % Tq;
    float p = 0.0f;
    if (!(causal && j > qi)) p = expf(S[idx] - LSE[row]);
    P[idx] = p;
}

__global__ void p_from_lse_half_kernel(const __half* S, const float* LSE, __half* P,
                                       int64_t total, int Tk, int Tq, int causal) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t row = idx / Tk;
    int64_t j = idx % Tk;
    int64_t qi = row % Tq;
    float p = 0.0f;
    if (!(causal && j > qi)) p = expf(__half2float(S[idx]) - LSE[row]);
    P[idx] = __float2half(p);
}

static void launch_p_from_lse(const float* S, const float* LSE, float* P,
                              int64_t total, int Tk, int Tq, int causal) {
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    p_from_lse_kernel<<<grid, block, 0, dev_stream(current_device())>>>(S, LSE, P, total, Tk, Tq, causal);
}

static void launch_p_from_lse_half(const __half* S, const float* LSE, __half* P,
                                   int64_t total, int Tk, int Tq, int causal) {
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    p_from_lse_half_kernel<<<grid, block, 0, dev_stream(current_device())>>>(S, LSE, P, total, Tk, Tq, causal);
}

// ============================================================================
// GQA-aware GEMM helpers for flash attention
// Q/O/dO/P/dS have H heads [B,H,T,D]; K/V/dK/dV have H_kv heads [B,H_kv,T,D]
// Query heads [hk*G, hk*G+G) share KV head hk, where G = H/H_kv
// ============================================================================
#ifndef __HIP_PLATFORM_AMD__
typedef cublasHandle_t GqaHandle;
#else
typedef rocblas_handle GqaHandle;
#endif

// S = alpha * K^T @ Q ; K:[B,H_kv,Tk,D] Q:[B,H,Tq,D] -> S:[B,H,Tq,Tk]
static void gqa_gemm_s_f32(GqaHandle handle,
    const float* k_ptr, const float* q_ptr, float* s_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D,
    float alpha) {
    float beta = 0.0f;
    long long sQ = Tq * D, sS = Tq * Tk;
    if (H == H_kv) {
        long long sK = Tk * D;
#ifndef __HIP_PLATFORM_AMD__
        cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)Tk, (int)Tq, (int)D, &alpha,
            k_ptr, (int)D, sK, q_ptr, (int)D, sQ, &beta,
            s_ptr, (int)Tk, sS, (int)(B * H));
#else
        rocblas_sgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none,
            Tk, Tq, D, &alpha, k_ptr, D, sK, q_ptr, D, sQ, &beta,
            s_ptr, Tk, sS, B * H);
#endif
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hk = 0; hk < H_kv; ++hk) {
#ifndef __HIP_PLATFORM_AMD__
            cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                (int)Tk, (int)Tq, (int)D, &alpha,
                k_ptr + (b * H_kv + hk) * Tk * D, (int)D, 0,
                q_ptr + (b * H + hk * G) * Tq * D, (int)D, sQ, &beta,
                s_ptr + (b * H + hk * G) * Tq * Tk, (int)Tk, sS, (int)G);
#else
            rocblas_sgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none,
                Tk, Tq, D, &alpha,
                k_ptr + (b * H_kv + hk) * Tk * D, D, 0,
                q_ptr + (b * H + hk * G) * Tq * D, D, sQ, &beta,
                s_ptr + (b * H + hk * G) * Tq * Tk, Tk, sS, G);
#endif
        }
    }
}

// O = V @ P ; V:[B,H_kv,Tk,D] P:[B,H,Tq,Tk] -> O:[B,H,Tq,D]
static void gqa_gemm_o_f32(GqaHandle handle,
    const float* v_ptr, const float* p_ptr, float* o_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    float one = 1.0f, zero = 0.0f;
    long long sS = Tq * Tk, sO = Tq * D;
    if (H == H_kv) {
        long long sV = Tk * D;
#ifndef __HIP_PLATFORM_AMD__
        cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            (int)D, (int)Tq, (int)Tk, &one,
            v_ptr, (int)D, sV, p_ptr, (int)Tk, sS, &zero,
            o_ptr, (int)D, sO, (int)(B * H));
#else
        rocblas_sgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
            D, Tq, Tk, &one, v_ptr, D, sV, p_ptr, Tk, sS, &zero,
            o_ptr, D, sO, B * H);
#endif
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hk = 0; hk < H_kv; ++hk) {
#ifndef __HIP_PLATFORM_AMD__
            cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)D, (int)Tq, (int)Tk, &one,
                v_ptr + (b * H_kv + hk) * Tk * D, (int)D, 0,
                p_ptr + (b * H + hk * G) * Tq * Tk, (int)Tk, sS, &zero,
                o_ptr + (b * H + hk * G) * Tq * D, (int)D, sO, (int)G);
#else
            rocblas_sgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
                D, Tq, Tk, &one,
                v_ptr + (b * H_kv + hk) * Tk * D, D, 0,
                p_ptr + (b * H + hk * G) * Tq * Tk, Tk, sS, &zero,
                o_ptr + (b * H + hk * G) * Tq * D, D, sO, G);
#endif
        }
    }
}

// dV[b,hk] += sum_g dO[b,hk*G+g] @ P[b,hk*G+g]^T (accumulate, zeroed first)
static void gqa_gemm_dv_f32(GqaHandle handle,
    const float* do_ptr, const float* p_ptr, float* dv_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    float one = 1.0f, zero = 0.0f;
    GPU_API(MemsetAsync)(dv_ptr, 0, (size_t)B * H_kv * Tk * D * sizeof(float),
                         dev_stream(current_device()));
    long long sO = Tq * D, sS = Tq * Tk;
    if (H == H_kv) {
        long long sV = Tk * D;
#ifndef __HIP_PLATFORM_AMD__
        cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            (int)D, (int)Tk, (int)Tq, &one,
            do_ptr, (int)D, sO, p_ptr, (int)Tk, sS, &zero,
            dv_ptr, (int)D, sV, (int)(B * H));
#else
        rocblas_sgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_transpose,
            D, Tk, Tq, &one, do_ptr, D, sO, p_ptr, Tk, sS, &zero,
            dv_ptr, D, sV, B * H);
#endif
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t hk = h / G;
#ifndef __HIP_PLATFORM_AMD__
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)D, (int)Tk, (int)Tq, &one,
                do_ptr + (b * H + h) * Tq * D, (int)D,
                p_ptr + (b * H + h) * Tq * Tk, (int)Tk,
                &one, dv_ptr + (b * H_kv + hk) * Tk * D, (int)D);
#else
            rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                D, Tk, Tq, &one,
                do_ptr + (b * H + h) * Tq * D, D,
                p_ptr + (b * H + h) * Tq * Tk, Tk,
                &one, dv_ptr + (b * H_kv + hk) * Tk * D, D);
#endif
        }
    }
}

// dS = V^T @ dO ; V:[B,H_kv,Tk,D] dO:[B,H,Tq,D] -> dS:[B,H,Tq,Tk]
static void gqa_gemm_ds_f32(GqaHandle handle,
    const float* v_ptr, const float* do_ptr, float* ds_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    float one = 1.0f, zero = 0.0f;
    long long sO = Tq * D, sS = Tq * Tk;
    if (H == H_kv) {
        long long sV = Tk * D;
#ifndef __HIP_PLATFORM_AMD__
        cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)Tk, (int)Tq, (int)D, &one,
            v_ptr, (int)D, sV, do_ptr, (int)D, sO, &zero,
            ds_ptr, (int)Tk, sS, (int)(B * H));
#else
        rocblas_sgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none,
            Tk, Tq, D, &one, v_ptr, D, sV, do_ptr, D, sO, &zero,
            ds_ptr, Tk, sS, B * H);
#endif
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hk = 0; hk < H_kv; ++hk) {
#ifndef __HIP_PLATFORM_AMD__
            cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                (int)Tk, (int)Tq, (int)D, &one,
                v_ptr + (b * H_kv + hk) * Tk * D, (int)D, 0,
                do_ptr + (b * H + hk * G) * Tq * D, (int)D, sO, &zero,
                ds_ptr + (b * H + hk * G) * Tq * Tk, (int)Tk, sS, (int)G);
#else
            rocblas_sgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none,
                Tk, Tq, D, &one,
                v_ptr + (b * H_kv + hk) * Tk * D, D, 0,
                do_ptr + (b * H + hk * G) * Tq * D, D, sO, &zero,
                ds_ptr + (b * H + hk * G) * Tq * Tk, Tk, sS, G);
#endif
        }
    }
}

// dQ = K @ dS ; K:[B,H_kv,Tk,D] dS:[B,H,Tq,Tk] -> dQ:[B,H,Tq,D]
static void gqa_gemm_dq_f32(GqaHandle handle,
    const float* k_ptr, const float* ds_ptr, float* dq_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    float one = 1.0f, zero = 0.0f;
    long long sS = Tq * Tk, sQ = Tq * D;
    if (H == H_kv) {
        long long sK = Tk * D;
#ifndef __HIP_PLATFORM_AMD__
        cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            (int)D, (int)Tq, (int)Tk, &one,
            k_ptr, (int)D, sK, ds_ptr, (int)Tk, sS, &zero,
            dq_ptr, (int)D, sQ, (int)(B * H));
#else
        rocblas_sgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
            D, Tq, Tk, &one, k_ptr, D, sK, ds_ptr, Tk, sS, &zero,
            dq_ptr, D, sQ, B * H);
#endif
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hk = 0; hk < H_kv; ++hk) {
#ifndef __HIP_PLATFORM_AMD__
            cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)D, (int)Tq, (int)Tk, &one,
                k_ptr + (b * H_kv + hk) * Tk * D, (int)D, 0,
                ds_ptr + (b * H + hk * G) * Tq * Tk, (int)Tk, sS, &zero,
                dq_ptr + (b * H + hk * G) * Tq * D, (int)D, sQ, (int)G);
#else
            rocblas_sgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
                D, Tq, Tk, &one,
                k_ptr + (b * H_kv + hk) * Tk * D, D, 0,
                ds_ptr + (b * H + hk * G) * Tq * Tk, Tk, sS, &zero,
                dq_ptr + (b * H + hk * G) * Tq * D, D, sQ, G);
#endif
        }
    }
}

// dK[b,hk] += sum_g Q[b,hk*G+g]^T @ dS[b,hk*G+g] (accumulate, zeroed first)
static void gqa_gemm_dk_f32(GqaHandle handle,
    const float* q_ptr, const float* ds_ptr, float* dk_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    float one = 1.0f, zero = 0.0f;
    GPU_API(MemsetAsync)(dk_ptr, 0, (size_t)B * H_kv * Tk * D * sizeof(float),
                         dev_stream(current_device()));
    long long sQ = Tq * D, sS = Tq * Tk;
    if (H == H_kv) {
        long long sK = Tk * D;
#ifndef __HIP_PLATFORM_AMD__
        cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            (int)D, (int)Tk, (int)Tq, &one,
            q_ptr, (int)D, sQ, ds_ptr, (int)Tk, sS, &zero,
            dk_ptr, (int)D, sK, (int)(B * H));
#else
        rocblas_sgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_transpose,
            D, Tk, Tq, &one, q_ptr, D, sQ, ds_ptr, Tk, sS, &zero,
            dk_ptr, D, sK, B * H);
#endif
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t hk = h / G;
#ifndef __HIP_PLATFORM_AMD__
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)D, (int)Tk, (int)Tq, &one,
                q_ptr + (b * H + h) * Tq * D, (int)D,
                ds_ptr + (b * H + h) * Tq * Tk, (int)Tk,
                &one, dk_ptr + (b * H_kv + hk) * Tk * D, (int)D);
#else
            rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                D, Tk, Tq, &one,
                q_ptr + (b * H + h) * Tq * D, D,
                ds_ptr + (b * H + h) * Tq * Tk, Tk,
                &one, dk_ptr + (b * H_kv + hk) * Tk * D, D);
#endif
        }
    }
}

#ifdef __HIP_PLATFORM_AMD__
// HIP half-precision GQA helpers (rocblas)
static void gqa_gemm_s_f16(rocblas_handle handle,
    const __half* k_ptr, const __half* q_ptr, __half* s_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D,
    float alpha) {
    __half h_alpha = __float2half(alpha);
    __half h_beta = __float2half(0.0f);
    long long sQ = Tq * D, sS = Tq * Tk;
    if (H == H_kv) {
        long long sK = Tk * D;
        rocblas_hgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none,
            Tk, Tq, D, &h_alpha, k_ptr, D, sK, q_ptr, D, sQ, &h_beta,
            s_ptr, Tk, sS, B * H);
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hk = 0; hk < H_kv; ++hk) {
            rocblas_hgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none,
                Tk, Tq, D, &h_alpha,
                k_ptr + (b * H_kv + hk) * Tk * D, D, 0,
                q_ptr + (b * H + hk * G) * Tq * D, D, sQ, &h_beta,
                s_ptr + (b * H + hk * G) * Tq * Tk, Tk, sS, G);
        }
    }
}

static void gqa_gemm_o_f16(rocblas_handle handle,
    const __half* v_ptr, const __half* p_ptr, __half* o_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    __half h_one = __float2half(1.0f), h_zero = __float2half(0.0f);
    long long sS = Tq * Tk, sO = Tq * D;
    if (H == H_kv) {
        long long sV = Tk * D;
        rocblas_hgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
            D, Tq, Tk, &h_one, v_ptr, D, sV, p_ptr, Tk, sS, &h_zero,
            o_ptr, D, sO, B * H);
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hk = 0; hk < H_kv; ++hk) {
            rocblas_hgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
                D, Tq, Tk, &h_one,
                v_ptr + (b * H_kv + hk) * Tk * D, D, 0,
                p_ptr + (b * H + hk * G) * Tq * Tk, Tk, sS, &h_zero,
                o_ptr + (b * H + hk * G) * Tq * D, D, sO, G);
        }
    }
}

static void gqa_gemm_dv_f16(rocblas_handle handle,
    const __half* do_ptr, const __half* p_ptr, __half* dv_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    __half h_one = __float2half(1.0f), h_zero = __float2half(0.0f);
    GPU_API(MemsetAsync)(dv_ptr, 0, (size_t)B * H_kv * Tk * D * sizeof(__half),
                         dev_stream(current_device()));
    long long sO = Tq * D, sS = Tq * Tk;
    if (H == H_kv) {
        long long sV = Tk * D;
        rocblas_hgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_transpose,
            D, Tk, Tq, &h_one, do_ptr, D, sO, p_ptr, Tk, sS, &h_zero,
            dv_ptr, D, sV, B * H);
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t hk = h / G;
            rocblas_hgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                D, Tk, Tq, &h_one,
                do_ptr + (b * H + h) * Tq * D, D,
                p_ptr + (b * H + h) * Tq * Tk, Tk,
                &h_one, dv_ptr + (b * H_kv + hk) * Tk * D, D);
        }
    }
}

static void gqa_gemm_ds_f16(rocblas_handle handle,
    const __half* v_ptr, const __half* do_ptr, __half* ds_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    __half h_one = __float2half(1.0f), h_zero = __float2half(0.0f);
    long long sO = Tq * D, sS = Tq * Tk;
    if (H == H_kv) {
        long long sV = Tk * D;
        rocblas_hgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none,
            Tk, Tq, D, &h_one, v_ptr, D, sV, do_ptr, D, sO, &h_zero,
            ds_ptr, Tk, sS, B * H);
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hk = 0; hk < H_kv; ++hk) {
            rocblas_hgemm_strided_batched(handle, rocblas_operation_transpose, rocblas_operation_none,
                Tk, Tq, D, &h_one,
                v_ptr + (b * H_kv + hk) * Tk * D, D, 0,
                do_ptr + (b * H + hk * G) * Tq * D, D, sO, &h_zero,
                ds_ptr + (b * H + hk * G) * Tq * Tk, Tk, sS, G);
        }
    }
}

static void gqa_gemm_dq_f16(rocblas_handle handle,
    const __half* k_ptr, const __half* ds_ptr, __half* dq_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    __half h_one = __float2half(1.0f), h_zero = __float2half(0.0f);
    long long sS = Tq * Tk, sQ = Tq * D;
    if (H == H_kv) {
        long long sK = Tk * D;
        rocblas_hgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
            D, Tq, Tk, &h_one, k_ptr, D, sK, ds_ptr, Tk, sS, &h_zero,
            dq_ptr, D, sQ, B * H);
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hk = 0; hk < H_kv; ++hk) {
            rocblas_hgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
                D, Tq, Tk, &h_one,
                k_ptr + (b * H_kv + hk) * Tk * D, D, 0,
                ds_ptr + (b * H + hk * G) * Tq * Tk, Tk, sS, &h_zero,
                dq_ptr + (b * H + hk * G) * Tq * D, D, sQ, G);
        }
    }
}

static void gqa_gemm_dk_f16(rocblas_handle handle,
    const __half* q_ptr, const __half* ds_ptr, __half* dk_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    __half h_one = __float2half(1.0f), h_zero = __float2half(0.0f);
    GPU_API(MemsetAsync)(dk_ptr, 0, (size_t)B * H_kv * Tk * D * sizeof(__half),
                         dev_stream(current_device()));
    long long sQ = Tq * D, sS = Tq * Tk;
    if (H == H_kv) {
        long long sK = Tk * D;
        rocblas_hgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_transpose,
            D, Tk, Tq, &h_one, q_ptr, D, sQ, ds_ptr, Tk, sS, &h_zero,
            dk_ptr, D, sK, B * H);
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t hk = h / G;
            rocblas_hgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                D, Tk, Tq, &h_one,
                q_ptr + (b * H + h) * Tq * D, D,
                ds_ptr + (b * H + h) * Tq * Tk, Tk,
                &h_one, dk_ptr + (b * H_kv + hk) * Tk * D, D);
        }
    }
}
#else
// Half-precision helpers (CUDA only; HIP uses float path via separate kernels)
static void gqa_gemm_s_f16(cublasHandle_t handle,
    const __half* k_ptr, const __half* q_ptr, __half* s_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D,
    float alpha) {
    __half h_alpha = __float2half(alpha);
    __half h_beta = __float2half(0.0f);
    long long sQ = Tq * D, sS = Tq * Tk;
    if (H == H_kv) {
        long long sK = Tk * D;
        cublasHgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)Tk, (int)Tq, (int)D, &h_alpha,
            k_ptr, (int)D, sK, q_ptr, (int)D, sQ, &h_beta,
            s_ptr, (int)Tk, sS, (int)(B * H));
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hk = 0; hk < H_kv; ++hk) {
            cublasHgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                (int)Tk, (int)Tq, (int)D, &h_alpha,
                k_ptr + (b * H_kv + hk) * Tk * D, (int)D, 0,
                q_ptr + (b * H + hk * G) * Tq * D, (int)D, sQ, &h_beta,
                s_ptr + (b * H + hk * G) * Tq * Tk, (int)Tk, sS, (int)G);
        }
    }
}

static void gqa_gemm_o_f16(cublasHandle_t handle,
    const __half* v_ptr, const __half* p_ptr, __half* o_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    __half h_one = __float2half(1.0f), h_zero = __float2half(0.0f);
    long long sS = Tq * Tk, sO = Tq * D;
    if (H == H_kv) {
        long long sV = Tk * D;
        cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            (int)D, (int)Tq, (int)Tk, &h_one,
            v_ptr, (int)D, sV, p_ptr, (int)Tk, sS, &h_zero,
            o_ptr, (int)D, sO, (int)(B * H));
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hk = 0; hk < H_kv; ++hk) {
            cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)D, (int)Tq, (int)Tk, &h_one,
                v_ptr + (b * H_kv + hk) * Tk * D, (int)D, 0,
                p_ptr + (b * H + hk * G) * Tq * Tk, (int)Tk, sS, &h_zero,
                o_ptr + (b * H + hk * G) * Tq * D, (int)D, sO, (int)G);
        }
    }
}

static void gqa_gemm_dv_f16(cublasHandle_t handle,
    const __half* do_ptr, const __half* p_ptr, __half* dv_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    __half h_one = __float2half(1.0f), h_zero = __float2half(0.0f);
    GPU_API(MemsetAsync)(dv_ptr, 0, (size_t)B * H_kv * Tk * D * sizeof(__half),
                         dev_stream(current_device()));
    long long sO = Tq * D, sS = Tq * Tk;
    if (H == H_kv) {
        long long sV = Tk * D;
        cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            (int)D, (int)Tk, (int)Tq, &h_one,
            do_ptr, (int)D, sO, p_ptr, (int)Tk, sS, &h_zero,
            dv_ptr, (int)D, sV, (int)(B * H));
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t hk = h / G;
            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)D, (int)Tk, (int)Tq, &h_one,
                do_ptr + (b * H + h) * Tq * D, (int)D,
                p_ptr + (b * H + h) * Tq * Tk, (int)Tk,
                &h_one, dv_ptr + (b * H_kv + hk) * Tk * D, (int)D);
        }
    }
}

static void gqa_gemm_ds_f16(cublasHandle_t handle,
    const __half* v_ptr, const __half* do_ptr, __half* ds_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    __half h_one = __float2half(1.0f), h_zero = __float2half(0.0f);
    long long sO = Tq * D, sS = Tq * Tk;
    if (H == H_kv) {
        long long sV = Tk * D;
        cublasHgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)Tk, (int)Tq, (int)D, &h_one,
            v_ptr, (int)D, sV, do_ptr, (int)D, sO, &h_zero,
            ds_ptr, (int)Tk, sS, (int)(B * H));
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hk = 0; hk < H_kv; ++hk) {
            cublasHgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                (int)Tk, (int)Tq, (int)D, &h_one,
                v_ptr + (b * H_kv + hk) * Tk * D, (int)D, 0,
                do_ptr + (b * H + hk * G) * Tq * D, (int)D, sO, &h_zero,
                ds_ptr + (b * H + hk * G) * Tq * Tk, (int)Tk, sS, (int)G);
        }
    }
}

static void gqa_gemm_dq_f16(cublasHandle_t handle,
    const __half* k_ptr, const __half* ds_ptr, __half* dq_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    __half h_one = __float2half(1.0f), h_zero = __float2half(0.0f);
    long long sS = Tq * Tk, sQ = Tq * D;
    if (H == H_kv) {
        long long sK = Tk * D;
        cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            (int)D, (int)Tq, (int)Tk, &h_one,
            k_ptr, (int)D, sK, ds_ptr, (int)Tk, sS, &h_zero,
            dq_ptr, (int)D, sQ, (int)(B * H));
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hk = 0; hk < H_kv; ++hk) {
            cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)D, (int)Tq, (int)Tk, &h_one,
                k_ptr + (b * H_kv + hk) * Tk * D, (int)D, 0,
                ds_ptr + (b * H + hk * G) * Tq * Tk, (int)Tk, sS, &h_zero,
                dq_ptr + (b * H + hk * G) * Tq * D, (int)D, sQ, (int)G);
        }
    }
}

static void gqa_gemm_dk_f16(cublasHandle_t handle,
    const __half* q_ptr, const __half* ds_ptr, __half* dk_ptr,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D) {
    __half h_one = __float2half(1.0f), h_zero = __float2half(0.0f);
    GPU_API(MemsetAsync)(dk_ptr, 0, (size_t)B * H_kv * Tk * D * sizeof(__half),
                         dev_stream(current_device()));
    long long sQ = Tq * D, sS = Tq * Tk;
    if (H == H_kv) {
        long long sK = Tk * D;
        cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            (int)D, (int)Tk, (int)Tq, &h_one,
            q_ptr, (int)D, sQ, ds_ptr, (int)Tk, sS, &h_zero,
            dk_ptr, (int)D, sK, (int)(B * H));
        return;
    }
    int64_t G = H / H_kv;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t hk = h / G;
            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)D, (int)Tk, (int)Tq, &h_one,
                q_ptr + (b * H + h) * Tq * D, (int)D,
                ds_ptr + (b * H + h) * Tq * Tk, (int)Tk,
                &h_one, dk_ptr + (b * H_kv + hk) * Tk * D, (int)D);
        }
    }
}
#endif

extern "C" void gpu_flash_attention(void *Q, int64_t q_off, void *K, int64_t k_off,
                                    void *V, int64_t v_off, void *O, int64_t o_off,
                                    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk,
                                    int64_t D, float scale, int causal) {
    auto_set_device(Q);
    if (g_fa3_fwd_fn) {
        g_fa3_fwd_fn((const char*)Q + q_off * sizeof(float), (const char*)K + k_off * sizeof(float),
                     (const char*)V + v_off * sizeof(float), (char*)O + o_off * sizeof(float),
                     B, H, H_kv, Tq, Tk, D, scale, (void*)dev_stream(current_device()), causal);
        return;
    }

    launch_fa_fwd<float>((const float*)Q + q_off, (const float*)K + k_off, (const float*)V + v_off,
                         (float*)O + o_off, (float*)nullptr,
                         B, H, H_kv, Tq, Tk, D, scale, causal);
}

extern "C" int gpu_flash_attention_forward_save_lse(void *Q, int64_t q_off, void *K, int64_t k_off,
                                    void *V, int64_t v_off, void *O, int64_t o_off,
                                    void *LSE, int64_t lse_off,
                                    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk,
                                    int64_t D, float scale, int causal) {
    auto_set_device(Q);
    if (g_fa3_fwd_fn) {
        g_fa3_fwd_fn((const char*)Q + q_off * sizeof(float), (const char*)K + k_off * sizeof(float),
                     (const char*)V + v_off * sizeof(float), (char*)O + o_off * sizeof(float),
                     B, H, H_kv, Tq, Tk, D, scale, (void*)dev_stream(current_device()), causal);
        return 0;
    }
    launch_fa_fwd<float>((const float*)Q + q_off, (const float*)K + k_off, (const float*)V + v_off,
                         (float*)O + o_off, (float*)LSE + lse_off,
                         B, H, H_kv, Tq, Tk, D, scale, causal);
    return 1;
}

extern "C" void gpu_flash_attention_half(void *Q, int64_t q_off, void *K, int64_t k_off,
                                         void *V, int64_t v_off, void *O, int64_t o_off,
                                         int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk,
                                         int64_t D, float scale, int causal) {
    auto_set_device(Q);
    if (g_fa3_fwd_fn) {
        g_fa3_fwd_fn((const char*)Q + q_off * sizeof(__half), (const char*)K + k_off * sizeof(__half),
                     (const char*)V + v_off * sizeof(__half), (char*)O + o_off * sizeof(__half),
                     B, H, H_kv, Tq, Tk, D, scale, (void*)dev_stream(current_device()), causal);
        return;
    }

    launch_fa_fwd<__half>((const __half*)Q + q_off, (const __half*)K + k_off, (const __half*)V + v_off,
                          (__half*)O + o_off, (float*)nullptr,
                          B, H, H_kv, Tq, Tk, D, scale, causal);
}

extern "C" int gpu_flash_attention_half_forward_save_lse(void *Q, int64_t q_off, void *K, int64_t k_off,
                                    void *V, int64_t v_off, void *O, int64_t o_off,
                                    void *LSE, int64_t lse_off,
                                    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk,
                                    int64_t D, float scale, int causal) {
    auto_set_device(Q);
    if (g_fa3_fwd_fn) {
        g_fa3_fwd_fn((const char*)Q + q_off * sizeof(__half), (const char*)K + k_off * sizeof(__half),
                     (const char*)V + v_off * sizeof(__half), (char*)O + o_off * sizeof(__half),
                     B, H, H_kv, Tq, Tk, D, scale, (void*)dev_stream(current_device()), causal);
        return 0;
    }
    launch_fa_fwd<__half>((const __half*)Q + q_off, (const __half*)K + k_off, (const __half*)V + v_off,
                          (__half*)O + o_off, (float*)LSE + lse_off,
                          B, H, H_kv, Tq, Tk, D, scale, causal);
    return 1;
}

extern "C" void gpu_flash_attention_backward(void *dQ, int64_t dq_off, void *dK, int64_t dk_off,
                                             void *dV, int64_t dv_off, void *O, int64_t o_off, void *dO,
                                             int64_t do_off, void *Q, int64_t q_off, void *K, int64_t k_off,
                                             void *V, int64_t v_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq,
                                             int64_t Tk, int64_t D, float scale, int causal) {
    auto_set_device(dQ);
    if (g_fa3_bwd_fn) {
        g_fa3_bwd_fn((char*)dQ + dq_off * sizeof(float), (char*)dK + dk_off * sizeof(float),
                     (char*)dV + dv_off * sizeof(float), (const char*)O + o_off * sizeof(float),
                     (const char*)dO + do_off * sizeof(float), (const char*)Q + q_off * sizeof(float),
                     (const char*)K + k_off * sizeof(float), (const char*)V + v_off * sizeof(float),
                     B, H, H_kv, Tq, Tk, D, scale, (void*)dev_stream(current_device()), causal);
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
    cublasHandle_t handle = get_cublas_handle(dq_ptr);
    float alpha = scale;

    gqa_gemm_s_f32(handle, k_ptr, q_ptr, g_attn_scratch_s,
                   B, H, H_kv, Tq, Tk, D, alpha);

    int total_rows = B * H * Tq;
    launch_softmax_fwd(g_attn_scratch_s, g_attn_scratch_p, total_rows, Tk, (int)Tq, causal != 0);

    gqa_gemm_dv_f32(handle, do_ptr, g_attn_scratch_p, dv_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_ds_f32(handle, v_ptr, do_ptr, g_attn_scratch_dp,
                    B, H, H_kv, Tq, Tk, D);

    launch_softmax_bwd(g_attn_scratch_dp, g_attn_scratch_p, g_attn_scratch_ds, total_rows, Tk, scale);

    gqa_gemm_dq_f32(handle, k_ptr, g_attn_scratch_ds, dq_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_dk_f32(handle, q_ptr, g_attn_scratch_ds, dk_ptr,
                    B, H, H_kv, Tq, Tk, D);
#else
    rocblas_handle handle = get_rocblas_handle();
    float alpha = scale;

    gqa_gemm_s_f32(handle, k_ptr, q_ptr, g_attn_scratch_s,
                   B, H, H_kv, Tq, Tk, D, alpha);

    int total_rows = B * H * Tq;
    launch_softmax_fwd_hip(g_attn_scratch_s, g_attn_scratch_p, total_rows, Tk, (int)Tq, causal != 0);

    gqa_gemm_dv_f32(handle, do_ptr, g_attn_scratch_p, dv_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_ds_f32(handle, v_ptr, do_ptr, g_attn_scratch_dp,
                    B, H, H_kv, Tq, Tk, D);

    launch_softmax_bwd_hip(g_attn_scratch_dp, g_attn_scratch_p, g_attn_scratch_ds, total_rows, Tk, scale);

    gqa_gemm_dq_f32(handle, k_ptr, g_attn_scratch_ds, dq_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_dk_f32(handle, q_ptr, g_attn_scratch_ds, dk_ptr,
                    B, H, H_kv, Tq, Tk, D);
#endif
}

extern "C" void gpu_flash_attention_backward_with_lse(void *dQ, int64_t dq_off, void *dK, int64_t dk_off,
                                             void *dV, int64_t dv_off, void *O, int64_t o_off, void *dO,
                                             int64_t do_off, void *Q, int64_t q_off, void *K, int64_t k_off,
                                             void *V, int64_t v_off, void *LSE, int64_t lse_off,
                                             int64_t B, int64_t H, int64_t H_kv, int64_t Tq,
                                             int64_t Tk, int64_t D, float scale, int causal) {
    auto_set_device(dQ);
    size_t total_elements = (size_t)B * H * Tq * Tk;
    ensure_attn_scratch(total_elements);

    float* dq_ptr = (float*)dQ + dq_off;
    float* dk_ptr = (float*)dK + dk_off;
    float* dv_ptr = (float*)dV + dv_off;
    const float* do_ptr = (const float*)dO + do_off;
    const float* q_ptr = (const float*)Q + q_off;
    const float* k_ptr = (const float*)K + k_off;
    const float* v_ptr = (const float*)V + v_off;
    const float* lse_ptr = (const float*)LSE + lse_off;

#ifndef __HIP_PLATFORM_AMD__
    cublasHandle_t handle = get_cublas_handle(dq_ptr);

    int total_rows = B * H * Tq;

    gqa_gemm_s_f32(handle, k_ptr, q_ptr, g_attn_scratch_s,
                   B, H, H_kv, Tq, Tk, D, scale);

    launch_p_from_lse(g_attn_scratch_s, lse_ptr, g_attn_scratch_p,
                      (int64_t)total_elements, (int)Tk, (int)Tq, causal);

    gqa_gemm_dv_f32(handle, do_ptr, g_attn_scratch_p, dv_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_ds_f32(handle, v_ptr, do_ptr, g_attn_scratch_dp,
                    B, H, H_kv, Tq, Tk, D);

    launch_softmax_bwd(g_attn_scratch_dp, g_attn_scratch_p, g_attn_scratch_ds, total_rows, Tk, scale);

    gqa_gemm_dq_f32(handle, k_ptr, g_attn_scratch_ds, dq_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_dk_f32(handle, q_ptr, g_attn_scratch_ds, dk_ptr,
                    B, H, H_kv, Tq, Tk, D);
#else
    rocblas_handle handle = get_rocblas_handle();

    int total_rows = B * H * Tq;

    gqa_gemm_s_f32(handle, k_ptr, q_ptr, g_attn_scratch_s,
                   B, H, H_kv, Tq, Tk, D, scale);

    launch_p_from_lse(g_attn_scratch_s, lse_ptr, g_attn_scratch_p,
                      (int64_t)total_elements, (int)Tk, (int)Tq, causal);

    gqa_gemm_dv_f32(handle, do_ptr, g_attn_scratch_p, dv_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_ds_f32(handle, v_ptr, do_ptr, g_attn_scratch_dp,
                    B, H, H_kv, Tq, Tk, D);

    launch_softmax_bwd_hip(g_attn_scratch_dp, g_attn_scratch_p, g_attn_scratch_ds, total_rows, Tk, scale);

    gqa_gemm_dq_f32(handle, k_ptr, g_attn_scratch_ds, dq_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_dk_f32(handle, q_ptr, g_attn_scratch_ds, dk_ptr,
                    B, H, H_kv, Tq, Tk, D);
#endif
}

extern "C" void gpu_flash_attention_backward_half(
    void *dQ, int64_t dq_off, void *dK, int64_t dk_off, void *dV, int64_t dv_off, void *O,
    int64_t o_off, void *dO, int64_t do_off, void *Q, int64_t q_off, void *K, int64_t k_off,
    void *V, int64_t v_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D,
    float scale, int causal) {
    auto_set_device(dQ);
    if (g_fa3_bwd_fn) {
        g_fa3_bwd_fn((char*)dQ + dq_off * sizeof(__half), (char*)dK + dk_off * sizeof(__half),
                     (char*)dV + dv_off * sizeof(__half), (const char*)O + o_off * sizeof(__half),
                     (const char*)dO + do_off * sizeof(__half), (const char*)Q + q_off * sizeof(__half),
                     (const char*)K + k_off * sizeof(__half), (const char*)V + v_off * sizeof(__half),
                     B, H, H_kv, Tq, Tk, D, scale, (void*)dev_stream(current_device()), causal);
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
    cublasHandle_t handle = get_cublas_handle(dq_ptr);

    gqa_gemm_s_f16(handle, k_ptr, q_ptr, s_half,
                   B, H, H_kv, Tq, Tk, D, scale);

    int total_rows = B * H * Tq;
    launch_softmax_fwd_half(s_half, p_half, total_rows, Tk, (int)Tq, causal != 0);

    gqa_gemm_dv_f16(handle, do_ptr, p_half, dv_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_ds_f16(handle, v_ptr, do_ptr, dp_half,
                    B, H, H_kv, Tq, Tk, D);

    launch_softmax_bwd_half(dp_half, p_half, ds_half, total_rows, Tk, scale);

    gqa_gemm_dq_f16(handle, k_ptr, ds_half, dq_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_dk_f16(handle, q_ptr, ds_half, dk_ptr,
                    B, H, H_kv, Tq, Tk, D);
#else
    rocblas_handle handle = get_rocblas_handle();

    gqa_gemm_s_f16(handle, k_ptr, q_ptr, s_half,
                   B, H, H_kv, Tq, Tk, D, scale);

    int total_rows = B * H * Tq;
    launch_softmax_fwd_half_hip(s_half, p_half, total_rows, Tk, (int)Tq, causal != 0);

    gqa_gemm_dv_f16(handle, do_ptr, p_half, dv_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_ds_f16(handle, v_ptr, do_ptr, dp_half,
                    B, H, H_kv, Tq, Tk, D);

    launch_softmax_bwd_half_hip(dp_half, p_half, ds_half, total_rows, Tk, scale);

    gqa_gemm_dq_f16(handle, k_ptr, ds_half, dq_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_dk_f16(handle, q_ptr, ds_half, dk_ptr,
                    B, H, H_kv, Tq, Tk, D);
#endif
}

extern "C" void gpu_flash_attention_backward_half_with_lse(
    void *dQ, int64_t dq_off, void *dK, int64_t dk_off, void *dV, int64_t dv_off, void *O,
    int64_t o_off, void *dO, int64_t do_off, void *Q, int64_t q_off, void *K, int64_t k_off,
    void *V, int64_t v_off, void *LSE, int64_t lse_off,
    int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D,
    float scale, int causal) {
    auto_set_device(dQ);
    size_t total_elements = (size_t)B * H * Tq * Tk;
    ensure_attn_scratch(total_elements);

    __half* dq_ptr = (__half*)dQ + dq_off;
    __half* dk_ptr = (__half*)dK + dk_off;
    __half* dv_ptr = (__half*)dV + dv_off;
    const __half* do_ptr = (const __half*)dO + do_off;
    const __half* q_ptr = (const __half*)Q + q_off;
    const __half* k_ptr = (const __half*)K + k_off;
    const __half* v_ptr = (const __half*)V + v_off;
    const float* lse_ptr = (const float*)LSE + lse_off;
    __half* s_half = (__half*)g_attn_scratch_s;
    __half* p_half = (__half*)g_attn_scratch_p;
    __half* dp_half = (__half*)g_attn_scratch_dp;
    __half* ds_half = (__half*)g_attn_scratch_ds;

#ifndef __HIP_PLATFORM_AMD__
    cublasHandle_t handle = get_cublas_handle(dq_ptr);

    int total_rows = B * H * Tq;

    gqa_gemm_s_f16(handle, k_ptr, q_ptr, s_half,
                   B, H, H_kv, Tq, Tk, D, scale);

    launch_p_from_lse_half(s_half, lse_ptr, p_half,
                           (int64_t)total_elements, (int)Tk, (int)Tq, causal);

    gqa_gemm_dv_f16(handle, do_ptr, p_half, dv_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_ds_f16(handle, v_ptr, do_ptr, dp_half,
                    B, H, H_kv, Tq, Tk, D);

    launch_softmax_bwd_half(dp_half, p_half, ds_half, total_rows, Tk, scale);

    gqa_gemm_dq_f16(handle, k_ptr, ds_half, dq_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_dk_f16(handle, q_ptr, ds_half, dk_ptr,
                    B, H, H_kv, Tq, Tk, D);
#else
    rocblas_handle handle = get_rocblas_handle();

    int total_rows = B * H * Tq;

    gqa_gemm_s_f16(handle, k_ptr, q_ptr, s_half,
                   B, H, H_kv, Tq, Tk, D, scale);

    launch_p_from_lse_half(s_half, lse_ptr, p_half,
                           (int64_t)total_elements, (int)Tk, (int)Tq, causal);

    gqa_gemm_dv_f16(handle, do_ptr, p_half, dv_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_ds_f16(handle, v_ptr, do_ptr, dp_half,
                    B, H, H_kv, Tq, Tk, D);

    launch_softmax_bwd_half_hip(dp_half, p_half, ds_half, total_rows, Tk, scale);

    gqa_gemm_dq_f16(handle, k_ptr, ds_half, dq_ptr,
                    B, H, H_kv, Tq, Tk, D);

    gqa_gemm_dk_f16(handle, q_ptr, ds_half, dk_ptr,
                    B, H, H_kv, Tq, Tk, D);
#endif
}

__global__ void qkv_split_transpose_kernel(const float* qkv, float* q, float* k, float* v,
                                           int B, int T, int H, int D) {
    int C = H * D;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;
    if (idx >= total) return;
    int d = idx % D;
    int t = (idx / D) % T;
    int h = (idx / (D * T)) % H;
    int b = idx / (D * T * H);
    int c = h * D + d;
    int base = (b * T + t) * 3 * C + c;
    q[idx] = qkv[base];
    k[idx] = qkv[base + C];
    v[idx] = qkv[base + 2 * C];
}

__global__ void qkv_split_transpose_backward_kernel(const float* gq, const float* gk, const float* gv,
                                                    float* gqkv, int B, int T, int H, int D) {
    int C = H * D;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * 3 * C;
    if (idx >= total) return;
    int c3 = idx % (3 * C);
    int s = c3 / C;
    int c = c3 % C;
    int t = (idx / (3 * C)) % T;
    int b = idx / (3 * C * T);
    int h = c / D;
    int d = c % D;
    int src = ((b * H + h) * T + t) * D + d;
    gqkv[idx] = (s == 0) ? gq[src] : ((s == 1) ? gk[src] : gv[src]);
}

extern "C" void gpu_qkv_split_transpose(void* QKV, int64_t qkv_off, void* Q, int64_t q_off,
                                        void* K, int64_t k_off, void* V, int64_t v_off,
                                        int64_t B, int64_t T, int64_t H, int64_t D) {
    auto_set_device(QKV);
    const float* qkv_ptr = (const float*)QKV + qkv_off;
    float* q_ptr = (float*)Q + q_off;
    float* k_ptr = (float*)K + k_off;
    float* v_ptr = (float*)V + v_off;
    int total = (int)(B * T * H * D);
    int block = 256;
    int grid = (total + block - 1) / block;
    qkv_split_transpose_kernel<<<grid, block, 0, dev_stream(current_device())>>>(qkv_ptr, q_ptr, k_ptr, v_ptr,
                                                                     (int)B, (int)T, (int)H, (int)D);
}

extern "C" void gpu_qkv_split_transpose_backward(void* GQ, int64_t gq_off, void* GK, int64_t gk_off,
                                                 void* GV, int64_t gv_off, void* GQKV, int64_t gqkv_off,
                                                 int64_t B, int64_t T, int64_t H, int64_t D) {
    auto_set_device(GQ);
    const float* gq_ptr = (const float*)GQ + gq_off;
    const float* gk_ptr = (const float*)GK + gk_off;
    const float* gv_ptr = (const float*)GV + gv_off;
    float* gqkv_ptr = (float*)GQKV + gqkv_off;
    int total = (int)(B * T * 3 * H * D);
    int block = 256;
    int grid = (total + block - 1) / block;
    qkv_split_transpose_backward_kernel<<<grid, block, 0, dev_stream(current_device())>>>(gq_ptr, gk_ptr, gv_ptr, gqkv_ptr,
                                                                              (int)B, (int)T, (int)H, (int)D);
}

__global__ void qkv_extract_kernel(const float* qkv, float* out, int B, int T, int H, int D, int idx) {
    int C = H * D;
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;
    if (g >= total) return;
    int d = g % D;
    int t = (g / D) % T;
    int h = (g / (D * T)) % H;
    int b = g / (D * T * H);
    int c = h * D + d;
    out[g] = qkv[(b * T + t) * 3 * C + idx * C + c];
}

__global__ void qkv_extract_backward_kernel(const float* gout, float* gqkv,
                                            int B, int T, int H, int D, int idx) {
    int C = H * D;
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * 3 * C;
    if (g >= total) return;
    int c3 = g % (3 * C);
    int s = c3 / C;
    if (s != idx) {
        gqkv[g] = 0.0f;
        return;
    }
    int c = c3 % C;
    int t = (g / (3 * C)) % T;
    int b = g / (3 * C * T);
    int h = c / D;
    int d = c % D;
    gqkv[g] = gout[((b * H + h) * T + t) * D + d];
}

extern "C" void gpu_qkv_extract(void* QKV, int64_t qkv_off, void* OUT, int64_t out_off,
                                int64_t B, int64_t T, int64_t H, int64_t D, int64_t idx) {
    auto_set_device(QKV);
    const float* qkv_ptr = (const float*)QKV + qkv_off;
    float* out_ptr = (float*)OUT + out_off;
    int total = (int)(B * T * H * D);
    int block = 256;
    int grid = (total + block - 1) / block;
    qkv_extract_kernel<<<grid, block, 0, dev_stream(current_device())>>>(qkv_ptr, out_ptr,
                                                             (int)B, (int)T, (int)H, (int)D, (int)idx);
}

extern "C" void gpu_qkv_extract_backward(void* GOUT, int64_t gout_off, void* GQKV, int64_t gqkv_off,
                                          int64_t B, int64_t T, int64_t H, int64_t D, int64_t idx) {
    auto_set_device(GOUT);
    const float* gout_ptr = (const float*)GOUT + gout_off;
    float* gqkv_ptr = (float*)GQKV + gqkv_off;
    int total = (int)(B * T * 3 * H * D);
    int block = 256;
    int grid = (total + block - 1) / block;
    qkv_extract_backward_kernel<<<grid, block, 0, dev_stream(current_device())>>>(gout_ptr, gqkv_ptr,
                                                                      (int)B, (int)T, (int)H, (int)D, (int)idx);
}
