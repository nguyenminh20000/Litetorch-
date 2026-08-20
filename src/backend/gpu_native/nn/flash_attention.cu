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

__global__ void flash_attn_fwd_kernel(
    const float *Q, const float *K, const float *V, float *O, int B, int H,
    int H_kv, int Tq, int Tk, int D, float scale, int q_stride_b,
    int q_stride_h, int q_stride_t, int k_stride_b, int k_stride_h,
    int k_stride_t, int v_stride_b, int v_stride_h, int v_stride_t,
    int o_stride_b, int o_stride_h, int o_stride_t) {
  int b = blockIdx.x;
  int h = blockIdx.y;
  int h_kv = h / (H / H_kv);
  int t_tile = blockIdx.z;

  const int block_size = 128;
  int q_start = t_tile * block_size;
  if (q_start >= Tq)
    return;
  int q_end = min(q_start + block_size, Tq);
  int q_len = q_end - q_start;

  int tid = threadIdx.x;

  extern __shared__ float s_mem[];
  float *s_Q = s_mem;
  float *s_K = s_Q + block_size * D;
  float *s_V = s_K + block_size * D;

  float r_O[128] = {0.0f};
  float r_m = -1e37f;
  float r_l = 0.0f;

  for (int i = tid; i < q_len * D; i += blockDim.x) {
    int t = i / D;
    int d = i % D;
    s_Q[t * D + d] =
        Q[b * q_stride_b + h * q_stride_h + (q_start + t) * q_stride_t + d];
  }
  __syncthreads();

  for (int k_tile = 0; k_tile < (Tk + block_size - 1) / block_size; ++k_tile) {
    int k_start = k_tile * block_size;
    int k_end = min(k_start + block_size, Tk);
    int k_len = k_end - k_start;

    for (int i = tid; i < k_len * D; i += blockDim.x) {
      int t = i / D;
      int d = i % D;
      s_K[t * D + d] = K[b * k_stride_b + h_kv * k_stride_h +
                         (k_start + t) * k_stride_t + d];
      s_V[t * D + d] = V[b * v_stride_b + h_kv * v_stride_h +
                         (k_start + t) * v_stride_t + d];
    }
    __syncthreads();

    if (tid < q_len) {
      int q_idx = tid;
      float m_prev = r_m;
      float m_new = r_m;

      for (int k_idx = 0; k_idx < k_len; ++k_idx) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
          score += s_Q[q_idx * D + d] * s_K[k_idx * D + d];
        }
        score *= scale;
        if (score > m_new)
          m_new = score;
      }

      float exp_prev = expf(m_prev - m_new);
      float l_new = r_l * exp_prev;

      float temp_O[128] = {0.0f};
      for (int k_idx = 0; k_idx < k_len; ++k_idx) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
          score += s_Q[q_idx * D + d] * s_K[k_idx * D + d];
        }
        score *= scale;
        float p = expf(score - m_new);
        l_new += p;

        for (int d = 0; d < D && d < 128; ++d) {
          temp_O[d] += p * s_V[k_idx * D + d];
        }
      }

      for (int d = 0; d < D && d < 128; ++d) {
        r_O[d] = r_O[d] * exp_prev + temp_O[d];
      }
      r_m = m_new;
      r_l = l_new;
    }
    __syncthreads();
  }

  if (tid < q_len) {
    int q_idx = tid;
    float inv_l = 1.0f / (r_l + 1e-6f);
    for (int d = 0; d < D; ++d) {
      O[b * o_stride_b + h * o_stride_h + (q_start + q_idx) * o_stride_t + d] =
          r_O[d] * inv_l;
    }
  }
}

__global__ void flash_attn_fwd_half_kernel(
    const __half *Q, const __half *K, const __half *V, __half *O, int B, int H,
    int H_kv, int Tq, int Tk, int D, float scale, int q_stride_b,
    int q_stride_h, int q_stride_t, int k_stride_b, int k_stride_h,
    int k_stride_t, int v_stride_b, int v_stride_h, int v_stride_t,
    int o_stride_b, int o_stride_h, int o_stride_t) {
  int b = blockIdx.x;
  int h = blockIdx.y;
  int h_kv = h / (H / H_kv);
  int t_tile = blockIdx.z;

  const int block_size = 128;
  int q_start = t_tile * block_size;
  if (q_start >= Tq)
    return;
  int q_end = min(q_start + block_size, Tq);
  int q_len = q_end - q_start;

  int tid = threadIdx.x;

  extern __shared__ __half s_mem_half[];
  __half *s_Q = s_mem_half;
  __half *s_K = s_Q + block_size * D;
  __half *s_V = s_K + block_size * D;

  float r_O[128] = {0.0f};
  float r_m = -1e37f;
  float r_l = 0.0f;

  for (int i = tid; i < q_len * D; i += blockDim.x) {
    int t = i / D;
    int d = i % D;
    s_Q[t * D + d] =
        Q[b * q_stride_b + h * q_stride_h + (q_start + t) * q_stride_t + d];
  }
  __syncthreads();

  for (int k_tile = 0; k_tile < (Tk + block_size - 1) / block_size; ++k_tile) {
    int k_start = k_tile * block_size;
    int k_end = min(k_start + block_size, Tk);
    int k_len = k_end - k_start;

    for (int i = tid; i < k_len * D; i += blockDim.x) {
      int t = i / D;
      int d = i % D;
      s_K[t * D + d] = K[b * k_stride_b + h_kv * k_stride_h +
                         (k_start + t) * k_stride_t + d];
      s_V[t * D + d] = V[b * v_stride_b + h_kv * v_stride_h +
                         (k_start + t) * v_stride_t + d];
    }
    __syncthreads();

    if (tid < q_len) {
      int q_idx = tid;
      float m_prev = r_m;
      float m_new = r_m;

      for (int k_idx = 0; k_idx < k_len; ++k_idx) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
          score += __half2float(s_Q[q_idx * D + d]) *
                   __half2float(s_K[k_idx * D + d]);
        }
        score *= scale;
        if (score > m_new)
          m_new = score;
      }

      float exp_prev = expf(m_prev - m_new);
      float l_new = r_l * exp_prev;

      float temp_O[128] = {0.0f};
      for (int k_idx = 0; k_idx < k_len; ++k_idx) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
          score += __half2float(s_Q[q_idx * D + d]) *
                   __half2float(s_K[k_idx * D + d]);
        }
        score *= scale;
        float p = expf(score - m_new);
        l_new += p;

        for (int d = 0; d < D && d < 128; ++d) {
          temp_O[d] += p * __half2float(s_V[k_idx * D + d]);
        }
      }

      for (int d = 0; d < D && d < 128; ++d) {
        r_O[d] = r_O[d] * exp_prev + temp_O[d];
      }
      r_m = m_new;
      r_l = l_new;
    }
    __syncthreads();
  }

  if (tid < q_len) {
    int q_idx = tid;
    float inv_l = 1.0f / (r_l + 1e-6f);
    for (int d = 0; d < D; ++d) {
      O[b * o_stride_b + h * o_stride_h + (q_start + q_idx) * o_stride_t + d] =
          __float2half(r_O[d] * inv_l);
    }
  }
}

__global__ void flash_attn_bwd_kernel(float *dQ, float *dK, float *dV,
                                      const float *O, const float *dO,
                                      const float *Q, const float *K,
                                      const float *V, int B, int H, int H_kv,
                                      int Tq, int Tk, int D, float scale) {
  int b = blockIdx.x;
  int h = blockIdx.y;
  int h_kv = h / (H / H_kv);

  for (int q_idx = threadIdx.x; q_idx < Tq; q_idx += blockDim.x) {
    float D_i = 0.0f;
    for (int d = 0; d < D; ++d) {
      D_i += dO[b * H * Tq * D + h * Tq * D + q_idx * D + d] *
             O[b * H * Tq * D + h * Tq * D + q_idx * D + d];
    }

    float max_s = -1e37f;
    for (int k_idx = 0; k_idx < Tk; ++k_idx) {
      float score = 0.0f;
      for (int d = 0; d < D; ++d) {
        score += Q[b * H * Tq * D + h * Tq * D + q_idx * D + d] *
                 K[b * H_kv * Tk * D + h_kv * Tk * D + k_idx * D + d];
      }
      score *= scale;
      if (score > max_s) max_s = score;
    }

    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < Tk; ++k_idx) {
      float score = 0.0f;
      for (int d = 0; d < D; ++d) {
        score += Q[b * H * Tq * D + h * Tq * D + q_idx * D + d] *
                 K[b * H_kv * Tk * D + h_kv * Tk * D + k_idx * D + d];
      }
      score *= scale;
      sum_exp += expf(score - max_s);
    }
    float inv_sum = 1.0f / (sum_exp + 1e-9f);

    for (int k_idx = 0; k_idx < Tk; ++k_idx) {
      float score = 0.0f;
      for (int d = 0; d < D; ++d) {
        score += Q[b * H * Tq * D + h * Tq * D + q_idx * D + d] *
                 K[b * H_kv * Tk * D + h_kv * Tk * D + k_idx * D + d];
      }
      score *= scale;
      float p = expf(score - max_s) * inv_sum;

      float dP = 0.0f;
      for (int d = 0; d < D; ++d) {
        dP += dO[b * H * Tq * D + h * Tq * D + q_idx * D + d] *
              V[b * H_kv * Tk * D + h_kv * Tk * D + k_idx * D + d];
      }

      float dS = scale * p * (dP - D_i);

      for (int d = 0; d < D; ++d) {
        atomicAdd(&dQ[b * H * Tq * D + h * Tq * D + q_idx * D + d],
                  dS * K[b * H_kv * Tk * D + h_kv * Tk * D + k_idx * D + d]);
        atomicAdd(&dK[b * H_kv * Tk * D + h_kv * Tk * D + k_idx * D + d],
                  dS * Q[b * H * Tq * D + h * Tq * D + q_idx * D + d]);
        atomicAdd(&dV[b * H_kv * Tk * D + h_kv * Tk * D + k_idx * D + d],
                  p * dO[b * H * Tq * D + h * Tq * D + q_idx * D + d]);
      }
    }
  }
}

__global__ void flash_attn_bwd_half_kernel(__half *dQ, __half *dK, __half *dV,
                                           const __half *O, const __half *dO,
                                           const __half *Q, const __half *K,
                                           const __half *V, int B, int H,
                                           int H_kv, int Tq, int Tk, int D,
                                           float scale) {
  int b = blockIdx.x;
  int h = blockIdx.y;
  int h_kv = h / (H / H_kv);

  for (int q_idx = threadIdx.x; q_idx < Tq; q_idx += blockDim.x) {
    float D_i = 0.0f;
    for (int d = 0; d < D; ++d) {
      D_i += __half2float(dO[b * H * Tq * D + h * Tq * D + q_idx * D + d]) *
             __half2float(O[b * H * Tq * D + h * Tq * D + q_idx * D + d]);
    }

    float max_s = -1e37f;
    for (int k_idx = 0; k_idx < Tk; ++k_idx) {
      float score = 0.0f;
      for (int d = 0; d < D; ++d) {
        score +=
            __half2float(Q[b * H * Tq * D + h * Tq * D + q_idx * D + d]) *
            __half2float(K[b * H_kv * Tk * D + h_kv * Tk * D + k_idx * D + d]);
      }
      score *= scale;
      if (score > max_s) max_s = score;
    }

    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < Tk; ++k_idx) {
      float score = 0.0f;
      for (int d = 0; d < D; ++d) {
        score +=
            __half2float(Q[b * H * Tq * D + h * Tq * D + q_idx * D + d]) *
            __half2float(K[b * H_kv * Tk * D + h_kv * Tk * D + k_idx * D + d]);
      }
      score *= scale;
      sum_exp += expf(score - max_s);
    }
    float inv_sum = 1.0f / (sum_exp + 1e-9f);

    for (int k_idx = 0; k_idx < Tk; ++k_idx) {
      float score = 0.0f;
      for (int d = 0; d < D; ++d) {
        score +=
            __half2float(Q[b * H * Tq * D + h * Tq * D + q_idx * D + d]) *
            __half2float(K[b * H_kv * Tk * D + h_kv * Tk * D + k_idx * D + d]);
      }
      score *= scale;
      float p = expf(score - max_s) * inv_sum;

      float dP = 0.0f;
      for (int d = 0; d < D; ++d) {
        dP +=
            __half2float(dO[b * H * Tq * D + h * Tq * D + q_idx * D + d]) *
            __half2float(V[b * H_kv * Tk * D + h_kv * Tk * D + k_idx * D + d]);
      }

      float dS = scale * p * (dP - D_i);

      for (int d = 0; d < D; ++d) {
        float dq_val =
            dS * __half2float(K[b * H_kv * Tk * D + h_kv * Tk * D + k_idx * D + d]);
        float dk_val =
            dS * __half2float(Q[b * H * Tq * D + h * Tq * D + q_idx * D + d]);
        float dv_val =
            p * __half2float(dO[b * H * Tq * D + h * Tq * D + q_idx * D + d]);

        atomicAdd(&dQ[b * H * Tq * D + h * Tq * D + q_idx * D + d],
                  __float2half(dq_val));
        atomicAdd(&dK[b * H_kv * Tk * D + h_kv * Tk * D + k_idx * D + d],
                  __float2half(dk_val));
        atomicAdd(&dV[b * H_kv * Tk * D + h_kv * Tk * D + k_idx * D + d],
                  __float2half(dv_val));
      }
    }
  }
}

extern "C" void gpu_flash_attention(void *Q, int q_off, void *K, int k_off,
                                    void *V, int v_off, void *O, int o_off,
                                    int B, int H, int H_kv, int Tq, int Tk,
                                    int D, float scale) {
  int block_size = 128;
  int shared_mem_size = (block_size * D * 3) * sizeof(float);

  dim3 grid(B, H, (Tq + block_size - 1) / block_size);
  dim3 block(128, 1, 1);

#ifndef __HIP_PLATFORM_AMD__
  flash_attn_fwd_kernel<<<grid, block, shared_mem_size, g_compute_stream>>>(
      (const float *)Q + q_off, (const float *)K + k_off,
      (const float *)V + v_off, (float *)O + o_off, B, H, H_kv, Tq, Tk, D,
      scale, H * Tq * D, Tq * D, D, H_kv * Tk * D, Tk * D, D, H_kv * Tk * D,
      Tk * D, D, H * Tq * D, Tq * D, D);
#else
  hipLaunchKernelGGL(flash_attn_fwd_kernel, grid, block, shared_mem_size, g_compute_stream,
      (const float *)Q + q_off, (const float *)K + k_off,
      (const float *)V + v_off, (float *)O + o_off, B, H, H_kv, Tq, Tk, D,
      scale, H * Tq * D, Tq * D, D, H_kv * Tk * D, Tk * D, D, H_kv * Tk * D,
      Tk * D, D, H * Tq * D, Tq * D, D);
#endif
}

extern "C" void gpu_flash_attention_half(void *Q, int q_off, void *K, int k_off,
                                         void *V, int v_off, void *O, int o_off,
                                         int B, int H, int H_kv, int Tq, int Tk,
                                         int D, float scale) {
  int block_size = 128;
  int shared_mem_size = (block_size * D * 3) * sizeof(__half);

  dim3 grid(B, H, (Tq + block_size - 1) / block_size);
  dim3 block(128, 1, 1);

#ifndef __HIP_PLATFORM_AMD__
  flash_attn_fwd_half_kernel<<<grid, block, shared_mem_size,
                               g_compute_stream>>>(
      (const __half *)Q + q_off, (const __half *)K + k_off,
      (const __half *)V + v_off, (__half *)O + o_off, B, H, H_kv, Tq, Tk, D,
      scale, H * Tq * D, Tq * D, D, H_kv * Tk * D, Tk * D, D, H_kv * Tk * D,
      Tk * D, D, H * Tq * D, Tq * D, D);
#else
  hipLaunchKernelGGL(flash_attn_fwd_half_kernel, grid, block, shared_mem_size,
                     g_compute_stream,
      (const __half *)Q + q_off, (const __half *)K + k_off,
      (const __half *)V + v_off, (__half *)O + o_off, B, H, H_kv, Tq, Tk, D,
      scale, H * Tq * D, Tq * D, D, H_kv * Tk * D, Tk * D, D, H_kv * Tk * D,
      Tk * D, D, H * Tq * D, Tq * D, D);
#endif
}

extern "C" void
gpu_flash_attention_backward(void *dQ, int dq_off, void *dK, int dk_off,
                             void *dV, int dv_off, void *O, int o_off, void *dO,
                             int do_off, void *Q, int q_off, void *K, int k_off,
                             void *V, int v_off, int B, int H, int H_kv, int Tq,
                             int Tk, int D, float scale) {
  size_t dq_size = B * H * Tq * D * sizeof(float);
  size_t dk_size = B * H_kv * Tk * D * sizeof(float);
  size_t dv_size = B * H_kv * Tk * D * sizeof(float);

  GPU_API(MemsetAsync)((char *)dQ + dq_off * sizeof(float), 0, dq_size,
                  g_compute_stream);
  GPU_API(MemsetAsync)((char *)dK + dk_off * sizeof(float), 0, dk_size,
                  g_compute_stream);
  GPU_API(MemsetAsync)((char *)dV + dv_off * sizeof(float), 0, dv_size,
                  g_compute_stream);

  dim3 grid(B, H);
  dim3 block(256, 1, 1);

#ifndef __HIP_PLATFORM_AMD__
  flash_attn_bwd_kernel<<<grid, block, 0, g_compute_stream>>>(
      (float *)dQ + dq_off, (float *)dK + dk_off, (float *)dV + dv_off,
      (const float *)O + o_off, (const float *)dO + do_off,
      (const float *)Q + q_off, (const float *)K + k_off,
      (const float *)V + v_off, B, H, H_kv, Tq, Tk, D, scale);
#else
  hipLaunchKernelGGL(flash_attn_bwd_kernel, grid, block, 0, g_compute_stream,
      (float *)dQ + dq_off, (float *)dK + dk_off, (float *)dV + dv_off,
      (const float *)O + o_off, (const float *)dO + do_off,
      (const float *)Q + q_off, (const float *)K + k_off,
      (const float *)V + v_off, B, H, H_kv, Tq, Tk, D, scale);
#endif
}

extern "C" void gpu_flash_attention_backward_half(
    void *dQ, int dq_off, void *dK, int dk_off, void *dV, int dv_off, void *O,
    int o_off, void *dO, int do_off, void *Q, int q_off, void *K, int k_off,
    void *V, int v_off, int B, int H, int H_kv, int Tq, int Tk, int D,
    float scale) {
  size_t dq_size = B * H * Tq * D * sizeof(__half);
  size_t dk_size = B * H_kv * Tk * D * sizeof(__half);
  size_t dv_size = B * H_kv * Tk * D * sizeof(__half);

  GPU_API(MemsetAsync)((char *)dQ + dq_off * sizeof(__half), 0, dq_size,
                  g_compute_stream);
  GPU_API(MemsetAsync)((char *)dK + dk_off * sizeof(__half), 0, dk_size,
                  g_compute_stream);
  GPU_API(MemsetAsync)((char *)dV + dv_off * sizeof(__half), 0, dv_size,
                  g_compute_stream);

  dim3 grid(B, H);
  dim3 block(256, 1, 1);

#ifndef __HIP_PLATFORM_AMD__
  flash_attn_bwd_half_kernel<<<grid, block, 0, g_compute_stream>>>(
      (__half *)dQ + dq_off, (__half *)dK + dk_off, (__half *)dV + dv_off,
      (const __half *)O + o_off, (const __half *)dO + do_off,
      (const __half *)Q + q_off, (const __half *)K + k_off,
      (const __half *)V + v_off, B, H, H_kv, Tq, Tk, D, scale);
#else
  hipLaunchKernelGGL(flash_attn_bwd_half_kernel, grid, block, 0, g_compute_stream,
      (__half *)dQ + dq_off, (__half *)dK + dk_off, (__half *)dV + dv_off,
      (const __half *)O + o_off, (const __half *)dO + do_off,
      (const __half *)Q + q_off, (const __half *)K + k_off,
      (const __half *)V + v_off, B, H, H_kv, Tq, Tk, D, scale);
#endif
}
