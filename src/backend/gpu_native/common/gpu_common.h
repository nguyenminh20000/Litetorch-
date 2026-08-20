#ifndef LITETORCH_GPU_COMMON_H
#define LITETORCH_GPU_COMMON_H

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <rocprim/rocprim.hpp>
#ifdef USE_MIOPEN
#include <miopen/miopen.h>
#endif
#define GPU_API(name) hip##name
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#define GPU_API(name) cuda##name
#endif

#include "litetorch/platform.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>

extern GPU_API(Stream_t) g_compute_stream;

#ifndef __HIP_PLATFORM_AMD__
cublasHandle_t get_cublas_handle();
#ifdef USE_CUDNN
cudnnHandle_t get_cudnn_handle();
#endif
#else
rocblas_handle get_rocblas_handle();
#ifdef USE_MIOPEN
miopenHandle_t get_miopen_handle();
#endif
#endif

__device__ inline float clamp(float val, float min_val, float max_val) {
    return fminf(fmaxf(val, min_val), max_val);
}

__device__ inline float mad(float a, float b, float c) {
    return fmaf(a, b, c);
}

__device__ inline float sign(float x) {
    return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

__device__ inline float atomic_add_float(float* addr, float val) {
    return atomicAdd(addr, val);
}

__device__ inline unsigned short dev_float_to_half(float f) {
    __half h = __float2half(f);
    return *reinterpret_cast<unsigned short*>(&h);
}

__device__ inline float dev_half_to_float(unsigned short h) {
    __half val = *reinterpret_cast<const __half*>(&h);
    return __half2float(val);
}

__device__ inline unsigned short dev_float_to_bf16(float f) {
    unsigned int val = *reinterpret_cast<unsigned int*>(&f);
    return static_cast<unsigned short>(val >> 16);
}

__device__ inline float dev_bf16_to_float(unsigned short h) {
    unsigned int val = (static_cast<unsigned int>(h)) << 16;
    return *reinterpret_cast<float*>(&val);
}

__constant__ const float DEV_NF4_GRID[16] = {
    -1.0f, -0.6961917f, -0.525073f, -0.3930782f, -0.2753147f, -0.1651313f, -0.0596338f, 0.0f,
    0.0596338f, 0.1651313f, 0.2753147f, 0.3930782f, 0.525073f, 0.6961917f, 1.0f, 1.0f
};

__device__ inline unsigned char dev_float_to_nf4(float val) {
    float min_dist = 1e9f;
    unsigned char best_idx = 0;
    for (unsigned char i = 0; i < 16; ++i) {
        float dist = fabsf(DEV_NF4_GRID[i] - val);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

__device__ inline float dev_nf4_to_float(unsigned char idx) {
    unsigned char i = idx >= 16 ? 15 : idx;
    return DEV_NF4_GRID[i];
}

__device__ inline unsigned char dev_float_to_fp8_e4m3(float val) {
    if (isnan(val)) return 0x7F;
    unsigned int ui = *reinterpret_cast<unsigned int*>(&val);
    unsigned int sign_bit = (ui >> 31) & 1;
    unsigned int exp_bits = (ui >> 23) & 0xFF;
    unsigned int mant_bits = ui & 0x7FFFFF;
    if (exp_bits == 0) return static_cast<unsigned char>(sign_bit << 7);
    if (exp_bits == 0xFF) return static_cast<unsigned char>((sign_bit << 7) | 0x7F);
    int new_exp = static_cast<int>(exp_bits) - 127 + 7;
    if (new_exp <= 0) {
        int shift = 1 - new_exp;
        if (shift > 3) return static_cast<unsigned char>(sign_bit << 7);
        unsigned int m = (1 << 3) | (mant_bits >> 20);
        m >>= shift;
        return static_cast<unsigned char>((sign_bit << 7) | m);
    } else if (new_exp >= 15) {
        return static_cast<unsigned char>((sign_bit << 7) | 0x7E);
    }
    unsigned int m = mant_bits >> 20;
    return static_cast<unsigned char>((sign_bit << 7) | (new_exp << 3) | m);
}

__device__ inline float dev_fp8_e4m3_to_float(unsigned char val) {
    unsigned int sign_bit = (val >> 7) & 1;
    unsigned int exp_bits = (val >> 3) & 0x0F;
    unsigned int mant_bits = val & 0x07;
    if (exp_bits == 15) return NAN;
    if (exp_bits == 0) {
        if (mant_bits == 0) return sign_bit ? -0.0f : 0.0f;
        return (sign_bit ? -1.0f : 1.0f) * powf(2.0f, -6.0f) * (static_cast<float>(mant_bits) / 8.0f);
    }
    return (sign_bit ? -1.0f : 1.0f) * powf(2.0f, static_cast<float>(exp_bits) - 7.0f) * (1.0f + static_cast<float>(mant_bits) / 8.0f);
}

__device__ inline unsigned char dev_float_to_fp8_e5m2(float val) {
    if (isnan(val)) return 0x7F;
    unsigned int ui = *reinterpret_cast<unsigned int*>(&val);
    unsigned int sign_bit = (ui >> 31) & 1;
    unsigned int exp_bits = (ui >> 23) & 0xFF;
    unsigned int mant_bits = ui & 0x7FFFFF;
    if (exp_bits == 0) return static_cast<unsigned char>(sign_bit << 7);
    if (exp_bits == 0xFF) return static_cast<unsigned char>((sign_bit << 7) | 0x7F);
    int new_exp = static_cast<int>(exp_bits) - 127 + 15;
    if (new_exp <= 0) {
        int shift = 1 - new_exp;
        if (shift > 2) return static_cast<unsigned char>(sign_bit << 7);
        unsigned int m = (1 << 2) | (mant_bits >> 21);
        m >>= shift;
        return static_cast<unsigned char>((sign_bit << 7) | m);
    } else if (new_exp >= 31) {
        return static_cast<unsigned char>((sign_bit << 7) | 0x7E);
    }
    unsigned int m = mant_bits >> 21;
    return static_cast<unsigned char>((sign_bit << 7) | (new_exp << 2) | m);
}

__device__ inline float dev_fp8_e5m2_to_float(unsigned char val) {
    unsigned int sign_bit = (val >> 7) & 1;
    unsigned int exp_bits = (val >> 2) & 0x1F;
    unsigned int mant_bits = val & 0x03;
    if (exp_bits == 31) return NAN;
    if (exp_bits == 0) {
        if (mant_bits == 0) return sign_bit ? -0.0f : 0.0f;
        return (sign_bit ? -1.0f : 1.0f) * powf(2.0f, -14.0f) * (static_cast<float>(mant_bits) / 4.0f);
    }
    return (sign_bit ? -1.0f : 1.0f) * powf(2.0f, static_cast<float>(exp_bits) - 15.0f) * (1.0f + static_cast<float>(mant_bits) / 4.0f);
}

#endif
