#include <cuda_fp16.h>
#include <cstdint>

static __device__ __forceinline__ uint8_t float_to_e4m3(float x) {
    uint32_t bits = __float_as_uint(x);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;
    if ((bits & 0x7FFFFFFF) == 0) return (uint8_t)(sign << 7);
    int32_t e4 = exp + 8;
    if (e4 >= 15) return (uint8_t)((sign << 7) | 0x7E);
    if (e4 <= 0) return (uint8_t)(sign << 7);
    uint32_t m3 = mant >> 20;
    uint32_t round_bit = (mant >> 19) & 1;
    uint32_t sticky = mant & 0x7FFFF;
    if (round_bit && (sticky || (m3 & 1))) {
        m3++;
        if (m3 >= 8) { m3 = 0; e4++; if (e4 >= 15) return (uint8_t)((sign << 7) | 0x7E); }
    }
    return (uint8_t)((sign << 7) | (e4 << 3) | m3);
}

static __device__ __forceinline__ float e4m3_to_float(uint8_t x) {
    uint32_t sign = (x >> 7) & 0x1;
    uint32_t e4 = (x >> 3) & 0xF;
    uint32_t m3 = x & 0x7;
    if (e4 == 0) {
        if (m3 == 0) return sign ? -0.0f : 0.0f;
        float f = (float)m3 / 8.0f * 0.015625f;
        return sign ? -f : f;
    }
    if (e4 == 15 && m3 == 7) return sign ? -CUDART_NAN_F : CUDART_NAN_F;
    int32_t exp = (int32_t)e4 - 8 + 127;
    uint32_t bits = (sign << 31) | ((uint32_t)exp << 23) | (m3 << 20);
    return __uint_as_float(bits);
}

static __device__ __forceinline__ uint8_t float_to_e5m2(float x) {
    uint32_t bits = __float_as_uint(x);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;
    if ((bits & 0x7FFFFFFF) == 0) return (uint8_t)(sign << 7);
    int32_t e5 = exp + 16;
    if (e5 >= 31) return (uint8_t)((sign << 7) | 0x7C);
    if (e5 <= 0) return (uint8_t)(sign << 7);
    uint32_t m2 = mant >> 21;
    uint32_t round_bit = (mant >> 20) & 1;
    uint32_t sticky = mant & 0xFFFFF;
    if (round_bit && (sticky || (m2 & 1))) {
        m2++;
        if (m2 >= 4) { m2 = 0; e5++; if (e5 >= 31) return (uint8_t)((sign << 7) | 0x7C); }
    }
    return (uint8_t)((sign << 7) | (e5 << 2) | m2);
}

static __device__ __forceinline__ float e5m2_to_float(uint8_t x) {
    uint32_t sign = (x >> 7) & 0x1;
    uint32_t e5 = (x >> 2) & 0x1F;
    uint32_t m2 = x & 0x3;
    if (e5 == 0) {
        if (m2 == 0) return sign ? -0.0f : 0.0f;
        float f = (float)m2 / 4.0f * 0.00006103515625f;
        return sign ? -f : f;
    }
    if (e5 == 31) return m2 ? CUDART_NAN_F : (sign ? -CUDART_INF_F : CUDART_INF_F);
    int32_t exp = (int32_t)e5 - 16 + 127;
    uint32_t bits = (sign << 31) | ((uint32_t)exp << 23) | (m2 << 21);
    return __uint_as_float(bits);
}

__global__ void fp8_cast_e4m3_kernel(const float* __restrict__ src, uint8_t* __restrict__ dst,
                                      float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = float_to_e4m3(src[i] / scale);
}

__global__ void fp8_cast_e5m2_kernel(const float* __restrict__ src, uint8_t* __restrict__ dst,
                                      float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = float_to_e5m2(src[i] / scale);
}

__global__ void fp8_uncast_e4m3_kernel(const uint8_t* __restrict__ src, float* __restrict__ dst,
                                       float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = e4m3_to_float(src[i]) * scale;
}

__global__ void fp8_uncast_e5m2_kernel(const uint8_t* __restrict__ src, float* __restrict__ dst,
                                       float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = e5m2_to_float(src[i]) * scale;
}

__global__ void fp8_matmul_e4m3_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                        uint8_t* __restrict__ C,
                                        float a_scale, float b_scale, float out_scale,
                                        int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += e4m3_to_float(A[row * K + k]) * e4m3_to_float(B[k * N + col]);
        C[row * N + col] = float_to_e4m3(sum * a_scale * b_scale / out_scale);
    }
}
