#include "gpu_common.h"

extern "C" void gpu_matmul(void* A, int a_off, void* B, int b_off, void* C, int c_off, int M, int N, int K) {
#ifndef __HIP_PLATFORM_AMD__
    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f;
    float beta = 0.0f;
    const float* a_ptr = (const float*)A + a_off;
    const float* b_ptr = (const float*)B + b_off;
    float* c_ptr = (float*)C + c_off;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b_ptr, N, a_ptr, K, &beta, c_ptr, N);
#else
    rocblas_handle handle = get_rocblas_handle();
    float alpha = 1.0f;
    float beta = 0.0f;
    const float* a_ptr = (const float*)A + a_off;
    const float* b_ptr = (const float*)B + b_off;
    float* c_ptr = (float*)C + c_off;
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, N, M, K, &alpha, b_ptr, N, a_ptr, K, &beta, c_ptr, N);
#endif
}

extern "C" void gpu_bmm(void* A, int a_off, void* B, int b_off, void* C, int c_off, int batch_size, int M, int N, int K) {
#ifndef __HIP_PLATFORM_AMD__
    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f;
    float beta = 0.0f;
    const float* a_ptr = (const float*)A + a_off;
    const float* b_ptr = (const float*)B + b_off;
    float* c_ptr = (float*)C + c_off;
    long long strideA = M * K;
    long long strideB = K * N;
    long long strideC = M * N;
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b_ptr, N, strideB, a_ptr, K, strideA, &beta, c_ptr, N, strideC, batch_size);
#else
    rocblas_handle handle = get_rocblas_handle();
    float alpha = 1.0f;
    float beta = 0.0f;
    const float* a_ptr = (const float*)A + a_off;
    const float* b_ptr = (const float*)B + b_off;
    float* c_ptr = (float*)C + c_off;
    long long strideA = M * K;
    long long strideB = K * N;
    long long strideC = M * N;
    rocblas_sgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none, N, M, K, &alpha, b_ptr, N, strideB, a_ptr, K, strideA, &beta, c_ptr, N, strideC, batch_size);
#endif
}

extern "C" void gpu_matmul_half(void* A, int a_off, void* B, int b_off, void* C, int c_off, int M, int N, int K) {
#ifndef __HIP_PLATFORM_AMD__
    cublasHandle_t handle = get_cublas_handle();
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    const __half* a_ptr = (const __half*)A + a_off;
    const __half* b_ptr = (const __half*)B + b_off;
    __half* c_ptr = (__half*)C + c_off;
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b_ptr, N, a_ptr, K, &beta, c_ptr, N);
#else
    rocblas_handle handle = get_rocblas_handle();
    const __half alpha_h = __float2half(1.0f);
    const __half beta_h = __float2half(0.0f);
    const rocblas_half* alpha = reinterpret_cast<const rocblas_half*>(&alpha_h);
    const rocblas_half* beta = reinterpret_cast<const rocblas_half*>(&beta_h);
    const rocblas_half* a_ptr = (const rocblas_half*)A + a_off;
    const rocblas_half* b_ptr = (const rocblas_half*)B + b_off;
    rocblas_half* c_ptr = (rocblas_half*)C + c_off;
    rocblas_hgemm(handle, rocblas_operation_none, rocblas_operation_none, N, M, K, alpha, b_ptr, N, a_ptr, K, beta, c_ptr, N);
#endif
}

extern "C" void gpu_bmm_half(void* A, int a_off, void* B, int b_off, void* C, int c_off, int batch_size, int M, int N, int K) {
#ifndef __HIP_PLATFORM_AMD__
    cublasHandle_t handle = get_cublas_handle();
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    const __half* a_ptr = (const __half*)A + a_off;
    const __half* b_ptr = (const __half*)B + b_off;
    __half* c_ptr = (__half*)C + c_off;
    long long strideA = M * K;
    long long strideB = K * N;
    long long strideC = M * N;
    cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b_ptr, N, strideB, a_ptr, K, strideA, &beta, c_ptr, N, strideC, batch_size);
#else
    rocblas_handle handle = get_rocblas_handle();
    const __half alpha_h = __float2half(1.0f);
    const __half beta_h = __float2half(0.0f);
    const rocblas_half* alpha = reinterpret_cast<const rocblas_half*>(&alpha_h);
    const rocblas_half* beta = reinterpret_cast<const rocblas_half*>(&beta_h);
    const rocblas_half* a_ptr = (const rocblas_half*)A + a_off;
    const rocblas_half* b_ptr = (const rocblas_half*)B + b_off;
    rocblas_half* c_ptr = (rocblas_half*)C + c_off;
    long long strideA = M * K;
    long long strideB = K * N;
    long long strideC = M * N;
    rocblas_hgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none, N, M, K, alpha, b_ptr, N, strideB, a_ptr, K, strideA, beta, c_ptr, N, strideC, batch_size);
#endif
}
