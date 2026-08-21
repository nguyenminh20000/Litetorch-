#include "gpu_common.h"

extern "C" void gpu_matmul(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
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

extern "C" void gpu_matmul_ex(void* A, int64_t a_off, bool trans_a, int64_t lda,
                             void* B, int64_t b_off, bool trans_b, int64_t ldb,
                             void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
#ifndef __HIP_PLATFORM_AMD__
    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f;
    float beta = 0.0f;
    const float* a_ptr = (const float*)A + a_off;
    const float* b_ptr = (const float*)B + b_off;
    float* c_ptr = (float*)C + c_off;
    cublasOperation_t opB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasSgemm(handle, opB, opA, N, M, K, &alpha, b_ptr, ldb, a_ptr, lda, &beta, c_ptr, N);
#else
    rocblas_handle handle = get_rocblas_handle();
    float alpha = 1.0f;
    float beta = 0.0f;
    const float* a_ptr = (const float*)A + a_off;
    const float* b_ptr = (const float*)B + b_off;
    float* c_ptr = (float*)C + c_off;
    rocblas_operation opB = trans_b ? rocblas_operation_transpose : rocblas_operation_none;
    rocblas_operation opA = trans_a ? rocblas_operation_transpose : rocblas_operation_none;
    rocblas_sgemm(handle, opB, opA, N, M, K, &alpha, b_ptr, ldb, a_ptr, lda, &beta, c_ptr, N);
#endif
}

extern "C" void gpu_bmm(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t batch_size, int64_t M, int64_t N, int64_t K) {
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

extern "C" void gpu_matmul_half(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
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

extern "C" void gpu_bmm_half(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t batch_size, int64_t M, int64_t N, int64_t K) {
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
    rocblas_hgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none, N, M, K, alpha, b_ptr, N, strideB, a_ptr, K, strideA, beta, c_ptr, N, strideC, batch_size);
#endif
}

extern "C" void gpu_matmul_fp8(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K, float a_scale, float b_scale, float d_scale) {
#ifndef __HIP_PLATFORM_AMD__
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
    cublasLtHandle_t lt_handle = get_cublaslt_handle();
    cublasLtMatmulDesc_t matmulDesc = nullptr;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, N, K, N);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, N, M, N);
    
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(float));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(float));
    
    float alpha = 1.0f / (d_scale > 0.0f ? d_scale : 1.0f);
    float beta = 0.0f;
    const void* a_ptr = (const char*)A + a_off;
    const void* b_ptr = (const char*)B + b_off;
    void* c_ptr = (char*)C + c_off;
    
    cublasLtMatmul(lt_handle, matmulDesc, &alpha, b_ptr, Bdesc, a_ptr, Adesc, &beta, c_ptr, Cdesc, c_ptr, Cdesc, nullptr, nullptr, 0, g_compute_stream);
    
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(matmulDesc);
#else
    gpu_matmul_half(A, a_off, B, b_off, C, c_off, M, N, K);
#endif
#else
    gpu_matmul_half(A, a_off, B, b_off, C, c_off, M, N, K);
#endif
}

extern "C" void gpu_matmul_bf16(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
#ifndef __HIP_PLATFORM_AMD__
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f;
    float beta = 0.0f;
    const void* a_ptr = (const char*)A + a_off * sizeof(unsigned short);
    const void* b_ptr = (const char*)B + b_off * sizeof(unsigned short);
    void* c_ptr = (char*)C + c_off * sizeof(unsigned short);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b_ptr, CUDA_R_16BF, N, a_ptr, CUDA_R_16BF, K, &beta, c_ptr, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
#else
    gpu_matmul_half(A, a_off, B, b_off, C, c_off, M, N, K);
#endif
#else
    gpu_matmul_half(A, a_off, B, b_off, C, c_off, M, N, K);
#endif
}
