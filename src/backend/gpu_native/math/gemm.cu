#include "gpu_common.h"
#include <mutex>
#include <unordered_map>

extern "C" void* gpu_allocate(size_t size);
extern "C" bool gpu_is_tf32_enabled();

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
    void* c_ptr = (char*)C + c_off * sizeof(__half);
    
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
    rocblas_handle handle = get_rocblas_handle();
    float alpha = 1.0f;
    float beta = 0.0f;
    const void* a_ptr = (const char*)A + a_off * sizeof(unsigned short);
    const void* b_ptr = (const char*)B + b_off * sizeof(unsigned short);
    void* c_ptr = (char*)C + c_off * sizeof(unsigned short);
    rocblas_status status = rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none,
                    N, M, K, &alpha,
                    b_ptr, rocblas_datatype_bf16_r, N,
                    a_ptr, rocblas_datatype_bf16_r, K,
                    &beta,
                    c_ptr, rocblas_datatype_bf16_r, N,
                    c_ptr, rocblas_datatype_bf16_r, N,
                    rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    if (status != rocblas_status_success) {
        gpu_matmul_half(A, a_off, B, b_off, C, c_off, M, N, K);
    }
#endif
}

#ifndef __HIP_PLATFORM_AMD__

struct LtMatmulKey {
    int64_t M, N, K, lda, ldb;
    bool trans_a, trans_b, tf32, has_bias;
    bool operator==(const LtMatmulKey& o) const {
        return M == o.M && N == o.N && K == o.K && lda == o.lda && ldb == o.ldb &&
               trans_a == o.trans_a && trans_b == o.trans_b && tf32 == o.tf32 && has_bias == o.has_bias;
    }
};

struct LtMatmulKeyHash {
    size_t operator()(const LtMatmulKey& k) const {
        size_t h = 1469598103934665603ULL;
        auto mix = [&](int64_t v) { h ^= (size_t)v; h *= 1099511628211ULL; };
        mix(k.M); mix(k.N); mix(k.K); mix(k.lda); mix(k.ldb);
        mix(k.trans_a ? 1 : 0); mix(k.trans_b ? 1 : 0); mix(k.tf32 ? 1 : 0); mix(k.has_bias ? 1 : 0);
        return h;
    }
};

static std::unordered_map<LtMatmulKey, cublasLtMatmulAlgo_t, LtMatmulKeyHash> lt_algo_cache;
static std::mutex lt_algo_cache_mutex;

static const size_t LT_WS_BYTES = 32ULL * 1024ULL * 1024ULL;

static void* lt_workspace() {
    thread_local void* ws = nullptr;
    if (!ws) ws = gpu_allocate(LT_WS_BYTES);
    return ws;
}

static bool matmul_lt_find_algo(cublasLtHandle_t lt_handle,
                                cublasLtMatmulDesc_t desc,
                                cublasLtMatrixLayout_t Adesc,
                                cublasLtMatrixLayout_t Bdesc,
                                cublasLtMatrixLayout_t Cdesc,
                                const float* a_ptr, const float* b_ptr, float* c_ptr,
                                const LtMatmulKey& key,
                                cublasLtMatmulAlgo_t* out_algo) {
    {
        std::lock_guard<std::mutex> lock(lt_algo_cache_mutex);
        auto it = lt_algo_cache.find(key);
        if (it != lt_algo_cache.end()) {
            *out_algo = it->second;
            return true;
        }
    }
    void* ws = lt_workspace();
    if (!ws) return false;
    cublasLtMatmulPreference_t pref = nullptr;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return false;
    size_t ws_cap = LT_WS_BYTES;
    bool pref_ok = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_cap, sizeof(ws_cap)) == CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulHeuristicResult_t heuristics[8];
    int retCount = 0;
    bool heur_ok = false;
    if (pref_ok) {
        heur_ok = cublasLtMatmulAlgoGetHeuristic(lt_handle, desc, Adesc, Bdesc, Cdesc, Cdesc, pref, 8, heuristics, &retCount) == CUBLAS_STATUS_SUCCESS && retCount > 0;
    }
    cublasLtMatmulPreferenceDestroy(pref);
    if (!heur_ok) return false;
    float alpha = 1.0f, beta = 0.0f;
    cudaStreamCaptureStatus capStatus = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(g_compute_stream, &capStatus);
    cudaEvent_t start = nullptr, stop = nullptr;
    bool timed = capStatus == cudaStreamCaptureStatusNone &&
                 cudaEventCreate(&start) == cudaSuccess &&
                 cudaEventCreate(&stop) == cudaSuccess;
    int best = -1;
    float best_ms = 0.0f;
    for (int i = 0; i < retCount; ++i) {
        if (heuristics[i].state != CUBLAS_STATUS_SUCCESS) continue;
        if (heuristics[i].workspaceSize > LT_WS_BYTES) continue;
        if (!timed) {
            if (best < 0) best = i;
            continue;
        }
        if (cublasLtMatmul(lt_handle, desc, &alpha, a_ptr, Adesc, b_ptr, Bdesc, &beta,
                           c_ptr, Cdesc, c_ptr, Cdesc, &heuristics[i].algo,
                           ws, LT_WS_BYTES, g_compute_stream) != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        float total = 0.0f;
        bool ok = true;
        for (int r = 0; r < 5; ++r) {
            cudaEventRecord(start, g_compute_stream);
            if (cublasLtMatmul(lt_handle, desc, &alpha, a_ptr, Adesc, b_ptr, Bdesc, &beta,
                               c_ptr, Cdesc, c_ptr, Cdesc, &heuristics[i].algo,
                               ws, LT_WS_BYTES, g_compute_stream) != CUBLAS_STATUS_SUCCESS) {
                ok = false;
                break;
            }
            cudaEventRecord(stop, g_compute_stream);
            if (cudaEventSynchronize(stop) != cudaSuccess) {
                ok = false;
                break;
            }
            float e = 0.0f;
            cudaEventElapsedTime(&e, start, stop);
            total += e;
        }
        if (!ok) continue;
        if (best < 0 || total < best_ms) {
            best = i;
            best_ms = total;
        }
    }
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    if (best < 0) return false;
    {
        std::lock_guard<std::mutex> lock(lt_algo_cache_mutex);
        lt_algo_cache.emplace(key, heuristics[best].algo);
    }
    *out_algo = heuristics[best].algo;
    return true;
}

#endif

extern "C" void gpu_matmul_ex_lt(void* A, int64_t a_off, bool trans_a, int64_t lda,
                                 void* B, int64_t b_off, bool trans_b, int64_t ldb,
                                 void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
#ifndef __HIP_PLATFORM_AMD__
    if (M <= 0 || N <= 0 || K <= 0) {
        gpu_matmul_ex(A, a_off, trans_a, lda, B, b_off, trans_b, ldb, C, c_off, M, N, K);
        return;
    }
    const float* a_ptr = (const float*)A + a_off;
    const float* b_ptr = (const float*)B + b_off;
    float* c_ptr = (float*)C + c_off;
    bool tf32 = gpu_is_tf32_enabled();
    LtMatmulKey key{M, N, K, lda, ldb, trans_a, trans_b, tf32, false};
    cublasLtHandle_t lt_handle = get_cublaslt_handle();
    cublasComputeType_t compute = tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    bool built = false;
    if (cublasLtMatmulDescCreate(&desc, compute, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS) {
        cublasOperation_t transA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t transB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        uint64_t a_rows = (uint64_t)(trans_b ? K : N);
        uint64_t a_cols = (uint64_t)(trans_b ? N : K);
        uint64_t b_rows = (uint64_t)(trans_a ? M : K);
        uint64_t b_cols = (uint64_t)(trans_a ? K : M);
        if (cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, a_rows, a_cols, (int64_t)ldb) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, b_rows, b_cols, (int64_t)lda) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, (uint64_t)N, (uint64_t)M, (int64_t)N) == CUBLAS_STATUS_SUCCESS) {
            built = true;
        }
    }
    cublasLtMatmulAlgo_t algo{};
    bool have_algo = built && matmul_lt_find_algo(lt_handle, desc, Adesc, Bdesc, Cdesc, b_ptr, a_ptr, c_ptr, key, &algo);
    if (have_algo) {
        void* ws = lt_workspace();
        float alpha = 1.0f, beta = 0.0f;
        if (ws && cublasLtMatmul(lt_handle, desc, &alpha, b_ptr, Adesc, a_ptr, Bdesc, &beta,
                                c_ptr, Cdesc, c_ptr, Cdesc, &algo,
                                ws, LT_WS_BYTES, g_compute_stream) == CUBLAS_STATUS_SUCCESS) {
            cublasLtMatrixLayoutDestroy(Cdesc);
            cublasLtMatrixLayoutDestroy(Bdesc);
            cublasLtMatrixLayoutDestroy(Adesc);
            cublasLtMatmulDescDestroy(desc);
            return;
        }
    }
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (desc) cublasLtMatmulDescDestroy(desc);
    gpu_matmul_ex(A, a_off, trans_a, lda, B, b_off, trans_b, ldb, C, c_off, M, N, K);
#else
    gpu_matmul_ex(A, a_off, trans_a, lda, B, b_off, trans_b, ldb, C, c_off, M, N, K);
#endif
}

extern "C" __global__ void matmul_bias_add_kernel(float* out, int64_t out_off, const float* bias, int64_t bias_off, int64_t M, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    if (idx < total) {
        int64_t n = idx % N;
        out[out_off + idx] += bias[bias_off + n];
    }
}

extern "C" void gpu_matmul_ex_lt_bias(void* A, int64_t a_off, bool trans_a, int64_t lda,
                                      void* B, int64_t b_off, bool trans_b, int64_t ldb,
                                      void* BIAS, int64_t bias_off,
                                      void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
#ifndef __HIP_PLATFORM_AMD__
    if (M <= 0 || N <= 0 || K <= 0) return;
    const float* a_ptr = (const float*)A + a_off;
    const float* b_ptr = (const float*)B + b_off;
    const float* bias_ptr = (const float*)BIAS + bias_off;
    float* c_ptr = (float*)C + c_off;
    bool tf32 = gpu_is_tf32_enabled();
    LtMatmulKey key{M, N, K, lda, ldb, trans_a, trans_b, tf32, true};
    cublasLtHandle_t lt_handle = get_cublaslt_handle();
    cublasComputeType_t compute = tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    bool built = false;
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    if (cublasLtMatmulDescCreate(&desc, compute, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS) {
        cublasOperation_t opA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        uint64_t a_rows = (uint64_t)(trans_a ? K : M);
        uint64_t a_cols = (uint64_t)(trans_a ? M : K);
        uint64_t b_rows = (uint64_t)(trans_b ? N : K);
        uint64_t b_cols = (uint64_t)(trans_b ? K : N);
        if (cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, a_rows, a_cols, (int64_t)lda) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, b_rows, b_cols, (int64_t)ldb) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, (uint64_t)M, (uint64_t)N, (int64_t)N) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)) == CUBLAS_STATUS_SUCCESS) {
            built = true;
        }
    }
    cublasLtMatmulAlgo_t algo{};
    bool have_algo = built && matmul_lt_find_algo(lt_handle, desc, Adesc, Bdesc, Cdesc, a_ptr, b_ptr, c_ptr, key, &algo);
    if (have_algo) {
        void* ws = lt_workspace();
        float alpha = 1.0f, beta = 0.0f;
        if (ws && cublasLtMatmul(lt_handle, desc, &alpha, a_ptr, Adesc, b_ptr, Bdesc, &beta,
                                c_ptr, Cdesc, c_ptr, Cdesc, &algo,
                                ws, LT_WS_BYTES, g_compute_stream) == CUBLAS_STATUS_SUCCESS) {
            cublasLtMatrixLayoutDestroy(Cdesc);
            cublasLtMatrixLayoutDestroy(Bdesc);
            cublasLtMatrixLayoutDestroy(Adesc);
            cublasLtMatmulDescDestroy(desc);
            return;
        }
    }
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (desc) cublasLtMatmulDescDestroy(desc);
    gpu_matmul_ex(A, a_off, trans_a, lda, B, b_off, trans_b, ldb, C, c_off, M, N, K);
    int64_t total = M * N;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    matmul_bias_add_kernel<<<(unsigned)blocks, threads, 0, g_compute_stream>>>(c_ptr, (int64_t)0, bias_ptr, (int64_t)0, M, N);
#else
    gpu_matmul_ex(A, a_off, trans_a, lda, B, b_off, trans_b, ldb, C, c_off, M, N, K);
#endif
}
