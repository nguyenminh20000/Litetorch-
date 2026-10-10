#include "../common/tpu_common.h"
#include "litetorch/thread_pool.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace litetorch {
namespace tpu_internal {

namespace {

constexpr int64_t TPU_GEMM_BM = 128;
constexpr int64_t TPU_GEMM_BN = 128;
constexpr int64_t TPU_GEMM_BK = 256;

inline void blocked_gemm_rows(const float* __restrict__ A, int64_t lda,
                              const float* __restrict__ B, int64_t ldb,
                              float* __restrict__ C, int64_t ldc,
                              int64_t i0, int64_t i1, int64_t N, int64_t K) {
    for (int64_t k0 = 0; k0 < K; k0 += TPU_GEMM_BK) {
        int64_t k1 = k0 + TPU_GEMM_BK < K ? k0 + TPU_GEMM_BK : K;
        for (int64_t j0 = 0; j0 < N; j0 += TPU_GEMM_BN) {
            int64_t j1 = j0 + TPU_GEMM_BN < N ? j0 + TPU_GEMM_BN : N;
            for (int64_t i = i0; i < i1; ++i) {
                const float* __restrict__ a_row = A + i * lda;
                float* __restrict__ c_row = C + i * ldc;
                for (int64_t k = k0; k < k1; ++k) {
                    float aval = a_row[k];
                    if (aval == 0.0f) continue;
                    const float* __restrict__ b_row = B + k * ldb;
                    int64_t j = j0;
                    for (; j + 4 <= j1; j += 4) {
                        c_row[j]     += aval * b_row[j];
                        c_row[j + 1] += aval * b_row[j + 1];
                        c_row[j + 2] += aval * b_row[j + 2];
                        c_row[j + 3] += aval * b_row[j + 3];
                    }
                    for (; j < j1; ++j) {
                        c_row[j] += aval * b_row[j];
                    }
                }
            }
        }
    }
}

}

void tpu_systolic_matmul(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    std::memset(C, 0, (size_t)(M * N) * sizeof(float));
    int64_t n_bm = (M + TPU_GEMM_BM - 1) / TPU_GEMM_BM;
    ThreadPool::get().parallel_for(0, n_bm, [&](int64_t bm) {
        int64_t i0 = bm * TPU_GEMM_BM;
        int64_t i1 = i0 + TPU_GEMM_BM < M ? i0 + TPU_GEMM_BM : M;
        blocked_gemm_rows(A, K, B, N, C, N, i0, i1, N, K);
    });
}

void tpu_systolic_matmul_ex(const float* A, bool trans_a, int64_t lda,
                            const float* B, bool trans_b, int64_t ldb,
                            float* C, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    std::memset(C, 0, (size_t)(M * N) * sizeof(float));

    if (!trans_a && !trans_b) {
        int64_t n_bm = (M + TPU_GEMM_BM - 1) / TPU_GEMM_BM;
        ThreadPool::get().parallel_for(0, n_bm, [&](int64_t bm) {
            int64_t i0 = bm * TPU_GEMM_BM;
            int64_t i1 = i0 + TPU_GEMM_BM < M ? i0 + TPU_GEMM_BM : M;
            blocked_gemm_rows(A, lda, B, ldb, C, N, i0, i1, N, K);
        });
    } else if (trans_a && !trans_b) {
        ThreadPool::get().parallel_for(0, M, [&](int64_t i) {
            float* C_row = C + i * N;
            for (int64_t k = 0; k < K; ++k) {
                float aval = A[k * lda + i];
                if (aval == 0.0f) continue;
                const float* B_row = B + k * ldb;
                int64_t j = 0;
                for (; j + 4 <= N; j += 4) {
                    C_row[j]     += aval * B_row[j];
                    C_row[j + 1] += aval * B_row[j + 1];
                    C_row[j + 2] += aval * B_row[j + 2];
                    C_row[j + 3] += aval * B_row[j + 3];
                }
                for (; j < N; ++j) {
                    C_row[j] += aval * B_row[j];
                }
            }
        });
    } else if (!trans_a && trans_b) {
        ThreadPool::get().parallel_for(0, M, [&](int64_t i) {
            float* C_row = C + i * N;
            const float* A_row = A + i * lda;
            for (int64_t j = 0; j < N; ++j) {
                const float* B_row = B + j * ldb;
                float dot = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    dot += A_row[k] * B_row[k];
                }
                C_row[j] = dot;
            }
        });
    } else {
        ThreadPool::get().parallel_for(0, M, [&](int64_t i) {
            float* C_row = C + i * N;
            for (int64_t j = 0; j < N; ++j) {
                const float* B_row = B + j * ldb;
                float dot = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    dot += A[k * lda + i] * B_row[k];
                }
                C_row[j] = dot;
            }
        });
    }
}

void tpu_systolic_bmm(const float* A, const float* B, float* C, int64_t B_batch, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    ThreadPool::get().parallel_for(0, B_batch, [&](int64_t b) {
        const float* Ab = A + b * (M * K);
        const float* Bb = B + b * (K * N);
        float* Cb = C + b * (M * N);
        std::memset(Cb, 0, (size_t)(M * N) * sizeof(float));
        int64_t n_bm = (M + TPU_GEMM_BM - 1) / TPU_GEMM_BM;
        for (int64_t bm = 0; bm < n_bm; ++bm) {
            int64_t i0 = bm * TPU_GEMM_BM;
            int64_t i1 = i0 + TPU_GEMM_BM < M ? i0 + TPU_GEMM_BM : M;
            blocked_gemm_rows(Ab, K, Bb, N, Cb, N, i0, i1, N, K);
        }
    });
}

}
}
