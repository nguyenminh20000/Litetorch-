#include "../common/tpu_common.h"
#include <algorithm>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace litetorch {
namespace tpu_internal {

void tpu_systolic_matmul(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    std::memset(C, 0, M * N * sizeof(float));
    constexpr int64_t TILE = TPU_SYSTOLIC_TILE_SIZE;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t m0 = 0; m0 < M; m0 += TILE) {
        for (int64_t n0 = 0; n0 < N; n0 += TILE) {
            int64_t m_max = std::min(m0 + TILE, M);
            int64_t n_max = std::min(n0 + TILE, N);
            for (int64_t k0 = 0; k0 < K; k0 += TILE) {
                int64_t k_max = std::min(k0 + TILE, K);
                for (int64_t m = m0; m < m_max; ++m) {
                    for (int64_t k = k0; k < k_max; ++k) {
                        float a_val = A[m * K + k];
                        for (int64_t n = n0; n < n_max; ++n) {
                            C[m * N + n] += a_val * B[k * N + n];
                        }
                    }
                }
            }
        }
    }
}

void tpu_systolic_matmul_ex(const float* A, bool trans_a, int64_t lda,
                            const float* B, bool trans_b, int64_t ldb,
                            float* C, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum_val = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                float a_val = trans_a ? A[k * lda + i] : A[i * lda + k];
                float b_val = trans_b ? B[j * ldb + k] : B[k * ldb + j];
                sum_val += a_val * b_val;
            }
            C[i * N + j] = sum_val;
        }
    }
}

void tpu_systolic_bmm(const float* A, const float* B, float* C, int64_t B_batch, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    for (int64_t b = 0; b < B_batch; ++b) {
        const float* a_ptr = A + b * M * K;
        const float* b_ptr = B + b * K * N;
        float* c_ptr = C + b * M * N;
        tpu_systolic_matmul(a_ptr, b_ptr, c_ptr, M, N, K);
    }
}

}
}
