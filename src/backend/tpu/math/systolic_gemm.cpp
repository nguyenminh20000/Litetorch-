#include "../common/tpu_common.h"
#include "litetorch/thread_pool.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace litetorch {
namespace tpu_internal {

void tpu_systolic_matmul(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    std::memset(C, 0, M * N * sizeof(float));

    ThreadPool::get().parallel_for(0, M, [&](int64_t i) {
        float* C_row = C + i * N;
        for (int64_t k = 0; k < K; ++k) {
            float aval = A[i * K + k];
            const float* B_row = B + k * N;
            for (int64_t j = 0; j < N; ++j) {
                C_row[j] += aval * B_row[j];
            }
        }
    });
}

void tpu_systolic_matmul_ex(const float* A, bool trans_a, int64_t lda,
                            const float* B, bool trans_b, int64_t ldb,
                            float* C, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    std::memset(C, 0, M * N * sizeof(float));

    if (!trans_a && !trans_b) {
        ThreadPool::get().parallel_for(0, M, [&](int64_t i) {
            float* C_row = C + i * N;
            const float* A_row = A + i * lda;
            for (int64_t k = 0; k < K; ++k) {
                float aval = A_row[k];
                const float* B_row = B + k * ldb;
                for (int64_t j = 0; j < N; ++j) {
                    C_row[j] += aval * B_row[j];
                }
            }
        });
    } else if (trans_a && !trans_b) {
        ThreadPool::get().parallel_for(0, M, [&](int64_t i) {
            float* C_row = C + i * N;
            for (int64_t k = 0; k < K; ++k) {
                float aval = A[k * lda + i];
                const float* B_row = B + k * ldb;
                for (int64_t j = 0; j < N; ++j) {
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
        const float* a_ptr = A + b * M * K;
        const float* b_ptr = B + b * K * N;
        float* c_ptr = C + b * M * N;
        std::memset(c_ptr, 0, M * N * sizeof(float));
        for (int64_t i = 0; i < M; ++i) {
            float* C_row = c_ptr + i * N;
            for (int64_t k = 0; k < K; ++k) {
                float aval = a_ptr[i * K + k];
                const float* B_row = b_ptr + k * N;
                for (int64_t j = 0; j < N; ++j) {
                    C_row[j] += aval * B_row[j];
                }
            }
        }
    });
}

}
}
