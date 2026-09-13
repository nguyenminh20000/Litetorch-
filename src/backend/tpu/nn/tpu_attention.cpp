#include "../common/tpu_common.h"
#include "litetorch/thread_pool.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace litetorch {
namespace tpu_internal {

void tpu_flash_attention_forward(const float* Q, const float* K, const float* V, float* O,
                                 int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) {
    if (!Q || !K || !V || !O) return;

    int64_t group_ratio = H / H_kv;
    int64_t total_tokens = B * H * Tq;

    ThreadPool::get().parallel_for(0, total_tokens, [&](int64_t idx) {
        int64_t b = idx / (H * Tq);
        int64_t rem = idx % (H * Tq);
        int64_t h = rem / Tq;
        int64_t i = rem % Tq;
        int64_t h_kv = h / group_ratio;

        thread_local std::vector<float> scores;
        if (static_cast<int64_t>(scores.size()) < Tk) {
            scores.resize(Tk);
        }

        const float* q_vec = Q + b * (H * Tq * D) + h * (Tq * D) + i * D;
        float max_score = -1e9f;
        for (int64_t j = 0; j < Tk; ++j) {
            const float* k_vec = K + b * (H_kv * Tk * D) + h_kv * (Tk * D) + j * D;
            float dot = 0.0f;
            for (int64_t d = 0; d < D; ++d) {
                dot += q_vec[d] * k_vec[d];
            }
            float sc = dot * scale;
            scores[j] = sc;
            if (sc > max_score) max_score = sc;
        }

        float sum_exp = 0.0f;
        for (int64_t j = 0; j < Tk; ++j) {
            scores[j] = std::exp(scores[j] - max_score);
            sum_exp += scores[j];
        }

        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (int64_t j = 0; j < Tk; ++j) {
            scores[j] *= inv_sum;
        }

        float* out_vec = O + b * (H * Tq * D) + h * (Tq * D) + i * D;
        std::memset(out_vec, 0, D * sizeof(float));
        for (int64_t j = 0; j < Tk; ++j) {
            float sc = scores[j];
            if (sc <= 0.0f) continue;
            const float* v_vec = V + b * (H_kv * Tk * D) + h_kv * (Tk * D) + j * D;
            for (int64_t d = 0; d < D; ++d) {
                out_vec[d] += sc * v_vec[d];
            }
        }
    });
}

void tpu_flash_attention_backward(const float* Q, const float* K, const float* V, const float* O, const float* dO,
                                  float* dQ, float* dK, float* dV,
                                  int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) {
    if (!Q || !K || !V || !O || !dO || !dQ || !dK || !dV) return;

    std::memset(dQ, 0, B * H * Tq * D * sizeof(float));
    std::memset(dK, 0, B * H_kv * Tk * D * sizeof(float));
    std::memset(dV, 0, B * H_kv * Tk * D * sizeof(float));

    int64_t group_ratio = H / H_kv;
    std::vector<std::mutex> kv_locks(B * H_kv);

    ThreadPool::get().parallel_for(0, B * H, [&](int64_t idx) {
        int64_t b = idx / H;
        int64_t h = idx % H;
        int64_t h_kv = h / group_ratio;
        int64_t kv_idx = b * H_kv + h_kv;

        std::vector<float> local_dk(Tk * D, 0.0f);
        std::vector<float> local_dv(Tk * D, 0.0f);
        std::vector<float> scores(Tk);

        const float* k_base = K + b * (H_kv * Tk * D) + h_kv * (Tk * D);
        const float* v_base = V + b * (H_kv * Tk * D) + h_kv * (Tk * D);

        for (int64_t i = 0; i < Tq; ++i) {
            const float* q_vec = Q + b * (H * Tq * D) + h * (Tq * D) + i * D;
            const float* o_vec = O + b * (H * Tq * D) + h * (Tq * D) + i * D;
            const float* do_vec = dO + b * (H * Tq * D) + h * (Tq * D) + i * D;
            float* dq_vec = dQ + b * (H * Tq * D) + h * (Tq * D) + i * D;

            float max_score = -1e9f;
            for (int64_t j = 0; j < Tk; ++j) {
                const float* k_vec = k_base + j * D;
                float dot = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    dot += q_vec[d] * k_vec[d];
                }
                float sc = dot * scale;
                scores[j] = sc;
                if (sc > max_score) max_score = sc;
            }

            float sum_exp = 0.0f;
            for (int64_t j = 0; j < Tk; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }

            float inv_sum = 1.0f / (sum_exp + 1e-8f);
            for (int64_t j = 0; j < Tk; ++j) {
                scores[j] *= inv_sum;
            }

            float Di = 0.0f;
            for (int64_t d = 0; d < D; ++d) {
                Di += do_vec[d] * o_vec[d];
            }

            for (int64_t j = 0; j < Tk; ++j) {
                float P_j = scores[j];
                const float* v_vec = v_base + j * D;
                const float* k_vec = k_base + j * D;

                float dP_j = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    dP_j += do_vec[d] * v_vec[d];
                }

                float dS_j = P_j * (dP_j - Di) * scale;

                for (int64_t d = 0; d < D; ++d) {
                    dq_vec[d] += dS_j * k_vec[d];
                    local_dk[j * D + d] += dS_j * q_vec[d];
                    local_dv[j * D + d] += P_j * do_vec[d];
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(kv_locks[kv_idx]);
            float* dk_base = dK + b * (H_kv * Tk * D) + h_kv * (Tk * D);
            float* dv_base = dV + b * (H_kv * Tk * D) + h_kv * (Tk * D);
            for (int64_t n = 0; n < Tk * D; ++n) {
                dk_base[n] += local_dk[n];
                dv_base[n] += local_dv[n];
            }
        }
    });
}

}
}
