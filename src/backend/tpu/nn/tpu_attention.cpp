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

}
}
