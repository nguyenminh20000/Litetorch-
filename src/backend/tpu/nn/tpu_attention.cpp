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
    ThreadPool::get().parallel_for(0, B * H, [&](int64_t bh) {
        int64_t b = bh / H;
        int64_t h = bh % H;
        int64_t h_kv = h / group_ratio;

        thread_local std::vector<float> scores;
        if (static_cast<int64_t>(scores.size()) < Tk) {
            scores.resize(Tk);
        }

        for (int64_t i = 0; i < Tq; ++i) {
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
            for (int64_t d = 0; d < D; ++d) {
                float val = 0.0f;
                for (int64_t j = 0; j < Tk; ++j) {
                    const float* v_vec = V + b * (H_kv * Tk * D) + h_kv * (Tk * D) + j * D;
                    val += scores[j] * v_vec[d];
                }
                out_vec[d] = val;
            }
        }
    });
}

}
}
