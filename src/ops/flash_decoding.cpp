#include "litetorch/ops.h"
#include "litetorch/tensor.h"
#include "litetorch/thread_pool.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace litetorch {
namespace Ops {

std::shared_ptr<Tensor> flash_decoding(
    std::shared_ptr<Tensor> q,
    std::shared_ptr<Tensor> k,
    std::shared_ptr<Tensor> v,
    int num_splits)
{
    int64_t B = q->shape[0];
    int64_t H = q->shape[1];
    int64_t T_q = q->shape[2];
    int64_t D = q->shape[3];
    int64_t T_k = k->shape[2];

    if (T_q != 1 || q->device.type == DeviceType::GPU) {
        return flash_attention(q, k, v);
    }

    if (num_splits <= 1) {
        return flash_attention(q, k, v);
    }

    auto out = Tensor::zeros({B, H, 1, D}, q->device, false);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    float* q_ptr = q->data_ptr();
    float* k_ptr = k->data_ptr();
    float* v_ptr = v->data_ptr();
    float* out_ptr = out->data_ptr();

    int64_t chunk_size = (T_k + num_splits - 1) / num_splits;

    ThreadPool::get().parallel_for(0, B * H, [&](int64_t idx) {
        int64_t b = idx / H;
        int64_t h = idx % H;

        std::vector<float> m_s(num_splits, -std::numeric_limits<float>::infinity());
        std::vector<float> d_s(num_splits, 0.0f);
        std::vector<std::vector<float>> O_s(num_splits, std::vector<float>(D, 0.0f));

        for (int s = 0; s < num_splits; ++s) {
            int64_t start_idx = s * chunk_size;
            int64_t end_idx = std::min(start_idx + chunk_size, T_k);
            if (start_idx >= end_idx) continue;

            float m_local = -std::numeric_limits<float>::infinity();
            std::vector<float> S(end_idx - start_idx);

            for (int64_t j = start_idx; j < end_idx; ++j) {
                float dot = 0.0f;
                int64_t q_offset = b * (H * T_q * D) + h * (T_q * D);
                int64_t k_offset = b * (H * T_k * D) + h * (T_k * D) + j * D;
                for (int64_t d = 0; d < D; ++d) {
                    dot += q_ptr[q_offset + d] * k_ptr[k_offset + d];
                }
                dot *= scale;
                S[j - start_idx] = dot;
                if (dot > m_local) m_local = dot;
            }

            float sum_exp = 0.0f;
            std::vector<float> exp_S(end_idx - start_idx);
            for (int64_t j = start_idx; j < end_idx; ++j) {
                float e = std::exp(S[j - start_idx] - m_local);
                exp_S[j - start_idx] = e;
                sum_exp += e;
            }

            m_s[s] = m_local;
            d_s[s] = sum_exp;

            for (int64_t d = 0; d < D; ++d) {
                float sum_v = 0.0f;
                for (int64_t j = start_idx; j < end_idx; ++j) {
                    int64_t v_offset = b * (H * T_k * D) + h * (T_k * D) + j * D;
                    sum_v += exp_S[j - start_idx] * v_ptr[v_offset + d];
                }
                O_s[s][d] = sum_v;
            }
        }

        float m_global = -std::numeric_limits<float>::infinity();
        for (int s = 0; s < num_splits; ++s) {
            if (m_s[s] > m_global) m_global = m_s[s];
        }

        float d_global = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            if (m_s[s] == -std::numeric_limits<float>::infinity()) continue;
            d_global += d_s[s] * std::exp(m_s[s] - m_global);
        }

        int64_t out_offset = b * (H * T_q * D) + h * (T_q * D);
        for (int64_t d = 0; d < D; ++d) {
            float sum_O = 0.0f;
            for (int s = 0; s < num_splits; ++s) {
                if (m_s[s] == -std::numeric_limits<float>::infinity()) continue;
                sum_O += O_s[s][d] * std::exp(m_s[s] - m_global);
            }
            out_ptr[out_offset + d] = sum_O / (d_global + 1e-15f);
        }
    });

    return out;
}

}
}
