#include "litetorch/ops.h"
#include "litetorch/tensor.h"
#include "litetorch/distributed.h"
#include "litetorch/thread_pool.h"
#include <cmath>
#include <stdexcept>
#include <limits>
#include <vector>

namespace litetorch {
namespace Ops {

std::shared_ptr<Tensor> ring_attention(
    std::shared_ptr<Tensor> q,
    std::shared_ptr<Tensor> k,
    std::shared_ptr<Tensor> v,
    std::shared_ptr<distributed::ProcessGroup> pg)
{
    if (!pg || pg->get_world_size() <= 1) {
        return flash_attention(q, k, v);
    }

    int rank = pg->get_rank();
    int world_size = pg->get_world_size();

    int64_t B = q->shape[0];
    int64_t H = q->shape[1];
    int64_t T_local = q->shape[2];
    int64_t D = q->shape[3];

    auto k_curr = k->is_contiguous() ? k : k->contiguous();
    auto v_curr = v->is_contiguous() ? v : v->contiguous();



    auto out = Tensor::zeros({B, H, T_local, D}, q->device, false);
    auto M = Tensor::create({B, H, T_local}, q->device, false);
    auto L = Tensor::zeros({B, H, T_local}, q->device, false);

    float* M_ptr = M->data_ptr();
    for (size_t i = 0; i < M->numel(); ++i) {
        M_ptr[i] = -std::numeric_limits<float>::infinity();
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    int dst = (rank + 1) % world_size;
    int src = (rank - 1 + world_size) % world_size;

    for (int step = 0; step < world_size; ++step) {
        float* q_ptr = q->data_ptr();
        float* k_ptr = k_curr->data_ptr();
        float* v_ptr = v_curr->data_ptr();
        float* out_ptr = out->data_ptr();
        float* L_ptr = L->data_ptr();

        ThreadPool::get().parallel_for(0, B * H * T_local, [&](int64_t idx) {
            int64_t b = idx / (H * T_local);
            int64_t h = (idx % (H * T_local)) / T_local;
            int64_t t = idx % T_local;

            float m_old = M_ptr[idx];
            float l_old = L_ptr[idx];

            std::vector<float> S(T_local);
            float m_local = -std::numeric_limits<float>::infinity();

            for (int64_t j = 0; j < T_local; ++j) {
                float dot = 0.0f;
                int64_t q_offset = b * (H * T_local * D) + h * (T_local * D) + t * D;
                int64_t k_offset = b * (H * T_local * D) + h * (T_local * D) + j * D;
                for (int64_t d = 0; d < D; ++d) {
                    dot += q_ptr[q_offset + d] * k_ptr[k_offset + d];
                }
                dot *= scale;
                S[j] = dot;
                if (dot > m_local) m_local = dot;
            }

            float m_new = std::max(m_old, m_local);
            float exp_old = std::exp(m_old - m_new);

            float sum_exp = 0.0f;
            std::vector<float> exp_S(T_local);
            for (int64_t j = 0; j < T_local; ++j) {
                float e = std::exp(S[j] - m_new);
                exp_S[j] = e;
                sum_exp += e;
            }

            float l_new = l_old * exp_old + sum_exp;

            int64_t out_offset = b * (H * T_local * D) + h * (T_local * D) + t * D;
            for (int64_t d = 0; d < D; ++d) {
                float val_old = out_ptr[out_offset + d] * l_old * exp_old;
                float val_new = 0.0f;
                for (int64_t j = 0; j < T_local; ++j) {
                    int64_t v_offset = b * (H * T_local * D) + h * (T_local * D) + j * D;
                    val_new += exp_S[j] * v_ptr[v_offset + d];
                }
                out_ptr[out_offset + d] = (val_old + val_new) / (l_new + 1e-15f);
            }

            M_ptr[idx] = m_new;
            L_ptr[idx] = l_new;
        });

        if (step < world_size - 1) {
            auto k_next = Tensor::create(k_curr->shape, k_curr->device, false);
            auto v_next = Tensor::create(v_curr->shape, v_curr->device, false);

            if (rank % 2 == 0) {
                pg->send_tensor(k_curr, dst);
                pg->recv_tensor(k_next, src);
                pg->send_tensor(v_curr, dst);
                pg->recv_tensor(v_next, src);
            } else {
                pg->recv_tensor(k_next, src);
                pg->send_tensor(k_curr, dst);
                pg->recv_tensor(v_next, src);
                pg->send_tensor(v_curr, dst);
            }

            k_curr = k_next;
            v_curr = v_next;
        }
    }

    return out;
}

}
}
