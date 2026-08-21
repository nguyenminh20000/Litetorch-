#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
#include "litetorch/thread_pool.h"
#include "litetorch/amp.h"
#include <cmath>
#include <stdexcept>
#include <limits>
#include <vector>

namespace litetorch {

namespace {
struct StorageUseGuard {
    std::vector<std::shared_ptr<StorageImpl>> storages;
    StorageUseGuard(const std::vector<std::shared_ptr<StorageImpl>>& list) : storages(list) {
        for (auto& s : storages) {
            if (s) s->in_use = true;
        }
    }
    ~StorageUseGuard() {
        for (auto& s : storages) {
            if (s) s->in_use = false;
        }
    }
};
}

class FlashAttentionNode : public Node {
public:
    FlashAttentionNode() : Node("FlashAttention") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto q = saved_tensors[0]->is_contiguous() ? saved_tensors[0] : saved_tensors[0]->contiguous();
        auto k = saved_tensors[1]->is_contiguous() ? saved_tensors[1] : saved_tensors[1]->contiguous();
        auto v = saved_tensors[2]->is_contiguous() ? saved_tensors[2] : saved_tensors[2]->contiguous();

        int64_t B = q->shape[0];
        int64_t H = q->shape[1];
        int64_t Tq = q->shape[2];
        int64_t D = q->shape[3];
        int64_t H_kv = k->shape[1];
        int64_t Tk = k->shape[2];

        float scale = 1.0f / std::sqrt(static_cast<float>(D));

        auto out_shared = output.lock();
        if (!out_shared) return { nullptr, nullptr, nullptr };
        auto out_c = out_shared->is_contiguous() ? out_shared : out_shared->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();

        if (q->device.type == DeviceType::GPU && BackendDispatcher::get().get_backend() && BackendDispatcher::get().get_backend()->is_available()) {
            auto grad_q = Tensor::create(q->shape, q->device, false, q->dtype);
            auto grad_k = Tensor::create(k->shape, k->device, false, k->dtype);
            auto grad_v = Tensor::create(v->shape, v->device, false, v->dtype);
            std::vector<std::shared_ptr<StorageImpl>> storages;
            storages.push_back(q->storage);
            storages.push_back(k->storage);
            storages.push_back(v->storage);
            storages.push_back(out_c->storage);
            storages.push_back(gout_c->storage);
            storages.push_back(grad_q->storage);
            storages.push_back(grad_k->storage);
            storages.push_back(grad_v->storage);
            StorageUseGuard guard(storages);
            if (q->dtype == DataType::FP16) {
                BackendDispatcher::get().get_backend()->flash_attention_backward_half(
                    grad_q->gpu_data(), grad_q->offset, grad_k->gpu_data(), grad_k->offset, grad_v->gpu_data(), grad_v->offset,
                    out_c->gpu_data(), out_c->offset, gout_c->gpu_data(), gout_c->offset,
                    q->gpu_data(), q->offset, k->gpu_data(), k->offset, v->gpu_data(), v->offset,
                    B, H, H_kv, Tq, Tk, D, scale
                );
            } else {
                BackendDispatcher::get().get_backend()->flash_attention_backward(
                    grad_q->gpu_data(), grad_q->offset, grad_k->gpu_data(), grad_k->offset, grad_v->gpu_data(), grad_v->offset,
                    out_c->gpu_data(), out_c->offset, gout_c->gpu_data(), gout_c->offset,
                    q->gpu_data(), q->offset, k->gpu_data(), k->offset, v->gpu_data(), v->offset,
                    B, H, H_kv, Tq, Tk, D, scale
                );
            }
            std::vector<std::shared_ptr<Tensor>> ret;
            ret.push_back(grad_q);
            ret.push_back(grad_k);
            ret.push_back(grad_v);
            return ret;
        }

        auto k_rep = k;
        auto v_rep = v;
        if (H > H_kv) {
            auto k_vec = k->to_vector();
            auto v_vec = v->to_vector();
            std::vector<float> k_rep_vec(B * H * Tk * D);
            std::vector<float> v_rep_vec(B * H * Tk * D);
            int64_t g = H / H_kv;
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t h = 0; h < H; ++h) {
                    int64_t h_kv = h / g;
                    for (int64_t t = 0; t < Tk; ++t) {
                        for (int64_t d = 0; d < D; ++d) {
                            int64_t src_idx = ((b * H_kv + h_kv) * Tk + t) * D + d;
                            int64_t dst_idx = ((b * H + h) * Tk + t) * D + d;
                            k_rep_vec[dst_idx] = k_vec[src_idx];
                            v_rep_vec[dst_idx] = v_vec[src_idx];
                        }
                    }
                }
            }
            k_rep = Tensor::from_vector(k_rep_vec, {B, H, Tk, D}, q->device);
            v_rep = Tensor::from_vector(v_rep_vec, {B, H, Tk, D}, q->device);
        }

        auto q_3d = q->view({B * H, Tq, D});
        auto k_t_3d = k_rep->transpose(2, 3)->contiguous()->view({B * H, D, Tk});
        auto scores_3d = Ops::bmm(q_3d, k_t_3d);
        auto scores_4d = scores_3d->view({B, H, Tq, Tk});

        auto scale_tensor = Tensor::from_vector({scale}, {1}, q->device);
        auto scaled_scores = Ops::mul(scores_4d, scale_tensor);
        auto attn = Ops::softmax(scaled_scores, -1);

        auto attn_3d = attn->is_contiguous() ? attn->view({B * H, Tq, Tk}) : attn->contiguous()->view({B * H, Tq, Tk});
        auto v_3d = v_rep->is_contiguous() ? v_rep->view({B * H, Tk, D}) : v_rep->contiguous()->view({B * H, Tk, D});

        auto attn_t_3d = attn_3d->transpose(1, 2)->contiguous();
        auto grad_output_3d = gout_c->is_contiguous() ? gout_c->view({B * H, Tq, D}) : gout_c->contiguous()->view({B * H, Tq, D});
        auto grad_v_3d = Ops::bmm(attn_t_3d, grad_output_3d);
        auto grad_v = grad_v_3d->is_contiguous() ? grad_v_3d->view({B, H, Tk, D}) : grad_v_3d->contiguous()->view({B, H, Tk, D});

        auto v_t_3d = v_3d->transpose(1, 2)->contiguous();
        auto grad_attn_3d = Ops::bmm(grad_output_3d, v_t_3d);
        auto grad_attn = grad_attn_3d->is_contiguous() ? grad_attn_3d->view({B, H, Tq, Tk}) : grad_attn_3d->contiguous()->view({B, H, Tq, Tk});

        auto grad_attn_attn = Ops::mul(grad_attn, attn);
        auto sum_grad_attn_attn = Ops::reduce_broadcast(grad_attn_attn, {B, H, Tq, 1});
        auto grad_scores_unscaled = Ops::mul(attn, Ops::sub(grad_attn, sum_grad_attn_attn));
        auto grad_scores = Ops::mul(grad_scores_unscaled, scale_tensor);

        auto grad_scores_3d = grad_scores->is_contiguous() ? grad_scores->view({B * H, Tq, Tk}) : grad_scores->contiguous()->view({B * H, Tq, Tk});
        auto k_3d = k_rep->is_contiguous() ? k_rep->view({B * H, Tk, D}) : k_rep->contiguous()->view({B * H, Tk, D});
        auto grad_q_3d = Ops::bmm(grad_scores_3d, k_3d);
        auto grad_q = grad_q_3d->is_contiguous() ? grad_q_3d->view({B, H, Tq, D}) : grad_q_3d->contiguous()->view({B, H, Tq, D});

        auto grad_scores_t_3d = grad_scores_3d->transpose(1, 2)->contiguous();
        auto grad_k_3d = Ops::bmm(grad_scores_t_3d, q_3d);
        auto grad_k = grad_k_3d->is_contiguous() ? grad_k_3d->view({B, H, Tk, D}) : grad_k_3d->contiguous()->view({B, H, Tk, D});

        if (H > H_kv) {
            auto grad_k_vec = grad_k->to_vector();
            auto grad_v_vec = grad_v->to_vector();
            std::vector<float> gk_red(B * H_kv * Tk * D, 0.0f);
            std::vector<float> gv_red(B * H_kv * Tk * D, 0.0f);
            int64_t g = H / H_kv;
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t h_kv = 0; h_kv < H_kv; ++h_kv) {
                    for (int64_t t = 0; t < Tk; ++t) {
                        for (int64_t d = 0; d < D; ++d) {
                            float sum_k = 0.0f;
                            float sum_v = 0.0f;
                            for (int64_t j = 0; j < g; ++j) {
                                int64_t h = h_kv * g + j;
                                int64_t src_idx = ((b * H + h) * Tk + t) * D + d;
                                sum_k += grad_k_vec[src_idx];
                                sum_v += grad_v_vec[src_idx];
                            }
                            int64_t dst_idx = ((b * H_kv + h_kv) * Tk + t) * D + d;
                            gk_red[dst_idx] = sum_k;
                            gv_red[dst_idx] = sum_v;
                        }
                    }
                }
            }
            grad_k = Tensor::from_vector(gk_red, {B, H_kv, Tk, D}, q->device);
            grad_v = Tensor::from_vector(gv_red, {B, H_kv, Tk, D}, q->device);
        }

        std::vector<std::shared_ptr<Tensor>> ret;
        ret.push_back(grad_q);
        ret.push_back(grad_k);
        ret.push_back(grad_v);
        return ret;
    }
};

namespace Ops {

std::shared_ptr<Tensor> flash_attention(std::shared_ptr<Tensor> q, std::shared_ptr<Tensor> k, std::shared_ptr<Tensor> v) {
    if (q->shape.size() != 4 || k->shape.size() != 4 || v->shape.size() != 4) {
        throw std::runtime_error("[litetorch Error] flash_attention inputs must be 4D tensors of shape (B, H, T, D)");
    }
    int64_t B = q->shape[0];
    int64_t H = q->shape[1];
    int64_t Tq = q->shape[2];
    int64_t D = q->shape[3];
    int64_t H_kv = k->shape[1];
    int64_t Tk = k->shape[2];

    if (k->shape[0] != B || k->shape[3] != D ||
        v->shape[0] != B || v->shape[1] != H_kv || v->shape[2] != Tk || v->shape[3] != D) {
        throw std::runtime_error("[litetorch Error] Dimension mismatch in flash_attention inputs");
    }

    if (H % H_kv != 0) {
        throw std::runtime_error("[litetorch Error] Query heads H must be divisible by Key/Value heads H_kv");
    }

    if (amp::AutocastGuard::is_enabled()) {
        auto cast_dtype = amp::AutocastGuard::get_dtype();
        if (q->dtype != cast_dtype) q = Ops::cast(q, cast_dtype);
        if (k->dtype != cast_dtype) k = Ops::cast(k, cast_dtype);
        if (v->dtype != cast_dtype) v = Ops::cast(v, cast_dtype);
    }

    auto q_c = q->is_contiguous() ? q : q->contiguous();
    auto k_c = k->is_contiguous() ? k : k->contiguous();
    auto v_c = v->is_contiguous() ? v : v->contiguous();

    auto out = Tensor::create({B, H, Tq, D}, q_c->device, false, q_c->dtype);

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    bool run_gpu = false;
    if (q_c->device.type == DeviceType::GPU) {
        StorageUseGuard guard({q_c->storage, k_c->storage, v_c->storage, out->storage});
        auto native = BackendDispatcher::get().get_backend();
        if (native && native->is_available()) {
            run_gpu = true;
            if (q_c->dtype == DataType::FP16) {
                native->flash_attention_half(q_c->gpu_data(), q_c->offset, k_c->gpu_data(), k_c->offset, v_c->gpu_data(), v_c->offset, out->gpu_data(), out->offset, B, H, H_kv, Tq, Tk, D, scale);
            } else {
                native->flash_attention(q_c->gpu_data(), q_c->offset, k_c->gpu_data(), k_c->offset, v_c->gpu_data(), v_c->offset, out->gpu_data(), out->offset, B, H, H_kv, Tq, Tk, D, scale);
            }
        } else {
            void* kernel = nullptr;
            if (q_c->dtype == DataType::FP16) {
                kernel = CLBackend::get().get_kernel("", "", "flash_attention_half_forward");
            } else {
                kernel = CLBackend::get().get_kernel(KernelID::FlashAttentionForward);
            }
            if (kernel) {
                run_gpu = true;
                cl_mem q_mem = q_c->gpu_data();
                int q_off = q_c->offset;
                cl_mem k_mem = k_c->gpu_data();
                int k_off = k_c->offset;
                cl_mem v_mem = v_c->gpu_data();
                int v_off = v_c->offset;
                cl_mem out_mem = out->gpu_data();
                int out_off = out->offset;

                int b_val = B;
                int h_val = H;
                int h_kv_val = H_kv;
                int tq_val = Tq;
                int tk_val = Tk;
                int d_val = D;

                int total_threads = B * H * Tq;
                CLBackend::get().launch(kernel, {static_cast<size_t>(total_threads)}, {},
                    {&q_mem, &q_off, &k_mem, &k_off, &v_mem, &v_off, &out_mem, &out_off, &b_val, &h_val, &h_kv_val, &tq_val, &tk_val, &d_val, &scale},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int),
                     sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(float)});
            }
        }
    }
    if (!run_gpu) {
        if (q_c->dtype != DataType::FP32 || k_c->dtype != DataType::FP32 || v_c->dtype != DataType::FP32) {
            amp::AutocastGuard disable_guard(false);
            auto q_fp32 = q_c->dtype == DataType::FP32 ? q_c : q_c->cast(DataType::FP32);
            auto k_fp32 = k_c->dtype == DataType::FP32 ? k_c : k_c->cast(DataType::FP32);
            auto v_fp32 = v_c->dtype == DataType::FP32 ? v_c : v_c->cast(DataType::FP32);
            auto out_fp32 = flash_attention(q_fp32, k_fp32, v_fp32);
            auto out_fp16 = out_fp32->cast(q_c->dtype);
            out->copy_(out_fp16);
        } else {
            StorageUseGuard guard({q_c->storage, k_c->storage, v_c->storage, out->storage});
            float* q_ptr = q_c->data_ptr();
            float* k_ptr = k_c->data_ptr();
            float* v_ptr = v_c->data_ptr();
            float* out_ptr = out->data_ptr();

            int64_t g = H / H_kv;
            ThreadPool::get().parallel_for(0, B * H, [&](int64_t bh) {
                int64_t b = bh / H;
                int64_t h = bh % H;
                int64_t h_kv = h / g;

                int64_t q_offset = (b * H + h) * Tq * D;
                int64_t k_offset = (b * H_kv + h_kv) * Tk * D;
                int64_t v_offset = (b * H_kv + h_kv) * Tk * D;
                int64_t out_offset = (b * H + h) * Tq * D;

                thread_local std::vector<float> acc;
                thread_local std::vector<float> S;
                if (acc.size() < static_cast<size_t>(D)) acc.resize(D);
                if (S.size() < static_cast<size_t>(Tk)) S.resize(Tk);

                for (int64_t i = 0; i < Tq; ++i) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    float denominator = 0.0f;
                    std::fill(acc.begin(), acc.begin() + D, 0.0f);

                    for (int64_t j = 0; j < Tk; ++j) {
                        float dot = 0.0f;
                        for (int k_idx = 0; k_idx < D; ++k_idx) {
                            dot += q_ptr[q_offset + i * D + k_idx] * k_ptr[k_offset + j * D + k_idx];
                        }
                        dot *= scale;
                        S[j] = dot;
                        if (dot > max_val) max_val = dot;
                    }

                    for (int64_t j = 0; j < Tk; ++j) {
                        float exp_val = std::exp(S[j] - max_val);
                        denominator += exp_val;
                        for (int64_t k_idx = 0; k_idx < D; ++k_idx) {
                            acc[k_idx] += exp_val * v_ptr[v_offset + j * D + k_idx];
                        }
                    }

                    for (int64_t k_idx = 0; k_idx < D; ++k_idx) {
                        out_ptr[out_offset + i * D + k_idx] = acc[k_idx] / denominator;
                    }
                }
            });
            if (out->device.type == DeviceType::GPU) {
                CLBackend::get().write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
            }
        }
    }

    if (q->requires_grad || k->requires_grad || v->requires_grad) {
        auto node = std::make_shared<FlashAttentionNode>();
        node->inputs = { {q, q->requires_grad}, {k, k->requires_grad}, {v, v->requires_grad} };
        node->next_nodes = { q->creator, k->creator, v->creator };
        node->saved_tensors = { q, k, v };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }

    return out;
}

std::shared_ptr<Tensor> paged_attention(
    std::shared_ptr<Tensor> q,
    std::shared_ptr<Tensor> k_cache,
    std::shared_ptr<Tensor> v_cache,
    std::shared_ptr<Tensor> block_tables,
    std::shared_ptr<Tensor> context_lens,
    int block_size)
{
    auto out = Tensor::create(q->shape, q->device, false);
    int64_t num_seqs = q->shape[0];
    int64_t num_heads = q->shape[1];
    int64_t head_dim = q->shape[2];
    int64_t num_blocks = k_cache->shape[0];
    int64_t num_kv_heads = k_cache->shape[1];
    int64_t max_num_blocks_per_seq = block_tables->shape[1];
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    if (q->device.type == DeviceType::CPU) {
        float* q_ptr = q->data_ptr();
        void* k_void = k_cache->storage->cpu_data;
        void* v_void = v_cache->storage->cpu_data;
        DataType k_dtype = k_cache->dtype;
        DataType v_dtype = v_cache->dtype;

        int* bt_ptr = reinterpret_cast<int*>(block_tables->storage->cpu_data);
        int* cl_ptr = reinterpret_cast<int*>(context_lens->storage->cpu_data);
        float* out_ptr = out->data_ptr();

        auto get_val = [](void* ptr, int64_t idx, DataType dtype) -> float {
            if (dtype == DataType::FP32) {
                return static_cast<float*>(ptr)[idx];
            } else if (dtype == DataType::FP8_E4M3) {
                uint8_t val = static_cast<uint8_t*>(ptr)[idx];
                uint32_t sign = (val >> 7) & 1;
                uint32_t exp = (val >> 3) & 0x0F;
                uint32_t mant = val & 0x07;
                if (exp == 15) return 0.0f;
                if (exp == 0) {
                    if (mant == 0) return 0.0f;
                    return (sign ? -1.0f : 1.0f) * std::pow(2.0f, -6.0f) * (static_cast<float>(mant) / 8.0f);
                }
                return (sign ? -1.0f : 1.0f) * std::pow(2.0f, static_cast<float>(exp) - 7.0f) * (1.0f + static_cast<float>(mant) / 8.0f);
            } else if (dtype == DataType::INT8) {
                return static_cast<int8_t*>(ptr)[idx] / 127.0f;
            }
            return 0.0f;
        };

        ThreadPool::get().parallel_for(0, num_seqs * num_heads, [&](int64_t id) {
            int64_t seq_idx = id / num_heads;
            int64_t head_idx = id % num_heads;
            int64_t kv_head_idx = head_idx / (num_heads / num_kv_heads);
            int64_t context_len = cl_ptr[seq_idx];
            if (context_len <= 0) return;

            float max_val = -std::numeric_limits<float>::infinity();
            std::vector<float> scores(context_len);

            for (int64_t t = 0; t < context_len; ++t) {
                int64_t block_idx = bt_ptr[seq_idx * max_num_blocks_per_seq + t / block_size];
                int64_t block_offset = t % block_size;
                int64_t k_idx = block_idx * (num_kv_heads * block_size * head_dim) + kv_head_idx * (block_size * head_dim) + block_offset * head_dim;

                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += q_ptr[seq_idx * (num_heads * head_dim) + head_idx * head_dim + d] * get_val(k_void, k_idx + d, k_dtype);
                }
                dot *= scale;
                scores[t] = dot;
                if (dot > max_val) max_val = dot;
            }

            float denominator = 0.0f;
            std::vector<float> acc(head_dim, 0.0f);

            for (int64_t t = 0; t < context_len; ++t) {
                float exp_val = std::exp(scores[t] - max_val);
                denominator += exp_val;

                int64_t block_idx = bt_ptr[seq_idx * max_num_blocks_per_seq + t / block_size];
                int64_t block_offset = t % block_size;
                int64_t v_idx = block_idx * (num_kv_heads * block_size * head_dim) + kv_head_idx * (block_size * head_dim) + block_offset * head_dim;

                for (int64_t d = 0; d < head_dim; ++d) {
                    acc[d] += exp_val * get_val(v_void, v_idx + d, v_dtype);
                }
            }

            for (int64_t d = 0; d < head_dim; ++d) {
                out_ptr[seq_idx * (num_heads * head_dim) + head_idx * head_dim + d] = acc[d] / denominator;
            }
        });
    } else if (q->device.type == DeviceType::GPU) {
        auto k_to_use = k_cache->dtype == DataType::FP32 ? k_cache : k_cache->cast(DataType::FP32);
        auto v_to_use = v_cache->dtype == DataType::FP32 ? v_cache : v_cache->cast(DataType::FP32);

        StorageUseGuard guard({q->storage, k_to_use->storage, v_to_use->storage, block_tables->storage, context_lens->storage, out->storage});
        cl_mem q_mem = q->gpu_data();
        int q_off = q->offset;
        cl_mem k_mem = k_to_use->gpu_data();
        int k_off = k_to_use->offset;
        cl_mem v_mem = v_to_use->gpu_data();
        int v_off = v_to_use->offset;
        cl_mem bt_mem = block_tables->gpu_data();
        int bt_off = block_tables->offset;
        cl_mem cl_mem_val = context_lens->gpu_data();
        int cl_off = context_lens->offset;
        cl_mem o_mem = out->gpu_data();
        int o_off = out->offset;

        int num_seqs_val = static_cast<int>(num_seqs);
        int num_heads_val = static_cast<int>(num_heads);
        int num_kv_heads_val = static_cast<int>(num_kv_heads);
        int head_dim_val = static_cast<int>(head_dim);
        int max_num_blocks_val = static_cast<int>(max_num_blocks_per_seq);

        auto kernel = CLBackend::get().get_kernel(KernelID::PagedAttentionForward);
        CLBackend::get().launch(kernel, {static_cast<size_t>(num_seqs * num_heads)}, {},
            {&q_mem, &q_off, &k_mem, &k_off, &v_mem, &v_off, &bt_mem, &bt_off, &cl_mem_val, &cl_off, &o_mem, &o_off,
             &num_seqs_val, &num_heads_val, &num_kv_heads_val, &head_dim_val, &max_num_blocks_val, &block_size, &scale},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int),
             sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int),
             sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(float)});
    }

    return out;
}

}
}
