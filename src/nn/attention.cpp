#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace litetorch {
namespace nn {

namespace {
std::shared_ptr<Tensor> ensure_contiguous(std::shared_ptr<Tensor> t) {
    return t->is_contiguous() ? t : t->contiguous();
}

std::shared_ptr<Tensor> build_additive_mask(std::shared_ptr<Tensor> attn_mask,
                                            std::shared_ptr<Tensor> key_padding_mask,
                                            int64_t B, int64_t H, int64_t Tq, int64_t Tk,
                                            const Device& device) {
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    int64_t BH = B * H;
    auto mask = Tensor::create({BH, Tq, Tk}, device, false, DataType::FP32);
    float* mptr = mask->data_ptr();
    std::fill(mptr, mptr + mask->numel(), 0.0f);
    if (attn_mask) {
        if (attn_mask->device.type != DeviceType::CPU) {
            throw std::runtime_error("[litetorch Error] MultiHeadAttention: attn_mask must be a CPU tensor");
        }
        auto mc = attn_mask->is_contiguous() ? attn_mask : attn_mask->contiguous();
        auto& sh = mc->shape;
        bool is_2d = (sh.size() == 2 && sh[0] == Tq && sh[1] == Tk);
        bool is_3d = (sh.size() == 3 && sh[0] == BH && sh[1] == Tq && sh[2] == Tk);
        if (!is_2d && !is_3d) {
            throw std::runtime_error("[litetorch Error] MultiHeadAttention: attn_mask must have shape (Tq, Tk) or (B*num_heads, Tq, Tk)");
        }
        bool is_float = (mc->dtype == DataType::FP32 || mc->dtype == DataType::FP64);
        auto vals = mc->to_vector();
        for (int64_t bh = 0; bh < BH; ++bh) {
            for (int64_t i = 0; i < Tq; ++i) {
                for (int64_t j = 0; j < Tk; ++j) {
                    float v = vals[is_2d ? (i * Tk + j) : ((bh * Tq + i) * Tk + j)];
                    float add = is_float ? v : (v != 0.0f ? NEG_INF : 0.0f);
                    mptr[(bh * Tq + i) * Tk + j] += add;
                }
            }
        }
    }
    if (key_padding_mask) {
        if (key_padding_mask->device.type != DeviceType::CPU) {
            throw std::runtime_error("[litetorch Error] MultiHeadAttention: key_padding_mask must be a CPU tensor");
        }
        auto mc = key_padding_mask->is_contiguous() ? key_padding_mask : key_padding_mask->contiguous();
        auto& sh = mc->shape;
        if (!(sh.size() == 2 && sh[0] == B && sh[1] == Tk)) {
            throw std::runtime_error("[litetorch Error] MultiHeadAttention: key_padding_mask must have shape (B, Tk)");
        }
        auto vals = mc->to_vector();
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t j = 0; j < Tk; ++j) {
                if (vals[b * Tk + j] == 0.0f) continue;
                for (int64_t h = 0; h < H; ++h) {
                    float* col = mptr + ((b * H + h) * Tq) * Tk + j;
                    for (int64_t i = 0; i < Tq; ++i) col[i * Tk] = NEG_INF;
                }
            }
        }
    }
    return mask;
}
}

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads, double dropout, bool bias,
                                       int kdim, int vdim, bool batch_first)
    : embed_dim(embed_dim), num_heads(num_heads),
      dropout_p(dropout), use_bias(bias), batch_first(batch_first) {
    if (embed_dim % num_heads != 0) {
        throw std::runtime_error("embed_dim must be divisible by num_heads");
    }
    if (dropout < 0.0 || dropout >= 1.0) {
        throw std::runtime_error("dropout probability must lie in [0, 1)");
    }
    head_dim = embed_dim / num_heads;
    this->kdim = (kdim < 0) ? embed_dim : kdim;
    this->vdim = (vdim < 0) ? embed_dim : vdim;

    out_proj = std::make_shared<Linear>(embed_dim, embed_dim, bias);
    attn_dropout = std::make_shared<Dropout>(static_cast<float>(dropout));
    if (this->kdim == embed_dim && this->vdim == embed_dim) {
        qkv_proj = std::make_shared<Linear>(embed_dim, 3 * embed_dim, bias);
    } else {
        q_proj = std::make_shared<Linear>(embed_dim, embed_dim, bias);
        k_proj = std::make_shared<Linear>(this->kdim, embed_dim, bias);
        v_proj = std::make_shared<Linear>(this->vdim, embed_dim, bias);
    }
}

std::shared_ptr<Tensor> MultiHeadAttention::forward(std::shared_ptr<Tensor> input) {
    return forward(input, input, input);
}

std::shared_ptr<Tensor> MultiHeadAttention::forward(std::shared_ptr<Tensor> query, std::shared_ptr<Tensor> key, std::shared_ptr<Tensor> value) {
    return forward(query, key, value, nullptr, nullptr, false).first;
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> MultiHeadAttention::forward(
    std::shared_ptr<Tensor> query, std::shared_ptr<Tensor> key, std::shared_ptr<Tensor> value,
    std::shared_ptr<Tensor> attn_mask, std::shared_ptr<Tensor> key_padding_mask, bool need_weights) {
    if (!batch_first) {
        query = ensure_contiguous(query->transpose(0, 1));
        key = ensure_contiguous(key->transpose(0, 1));
        value = ensure_contiguous(value->transpose(0, 1));
    }
    if (query->shape.size() != 3 || key->shape.size() != 3 || value->shape.size() != 3) {
        throw std::runtime_error("MultiHeadAttention inputs must be 3D tensors of shape (N, L, E) when batch_first=true or (L, N, E) otherwise");
    }
    int64_t B = query->shape[0];
    int64_t Tq = query->shape[1];
    int64_t Cq = query->shape[2];

    int64_t Tk = key->shape[1];
    int64_t Ck = key->shape[2];

    int64_t Tv = value->shape[1];
    int64_t Cv = value->shape[2];

    if (Cq != embed_dim || Ck != kdim || Cv != vdim) {
        throw std::runtime_error("Input channel dimensions do not match embed_dim/kdim/vdim");
    }
    if (key->shape[0] != B || value->shape[0] != B) {
        throw std::runtime_error("Batch dimensions of query, key, and value must match");
    }
    if (Tk != Tv) {
        throw std::runtime_error("Sequence length of key and value must match");
    }

    bool self_attn = (query == key && key == value);
    bool use_masks = (attn_mask != nullptr) || (key_padding_mask != nullptr);
    std::shared_ptr<Tensor> out_4d;
    std::shared_ptr<Tensor> attn_weights = nullptr;
    if (qkv_proj && self_attn && dropout_p == 0.0 && !use_masks && !need_weights) {
        auto qkv = qkv_proj->forward(query);
        out_4d = Ops::flash_attention_qkv(ensure_contiguous(qkv), num_heads);
    } else {
        std::shared_ptr<Tensor> q4d, k4d, v4d;
        if (qkv_proj) {
            auto qq = ensure_contiguous(qkv_proj->forward(query));
            auto kk = ensure_contiguous(qkv_proj->forward(key));
            auto vv = ensure_contiguous(qkv_proj->forward(value));
            if (qq->dtype == DataType::FP32) {
                q4d = Ops::qkv_extract(qq, num_heads, 0);
                k4d = Ops::qkv_extract(kk, num_heads, 1);
                v4d = Ops::qkv_extract(vv, num_heads, 2);
            } else {
                auto split_qkv = [&](std::shared_ptr<Tensor> x, int64_t T, int64_t index) {
                    auto parts = Ops::chunk(x, 3, 2);
                    auto p = parts[static_cast<size_t>(index)];
                    auto pc = p->is_contiguous() ? p : p->contiguous();
                    return pc->view({B, T, num_heads, head_dim})->transpose(1, 2);
                };
                q4d = split_qkv(qq, Tq, 0);
                k4d = split_qkv(kk, Tk, 1);
                v4d = split_qkv(vv, Tv, 2);
            }
        } else {
            auto split_heads = [&](std::shared_ptr<Tensor> x, int64_t T) {
                auto r = ensure_contiguous(x)->view({B, T, num_heads, head_dim});
                return ensure_contiguous(r->transpose(1, 2));
            };
            q4d = split_heads(q_proj->forward(query), Tq);
            k4d = split_heads(k_proj->forward(key), Tk);
            v4d = split_heads(v_proj->forward(value), Tv);
        }
        auto qbh = ensure_contiguous(q4d)->view({B * num_heads, Tq, head_dim});
        auto kbh = ensure_contiguous(k4d)->view({B * num_heads, Tk, head_dim});
        auto vbh = ensure_contiguous(v4d)->view({B * num_heads, Tv, head_dim});
        auto kbh_t = ensure_contiguous(kbh->transpose(1, 2));
        auto scores = Ops::bmm(qbh, kbh_t);
        auto scale = Tensor::from_vector({1.0f / std::sqrt(static_cast<float>(head_dim))}, {1}, scores->device, false, scores->dtype);
        scores = Ops::mul(scores, scale);
        if (use_masks) {
            if (scores->device.type != DeviceType::CPU) {
                throw std::runtime_error("[litetorch Error] MultiHeadAttention: attn_mask/key_padding_mask on GPU is not implemented; use CPU tensors (deferred to Phase 3)");
            }
            auto add_mask = build_additive_mask(attn_mask, key_padding_mask, B, num_heads, Tq, Tk, scores->device);
            auto add_cast = (add_mask->dtype == scores->dtype) ? add_mask : add_mask->cast(scores->dtype);
            scores = Ops::add(scores, add_cast);
        }
        auto probs = Ops::softmax(scores, -1);
        if (need_weights) {
            attn_weights = ensure_contiguous(probs)->view({B, num_heads, Tq, Tk});
        }
        if (dropout_p > 0.0) {
            probs = attn_dropout->forward(probs);
        }
        auto ctx = Ops::bmm(probs, vbh);
        out_4d = ensure_contiguous(ctx)->view({B, num_heads, Tq, head_dim});
    }
    auto out_transposed = out_4d->transpose(1, 2);
    auto out_contiguous = ensure_contiguous(out_transposed);
    auto out_flat = out_contiguous->view({B, Tq, embed_dim});

    auto proj_out = out_proj->forward(out_flat);
    if (!batch_first) {
        proj_out = ensure_contiguous(proj_out->transpose(0, 1));
    }
    return {proj_out, attn_weights};
}

std::vector<std::shared_ptr<Tensor>> MultiHeadAttention::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto& m : {qkv_proj, q_proj, k_proj, v_proj, out_proj}) {
        if (!m) continue;
        auto mp = m->parameters();
        params.insert(params.end(), mp.begin(), mp.end());
    }
    return params;
}

void MultiHeadAttention::to(const Device& device) {
    for (auto& m : {qkv_proj, q_proj, k_proj, v_proj, out_proj}) {
        if (m) m->to(device);
    }
}

void MultiHeadAttention::train() {
    training = true;
    for (auto& c : children()) {
        if (c) c->train();
    }
}

void MultiHeadAttention::eval() {
    training = false;
    for (auto& c : children()) {
        if (c) c->eval();
    }
}

}
}
