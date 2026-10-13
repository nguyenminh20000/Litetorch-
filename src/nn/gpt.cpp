#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <cmath>
#include <vector>

namespace litetorch {
namespace nn {

RMSNorm::RMSNorm(const std::vector<int64_t>& normalized_shape, float eps)
    : normalized_shape(normalized_shape), eps(eps) {
    int64_t dim = normalized_shape.back();
    weight = Tensor::create({dim}, Device(DeviceType::CPU, 0), true);
    float* data = weight->data_ptr();
    for (int64_t i = 0; i < dim; ++i) {
        data[i] = 1.0f;
    }
}

std::shared_ptr<Tensor> RMSNorm::forward(std::shared_ptr<Tensor> x) {
    auto w = weight;
    if (w->device != x->device || w->dtype != x->dtype) {
        w = w->to(x->device, x->dtype);
    }
    auto out = Ops::rmsnorm(x, w, eps);
    return out->view(x->shape);
}

std::vector<std::shared_ptr<Tensor>> RMSNorm::parameters() {
    return {weight};
}

void RMSNorm::to(const Device& device) {
    if (weight->device != device) {
        weight = weight->to(device);
    }
}

void RMSNorm::half() {
    weight = weight->cast(DataType::FP16);
}

GPTBlock::GPTBlock(int d_model, int n_q_heads, int n_kv_heads, int d_ff, float rope_theta)
    : d_model(d_model), n_q_heads(n_q_heads), n_kv_heads(n_kv_heads), d_ff(d_ff),
      head_dim(d_model / n_q_heads), rope_theta(rope_theta) {
    q_proj = std::make_shared<Linear>(d_model, n_q_heads * head_dim, false);
    k_proj = std::make_shared<Linear>(d_model, n_kv_heads * head_dim, false);
    v_proj = std::make_shared<Linear>(d_model, n_kv_heads * head_dim, false);
    o_proj = std::make_shared<Linear>(d_model, d_model, false);
    q_norm = std::make_shared<RMSNorm>(std::vector<int64_t>{head_dim});
    k_norm = std::make_shared<RMSNorm>(std::vector<int64_t>{head_dim});
    gate_proj = std::make_shared<Linear>(d_model, d_ff, false);
    up_proj = std::make_shared<Linear>(d_model, d_ff, false);
    down_proj = std::make_shared<Linear>(d_ff, d_model, false);
    attn_norm = std::make_shared<RMSNorm>(std::vector<int64_t>{d_model});
    ffn_norm = std::make_shared<RMSNorm>(std::vector<int64_t>{d_model});
}

std::shared_ptr<Tensor> GPTBlock::causal_mask(int64_t T, const Device& device) {
    auto it = mask_cache_.find(T);
    if (it != mask_cache_.end() && it->second->device == device) {
        return it->second;
    }
    std::vector<float> data(static_cast<size_t>(T * T));
    for (int64_t i = 0; i < T; ++i) {
        for (int64_t j = 0; j < T; ++j) {
            data[static_cast<size_t>(i * T + j)] = (j <= i) ? 0.0f : -1e9f;
        }
    }
    auto mask = Tensor::from_vector(data, {T, T}, device, false);
    mask_cache_[T] = mask;
    return mask;
}

void GPTBlock::ensure_rope_cache(int64_t T, int64_t D, const Device& device) {
    if (rope_cache_valid_ && rope_cos_cache_ && rope_cos_cache_->shape[0] == T &&
        rope_cos_cache_->shape[1] == D / 2 && rope_cache_device_ == device) {
        return;
    }
    int64_t half = D / 2;
    std::vector<float> cos_data(static_cast<size_t>(T * half));
    std::vector<float> sin_data(static_cast<size_t>(T * half));
    for (int64_t m = 0; m < T; ++m) {
        for (int64_t i = 0; i < half; ++i) {
            float freq = std::pow(rope_theta, -2.0f * static_cast<float>(i) / static_cast<float>(D));
            float angle = static_cast<float>(m) * freq;
            cos_data[static_cast<size_t>(m * half + i)] = std::cos(angle);
            sin_data[static_cast<size_t>(m * half + i)] = std::sin(angle);
        }
    }
    rope_cos_cache_ = Tensor::from_vector(cos_data, {T, half}, device, false);
    rope_sin_cache_ = Tensor::from_vector(sin_data, {T, half}, device, false);
    rope_cache_device_ = device;
    rope_cache_valid_ = true;
}

std::shared_ptr<Tensor> GPTBlock::forward(std::shared_ptr<Tensor> x) {
    int64_t B = x->shape[0];
    int64_t T = x->shape[1];

    auto h = attn_norm->forward(x);

    std::shared_ptr<Tensor> q, k, v;
    bool fused_qkv_ok = h->device.type == DeviceType::CPU &&
        q_proj->weight->dtype != DataType::INT8 &&
        k_proj->weight->dtype != DataType::INT8 &&
        v_proj->weight->dtype != DataType::INT8 &&
        !q_proj->bias && !k_proj->bias && !v_proj->bias &&
        !q_proj->scales && !k_proj->scales && !v_proj->scales;
    if (fused_qkv_ok) {
        auto qkv = Ops::fused_qkv_proj(h, q_proj->weight, k_proj->weight, v_proj->weight);
        q = std::get<0>(qkv);
        k = std::get<1>(qkv);
        v = std::get<2>(qkv);
    } else {
        q = q_proj->forward(h);
        k = k_proj->forward(h);
        v = v_proj->forward(h);
    }

    auto q_c = q->is_contiguous() ? q : q->contiguous();
    auto k_c = k->is_contiguous() ? k : k->contiguous();
    auto v_c = v->is_contiguous() ? v : v->contiguous();

    auto q_4d = q_c->view({B, T, n_q_heads, head_dim})->transpose(1, 2);
    auto k_4d = k_c->view({B, T, n_kv_heads, head_dim})->transpose(1, 2);
    auto v_4d = v_c->view({B, T, n_kv_heads, head_dim})->transpose(1, 2);

    auto q_4d_c = q_4d->is_contiguous() ? q_4d : q_4d->contiguous();
    auto k_4d_c = k_4d->is_contiguous() ? k_4d : k_4d->contiguous();
    q_4d = q_norm->forward(q_4d_c);
    k_4d = k_norm->forward(k_4d_c);

    ensure_rope_cache(T, head_dim, x->device);
    q_4d = Ops::rope(q_4d, rope_cos_cache_, rope_sin_cache_);
    k_4d = Ops::rope(k_4d, rope_cos_cache_, rope_sin_cache_);

    auto attn_4d = Ops::flash_attention(q_4d, k_4d, v_4d, true);
    auto attn_t = attn_4d->transpose(1, 2);
    auto attn_tc = attn_t->is_contiguous() ? attn_t : attn_t->contiguous();
    auto attn_flat = attn_tc->view({B, T, d_model});
    auto attn_out = o_proj->forward(attn_flat);

    auto x1 = Ops::add(x, attn_out);

    auto h2 = ffn_norm->forward(x1);
    auto gate = gate_proj->forward(h2);
    auto up = up_proj->forward(h2);
    std::shared_ptr<Tensor> swiglu;
    if (h2->device.type == DeviceType::CPU) {
        swiglu = Ops::fused_swiglu(gate, up);
    } else {
        auto silu_gate = Ops::mul(gate, Ops::sigmoid(gate));
        swiglu = Ops::mul(silu_gate, up);
    }
    auto ffn_out = down_proj->forward(swiglu);

    return Ops::add(x1, ffn_out);
}

std::vector<std::shared_ptr<Tensor>> GPTBlock::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    auto p1 = q_proj->parameters();
    auto p2 = k_proj->parameters();
    auto p3 = v_proj->parameters();
    auto p4 = o_proj->parameters();
    auto p5 = q_norm->parameters();
    auto p6 = k_norm->parameters();
    auto p7 = gate_proj->parameters();
    auto p8 = up_proj->parameters();
    auto p9 = down_proj->parameters();
    auto p10 = attn_norm->parameters();
    auto p11 = ffn_norm->parameters();
    params.insert(params.end(), p1.begin(), p1.end());
    params.insert(params.end(), p2.begin(), p2.end());
    params.insert(params.end(), p3.begin(), p3.end());
    params.insert(params.end(), p4.begin(), p4.end());
    params.insert(params.end(), p5.begin(), p5.end());
    params.insert(params.end(), p6.begin(), p6.end());
    params.insert(params.end(), p7.begin(), p7.end());
    params.insert(params.end(), p8.begin(), p8.end());
    params.insert(params.end(), p9.begin(), p9.end());
    params.insert(params.end(), p10.begin(), p10.end());
    params.insert(params.end(), p11.begin(), p11.end());
    return params;
}

void GPTBlock::to(const Device& device) {
    q_proj->to(device);
    k_proj->to(device);
    v_proj->to(device);
    o_proj->to(device);
    q_norm->to(device);
    k_norm->to(device);
    gate_proj->to(device);
    up_proj->to(device);
    down_proj->to(device);
    attn_norm->to(device);
    ffn_norm->to(device);
    mask_cache_.clear();
    rope_cache_valid_ = false;
}

}
}
