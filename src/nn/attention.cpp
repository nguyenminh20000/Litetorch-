#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <cmath>
#include <stdexcept>

namespace litetorch {
namespace nn {

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads)
    : embed_dim(embed_dim), num_heads(num_heads) {
    if (embed_dim % num_heads != 0) {
        throw std::runtime_error("embed_dim must be divisible by num_heads");
    }
    head_dim = embed_dim / num_heads;

    qkv_proj = std::make_shared<Linear>(embed_dim, 3 * embed_dim);
    out_proj = std::make_shared<Linear>(embed_dim, embed_dim);
}

std::shared_ptr<Tensor> MultiHeadAttention::forward(std::shared_ptr<Tensor> input) {
    return forward(input, input, input);
}

std::shared_ptr<Tensor> MultiHeadAttention::forward(std::shared_ptr<Tensor> query, std::shared_ptr<Tensor> key, std::shared_ptr<Tensor> value) {
    if (query->shape.size() != 3 || key->shape.size() != 3 || value->shape.size() != 3) {
        throw std::runtime_error("MultiHeadAttention inputs must be 3D tensors of shape (B, T, C)");
    }
    int64_t B = query->shape[0];
    int64_t Tq = query->shape[1];
    int64_t Cq = query->shape[2];

    int64_t Tk = key->shape[1];
    int64_t Ck = key->shape[2];

    int64_t Tv = value->shape[1];
    int64_t Cv = value->shape[2];

    if (Cq != embed_dim || Ck != embed_dim || Cv != embed_dim) {
        throw std::runtime_error("Input channel dimensions do not match embed_dim");
    }
    if (key->shape[0] != B || value->shape[0] != B) {
        throw std::runtime_error("Batch dimensions of query, key, and value must match");
    }
    if (Tk != Tv) {
        throw std::runtime_error("Sequence length of key and value must match");
    }

    std::shared_ptr<Tensor> out_4d;
    if (query == key && key == value) {
        auto qkv = qkv_proj->forward(query);
        auto qkv_c = qkv->is_contiguous() ? qkv : qkv->contiguous();
        out_4d = Ops::flash_attention_qkv(qkv_c, num_heads);
    } else {
        auto qkv_q = qkv_proj->forward(query);
        auto qkv_k = qkv_proj->forward(key);
        auto qkv_v = qkv_proj->forward(value);
        auto q = Ops::qkv_extract(qkv_q->is_contiguous() ? qkv_q : qkv_q->contiguous(), num_heads, 0);
        auto k = Ops::qkv_extract(qkv_k->is_contiguous() ? qkv_k : qkv_k->contiguous(), num_heads, 1);
        auto v = Ops::qkv_extract(qkv_v->is_contiguous() ? qkv_v : qkv_v->contiguous(), num_heads, 2);
        out_4d = Ops::flash_attention(q, k, v);
    }
    auto out_transposed = out_4d->transpose(1, 2);
    auto out_contiguous = out_transposed->is_contiguous() ? out_transposed : out_transposed->contiguous();
    auto out_flat = out_contiguous->view({B, Tq, embed_dim});

    return out_proj->forward(out_flat);
}

std::vector<std::shared_ptr<Tensor>> MultiHeadAttention::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    auto qkv_params = qkv_proj->parameters();
    auto out_params = out_proj->parameters();
    params.insert(params.end(), qkv_params.begin(), qkv_params.end());
    params.insert(params.end(), out_params.begin(), out_params.end());
    return params;
}

void MultiHeadAttention::to(const Device& device) {
    qkv_proj->to(device);
    out_proj->to(device);
}

}
}
