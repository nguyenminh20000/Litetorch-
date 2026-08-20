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

    q_proj = std::make_shared<Linear>(embed_dim, embed_dim);
    k_proj = std::make_shared<Linear>(embed_dim, embed_dim);
    v_proj = std::make_shared<Linear>(embed_dim, embed_dim);
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

    auto q = q_proj->forward(query);
    auto k = k_proj->forward(key);
    auto v = v_proj->forward(value);

    auto q_c = q->is_contiguous() ? q : q->contiguous();
    auto k_c = k->is_contiguous() ? k : k->contiguous();
    auto v_c = v->is_contiguous() ? v : v->contiguous();

    auto q_reshaped = q_c->view({B, Tq, num_heads, head_dim});
    auto k_reshaped = k_c->view({B, Tk, num_heads, head_dim});
    auto v_reshaped = v_c->view({B, Tv, num_heads, head_dim});

    auto q_transposed = q_reshaped->transpose(1, 2);
    auto k_transposed = k_reshaped->transpose(1, 2);
    auto v_transposed = v_reshaped->transpose(1, 2);

    auto out_4d = Ops::flash_attention(q_transposed, k_transposed, v_transposed);
    auto out_transposed = out_4d->transpose(1, 2);
    auto out_contiguous = out_transposed->is_contiguous() ? out_transposed : out_transposed->contiguous();
    auto out_flat = out_contiguous->view({B, Tq, embed_dim});

    return out_proj->forward(out_flat);
}

std::vector<std::shared_ptr<Tensor>> MultiHeadAttention::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    auto q_params = q_proj->parameters();
    auto k_params = k_proj->parameters();
    auto v_params = v_proj->parameters();
    auto out_params = out_proj->parameters();
    params.insert(params.end(), q_params.begin(), q_params.end());
    params.insert(params.end(), k_params.begin(), k_params.end());
    params.insert(params.end(), v_params.begin(), v_params.end());
    params.insert(params.end(), out_params.begin(), out_params.end());
    return params;
}

void MultiHeadAttention::to(const Device& device) {
    q_proj->to(device);
    k_proj->to(device);
    v_proj->to(device);
    out_proj->to(device);
}

}
}
