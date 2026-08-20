#include "litetorch/nn.h"
#include "litetorch/ops.h"

namespace litetorch {
namespace nn {

TransformerDecoderLayer::TransformerDecoderLayer(int embed_dim, int num_heads, int dim_feedforward)
    : embed_dim(embed_dim), num_heads(num_heads), dim_feedforward(dim_feedforward) {
    self_attn = std::make_shared<MultiHeadAttention>(embed_dim, num_heads);
    multihead_attn = std::make_shared<MultiHeadAttention>(embed_dim, num_heads);
    linear1 = std::make_shared<Linear>(embed_dim, dim_feedforward);
    linear2 = std::make_shared<Linear>(dim_feedforward, embed_dim);
    norm1 = std::make_shared<LayerNorm>(std::vector<int64_t>{embed_dim});
    norm2 = std::make_shared<LayerNorm>(std::vector<int64_t>{embed_dim});
    norm3 = std::make_shared<LayerNorm>(std::vector<int64_t>{embed_dim});
}

std::shared_ptr<Tensor> TransformerDecoderLayer::forward(std::shared_ptr<Tensor> input) {
    return forward(input, nullptr);
}

std::shared_ptr<Tensor> TransformerDecoderLayer::forward(std::shared_ptr<Tensor> tgt, std::shared_ptr<Tensor> memory) {
    auto tgt2 = self_attn->forward(tgt);
    auto x1 = Ops::fused_add_layernorm(tgt, tgt2, norm1->normalized_shape, norm1->weight, norm1->bias, norm1->eps);

    std::shared_ptr<Tensor> x2;
    if (memory != nullptr) {
        auto tgt3 = multihead_attn->forward(x1, memory, memory);
        x2 = Ops::fused_add_layernorm(x1, tgt3, norm2->normalized_shape, norm2->weight, norm2->bias, norm2->eps);
    } else {
        x2 = x1;
    }

    auto ffn1 = linear1->forward(x2);
    auto act = Ops::gelu(ffn1);
    auto ffn2 = linear2->forward(act);
    auto out = Ops::fused_add_layernorm(x2, ffn2, norm3->normalized_shape, norm3->weight, norm3->bias, norm3->eps);

    return out;
}

std::vector<std::shared_ptr<Tensor>> TransformerDecoderLayer::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    auto p1 = self_attn->parameters();
    auto p2 = multihead_attn->parameters();
    auto p3 = linear1->parameters();
    auto p4 = linear2->parameters();
    auto p5 = norm1->parameters();
    auto p6 = norm2->parameters();
    auto p7 = norm3->parameters();
    params.insert(params.end(), p1.begin(), p1.end());
    params.insert(params.end(), p2.begin(), p2.end());
    params.insert(params.end(), p3.begin(), p3.end());
    params.insert(params.end(), p4.begin(), p4.end());
    params.insert(params.end(), p5.begin(), p5.end());
    params.insert(params.end(), p6.begin(), p6.end());
    params.insert(params.end(), p7.begin(), p7.end());
    return params;
}

void TransformerDecoderLayer::to(const Device& device) {
    self_attn->to(device);
    multihead_attn->to(device);
    linear1->to(device);
    linear2->to(device);
    norm1->to(device);
    norm2->to(device);
    norm3->to(device);
}

}
}
