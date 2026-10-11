#include "litetorch/nn.h"
#include "litetorch/ops.h"

namespace litetorch {
namespace nn {

GPT2B::GPT2B(int vocab_size, int d_model, int n_layers,
             int n_q_heads, int n_kv_heads, int d_ff, float rope_theta)
    : vocab_size(vocab_size), d_model(d_model), n_layers(n_layers),
      n_q_heads(n_q_heads), n_kv_heads(n_kv_heads), d_ff(d_ff), rope_theta(rope_theta) {
    tok_emb = std::make_shared<Embedding>(vocab_size, d_model);
    blocks.reserve(static_cast<size_t>(n_layers));
    for (int i = 0; i < n_layers; ++i) {
        blocks.push_back(std::make_shared<GPTBlock>(d_model, n_q_heads, n_kv_heads, d_ff, rope_theta));
    }
    final_norm = std::make_shared<RMSNorm>(std::vector<int64_t>{d_model});
    lm_head = std::make_shared<Linear>(d_model, vocab_size, false);
    tie_weights();
}

void GPT2B::tie_weights() {
    lm_head->weight = tok_emb->weight;
    tied = true;
}

void GPT2B::pipeline_to(const Device& d0, const Device& d1, int split) {
    if (split < 0) split = n_layers / 2;
    stage_split = split;
    stage_dev0 = d0;
    stage_dev1 = d1;
    tok_emb->to(d0);
    for (int i = 0; i < n_layers; ++i) {
        blocks[i]->to(i < split ? d0 : d1);
    }
    final_norm->to(d1);
    lm_head->weight = lm_head->weight->to(d1);
    tied = false;
    n_stages = 2;
}

std::shared_ptr<Tensor> GPT2B::stage_forward(std::shared_ptr<Tensor> input, int stage) {
    if (n_stages == 1) return forward(input);
    if (stage == 0) {
        auto h = tok_emb->forward(input);
        for (int i = 0; i < stage_split; ++i) h = blocks[i]->forward(h);
        return h;
    }
    auto h = input;
    for (int i = stage_split; i < n_layers; ++i) h = blocks[i]->forward(h);
    h = final_norm->forward(h);
    return lm_head->forward(h);
}

std::shared_ptr<Tensor> GPT2B::forward(std::shared_ptr<Tensor> input) {
    if (n_stages == 2) {
        auto h = stage_forward(input, 0);
        h = h->to(stage_dev1);
        return stage_forward(h, 1);
    }
    auto h = tok_emb->forward(input);
    for (auto& b : blocks) h = b->forward(h);
    h = final_norm->forward(h);
    return lm_head->forward(h);
}

std::vector<std::shared_ptr<Tensor>> GPT2B::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    params.push_back(tok_emb->weight);
    for (auto& b : blocks) {
        auto bp = b->parameters();
        params.insert(params.end(), bp.begin(), bp.end());
    }
    params.push_back(final_norm->weight);
    if (lm_head->weight.get() != tok_emb->weight.get()) {
        params.push_back(lm_head->weight);
    }
    return params;
}

void GPT2B::to(const Device& device) {
    tok_emb->to(device);
    for (auto& b : blocks) b->to(device);
    final_norm->to(device);
    if (tied) {
        lm_head->weight = tok_emb->weight;
    } else {
        lm_head->to(device);
    }
    n_stages = 1;
}

std::vector<std::shared_ptr<Module>> GPT2B::children() {
    std::vector<std::shared_ptr<Module>> c;
    c.push_back(tok_emb);
    for (auto& b : blocks) c.push_back(b);
    c.push_back(final_norm);
    c.push_back(lm_head);
    return c;
}

}
}
