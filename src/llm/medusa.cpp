#include "litetorch/llm_serving.h"
#include "litetorch/ops.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace litetorch {
namespace nn {

MedusaHead::MedusaHead(int in_features, int vocab_size) {
    weight = Tensor::create({vocab_size, in_features}, Device(DeviceType::CPU), true);
    bias = Tensor::create({vocab_size}, Device(DeviceType::CPU), true);
    
    float scale = 1.0f / std::sqrt(static_cast<float>(in_features));
    float* w_ptr = weight->data_ptr();
    float* b_ptr = bias->data_ptr();
    for (size_t i = 0; i < weight->numel(); ++i) {
        w_ptr[i] = ((static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f) * scale;
    }
    for (size_t i = 0; i < bias->numel(); ++i) {
        b_ptr[i] = 0.0f;
    }
}

std::shared_ptr<Tensor> MedusaHead::forward(std::shared_ptr<Tensor> input) {
    auto out = Ops::matmul(input, weight->transpose(0, 1));
    return Ops::add(out, bias);
}

std::vector<std::shared_ptr<Tensor>> MedusaHead::parameters() {
    return {weight, bias};
}

void MedusaHead::to(const Device& device) {
    weight = weight->to(device);
    bias = bias->to(device);
}

MedusaEngine::MedusaEngine(std::shared_ptr<Module> base, const std::vector<std::shared_ptr<MedusaHead>>& heads)
    : base_model(base), medusa_heads(heads) {}

std::vector<int64_t> MedusaEngine::generate(const std::vector<int64_t>& prompt, int max_new_tokens, int lookahead) {
    if (medusa_heads.empty()) {
        throw std::runtime_error("[litetorch Error] MedusaEngine: no medusa heads");
    }
    size_t n_heads = std::min(static_cast<size_t>(lookahead > 0 ? lookahead : 0), medusa_heads.size());
    std::vector<int64_t> tokens = prompt;
    int vocab_size = medusa_heads[0]->weight->shape[0];

    while (tokens.size() - prompt.size() < static_cast<size_t>(max_new_tokens)) {
        int64_t seq_len = tokens.size();
        auto input = Tensor::create({1, seq_len}, Device(DeviceType::CPU));
        float* in_ptr = input->data_ptr();
        for (int64_t i = 0; i < seq_len; ++i) {
            in_ptr[i] = static_cast<float>(tokens[i]);
        }

        auto logits = base_model->forward(input);
        std::vector<float> logits_vec = logits->to_vector();

        float max_logit = -1e9f;
        int next_token = 0;
        for (int v = 0; v < vocab_size; ++v) {
            float logit = logits_vec[(seq_len - 1) * vocab_size + v];
            if (logit > max_logit) {
                max_logit = logit;
                next_token = v;
            }
        }

        tokens.push_back(next_token);

        std::vector<int> candidates;
        for (size_t h = 0; h < n_heads; ++h) {
            auto head_out = medusa_heads[h]->forward(logits);
            std::vector<float> ho_vec = head_out->to_vector();

            float max_ho = -1e9f;
            int cand = 0;
            for (int v = 0; v < vocab_size; ++v) {
                float logit = ho_vec[(seq_len - 1) * vocab_size + v];
                if (logit > max_ho) {
                    max_ho = logit;
                    cand = v;
                }
            }
            candidates.push_back(cand);
        }

        int64_t T = tokens.size();
        std::vector<int64_t> test_tokens = tokens;
        for (int cand : candidates) {
            test_tokens.push_back(cand);
        }

        auto cand_input = Tensor::create({1, static_cast<int64_t>(test_tokens.size())}, Device(DeviceType::CPU));
        float* ci_ptr = cand_input->data_ptr();
        for (size_t i = 0; i < test_tokens.size(); ++i) {
            ci_ptr[i] = static_cast<float>(test_tokens[i]);
        }

        auto cand_logits = base_model->forward(cand_input);
        std::vector<float> cl_vec = cand_logits->to_vector();

        for (size_t h = 0; h < candidates.size(); ++h) {
            int64_t step_idx = T - 1 + h;
            float max_cl = -1e9f;
            int act_token = 0;
            for (int v = 0; v < vocab_size; ++v) {
                float logit = cl_vec[step_idx * vocab_size + v];
                if (logit > max_cl) {
                    max_cl = logit;
                    act_token = v;
                }
            }

            if (act_token == candidates[h]) {
                tokens.push_back(candidates[h]);
            } else {
                break;
            }
        }

        if (next_token == 2) break;
    }

    return tokens;
}

}
}
