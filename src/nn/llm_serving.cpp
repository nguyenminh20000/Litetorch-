#include "litetorch/llm_serving.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <sstream>
#include <algorithm>
#include <cmath>

namespace litetorch {
namespace nn {

PrefixCache& PrefixCache::get() {
    static PrefixCache instance;
    return instance;
}

std::string PrefixCache::hash_prefix(const std::vector<int64_t>& prefix) {
    std::stringstream ss;
    for (size_t i = 0; i < prefix.size(); ++i) {
        ss << prefix[i];
        if (i + 1 < prefix.size()) ss << ",";
    }
    return ss.str();
}

void PrefixCache::put(const std::vector<int64_t>& prefix, const std::vector<std::shared_ptr<Tensor>>& kv_cache) {
    cache_[hash_prefix(prefix)] = kv_cache;
}

std::vector<std::shared_ptr<Tensor>> PrefixCache::get_cache(const std::vector<int64_t>& prefix) {
    std::string key = hash_prefix(prefix);
    if (cache_.find(key) != cache_.end()) {
        return cache_[key];
    }
    return {};
}

void PrefixCache::clear() {
    cache_.clear();
}

SpeculativeEngine::SpeculativeEngine(std::shared_ptr<Module> draft, std::shared_ptr<Module> target)
    : draft_model(draft), target_model(target) {}

std::vector<int64_t> SpeculativeEngine::generate(const std::vector<int64_t>& prompt, int max_new_tokens, int lookahead) {
    std::vector<int64_t> result = prompt;
    int generated = 0;

    while (generated < max_new_tokens) {
        std::vector<int64_t> draft_tokens;
        std::vector<int64_t> current_seq = result;

        for (int k = 0; k < lookahead; ++k) {
            std::vector<float> input_vec(current_seq.size());
            for (size_t idx = 0; idx < current_seq.size(); ++idx) {
                input_vec[idx] = static_cast<float>(current_seq[idx]);
            }
            auto input_tensor = Tensor::from_vector(input_vec, {1, static_cast<int64_t>(current_seq.size())});
            
            auto logits = (*draft_model)(input_tensor);
            float* logits_ptr = logits->data_ptr();
            int64_t vocab_size = logits->shape.back();
            int64_t seq_len = logits->shape[1];

            float max_val = -1e9f;
            int64_t next_token = 0;
            int64_t last_step_offset = (seq_len - 1) * vocab_size;

            for (int64_t v = 0; v < vocab_size; ++v) {
                if (logits_ptr[last_step_offset + v] > max_val) {
                    max_val = logits_ptr[last_step_offset + v];
                    next_token = v;
                }
            }

            draft_tokens.push_back(next_token);
            current_seq.push_back(next_token);
        }

        std::vector<float> target_input_vec(current_seq.size());
        for (size_t idx = 0; idx < current_seq.size(); ++idx) {
            target_input_vec[idx] = static_cast<float>(current_seq[idx]);
        }
        auto target_input_tensor = Tensor::from_vector(target_input_vec, {1, static_cast<int64_t>(current_seq.size())});

        auto target_logits = (*target_model)(target_input_tensor);
        float* t_logits_ptr = target_logits->data_ptr();
        int64_t vocab_size = target_logits->shape.back();

        int accept_count = 0;
        std::vector<int64_t> target_predictions;

        int64_t base_len = result.size();
        for (int k = 0; k < lookahead; ++k) {
            int64_t step_idx = base_len - 1 + k;
            int64_t step_offset = step_idx * vocab_size;

            float max_val = -1e9f;
            int64_t t_token = 0;
            for (int64_t v = 0; v < vocab_size; ++v) {
                if (t_logits_ptr[step_offset + v] > max_val) {
                    max_val = t_logits_ptr[step_offset + v];
                    t_token = v;
                }
            }
            target_predictions.push_back(t_token);

            if (draft_tokens[k] == t_token) {
                accept_count++;
            } else {
                break;
            }
        }

        for (int a = 0; a < accept_count; ++a) {
            result.push_back(draft_tokens[a]);
            generated++;
        }

        if (generated >= max_new_tokens) {
            break;
        }

        int64_t last_validated_step = base_len - 1 + accept_count;
        int64_t step_offset = last_validated_step * vocab_size;
        float max_val = -1e9f;
        int64_t next_t_token = 0;
        for (int64_t v = 0; v < vocab_size; ++v) {
            if (t_logits_ptr[step_offset + v] > max_val) {
                max_val = t_logits_ptr[step_offset + v];
                next_t_token = v;
            }
        }

        result.push_back(next_t_token);
        generated++;
    }

    return result;
}

}
}
