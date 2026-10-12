#include "litetorch/llm_serving.h"
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <sstream>
#include <algorithm>
#include <cmath>

namespace litetorch {
namespace nn {

namespace {
Device serving_model_device(std::shared_ptr<Module> m) {
    auto ps = m->parameters();
    if (!ps.empty() && ps[0]) return ps[0]->device;
    return Device(DeviceType::CPU, 0);
}

std::shared_ptr<Tensor> tokens_to_tensor(const std::vector<int64_t>& seq, const Device& device) {
    std::vector<float> v(seq.size());
    for (size_t i = 0; i < seq.size(); ++i) v[i] = static_cast<float>(seq[i]);
    return Tensor::from_vector(v, {1, static_cast<int64_t>(seq.size())}, device);
}

int64_t argmax_at_step(const std::shared_ptr<Tensor>& logits, int64_t step) {
    auto l = logits->dtype == DataType::FP32 ? logits : logits->cast(DataType::FP32);
    float* p = l->data_ptr();
    int64_t vocab = l->shape.back();
    int64_t off = step * vocab;
    float best = -1e30f;
    int64_t idx = 0;
    for (int64_t v = 0; v < vocab; ++v) {
        if (p[off + v] > best) { best = p[off + v]; idx = v; }
    }
    return idx;
}
}

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
    Device draft_dev = serving_model_device(draft_model);
    Device target_dev = serving_model_device(target_model);

    while (generated < max_new_tokens) {
        std::vector<int64_t> draft_tokens;
        std::vector<int64_t> current_seq = result;

        for (int k = 0; k < lookahead; ++k) {
            auto input_tensor = tokens_to_tensor(current_seq, draft_dev);
            auto logits = (*draft_model)(input_tensor);
            int64_t seq_len = logits->shape[logits->shape.size() - 2];
            int64_t next_token = argmax_at_step(logits, seq_len - 1);
            draft_tokens.push_back(next_token);
            current_seq.push_back(next_token);
        }

        auto target_input_tensor = tokens_to_tensor(current_seq, target_dev);
        auto target_logits = (*target_model)(target_input_tensor);
        int64_t t_seq = target_logits->shape[target_logits->shape.size() - 2];

        int accept_count = 0;
        int64_t base_len = static_cast<int64_t>(result.size());
        for (int k = 0; k < lookahead; ++k) {
            int64_t step_idx = base_len - 1 + k;
            if (step_idx < 0 || step_idx >= t_seq) break;
            int64_t t_token = argmax_at_step(target_logits, step_idx);
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

        int64_t last_step = base_len - 1 + accept_count;
        if (last_step < 0) last_step = 0;
        if (last_step >= t_seq) last_step = t_seq - 1;
        int64_t next_t_token = argmax_at_step(target_logits, last_step);
        result.push_back(next_t_token);
        generated++;
    }

    return result;
}

}
}
