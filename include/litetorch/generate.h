#ifndef LITETORCH_GENERATE_H
#define LITETORCH_GENERATE_H

#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include <vector>
#include <string>
#include <functional>
#include <cstdint>

namespace litetorch {
namespace nn {

struct GenerateConfig {
    int64_t max_new_tokens = 32;
    bool do_sample = false;
    float temperature = 1.0f;
    int64_t top_k = 0;
    float top_p = 1.0f;
    int64_t num_beams = 1;
    float repetition_penalty = 1.0f;
    int64_t eos_token_id = -1;
    std::vector<int64_t> stop_token_ids;
    std::vector<std::vector<int64_t>> stop_sequences;
    uint64_t seed = 0;
};

std::vector<int64_t> generate(std::shared_ptr<Module> model,
                              const std::vector<int64_t>& prompt_tokens,
                              const GenerateConfig& config);

using TokenCallback = std::function<bool(int64_t token)>;
std::vector<int64_t> generate_stream(std::shared_ptr<Module> model,
                                     const std::vector<int64_t>& prompt_tokens,
                                     const GenerateConfig& config,
                                     TokenCallback on_token);

std::vector<float> apply_repetition_penalty(const std::vector<float>& logits,
                                           const std::vector<int64_t>& seen,
                                           float penalty);
std::vector<float> apply_top_k(const std::vector<float>& logits, int64_t k);
std::vector<float> apply_top_p(std::vector<float> logits, float p);
int64_t sample_token(const std::vector<float>& logits, float temperature, uint64_t& rng_state);

}
}

#endif
