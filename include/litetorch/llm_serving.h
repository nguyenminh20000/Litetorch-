#ifndef LITETORCH_LLM_SERVING_H
#define LITETORCH_LLM_SERVING_H

#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

namespace litetorch {
namespace nn {

class PrefixCache {
public:
    static PrefixCache& get();

    void put(const std::vector<int64_t>& prefix, const std::vector<std::shared_ptr<Tensor>>& kv_cache);
    std::vector<std::shared_ptr<Tensor>> get_cache(const std::vector<int64_t>& prefix);
    void clear();

private:
    PrefixCache() = default;
    std::unordered_map<std::string, std::vector<std::shared_ptr<Tensor>>> cache_;
    std::string hash_prefix(const std::vector<int64_t>& prefix);
};

class SpeculativeEngine {
public:
    std::shared_ptr<Module> draft_model;
    std::shared_ptr<Module> target_model;

    SpeculativeEngine(std::shared_ptr<Module> draft, std::shared_ptr<Module> target);

    std::vector<int64_t> generate(const std::vector<int64_t>& prompt, int max_new_tokens, int lookahead = 4);
};

class MedusaHead : public Module {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    MedusaHead(int in_features, int vocab_size);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class MedusaEngine {
public:
    std::shared_ptr<Module> base_model;
    std::vector<std::shared_ptr<MedusaHead>> medusa_heads;

    MedusaEngine(std::shared_ptr<Module> base, const std::vector<std::shared_ptr<MedusaHead>>& heads);

    std::vector<int64_t> generate(const std::vector<int64_t>& prompt, int max_new_tokens, int lookahead = 2);
};

}
}

#endif
