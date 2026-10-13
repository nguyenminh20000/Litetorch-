#ifndef LITETORCH_HF_H
#define LITETORCH_HF_H

#include "litetorch/tensor.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace litetorch {
namespace hf {

struct ModelConfig {
    std::string model_type;
    int64_t hidden_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t intermediate_size = 0;
    int64_t vocab_size = 0;
    int64_t max_position_embeddings = 0;
    std::string hidden_act;
    double rms_norm_eps = 1e-6;
    double layer_norm_eps = 1e-5;
    int64_t n_layer = 0;
    int64_t n_head = 0;
    int64_t n_embd = 0;
    int64_t n_positions = 0;
};

struct PretrainedBundle {
    ModelConfig config;
    std::map<std::string, std::shared_ptr<Tensor>> state_dict;
};

ModelConfig load_config(const std::string& repo_or_dir, const std::string& revision = "main");
PretrainedBundle from_pretrained(const std::string& repo_id, const std::string& revision = "main");
std::map<std::string, std::string> weight_name_map(const std::string& model_type);
std::map<std::string, std::shared_ptr<Tensor>> remap_state_dict(
    const std::map<std::string, std::shared_ptr<Tensor>>& sd, const std::string& model_type);

}
}

#endif
