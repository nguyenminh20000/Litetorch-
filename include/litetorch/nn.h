#ifndef LITETORCH_NN_H
#define LITETORCH_NN_H

#include "litetorch/tensor.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <typeindex>
#include <utility>

namespace litetorch {
namespace nn {

#include <functional>

class Module : public std::enable_shared_from_this<Module> {
public:
    bool training = true;
    bool is_compiled = false;
    bool is_graph_recorded = false;
    void* compiled_graph_exec = nullptr;
    std::vector<std::function<void(Module*, std::shared_ptr<Tensor>)>> forward_pre_hooks;
    std::vector<std::function<void(Module*, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>)>> forward_hooks;

    virtual ~Module() = default;
    virtual void compile() { is_compiled = true; }
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> parameters();
    virtual void to(const Device& device);
    virtual void half() {
        for (auto& child : children()) {
            if (child) child->half();
        }
    }
    virtual void train();
    virtual void eval();
    virtual std::vector<std::shared_ptr<Module>> children() { return {}; }
    virtual void save_pretrained(const std::string& path, int num_shards = 1);
    virtual void load_pretrained(const std::string& path);

    void register_parameter(const std::string& name, std::shared_ptr<Tensor> tensor);
    void register_buffer(const std::string& name, std::shared_ptr<Tensor> tensor);
    void add_module(const std::string& name, std::shared_ptr<Module> module);
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> named_parameters(const std::string& prefix = "");
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> named_buffers(const std::string& prefix = "");
    std::vector<std::shared_ptr<Tensor>> buffers();
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> named_children();
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> named_modules(const std::string& prefix = "");
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> state_dict();
    std::pair<std::vector<std::string>, std::vector<std::string>> load_state_dict(
        const std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>& state, bool strict = true);
    void zero_grad();
    void apply(const std::function<void(std::shared_ptr<Module>)>& fn);

    void register_forward_pre_hook(std::function<void(Module*, std::shared_ptr<Tensor>)> hook) {
        forward_pre_hooks.push_back(hook);
    }

    void register_forward_hook(std::function<void(Module*, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>)> hook) {
        forward_hooks.push_back(hook);
    }

    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input) {
        for (auto& hook : forward_pre_hooks) {
            hook(this, input);
        }
        auto output = forward(input);
        for (auto& hook : forward_hooks) {
            hook(this, input, output);
        }
        return output;
    }

protected:
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> named_params_;
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> named_buffers_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> named_children_;

private:
    struct LegacyLocal {
        std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> params;
        std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> buffers;
        std::vector<std::pair<std::string, std::shared_ptr<Module>>> children;
    };
    using LegacyFill = std::function<void(const Module*, LegacyLocal&)>;
    static const std::unordered_map<std::type_index, LegacyFill>& legacy_fill_map();
    LegacyLocal legacy_local() const;
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> local_params() const;
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> local_buffers() const;
    void state_dict_rec(const std::string& prefix,
                        std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>& out);
};

class Parameter : public Tensor {
public:
    Parameter(const std::vector<int64_t>& shape, const Device& device = Device(DeviceType::CPU, 0),
              DataType dtype = DataType::FP32);
};

class Buffer : public Tensor {
public:
    Buffer(const std::vector<int64_t>& shape, const Device& device = Device(DeviceType::CPU, 0),
           DataType dtype = DataType::FP32);
};

class Linear : public Module {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> scales = nullptr;

    Linear(int in_features, int out_features, bool has_bias = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    void half() override;
};

class QLoRALinear : public Module {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> lora_A;
    std::shared_ptr<Tensor> lora_B;
    float scaling;

    QLoRALinear(int in_features, int out_features, int r = 8, float lora_alpha = 16.0f, bool has_bias = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};class MoELinear : public Module {
public:
    std::shared_ptr<Tensor> gate_weight;
    std::vector<std::shared_ptr<Linear>> experts;
    int num_experts;
    int top_k;

    MoELinear(int in_features, int out_features, int num_experts, int top_k);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    std::vector<std::shared_ptr<Module>> children() override {
        std::vector<std::shared_ptr<Module>> c;
        for (auto& e : experts) c.push_back(e);
        return c;
    }
};

class ColumnParallelLinear : public Module {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    ColumnParallelLinear(int in_features, int out_features, bool has_bias = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class RowParallelLinear : public Module {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    RowParallelLinear(int in_features, int out_features, bool has_bias = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class Conv2d : public Module {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    int stride;
    int padding;
    int groups;
    std::vector<int> kernel_size;
    std::vector<int> stride_vec;
    std::vector<int> padding_vec;

    Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool has_bias = true, int groups = 1);
    Conv2d(int in_channels, int out_channels, const std::vector<int>& kernel_size,
           const std::vector<int>& stride, const std::vector<int>& padding, bool has_bias = true, int groups = 1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class Conv3d : public Module {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    int stride;
    int padding;
    int groups;

    Conv3d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool has_bias = true, int groups = 1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class Sequential : public Module {
public:
    std::vector<std::shared_ptr<Module>> modules;
    std::shared_ptr<Tensor> compiled_graph_input = nullptr;
    std::shared_ptr<Tensor> compiled_graph_output = nullptr;

    Sequential() = default;
    Sequential(const std::vector<std::shared_ptr<Module>>& modules);

    void add(std::shared_ptr<Module> module);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    void train() override;
    void eval() override;
    std::vector<std::shared_ptr<Module>> children() override { return modules; }
};

class ModuleList : public Module {
public:
    std::vector<std::shared_ptr<Module>> modules;

    ModuleList() = default;
    ModuleList(const std::vector<std::shared_ptr<Module>>& modules);

    void append(std::shared_ptr<Module> module);
    void extend(const std::vector<std::shared_ptr<Module>>& ms);
    std::shared_ptr<Module>& operator[](size_t i);
    const std::shared_ptr<Module>& operator[](size_t i) const;
    size_t size() const;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    void train() override;
    void eval() override;
    std::vector<std::shared_ptr<Module>> children() override { return modules; }
};

class ModuleDict : public Module {
public:
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> items;

    ModuleDict() = default;

    void put(const std::string& key, std::shared_ptr<Module> module);
    std::shared_ptr<Module> get(const std::string& key) const;
    bool contains(const std::string& key) const;
    void erase(const std::string& key);
    void clear();
    size_t size() const;
    std::vector<std::string> keys() const;
    std::shared_ptr<Module>& operator[](const std::string& key);
    const std::shared_ptr<Module>& operator[](const std::string& key) const;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    void train() override;
    void eval() override;
    std::vector<std::shared_ptr<Module>> children() override;
};

class LeakyReLU : public Module {
public:
    float negative_slope;

    LeakyReLU(float negative_slope = 0.01f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class MaxPool2d : public Module {
public:
    int kernel_size;
    int stride;
    int padding;

    MaxPool2d(int kernel_size, int stride = -1, int padding = 0);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class MaxPool3d : public Module {
public:
    int kernel_size;
    int stride;
    int padding;

    MaxPool3d(int kernel_size, int stride = -1, int padding = 0);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class BatchNorm2d : public Module {
public:
    int num_features;
    float eps;
    float momentum;
    bool affine;
    bool track_running_stats;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> running_mean;
    std::shared_ptr<Tensor> running_var;

    BatchNorm2d(int num_features, float eps = 1e-5f, float momentum = 0.1f,
                bool affine = true, bool track_running_stats = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class Dropout : public Module {
public:
    float p;

    Dropout(float p = 0.5f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Sigmoid : public Module {
public:
    Sigmoid() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Tanh : public Module {
public:
    Tanh() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class ReLU : public Module {
public:
    ReLU() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class MSELoss : public Module {
public:
    std::string reduction;

    MSELoss(const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class CrossEntropyLoss : public Module {
public:
    std::shared_ptr<Tensor> weight;
    int64_t ignore_index;
    std::string reduction;
    float label_smoothing;

    CrossEntropyLoss(std::shared_ptr<Tensor> weight = nullptr, int64_t ignore_index = -100,
                     const std::string& reduction = "mean", float label_smoothing = 0.0f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class BCELoss : public Module {
public:
    std::string reduction;

    BCELoss(const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class L1Loss : public Module {
public:
    std::string reduction;

    L1Loss(const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class Embedding : public Module {
public:
    int num_embeddings;
    int embedding_dim;
    int64_t padding_idx;
    std::shared_ptr<Tensor> weight;

    Embedding(int num_embeddings, int embedding_dim, int64_t padding_idx = -1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class Softmax : public Module {
public:
    int64_t dim;
    Softmax(int64_t dim = -1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class GELU : public Module {
public:
    GELU() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class LayerNorm : public Module {
public:
    std::vector<int64_t> normalized_shape;
    float eps;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    LayerNorm(const std::vector<int64_t>& normalized_shape, float eps = 1e-5f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class MultiHeadAttention : public Module {
public:
    int embed_dim;
    int num_heads;
    int head_dim;
    int kdim;
    int vdim;
    double dropout_p;
    bool use_bias;
    bool batch_first;

    std::shared_ptr<Linear> qkv_proj;
    std::shared_ptr<Linear> q_proj;
    std::shared_ptr<Linear> k_proj;
    std::shared_ptr<Linear> v_proj;
    std::shared_ptr<Linear> out_proj;
    std::shared_ptr<Dropout> attn_dropout;

    MultiHeadAttention(int embed_dim, int num_heads, double dropout = 0.0, bool bias = true,
                       int kdim = -1, int vdim = -1, bool batch_first = false);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> query, std::shared_ptr<Tensor> key, std::shared_ptr<Tensor> value);
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> forward(std::shared_ptr<Tensor> query, std::shared_ptr<Tensor> key,
        std::shared_ptr<Tensor> value, std::shared_ptr<Tensor> attn_mask, std::shared_ptr<Tensor> key_padding_mask, bool need_weights);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    void train() override;
    void eval() override;
    std::vector<std::shared_ptr<Module>> children() override {
        std::vector<std::shared_ptr<Module>> ch;
        if (qkv_proj) ch.push_back(qkv_proj);
        if (q_proj) ch.push_back(q_proj);
        if (k_proj) ch.push_back(k_proj);
        if (v_proj) ch.push_back(v_proj);
        if (out_proj) ch.push_back(out_proj);
        if (attn_dropout) ch.push_back(attn_dropout);
        return ch;
    }
};

class AdaptiveAvgPool2d : public Module {
public:
    int output_height;
    int output_width;

    AdaptiveAvgPool2d(int output_height, int output_width);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Flatten : public Module {
public:
    Flatten() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class TransformerDecoderLayer : public Module {
public:
    int embed_dim;
    int num_heads;
    int dim_feedforward;

    std::shared_ptr<MultiHeadAttention> self_attn;
    std::shared_ptr<MultiHeadAttention> multihead_attn;
    std::shared_ptr<Linear> linear1;
    std::shared_ptr<Linear> linear2;
    std::shared_ptr<LayerNorm> norm1;
    std::shared_ptr<LayerNorm> norm2;
    std::shared_ptr<LayerNorm> norm3;

    TransformerDecoderLayer(int embed_dim, int num_heads, int dim_feedforward = 2048);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> tgt, std::shared_ptr<Tensor> memory);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    std::vector<std::shared_ptr<Module>> children() override {
        return {self_attn, multihead_attn, linear1, linear2, norm1, norm2, norm3};
    }
};

class RMSNorm : public Module {
public:
    std::vector<int64_t> normalized_shape;
    float eps;
    std::shared_ptr<Tensor> weight;

    RMSNorm(const std::vector<int64_t>& normalized_shape, float eps = 1e-5f);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    void half() override;
};

class GPTBlock : public Module {
public:
    int d_model;
    int n_q_heads;
    int n_kv_heads;
    int d_ff;
    int head_dim;
    float rope_theta;

    std::shared_ptr<Linear> q_proj;
    std::shared_ptr<Linear> k_proj;
    std::shared_ptr<Linear> v_proj;
    std::shared_ptr<Linear> o_proj;
    std::shared_ptr<RMSNorm> q_norm;
    std::shared_ptr<RMSNorm> k_norm;
    std::shared_ptr<Linear> gate_proj;
    std::shared_ptr<Linear> up_proj;
    std::shared_ptr<Linear> down_proj;
    std::shared_ptr<RMSNorm> attn_norm;
    std::shared_ptr<RMSNorm> ffn_norm;

    GPTBlock(int d_model = 2560, int n_q_heads = 20, int n_kv_heads = 5, int d_ff = 6912, float rope_theta = 500000.0f);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    std::vector<std::shared_ptr<Module>> children() override {
        return {q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, gate_proj, up_proj, down_proj, attn_norm, ffn_norm};
    }

private:
    std::unordered_map<int64_t, std::shared_ptr<Tensor>> mask_cache_;
    std::shared_ptr<Tensor> rope_cos_cache_;
    std::shared_ptr<Tensor> rope_sin_cache_;
    Device rope_cache_device_;
    bool rope_cache_valid_ = false;
    std::shared_ptr<Tensor> causal_mask(int64_t T, const Device& device);
    void ensure_rope_cache(int64_t T, int64_t D, const Device& device);
};

class GPT2B : public Module {
public:
    int vocab_size;
    int d_model;
    int n_layers;
    int n_q_heads;
    int n_kv_heads;
    int d_ff;
    float rope_theta;
    bool tied = true;

    std::shared_ptr<Embedding> tok_emb;
    std::vector<std::shared_ptr<GPTBlock>> blocks;
    std::shared_ptr<RMSNorm> final_norm;
    std::shared_ptr<Linear> lm_head;

    int n_stages = 1;
    int stage_split = 0;
    Device stage_dev0{DeviceType::CPU, 0};
    Device stage_dev1{DeviceType::CPU, 0};

    GPT2B(int vocab_size = 50257, int d_model = 2560, int n_layers = 28,
          int n_q_heads = 20, int n_kv_heads = 5, int d_ff = 6912,
          float rope_theta = 500000.0f);

    void tie_weights();
    void pipeline_to(const Device& d0, const Device& d1, int split = -1);
    std::shared_ptr<Tensor> stage_forward(std::shared_ptr<Tensor> input, int stage);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    void half() override;
    std::vector<std::shared_ptr<Module>> children() override;
};

namespace init {
std::shared_ptr<Tensor> kaiming_uniform_(std::shared_ptr<Tensor> tensor, float a = 0.0f, const std::string& mode = "fan_in", const std::string& nonlinearity = "relu");
std::shared_ptr<Tensor> kaiming_normal_(std::shared_ptr<Tensor> tensor, float a = 0.0f, const std::string& mode = "fan_in", const std::string& nonlinearity = "relu");
std::shared_ptr<Tensor> xavier_uniform_(std::shared_ptr<Tensor> tensor, float gain = 1.0f);
std::shared_ptr<Tensor> xavier_normal_(std::shared_ptr<Tensor> tensor, float gain = 1.0f);
std::shared_ptr<Tensor> normal_(std::shared_ptr<Tensor> tensor, float mean = 0.0f, float std = 1.0f);
std::shared_ptr<Tensor> uniform_(std::shared_ptr<Tensor> tensor, float a = 0.0f, float b = 1.0f);
std::shared_ptr<Tensor> zeros_(std::shared_ptr<Tensor> tensor);
std::shared_ptr<Tensor> ones_(std::shared_ptr<Tensor> tensor);
std::shared_ptr<Tensor> fill_(std::shared_ptr<Tensor> tensor, float value);
// === W3-T4 ===
std::shared_ptr<Tensor> trunc_normal_(std::shared_ptr<Tensor> tensor, float mean = 0.0f, float std = 1.0f, float a = -2.0f, float b = 2.0f);
std::shared_ptr<Tensor> orthogonal_(std::shared_ptr<Tensor> tensor, float gain = 1.0f);
}

namespace utils {
void clip_grad_value_(const std::vector<std::shared_ptr<Tensor>>& parameters, float clip_value);
std::shared_ptr<Tensor> parameters_to_vector(const std::vector<std::shared_ptr<Tensor>>& parameters);
void vector_to_parameters(std::shared_ptr<Tensor> vec, const std::vector<std::shared_ptr<Tensor>>& parameters);
std::shared_ptr<Tensor> weight_norm(std::shared_ptr<Tensor> v, std::shared_ptr<Tensor> g, int64_t dim = 0);
std::shared_ptr<Tensor> spectral_norm(std::shared_ptr<Tensor> weight, int n_power_iterations = 1);
}
// === end W3-T4 ===

// === W1-T4 ===
class GroupNorm : public Module {
public:
    int num_groups;
    int num_channels;
    float eps;
    bool affine;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    GroupNorm(int num_groups, int num_channels, float eps = 1e-5f, bool affine = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class BatchNorm1d : public Module {
public:
    int num_features;
    float eps;
    float momentum;
    bool affine;
    bool track_running_stats;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> running_mean;
    std::shared_ptr<Tensor> running_var;

    BatchNorm1d(int num_features, float eps = 1e-5f, float momentum = 0.1f,
                bool affine = true, bool track_running_stats = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class Conv1d : public Module {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    int kernel_size;
    int stride;
    int padding;

    Conv1d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool has_bias = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class AvgPool2d : public Module {
public:
    int kernel_size;
    int stride;
    int padding;

    AvgPool2d(int kernel_size, int stride = -1, int padding = 0);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class NLLLoss : public Module {
public:
    std::shared_ptr<Tensor> weight;
    int64_t ignore_index;
    std::string reduction;

    NLLLoss(std::shared_ptr<Tensor> weight = nullptr, int64_t ignore_index = -100,
            const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class BCEWithLogitsLoss : public Module {
public:
    std::shared_ptr<Tensor> weight;
    std::string reduction;
    std::shared_ptr<Tensor> pos_weight;

    BCEWithLogitsLoss(std::shared_ptr<Tensor> weight = nullptr, const std::string& reduction = "mean",
                      std::shared_ptr<Tensor> pos_weight = nullptr);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class SiLU : public Module {
public:
    SiLU() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

// === W3-T4 ===
class KLDivLoss : public Module {
public:
    std::string reduction;

    KLDivLoss(const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class SmoothL1Loss : public Module {
public:
    float beta;
    std::string reduction;

    SmoothL1Loss(float beta = 1.0f, const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class HuberLoss : public Module {
public:
    float delta;
    std::string reduction;

    HuberLoss(float delta = 1.0f, const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class CTCLoss : public Module {
public:
    int blank;
    std::string reduction;
    bool zero_infinity;

    CTCLoss(int blank = 0, const std::string& reduction = "mean", bool zero_infinity = false);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> log_probs, std::shared_ptr<Tensor> targets,
                                    std::shared_ptr<Tensor> input_lengths, std::shared_ptr<Tensor> target_lengths);
};
// === end W3-T4 ===

class LogSoftmax : public Module {
public:
    int64_t dim;

    LogSoftmax(int64_t dim = -1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

// === W3-T3 recurrent+ ===

class RNNCell : public Module {
public:
    int input_size;
    int hidden_size;
    bool has_bias;
    std::string nonlinearity;
    std::shared_ptr<Tensor> weight_ih;
    std::shared_ptr<Tensor> weight_hh;
    std::shared_ptr<Tensor> bias_ih;
    std::shared_ptr<Tensor> bias_hh;

    RNNCell(int input_size, int hidden_size, bool bias = true, const std::string& nonlinearity = "tanh");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> hx);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class LSTMCell : public Module {
public:
    int input_size;
    int hidden_size;
    bool has_bias;
    std::shared_ptr<Tensor> weight_ih;
    std::shared_ptr<Tensor> weight_hh;
    std::shared_ptr<Tensor> bias_ih;
    std::shared_ptr<Tensor> bias_hh;

    LSTMCell(int input_size, int hidden_size, bool bias = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> forward(
        std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> hx, std::shared_ptr<Tensor> cx);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class GRUCell : public Module {
public:
    int input_size;
    int hidden_size;
    bool has_bias;
    std::shared_ptr<Tensor> weight_ih;
    std::shared_ptr<Tensor> weight_hh;
    std::shared_ptr<Tensor> bias_ih;
    std::shared_ptr<Tensor> bias_hh;

    GRUCell(int input_size, int hidden_size, bool bias = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> hx);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class RNN : public Module {
public:
    int input_size;
    int hidden_size;
    int num_layers;
    bool has_bias;
    bool batch_first;
    float dropout_p;
    bool bidirectional;
    std::string nonlinearity;
    std::vector<std::shared_ptr<RNNCell>> cells;
    std::shared_ptr<Dropout> dropout;

    RNN(int input_size, int hidden_size, int num_layers = 1, const std::string& nonlinearity = "tanh",
        bool bias = true, bool batch_first = false, float dropout = 0.0f, bool bidirectional = false);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> forward_full(
        std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> h0 = nullptr);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    void train() override;
    void eval() override;
    std::vector<std::shared_ptr<Module>> children() override;
};

class LSTM : public Module {
public:
    int input_size;
    int hidden_size;
    int num_layers;
    bool has_bias;
    bool batch_first;
    float dropout_p;
    bool bidirectional;
    std::vector<std::shared_ptr<LSTMCell>> cells;
    std::shared_ptr<Dropout> dropout;

    LSTM(int input_size, int hidden_size, int num_layers = 1, bool bias = true,
         bool batch_first = false, float dropout = 0.0f, bool bidirectional = false);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::pair<std::shared_ptr<Tensor>, std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> forward_full(
        std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> h0 = nullptr, std::shared_ptr<Tensor> c0 = nullptr);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    void train() override;
    void eval() override;
    std::vector<std::shared_ptr<Module>> children() override;
};

class GRU : public Module {
public:
    int input_size;
    int hidden_size;
    int num_layers;
    bool has_bias;
    bool batch_first;
    float dropout_p;
    bool bidirectional;
    std::vector<std::shared_ptr<GRUCell>> cells;
    std::shared_ptr<Dropout> dropout;

    GRU(int input_size, int hidden_size, int num_layers = 1, bool bias = true,
        bool batch_first = false, float dropout = 0.0f, bool bidirectional = false);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> forward_full(
        std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> h0 = nullptr);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    void train() override;
    void eval() override;
    std::vector<std::shared_ptr<Module>> children() override;
};

class TransformerEncoderLayer : public Module {
public:
    int d_model;
    int nhead;
    int dim_feedforward;
    float dropout_p;
    std::string activation;
    bool batch_first;
    std::shared_ptr<MultiHeadAttention> self_attn;
    std::shared_ptr<Linear> linear1;
    std::shared_ptr<Linear> linear2;
    std::shared_ptr<LayerNorm> norm1;
    std::shared_ptr<LayerNorm> norm2;
    std::shared_ptr<Dropout> dropout1;
    std::shared_ptr<Dropout> dropout2;
    std::shared_ptr<Dropout> dropout3;

    TransformerEncoderLayer(int d_model, int nhead, int dim_feedforward = 2048, float dropout = 0.1f,
                            const std::string& activation = "relu", bool batch_first = false);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> src, std::shared_ptr<Tensor> src_mask,
                                    std::shared_ptr<Tensor> src_key_padding_mask);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    void train() override;
    void eval() override;
    std::vector<std::shared_ptr<Module>> children() override;
};

class TransformerEncoder : public Module {
public:
    std::shared_ptr<ModuleList> layers;
    std::shared_ptr<LayerNorm> norm;

    TransformerEncoder(std::shared_ptr<TransformerEncoderLayer> encoder_layer, int num_layers,
                       std::shared_ptr<LayerNorm> norm = nullptr);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> src, std::shared_ptr<Tensor> mask,
                                    std::shared_ptr<Tensor> src_key_padding_mask);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    void train() override;
    void eval() override;
    std::vector<std::shared_ptr<Module>> children() override;
};

class EmbeddingBag : public Module {
public:
    int num_embeddings;
    int embedding_dim;
    std::string mode;
    std::shared_ptr<Tensor> weight;

    EmbeddingBag(int num_embeddings, int embedding_dim, const std::string& mode = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> offsets);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class ConvTranspose1d : public Module {
public:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int output_padding;
    int groups;
    int dilation;
    bool has_bias;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    ConvTranspose1d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0,
                    int output_padding = 0, int groups = 1, bool bias = true, int dilation = 1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class ConvTranspose2d : public Module {
public:
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride;
    int padding;
    int output_padding;
    int groups;
    int dilation;
    bool has_bias;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    ConvTranspose2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0,
                    int output_padding = 0, int groups = 1, bool bias = true, int dilation = 1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class ConvTranspose3d : public Module {
public:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int output_padding;
    int groups;
    int dilation;
    bool has_bias;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    ConvTranspose3d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0,
                    int output_padding = 0, int groups = 1, bool bias = true, int dilation = 1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class InstanceNorm1d : public Module {
public:
    int num_features;
    float eps;
    float momentum;
    bool affine;
    bool track_running_stats;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> running_mean;
    std::shared_ptr<Tensor> running_var;

    InstanceNorm1d(int num_features, float eps = 1e-5f, float momentum = 0.1f, bool affine = false,
                   bool track_running_stats = false);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class InstanceNorm2d : public Module {
public:
    int num_features;
    float eps;
    float momentum;
    bool affine;
    bool track_running_stats;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> running_mean;
    std::shared_ptr<Tensor> running_var;

    InstanceNorm2d(int num_features, float eps = 1e-5f, float momentum = 0.1f, bool affine = false,
                   bool track_running_stats = false);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class InstanceNorm3d : public Module {
public:
    int num_features;
    float eps;
    float momentum;
    bool affine;
    bool track_running_stats;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> running_mean;
    std::shared_ptr<Tensor> running_var;

    InstanceNorm3d(int num_features, float eps = 1e-5f, float momentum = 0.1f, bool affine = false,
                   bool track_running_stats = false);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class BatchNorm3d : public Module {
public:
    int num_features;
    float eps;
    float momentum;
    bool affine;
    bool track_running_stats;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> running_mean;
    std::shared_ptr<Tensor> running_var;

    BatchNorm3d(int num_features, float eps = 1e-5f, float momentum = 0.1f, bool affine = true,
                bool track_running_stats = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class SyncBatchNorm : public Module {
public:
    int num_features;
    float eps;
    float momentum;
    bool affine;
    bool track_running_stats;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> running_mean;
    std::shared_ptr<Tensor> running_var;

    SyncBatchNorm(int num_features, float eps = 1e-5f, float momentum = 0.1f, bool affine = true,
                  bool track_running_stats = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class ELU : public Module {
public:
    float alpha;

    ELU(float alpha = 1.0f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class PReLU : public Module {
public:
    int num_parameters;
    std::shared_ptr<Tensor> weight;

    PReLU(int num_parameters = 1, float init = 0.25f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class GLU : public Module {
public:
    int64_t dim;

    GLU(int64_t dim = -1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class PixelShuffle : public Module {
public:
    int upscale_factor;

    PixelShuffle(int upscale_factor);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class PixelUnshuffle : public Module {
public:
    int downscale_factor;

    PixelUnshuffle(int downscale_factor);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Identity : public Module {
public:
    Identity() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Unflatten : public Module {
public:
    int64_t dim;
    std::vector<int64_t> unflattened_size;

    Unflatten(int64_t dim, const std::vector<int64_t>& unflattened_size);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
// === end W3-T3 ===

// === DT-A2 (LT-ENG-019): tensor-parallel attention + vocab-parallel embedding ===
class TPAttention : public Module {
public:
    int embed_dim;
    int num_heads;
    int n_kv_heads;
    int head_dim;
    int tp_size;
    int tp_rank;
    int q_heads_per_rank;
    int kv_heads_per_rank;
    bool has_bias;

    std::shared_ptr<Tensor> w_q;
    std::shared_ptr<Tensor> b_q;
    std::shared_ptr<Tensor> w_k;
    std::shared_ptr<Tensor> b_k;
    std::shared_ptr<Tensor> w_v;
    std::shared_ptr<Tensor> b_v;
    std::shared_ptr<Tensor> w_o;
    std::shared_ptr<Tensor> b_o;

    TPAttention(int embed_dim, int num_heads, int n_kv_heads = -1, bool has_bias = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class VocabParallelEmbedding : public Module {
public:
    int vocab_size;
    int embed_dim;
    int64_t padding_idx;
    int tp_size;
    int tp_rank;
    int vocab_per_rank;
    int64_t vocab_start;
    int64_t vocab_end;

    std::shared_ptr<Tensor> weight;

    VocabParallelEmbedding(int vocab_size, int embed_dim, int64_t padding_idx = -1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> logits(std::shared_ptr<Tensor> hidden);
    std::shared_ptr<Tensor> sharded_cross_entropy(std::shared_ptr<Tensor> logits_shard,
                                                  std::shared_ptr<Tensor> targets);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};
// === end DT-A2 ===

// === W4-T3 (LT-ENG-028): extra activations ===
class ReLU6 : public Module {
public:
    ReLU6() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class RReLU : public Module {
public:
    float lower;
    float upper;
    RReLU(float lower = 0.125f, float upper = 0.3333333333333333f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class CELU : public Module {
public:
    float alpha;
    CELU(float alpha = 1.0f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class SELU : public Module {
public:
    SELU() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Mish : public Module {
public:
    Mish() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Softplus : public Module {
public:
    float beta;
    float threshold;
    Softplus(float beta = 1.0f, float threshold = 20.0f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Softshrink : public Module {
public:
    float lambd;
    Softshrink(float lambd = 0.5f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Hardshrink : public Module {
public:
    float lambd;
    Hardshrink(float lambd = 0.5f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Tanhshrink : public Module {
public:
    Tanhshrink() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Softsign : public Module {
public:
    Softsign() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Hardtanh : public Module {
public:
    float min_val;
    float max_val;
    Hardtanh(float min_val = -1.0f, float max_val = 1.0f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Threshold : public Module {
public:
    float threshold;
    float value;
    Threshold(float threshold, float value);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
// === end W4-T3 activations ===

// === W4-T3 (LT-ENG-028): extra losses ===
class HingeEmbeddingLoss : public Module {
public:
    float margin;
    std::string reduction;
    HingeEmbeddingLoss(float margin = 1.0f, const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class CosineEmbeddingLoss : public Module {
public:
    float margin;
    std::string reduction;
    CosineEmbeddingLoss(float margin = 0.0f, const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input1, std::shared_ptr<Tensor> input2,
                                    std::shared_ptr<Tensor> target);
};

class MarginRankingLoss : public Module {
public:
    float margin;
    std::string reduction;
    MarginRankingLoss(float margin = 0.0f, const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input1, std::shared_ptr<Tensor> input2,
                                    std::shared_ptr<Tensor> target);
};

class MultiMarginLoss : public Module {
public:
    int p;
    float margin;
    std::string reduction;
    MultiMarginLoss(int p = 1, float margin = 1.0f, const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class MultiLabelMarginLoss : public Module {
public:
    std::string reduction;
    MultiLabelMarginLoss(const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class SoftMarginLoss : public Module {
public:
    std::string reduction;
    SoftMarginLoss(const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class MultiLabelSoftMarginLoss : public Module {
public:
    std::string reduction;
    MultiLabelSoftMarginLoss(const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class TripletMarginLoss : public Module {
public:
    float margin;
    int p;
    float eps;
    bool swap;
    std::string reduction;
    TripletMarginLoss(float margin = 1.0f, int p = 2, float eps = 1e-6f, bool swap = false,
                      const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> anchor, std::shared_ptr<Tensor> positive,
                                    std::shared_ptr<Tensor> negative);
};

class TripletMarginWithDistanceLoss : public Module {
public:
    float margin;
    std::string reduction;
    TripletMarginWithDistanceLoss(float margin = 1.0f, const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> anchor, std::shared_ptr<Tensor> positive,
                                    std::shared_ptr<Tensor> negative);
};

class PoissonNLLLoss : public Module {
public:
    bool log_input;
    bool full;
    float eps;
    std::string reduction;
    PoissonNLLLoss(bool log_input = true, bool full = false, float eps = 1e-8f,
                   const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class GaussianNLLLoss : public Module {
public:
    bool full;
    float eps;
    std::string reduction;
    GaussianNLLLoss(bool full = false, float eps = 1e-6f, const std::string& reduction = "mean");
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
                                    std::shared_ptr<Tensor> var);
};
// === end W4-T3 losses ===

// === W4-T3 (LT-ENG-028): Bilinear, Fold, Unfold ===
class Bilinear : public Module {
public:
    int in1_features;
    int in2_features;
    int out_features;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    Bilinear(int in1_features, int in2_features, int out_features, bool bias = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input1, std::shared_ptr<Tensor> input2);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class Fold : public Module {
public:
    std::vector<int64_t> output_size;
    std::vector<int64_t> kernel_size;
    std::vector<int64_t> dilation;
    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
    Fold(const std::vector<int64_t>& output_size, const std::vector<int64_t>& kernel_size,
         const std::vector<int64_t>& dilation = {1, 1}, const std::vector<int64_t>& padding = {0, 0},
         const std::vector<int64_t>& stride = {1, 1});
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Unfold : public Module {
public:
    std::vector<int64_t> kernel_size;
    std::vector<int64_t> dilation;
    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
    Unfold(const std::vector<int64_t>& kernel_size,
           const std::vector<int64_t>& dilation = {1, 1}, const std::vector<int64_t>& padding = {0, 0},
           const std::vector<int64_t>& stride = {1, 1});
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
// === end W4-T3 misc ===

}

}

#endif
