#ifndef LITETORCH_NN_H
#define LITETORCH_NN_H

#include "litetorch/tensor.h"
#include <vector>
#include <memory>

namespace litetorch {
namespace nn {

#include <functional>

class Module {
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
    virtual std::vector<std::shared_ptr<Tensor>> parameters() { return {}; }
    virtual void to(const Device&) {}
    virtual void train() { training = true; }
    virtual void eval() { training = false; }
    virtual std::vector<std::shared_ptr<Module>> children() { return {}; }

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

    Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool has_bias = true);
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

    Conv3d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool has_bias = true);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class Sequential : public Module {
public:
    std::vector<std::shared_ptr<Module>> modules;

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
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> running_mean;
    std::shared_ptr<Tensor> running_var;

    BatchNorm2d(int num_features, float eps = 1e-5f, float momentum = 0.1f);
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
    MSELoss() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class CrossEntropyLoss : public Module {
public:
    CrossEntropyLoss() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class BCELoss : public Module {
public:
    BCELoss() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class L1Loss : public Module {
public:
    L1Loss() = default;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
};

class Embedding : public Module {
public:
    int num_embeddings;
    int embedding_dim;
    std::shared_ptr<Tensor> weight;

    Embedding(int num_embeddings, int embedding_dim);
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

    std::shared_ptr<Linear> q_proj;
    std::shared_ptr<Linear> k_proj;
    std::shared_ptr<Linear> v_proj;
    std::shared_ptr<Linear> out_proj;

    MultiHeadAttention(int embed_dim, int num_heads);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> query, std::shared_ptr<Tensor> key, std::shared_ptr<Tensor> value);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
    std::vector<std::shared_ptr<Module>> children() override {
        return {q_proj, k_proj, v_proj, out_proj};
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

}
}

#endif
