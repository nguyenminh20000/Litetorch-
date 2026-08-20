#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace litetorch;

void test_transformer(const Device& dev) {
    std::cout << "\n--- Testing TransformerDecoderLayer on " << (dev.type == DeviceType::GPU ? "GPU" : "CPU") << " ---\n";

    int B = 2;
    int T_tgt = 4;
    int T_mem = 6;
    int C = 8;
    int num_heads = 2;
    int dim_feedforward = 16;

    std::vector<float> tgt_data;
    for (int i = 0; i < B * T_tgt * C; ++i) {
        tgt_data.push_back(sin(static_cast<float>(i)));
    }
    auto tgt = Tensor::from_vector(tgt_data, {B, T_tgt, C}, dev, true);

    std::vector<float> mem_data;
    for (int i = 0; i < B * T_mem * C; ++i) {
        mem_data.push_back(cos(static_cast<float>(i)));
    }
    auto memory = Tensor::from_vector(mem_data, {B, T_mem, C}, dev, true);

    auto layer = std::make_shared<nn::TransformerDecoderLayer>(C, num_heads, dim_feedforward);
    layer->to(dev);

    std::cout << "1. Testing Decoder-only mode (self-attention + FFN)...\n";
    auto out_self = layer->forward(tgt);
    std::cout << "Output shape: (" << out_self->shape[0] << ", " << out_self->shape[1] << ", " << out_self->shape[2] << ")\n";
    assert(out_self->shape[0] == B);
    assert(out_self->shape[1] == T_tgt);
    assert(out_self->shape[2] == C);

    auto loss_self = Ops::sum(out_self);
    std::cout << "Forward self-attention success. Loss: " << loss_self->item() << "\n";
    loss_self->backward();

    float tgt_grad_sum = 0.0f;
    if (tgt->grad) {
        auto grad_vec = tgt->grad->to_vector();
        for (float g : grad_vec) tgt_grad_sum += std::abs(g);
    }
    std::cout << "Tgt gradient absolute sum: " << tgt_grad_sum << "\n";
    assert(tgt_grad_sum > 0.0f);

    float w_grad_sum = 0.0f;
    if (layer->self_attn->q_proj->weight->grad) {
        auto w_grad_vec = layer->self_attn->q_proj->weight->grad->to_vector();
        for (float g : w_grad_vec) w_grad_sum += std::abs(g);
    }
    std::cout << "Self-attn Q-proj weight gradient absolute sum: " << w_grad_sum << "\n";
    assert(w_grad_sum > 0.0f);

    for (auto& p : layer->parameters()) {
        if (p->grad) {
            p->grad = nullptr;
        }
    }
    tgt->grad = nullptr;
    memory->grad = nullptr;

    std::cout << "2. Testing Encoder-Decoder mode (self-attention + cross-attention + FFN)...\n";
    auto out_cross = layer->forward(tgt, memory);
    std::cout << "Output shape: (" << out_cross->shape[0] << ", " << out_cross->shape[1] << ", " << out_cross->shape[2] << ")\n";
    assert(out_cross->shape[0] == B);
    assert(out_cross->shape[1] == T_tgt);
    assert(out_cross->shape[2] == C);

    auto loss_cross = Ops::sum(out_cross);
    std::cout << "Forward cross-attention success. Loss: " << loss_cross->item() << "\n";
    loss_cross->backward();

    float tgt_grad_cross_sum = 0.0f;
    if (tgt->grad) {
        auto grad_vec = tgt->grad->to_vector();
        for (float g : grad_vec) tgt_grad_cross_sum += std::abs(g);
    }
    std::cout << "Tgt gradient absolute sum (cross): " << tgt_grad_cross_sum << "\n";
    assert(tgt_grad_cross_sum > 0.0f);

    float mem_grad_cross_sum = 0.0f;
    if (memory->grad) {
        auto grad_vec = memory->grad->to_vector();
        for (float g : grad_vec) mem_grad_cross_sum += std::abs(g);
    }
    std::cout << "Memory gradient absolute sum (cross): " << mem_grad_cross_sum << "\n";
    assert(mem_grad_cross_sum > 0.0f);

    float cross_w_grad_sum = 0.0f;
    if (layer->multihead_attn->q_proj->weight->grad) {
        auto w_grad_vec = layer->multihead_attn->q_proj->weight->grad->to_vector();
        for (float g : w_grad_vec) cross_w_grad_sum += std::abs(g);
    }
    std::cout << "Cross-attn Q-proj weight gradient absolute sum: " << cross_w_grad_sum << "\n";
    assert(cross_w_grad_sum > 0.0f);

    std::cout << "[RESULT] SUCCESS: TransformerDecoderLayer passed on " << (dev.type == DeviceType::GPU ? "GPU" : "CPU") << "\n";
}

int main() {
    std::cout << "==================================================\n";
    std::cout << "       LITETORCH TRANSFORMER DECODER TEST         \n";
    std::cout << "==================================================\n";

    std::vector<Device> devices = { Device(DeviceType::CPU, 0) };
    if (CLBackend::get().is_available()) {
        devices.push_back(Device(DeviceType::GPU, 0));
    }

    for (const auto& dev : devices) {
        test_transformer(dev);
    }

    std::cout << "==================================================\n";
    return 0;
}
