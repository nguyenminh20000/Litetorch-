#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/backend.h"
#include <iostream>
#include <cmath>

namespace litetorch {
namespace nn {

class MoeForwardNode : public Node {
public:
    int num_experts;
    int top_k;
    std::shared_ptr<Tensor> indices;
    std::shared_ptr<Tensor> probs;

    MoeForwardNode(int num_experts, int top_k, std::shared_ptr<Tensor> indices, std::shared_ptr<Tensor> probs)
        : Node("MoeForwardNode"), num_experts(num_experts), top_k(top_k), indices(indices), probs(probs) {}

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto input = saved_tensors[0];
        auto gate_weight = saved_tensors[1];

        int64_t N = input->shape[0];
        int64_t D = input->shape[1];
        int64_t out_features = grad_output->shape[1];

        auto grad_input = Tensor::zeros(input->shape, input->device, false);
        auto grad_gate_weight = Tensor::zeros(gate_weight->shape, gate_weight->device, false);

        std::vector<std::shared_ptr<Tensor>> grads = {grad_input, grad_gate_weight};
        auto grad_probs = Tensor::zeros(probs->shape, probs->device, false);

        bool run_gpu = false;
        if (input->device.type == DeviceType::GPU) {
            auto backend = BackendDispatcher::get().get_backend();
            if (backend && backend->is_available()) {
                run_gpu = true;
                for (int e = 0; e < num_experts; ++e) {
                    auto w = saved_tensors[2 + e * 2];
                    auto b = saved_tensors[2 + e * 2 + 1];

                    auto gw = Tensor::zeros(w->shape, w->device, false);
                    auto gb = b ? Tensor::zeros(b->shape, b->device, false) : nullptr;
                    void* gb_ptr = gb ? gb->gpu_data() : nullptr;
                    int gb_off = gb ? gb->offset : 0;
                    void* b_ptr = b ? b->gpu_data() : nullptr;
                    int b_off = b ? b->offset : 0;

                    backend->moe_expert_backward(grad_output->gpu_data(), grad_output->offset, input->gpu_data(), input->offset, w->gpu_data(), w->offset, b_ptr, b_off, probs->gpu_data(), probs->offset, indices->gpu_data(), indices->offset, grad_input->gpu_data(), grad_input->offset, gw->gpu_data(), gw->offset, gb_ptr, gb_off, grad_probs->gpu_data(), grad_probs->offset, N, D, out_features, e, top_k);

                    grads.push_back(gw);
                    grads.push_back(gb);
                }
            }
        }
        
        if (!run_gpu) {
            auto cpu_gout = grad_output->to(Device(DeviceType::CPU, 0));
            auto cpu_in = input->to(Device(DeviceType::CPU, 0));
            auto cpu_probs = probs->to(Device(DeviceType::CPU, 0));
            auto cpu_indices = indices->to(Device(DeviceType::CPU, 0));
            
            auto cpu_gin = grad_input->to(Device(DeviceType::CPU, 0));
            auto cpu_gprobs = grad_probs->to(Device(DeviceType::CPU, 0));

            float* d_out = cpu_gout->data_ptr();
            float* in_ptr = cpu_in->data_ptr();
            float* idx_ptr = cpu_indices->data_ptr();
            float* p_ptr = cpu_probs->data_ptr();
            float* gi_ptr = cpu_gin->data_ptr();
            float* gp_ptr = cpu_gprobs->data_ptr();

            for (int e = 0; e < num_experts; ++e) {
                auto w = saved_tensors[2 + e * 2];
                auto b = saved_tensors[2 + e * 2 + 1];

                auto gw = Tensor::zeros(w->shape, w->device, false);
                auto gb = b ? Tensor::zeros(b->shape, b->device, false) : nullptr;
                
                auto cpu_w = w->to(Device(DeviceType::CPU, 0));
                auto cpu_b = b ? b->to(Device(DeviceType::CPU, 0)) : nullptr;
                auto cpu_gw = gw->to(Device(DeviceType::CPU, 0));
                auto cpu_gb = gb ? gb->to(Device(DeviceType::CPU, 0)) : nullptr;

                float* w_ptr = cpu_w->data_ptr();
                float* b_ptr = cpu_b ? cpu_b->data_ptr() : nullptr;
                float* gw_ptr = cpu_gw->data_ptr();
                float* gb_ptr = cpu_gb ? cpu_gb->data_ptr() : nullptr;

                for (int64_t i = 0; i < N; ++i) {
                    for (int k = 0; k < top_k; ++k) {
                        int exp_idx = static_cast<int>(idx_ptr[i * top_k + k]);
                        if (exp_idx == e) {
                            float p = p_ptr[i * top_k + k];

                            for (int64_t o = 0; o < out_features; ++o) {
                                float val = 0.0f;
                                for (int64_t d_idx = 0; d_idx < D; ++d_idx) {
                                    val += in_ptr[i * D + d_idx] * w_ptr[o * D + d_idx];
                                }
                                if (b_ptr) val += b_ptr[o];

                                gp_ptr[i * top_k + k] += d_out[i * out_features + o] * val;
                            }

                            for (int64_t o = 0; o < out_features; ++o) {
                                float grad_val = d_out[i * out_features + o] * p;
                                if (gb_ptr) {
                                    gb_ptr[o] += grad_val;
                                }
                                for (int64_t d_idx = 0; d_idx < D; ++d_idx) {
                                    gw_ptr[o * D + d_idx] += grad_val * in_ptr[i * D + d_idx];
                                    gi_ptr[i * D + d_idx] += grad_val * w_ptr[o * D + d_idx];
                                }
                            }
                        }
                    }
                }
                gw->copy_(cpu_gw);
                if (gb) gb->copy_(cpu_gb);
                grads.push_back(gw);
                grads.push_back(gb);
            }
            grad_input->copy_(cpu_gin);
            grad_probs->copy_(cpu_gprobs);
        }

        auto probs_grad_node = probs->creator;
        if (probs_grad_node) {
            auto probs_grads = probs_grad_node->backward(grad_probs);
            if (probs_grads.size() > 0 && probs_grads[0]) {
                float* pg_in_ptr = grads[0]->data_ptr();
                float* pg_add_ptr = probs_grads[0]->data_ptr();
                for (size_t i = 0; i < grads[0]->numel(); ++i) {
                    pg_in_ptr[i] += pg_add_ptr[i];
                }
            }
            if (probs_grads.size() > 1 && probs_grads[1]) {
                float* pg_gw_ptr = grads[1]->data_ptr();
                float* pg_add_gw = probs_grads[1]->data_ptr();
                for (size_t i = 0; i < grads[1]->numel(); ++i) {
                    pg_gw_ptr[i] += pg_add_gw[i];
                }
            }
        }

        return grads;
    }
};

MoELinear::MoELinear(int in_features, int out_features, int num_experts, int top_k)
    : num_experts(num_experts), top_k(top_k) {
    gate_weight = Tensor::create({num_experts, in_features}, Device(DeviceType::CPU), true);
    float scale = 1.0f / std::sqrt(static_cast<float>(in_features));
    float* gw_ptr = gate_weight->data_ptr();
    for (size_t i = 0; i < gate_weight->numel(); ++i) {
        gw_ptr[i] = ((static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f) * scale;
    }

    for (int i = 0; i < num_experts; ++i) {
        experts.push_back(std::make_shared<Linear>(in_features, out_features));
    }
}

std::shared_ptr<Tensor> MoELinear::forward(std::shared_ptr<Tensor> input) {
    int64_t N = input->shape[0];
    int64_t D = input->shape[1];
    int64_t out_features = experts[0]->weight->shape[0];

    std::shared_ptr<Tensor> indices;
    auto probs = Ops::moe_gate(input, gate_weight, top_k, indices);

    auto out = Tensor::zeros({N, out_features}, input->device, false);

    bool run_gpu = false;
    if (input->device.type == DeviceType::GPU) {
        auto backend = BackendDispatcher::get().get_backend();
        if (backend && backend->is_available()) {
            run_gpu = true;
            for (int e = 0; e < num_experts; ++e) {
                void* b_ptr = experts[e]->bias ? experts[e]->bias->gpu_data() : nullptr;
                int b_off = experts[e]->bias ? experts[e]->bias->offset : 0;
                backend->moe_expert_forward(input->gpu_data(), input->offset, experts[e]->weight->gpu_data(), experts[e]->weight->offset, b_ptr, b_off, probs->gpu_data(), probs->offset, indices->gpu_data(), indices->offset, out->gpu_data(), out->offset, N, D, out_features, e, top_k);
            }
        }
    }
    
    if (!run_gpu) {
        auto cpu_in = input->to(Device(DeviceType::CPU, 0));
        auto cpu_probs = probs->to(Device(DeviceType::CPU, 0));
        auto cpu_indices = indices->to(Device(DeviceType::CPU, 0));
        auto cpu_out = out->to(Device(DeviceType::CPU, 0));
        
        float* in_ptr = cpu_in->data_ptr();
        float* p_ptr = cpu_probs->data_ptr();
        float* idx_ptr = cpu_indices->data_ptr();
        float* out_ptr = cpu_out->data_ptr();

        for (int e = 0; e < num_experts; ++e) {
            auto cpu_w = experts[e]->weight->to(Device(DeviceType::CPU, 0));
            auto cpu_b = experts[e]->bias ? experts[e]->bias->to(Device(DeviceType::CPU, 0)) : nullptr;
            
            float* w_ptr = cpu_w->data_ptr();
            float* b_ptr = cpu_b ? cpu_b->data_ptr() : nullptr;

            for (int64_t i = 0; i < N; ++i) {
                for (int k = 0; k < top_k; ++k) {
                    int exp_idx = static_cast<int>(idx_ptr[i * top_k + k]);
                    if (exp_idx == e) {
                        float p = p_ptr[i * top_k + k];
                        for (int64_t o = 0; o < out_features; ++o) {
                            float val = 0.0f;
                            for (int64_t d_idx = 0; d_idx < D; ++d_idx) {
                                val += in_ptr[i * D + d_idx] * w_ptr[o * D + d_idx];
                            }
                            if (b_ptr) val += b_ptr[o];
                            out_ptr[i * out_features + o] += p * val;
                        }
                    }
                }
            }
        }
        out->copy_(cpu_out);
    }

    bool requires_grad = input->requires_grad || gate_weight->requires_grad;
    for (int e = 0; e < num_experts; ++e) {
        requires_grad = requires_grad || experts[e]->weight->requires_grad || (experts[e]->bias && experts[e]->bias->requires_grad);
    }

    if (Autograd::active_tensors.size() > 0 || requires_grad) {
        out->requires_grad = requires_grad;
        auto node = std::make_shared<MoeForwardNode>(num_experts, top_k, indices, probs);
        node->next_nodes = {input->creator, gate_weight->creator};
        node->inputs = {{input, input->requires_grad}, {gate_weight, gate_weight->requires_grad}};
        node->saved_tensors = {input, gate_weight};

        for (int e = 0; e < num_experts; ++e) {
            node->next_nodes.push_back(experts[e]->weight->creator);
            node->inputs.push_back({experts[e]->weight, experts[e]->weight->requires_grad});
            node->saved_tensors.push_back(experts[e]->weight);

            if (experts[e]->bias) {
                node->next_nodes.push_back(experts[e]->bias->creator);
                node->inputs.push_back({experts[e]->bias, experts[e]->bias->requires_grad});
                node->saved_tensors.push_back(experts[e]->bias);
            } else {
                node->next_nodes.push_back(nullptr);
                node->inputs.push_back(NodeInput{std::weak_ptr<Tensor>(), false});
                node->saved_tensors.push_back(nullptr);
            }
        }

        out->creator = node;
        node->output = out;
    }

    return out;
}

std::vector<std::shared_ptr<Tensor>> MoELinear::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    params.push_back(gate_weight);
    for (int i = 0; i < num_experts; ++i) {
        auto ep = experts[i]->parameters();
        params.insert(params.end(), ep.begin(), ep.end());
    }
    return params;
}

void MoELinear::to(const Device& device) {
    gate_weight = gate_weight->to(device);
    for (int i = 0; i < num_experts; ++i) {
        experts[i]->to(device);
    }
}

}
}
