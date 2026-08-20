#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/tensor.h"
#include "litetorch/backend.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace litetorch {

class MoeGateNode : public Node {
public:
    int top_k;
    std::shared_ptr<Tensor> indices;

    MoeGateNode(int top_k, std::shared_ptr<Tensor> indices)
        : Node("MoeGateNode"), top_k(top_k), indices(indices) {}

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto input = saved_tensors[0];
        auto gate_weight = saved_tensors[1];
        auto probs = saved_tensors[2];

        int64_t N = input->shape[0];
        int64_t D = input->shape[1];
        int64_t E = gate_weight->shape[0];

        auto grad_input = Tensor::zeros(input->shape, input->device, false);
        auto grad_gate_weight = Tensor::zeros(gate_weight->shape, gate_weight->device, false);

        bool run_gpu = false;
        if (grad_output->device.type == DeviceType::GPU) {
            auto backend = BackendDispatcher::get().get_backend();
            if (backend && backend->is_available()) {
                run_gpu = true;
                backend->moe_gate_backward(grad_output->gpu_data(), grad_output->offset, input->gpu_data(), input->offset, gate_weight->gpu_data(), gate_weight->offset, probs->gpu_data(), probs->offset, indices->gpu_data(), indices->offset, grad_input->gpu_data(), grad_input->offset, grad_gate_weight->gpu_data(), grad_gate_weight->offset, N, D, E, top_k);
            }
        }
        
        if (!run_gpu) {
            auto cpu_gout = grad_output->to(Device(DeviceType::CPU, 0));
            auto cpu_in = input->to(Device(DeviceType::CPU, 0));
            auto cpu_gw = gate_weight->to(Device(DeviceType::CPU, 0));
            auto cpu_probs = probs->to(Device(DeviceType::CPU, 0));
            auto cpu_indices = indices->to(Device(DeviceType::CPU, 0));
            
            auto cpu_gin = grad_input->to(Device(DeviceType::CPU, 0));
            auto cpu_ggw = grad_gate_weight->to(Device(DeviceType::CPU, 0));
            
            float* d_out = cpu_gout->data_ptr();
            float* in_ptr = cpu_in->data_ptr();
            float* gw_ptr = cpu_gw->data_ptr();
            float* p_ptr = cpu_probs->data_ptr();
            float* idx_ptr = cpu_indices->data_ptr();
            float* gi_ptr = cpu_gin->data_ptr();
            float* ggw_ptr = cpu_ggw->data_ptr();

            std::vector<float> dS(N * E, 0.0f);
            std::vector<float> S(N * E, 0.0f);

            for (int64_t i = 0; i < N; ++i) {
                for (int k = 0; k < top_k; ++k) {
                    int e = static_cast<int>(idx_ptr[i * top_k + k]);
                    float p = p_ptr[i * top_k + k];
                    S[i * E + e] = p;
                    dS[i * E + e] = d_out[i * top_k + k];
                }
            }

            std::vector<float> dH(N * E, 0.0f);
            for (int64_t i = 0; i < N; ++i) {
                float sum_ds_s = 0.0f;
                for (int e = 0; e < E; ++e) {
                    sum_ds_s += dS[i * E + e] * S[i * E + e];
                }
                for (int e = 0; e < E; ++e) {
                    dH[i * E + e] = S[i * E + e] * (dS[i * E + e] - sum_ds_s);
                }
            }

            for (int64_t i = 0; i < N; ++i) {
                for (int64_t d = 0; d < D; ++d) {
                    float sum_gi = 0.0f;
                    for (int e = 0; e < E; ++e) {
                        sum_gi += dH[i * E + e] * gw_ptr[e * D + d];
                    }
                    gi_ptr[i * D + d] = sum_gi;
                }
            }

            for (int e = 0; e < E; ++e) {
                for (int64_t d = 0; d < D; ++d) {
                    float sum_ggw = 0.0f;
                    for (int64_t i = 0; i < N; ++i) {
                        sum_ggw += dH[i * E + e] * in_ptr[i * D + d];
                    }
                    ggw_ptr[e * D + d] = sum_ggw;
                }
            }
            grad_input->copy_(cpu_gin);
            grad_gate_weight->copy_(cpu_ggw);
        }

        return {grad_input, grad_gate_weight};
    }
};

namespace Ops {

std::shared_ptr<Tensor> moe_gate(
    std::shared_ptr<Tensor> input,
    std::shared_ptr<Tensor> gate_weight,
    int top_k,
    std::shared_ptr<Tensor>& indices_out
) {
    int64_t N = input->shape[0];
    int64_t D = input->shape[1];
    int64_t E = gate_weight->shape[0];

    auto logits = Ops::matmul(input, gate_weight->transpose(0, 1));
    auto out_probs = Tensor::create({N, top_k}, input->device, false);
    indices_out = Tensor::create({N, top_k}, input->device, false);

    bool run_gpu = false;
    if (input->device.type == DeviceType::GPU) {
        auto backend = BackendDispatcher::get().get_backend();
        if (backend && backend->is_available()) {
            run_gpu = true;
            backend->moe_gate(logits->gpu_data(), logits->offset, out_probs->gpu_data(), out_probs->offset, indices_out->gpu_data(), indices_out->offset, N, E, top_k);
        }
    }
    
    if (!run_gpu) {
        auto cpu_logits = logits->to(Device(DeviceType::CPU, 0));
        float* logit_ptr = cpu_logits->data_ptr();

        auto probs_full = Tensor::create({N, E}, Device(DeviceType::CPU, 0), false);
        float* pf_ptr = probs_full->data_ptr();

        for (int64_t i = 0; i < N; ++i) {
            float max_val = -1e9f;
            for (int e = 0; e < E; ++e) {
                max_val = std::max(max_val, logit_ptr[i * E + e]);
            }
            float sum_exp = 0.0f;
            for (int e = 0; e < E; ++e) {
                pf_ptr[i * E + e] = std::exp(logit_ptr[i * E + e] - max_val);
                sum_exp += pf_ptr[i * E + e];
            }
            for (int e = 0; e < E; ++e) {
                pf_ptr[i * E + e] /= sum_exp;
            }
        }

        auto cpu_out_probs = out_probs->to(Device(DeviceType::CPU, 0));
        auto cpu_indices_out = indices_out->to(Device(DeviceType::CPU, 0));
        float* op_ptr = cpu_out_probs->data_ptr();
        float* io_ptr = cpu_indices_out->data_ptr();

        for (int64_t i = 0; i < N; ++i) {
            std::vector<std::pair<float, int>> exp_probs;
            for (int e = 0; e < E; ++e) {
                exp_probs.push_back({pf_ptr[i * E + e], e});
            }
            std::sort(exp_probs.begin(), exp_probs.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });
            float sum_top_k = 0.0f;
            for (int k = 0; k < top_k; ++k) {
                sum_top_k += exp_probs[k].first;
            }
            for (int k = 0; k < top_k; ++k) {
                op_ptr[i * top_k + k] = exp_probs[k].first / (sum_top_k + 1e-9f);
                io_ptr[i * top_k + k] = static_cast<float>(exp_probs[k].second);
            }
        }
        out_probs->copy_(cpu_out_probs);
        indices_out->copy_(cpu_indices_out);
    }

    bool requires_grad = input->requires_grad || gate_weight->requires_grad;
    if (Autograd::active_tensors.size() > 0 || requires_grad) {
        out_probs->requires_grad = requires_grad;
        auto node = std::make_shared<MoeGateNode>(top_k, indices_out);
        node->next_nodes = {input->creator, gate_weight->creator};
        node->inputs = {{input, input->requires_grad}, {gate_weight, gate_weight->requires_grad}};
        node->saved_tensors = {input, gate_weight, out_probs};
        out_probs->creator = node;
        node->output = out_probs;
    }

    return out_probs;
}

}
}
