#include "litetorch/ops.h"
#include "litetorch/tensor.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace litetorch {

class FusedLinearCrossEntropyNode : public Node {
public:
    FusedLinearCrossEntropyNode() : Node("FusedLinearCrossEntropy") {}

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto x = saved_tensors[0];
        auto weight = saved_tensors[1];
        auto target = saved_tensors[2];
        auto bias = (saved_tensors.size() > 3) ? saved_tensors[3] : nullptr;

        float d_loss = grad_output->to_vector()[0];

        auto grad_x = Tensor::zeros(x->shape, x->device, false);
        auto grad_weight = Tensor::zeros(weight->shape, weight->device, false);
        std::shared_ptr<Tensor> grad_bias = nullptr;
        if (bias) {
            grad_bias = Tensor::zeros(bias->shape, bias->device, false);
        }

        int64_t N = x->shape[0];
        int64_t D = x->shape[1];
        int64_t Vocab = weight->shape[0];

        float* x_ptr = x->data_ptr();
        float* w_ptr = weight->data_ptr();
        float* t_ptr = target->data_ptr();
        float* b_ptr = bias ? bias->data_ptr() : nullptr;

        float* gx_ptr = grad_x->data_ptr();
        float* gw_ptr = grad_weight->data_ptr();
        float* gb_ptr = grad_bias ? grad_bias->data_ptr() : nullptr;

        int64_t chunk_size = 512;
        int64_t num_chunks = (N + chunk_size - 1) / chunk_size;

        for (int64_t c = 0; c < num_chunks; ++c) {
            int64_t start = c * chunk_size;
            int64_t end = std::min(start + chunk_size, N);
            int64_t curr_size = end - start;

            std::vector<float> dS(curr_size * Vocab, 0.0f);

            for (int64_t i = 0; i < curr_size; ++i) {
                int64_t idx = start + i;
                int64_t target_val = static_cast<int64_t>(t_ptr[idx]);

                float max_logit = -1e9f;
                std::vector<float> logits(Vocab, 0.0f);

                for (int64_t v = 0; v < Vocab; ++v) {
                    float val = b_ptr ? b_ptr[v] : 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        val += x_ptr[idx * D + d] * w_ptr[v * D + d];
                    }
                    logits[v] = val;
                    if (val > max_logit) max_logit = val;
                }

                float sum_exp = 0.0f;
                for (int64_t v = 0; v < Vocab; ++v) {
                    logits[v] = std::exp(logits[v] - max_logit);
                    sum_exp += logits[v];
                }

                for (int64_t v = 0; v < Vocab; ++v) {
                    float prob = logits[v] / sum_exp;
                    float delta = (v == target_val) ? 1.0f : 0.0f;
                    dS[i * Vocab + v] = (prob - delta) / static_cast<float>(N) * d_loss;
                }
            }

            for (int64_t i = 0; i < curr_size; ++i) {
                int64_t idx = start + i;
                for (int64_t d = 0; d < D; ++d) {
                    float sum_x = 0.0f;
                    for (int64_t v = 0; v < Vocab; ++v) {
                        sum_x += dS[i * Vocab + v] * w_ptr[v * D + d];
                        gw_ptr[v * D + d] += dS[i * Vocab + v] * x_ptr[idx * D + d];
                    }
                    gx_ptr[idx * D + d] = sum_x;
                }
                if (gb_ptr) {
                    for (int64_t v = 0; v < Vocab; ++v) {
                        gb_ptr[v] += dS[i * Vocab + v];
                    }
                }
            }
        }

        std::vector<std::shared_ptr<Tensor>> grads = {grad_x, grad_weight, std::shared_ptr<Tensor>()};
        if (bias) {
            grads.push_back(grad_bias);
        }
        return grads;
    }
};

namespace Ops {

std::shared_ptr<Tensor> fused_linear_cross_entropy(
    std::shared_ptr<Tensor> x,
    std::shared_ptr<Tensor> weight,
    std::shared_ptr<Tensor> target,
    std::shared_ptr<Tensor> bias)
{
    if (x->device.type != DeviceType::CPU || weight->device.type != DeviceType::CPU) {
        throw std::runtime_error("Fused Linear Cross Entropy only supported on CPU currently");
    }

    auto x_2d = x;
    if (x->shape.size() == 3) {
        x_2d = x->view({x->shape[0] * x->shape[1], x->shape[2]});
    }

    auto target_1d = target;
    if (target->shape.size() == 2) {
        target_1d = target->view({target->shape[0] * target->shape[1]});
    }

    int64_t N = x_2d->shape[0];
    int64_t D = x_2d->shape[1];
    int64_t Vocab = weight->shape[0];

    float* x_ptr = x_2d->data_ptr();
    float* w_ptr = weight->data_ptr();
    float* t_ptr = target_1d->data_ptr();
    float* b_ptr = bias ? bias->data_ptr() : nullptr;

    double total_loss = 0.0;
    int64_t chunk_size = 512;
    int64_t num_chunks = (N + chunk_size - 1) / chunk_size;

    for (int64_t c = 0; c < num_chunks; ++c) {
        int64_t start = c * chunk_size;
        int64_t end = std::min(start + chunk_size, N);
        int64_t curr_size = end - start;

        for (int64_t i = 0; i < curr_size; ++i) {
            int64_t idx = start + i;
            int64_t target_val = static_cast<int64_t>(t_ptr[idx]);

            float max_logit = -1e9f;
            std::vector<float> logits(Vocab, 0.0f);

            for (int64_t v = 0; v < Vocab; ++v) {
                float val = b_ptr ? b_ptr[v] : 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    val += x_ptr[idx * D + d] * w_ptr[v * D + d];
                }
                logits[v] = val;
                if (val > max_logit) max_logit = val;
            }

            float sum_exp = 0.0f;
            for (int64_t v = 0; v < Vocab; ++v) {
                logits[v] = std::exp(logits[v] - max_logit);
                sum_exp += logits[v];
            }

            float prob_target = logits[target_val] / sum_exp;
            total_loss += -std::log(prob_target + 1e-15f);
        }
    }

    bool requires_grad = x->requires_grad || weight->requires_grad || (bias && bias->requires_grad);
    float final_loss = static_cast<float>(total_loss / N);
    auto loss_tensor = Tensor::from_vector({final_loss}, {1}, x->device, requires_grad);

    if (Autograd::active_tensors.size() > 0 || requires_grad) {
        auto node = std::make_shared<FusedLinearCrossEntropyNode>();
        node->next_nodes = {x_2d->creator, weight->creator, target_1d->creator};
        node->inputs = {{x_2d, x_2d->requires_grad}, {weight, weight->requires_grad}, {target_1d, target_1d->requires_grad}};
        node->saved_tensors = {x_2d, weight, target_1d};
        if (bias) {
            node->next_nodes.push_back(bias->creator);
            node->inputs.push_back({bias, bias->requires_grad});
            node->saved_tensors.push_back(bias);
        }
        loss_tensor->creator = node;
        node->output = loss_tensor;
    }

    return loss_tensor;
}

}
}
