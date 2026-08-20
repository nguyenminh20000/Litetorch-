#ifndef LITETORCH_OPS_H
#define LITETORCH_OPS_H

#include "litetorch/tensor.h"
#include <memory>
#include <vector>

namespace litetorch {
namespace Ops {

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> bmm(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> conv2d(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias = nullptr, int stride = 1, int padding = 0);
std::shared_ptr<Tensor> conv3d(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias = nullptr, int stride = 1, int padding = 0);
std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> reduce_broadcast(std::shared_ptr<Tensor> grad, const std::vector<int64_t>& orig_shape);

std::shared_ptr<Tensor> pow(std::shared_ptr<Tensor> a, float exponent);
std::shared_ptr<Tensor> sqrt(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> exp(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> log(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> abs(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> neg(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> leaky_relu(std::shared_ptr<Tensor> a, float negative_slope = 0.01f);
std::shared_ptr<Tensor> cat(const std::vector<std::shared_ptr<Tensor>>& tensors, int64_t dim = 0);
std::shared_ptr<Tensor> squeeze(std::shared_ptr<Tensor> a, int64_t dim = -1);
std::shared_ptr<Tensor> unsqueeze(std::shared_ptr<Tensor> a, int64_t dim);
std::shared_ptr<Tensor> max_pool2d(std::shared_ptr<Tensor> input, int kernel_size, int stride = -1, int padding = 0);
std::shared_ptr<Tensor> max_pool3d(std::shared_ptr<Tensor> input, int kernel_size, int stride = -1, int padding = 0);
std::shared_ptr<Tensor> batch_norm2d(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> running_mean, std::shared_ptr<Tensor> running_var, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias, bool training, float momentum = 0.1f, float eps = 1e-5f);
std::shared_ptr<Tensor> l1_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> bce_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);

std::shared_ptr<Tensor> clamp(std::shared_ptr<Tensor> a, float min_val, float max_val);
std::shared_ptr<Tensor> sin(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> cos(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> softmax(std::shared_ptr<Tensor> a, int64_t dim = -1);
std::shared_ptr<Tensor> gelu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> layer_norm(std::shared_ptr<Tensor> input, const std::vector<int64_t>& normalized_shape, std::shared_ptr<Tensor> weight = nullptr, std::shared_ptr<Tensor> bias = nullptr, float eps = 1e-5f);
std::shared_ptr<Tensor> fused_add_layernorm(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> residual, const std::vector<int64_t>& normalized_shape, std::shared_ptr<Tensor> weight = nullptr, std::shared_ptr<Tensor> bias = nullptr, float eps = 1e-5f);
std::shared_ptr<Tensor> adaptive_avg_pool2d(std::shared_ptr<Tensor> input, int output_height, int output_width);
std::shared_ptr<Tensor> cast(std::shared_ptr<Tensor> a, DataType target_dtype);
std::shared_ptr<Tensor> fake_quantize(std::shared_ptr<Tensor> input, float scale, float zero_point = 0.0f, int bits = 8);
std::shared_ptr<Tensor> flash_attention(std::shared_ptr<Tensor> q, std::shared_ptr<Tensor> k, std::shared_ptr<Tensor> v);
float clip_grad_norm_(const std::vector<std::shared_ptr<Tensor>>& params, float max_norm, float norm_type = 2.0f);
std::shared_ptr<Tensor> rope(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> cos, std::shared_ptr<Tensor> sin);
std::shared_ptr<Tensor> paged_attention(
    std::shared_ptr<Tensor> q,
    std::shared_ptr<Tensor> k_cache,
    std::shared_ptr<Tensor> v_cache,
    std::shared_ptr<Tensor> block_tables,
    std::shared_ptr<Tensor> context_lens,
    int block_size
);
std::shared_ptr<Tensor> scaled_matmul(
    std::shared_ptr<Tensor> a,
    std::shared_ptr<Tensor> b,
    float a_scale,
    float b_scale,
    std::shared_ptr<Tensor> bias = nullptr,
    DataType out_dtype = DataType::FP32
);
std::shared_ptr<Tensor> ring_attention(
    std::shared_ptr<Tensor> q,
    std::shared_ptr<Tensor> k,
    std::shared_ptr<Tensor> v,
    std::shared_ptr<distributed::ProcessGroup> pg
);
std::shared_ptr<Tensor> flash_decoding(
    std::shared_ptr<Tensor> q,
    std::shared_ptr<Tensor> k,
    std::shared_ptr<Tensor> v,
    int num_splits
);
std::shared_ptr<Tensor> fused_linear_cross_entropy(
    std::shared_ptr<Tensor> x,
    std::shared_ptr<Tensor> weight,
    std::shared_ptr<Tensor> target,
    std::shared_ptr<Tensor> bias = nullptr
);
std::shared_ptr<Tensor> moe_gate(
    std::shared_ptr<Tensor> input,
    std::shared_ptr<Tensor> gate_weight,
    int top_k,
    std::shared_ptr<Tensor>& indices_out
);
std::shared_ptr<Tensor> w8a8_matmul(
    std::shared_ptr<Tensor> x,
    std::shared_ptr<Tensor> w,
    float x_scale,
    float w_scale
);
}
namespace nn {
class Module;
}
namespace Ops {
std::shared_ptr<Tensor> checkpoint(std::shared_ptr<nn::Module> module, std::shared_ptr<Tensor> input);
}
}

#endif
