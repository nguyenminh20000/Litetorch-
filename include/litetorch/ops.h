#ifndef LITETORCH_OPS_H
#define LITETORCH_OPS_H

#include "litetorch/tensor.h"
#include <memory>
#include <vector>
#include <cstdint>
#include <utility>
#include <tuple>

namespace litetorch {
namespace Ops {

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> matmul_bias(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, std::shared_ptr<Tensor> bias);
std::shared_ptr<Tensor> bmm(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim = false);
std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> conv2d(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias = nullptr, int stride = 1, int padding = 0);
std::shared_ptr<Tensor> conv2d_relu(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias = nullptr, int stride = 1, int padding = 0);
std::shared_ptr<Tensor> conv3d(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias = nullptr, int stride = 1, int padding = 0);
std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
    std::shared_ptr<Tensor> weight = nullptr, int64_t ignore_index = -100, const std::string& reduction = "mean",
    float label_smoothing = 0.0f);
std::shared_ptr<Tensor> reduce_broadcast(std::shared_ptr<Tensor> grad, const std::vector<int64_t>& orig_shape);

std::shared_ptr<Tensor> pow(std::shared_ptr<Tensor> a, float exponent);
std::shared_ptr<Tensor> sqrt(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> exp(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> log(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> abs(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> neg(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim = false);
std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim = false);
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> max_with_indices(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim = false);
std::shared_ptr<Tensor> permute(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims);
std::shared_ptr<Tensor> expand(std::shared_ptr<Tensor> a, const std::vector<int64_t>& shape);
std::shared_ptr<Tensor> stack(const std::vector<std::shared_ptr<Tensor>>& tensors, int64_t dim = 0);
std::shared_ptr<Tensor> gather(std::shared_ptr<Tensor> input, int64_t dim, std::shared_ptr<Tensor> index);
std::shared_ptr<Tensor> argmax(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim = false);
std::shared_ptr<Tensor> argmin(std::shared_ptr<Tensor> a, int64_t dim, bool keepdim = false);
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> topk(std::shared_ptr<Tensor> a, int64_t k, int64_t dim = -1, bool largest = true, bool sorted = true);
std::shared_ptr<Tensor> eq(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> ne(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> lt(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> gt(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> le(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> ge(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> where(std::shared_ptr<Tensor> cond, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> log_softmax(std::shared_ptr<Tensor> a, int64_t dim = -1);
std::shared_ptr<Tensor> silu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> leaky_relu(std::shared_ptr<Tensor> a, float negative_slope = 0.01f);
std::shared_ptr<Tensor> cat(const std::vector<std::shared_ptr<Tensor>>& tensors, int64_t dim = 0);
std::vector<std::shared_ptr<Tensor>> split(std::shared_ptr<Tensor> input, const std::vector<int64_t>& sizes, int64_t dim = -1);
std::vector<std::shared_ptr<Tensor>> chunk(std::shared_ptr<Tensor> input, int64_t chunks, int64_t dim = 0);
constexpr int64_t SQUEEZE_ALL_DIMS = INT64_MIN;
std::shared_ptr<Tensor> squeeze(std::shared_ptr<Tensor> a, int64_t dim = SQUEEZE_ALL_DIMS);
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
std::shared_ptr<Tensor> rmsnorm(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight, float eps = 1e-5f);
std::shared_ptr<Tensor> fused_add_layernorm(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> residual, const std::vector<int64_t>& normalized_shape, std::shared_ptr<Tensor> weight = nullptr, std::shared_ptr<Tensor> bias = nullptr, float eps = 1e-5f);
std::shared_ptr<Tensor> adaptive_avg_pool2d(std::shared_ptr<Tensor> input, int output_height, int output_width);
std::shared_ptr<Tensor> cast(std::shared_ptr<Tensor> a, DataType target_dtype);
std::shared_ptr<Tensor> fake_quantize(std::shared_ptr<Tensor> input, float scale, float zero_point = 0.0f, int bits = 8);
std::shared_ptr<Tensor> flash_attention(std::shared_ptr<Tensor> q, std::shared_ptr<Tensor> k, std::shared_ptr<Tensor> v, bool causal = false);
std::shared_ptr<Tensor> flash_attention_qkv(std::shared_ptr<Tensor> qkv, int64_t num_heads, bool causal = false);
std::shared_ptr<Tensor> qkv_extract(std::shared_ptr<Tensor> qkv, int64_t num_heads, int64_t index);
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
    std::shared_ptr<distributed::ProcessGroup> pg,
    bool causal = false
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
std::shared_ptr<Tensor> fp8_cast_scaled(std::shared_ptr<Tensor> x, float scale, DataType dtype = DataType::FP8_E4M3);
std::shared_ptr<Tensor> fp8_uncast_scaled(std::shared_ptr<Tensor> x, float scale);
std::shared_ptr<Tensor> fp8_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
                                    float a_scale, float b_scale, float out_scale,
                                    DataType out_dtype = DataType::FP8_E4M3);
std::shared_ptr<Tensor> int4_pack(std::shared_ptr<Tensor> x);
std::shared_ptr<Tensor> int4_unpack(std::shared_ptr<Tensor> packed, int64_t numel);
std::shared_ptr<Tensor> int4_weight_only_linear(std::shared_ptr<Tensor> x,
                                                 std::shared_ptr<Tensor> packed_w,
                                                 std::shared_ptr<Tensor> w_scale,
                                                 std::shared_ptr<Tensor> bias = nullptr);
// === W1-T4 ===
std::shared_ptr<Tensor> conv1d(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias = nullptr, int stride = 1, int padding = 0);
std::shared_ptr<Tensor> avg_pool2d(std::shared_ptr<Tensor> input, int kernel_size, int stride = -1, int padding = 0);
std::shared_ptr<Tensor> nll_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
    std::shared_ptr<Tensor> weight = nullptr, int64_t ignore_index = -100, const std::string& reduction = "mean");
std::shared_ptr<Tensor> bce_with_logits_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
    std::shared_ptr<Tensor> weight = nullptr, const std::string& reduction = "mean", std::shared_ptr<Tensor> pos_weight = nullptr);
}
namespace nn {
class Module;
}
namespace Ops {
std::shared_ptr<Tensor> checkpoint(std::shared_ptr<nn::Module> module, std::shared_ptr<Tensor> input);
}

// === W1-T1 core-ops ===
namespace Ops {
std::shared_ptr<Tensor> from_data(const std::vector<float>& data, const std::vector<int64_t>& shape,
    const Device& device = Device(DeviceType::CPU, 0), bool requires_grad = false, DataType dtype = DataType::FP32);
std::shared_ptr<Tensor> empty(const std::vector<int64_t>& shape,
    const Device& device = Device(DeviceType::CPU, 0), bool requires_grad = false, DataType dtype = DataType::FP32);
std::shared_ptr<Tensor> full(const std::vector<int64_t>& shape, float fill_value,
    const Device& device = Device(DeviceType::CPU, 0), bool requires_grad = false, DataType dtype = DataType::FP32);
std::shared_ptr<Tensor> linspace(float start, float end, int64_t steps,
    const Device& device = Device(DeviceType::CPU, 0), DataType dtype = DataType::FP32);
std::shared_ptr<Tensor> logspace(float start, float end, int64_t steps, float base = 10.0f,
    const Device& device = Device(DeviceType::CPU, 0), DataType dtype = DataType::FP32);
std::shared_ptr<Tensor> eye(int64_t n, int64_t m = -1,
    const Device& device = Device(DeviceType::CPU, 0), DataType dtype = DataType::FP32);
std::shared_ptr<Tensor> rand(const std::vector<int64_t>& shape,
    const Device& device = Device(DeviceType::CPU, 0), bool requires_grad = false, DataType dtype = DataType::FP32);
std::shared_ptr<Tensor> randint(int64_t low, int64_t high, const std::vector<int64_t>& shape,
    const Device& device = Device(DeviceType::CPU, 0), DataType dtype = DataType::INT32);
std::shared_ptr<Tensor> randperm(int64_t n,
    const Device& device = Device(DeviceType::CPU, 0), DataType dtype = DataType::INT32);
std::shared_ptr<Tensor> zeros_like(std::shared_ptr<Tensor> a, const Device& device, DataType dtype);
std::shared_ptr<Tensor> ones_like(std::shared_ptr<Tensor> a, const Device& device, DataType dtype);
std::shared_ptr<Tensor> empty_like(std::shared_ptr<Tensor> a, const Device& device, DataType dtype);
std::shared_ptr<Tensor> full_like(std::shared_ptr<Tensor> a, float fill_value, const Device& device, DataType dtype);
std::shared_ptr<Tensor> rand_like(std::shared_ptr<Tensor> a, const Device& device, DataType dtype);
std::shared_ptr<Tensor> randn_like(std::shared_ptr<Tensor> a, const Device& device, DataType dtype);

std::shared_ptr<Tensor> log2(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> log10(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> rsqrt(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> tan(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> reciprocal(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sign(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> floor(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> ceil(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> round(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> trunc(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> clamp_min(std::shared_ptr<Tensor> a, float min_val);
std::shared_ptr<Tensor> clamp_max(std::shared_ptr<Tensor> a, float max_val);

std::shared_ptr<Tensor> min(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> amax(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim = false);
std::shared_ptr<Tensor> amin(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim = false);
std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim = false);
std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim = false);
std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a, const std::vector<int64_t>& dims, bool keepdim = false);
bool allclose(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, float rtol = 1e-5f, float atol = 1e-8f, bool equal_nan = false);
std::shared_ptr<Tensor> isclose(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, float rtol = 1e-5f, float atol = 1e-8f, bool equal_nan = false);

std::shared_ptr<Tensor> triu(std::shared_ptr<Tensor> a, int64_t diagonal = 0);
std::shared_ptr<Tensor> tril(std::shared_ptr<Tensor> a, int64_t diagonal = 0);
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> sort(std::shared_ptr<Tensor> a, int64_t dim = -1, bool descending = false);
std::shared_ptr<Tensor> argsort(std::shared_ptr<Tensor> a, int64_t dim = -1, bool descending = false);
std::shared_ptr<Tensor> cumsum(std::shared_ptr<Tensor> a, int64_t dim);
std::shared_ptr<Tensor> scatter(std::shared_ptr<Tensor> input, int64_t dim, std::shared_ptr<Tensor> index, std::shared_ptr<Tensor> src);
std::shared_ptr<Tensor> scatter_add(std::shared_ptr<Tensor> input, int64_t dim, std::shared_ptr<Tensor> index, std::shared_ptr<Tensor> src);
std::shared_ptr<Tensor> flatten(std::shared_ptr<Tensor> a, int64_t start_dim = 0, int64_t end_dim = -1);
std::shared_ptr<Tensor> repeat(std::shared_ptr<Tensor> a, const std::vector<int64_t>& repeats);
std::shared_ptr<Tensor> narrow(std::shared_ptr<Tensor> a, int64_t dim, int64_t start, int64_t length);
std::shared_ptr<Tensor> t(std::shared_ptr<Tensor> a);

// === LT-ENG-022 fused MFU kernels ===
std::shared_ptr<Tensor> fused_swiglu(std::shared_ptr<Tensor> gate, std::shared_ptr<Tensor> up);
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> fused_qkv_proj(
    std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> wq,
    std::shared_ptr<Tensor> wk, std::shared_ptr<Tensor> wv);
// === end LT-ENG-022 ===

// === W4-T3 (LT-ENG-028): bilinear, fold, unfold ===
std::shared_ptr<Tensor> bilinear(std::shared_ptr<Tensor> input1, std::shared_ptr<Tensor> input2,
    std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias = nullptr);
std::shared_ptr<Tensor> fold(std::shared_ptr<Tensor> input, const std::vector<int64_t>& output_size,
    const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& dilation,
    const std::vector<int64_t>& padding, const std::vector<int64_t>& stride);
std::shared_ptr<Tensor> unfold(std::shared_ptr<Tensor> input, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& dilation, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& stride);
// === end W4-T3 ===
}
}

#endif
