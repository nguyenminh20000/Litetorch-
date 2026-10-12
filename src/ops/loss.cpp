#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
#include "litetorch/backend_route.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace litetorch {

extern const std::string litetorch_kernels_src;

namespace {
struct StorageUseGuard {
    std::vector<std::shared_ptr<StorageImpl>> storages;
    StorageUseGuard(const std::vector<std::shared_ptr<StorageImpl>>& list) : storages(list) {
        for (auto& s : storages) {
            if (s) s->in_use = true;
        }
    }
    ~StorageUseGuard() {
        for (auto& s : storages) {
            if (s) s->in_use = false;
        }
    }
};

inline std::shared_ptr<Tensor> upcast_fp32(std::shared_ptr<Tensor> t) {
    return (t && t->dtype != DataType::FP32) ? t->cast(DataType::FP32) : t;
}

inline std::vector<int> ce_target_indices(std::shared_ptr<Tensor> target) {
    auto tc = target->is_contiguous() ? target : target->contiguous();
    size_t n = tc->numel();
    size_t esz = tc->storage->element_size();
    const char* base = reinterpret_cast<const char*>(tc->storage->get_cpu_ptr()) + tc->offset * esz;
    std::vector<int> out(n);
    for (size_t i = 0; i < n; ++i) {
        switch (tc->dtype) {
            case DataType::FP32: out[i] = static_cast<int>(reinterpret_cast<const float*>(base)[i]); break;
            case DataType::INT32: out[i] = static_cast<int>(reinterpret_cast<const int32_t*>(base)[i]); break;
            case DataType::INT16: out[i] = static_cast<int>(reinterpret_cast<const int16_t*>(base)[i]); break;
            case DataType::INT8: out[i] = static_cast<int>(reinterpret_cast<const int8_t*>(base)[i]); break;
            default: throw std::runtime_error("[litetorch Error] cross_entropy_loss: target must have integer or float dtype");
        }
    }
    return out;
}

inline void ce_validate_indices(const std::vector<int>& idx, int C, int ignore_idx) {
    for (int t : idx) {
        if (t == ignore_idx) continue;
        if (t < 0 || t >= C) throw std::runtime_error("[litetorch Error] cross_entropy_loss: target index out of bounds");
    }
}

inline void ce_check_same_shape(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target, const char* name) {
    if (input->shape != target->shape) {
        throw std::runtime_error(std::string("[litetorch Error] ") + name + ": input and target must have the same shape");
    }
}
}

class MseLossNode : public Node {
public:
    MseLossNode() : Node("MSELoss") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto target = saved_tensors[1];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto target_c = target->is_contiguous() ? target : target->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto input_w = upcast_fp32(input_c);
        auto target_w = upcast_fp32(target_c);
        auto gout_w = upcast_fp32(gout_c);
        auto grad_w = Tensor::create(input_w->shape, input_w->device, false, DataType::FP32);
        float scale = 2.0f / input_w->numel();
        StorageUseGuard guard({input_w->storage, target_w->storage, gout_w->storage, grad_w->storage});
        if (input_w->device.type == DeviceType::GPU) {
            auto kernel = rt_gpu_get_kernel_by_id(KernelID::MseLossBackward);
            cl_mem in_mem = input_w->gpu_data();
            int in_off = input_w->offset;
            cl_mem tgt_mem = target_w->gpu_data();
            int tgt_off = target_w->offset;
            cl_mem gout_mem = gout_w->gpu_data();
            int gout_off = gout_w->offset;
            cl_mem gin_mem = grad_w->gpu_data();
            int gin_off = grad_w->offset;
            int size = input_w->numel();
            rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                {&in_mem, &in_off, &tgt_mem, &tgt_off, &gout_mem, &gout_off, &gin_mem, &gin_off, &scale, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(int)});
        } else {
            auto diff = Ops::sub(input_w, target_w);
            auto scale_t = Tensor::from_vector({scale * gout_w->item()}, {1}, input_w->device);
            grad_w = Ops::mul(diff, scale_t);
        }
        auto grad_input = grad_w->dtype == input_c->dtype ? grad_w : grad_w->cast(input_c->dtype);
        return { grad_input, nullptr };
    }
};

class CrossEntropyLossNode : public Node {
public:
    std::shared_ptr<Tensor> weight;
    int64_t ignore_index;
    std::string reduction;
    float normalizer;
    float label_smoothing;
    CrossEntropyLossNode(std::shared_ptr<Tensor> weight, int64_t ignore_index,
                         const std::string& reduction, float normalizer, float label_smoothing = 0.0f)
        : Node("CrossEntropyLoss"), weight(weight), ignore_index(ignore_index),
          reduction(reduction), normalizer(normalizer), label_smoothing(label_smoothing) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto target = saved_tensors[1];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto target_c = target->is_contiguous() ? target : target->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto weight_c = weight ? (weight->is_contiguous() ? weight : weight->contiguous()) : nullptr;
        auto input_w = upcast_fp32(input_c);
        auto gout_w = upcast_fp32(gout_c);
        auto weight_w = upcast_fp32(weight_c);

        int N = input_w->shape[0];
        int C = input_w->shape[1];
        bool per_sample = (reduction == "none");
        float norm_inv = (reduction == "mean") ? (1.0f / normalizer) : 1.0f;
        int ignore_idx = static_cast<int>(ignore_index);
        int has_weight = weight_w ? 1 : 0;

        auto t_idx = ce_target_indices(target_c);
        ce_validate_indices(t_idx, C, ignore_idx);
        std::vector<float> t_f(t_idx.size());
        for (size_t i = 0; i < t_idx.size(); ++i) t_f[i] = static_cast<float>(t_idx[i]);
        auto target_w = Tensor::from_vector(t_f, {static_cast<int64_t>(N)}, input_w->device);

        auto grad_w = Tensor::create(input_w->shape, input_w->device, false, DataType::FP32);

        StorageUseGuard guard({input_w->storage, target_w->storage, gout_w->storage, grad_w->storage,
                               weight_w ? weight_w->storage : nullptr});

        if (input_w->device.type == DeviceType::GPU) {
            auto kernel = rt_gpu_get_kernel_by_id(KernelID::CrossEntropyLossBackward);
            cl_mem in_mem = input_w->gpu_data();
            int in_off = input_w->offset;
            cl_mem tgt_mem = target_w->gpu_data();
            int tgt_off = target_w->offset;
            cl_mem w_mem = weight_w ? weight_w->gpu_data() : nullptr;
            int w_off = weight_w ? weight_w->offset : 0;
            cl_mem gout_mem = gout_w->gpu_data();
            int gout_off = gout_w->offset;
            cl_mem gin_mem = grad_w->gpu_data();
            int gin_off = grad_w->offset;
            int per_sample_val = per_sample ? 1 : 0;

            rt_gpu_launch(kernel, {static_cast<size_t>(N)}, {},
                {&in_mem, &in_off, &tgt_mem, &tgt_off, &w_mem, &w_off, &has_weight,
                 &gout_mem, &gout_off, &gin_mem, &gin_off, &N, &C, &ignore_idx, &norm_inv, &per_sample_val},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int),
                 sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(float), sizeof(int)});
        } else {
            float* in_ptr = input_w->data_ptr();
            float* gin_ptr = grad_w->data_ptr();
            float* gout_ptr = gout_w->data_ptr();
            float* w_ptr = weight_w ? weight_w->data_ptr() : nullptr;
            float gout_val = per_sample ? 0.0f : gout_ptr[0];

            ThreadPool::get().parallel_for(0, N, [&](int64_t i) {
                float max_val = in_ptr[i * C];
                for (int j = 1; j < C; ++j) {
                    if (in_ptr[i * C + j] > max_val) max_val = in_ptr[i * C + j];
                }

                float sum = 0.0f;
                std::vector<float> probs(C);
                for (int j = 0; j < C; ++j) {
                    probs[j] = std::exp(in_ptr[i * C + j] - max_val);
                    sum += probs[j];
                }

                int target_idx = t_idx[i];
                float* grow = gin_ptr + i * C;
                if (target_idx == ignore_idx) {
                    std::fill(grow, grow + C, 0.0f);
                    return;
                }
                float wt = w_ptr ? w_ptr[target_idx] : 1.0f;
                float go = per_sample ? gout_ptr[i] : gout_val;
                float eps = label_smoothing;
                float base = (1.0f - eps) * wt;
                float coef = base * norm_inv * go;
                float scoef = eps * norm_inv * go;
                for (int j = 0; j < C; ++j) {
                    probs[j] /= sum;
                    float indicator = (j == target_idx) ? 1.0f : 0.0f;
                    grow[j] = (probs[j] - indicator) * coef + (probs[j] - 1.0f / static_cast<float>(C)) * scoef;
                }
            });

            if (grad_w->device.type == DeviceType::TPU) {
                auto tpu = BackendDispatcher::get().get_tpu_backend();
                if (tpu) tpu->write(grad_w->gpu_data(), grad_w->numel() * sizeof(float), gin_ptr);
            }
        }
        auto grad_input = grad_w->dtype == input_c->dtype ? grad_w : grad_w->cast(input_c->dtype);
        return { grad_input, nullptr };
    }
};

class L1LossNode : public Node {
public:
    L1LossNode() : Node("L1Loss") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto target = saved_tensors[1];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto target_c = target->is_contiguous() ? target : target->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto input_w = upcast_fp32(input_c);
        auto target_w = upcast_fp32(target_c);
        auto gout_w = upcast_fp32(gout_c);
        auto grad_w = Tensor::create(input_w->shape, input_w->device, false, DataType::FP32);
        int size = input_w->numel();
        StorageUseGuard guard({input_w->storage, target_w->storage, gout_w->storage, grad_w->storage});
        if (input_w->device.type == DeviceType::GPU) {
            auto kernel = rt_gpu_get_kernel_by_id(KernelID::L1LossBackward);
            cl_mem in_mem = input_w->gpu_data();
            int in_off = input_w->offset;
            cl_mem tgt_mem = target_w->gpu_data();
            int tgt_off = target_w->offset;
            cl_mem gout_mem = gout_w->gpu_data();
            int gout_off = gout_w->offset;
            cl_mem gin_mem = grad_w->gpu_data();
            int gin_off = grad_w->offset;
            float scale = 1.0f / size;
            rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                {&in_mem, &in_off, &tgt_mem, &tgt_off, &gout_mem, &gout_off, &gin_mem, &gin_off, &scale, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(int)});
        } else {
            float* in_ptr = input_w->data_ptr();
            float* tgt_ptr = target_w->data_ptr();
            float* gin_ptr = grad_w->data_ptr();
            float scale = gout_w->item() / size;
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                float diff = in_ptr[i] - tgt_ptr[i];
                gin_ptr[i] = (diff > 0.0f) ? scale : ((diff < 0.0f) ? -scale : 0.0f);
            });
        }
        auto grad_input = grad_w->dtype == input_c->dtype ? grad_w : grad_w->cast(input_c->dtype);
        return { grad_input, nullptr };
    }
};

class BceLossNode : public Node {
public:
    BceLossNode() : Node("BCELoss") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto target = saved_tensors[1];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto target_c = target->is_contiguous() ? target : target->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto input_w = upcast_fp32(input_c);
        auto target_w = upcast_fp32(target_c);
        auto gout_w = upcast_fp32(gout_c);
        auto grad_w = Tensor::create(input_w->shape, input_w->device, false, DataType::FP32);
        int size = input_w->numel();
        StorageUseGuard guard({input_w->storage, target_w->storage, gout_w->storage, grad_w->storage});
        if (input_w->device.type == DeviceType::GPU) {
            auto kernel = rt_gpu_get_kernel_by_id(KernelID::BceLossBackward);
            cl_mem in_mem = input_w->gpu_data();
            int in_off = input_w->offset;
            cl_mem tgt_mem = target_w->gpu_data();
            int tgt_off = target_w->offset;
            cl_mem gout_mem = gout_w->gpu_data();
            int gout_off = gout_w->offset;
            cl_mem gin_mem = grad_w->gpu_data();
            int gin_off = grad_w->offset;
            float scale = 1.0f / size;
            rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                {&in_mem, &in_off, &tgt_mem, &tgt_off, &gout_mem, &gout_off, &gin_mem, &gin_off, &scale, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(int)});
        } else {
            float* in_ptr = input_w->data_ptr();
            float* tgt_ptr = target_w->data_ptr();
            float* gin_ptr = grad_w->data_ptr();
            float scale = gout_w->item() / size;
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                float x = in_ptr[i];
                float y = tgt_ptr[i];
                if (x < 1e-7f) x = 1e-7f;
                if (x > 1.0f - 1e-7f) x = 1.0f - 1e-7f;
                gin_ptr[i] = scale * (x - y) / (x * (1.0f - x));
            });
        }
        auto grad_input = grad_w->dtype == input_c->dtype ? grad_w : grad_w->cast(input_c->dtype);
        return { grad_input, nullptr };
    }
};

class NLLLossNode : public Node {
public:
    std::shared_ptr<Tensor> weight;
    int64_t ignore_index;
    std::string reduction;
    float normalizer;
    NLLLossNode(std::shared_ptr<Tensor> weight, int64_t ignore_index,
                const std::string& reduction, float normalizer)
        : Node("NLLLoss"), weight(weight), ignore_index(ignore_index),
          reduction(reduction), normalizer(normalizer) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto target = saved_tensors[1];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto target_c = target->is_contiguous() ? target : target->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto weight_c = weight ? (weight->is_contiguous() ? weight : weight->contiguous()) : nullptr;
        auto input_w = upcast_fp32(input_c);
        auto gout_w = upcast_fp32(gout_c);
        auto weight_w = upcast_fp32(weight_c);
        int N = input_w->shape[0];
        int C = input_w->shape[1];
        bool per_sample = (reduction == "none");
        float norm_inv = (reduction == "mean") ? (1.0f / normalizer) : 1.0f;
        int ignore_idx = static_cast<int>(ignore_index);
        auto t_idx = ce_target_indices(target_c);
        ce_validate_indices(t_idx, C, ignore_idx);
        auto grad_w = Tensor::create(input_w->shape, input_w->device, false, DataType::FP32);
        StorageUseGuard guard({input_w->storage, gout_w->storage, grad_w->storage,
                               weight_w ? weight_w->storage : nullptr});
        float* gin_ptr = grad_w->data_ptr();
        float* gout_ptr = gout_w->data_ptr();
        float* w_ptr = weight_w ? weight_w->data_ptr() : nullptr;
        std::fill(gin_ptr, gin_ptr + grad_w->numel(), 0.0f);
        float go_scalar = per_sample ? 0.0f : gout_ptr[0];
        ThreadPool::get().parallel_for(0, N, [&](int64_t i) {
            int t = t_idx[i];
            if (t == ignore_idx) return;
            float wt = w_ptr ? w_ptr[t] : 1.0f;
            float go = per_sample ? gout_ptr[i] : go_scalar;
            gin_ptr[i * C + t] = -wt * norm_inv * go;
        });
        auto grad_input = grad_w->dtype == input_c->dtype ? grad_w : grad_w->cast(input_c->dtype);
        return { grad_input, nullptr };
    }
};

class BCEWithLogitsLossNode : public Node {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> pos_weight;
    std::string reduction;
    BCEWithLogitsLossNode(std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> pos_weight,
                          const std::string& reduction)
        : Node("BCEWithLogitsLoss"), weight(weight), pos_weight(pos_weight), reduction(reduction) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto target = saved_tensors[1];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto target_c = target->is_contiguous() ? target : target->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto weight_c = weight ? (weight->is_contiguous() ? weight : weight->contiguous()) : nullptr;
        auto pos_c = pos_weight ? (pos_weight->is_contiguous() ? pos_weight : pos_weight->contiguous()) : nullptr;
        auto input_w = upcast_fp32(input_c);
        auto target_w = upcast_fp32(target_c);
        auto gout_w = upcast_fp32(gout_c);
        auto weight_w = upcast_fp32(weight_c);
        auto pos_w = upcast_fp32(pos_c);
        int64_t size = input_w->numel();
        bool per_sample = (reduction == "none");
        float norm = (reduction == "mean") ? (1.0f / static_cast<float>(size)) : 1.0f;
        auto grad_w = Tensor::create(input_w->shape, input_w->device, false, DataType::FP32);
        StorageUseGuard guard({input_w->storage, target_w->storage, gout_w->storage, grad_w->storage,
                               weight_w ? weight_w->storage : nullptr, pos_w ? pos_w->storage : nullptr});
        float* in_ptr = input_w->data_ptr();
        float* tgt_ptr = target_w->data_ptr();
        float* gout_ptr = gout_w->data_ptr();
        float* gin_ptr = grad_w->data_ptr();
        float* w_ptr = weight_w ? weight_w->data_ptr() : nullptr;
        float* p_ptr = pos_w ? pos_w->data_ptr() : nullptr;
        int64_t p_numel = pos_w ? static_cast<int64_t>(pos_w->numel()) : 0;
        int64_t last_dim = input_w->shape.empty() ? 1 : input_w->shape.back();
        bool p_broadcast_last = (pos_w && p_numel > 1 && p_numel == last_dim && pos_w->shape.size() == 1);
        float go_scalar = per_sample ? 0.0f : gout_ptr[0];
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            float x = in_ptr[i];
            float y = tgt_ptr[i];
            float p = 1.0f;
            if (p_ptr) {
                if (p_numel == 1) p = p_ptr[0];
                else if (p_broadcast_last) p = p_ptr[i % last_dim];
                else p = p_ptr[i];
            }
            float s = 1.0f / (1.0f + std::exp(-x));
            float d = s * (1.0f + (p - 1.0f) * y) - p * y;
            if (w_ptr) d *= (weight_w->numel() == 1 ? w_ptr[0] : w_ptr[i]);
            float go = per_sample ? gout_ptr[i] : go_scalar;
            gin_ptr[i] = d * norm * go;
        });
        auto grad_input = grad_w->dtype == input_c->dtype ? grad_w : grad_w->cast(input_c->dtype);
        return { grad_input, nullptr };
    }
};

namespace Ops {

std::shared_ptr<Tensor> l1_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    if (input->device != target->device) {
        target = target->to(input->device);
    }
    ce_check_same_shape(input, target, "l1_loss");
    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto target_c = target->is_contiguous() ? target : target->contiguous();
    auto input_w = upcast_fp32(input_c);
    auto target_w = upcast_fp32(target_c);
    auto out = Tensor::create({1}, input_w->device);
    StorageUseGuard guard({input_w->storage, target_w->storage, out->storage});
    size_t size = input_w->numel();
    float inv_n = 1.0f / size;

    if (input_w->device.type == DeviceType::GPU) {
        auto kernel = rt_gpu_get_kernel_by_id(KernelID::L1LossForward);
        cl_mem in_mem = input_w->gpu_data();
        int in_off = input_w->offset;
        cl_mem tgt_mem = target_w->gpu_data();
        int tgt_off = target_w->offset;
        cl_mem out_mem = out->gpu_data();
        int out_off = out->offset;
        int size_val = size;
        float zero = 0.0f;
        rt_gpu_write(out_mem, sizeof(float), &zero, out_off);
        size_t blocks = std::min(static_cast<size_t>(32), (size + 255) / 256);
        if (blocks < 1) blocks = 1;
        rt_gpu_launch(kernel, {blocks * 256}, {256},
            {&in_mem, &in_off, &tgt_mem, &tgt_off, &out_mem, &out_off, &size_val, &inv_n},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(float)});
    } else {
        float* in_ptr = input_w->data_ptr();
        float* tgt_ptr = target_w->data_ptr();
        float total = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            total += std::abs(in_ptr[i] - tgt_ptr[i]);
        }
        out->data_ptr()[0] = total * inv_n;
    }

    if (Autograd::is_grad_enabled() && (input->requires_grad)) {
        auto node = std::make_shared<L1LossNode>();
        node->inputs = { {input, true}, {target, false} };
        node->next_nodes = { input->creator, nullptr };
        node->saved_tensors = { input, target };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> bce_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    if (input->device != target->device) {
        target = target->to(input->device);
    }
    ce_check_same_shape(input, target, "bce_loss");
    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto target_c = target->is_contiguous() ? target : target->contiguous();
    auto input_w = upcast_fp32(input_c);
    auto target_w = upcast_fp32(target_c);
    auto out = Tensor::create({1}, input_w->device);
    StorageUseGuard guard({input_w->storage, target_w->storage, out->storage});
    size_t size = input_w->numel();
    float inv_n = 1.0f / size;

    if (input_w->device.type == DeviceType::GPU) {
        auto kernel = rt_gpu_get_kernel_by_id(KernelID::BceLossForward);
        cl_mem in_mem = input_w->gpu_data();
        int in_off = input_w->offset;
        cl_mem tgt_mem = target_w->gpu_data();
        int tgt_off = target_w->offset;
        cl_mem out_mem = out->gpu_data();
        int out_off = out->offset;
        int size_val = size;
        float zero = 0.0f;
        rt_gpu_write(out_mem, sizeof(float), &zero, out_off);
        size_t blocks = std::min(static_cast<size_t>(32), (size + 255) / 256);
        if (blocks < 1) blocks = 1;
        rt_gpu_launch(kernel, {blocks * 256}, {256},
            {&in_mem, &in_off, &tgt_mem, &tgt_off, &out_mem, &out_off, &size_val, &inv_n},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(float)});
    } else {
        float* in_ptr = input_w->data_ptr();
        float* tgt_ptr = target_w->data_ptr();
        float total_loss = 0.0f;

        std::vector<float> losses(size, 0.0f);
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            float x = in_ptr[i];
            float y = tgt_ptr[i];
            if (x < 1e-7f) x = 1e-7f;
            if (x > 1.0f - 1e-7f) x = 1.0f - 1e-7f;
            losses[i] = -(y * std::log(x) + (1.0f - y) * std::log(1.0f - x));
        });

        for (size_t i = 0; i < size; ++i) {
            total_loss += losses[i];
        }
        out->data_ptr()[0] = total_loss / size;
    }

    if (Autograd::is_grad_enabled() && (input->requires_grad)) {
        auto node = std::make_shared<BceLossNode>();
        node->inputs = { {input, true}, {target, false} };
        node->next_nodes = { input->creator, nullptr };
        node->saved_tensors = { input, target };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    if (input->device != target->device) {
        target = target->to(input->device);
    }
    ce_check_same_shape(input, target, "mse_loss");
    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto target_c = target->is_contiguous() ? target : target->contiguous();
    auto input_w = upcast_fp32(input_c);
    auto target_w = upcast_fp32(target_c);
    auto out = Tensor::create({1}, input_w->device);
    StorageUseGuard guard({input_w->storage, target_w->storage, out->storage});
    size_t size = input_w->numel();
    float inv_n = 1.0f / size;

    if (input_w->device.type == DeviceType::GPU) {
        auto kernel = rt_gpu_get_kernel_by_id(KernelID::MseLossForward);
        cl_mem in_mem = input_w->gpu_data();
        int in_off = input_w->offset;
        cl_mem tgt_mem = target_w->gpu_data();
        int tgt_off = target_w->offset;
        cl_mem out_mem = out->gpu_data();
        int out_off = out->offset;
        int size_val = size;
        float zero = 0.0f;
        rt_gpu_write(out_mem, sizeof(float), &zero, out_off);
        size_t blocks = std::min(static_cast<size_t>(32), (size + 255) / 256);
        if (blocks < 1) blocks = 1;
        rt_gpu_launch(kernel, {blocks * 256}, {256},
            {&in_mem, &in_off, &tgt_mem, &tgt_off, &out_mem, &out_off, &size_val, &inv_n},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(float)});
    } else {
        float* in_ptr = input_w->data_ptr();
        float* tgt_ptr = target_w->data_ptr();
        float total = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            float d = in_ptr[i] - tgt_ptr[i];
            total += d * d;
        }
        out->data_ptr()[0] = total * inv_n;
    }

    if (Autograd::is_grad_enabled() && (input->requires_grad)) {
        auto node = std::make_shared<MseLossNode>();
        node->inputs = { {input, true}, {target, false} };
        node->next_nodes = { input->creator, nullptr };
        node->saved_tensors = { input, target };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
    std::shared_ptr<Tensor> weight, int64_t ignore_index, const std::string& reduction, float label_smoothing) {
    if (reduction != "mean" && reduction != "sum" && reduction != "none") {
        throw std::invalid_argument("[litetorch Error] cross_entropy_loss reduction must be 'mean', 'sum' or 'none'");
    }
    if (label_smoothing < 0.0f || label_smoothing > 1.0f) {
        throw std::invalid_argument("[litetorch Error] cross_entropy_loss label_smoothing must be in [0, 1]");
    }
    if (input->device != target->device) {
        target = target->to(input->device);
    }
    if (weight && weight->device != input->device) {
        weight = weight->to(input->device);
    }
    if (input->shape.size() != 2 || target->shape.size() != 1) {
        throw std::runtime_error("[litetorch Error] CrossEntropyLoss requires 2D input (logits) and 1D target");
    }

    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto target_c = target->is_contiguous() ? target : target->contiguous();
    auto weight_c = weight ? (weight->is_contiguous() ? weight : weight->contiguous()) : nullptr;
    auto input_w = upcast_fp32(input_c);
    auto weight_w = upcast_fp32(weight_c);
    int N = input_w->shape[0];
    int C = input_w->shape[1];
    if (weight_c && (weight_c->shape.size() != 1 || weight_c->shape[0] != C)) {
        throw std::runtime_error("[litetorch Error] CrossEntropyLoss weight must be 1D with num_classes elements");
    }
    int ignore_idx = static_cast<int>(ignore_index);
    bool per_sample = (reduction == "none");

    auto t_idx = ce_target_indices(target_c);
    if (static_cast<int>(t_idx.size()) != N) {
        throw std::runtime_error("[litetorch Error] CrossEntropyLoss target batch size must match input");
    }
    ce_validate_indices(t_idx, C, ignore_idx);
    std::vector<float> t_f(t_idx.size());
    for (size_t i = 0; i < t_idx.size(); ++i) t_f[i] = static_cast<float>(t_idx[i]);
    auto target_w = Tensor::from_vector(t_f, {static_cast<int64_t>(N)}, input_w->device);

    auto out = Tensor::create(per_sample ? std::vector<int64_t>{N} : std::vector<int64_t>{1}, input_w->device);
    StorageUseGuard guard({input_w->storage, target_w->storage, out->storage,
                           weight_w ? weight_w->storage : nullptr});

    float normalizer = 0.0f;
    if (reduction == "mean") {
        std::vector<float> h_w;
        if (weight_w) {
            h_w.resize(C);
            if (input_w->device.type == DeviceType::GPU) {
                cl_mem w_mem = weight_w->gpu_data();
                rt_gpu_read(w_mem, C * sizeof(float), h_w.data(), static_cast<size_t>(weight_w->offset) * sizeof(float));
            } else {
                float* w_ptr = weight_w->data_ptr();
                h_w.assign(w_ptr, w_ptr + C);
            }
        }
        for (int i = 0; i < N; ++i) {
            int t = t_idx[i];
            if (t == ignore_idx) continue;
            normalizer += weight_w ? h_w[t] : 1.0f;
        }
    }
    float norm_inv = (reduction == "mean") ? (1.0f / normalizer) : 1.0f;

    if (input_w->device.type == DeviceType::GPU && label_smoothing != 0.0f) {
        throw std::runtime_error("[litetorch Error] cross_entropy_loss: label_smoothing on GPU is not implemented");
    }

    if (input_w->device.type == DeviceType::GPU) {
        auto kernel = rt_gpu_get_kernel_by_id(KernelID::CrossEntropyLossForward);
        cl_mem in_mem = input_w->gpu_data();
        int in_off = input_w->offset;
        cl_mem tgt_mem = target_w->gpu_data();
        int tgt_off = target_w->offset;
        cl_mem w_mem = weight_w ? weight_w->gpu_data() : nullptr;
        int w_off = weight_w ? weight_w->offset : 0;
        int has_weight = weight_w ? 1 : 0;
        cl_mem out_mem = out->gpu_data();
        int out_off = out->offset;
        int per_sample_val = per_sample ? 1 : 0;
        if (!per_sample) {
            float zero = 0.0f;
            rt_gpu_write(out_mem, sizeof(float), &zero, static_cast<size_t>(out_off) * sizeof(float));
        }
        rt_gpu_launch(kernel, {static_cast<size_t>(N)}, {},
            {&in_mem, &in_off, &tgt_mem, &tgt_off, &w_mem, &w_off, &has_weight,
             &out_mem, &out_off, &N, &C, &ignore_idx, &norm_inv, &per_sample_val},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int),
             sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(float), sizeof(int)});
    } else {
        float* in_ptr = input_w->data_ptr();
        float* w_ptr = weight_w ? weight_w->data_ptr() : nullptr;
        float* out_ptr = out->data_ptr();

        std::vector<float> losses(N, 0.0f);
        ThreadPool::get().parallel_for(0, N, [&](int64_t i) {
            int target_idx = t_idx[i];
            if (target_idx == ignore_idx) {
                losses[i] = 0.0f;
                return;
            }
            float max_val = in_ptr[i * C];
            for (int j = 1; j < C; ++j) {
                if (in_ptr[i * C + j] > max_val) max_val = in_ptr[i * C + j];
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < C; ++j) {
                sum_exp += std::exp(in_ptr[i * C + j] - max_val);
            }
            float lse = max_val + std::log(sum_exp);
            float correct_logit = in_ptr[i * C + target_idx];
            float wt = w_ptr ? w_ptr[target_idx] : 1.0f;
            float loss = (1.0f - label_smoothing) * wt * (lse - correct_logit);
            if (label_smoothing != 0.0f) {
                float mean_logit = 0.0f;
                for (int j = 0; j < C; ++j) mean_logit += in_ptr[i * C + j];
                mean_logit /= static_cast<float>(C);
                loss += label_smoothing * (lse - mean_logit);
            }
            losses[i] = loss;
        });

        if (per_sample) {
            std::copy(losses.begin(), losses.end(), out_ptr);
        } else {
            float total_loss = 0.0f;
            for (int i = 0; i < N; ++i) total_loss += losses[i];
            out_ptr[0] = total_loss * norm_inv;
        }
        if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), out->numel() * sizeof(float), out_ptr);
        }
    }

    if (Autograd::is_grad_enabled() && (input->requires_grad)) {
        auto node = std::make_shared<CrossEntropyLossNode>(weight_c, ignore_index, reduction, normalizer, label_smoothing);
        node->inputs = { {input, true}, {target, false} };
        node->next_nodes = { input->creator, nullptr };
        node->saved_tensors = { input, target };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> nll_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
    std::shared_ptr<Tensor> weight, int64_t ignore_index, const std::string& reduction) {
    if (reduction != "mean" && reduction != "sum" && reduction != "none") {
        throw std::invalid_argument("[litetorch Error] nll_loss reduction must be 'mean', 'sum' or 'none'");
    }
    if (input->device.type == DeviceType::GPU) {
        throw std::runtime_error("[litetorch Error] Ops::nll_loss: GPU kernel not implemented; use CPU tensors (deferred to Phase 3)");
    }
    if (input->device != target->device) {
        target = target->to(input->device);
    }
    if (weight && weight->device != input->device) {
        weight = weight->to(input->device);
    }
    if (input->shape.size() != 2 || target->shape.size() != 1) {
        throw std::runtime_error("[litetorch Error] NLLLoss requires 2D input (log-probs) and 1D target");
    }
    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto target_c = target->is_contiguous() ? target : target->contiguous();
    auto weight_c = weight ? (weight->is_contiguous() ? weight : weight->contiguous()) : nullptr;
    auto input_w = upcast_fp32(input_c);
    auto weight_w = upcast_fp32(weight_c);
    int N = input_w->shape[0];
    int C = input_w->shape[1];
    if (weight_c && (weight_c->shape.size() != 1 || weight_c->shape[0] != C)) {
        throw std::runtime_error("[litetorch Error] NLLLoss weight must be 1D with num_classes elements");
    }
    int ignore_idx = static_cast<int>(ignore_index);
    bool per_sample = (reduction == "none");
    auto t_idx = ce_target_indices(target_c);
    if (static_cast<int>(t_idx.size()) != N) {
        throw std::runtime_error("[litetorch Error] NLLLoss target batch size must match input");
    }
    ce_validate_indices(t_idx, C, ignore_idx);
    auto out = Tensor::create(per_sample ? std::vector<int64_t>{N} : std::vector<int64_t>{1}, input_w->device);
    StorageUseGuard guard({input_w->storage, out->storage, weight_w ? weight_w->storage : nullptr});
    float normalizer = 0.0f;
    float* w_ptr = weight_w ? weight_w->data_ptr() : nullptr;
    if (reduction == "mean") {
        for (int i = 0; i < N; ++i) {
            int t = t_idx[i];
            if (t == ignore_idx) continue;
            normalizer += w_ptr ? w_ptr[t] : 1.0f;
        }
    }
    float norm_inv = (reduction == "mean") ? (1.0f / normalizer) : 1.0f;
    float* in_ptr = input_w->data_ptr();
    float* out_ptr = out->data_ptr();
    std::vector<float> losses(N, 0.0f);
    ThreadPool::get().parallel_for(0, N, [&](int64_t i) {
        int t = t_idx[i];
        if (t == ignore_idx) {
            losses[i] = 0.0f;
            return;
        }
        float wt = w_ptr ? w_ptr[t] : 1.0f;
        losses[i] = -wt * in_ptr[i * C + t];
    });
    if (per_sample) {
        std::copy(losses.begin(), losses.end(), out_ptr);
    } else {
        float total = 0.0f;
        for (int i = 0; i < N; ++i) total += losses[i];
        out_ptr[0] = total * norm_inv;
    }
    if (Autograd::is_grad_enabled() && (input->requires_grad)) {
        auto node = std::make_shared<NLLLossNode>(weight_c, ignore_index, reduction, normalizer);
        node->inputs = { {input, true}, {target, false} };
        node->next_nodes = { input->creator, nullptr };
        node->saved_tensors = { input, target };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> bce_with_logits_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
    std::shared_ptr<Tensor> weight, const std::string& reduction, std::shared_ptr<Tensor> pos_weight) {
    if (reduction != "mean" && reduction != "sum" && reduction != "none") {
        throw std::invalid_argument("[litetorch Error] bce_with_logits_loss reduction must be 'mean', 'sum' or 'none'");
    }
    if (input->device.type == DeviceType::GPU) {
        throw std::runtime_error("[litetorch Error] Ops::bce_with_logits_loss: GPU kernel not implemented; use CPU tensors (deferred to Phase 3)");
    }
    if (input->device != target->device) {
        target = target->to(input->device);
    }
    if (weight && weight->device != input->device) {
        weight = weight->to(input->device);
    }
    if (pos_weight && pos_weight->device != input->device) {
        pos_weight = pos_weight->to(input->device);
    }
    ce_check_same_shape(input, target, "bce_with_logits_loss");
    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto target_c = target->is_contiguous() ? target : target->contiguous();
    auto weight_c = weight ? (weight->is_contiguous() ? weight : weight->contiguous()) : nullptr;
    auto pos_c = pos_weight ? (pos_weight->is_contiguous() ? pos_weight : pos_weight->contiguous()) : nullptr;
    auto input_w = upcast_fp32(input_c);
    auto target_w = upcast_fp32(target_c);
    auto weight_w = upcast_fp32(weight_c);
    auto pos_w = upcast_fp32(pos_c);
    int64_t size = input_w->numel();
    bool per_sample = (reduction == "none");
    if (weight_w && weight_w->numel() != 1 && weight_w->numel() != static_cast<size_t>(size)) {
        throw std::runtime_error("[litetorch Error] bce_with_logits_loss: weight must be scalar or match input numel");
    }
    int64_t p_numel = pos_w ? static_cast<int64_t>(pos_w->numel()) : 0;
    int64_t last_dim = input_w->shape.empty() ? 1 : input_w->shape.back();
    if (pos_w && !(p_numel == 1 || p_numel == size || (pos_w->shape.size() == 1 && p_numel == last_dim))) {
        throw std::runtime_error("[litetorch Error] bce_with_logits_loss: pos_weight must be scalar, 1D with last-dim size, or match input numel");
    }
    bool p_broadcast_last = (pos_w && p_numel > 1 && p_numel == last_dim && pos_w->shape.size() == 1);
    auto out = Tensor::create(per_sample ? input_w->shape : std::vector<int64_t>{1}, input_w->device);
    StorageUseGuard guard({input_w->storage, target_w->storage, out->storage,
                           weight_w ? weight_w->storage : nullptr, pos_w ? pos_w->storage : nullptr});
    float* in_ptr = input_w->data_ptr();
    float* tgt_ptr = target_w->data_ptr();
    float* out_ptr = out->data_ptr();
    float* w_ptr = weight_w ? weight_w->data_ptr() : nullptr;
    float* p_ptr = pos_w ? pos_w->data_ptr() : nullptr;
    std::vector<float> losses(size, 0.0f);
    ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
        float x = in_ptr[i];
        float y = tgt_ptr[i];
        float p = 1.0f;
        if (p_ptr) {
            if (p_numel == 1) p = p_ptr[0];
            else if (p_broadcast_last) p = p_ptr[i % last_dim];
            else p = p_ptr[i];
        }
        float ax = std::abs(x);
        float max_neg = (x < 0.0f) ? -x : 0.0f;
        float log_term = std::log1p(std::exp(-ax));
        float pw = 1.0f + (p - 1.0f) * y;
        float li = (1.0f - y) * x + pw * (max_neg + log_term);
        if (w_ptr) li *= (weight_w->numel() == 1 ? w_ptr[0] : w_ptr[i]);
        losses[i] = li;
    });
    if (per_sample) {
        std::copy(losses.begin(), losses.end(), out_ptr);
    } else {
        float total = 0.0f;
        for (int64_t i = 0; i < size; ++i) total += losses[i];
        out_ptr[0] = (reduction == "mean") ? total / static_cast<float>(size) : total;
    }
    if (Autograd::is_grad_enabled() && (input->requires_grad)) {
        auto node = std::make_shared<BCEWithLogitsLossNode>(weight_c, pos_c, reduction);
        node->inputs = { {input, true}, {target, false} };
        node->next_nodes = { input->creator, nullptr };
        node->saved_tensors = { input, target };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

}
}
