#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
#include "litetorch/backend_route.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

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
        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        float scale = 2.0f / input_c->numel();
        StorageUseGuard guard({input_c->storage, target_c->storage, grad_output->storage, grad_input->storage});
        if (input_c->device.type == DeviceType::GPU) {
            auto kernel = rt_gpu_get_kernel_by_id(KernelID::MseLossBackward);
            cl_mem in_mem = input_c->gpu_data();
            int in_off = input_c->offset;
            cl_mem tgt_mem = target_c->gpu_data();
            int tgt_off = target_c->offset;
            cl_mem gout_mem = grad_output->gpu_data();
            int gout_off = grad_output->offset;
            cl_mem gin_mem = grad_input->gpu_data();
            int gin_off = grad_input->offset;
            int size = input_c->numel();
            rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                {&in_mem, &in_off, &tgt_mem, &tgt_off, &gout_mem, &gout_off, &gin_mem, &gin_off, &scale, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(int)});
        } else {
            auto diff = Ops::sub(input_c, target_c);
            auto scale_t = Tensor::from_vector({scale * grad_output->item()}, {1}, input_c->device);
            grad_input = Ops::mul(diff, scale_t);
        }
        return { grad_input, nullptr };
    }
};

class CrossEntropyLossNode : public Node {
public:
    std::shared_ptr<Tensor> weight;
    int64_t ignore_index;
    std::string reduction;
    float normalizer;
    CrossEntropyLossNode(std::shared_ptr<Tensor> weight, int64_t ignore_index,
                         const std::string& reduction, float normalizer)
        : Node("CrossEntropyLoss"), weight(weight), ignore_index(ignore_index),
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
        auto grad_input = Tensor::create(input_c->shape, input_c->device);

        int N = input_c->shape[0];
        int C = input_c->shape[1];
        bool per_sample = (reduction == "none");
        float norm_inv = (reduction == "mean") ? (1.0f / normalizer) : 1.0f;
        int ignore_idx = static_cast<int>(ignore_index);
        int has_weight = weight_c ? 1 : 0;

        StorageUseGuard guard({input_c->storage, target_c->storage, gout_c->storage, grad_input->storage,
                               weight_c ? weight_c->storage : nullptr});

        if (input_c->device.type == DeviceType::GPU) {
            auto kernel = rt_gpu_get_kernel_by_id(KernelID::CrossEntropyLossBackward);
            cl_mem in_mem = input_c->gpu_data();
            int in_off = input_c->offset;
            cl_mem tgt_mem = target_c->gpu_data();
            int tgt_off = target_c->offset;
            cl_mem w_mem = weight_c ? weight_c->gpu_data() : nullptr;
            int w_off = weight_c ? weight_c->offset : 0;
            cl_mem gout_mem = gout_c->gpu_data();
            int gout_off = gout_c->offset;
            cl_mem gin_mem = grad_input->gpu_data();
            int gin_off = grad_input->offset;
            int per_sample_val = per_sample ? 1 : 0;

            rt_gpu_launch(kernel, {static_cast<size_t>(N)}, {},
                {&in_mem, &in_off, &tgt_mem, &tgt_off, &w_mem, &w_off, &has_weight,
                 &gout_mem, &gout_off, &gin_mem, &gin_off, &N, &C, &ignore_idx, &norm_inv, &per_sample_val},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int),
                 sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(float), sizeof(int)});
        } else {
            float* in_ptr = input_c->data_ptr();
            float* tgt_ptr = target_c->data_ptr();
            float* gin_ptr = grad_input->data_ptr();
            float* gout_ptr = gout_c->data_ptr();
            float* w_ptr = weight_c ? weight_c->data_ptr() : nullptr;
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

                int target_idx = static_cast<int>(tgt_ptr[i]);
                float* grow = gin_ptr + i * C;
                if (target_idx == ignore_idx) {
                    std::fill(grow, grow + C, 0.0f);
                    return;
                }
                float wt = w_ptr ? w_ptr[target_idx] : 1.0f;
                float go = per_sample ? gout_ptr[i] : gout_val;
                float scale = wt * norm_inv * go;
                for (int j = 0; j < C; ++j) {
                    probs[j] /= sum;
                    float indicator = (j == target_idx) ? 1.0f : 0.0f;
                    grow[j] = (probs[j] - indicator) * scale;
                }
            });

            if (grad_input->device.type == DeviceType::TPU) {
                auto tpu = BackendDispatcher::get().get_tpu_backend();
                if (tpu) tpu->write(grad_input->gpu_data(), grad_input->numel() * sizeof(float), gin_ptr);
            }
        }
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
        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        int size = input_c->numel();
        StorageUseGuard guard({input_c->storage, target_c->storage, grad_output->storage, grad_input->storage});
        if (input_c->device.type == DeviceType::GPU) {
            auto kernel = rt_gpu_get_kernel_by_id(KernelID::L1LossBackward);
            cl_mem in_mem = input_c->gpu_data();
            int in_off = input_c->offset;
            cl_mem tgt_mem = target_c->gpu_data();
            int tgt_off = target_c->offset;
            cl_mem gout_mem = grad_output->gpu_data();
            int gout_off = grad_output->offset;
            cl_mem gin_mem = grad_input->gpu_data();
            int gin_off = grad_input->offset;
            float scale = 1.0f / size;
            rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                {&in_mem, &in_off, &tgt_mem, &tgt_off, &gout_mem, &gout_off, &gin_mem, &gin_off, &scale, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(int)});
        } else {
            float* in_ptr = input_c->data_ptr();
            float* tgt_ptr = target_c->data_ptr();
            float* gin_ptr = grad_input->data_ptr();
            float scale = grad_output->item() / size;
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                float diff = in_ptr[i] - tgt_ptr[i];
                gin_ptr[i] = (diff > 0.0f) ? scale : ((diff < 0.0f) ? -scale : 0.0f);
            });
        }
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
        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        int size = input_c->numel();
        StorageUseGuard guard({input_c->storage, target_c->storage, grad_output->storage, grad_input->storage});
        if (input_c->device.type == DeviceType::GPU) {
            auto kernel = rt_gpu_get_kernel_by_id(KernelID::BceLossBackward);
            cl_mem in_mem = input_c->gpu_data();
            int in_off = input_c->offset;
            cl_mem tgt_mem = target_c->gpu_data();
            int tgt_off = target_c->offset;
            cl_mem gout_mem = grad_output->gpu_data();
            int gout_off = grad_output->offset;
            cl_mem gin_mem = grad_input->gpu_data();
            int gin_off = grad_input->offset;
            float scale = 1.0f / size;
            rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                {&in_mem, &in_off, &tgt_mem, &tgt_off, &gout_mem, &gout_off, &gin_mem, &gin_off, &scale, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(int)});
        } else {
            float* in_ptr = input_c->data_ptr();
            float* tgt_ptr = target_c->data_ptr();
            float* gin_ptr = grad_input->data_ptr();
            float scale = grad_output->item() / size;
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                float x = in_ptr[i];
                float y = tgt_ptr[i];
                if (x < 1e-7f) x = 1e-7f;
                if (x > 1.0f - 1e-7f) x = 1.0f - 1e-7f;
                gin_ptr[i] = scale * (x - y) / (x * (1.0f - x));
            });
        }
        return { grad_input, nullptr };
    }
};

namespace Ops {

std::shared_ptr<Tensor> l1_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    if (input->device != target->device) {
        target = target->to(input->device);
    }
    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto target_c = target->is_contiguous() ? target : target->contiguous();
    auto out = Tensor::create({1}, input_c->device);
    StorageUseGuard guard({input_c->storage, target_c->storage, out->storage});
    size_t size = input_c->numel();
    float inv_n = 1.0f / size;

    if (input_c->device.type == DeviceType::GPU) {
        auto kernel = rt_gpu_get_kernel_by_id(KernelID::L1LossForward);
        cl_mem in_mem = input_c->gpu_data();
        int in_off = input_c->offset;
        cl_mem tgt_mem = target_c->gpu_data();
        int tgt_off = target_c->offset;
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
        float* in_ptr = input_c->data_ptr();
        float* tgt_ptr = target_c->data_ptr();
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
    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto target_c = target->is_contiguous() ? target : target->contiguous();
    auto out = Tensor::create({1}, input_c->device);
    StorageUseGuard guard({input_c->storage, target_c->storage, out->storage});
    size_t size = input_c->numel();
    float inv_n = 1.0f / size;

    if (input_c->device.type == DeviceType::GPU) {
        auto kernel = rt_gpu_get_kernel_by_id(KernelID::BceLossForward);
        cl_mem in_mem = input_c->gpu_data();
        int in_off = input_c->offset;
        cl_mem tgt_mem = target_c->gpu_data();
        int tgt_off = target_c->offset;
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
        float* in_ptr = input_c->data_ptr();
        float* tgt_ptr = target_c->data_ptr();
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
    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto target_c = target->is_contiguous() ? target : target->contiguous();
    auto out = Tensor::create({1}, input_c->device);
    StorageUseGuard guard({input_c->storage, target_c->storage, out->storage});
    size_t size = input_c->numel();
    float inv_n = 1.0f / size;

    if (input_c->device.type == DeviceType::GPU) {
        auto kernel = rt_gpu_get_kernel_by_id(KernelID::MseLossForward);
        cl_mem in_mem = input_c->gpu_data();
        int in_off = input_c->offset;
        cl_mem tgt_mem = target_c->gpu_data();
        int tgt_off = target_c->offset;
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
        float* in_ptr = input_c->data_ptr();
        float* tgt_ptr = target_c->data_ptr();
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
    std::shared_ptr<Tensor> weight, int64_t ignore_index, const std::string& reduction) {
    if (reduction != "mean" && reduction != "sum" && reduction != "none") {
        throw std::invalid_argument("[litetorch Error] cross_entropy_loss reduction must be 'mean', 'sum' or 'none'");
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
    int N = input_c->shape[0];
    int C = input_c->shape[1];
    if (weight_c && (weight_c->shape.size() != 1 || weight_c->shape[0] != C)) {
        throw std::runtime_error("[litetorch Error] CrossEntropyLoss weight must be 1D with num_classes elements");
    }
    int ignore_idx = static_cast<int>(ignore_index);
    bool per_sample = (reduction == "none");

    auto out = Tensor::create(per_sample ? std::vector<int64_t>{N} : std::vector<int64_t>{1}, input_c->device);
    StorageUseGuard guard({input_c->storage, target_c->storage, out->storage,
                           weight_c ? weight_c->storage : nullptr});

    float normalizer = 0.0f;
    if (reduction == "mean") {
        std::vector<float> h_tgt(N);
        std::vector<float> h_w;
        if (input_c->device.type == DeviceType::GPU) {
            cl_mem tgt_mem = target_c->gpu_data();
            rt_gpu_read(tgt_mem, N * sizeof(float), h_tgt.data(), static_cast<size_t>(target_c->offset) * sizeof(float));
            if (weight_c) {
                h_w.resize(C);
                cl_mem w_mem = weight_c->gpu_data();
                rt_gpu_read(w_mem, C * sizeof(float), h_w.data(), static_cast<size_t>(weight_c->offset) * sizeof(float));
            }
        } else {
            float* tgt_ptr = target_c->data_ptr();
            std::copy(tgt_ptr, tgt_ptr + N, h_tgt.begin());
            if (weight_c) {
                float* w_ptr = weight_c->data_ptr();
                h_w.assign(w_ptr, w_ptr + C);
            }
        }
        for (int i = 0; i < N; ++i) {
            int t = static_cast<int>(h_tgt[i]);
            if (t == ignore_idx) continue;
            normalizer += weight_c ? h_w[t] : 1.0f;
        }
    }
    float norm_inv = (reduction == "mean") ? (1.0f / normalizer) : 1.0f;

    if (input_c->device.type == DeviceType::GPU) {
        auto kernel = rt_gpu_get_kernel_by_id(KernelID::CrossEntropyLossForward);
        cl_mem in_mem = input_c->gpu_data();
        int in_off = input_c->offset;
        cl_mem tgt_mem = target_c->gpu_data();
        int tgt_off = target_c->offset;
        cl_mem w_mem = weight_c ? weight_c->gpu_data() : nullptr;
        int w_off = weight_c ? weight_c->offset : 0;
        int has_weight = weight_c ? 1 : 0;
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
        float* in_ptr = input_c->data_ptr();
        float* tgt_ptr = target_c->data_ptr();
        float* w_ptr = weight_c ? weight_c->data_ptr() : nullptr;
        float* out_ptr = out->data_ptr();

        std::vector<float> losses(N, 0.0f);
        ThreadPool::get().parallel_for(0, N, [&](int64_t i) {
            int target_idx = static_cast<int>(tgt_ptr[i]);
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
            float correct_logit = in_ptr[i * C + target_idx];
            float wt = w_ptr ? w_ptr[target_idx] : 1.0f;
            losses[i] = wt * (-correct_logit + max_val + std::log(sum_exp));
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
        auto node = std::make_shared<CrossEntropyLossNode>(weight_c, ignore_index, reduction, normalizer);
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
