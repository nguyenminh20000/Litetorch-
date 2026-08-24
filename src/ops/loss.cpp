#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
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
        auto input = saved_tensors[0];
        auto target = saved_tensors[1];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto target_c = target->is_contiguous() ? target : target->contiguous();
        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        float scale = 2.0f / input_c->numel();
        StorageUseGuard guard({input_c->storage, target_c->storage, grad_output->storage, grad_input->storage});
        if (input_c->device.type == DeviceType::GPU) {
            auto kernel = CLBackend::get().get_kernel(KernelID::MseLossBackward);
            cl_mem in_mem = input_c->gpu_data();
            int in_off = input_c->offset;
            cl_mem tgt_mem = target_c->gpu_data();
            int tgt_off = target_c->offset;
            cl_mem gout_mem = grad_output->gpu_data();
            int gout_off = grad_output->offset;
            cl_mem gin_mem = grad_input->gpu_data();
            int gin_off = grad_input->offset;
            int size = input_c->numel();
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
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
    CrossEntropyLossNode() : Node("CrossEntropyLoss") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto input = saved_tensors[0];
        auto target = saved_tensors[1];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto target_c = target->is_contiguous() ? target : target->contiguous();
        auto grad_input = Tensor::create(input_c->shape, input_c->device);

        int N = input_c->shape[0];
        int C = input_c->shape[1];

        StorageUseGuard guard({input_c->storage, target_c->storage, grad_output->storage, grad_input->storage});

        if (input_c->device.type == DeviceType::GPU) {
            auto kernel = CLBackend::get().get_kernel(KernelID::CrossEntropyLossBackward);
            cl_mem in_mem = input_c->gpu_data();
            int in_off = input_c->offset;
            cl_mem tgt_mem = target_c->gpu_data();
            int tgt_off = target_c->offset;
            cl_mem gout_mem = grad_output->gpu_data();
            int gout_off = grad_output->offset;
            cl_mem gin_mem = grad_input->gpu_data();
            int gin_off = grad_input->offset;

            CLBackend::get().launch(kernel, {static_cast<size_t>(N)}, {},
                {&in_mem, &in_off, &tgt_mem, &tgt_off, &gout_mem, &gout_off, &gin_mem, &gin_off, &N, &C},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int)});
        } else {
            float* in_ptr = input_c->data_ptr();
            float* tgt_ptr = target_c->data_ptr();
            float* gin_ptr = grad_input->data_ptr();
            float gout_val = grad_output->item();
            
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
                for (int j = 0; j < C; ++j) {
                    probs[j] /= sum;
                    float indicator = (j == target_idx) ? 1.0f : 0.0f;
                    gin_ptr[i * C + j] = (probs[j] - indicator) / N * gout_val;
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
        auto input = saved_tensors[0];
        auto target = saved_tensors[1];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto target_c = target->is_contiguous() ? target : target->contiguous();
        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        int size = input_c->numel();
        StorageUseGuard guard({input_c->storage, target_c->storage, grad_output->storage, grad_input->storage});
        if (input_c->device.type == DeviceType::GPU) {
            auto kernel = CLBackend::get().get_kernel(KernelID::L1LossBackward);
            cl_mem in_mem = input_c->gpu_data();
            int in_off = input_c->offset;
            cl_mem tgt_mem = target_c->gpu_data();
            int tgt_off = target_c->offset;
            cl_mem gout_mem = grad_output->gpu_data();
            int gout_off = grad_output->offset;
            cl_mem gin_mem = grad_input->gpu_data();
            int gin_off = grad_input->offset;
            float scale = 1.0f / size;
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
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
        auto input = saved_tensors[0];
        auto target = saved_tensors[1];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto target_c = target->is_contiguous() ? target : target->contiguous();
        auto grad_input = Tensor::create(input_c->shape, input_c->device);
        int size = input_c->numel();
        StorageUseGuard guard({input_c->storage, target_c->storage, grad_output->storage, grad_input->storage});
        if (input_c->device.type == DeviceType::GPU) {
            auto kernel = CLBackend::get().get_kernel(KernelID::BceLossBackward);
            cl_mem in_mem = input_c->gpu_data();
            int in_off = input_c->offset;
            cl_mem tgt_mem = target_c->gpu_data();
            int tgt_off = target_c->offset;
            cl_mem gout_mem = grad_output->gpu_data();
            int gout_off = grad_output->offset;
            cl_mem gin_mem = grad_input->gpu_data();
            int gin_off = grad_input->offset;
            float scale = 1.0f / size;
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
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
        auto kernel = CLBackend::get().get_kernel(KernelID::L1LossForward);
        cl_mem in_mem = input_c->gpu_data();
        int in_off = input_c->offset;
        cl_mem tgt_mem = target_c->gpu_data();
        int tgt_off = target_c->offset;
        cl_mem out_mem = out->gpu_data();
        int out_off = out->offset;
        int size_val = size;
        float zero = 0.0f;
        CLBackend::get().write(out_mem, sizeof(float), &zero, out_off);
        size_t blocks = std::min(static_cast<size_t>(32), (size + 255) / 256);
        if (blocks < 1) blocks = 1;
        CLBackend::get().launch(kernel, {blocks * 256}, {256},
            {&in_mem, &in_off, &tgt_mem, &tgt_off, &out_mem, &out_off, &size_val},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
    } else {
        float* in_ptr = input_c->data_ptr();
        float* tgt_ptr = target_c->data_ptr();
        float total = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            total += std::abs(in_ptr[i] - tgt_ptr[i]);
        }
        out->data_ptr()[0] = total * inv_n;
    }

    if (input->requires_grad) {
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

    if (input_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::BceLossForward);
        cl_mem in_mem = input_c->gpu_data();
        int in_off = input_c->offset;
        cl_mem tgt_mem = target_c->gpu_data();
        int tgt_off = target_c->offset;
        cl_mem out_mem = out->gpu_data();
        int out_off = out->offset;
        int size_val = size;
        float zero = 0.0f;
        CLBackend::get().write(out_mem, sizeof(float), &zero, out_off);
        size_t blocks = std::min(static_cast<size_t>(32), (size + 255) / 256);
        if (blocks < 1) blocks = 1;
        CLBackend::get().launch(kernel, {blocks * 256}, {256},
            {&in_mem, &in_off, &tgt_mem, &tgt_off, &out_mem, &out_off, &size_val},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
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

    if (input->requires_grad) {
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
        auto kernel = CLBackend::get().get_kernel(KernelID::MseLossForward);
        cl_mem in_mem = input_c->gpu_data();
        int in_off = input_c->offset;
        cl_mem tgt_mem = target_c->gpu_data();
        int tgt_off = target_c->offset;
        cl_mem out_mem = out->gpu_data();
        int out_off = out->offset;
        int size_val = size;
        float zero = 0.0f;
        CLBackend::get().write(out_mem, sizeof(float), &zero, out_off);
        size_t blocks = std::min(static_cast<size_t>(32), (size + 255) / 256);
        if (blocks < 1) blocks = 1;
        CLBackend::get().launch(kernel, {blocks * 256}, {256},
            {&in_mem, &in_off, &tgt_mem, &tgt_off, &out_mem, &out_off, &size_val},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
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

    if (input->requires_grad) {
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

std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target) {
    if (input->device != target->device) {
        target = target->to(input->device);
    }
    if (input->shape.size() != 2 || target->shape.size() != 1) {
        throw std::runtime_error("[litetorch Error] CrossEntropyLoss requires 2D input (logits) and 1D target");
    }

    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto target_c = target->is_contiguous() ? target : target->contiguous();
    int N = input_c->shape[0];
    int C = input_c->shape[1];
    auto out = Tensor::create({1}, input_c->device);
    StorageUseGuard guard({input_c->storage, target_c->storage, out->storage});

    if (input_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::CrossEntropyLossForward);
        cl_mem in_mem = input_c->gpu_data();
        int in_off = input_c->offset;
        cl_mem tgt_mem = target_c->gpu_data();
        int tgt_off = target_c->offset;
        cl_mem out_mem = out->gpu_data();
        int out_off = out->offset;
        float zero = 0.0f;
        CLBackend::get().write(out_mem, sizeof(float), &zero, out_off);
        CLBackend::get().launch(kernel, {static_cast<size_t>(N)}, {},
            {&in_mem, &in_off, &tgt_mem, &tgt_off, &out_mem, &out_off, &N, &C},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int)});
    } else {
        float* in_ptr = input_c->data_ptr();
        float* tgt_ptr = target_c->data_ptr();
        float total_loss = 0.0f;

        std::vector<float> losses(N, 0.0f);
        ThreadPool::get().parallel_for(0, N, [&](int64_t i) {
            float max_val = in_ptr[i * C];
            for (int j = 1; j < C; ++j) {
                if (in_ptr[i * C + j] > max_val) max_val = in_ptr[i * C + j];
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < C; ++j) {
                sum_exp += std::exp(in_ptr[i * C + j] - max_val);
            }
            int target_idx = static_cast<int>(tgt_ptr[i]);
            float correct_logit = in_ptr[i * C + target_idx];
            losses[i] = -correct_logit + max_val + std::log(sum_exp);
        });

        for (int i = 0; i < N; ++i) {
            total_loss += losses[i];
        }
        float avg_loss = total_loss / N;
        out->data_ptr()[0] = avg_loss;
        if (out->device.type == DeviceType::TPU) {
            auto tpu = BackendDispatcher::get().get_tpu_backend();
            if (tpu) tpu->write(out->gpu_data(), sizeof(float), &avg_loss);
        }
    }

    if (input->requires_grad) {
        auto node = std::make_shared<CrossEntropyLossNode>();
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
