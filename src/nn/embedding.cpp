#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/cl_backend.h"
#include "litetorch/thread_pool.h"
#include <random>
#include <algorithm>

namespace litetorch {
extern const std::string litetorch_kernels_src;
extern std::mt19937& get_generator();

namespace nn {

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

class EmbeddingNode : public litetorch::Node {
public:
    int num_embeddings;
    int embedding_dim;
    EmbeddingNode(int num_embeddings, int embedding_dim)
        : litetorch::Node("Embedding"), num_embeddings(num_embeddings), embedding_dim(embedding_dim) {}

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto input = saved_tensors[0];
        auto weight = saved_tensors[1];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto weight_c = weight->is_contiguous() ? weight : weight->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();

        auto grad_weight = Tensor::create(weight_c->shape, weight_c->device);
        StorageUseGuard guard({input_c->storage, weight_c->storage, gout_c->storage, grad_weight->storage});

        if (input_c->device.type == DeviceType::GPU) {
            auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "embedding_backward");
            cl_mem in_mem = input_c->gpu_data();
            int in_off = input_c->offset;
            cl_mem gout_mem = gout_c->gpu_data();
            int gout_off = gout_c->offset;
            cl_mem gw_mem = grad_weight->gpu_data();
            int gw_off = grad_weight->offset;
            int num_indices = input_c->numel();
            int num_embeddings_val = weight_c->shape[0];
            int embedding_dim_val = embedding_dim;

            int total_threads = num_embeddings_val * embedding_dim_val;
            CLBackend::get().launch(kernel, {static_cast<size_t>(total_threads)}, {},
                {&in_mem, &in_off, &gout_mem, &gout_off, &gw_mem, &gw_off, &num_indices, &num_embeddings_val, &embedding_dim_val},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
        } else {
            float* in_ptr = input_c->data_ptr();
            float* gout_ptr = gout_c->data_ptr();
            float* gw_ptr = grad_weight->data_ptr();
            std::fill(gw_ptr, gw_ptr + grad_weight->numel(), 0.0f);
            size_t num_indices = input_c->numel();
            for (size_t i = 0; i < num_indices; ++i) {
                int idx = static_cast<int>(in_ptr[i]);
                if (idx >= 0 && idx < num_embeddings) {
                    float* gw_row = gw_ptr + idx * embedding_dim;
                    float* gout_row = gout_ptr + i * embedding_dim;
                    for (int d = 0; d < embedding_dim; ++d) {
                        gw_row[d] += gout_row[d];
                    }
                }
            }
        }
        std::vector<std::shared_ptr<Tensor>> res = { nullptr, grad_weight };
        return res;
    }
};
}

Embedding::Embedding(int num_embeddings, int embedding_dim)
    : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
    weight = Tensor::create({num_embeddings, embedding_dim}, Device(DeviceType::CPU, 0), true);
    std::mt19937& gen = get_generator();
    std::normal_distribution<float> dis(0.0f, 1.0f);
    float* data = weight->data_ptr();
    for (size_t i = 0; i < weight->numel(); ++i) {
        data[i] = dis(gen);
    }
}

std::shared_ptr<Tensor> Embedding::forward(std::shared_ptr<Tensor> input) {
    if (weight->device != input->device) {
        weight = weight->to(input->device);
    }
    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto weight_c = weight->is_contiguous() ? weight : weight->contiguous();

    std::vector<int64_t> out_shape = input_c->shape;
    out_shape.push_back(embedding_dim);
    auto out = Tensor::create(out_shape, weight_c->device);
    StorageUseGuard guard({input_c->storage, weight_c->storage, out->storage});

    if (input_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "embedding_forward");
        cl_mem in_mem = input_c->gpu_data();
        int in_off = input_c->offset;
        cl_mem w_mem = weight_c->gpu_data();
        int w_off = weight_c->offset;
        cl_mem out_mem = out->gpu_data();
        int out_off = out->offset;
        int num_indices = input_c->numel();
        int num_embeddings_val = num_embeddings;
        int embedding_dim_val = embedding_dim;

        int total_threads = num_indices * embedding_dim_val;
        CLBackend::get().launch(kernel, {static_cast<size_t>(total_threads)}, {},
            {&in_mem, &in_off, &w_mem, &w_off, &out_mem, &out_off, &num_indices, &num_embeddings_val, &embedding_dim_val},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
    } else {
        float* in_ptr = input_c->data_ptr();
        float* w_ptr = weight_c->data_ptr();
        float* out_ptr = out->data_ptr();
        size_t num_indices = input_c->numel();
        litetorch::ThreadPool::get().parallel_for(0, num_indices, [&](int64_t i) {
            int idx = static_cast<int>(in_ptr[i]);
            if (idx < 0 || idx >= num_embeddings) {
                return;
            }
            std::copy(w_ptr + idx * embedding_dim, w_ptr + (idx + 1) * embedding_dim, out_ptr + i * embedding_dim);
        });
    }

    if (input->requires_grad || weight->requires_grad) {
        auto node = std::make_shared<EmbeddingNode>(num_embeddings, embedding_dim);
        node->inputs = { {input, input->requires_grad}, {weight, weight->requires_grad} };
        node->next_nodes = { input->creator, weight->creator };
        node->saved_tensors = { input, weight };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::vector<std::shared_ptr<Tensor>> Embedding::parameters() {
    return { weight };
}

void Embedding::to(const Device& device) {
    weight = weight->to(device);
}

}
}
