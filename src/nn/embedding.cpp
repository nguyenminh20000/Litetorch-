#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend_route.h"
#include "litetorch/thread_pool.h"
#include <random>
#include <algorithm>
#include <cstring>

namespace litetorch {
extern const std::string litetorch_kernels_src;
extern std::mt19937& get_generator();

namespace {
inline uint16_t emb_float_to_half(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t exp = (x >> 23) & 0xff;
    uint32_t mant = x & 0x7fffff;
    if (exp == 0) return static_cast<uint16_t>(sign);
    if (exp == 255) return static_cast<uint16_t>(sign | 0x7c00 | (mant ? 0x200 : 0));
    int nexp = static_cast<int>(exp) - 127 + 15;
    if (nexp >= 31) return static_cast<uint16_t>(sign | 0x7c00);
    if (nexp <= 0) return static_cast<uint16_t>(sign);
    return static_cast<uint16_t>(sign | (nexp << 10) | (mant >> 13));
}
inline float emb_half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) f = sign << 31;
    else if (exp == 31) f = (sign << 31) | 0x7f800000 | (mant << 13);
    else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    return *reinterpret_cast<float*>(&f);
}
}

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
    int64_t padding_idx;
    EmbeddingNode(int num_embeddings, int embedding_dim, int64_t padding_idx = -1)
        : litetorch::Node("Embedding"), num_embeddings(num_embeddings),
          embedding_dim(embedding_dim), padding_idx(padding_idx) {}

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto weight = saved_tensors[1];
        auto input_c = input->is_contiguous() ? input : input->contiguous();
        auto weight_c = weight->is_contiguous() ? weight : weight->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();

        auto grad_weight = Tensor::create(weight_c->shape, weight_c->device, false, weight_c->dtype);
        StorageUseGuard guard({input_c->storage, weight_c->storage, gout_c->storage, grad_weight->storage});
        DataType wdt = weight_c->dtype;
        size_t wesz = weight_c->storage->element_size();

        if (input_c->device.type == DeviceType::GPU && wdt == DataType::FP32) {
            auto kernel = rt_gpu_get_kernel("litetorch_kernels", litetorch_kernels_src, "embedding_backward");
            cl_mem in_mem = input_c->gpu_data();
            int in_off = input_c->offset;
            cl_mem gout_mem = gout_c->gpu_data();
            int gout_off = gout_c->offset;
            cl_mem gw_mem = grad_weight->gpu_data();
            int gw_off = grad_weight->offset;
            int num_indices = input_c->numel();
            int num_embeddings_val = weight_c->shape[0];
            int embedding_dim_val = embedding_dim;

            std::vector<float> zeros(grad_weight->numel(), 0.0f);
            rt_gpu_write(gw_mem, zeros.size() * sizeof(float), zeros.data(), static_cast<size_t>(gw_off) * sizeof(float));
            int total_threads = num_indices;
            rt_gpu_launch(kernel, {static_cast<size_t>(total_threads)}, {},
                {&in_mem, &in_off, &gout_mem, &gout_off, &gw_mem, &gw_off, &num_indices, &num_embeddings_val, &embedding_dim_val},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
            if (padding_idx >= 0 && padding_idx < num_embeddings) {
                std::vector<float> prow(embedding_dim, 0.0f);
                rt_gpu_write(gw_mem, prow.size() * sizeof(float), prow.data(),
                             static_cast<size_t>(gw_off + padding_idx * embedding_dim) * sizeof(float));
            }
        } else {
            auto cpu_input = input_c->device.type == DeviceType::CPU ? input_c : input_c->to(Device(DeviceType::CPU, 0));
            auto cpu_gout = gout_c->device.type == DeviceType::CPU ? gout_c : gout_c->to(Device(DeviceType::CPU, 0));
            if (!cpu_input->is_contiguous()) cpu_input = cpu_input->contiguous();
            if (!cpu_gout->is_contiguous()) cpu_gout = cpu_gout->contiguous();
            auto cpu_gw = Tensor::create(weight_c->shape, Device(DeviceType::CPU, 0), false, wdt);
            size_t gw_n = cpu_gw->numel();
            std::memset(cpu_gw->storage->get_cpu_ptr(), 0, gw_n * wesz);
            float* in_ptr = cpu_input->data_ptr();
            size_t num_indices = cpu_input->numel();
            if (wdt == DataType::FP32) {
                float* gout_ptr = cpu_gout->data_ptr();
                float* gw_ptr = reinterpret_cast<float*>(cpu_gw->storage->get_cpu_ptr());
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
                if (padding_idx >= 0 && padding_idx < num_embeddings) {
                    std::fill(gw_ptr + padding_idx * embedding_dim,
                              gw_ptr + (padding_idx + 1) * embedding_dim, 0.0f);
                }
            } else if (wdt == DataType::FP16) {
                uint16_t* gout_ptr = reinterpret_cast<uint16_t*>(cpu_gout->data_ptr());
                uint16_t* gw_ptr = reinterpret_cast<uint16_t*>(cpu_gw->storage->get_cpu_ptr());
                for (size_t i = 0; i < num_indices; ++i) {
                    int idx = static_cast<int>(in_ptr[i]);
                    if (idx >= 0 && idx < num_embeddings) {
                        uint16_t* gw_row = gw_ptr + idx * embedding_dim;
                        uint16_t* gout_row = gout_ptr + i * embedding_dim;
                        for (int d = 0; d < embedding_dim; ++d) {
                            float acc = emb_half_to_float(gw_row[d]) + emb_half_to_float(gout_row[d]);
                            gw_row[d] = emb_float_to_half(acc);
                        }
                    }
                }
            } else {
                auto gout_f32 = cpu_gout->cast(DataType::FP32);
                if (!gout_f32->is_contiguous()) gout_f32 = gout_f32->contiguous();
                auto tmp_gw = Tensor::create(weight_c->shape, Device(DeviceType::CPU, 0));
                float* gout_ptr = gout_f32->data_ptr();
                float* gw_ptr = reinterpret_cast<float*>(tmp_gw->storage->get_cpu_ptr());
                std::fill(gw_ptr, gw_ptr + tmp_gw->numel(), 0.0f);
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
                cpu_gw = tmp_gw->cast(wdt);
            }
            grad_weight->copy_(cpu_gw);
        }
        std::vector<std::shared_ptr<Tensor>> res = { nullptr, grad_weight };
        return res;
    }
};
}

Embedding::Embedding(int num_embeddings, int embedding_dim, int64_t padding_idx)
    : num_embeddings(num_embeddings), embedding_dim(embedding_dim), padding_idx(padding_idx) {
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
        throw std::runtime_error("[litetorch Error] Embedding weight device does not match input device; call Embedding::to(device) before forward instead of relying on an implicit move");
    }
    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto weight_c = weight->is_contiguous() ? weight : weight->contiguous();

    std::shared_ptr<Tensor> w_compute = weight_c;
    if (weight_c->dtype != DataType::FP32) {
        w_compute = weight_c->cast(DataType::FP32);
    }
    auto w_compute_c = w_compute->is_contiguous() ? w_compute : w_compute->contiguous();

    std::vector<int64_t> out_shape = input_c->shape;
    out_shape.push_back(embedding_dim);
    auto out_fp32 = Tensor::create(out_shape, w_compute_c->device);
    StorageUseGuard guard({input_c->storage, w_compute_c->storage, out_fp32->storage});

    if (input_c->device.type == DeviceType::GPU) {
        auto kernel = rt_gpu_get_kernel("litetorch_kernels", litetorch_kernels_src, "embedding_forward");
        cl_mem in_mem = input_c->gpu_data();
        int in_off = input_c->offset;
        cl_mem w_mem = w_compute_c->gpu_data();
        int w_off = w_compute_c->offset;
        cl_mem out_mem = out_fp32->gpu_data();
        int out_off = out_fp32->offset;
        int num_indices = input_c->numel();
        int num_embeddings_val = num_embeddings;
        int embedding_dim_val = embedding_dim;

        int total_threads = num_indices * embedding_dim_val;
        rt_gpu_launch(kernel, {static_cast<size_t>(total_threads)}, {},
            {&in_mem, &in_off, &w_mem, &w_off, &out_mem, &out_off, &num_indices, &num_embeddings_val, &embedding_dim_val},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
        if (padding_idx >= 0 && padding_idx < num_embeddings) {
            auto zkernel = rt_gpu_get_kernel("litetorch_kernels", litetorch_kernels_src, "embedding_zero_padding_rows");
            int num_indices_val = num_indices;
            int padding_idx_val = static_cast<int>(padding_idx);
            rt_gpu_launch(zkernel, {static_cast<size_t>(num_indices_val)}, {},
                {&in_mem, &in_off, &out_mem, &out_off, &num_indices_val, &embedding_dim_val, &padding_idx_val},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
        }
    } else {
        float* in_ptr = input_c->data_ptr();
        float* w_ptr = w_compute_c->data_ptr();
        float* out_ptr = out_fp32->data_ptr();
        size_t num_indices = input_c->numel();
        litetorch::ThreadPool::get().parallel_for(0, num_indices, [&](int64_t i) {
            int idx = static_cast<int>(in_ptr[i]);
            if (idx < 0 || idx >= num_embeddings) {
                return;
            }
            std::copy(w_ptr + idx * embedding_dim, w_ptr + (idx + 1) * embedding_dim, out_ptr + i * embedding_dim);
        });
        if (padding_idx >= 0 && padding_idx < num_embeddings) {
            for (size_t i = 0; i < num_indices; ++i) {
                if (static_cast<int>(in_ptr[i]) == padding_idx) {
                    std::fill(out_ptr + i * embedding_dim, out_ptr + (i + 1) * embedding_dim, 0.0f);
                }
            }
        }
    }

    std::shared_ptr<Tensor> out = out_fp32;
    if (weight_c->dtype != DataType::FP32) {
        out = out_fp32->cast(weight_c->dtype);
    }

    if (Autograd::is_grad_enabled() && (input->requires_grad || weight->requires_grad)) {
        auto node = std::make_shared<EmbeddingNode>(num_embeddings, embedding_dim, padding_idx);
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
