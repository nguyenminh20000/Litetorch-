#ifndef LITETORCH_FSDP_H
#define LITETORCH_FSDP_H

#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include "litetorch/distributed.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstring>
#include <algorithm>

#include "litetorch/backend.h"
#include "litetorch/cl_backend.h"

namespace litetorch {

extern const std::string litetorch_kernels_src;

namespace distributed {

class FullyShardedDataParallel {
public:
    FullyShardedDataParallel(std::shared_ptr<nn::Module> module) : module_(module) {
        int rank = ProcessGroup::get().get_rank();
        int world_size = ProcessGroup::get().get_world_size();

        for (auto& p : module_->parameters()) {
            size_t total_size = p->numel();
            size_t shard_size = (total_size + world_size - 1) / world_size;

            orig_shapes_[p] = p->shape;
            orig_sizes_[p] = total_size;
            shard_sizes_[p] = shard_size;

            auto sharded_storage = std::make_shared<StorageImpl>(shard_size, p->device, p->dtype);

            if (p->device.type == DeviceType::GPU) {
                auto backend = BackendDispatcher::get().get_backend();
                if (backend && backend->is_available()) {
                    size_t start_idx = rank * shard_size;
                    size_t copy_elements = 0;
                    if (total_size > start_idx) {
                        copy_elements = std::min(shard_size, total_size - start_idx);
                    }
                    if (copy_elements > 0) {
                        backend->copy(
                            p->storage->get_gpu_ptr(),
                            sharded_storage->get_gpu_ptr(),
                            copy_elements * p->storage->element_size(),
                            start_idx * p->storage->element_size(),
                            0
                        );
                    }
                } else if (CLBackend::get().is_available()) {
                    auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch::litetorch_kernels_src, "fill_zero");
                    int size_val = static_cast<int>(shard_size);
                    cl_mem gpu_ptr = sharded_storage->get_gpu_ptr();
                    CLBackend::get().launch(kernel, {shard_size}, {}, {&gpu_ptr, &size_val}, {sizeof(cl_mem), sizeof(int)});

                    size_t start_idx = rank * shard_size;
                    size_t copy_elements = 0;
                    if (total_size > start_idx) {
                        copy_elements = std::min(shard_size, total_size - start_idx);
                    }
                    if (copy_elements > 0) {
                        CLBackend::get().copy(
                            p->storage->get_gpu_ptr(),
                            gpu_ptr,
                            copy_elements * p->storage->element_size(),
                            start_idx * p->storage->element_size(),
                            0
                        );
                    }
                }
            } else {
                p->storage->ensure_cpu();
                float* cpu_data = p->storage->get_cpu_ptr();
                std::vector<float> local_shard(shard_size, 0.0f);
                size_t start_idx = rank * shard_size;
                for (size_t i = 0; i < shard_size; ++i) {
                    if (start_idx + i < total_size) {
                        local_shard[i] = cpu_data[start_idx + i];
                    }
                }
                std::memcpy(sharded_storage->get_cpu_ptr(), local_shard.data(), shard_size * sharded_storage->element_size());
            }

            sharded_storages_[p] = sharded_storage;
            p->storage = sharded_storage;
            p->shape = { (int64_t)shard_size };
            p->strides = { 1 };
            p->offset = 0;
        }
    }

    void gather_parameters() {
        int world_size = ProcessGroup::get().get_world_size();
        for (auto& p : module_->parameters()) {
            size_t shard_size = shard_sizes_[p];
            auto sharded_tensor = std::make_shared<Tensor>(sharded_storages_[p], std::vector<int64_t>{(int64_t)shard_size}, std::vector<int64_t>{1}, 0, p->device);
            
            auto full_tensor = Tensor::create({(int64_t)(shard_size * world_size)}, p->device);
            all_gather(sharded_tensor, full_tensor);

            p->storage = full_tensor->storage;
            p->shape = orig_shapes_[p];
            p->strides = default_strides(orig_shapes_[p]);
            p->offset = 0;
        }
    }

    void shard_parameters() {
        for (auto& p : module_->parameters()) {
            p->storage = sharded_storages_[p];
            p->shape = { (int64_t)shard_sizes_[p] };
            p->strides = { 1 };
            p->offset = 0;
        }
    }

    std::shared_ptr<nn::Module> get_module() const { return module_; }

private:
    std::shared_ptr<nn::Module> module_;
    std::unordered_map<std::shared_ptr<Tensor>, std::vector<int64_t>> orig_shapes_;
    std::unordered_map<std::shared_ptr<Tensor>, size_t> orig_sizes_;
    std::unordered_map<std::shared_ptr<Tensor>, size_t> shard_sizes_;
    std::unordered_map<std::shared_ptr<Tensor>, std::shared_ptr<StorageImpl>> sharded_storages_;
};

}
}

#endif
