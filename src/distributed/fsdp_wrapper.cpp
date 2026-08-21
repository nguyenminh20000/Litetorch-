#include "litetorch/fsdp_wrapper.h"
#include "litetorch/distributed.h"
#include "litetorch/autograd.h"
#include "litetorch/backend.h"
#include "litetorch/allocator.h"
#include <cstring>
#include <unordered_set>

namespace litetorch {
namespace distributed {

class FSDPOutputNode : public Node {
public:
    std::vector<std::shared_ptr<Tensor>> full_params;
    std::vector<std::shared_ptr<Tensor>> sharded_params;
    
    FSDPOutputNode(std::shared_ptr<Node> original_creator, 
                   std::vector<std::shared_ptr<Tensor>> full_params,
                   std::vector<std::shared_ptr<Tensor>> sharded_params)
        : Node("FSDPOutput"), full_params(full_params), sharded_params(sharded_params) {
        if (original_creator) {
            this->next_nodes = { original_creator };
        }
    }
    
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        if (NCCLBridge::get().is_available()) {
            NCCLBridge::get().group_start();
        }
        for (size_t i = 0; i < full_params.size(); ++i) {
            auto full = full_params[i];
            if (full->device.type == DeviceType::GPU && !full->storage->gpu_data) {
                auto backend = BackendDispatcher::get().get_backend();
                if (backend && backend->is_available()) {
                    full->storage->gpu_data = (cl_mem)backend->allocate(full->storage->size);
                } else if (CLBackend::get().is_available()) {
                    full->storage->gpu_data = CLBackend::get().allocate(full->storage->size);
                }
            } else if (full->device.type == DeviceType::CPU && !full->storage->cpu_data) {
                full->storage->cpu_data = (float*)CachingAllocator::get().allocate_cpu(full->storage->size);
            }
            ProcessGroup::get().all_gather(sharded_params[i], full);
        }
        if (NCCLBridge::get().is_available()) {
            NCCLBridge::get().group_end();
        }
        ProcessGroup::get().sync_comm();
        return { grad_output };
    }
};

static void shard_module_parameters(std::shared_ptr<nn::Module> module, 
                                  std::vector<std::shared_ptr<Tensor>>& full_params_out, 
                                  std::vector<std::shared_ptr<Tensor>>& sharded_params_out) {
    int rank = ProcessGroup::get().get_rank();
    int world_size = ProcessGroup::get().get_world_size();
    
    auto params = module->parameters();
    for (auto& p : params) {
        size_t total_elements = p->numel();
        size_t shard_size = (total_elements + world_size - 1) / world_size;
        
        std::vector<int64_t> shard_shape = { (int64_t)shard_size };
        auto shard = Tensor::create(shard_shape, p->device, p->requires_grad, p->dtype);
        size_t start_idx = rank * shard_size;
        size_t copy_size = std::min(shard_size, total_elements > start_idx ? total_elements - start_idx : 0);
        
        if (copy_size > 0) {
            if (p->device.type == DeviceType::CPU) {
                if (!p->storage->cpu_data) p->storage->ensure_cpu();
                std::memcpy(shard->data_ptr(), p->data_ptr() + start_idx, copy_size * p->storage->element_size());
            } else {
                auto cpu_p = p->to(Device(DeviceType::CPU, 0));
                std::memcpy(shard->storage->get_cpu_ptr(), cpu_p->data_ptr() + start_idx, copy_size * p->storage->element_size());
                shard->to(p->device);
            }
        }
        
        p->shard = shard;
        
        if (p->requires_grad) {
            p->register_hook([shard, p](std::shared_ptr<Tensor> grad) {
                auto sharded_grad = Tensor::create(shard->shape, shard->device, false, shard->dtype);
                if (NCCLBridge::get().is_available() && grad->device.type == DeviceType::GPU) {
                    OverlappedAllReducer::get().push_and_reduce_scatter(sharded_grad, grad);
                } else {
                    ProcessGroup::get().reduce_scatter(sharded_grad, grad);
                    ProcessGroup::get().sync_comm();
                }
                
                shard->grad = sharded_grad;
                
                p->storage->discard_gpu();
                grad->storage->discard_gpu();
                return grad;
            });
        }
        
        full_params_out.push_back(p);
        sharded_params_out.push_back(shard);
        
        p->storage->discard_gpu();
    }
}

void FSDP::fully_shard(std::shared_ptr<nn::Module> module) {
    auto children = module->children();
    if (!children.empty()) {
        for (auto& child : children) {
            fully_shard(child);
        }
        return;
    }
    
    auto params = module->parameters();
    if (params.empty()) return;
    
    std::shared_ptr<std::vector<std::shared_ptr<Tensor>>> full_params = std::make_shared<std::vector<std::shared_ptr<Tensor>>>();
    std::shared_ptr<std::vector<std::shared_ptr<Tensor>>> sharded_params = std::make_shared<std::vector<std::shared_ptr<Tensor>>>();
    
    shard_module_parameters(module, *full_params, *sharded_params);
    
    module->register_forward_pre_hook([full_params, sharded_params](nn::Module*, std::shared_ptr<Tensor>) {
        if (NCCLBridge::get().is_available()) {
            NCCLBridge::get().group_start();
        }
        for (size_t i = 0; i < full_params->size(); ++i) {
            auto full = (*full_params)[i];
            if (full->device.type == DeviceType::GPU && !full->storage->gpu_data) {
                auto backend = BackendDispatcher::get().get_backend();
                if (backend && backend->is_available()) {
                    full->storage->gpu_data = (cl_mem)backend->allocate(full->storage->size);
                } else if (CLBackend::get().is_available()) {
                    full->storage->gpu_data = CLBackend::get().allocate(full->storage->size);
                }
            } else if (full->device.type == DeviceType::CPU && !full->storage->cpu_data) {
                full->storage->cpu_data = (float*)CachingAllocator::get().allocate_cpu(full->storage->size);
            }
            ProcessGroup::get().all_gather((*sharded_params)[i], full);
        }
        if (NCCLBridge::get().is_available()) {
            NCCLBridge::get().group_end();
        }
        ProcessGroup::get().sync_comm();
    });
    
    module->register_forward_hook([full_params, sharded_params](nn::Module*, std::shared_ptr<Tensor>, std::shared_ptr<Tensor> out) {
        if (out->requires_grad) {
            auto out_node = std::make_shared<FSDPOutputNode>(out->creator, *full_params, *sharded_params);
            out->creator = out_node;
        }
        for (auto& p : *full_params) {
            p->storage->discard_gpu();
        }
    });
}

}
}
