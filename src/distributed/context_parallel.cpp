#include "litetorch/context_parallel.h"
#include "litetorch/distributed.h"
#include "litetorch/ops.h"
#include <vector>
#include <algorithm>
#include <cstring>

namespace litetorch {
namespace distributed {

std::shared_ptr<Tensor> context_parallel_forward(std::shared_ptr<Tensor> input, int cp_group_size) {
    if (!input || cp_group_size <= 1) {
        return input;
    }
    int rank = ProcessGroup::get().is_initialized() ? ProcessGroup::get().get_rank() : 0;
    int world_size = ProcessGroup::get().is_initialized() ? ProcessGroup::get().get_world_size() : 1;
    if (world_size <= 1) {
        return input;
    }
    auto shape = input->shape;
    if (shape.size() < 2) {
        return input;
    }
    int64_t batch_size = shape[0];
    int64_t total_seq_len = shape[1];
    int64_t shard_seq_len = (total_seq_len + cp_group_size - 1) / cp_group_size;
    int cp_rank = rank % cp_group_size;
    int64_t start_idx = cp_rank * shard_seq_len;
    int64_t end_idx = std::min(start_idx + shard_seq_len, total_seq_len);
    int64_t local_seq_len = std::max<int64_t>(0, end_idx - start_idx);

    std::vector<int64_t> shard_shape = shape;
    shard_shape[1] = local_seq_len;

    auto local_tensor = Tensor::create(shard_shape, input->device, input->requires_grad, input->dtype);
    size_t hidden_dim = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        hidden_dim *= shape[i];
    }

    size_t elem_sz = input->storage->element_size();
    if (input->device.type == DeviceType::CPU) {
        input->storage->ensure_cpu();
        char* src = (char*)input->data_ptr();
        char* dst = (char*)local_tensor->data_ptr();
        for (int64_t b = 0; b < batch_size; ++b) {
            char* b_src = src + (b * total_seq_len + start_idx) * hidden_dim * elem_sz;
            char* b_dst = dst + (b * local_seq_len) * hidden_dim * elem_sz;
            std::memcpy(b_dst, b_src, local_seq_len * hidden_dim * elem_sz);
        }
    } else {
        auto cpu_input = input->to(Device(DeviceType::CPU, 0));
        char* src = (char*)cpu_input->data_ptr();
        auto cpu_local = Tensor::create(shard_shape, Device(DeviceType::CPU, 0), false, input->dtype);
        char* dst = (char*)cpu_local->data_ptr();
        for (int64_t b = 0; b < batch_size; ++b) {
            char* b_src = src + (b * total_seq_len + start_idx) * hidden_dim * elem_sz;
            char* b_dst = dst + (b * local_seq_len) * hidden_dim * elem_sz;
            std::memcpy(b_dst, b_src, local_seq_len * hidden_dim * elem_sz);
        }
        local_tensor = cpu_local->to(input->device);
    }
    return local_tensor;
}

std::shared_ptr<Tensor> context_parallel_backward(std::shared_ptr<Tensor> grad_output, int cp_group_size) {
    if (!grad_output || cp_group_size <= 1) {
        return grad_output;
    }
    if (!ProcessGroup::get().is_initialized() || ProcessGroup::get().get_world_size() <= 1) {
        return grad_output;
    }
    auto full_shape = grad_output->shape;
    full_shape[1] *= cp_group_size;
    auto full_grad = Tensor::create(full_shape, grad_output->device, false, grad_output->dtype);
    ProcessGroup::get().all_gather(grad_output, full_grad);
    return full_grad;
}

}
}
