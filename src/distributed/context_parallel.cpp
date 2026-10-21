#include "litetorch/context_parallel.h"
#include "litetorch/distributed.h"
#include "litetorch/p2p_async.h"
#include "litetorch/autograd.h"
#include "litetorch/ops.h"
#include <vector>
#include <algorithm>
#include <cstring>

namespace litetorch {
namespace distributed {

namespace {

int cp_world() {
    auto& pg = ProcessGroup::get();
    return pg.is_initialized() ? pg.get_world_size() : 1;
}

int cp_rank() {
    auto& pg = ProcessGroup::get();
    return pg.is_initialized() ? pg.get_rank() : 0;
}

void cp_require_tiling(int cp_group_size) {
    int world = cp_world();
    if (world > 1 && world % cp_group_size != 0)
        throw std::runtime_error("[litetorch Error] context_parallel: world_size must be divisible by cp_group_size");
}

std::vector<int> cp_group_ranks(int cp_group_size) {
    int base = (cp_rank() / cp_group_size) * cp_group_size;
    std::vector<int> g;
    g.reserve(cp_group_size);
    for (int i = 0; i < cp_group_size; ++i) g.push_back(base + i);
    return g;
}

void cp_all_gather_into(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full, int cp_group_size) {
    auto group = cp_group_ranks(cp_group_size);
    int gsize = static_cast<int>(group.size());
    int crank = cp_rank() % cp_group_size;
    size_t n = shard->numel();
    size_t elem_sz = shard->storage->element_size();
    shard->storage->ensure_cpu();
    full->storage->ensure_cpu();
    char* dst = (char*)full->raw_data_ptr();
    std::memcpy(dst + (size_t)crank * n * elem_sz, shard->raw_data_ptr(), n * elem_sz);
    if (gsize <= 1) return;
    auto recv_buf = Tensor::create(shard->shape, shard->device, false, shard->dtype);
    for (int s = 1; s < gsize; ++s) {
        int send_to = group[(crank + s) % gsize];
        int recv_from = group[(crank - s + gsize * 2) % gsize];
        std::shared_ptr<Work> w_send;
        std::shared_ptr<Work> w_recv;
        if (crank % 2 == 0) {
            w_send = isend(shard, send_to);
            w_recv = irecv(recv_buf, recv_from);
        } else {
            w_recv = irecv(recv_buf, recv_from);
            w_send = isend(shard, send_to);
        }
        w_send->wait();
        w_recv->wait();
        int src_idx = (crank - s + gsize) % gsize;
        std::memcpy(dst + (size_t)src_idx * n * elem_sz, recv_buf->raw_data_ptr(), n * elem_sz);
    }
}

std::shared_ptr<Tensor> cp_shard_impl(std::shared_ptr<Tensor> input, int cp_group_size) {
    if (!input->is_contiguous()) {
        input = input->contiguous();
    }
    int rank = cp_rank();
    auto shape = input->shape;
    int64_t batch_size = shape[0];
    int64_t total_seq_len = shape[1];
    if (total_seq_len % cp_group_size != 0) {
        throw std::runtime_error("[litetorch Error] context_parallel: seq_len must be divisible by cp_group_size");
    }
    int64_t shard_seq_len = total_seq_len / cp_group_size;
    int cp_rank_idx = rank % cp_group_size;
    int64_t start_idx = cp_rank_idx * shard_seq_len;
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
        char* src = (char*)input->raw_data_ptr();
        char* dst = (char*)local_tensor->raw_data_ptr();
        for (int64_t b = 0; b < batch_size; ++b) {
            char* b_src = src + (b * total_seq_len + start_idx) * hidden_dim * elem_sz;
            char* b_dst = dst + (b * local_seq_len) * hidden_dim * elem_sz;
            std::memcpy(b_dst, b_src, local_seq_len * hidden_dim * elem_sz);
        }
    } else {
        auto cpu_input = input->to(Device(DeviceType::CPU, 0));
        char* src = (char*)cpu_input->raw_data_ptr();
        auto cpu_local = Tensor::create(shard_shape, Device(DeviceType::CPU, 0), false, input->dtype);
        char* dst = (char*)cpu_local->raw_data_ptr();
        for (int64_t b = 0; b < batch_size; ++b) {
            char* b_src = src + (b * total_seq_len + start_idx) * hidden_dim * elem_sz;
            char* b_dst = dst + (b * local_seq_len) * hidden_dim * elem_sz;
            std::memcpy(b_dst, b_src, local_seq_len * hidden_dim * elem_sz);
        }
        local_tensor = cpu_local->to(input->device);
    }
    return local_tensor;
}

class CPShardFn : public Function {
public:
    explicit CPShardFn(int cp_group_size) : cp_group_size_(cp_group_size) {}

    std::vector<std::shared_ptr<Tensor>> forward(FunctionContext* ctx,
        const std::vector<std::shared_ptr<Tensor>>& inputs) override {
        ctx->save_for_backward(inputs[0]);
        return {cp_shard_impl(inputs[0], cp_group_size_)};
    }

    std::vector<std::shared_ptr<Tensor>> backward(FunctionContext*,
        const std::vector<std::shared_ptr<Tensor>>& grad_outputs) override {
        return {context_parallel_backward(grad_outputs[0], cp_group_size_)};
    }

private:
    int cp_group_size_;
};

}

std::shared_ptr<Tensor> context_parallel_forward(std::shared_ptr<Tensor> input, int cp_group_size) {
    if (!input || cp_group_size <= 1) {
        return input;
    }
    if (cp_world() <= 1) {
        return input;
    }
    if (input->shape.size() < 2) {
        return input;
    }
    cp_require_tiling(cp_group_size);
    if (input->requires_grad) {
        auto fn = std::make_shared<CPShardFn>(cp_group_size);
        return Function::apply(fn, {input})[0];
    }
    return cp_shard_impl(input, cp_group_size);
}

std::shared_ptr<Tensor> context_parallel_backward(std::shared_ptr<Tensor> grad_output, int cp_group_size) {
    if (!grad_output || cp_group_size <= 1) {
        return grad_output;
    }
    if (!ProcessGroup::get().is_initialized() || cp_world() <= 1) {
        return grad_output;
    }
    cp_require_tiling(cp_group_size);
    auto full_shape = grad_output->shape;
    full_shape[1] *= cp_group_size;
    bool off_cpu = grad_output->device.type != DeviceType::CPU;
    auto shard = off_cpu ? grad_output->to(Device(DeviceType::CPU, 0)) : grad_output;
    auto full_cpu = Tensor::create(full_shape, Device(DeviceType::CPU, 0), false, shard->dtype);
    cp_all_gather_into(shard, full_cpu, cp_group_size);
    if (off_cpu) return full_cpu->to(grad_output->device);
    return full_cpu;
}

}
}
