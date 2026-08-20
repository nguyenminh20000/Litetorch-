#ifndef LITETORCH_DISTRIBUTED_H
#define LITETORCH_DISTRIBUTED_H

#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include <vector>
#include <string>
#include <memory>
#include <future>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

namespace litetorch {
namespace distributed {

struct ShmControl {
    std::atomic<int> steps[128];
};

uint16_t float_to_half(float f);
float half_to_float(uint16_t h);

class NCCLBridge;

class ProcessGroup {
public:
    friend class NCCLBridge;
    static ProcessGroup& get();
    void init(int rank, int world_size, const std::string& master_addr = "127.0.0.1", int master_port = 29500);
    void shutdown();

    int get_rank() const { return rank_; }
    int get_world_size() const { return world_size_; }
    bool is_initialized() const { return initialized_; }

    void all_reduce(std::shared_ptr<Tensor> tensor);
    std::future<void> all_reduce_async(std::shared_ptr<Tensor> tensor);
    void broadcast(std::shared_ptr<Tensor> tensor, int src);
    void all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    void reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    void send_tensor(std::shared_ptr<Tensor> tensor, int dst);
    void recv_tensor(std::shared_ptr<Tensor> tensor, int src);
    void sync_comm();
    ~ProcessGroup();

private:
    ProcessGroup() = default;

    int rank_ = 0;
    int world_size_ = 1;
    bool initialized_ = false;

    int server_fd_ = -1;
    std::vector<int> client_fds_;
    int peer_fd_ = -1;

    bool use_shm_ = false;
    int master_port_ = 0;
    int shm_ctrl_fd_ = -1;
    ShmControl* shm_ctrl_ = nullptr;
    std::vector<float*> shm_buffers_;
    size_t shm_buffer_size_ = 0;

    void init_shm(const std::string& master_addr, int master_port);
    void shutdown_shm();
    bool all_reduce_shm(std::shared_ptr<Tensor> tensor);
    bool broadcast_shm(std::shared_ptr<Tensor> tensor, int src);
    bool all_gather_shm(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    bool reduce_scatter_shm(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);

    std::thread async_worker_;
    std::queue<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<std::promise<void>>>> async_queue_;
    std::mutex async_mutex_;
    std::condition_variable async_cv_;
    bool stop_async_worker_ = false;
};

void init_process_group(int rank, int world_size, const std::string& master_addr, int master_port);
void shutdown();
void all_reduce(std::shared_ptr<Tensor> tensor);
std::future<void> all_reduce_async(std::shared_ptr<Tensor> tensor);
void broadcast(std::shared_ptr<Tensor> tensor, int src_rank);
void all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
void reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
void all_reduce_grads(std::shared_ptr<nn::Module> module);
void all_reduce_bucketed(const std::vector<std::shared_ptr<Tensor>>& tensors, size_t bucket_size_bytes = 25 * 1024 * 1024);
void send_tensor(std::shared_ptr<Tensor> tensor, int dst);
void recv_tensor(std::shared_ptr<Tensor> tensor, int src);

class PipelineParallelModule : public nn::Module {
public:
    std::shared_ptr<nn::Module> local_module;

    PipelineParallelModule(std::shared_ptr<nn::Module> sub_module) : local_module(sub_module) {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        int rank = ProcessGroup::get().is_initialized() ? ProcessGroup::get().get_rank() : 0;
        int world_size = ProcessGroup::get().is_initialized() ? ProcessGroup::get().get_world_size() : 1;

        std::shared_ptr<Tensor> x = input;
        if (rank > 0 && world_size > 1) {
            x = Tensor::create({1}, input ? input->device : Device(DeviceType::CPU));
            ProcessGroup::get().recv_tensor(x, rank - 1);
        }

        std::shared_ptr<Tensor> out = local_module->forward(x);

        if (rank < world_size - 1 && world_size > 1) {
            ProcessGroup::get().send_tensor(out, rank + 1);
        }
        return out;
    }

    std::vector<std::shared_ptr<Tensor>> forward_microbatches(const std::vector<std::shared_ptr<Tensor>>& microbatches) {
        return schedule_1f1b(microbatches);
    }

    std::vector<std::shared_ptr<Tensor>> schedule_1f1b(const std::vector<std::shared_ptr<Tensor>>& microbatches) {
        int rank = ProcessGroup::get().is_initialized() ? ProcessGroup::get().get_rank() : 0;
        int world_size = ProcessGroup::get().is_initialized() ? ProcessGroup::get().get_world_size() : 1;
        int num_microbatches = static_cast<int>(microbatches.size());
        std::vector<std::shared_ptr<Tensor>> outputs(num_microbatches);
        if (world_size <= 1 || num_microbatches == 0) {
            for (int i = 0; i < num_microbatches; ++i) {
                outputs[i] = local_module->forward(microbatches[i]);
            }
            return outputs;
        }

        int num_warmup = std::min(num_microbatches, world_size - rank - 1);
        int num_1f1b = num_microbatches - num_warmup;

        for (int i = 0; i < num_warmup; ++i) {
            std::shared_ptr<Tensor> x = microbatches[i];
            if (rank > 0) {
                x = Tensor::create({1}, x ? x->device : Device(DeviceType::CPU));
                ProcessGroup::get().recv_tensor(x, rank - 1);
            }
            auto out = local_module->forward(x);
            if (rank < world_size - 1) {
                ProcessGroup::get().send_tensor(out, rank + 1);
            }
            outputs[i] = out;
        }

        for (int i = 0; i < num_1f1b; ++i) {
            int mb_idx = num_warmup + i;
            std::shared_ptr<Tensor> x = microbatches[mb_idx];
            if (rank > 0) {
                x = Tensor::create({1}, x ? x->device : Device(DeviceType::CPU));
                ProcessGroup::get().recv_tensor(x, rank - 1);
            }
            auto out = local_module->forward(x);
            if (rank < world_size - 1) {
                ProcessGroup::get().send_tensor(out, rank + 1);
            }
            outputs[mb_idx] = out;
        }

        return outputs;
    }

    std::vector<std::shared_ptr<Tensor>> train_step_1f1b(
        const std::vector<std::shared_ptr<Tensor>>& microbatches,
        std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>, int)> loss_fn = nullptr) {
        
        int rank = ProcessGroup::get().is_initialized() ? ProcessGroup::get().get_rank() : 0;
        int world_size = ProcessGroup::get().is_initialized() ? ProcessGroup::get().get_world_size() : 1;
        int num_microbatches = static_cast<int>(microbatches.size());
        std::vector<std::shared_ptr<Tensor>> losses;
        if (world_size <= 1 || num_microbatches == 0) {
            for (int i = 0; i < num_microbatches; ++i) {
                auto out = local_module->forward(microbatches[i]);
                if (loss_fn) {
                    auto loss = loss_fn(out, i);
                    loss->backward();
                    losses.push_back(loss);
                }
            }
            return losses;
        }

        int num_warmup = std::min(num_microbatches, world_size - rank - 1);
        int num_1f1b = num_microbatches - num_warmup;

        std::queue<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> fifo_queue;
        int forward_idx = 0;

        auto run_forward_step = [&](int mb_idx) {
            std::shared_ptr<Tensor> x = (mb_idx < static_cast<int>(microbatches.size())) ? microbatches[mb_idx] : nullptr;
            if (rank > 0) {
                x = Tensor::create({1}, x ? x->device : Device(DeviceType::CPU));
                x->requires_grad = true;
                ProcessGroup::get().recv_tensor(x, rank - 1);
            }
            auto out = local_module->forward(x);
            if (rank < world_size - 1) {
                ProcessGroup::get().send_tensor(out, rank + 1);
            }
            fifo_queue.push({x, out});
        };

        auto run_backward_step = [&]() {
            if (fifo_queue.empty()) return;
            auto item = fifo_queue.front();
            fifo_queue.pop();
            auto x = item.first;
            auto out = item.second;

            std::shared_ptr<Tensor> grad_out = nullptr;
            if (rank == world_size - 1) {
                if (loss_fn) {
                    int mb_id = static_cast<int>(losses.size());
                    auto loss = loss_fn(out, mb_id);
                    losses.push_back(loss);
                    loss->backward();
                } else {
                    out->backward();
                }
            } else {
                grad_out = Tensor::create(out->shape, out->device);
                ProcessGroup::get().recv_tensor(grad_out, rank + 1);
                out->backward(grad_out);
            }

            if (rank > 0 && x && x->grad) {
                ProcessGroup::get().send_tensor(x->grad, rank - 1);
            }
        };

        for (int i = 0; i < num_warmup; ++i) {
            run_forward_step(forward_idx++);
        }

        for (int i = 0; i < num_1f1b; ++i) {
            run_forward_step(forward_idx++);
            run_backward_step();
        }

        while (!fifo_queue.empty()) {
            run_backward_step();
        }

        return losses;
    }

    std::vector<std::shared_ptr<Tensor>> run_1f1b_with_backward(
        const std::vector<std::shared_ptr<Tensor>>& microbatches,
        std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>, int)> loss_fn) {
        return train_step_1f1b(microbatches, loss_fn);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() override {
        return local_module->parameters();
    }

    void to(const Device& device) override {
        local_module->to(device);
    }
};

class NCCLBridge {
public:
    static NCCLBridge& get();
    bool is_available() const { return available_; }
    void init(int rank, int world_size);
    void init_groups(int rank, int world_size, int tp_size = 1, int pp_size = 1);
    bool all_reduce(std::shared_ptr<Tensor> tensor);
    bool broadcast(std::shared_ptr<Tensor> tensor, int src);
    bool all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    bool reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);

    bool tp_all_reduce(std::shared_ptr<Tensor> tensor);
    bool tp_all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    bool dp_reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    bool dp_all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);

    void sync_comm();
    void shutdown();

private:
    NCCLBridge();
    ~NCCLBridge();

    int map_dtype(DataType dtype);
    void* get_gpu_raw_ptr(std::shared_ptr<Tensor> t);

    bool initialized_ = false;
    bool available_ = false;
    void* lib_handle_ = nullptr;
    void* comm_ = nullptr;
    void* comm_stream_ = nullptr;

    void* tp_comm_ = nullptr;
    void* dp_comm_ = nullptr;
    void* pp_comm_ = nullptr;
    void* tp_stream_ = nullptr;
    void* dp_stream_ = nullptr;
    void* pp_stream_ = nullptr;

    void* get_unique_id_fn = nullptr;
    void* comm_init_rank_fn = nullptr;
    void* comm_split_fn = nullptr;
    void* all_reduce_fn = nullptr;
    void* broadcast_fn = nullptr;
    void* all_gather_fn = nullptr;
    void* reduce_scatter_fn = nullptr;
    void* comm_destroy_fn = nullptr;
};

class OverlappedAllReducer {
public:
    static OverlappedAllReducer& get();
    void push_and_all_reduce(std::shared_ptr<Tensor> tensor);
    void push_and_reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    void sync();
    ~OverlappedAllReducer();

private:
    OverlappedAllReducer();

    void* ready_event_ = nullptr;
    void* done_event_ = nullptr;
};

void overlapped_all_reduce(std::shared_ptr<Tensor> tensor);
void overlapped_reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
void sync_overlapped_all_reduce();

}
}

#endif
