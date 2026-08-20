#ifndef LITETORCH_DISTRIBUTED_H
#define LITETORCH_DISTRIBUTED_H

#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include <string>
#include <vector>
#include <memory>

#include <atomic>

#include <queue>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>

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
    ~ProcessGroup();

    void init(int rank, int world_size, const std::string& master_addr, int master_port);
    void shutdown();

    void all_reduce(std::shared_ptr<Tensor> tensor);
    std::future<void> all_reduce_async(std::shared_ptr<Tensor> tensor);
    void broadcast(std::shared_ptr<Tensor> tensor, int src);
    void all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    void reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    void send_tensor(std::shared_ptr<Tensor> tensor, int dst);
    void recv_tensor(std::shared_ptr<Tensor> tensor, int src);
    void sync_comm();

    int get_rank() const { return rank_; }
    int get_world_size() const { return world_size_; }
    bool is_initialized() const { return initialized_; }

private:
    ProcessGroup() = default;

    void init_shm(const std::string& master_addr, int master_port);
    void shutdown_shm();
    bool all_reduce_shm(std::shared_ptr<Tensor> tensor);
    bool broadcast_shm(std::shared_ptr<Tensor> tensor, int src);
    bool all_gather_shm(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    bool reduce_scatter_shm(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);

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
void send_tensor(std::shared_ptr<Tensor> tensor, int dst);
void recv_tensor(std::shared_ptr<Tensor> tensor, int src);

class PipelineParallelModule : public nn::Module {
public:
    std::shared_ptr<nn::Module> local_module;

    PipelineParallelModule(std::shared_ptr<nn::Module> sub_module) : local_module(sub_module) {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        int rank = ProcessGroup::get().get_rank();
        int world_size = ProcessGroup::get().get_world_size();

        std::shared_ptr<Tensor> x = input;
        if (rank > 0) {
            x = Tensor::create({1}, input ? input->device : Device(DeviceType::CPU));
            ProcessGroup::get().recv_tensor(x, rank - 1);
        }

        std::shared_ptr<Tensor> out = local_module->forward(x);

        if (rank < world_size - 1) {
            ProcessGroup::get().send_tensor(out, rank + 1);
        }
        return out;
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
    bool all_reduce(std::shared_ptr<Tensor> tensor);
    bool broadcast(std::shared_ptr<Tensor> tensor, int src);
    bool all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    bool reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
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

    void* get_unique_id_fn = nullptr;
    void* comm_init_rank_fn = nullptr;
    void* all_reduce_fn = nullptr;
    void* broadcast_fn = nullptr;
    void* all_gather_fn = nullptr;
    void* reduce_scatter_fn = nullptr;
    void* comm_destroy_fn = nullptr;
};

}
}

#endif
