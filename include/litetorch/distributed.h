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
#include <deque>
#include <tuple>
#include "litetorch/pipeline.h"

namespace litetorch {
namespace distributed {

enum class ReduceOp {
    SUM = 0,
    AVG = 1,
    PRODUCT = 2,
    MIN = 3,
    MAX = 4
};

struct ShmControl {
    std::atomic<int> steps[4096];
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
    void all_reduce(std::shared_ptr<Tensor> tensor, ReduceOp op);
    std::future<void> all_reduce_async(std::shared_ptr<Tensor> tensor, ReduceOp op);
    void barrier();
    void all_to_all(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output);
    void scatter(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, int src);
    void gather(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output, int dst);
    void reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full, ReduceOp op);
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
    bool all_reduce_shm(std::shared_ptr<Tensor> tensor, ReduceOp op);
    bool broadcast_shm(std::shared_ptr<Tensor> tensor, int src);
    bool all_gather_shm(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    bool reduce_scatter_shm(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    bool reduce_scatter_shm(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full, ReduceOp op);
    void all_reduce_socket_op(std::shared_ptr<Tensor> tensor, ReduceOp op);
    bool all_to_all_shm(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output);
    bool scatter_shm(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, int src);
    bool gather_shm(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output, int dst);
    void all_to_all_socket(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output);
    void scatter_socket(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, int src);
    void gather_socket(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output, int dst);
    void barrier_shm();
    void barrier_socket();

    std::thread async_worker_;
    std::queue<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<std::promise<void>>, ReduceOp>> async_queue_;
    std::mutex async_mutex_;
    std::condition_variable async_cv_;
    bool stop_async_worker_ = false;

    struct RelayMsg {
        int src = -1;
        int32_t ndim = 0;
        int32_t dtype_val = 0;
        std::vector<int64_t> shape;
        std::vector<char> data;
    };
    std::thread relay_thread_;
    bool stop_relay_ = false;
    std::mutex relay_send_mutex_;
    std::mutex relay_inbox_mutex_;
    std::condition_variable relay_inbox_cv_;
    std::deque<RelayMsg> relay_inbox_;
    void relay_loop();
    static void relay_send_all(int fd, const char* buf, size_t n);
    static size_t relay_dtype_bytes(int32_t dtype_val);
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


class NCCLBridge {
public:
    static NCCLBridge& get();
    bool is_available() const { return available_; }
    void init(int rank, int world_size);
    void init_groups(int rank, int world_size, int tp_size = 1, int pp_size = 1);
    bool all_reduce(std::shared_ptr<Tensor> tensor);
    bool all_reduce(std::shared_ptr<Tensor> tensor, ReduceOp op);
    bool broadcast(std::shared_ptr<Tensor> tensor, int src);
    bool all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    bool reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    bool reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full, ReduceOp op);

    bool tp_all_reduce(std::shared_ptr<Tensor> tensor);
    bool tp_all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    bool dp_reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);
    bool dp_all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full);

    void group_start();
    void group_end();

    void sync_comm();
    void shutdown();

private:
    NCCLBridge();
    ~NCCLBridge();

    int map_dtype(DataType dtype);
    int map_reduce_op(ReduceOp op);
    void* get_gpu_raw_ptr(std::shared_ptr<Tensor> t);
    void order_for_stream(void* stream);
    void sync_stream_(void* stream);
    static void nccl_check(int res, const char* op);

    bool initialized_ = false;
    bool available_ = false;
    int world_size_ = 0;
    void* lib_handle_ = nullptr;
    void* comm_ = nullptr;
    void* comm_stream_ = nullptr;
    void* plain_event_ = nullptr;

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
    void* group_start_fn = nullptr;
    void* group_end_fn = nullptr;
    void* comm_get_async_error_fn = nullptr;
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

void barrier();
void all_reduce(std::shared_ptr<Tensor> tensor, ReduceOp op);
std::future<void> all_reduce_async(std::shared_ptr<Tensor> tensor, ReduceOp op);
void all_to_all(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output);
void scatter(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, int src);
void gather(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output, int dst);
void reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full, ReduceOp op);
void all_reduce_grads(std::shared_ptr<nn::Module> module, ReduceOp op);

class DistributedDataParallel : public nn::Module {
public:
    std::shared_ptr<nn::Module> module;
    ReduceOp reduce_op = ReduceOp::AVG;

    DistributedDataParallel(std::shared_ptr<nn::Module> sub_module, ReduceOp op = ReduceOp::AVG);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    void backward(std::shared_ptr<Tensor> loss);
    void backward(std::shared_ptr<Tensor> loss, std::shared_ptr<Tensor> grad_output);
    void sync_grads();

    std::vector<std::shared_ptr<Tensor>> parameters() override;
    std::vector<std::shared_ptr<nn::Module>> children() override;
    void to(const Device& device) override;
    void train() override;
    void eval() override;
};

}
}

// === DT-A1 (LT-ENG-018): tensor-parallel collectives; see litetorch/tp_collectives.h ===
#include "litetorch/tp_collectives.h"

#endif
