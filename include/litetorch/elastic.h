#ifndef LITETORCH_ELASTIC_H
#define LITETORCH_ELASTIC_H

#include "litetorch/dist_checkpoint.h"
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace litetorch {

class Tensor;

namespace distributed {
namespace elastic {

class HeartbeatAgent {
public:
    HeartbeatAgent(const std::string& store_addr, int store_port, int rank, int interval_ms = 1000);
    ~HeartbeatAgent();
    HeartbeatAgent(const HeartbeatAgent&) = delete;
    HeartbeatAgent& operator=(const HeartbeatAgent&) = delete;
    void start();
    void stop();
    bool running() const { return running_.load(); }
private:
    void loop();
    std::string addr_;
    int port_;
    int rank_;
    int interval_ms_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex mu_;
};

std::vector<int> detect_dead_ranks(const std::string& store_addr, int store_port, int64_t timeout_ms);

void save_consistent(
    const std::map<std::string, std::shared_ptr<Tensor>>& local_shards,
    const std::map<std::string, dist_checkpoint::TensorShardInfo>& shard_specs,
    const std::string& ckpt_dir,
    int64_t step);

int64_t latest_consistent_step(const std::string& ckpt_dir);

void load_latest(
    std::map<std::string, std::shared_ptr<Tensor>>& local_shards,
    const std::string& ckpt_dir);

std::string step_dir(const std::string& ckpt_dir, int64_t step);
std::string tmp_dir(const std::string& ckpt_dir, int64_t step);

}
}
}

#endif
