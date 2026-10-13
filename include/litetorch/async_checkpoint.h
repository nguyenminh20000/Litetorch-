#ifndef LITETORCH_ASYNC_CHECKPOINT_H
#define LITETORCH_ASYNC_CHECKPOINT_H

#include "litetorch/tensor.h"
#include <string>
#include <map>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <exception>

namespace litetorch {
namespace async_checkpoint {

class AsyncCheckpointSaver {
public:
    AsyncCheckpointSaver();
    ~AsyncCheckpointSaver();

    AsyncCheckpointSaver(const AsyncCheckpointSaver&) = delete;
    AsyncCheckpointSaver& operator=(const AsyncCheckpointSaver&) = delete;

    void save_async(const std::map<std::string, std::shared_ptr<Tensor>>& tensors,
                    const std::string& filepath);
    void wait();
    bool is_saving() const;

private:
    struct Job {
        std::map<std::string, std::shared_ptr<Tensor>> staged;
        std::string filepath;
    };

    void worker_loop();
    std::map<std::string, std::shared_ptr<Tensor>> stage_locked(
        const std::map<std::string, std::shared_ptr<Tensor>>& tensors);

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::thread worker_;
    bool stop_ = false;
    bool busy_ = false;
    Job job_;
    std::exception_ptr job_error_;
    std::map<std::string, std::shared_ptr<Tensor>> staging_[2];
    int staging_idx_ = 0;
};

}
}

#endif
