#include "litetorch/async_checkpoint.h"
#include "litetorch/serialization.h"
#include <stdexcept>

namespace litetorch {
namespace async_checkpoint {

AsyncCheckpointSaver::AsyncCheckpointSaver()
    : worker_(&AsyncCheckpointSaver::worker_loop, this) {}

AsyncCheckpointSaver::~AsyncCheckpointSaver() {
    wait();
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

std::map<std::string, std::shared_ptr<Tensor>> AsyncCheckpointSaver::stage_locked(
    const std::map<std::string, std::shared_ptr<Tensor>>& tensors) {
    auto& buf = staging_[staging_idx_];
    buf.clear();
    for (const auto& kv : tensors) {
        if (!kv.second)
            throw std::runtime_error("[litetorch Error] async_checkpoint: null tensor for " + kv.first);
        if (kv.second->device.type == DeviceType::CPU)
            buf[kv.first] = kv.second->clone();
        else
            buf[kv.first] = kv.second->to(Device(DeviceType::CPU, 0));
    }
    staging_idx_ ^= 1;
    return buf;
}

void AsyncCheckpointSaver::save_async(
    const std::map<std::string, std::shared_ptr<Tensor>>& tensors,
    const std::string& filepath) {
    std::unique_lock<std::mutex> lk(mu_);
    if (busy_)
        throw std::runtime_error("[litetorch Error] async_checkpoint: a checkpoint is already in-flight; call wait() before starting another");
    if (job_error_) {
        auto e = job_error_;
        job_error_ = nullptr;
        lk.unlock();
        std::rethrow_exception(e);
    }
    if (tensors.empty())
        throw std::runtime_error("[litetorch Error] async_checkpoint: empty tensor map");
    if (filepath.empty())
        throw std::runtime_error("[litetorch Error] async_checkpoint: empty filepath");
    auto staged = stage_locked(tensors);
    job_.staged = std::move(staged);
    job_.filepath = filepath;
    busy_ = true;
    lk.unlock();
    cv_.notify_one();
}

void AsyncCheckpointSaver::wait() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] { return !busy_; });
    if (job_error_) {
        auto e = job_error_;
        job_error_ = nullptr;
        lk.unlock();
        std::rethrow_exception(e);
    }
}

bool AsyncCheckpointSaver::is_saving() const {
    std::lock_guard<std::mutex> lk(mu_);
    return busy_;
}

void AsyncCheckpointSaver::worker_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
        cv_.wait(lk, [this] { return stop_ || busy_; });
        if (stop_) return;
        Job job = std::move(job_);
        lk.unlock();
        std::exception_ptr err;
        try {
            save_safetensors(job.staged, job.filepath);
        } catch (...) {
            err = std::current_exception();
        }
        job.staged.clear();
        lk.lock();
        job_error_ = err;
        busy_ = false;
        cv_.notify_all();
    }
}

}
}
