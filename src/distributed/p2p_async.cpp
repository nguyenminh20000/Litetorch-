#include "litetorch/p2p_async.h"
#include "litetorch/distributed.h"
#include "litetorch/tensor.h"

#include <future>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <stdexcept>

namespace litetorch {
namespace distributed {

namespace {

class FutureWork : public Work {
public:
    explicit FutureWork(std::shared_future<void> f) : fut_(std::move(f)) {}
    void wait() override {
        fut_.wait();
        fut_.get();
    }
    bool is_completed() override {
        return fut_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
private:
    std::shared_future<void> fut_;
};

class CompletedWork : public Work {
public:
    void wait() override {}
    bool is_completed() override { return true; }
};

class P2PExecutor {
public:
    static P2PExecutor& instance() {
        static P2PExecutor e;
        return e;
    }

    std::shared_future<void> submit(std::function<void()> fn) {
        auto prom = std::make_shared<std::promise<void>>();
        std::shared_future<void> fut = prom->get_future().share();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_) {
                prom->set_exception(std::make_exception_ptr(
                    std::runtime_error("p2p_async: executor stopped")));
                return fut;
            }
            queue_.emplace(std::move(fn), std::move(prom));
        }
        cv_.notify_one();
        return fut;
    }

    ~P2PExecutor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
            while (!queue_.empty()) {
                queue_.front().second->set_exception(std::make_exception_ptr(
                    std::runtime_error("p2p_async: executor shutdown with pending op")));
                queue_.pop();
            }
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

private:
    P2PExecutor() {
        worker_ = std::thread([this]() {
            while (true) {
                std::pair<std::function<void()>, std::shared_ptr<std::promise<void>>> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this]() { return stopped_ || !queue_.empty(); });
                    if (stopped_ && queue_.empty()) return;
                    task = std::move(queue_.front());
                    queue_.pop();
                }
                try {
                    task.first();
                    task.second->set_value();
                } catch (...) {
                    task.second->set_exception(std::current_exception());
                }
            }
        });
    }

    P2PExecutor(const P2PExecutor&) = delete;
    P2PExecutor& operator=(const P2PExecutor&) = delete;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::pair<std::function<void()>, std::shared_ptr<std::promise<void>>>> queue_;
    std::thread worker_;
    bool stopped_ = false;
};

bool p2p_noop(int peer) {
    auto& pg = ProcessGroup::get();
    return !pg.is_initialized() || pg.get_world_size() <= 1 || pg.get_rank() == peer;
}

}

std::shared_ptr<Work> isend(std::shared_ptr<Tensor> tensor, int dst) {
    if (p2p_noop(dst)) return std::make_shared<CompletedWork>();
    auto snap = tensor->clone();
    auto fut = P2PExecutor::instance().submit([snap, dst]() {
        ProcessGroup::get().send_tensor(snap, dst);
    });
    return std::make_shared<FutureWork>(std::move(fut));
}

std::shared_ptr<Work> irecv(std::shared_ptr<Tensor> tensor, int src) {
    if (p2p_noop(src)) return std::make_shared<CompletedWork>();
    auto fut = P2PExecutor::instance().submit([tensor, src]() {
        ProcessGroup::get().recv_tensor(tensor, src);
    });
    return std::make_shared<FutureWork>(std::move(fut));
}

}
}
