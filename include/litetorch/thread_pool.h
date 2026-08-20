#ifndef LITETORCH_THREAD_POOL_H
#define LITETORCH_THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <algorithm>

namespace litetorch {

class ThreadPool {
public:
    static ThreadPool& get() {
        static ThreadPool instance;
        return instance;
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    template <typename F>
    void parallel_for(int64_t start, int64_t end, F&& func) {
        int64_t total = end - start;
        if (total <= 0) return;

        int num_threads = workers.size();
        if (total < 50000 || num_threads <= 1) {
            for (int64_t i = start; i < end; ++i) {
                func(i);
            }
            return;
        }

        int64_t chunk_size = (total + num_threads - 1) / num_threads;
        
        struct SyncContext {
            std::atomic<int> remaining{0};
            std::mutex mutex;
            std::condition_variable cv;
        };
        auto sync = std::make_shared<SyncContext>();

        int64_t actual_chunks = 0;
        for (int64_t t_start = start; t_start < end; t_start += chunk_size) {
            actual_chunks++;
        }
        
        sync->remaining.store(actual_chunks - 1);

        int64_t current_start = start;
        int64_t calling_start = -1;
        int64_t calling_end = -1;

        for (int t = 0; t < actual_chunks; ++t) {
            int64_t t_start = current_start;
            int64_t t_end = std::min(t_start + chunk_size, end);
            current_start = t_end;

            if (t == actual_chunks - 1) {
                calling_start = t_start;
                calling_end = t_end;
            } else {
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    if (stop) {
                        for (int64_t i = t_start; i < t_end; ++i) {
                            func(i);
                        }
                        sync->remaining.fetch_sub(1);
                        continue;
                    }
                    tasks.emplace([t_start, t_end, &func, sync]() {
                        for (int64_t i = t_start; i < t_end; ++i) {
                            func(i);
                        }
                        if (sync->remaining.fetch_sub(1) == 1) {
                            std::lock_guard<std::mutex> lk(sync->mutex);
                            sync->cv.notify_one();
                        }
                    });
                }
                condition.notify_one();
            }
        }

        if (calling_start != -1) {
            for (int64_t i = calling_start; i < calling_end; ++i) {
                func(i);
            }
        }

        if (sync->remaining.load() > 0) {
            std::unique_lock<std::mutex> lk(sync->mutex);
            sync->cv.wait(lk, [sync] { return sync->remaining.load() == 0; });
        }
    }

private:
    ThreadPool() : stop(false) {
        size_t threads = std::thread::hardware_concurrency();
        if (threads == 0) threads = 4;
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

}

#endif
