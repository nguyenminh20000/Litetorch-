#ifndef LITETORCH_MEMORY_MANAGER_H
#define LITETORCH_MEMORY_MANAGER_H

#include "litetorch/device.h"
#include "litetorch/cl_backend.h"
#include <string>
#include <vector>
#include <unordered_set>
#include <list>
#include <cstddef>
#include <mutex>
#include <atomic>

namespace litetorch {

class StorageImpl;

class MemoryManager {
public:
    static MemoryManager& get();

    void set_gpu_limit(size_t bytes);
    size_t get_gpu_limit() const { return gpu_limit_; }
    size_t get_gpu_used() const { return gpu_used_; }
    size_t get_lru_size() const;
    size_t get_active_gpu_storages() const;
    size_t get_last_evicted_bytes() const { return last_evicted_bytes_; }
    size_t get_last_eviction_required() const { return last_eviction_required_; }
    std::string gpu_oom_diagnostics(size_t requested_bytes) const;
    std::string cpu_oom_diagnostics(size_t requested_bytes) const;

    void register_gpu(StorageImpl* storage);
    void unregister_gpu(StorageImpl* storage);
    void touch(StorageImpl* storage);
    void ensure_gpu(StorageImpl* storage);
    void evict_to_free(size_t required_bytes);

    void register_gpu_impl(StorageImpl* storage);
    void unregister_gpu_impl(StorageImpl* storage);
    void touch_impl(StorageImpl* storage);
    size_t evict_to_free_impl(size_t required_bytes);

    std::recursive_mutex& get_mutex() { return mutex_; }

private:
    MemoryManager();
    ~MemoryManager() = default;

    std::atomic<size_t> gpu_limit_{512 * 1024 * 1024};
    std::atomic<size_t> gpu_used_{0};
    std::atomic<size_t> set_gpu_limit_calls_{0};
    std::atomic<size_t> last_set_gpu_limit_{0};
    std::atomic<size_t> last_evicted_bytes_{0};
    std::atomic<size_t> last_eviction_required_{0};

    std::list<StorageImpl*> lru_list_;
    std::unordered_set<StorageImpl*> active_gpu_storages_;
    mutable std::recursive_mutex mutex_;
};

}

#endif
