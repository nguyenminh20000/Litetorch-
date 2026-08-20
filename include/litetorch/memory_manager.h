#ifndef LITETORCH_MEMORY_MANAGER_H
#define LITETORCH_MEMORY_MANAGER_H

#include "litetorch/device.h"
#include "litetorch/cl_backend.h"
#include <vector>
#include <unordered_set>
#include <list>
#include <cstddef>
#include <mutex>

namespace litetorch {

class StorageImpl;

class MemoryManager {
public:
    static MemoryManager& get();

    void set_gpu_limit(size_t bytes);
    size_t get_gpu_limit() const { return gpu_limit_; }
    size_t get_gpu_used() const { return gpu_used_; }

    void register_gpu(StorageImpl* storage);
    void unregister_gpu(StorageImpl* storage);
    void touch(StorageImpl* storage);
    void ensure_gpu(StorageImpl* storage);
    void evict_to_free(size_t required_bytes);

    void register_gpu_impl(StorageImpl* storage);
    void unregister_gpu_impl(StorageImpl* storage);
    void touch_impl(StorageImpl* storage);
    void evict_to_free_impl(size_t required_bytes);

    std::mutex& get_mutex() { return mutex_; }

private:
    MemoryManager();
    ~MemoryManager() = default;

    size_t gpu_limit_ = 512 * 1024 * 1024;
    size_t gpu_used_ = 0;

    std::list<StorageImpl*> lru_list_;
    std::unordered_set<StorageImpl*> active_gpu_storages_;
    mutable std::mutex mutex_;
};

}

#endif
