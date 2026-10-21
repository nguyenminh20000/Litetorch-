#include "litetorch/memory_manager.h"
#include "litetorch/tensor.h"
#include "litetorch/allocator.h"
#include "litetorch/backend.h"
#include <algorithm>

namespace litetorch {

MemoryManager::MemoryManager() {}

MemoryManager& MemoryManager::get() {
    static MemoryManager* instance = new MemoryManager();
    return *instance;
}

void MemoryManager::set_gpu_limit(size_t bytes) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    gpu_limit_ = bytes;
    ++set_gpu_limit_calls_;
    last_set_gpu_limit_ = bytes;
    evict_to_free_impl(0);
}

void MemoryManager::register_gpu(StorageImpl* storage) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    register_gpu_impl(storage);
}

void MemoryManager::unregister_gpu(StorageImpl* storage) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    unregister_gpu_impl(storage);
}

void MemoryManager::touch(StorageImpl* storage) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    touch_impl(storage);
}

void MemoryManager::ensure_gpu(StorageImpl* storage) {
    if (!storage) return;
    bool swapped = false;
    {
        std::lock_guard<std::mutex> lock(storage->storage_mutex_);
        swapped = storage->is_swapped;
    }
    if (swapped) {
        storage->swap_in();
    } else {
        touch(storage);
    }
}

void MemoryManager::evict_to_free(size_t required_bytes) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    evict_to_free_impl(required_bytes);
}

void MemoryManager::register_gpu_impl(StorageImpl* storage) {
    if (active_gpu_storages_.find(storage) != active_gpu_storages_.end()) {
        return;
    }

    size_t size_bytes = storage->size * storage->element_size();
    evict_to_free_impl(size_bytes);

    active_gpu_storages_.insert(storage);
    lru_list_.push_back(storage);
    storage->lru_iterator = std::prev(lru_list_.end());
    storage->has_lru_iterator = true;
    gpu_used_ += size_bytes;
}

void MemoryManager::unregister_gpu_impl(StorageImpl* storage) {
    auto it = active_gpu_storages_.find(storage);
    if (it != active_gpu_storages_.end()) {
        active_gpu_storages_.erase(it);
        if (storage->has_lru_iterator) {
            lru_list_.erase(storage->lru_iterator);
            storage->has_lru_iterator = false;
        }
        gpu_used_ -= storage->size * storage->element_size();
    }
}

void MemoryManager::touch_impl(StorageImpl* storage) {
    if (active_gpu_storages_.find(storage) != active_gpu_storages_.end()) {
        if (storage->has_lru_iterator) {
            lru_list_.erase(storage->lru_iterator);
        }
        lru_list_.push_back(storage);
        storage->lru_iterator = std::prev(lru_list_.end());
        storage->has_lru_iterator = true;
    }
}

size_t MemoryManager::evict_to_free_impl(size_t required_bytes) {
    size_t evicted = 0;
    auto it = lru_list_.begin();
    while (gpu_used_ + required_bytes > gpu_limit_ && it != lru_list_.end()) {
        StorageImpl* storage = *it;
        if (!storage->in_use && !storage->is_swapped) {
            auto next_it = std::next(it);
            size_t bytes = storage->size * storage->element_size();
            {
                std::lock_guard<std::mutex> storage_lock(storage->storage_mutex_);
                storage->evict_impl();
            }
            evicted += bytes;
            it = next_it;
        } else {
            ++it;
        }
    }
    last_evicted_bytes_ = evicted;
    last_eviction_required_ = required_bytes;
    return evicted;
}

size_t MemoryManager::get_lru_size() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return lru_list_.size();
}

size_t MemoryManager::get_active_gpu_storages() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return active_gpu_storages_.size();
}

std::string MemoryManager::gpu_oom_diagnostics(size_t requested_bytes) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    size_t pool_cached = 0;
    std::string backend_err;
    auto backend = BackendDispatcher::get().get_backend();
    if (backend) {
        pool_cached = backend->get_pool_cached_bytes();
        backend_err = backend->last_alloc_error();
    }
    std::string s = "requested=" + std::to_string(requested_bytes) + "B"
        + " gpu_used=" + std::to_string(gpu_used_.load()) + "B"
        + " gpu_limit=" + std::to_string(gpu_limit_.load()) + "B"
        + " set_gpu_limit_calls=" + std::to_string(set_gpu_limit_calls_.load())
        + " set_gpu_limit_last=" + std::to_string(last_set_gpu_limit_.load()) + "B"
        + " allocator_cached=" + std::to_string(CachingAllocator::get().get_cached_gpu_bytes()) + "B"
        + " backend_pool_cached=" + std::to_string(pool_cached) + "B"
        + " lru_storages=" + std::to_string(lru_list_.size())
        + " active_storages=" + std::to_string(active_gpu_storages_.size())
        + " last_eviction_freed=" + std::to_string(last_evicted_bytes_.load()) + "B"
        + " last_eviction_required=" + std::to_string(last_eviction_required_.load()) + "B";
    if (!backend_err.empty()) s += " backend_error=" + backend_err;
    else s += " backend_error=<not reported by backend>";
    return s;
}

std::string MemoryManager::cpu_oom_diagnostics(size_t requested_bytes) const {
    return "requested=" + std::to_string(requested_bytes) + "B"
        + " allocator_cached=" + std::to_string(CachingAllocator::get().get_cached_cpu_bytes()) + "B";
}

}
