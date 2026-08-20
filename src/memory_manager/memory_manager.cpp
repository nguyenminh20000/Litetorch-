#include "litetorch/memory_manager.h"
#include "litetorch/tensor.h"
#include <algorithm>

namespace litetorch {

MemoryManager::MemoryManager() {}

MemoryManager& MemoryManager::get() {
    static MemoryManager instance;
    return instance;
}

void MemoryManager::set_gpu_limit(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    gpu_limit_ = bytes;
    evict_to_free_impl(0);
}

void MemoryManager::register_gpu(StorageImpl* storage) {
    std::lock_guard<std::mutex> lock(mutex_);
    register_gpu_impl(storage);
}

void MemoryManager::unregister_gpu(StorageImpl* storage) {
    std::lock_guard<std::mutex> lock(mutex_);
    unregister_gpu_impl(storage);
}

void MemoryManager::touch(StorageImpl* storage) {
    std::lock_guard<std::mutex> lock(mutex_);
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
    std::lock_guard<std::mutex> lock(mutex_);
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

void MemoryManager::evict_to_free_impl(size_t required_bytes) {
    auto it = lru_list_.begin();
    while (gpu_used_ + required_bytes > gpu_limit_ && it != lru_list_.end()) {
        StorageImpl* storage = *it;
        if (!storage->in_use && !storage->is_swapped) {
            auto next_it = std::next(it);
            {
                std::lock_guard<std::mutex> storage_lock(storage->storage_mutex_);
                storage->evict_impl();
            }
            it = next_it;
        } else {
            ++it;
        }
    }
}

}
