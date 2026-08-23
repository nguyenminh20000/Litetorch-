#include "litetorch/allocator.h"
#include "litetorch/backend.h"
#include <cstdlib>
#include <cstring>

namespace litetorch {

CachingAllocator::CachingAllocator()
    : max_cached_cpu_bytes_(128 * 1024 * 1024),
      cached_cpu_bytes_(0),
      cached_gpu_bytes_(0) {}

CachingAllocator& CachingAllocator::get() {
    static CachingAllocator instance;
    return instance;
}

CachingAllocator::~CachingAllocator() {
    empty_cache();
}

void CachingAllocator::set_max_cpu_cache_size(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_cached_cpu_bytes_ = bytes;
    while (cached_cpu_bytes_ > max_cached_cpu_bytes_ && !free_cpu_blocks_.empty()) {
        auto it = free_cpu_blocks_.begin();
        std::free(it->second);
        cached_cpu_bytes_ -= it->first;
        free_cpu_blocks_.erase(it);
    }
}

size_t CachingAllocator::get_cached_cpu_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cached_cpu_bytes_;
}

size_t CachingAllocator::get_cached_gpu_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cached_gpu_bytes_;
}

void* CachingAllocator::allocate_cpu(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = free_cpu_blocks_.lower_bound(size);
    if (it != free_cpu_blocks_.end() && it->first <= size * 2) {
        void* ptr = it->second;
        size_t actual_size = it->first;
        cached_cpu_bytes_ -= actual_size;
        free_cpu_blocks_.erase(it);
        allocated_cpu_blocks_[ptr] = actual_size;
        std::memset(ptr, 0, size);
        return ptr;
    }
    void* ptr = std::calloc(size, 1);
    if (ptr) {
        allocated_cpu_blocks_[ptr] = size;
    }
    return ptr;
}

void CachingAllocator::free_cpu(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocated_cpu_blocks_.find(ptr);
    if (it != allocated_cpu_blocks_.end()) {
        size_t size = it->second;
        allocated_cpu_blocks_.erase(it);
        if (cached_cpu_bytes_ + size <= max_cached_cpu_bytes_ && size < 64 * 1024 * 1024) {
            free_cpu_blocks_.insert({size, ptr});
            cached_cpu_bytes_ += size;
        } else {
            std::free(ptr);
        }
    } else {
        std::free(ptr);
    }
}

void* CachingAllocator::allocate_gpu(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = free_gpu_blocks_.lower_bound(size);
    if (it != free_gpu_blocks_.end() && it->first <= size * 2) {
        void* ptr = it->second;
        size_t actual_size = it->first;
        cached_gpu_bytes_ -= actual_size;
        free_gpu_blocks_.erase(it);
        allocated_gpu_blocks_[ptr] = actual_size;
        return ptr;
    }
    auto backend = BackendDispatcher::get().get_backend();
    void* ptr = nullptr;
    if (backend && backend->is_available()) {
        ptr = backend->allocate(size);
    }
    if (ptr) {
        allocated_gpu_blocks_[ptr] = size;
    }
    return ptr;
}

void CachingAllocator::free_gpu(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocated_gpu_blocks_.find(ptr);
    if (it != allocated_gpu_blocks_.end()) {
        size_t size = it->second;
        allocated_gpu_blocks_.erase(it);
        if (cached_gpu_bytes_ + size <= 512 * 1024 * 1024) {
            free_gpu_blocks_.insert({size, ptr});
            cached_gpu_bytes_ += size;
        } else {
            auto backend = BackendDispatcher::get().get_backend();
            if (backend && backend->is_available()) {
                backend->free(ptr);
            }
        }
    } else {
        auto backend = BackendDispatcher::get().get_backend();
        if (backend && backend->is_available()) {
            backend->free(ptr);
        }
    }
}

void CachingAllocator::empty_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : free_cpu_blocks_) {
        std::free(pair.second);
    }
    free_cpu_blocks_.clear();
    cached_cpu_bytes_ = 0;

    auto backend = BackendDispatcher::get().get_backend();
    if (backend && backend->is_available()) {
        for (auto& pair : free_gpu_blocks_) {
            backend->free(pair.second);
        }
    }
    free_gpu_blocks_.clear();
    cached_gpu_bytes_ = 0;
}

}
