#include "litetorch/allocator.h"
#include "litetorch/backend.h"
#include <cstdlib>
#include <cstring>

namespace litetorch {

CachingAllocator& CachingAllocator::get() {
    static CachingAllocator instance;
    return instance;
}

CachingAllocator::~CachingAllocator() {
    empty_cache();
}

void* CachingAllocator::allocate_cpu(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = free_cpu_blocks_.lower_bound(size);
    if (it != free_cpu_blocks_.end() && it->first <= size * 2) {
        void* ptr = it->second;
        size_t actual_size = it->first;
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
        free_cpu_blocks_.insert({it->second, ptr});
        allocated_cpu_blocks_.erase(it);
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
        free_gpu_blocks_.insert({it->second, ptr});
        allocated_gpu_blocks_.erase(it);
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

    auto backend = BackendDispatcher::get().get_backend();
    if (backend && backend->is_available()) {
        for (auto& pair : free_gpu_blocks_) {
            backend->free(pair.second);
        }
    }
    free_gpu_blocks_.clear();
}

}
