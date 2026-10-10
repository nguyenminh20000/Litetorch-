#include "litetorch/allocator.h"
#include "litetorch/backend.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace litetorch {
namespace {

struct PinnedHostFns {
    void* handle = nullptr;
    int (*host_alloc)(void**, size_t, unsigned int) = nullptr;
    int (*host_free)(void*) = nullptr;
    bool tried = false;
};

PinnedHostFns& pinned_host_fns() {
    static PinnedHostFns f;
    return f;
}

bool pinned_host_available() {
    PinnedHostFns& f = pinned_host_fns();
    if (f.tried) return f.host_alloc != nullptr;
    f.tried = true;
    struct LibEntry { const char* lib; const char* alloc_sym; const char* free_sym; };
    static const LibEntry entries[] = {
        {"libcudart.so", "cudaHostAlloc", "cudaFreeHost"},
        {"libcudart.so.1", "cudaHostAlloc", "cudaFreeHost"},
        {"libcudart.so.12", "cudaHostAlloc", "cudaFreeHost"},
        {"libcudart.so.13", "cudaHostAlloc", "cudaFreeHost"},
        {"libamdhip64.so", "hipHostMalloc", "hipHostFree"},
        {"libamdhip64.so.5", "hipHostMalloc", "hipHostFree"},
        {"libamdhip64.so.6", "hipHostMalloc", "hipHostFree"},
        {nullptr, nullptr, nullptr},
    };
    for (int i = 0; entries[i].lib; ++i) {
        f.handle = dlopen(entries[i].lib, RTLD_NOW | RTLD_NOLOAD);
        if (!f.handle) f.handle = dlopen(entries[i].lib, RTLD_NOW);
        if (!f.handle) continue;
        f.host_alloc = reinterpret_cast<int(*)(void**, size_t, unsigned int)>(dlsym(f.handle, entries[i].alloc_sym));
        f.host_free = reinterpret_cast<int(*)(void*)>(dlsym(f.handle, entries[i].free_sym));
        if (f.host_alloc && f.host_free) return true;
        f.host_alloc = nullptr;
        f.host_free = nullptr;
    }
    return false;
}

constexpr size_t kPinnedHostThreshold = 64 * 1024;
constexpr unsigned int kHostAllocPortable = 0x01;

}

CachingAllocator::CachingAllocator()
    : max_cached_cpu_bytes_(128 * 1024 * 1024),
      cached_cpu_bytes_(0),
      cached_gpu_bytes_(0) {}

CachingAllocator& CachingAllocator::get() {
    static CachingAllocator* instance = new CachingAllocator();
    return *instance;
}

CachingAllocator::~CachingAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : free_cpu_blocks_) {
        free_cpu_raw(pair.second);
    }
    free_cpu_blocks_.clear();
    cached_cpu_bytes_ = 0;
    free_gpu_blocks_.clear();
    cached_gpu_bytes_ = 0;
}

void CachingAllocator::set_max_cpu_cache_size(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_cached_cpu_bytes_ = bytes;
    while (cached_cpu_bytes_ > max_cached_cpu_bytes_ && !free_cpu_blocks_.empty()) {
        auto it = free_cpu_blocks_.begin();
        free_cpu_raw(it->second);
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
    if (size >= kPinnedHostThreshold && pinned_host_available()) {
        PinnedHostFns& f = pinned_host_fns();
        void* ptr = nullptr;
        if (f.host_alloc(&ptr, size, kHostAllocPortable) == 0 && ptr) {
            std::memset(ptr, 0, size);
            allocated_cpu_blocks_[ptr] = size;
            pinned_cpu_blocks_.insert(ptr);
            return ptr;
        }
    }
    void* ptr = std::calloc(size, 1);
    if (ptr) {
        allocated_cpu_blocks_[ptr] = size;
    }
    return ptr;
}

void CachingAllocator::free_cpu_raw(void* ptr) {
    auto pit = pinned_cpu_blocks_.find(ptr);
    if (pit != pinned_cpu_blocks_.end()) {
        pinned_cpu_blocks_.erase(pit);
        PinnedHostFns& f = pinned_host_fns();
        if (f.host_free) f.host_free(ptr);
        else std::free(ptr);
    } else {
        std::free(ptr);
    }
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
            free_cpu_raw(ptr);
        }
    } else {
        for (auto f = free_cpu_blocks_.begin(); f != free_cpu_blocks_.end(); ++f) {
            if (f->second == ptr) return;
        }
        free_cpu_raw(ptr);
    }
}

static int current_gpu_device() {
    auto backend = BackendDispatcher::get().get_backend();
    if (backend && backend->is_available()) return backend->get_device();
    return 0;
}

void* CachingAllocator::allocate_gpu(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    int dev = current_gpu_device();
    auto it = free_gpu_blocks_.lower_bound({dev, size});
    if (it != free_gpu_blocks_.end() && it->first.first == dev && it->first.second <= size * 2 && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        size_t actual_size = it->first.second;
        cached_gpu_bytes_ -= actual_size;
        if (it->second.empty()) free_gpu_blocks_.erase(it);
        allocated_gpu_blocks_[ptr] = {actual_size, dev};
        return ptr;
    }
    auto backend = BackendDispatcher::get().get_backend();
    void* ptr = nullptr;
    if (backend && backend->is_available()) {
        ptr = backend->allocate(size);
    }
    if (ptr) {
        allocated_gpu_blocks_[ptr] = {size, dev};
    }
    return ptr;
}

void CachingAllocator::free_gpu(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocated_gpu_blocks_.find(ptr);
    if (it != allocated_gpu_blocks_.end()) {
        size_t size = it->second.first;
        int dev = it->second.second;
        allocated_gpu_blocks_.erase(it);
        if (cached_gpu_bytes_ + size <= 512 * 1024 * 1024) {
            free_gpu_blocks_[{dev, size}].push_back(ptr);
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
        free_cpu_raw(pair.second);
    }
    free_cpu_blocks_.clear();
    cached_cpu_bytes_ = 0;

    auto backend = BackendDispatcher::get().get_backend();
    if (backend && backend->is_available()) {
        for (auto& pair : free_gpu_blocks_) {
            for (void* p : pair.second) backend->free(p);
        }
    }
    free_gpu_blocks_.clear();
    cached_gpu_bytes_ = 0;
}

}
