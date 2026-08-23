#ifndef LITETORCH_ALLOCATOR_H
#define LITETORCH_ALLOCATOR_H

#include <cstddef>
#include <mutex>
#include <map>

namespace litetorch {

class CachingAllocator {
public:
    static CachingAllocator& get();

    void* allocate_cpu(size_t size);
    void free_cpu(void* ptr);

    void* allocate_gpu(size_t size);
    void free_gpu(void* ptr);

    void empty_cache();
    void set_max_cpu_cache_size(size_t bytes);
    size_t get_cached_cpu_bytes() const;
    size_t get_cached_gpu_bytes() const;

private:
    CachingAllocator();
    ~CachingAllocator();

    mutable std::mutex mutex_;
    size_t max_cached_cpu_bytes_;
    size_t cached_cpu_bytes_;
    std::multimap<size_t, void*> free_cpu_blocks_;
    std::map<void*, size_t> allocated_cpu_blocks_;

    size_t cached_gpu_bytes_;
    std::multimap<size_t, void*> free_gpu_blocks_;
    std::map<void*, size_t> allocated_gpu_blocks_;
};

}

#endif
