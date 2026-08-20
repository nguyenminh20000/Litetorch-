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

private:
    CachingAllocator() = default;
    ~CachingAllocator();

    std::mutex mutex_;
    std::multimap<size_t, void*> free_cpu_blocks_;
    std::map<void*, size_t> allocated_cpu_blocks_;

    std::multimap<size_t, void*> free_gpu_blocks_;
    std::map<void*, size_t> allocated_gpu_blocks_;
};

}

#endif
