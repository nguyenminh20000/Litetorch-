#ifndef LITETORCH_ALLOCATOR_H
#define LITETORCH_ALLOCATOR_H

#include <cstddef>
#include <mutex>
#include <map>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace litetorch {

class CachingAllocator {
public:
    static CachingAllocator& get();

    void* allocate_cpu(size_t size);
    void free_cpu(void* ptr);

    void* allocate_gpu(size_t size);
    void free_gpu(void* ptr);
    void free_gpu_now(void* ptr);

    void empty_cache();
    void set_max_cpu_cache_size(size_t bytes);
    void set_max_gpu_cache_size(size_t bytes);
    size_t get_cached_cpu_bytes() const;
    size_t get_cached_gpu_bytes() const;

private:
    CachingAllocator();
    ~CachingAllocator();

    mutable std::mutex mutex_;
    size_t max_cached_cpu_bytes_;
    size_t max_cached_gpu_bytes_;
    size_t cached_cpu_bytes_;
    std::multimap<size_t, void*> free_cpu_blocks_;
    std::map<void*, size_t> allocated_cpu_blocks_;
    std::unordered_set<void*> pinned_cpu_blocks_;
    void free_cpu_raw(void* ptr);
    void empty_cache_impl();

    size_t cached_gpu_bytes_;
    std::map<std::pair<int, size_t>, std::vector<void*>> free_gpu_blocks_;
    std::map<void*, std::pair<size_t, int>> allocated_gpu_blocks_;
    std::unordered_map<void*, void*> gpu_block_base_;
    std::unordered_map<void*, size_t> gpu_base_refs_;
    static constexpr size_t kGpuSplitAlign = 16;
    static constexpr size_t kGpuMinRemainder = 64;
    void gpu_discard_free_piece(void* ptr, size_t size, bool can_free);
};

}

#endif
