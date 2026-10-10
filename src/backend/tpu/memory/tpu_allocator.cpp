#include "../common/tpu_common.h"
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <vector>

namespace litetorch {
namespace tpu_internal {

namespace {

constexpr size_t TPU_ALLOC_HEADER = sizeof(void*) + sizeof(size_t);
constexpr size_t TPU_ALLOC_CACHE_CAP = 256ULL * 1024ULL * 1024ULL;

struct TpuFreeBlock {
    void* raw;
    void* aligned;
    size_t total;
};

std::mutex g_tpu_alloc_mutex;
std::vector<TpuFreeBlock> g_tpu_free_list;
size_t g_tpu_cached_bytes = 0;

}

void* tpu_hbm_allocate(size_t size) {
    if (size == 0) return nullptr;
    size_t alignment = TPU_HBM_ALIGNMENT;
    size_t total_size = size + alignment + TPU_ALLOC_HEADER;
    {
        std::lock_guard<std::mutex> lock(g_tpu_alloc_mutex);
        for (size_t i = 0; i < g_tpu_free_list.size(); ++i) {
            if (g_tpu_free_list[i].total >= total_size) {
                void* aligned = g_tpu_free_list[i].aligned;
                g_tpu_cached_bytes -= g_tpu_free_list[i].total;
                g_tpu_free_list[i] = g_tpu_free_list.back();
                g_tpu_free_list.pop_back();
                return aligned;
            }
        }
    }
    void* raw = std::malloc(total_size);
    if (!raw) return nullptr;
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw) + TPU_ALLOC_HEADER;
    uintptr_t aligned_addr = (addr + (alignment - 1)) & ~(alignment - 1);
    void** storage = reinterpret_cast<void**>(aligned_addr - TPU_ALLOC_HEADER);
    storage[0] = raw;
    reinterpret_cast<size_t*>(storage)[1] = total_size;
    return reinterpret_cast<void*>(aligned_addr);
}

void tpu_hbm_free(void* ptr) {
    if (!ptr) return;
    uintptr_t aligned_addr = reinterpret_cast<uintptr_t>(ptr);
    void** storage = reinterpret_cast<void**>(aligned_addr - TPU_ALLOC_HEADER);
    void* raw = storage[0];
    size_t total_size = reinterpret_cast<size_t*>(storage)[1];
    std::lock_guard<std::mutex> lock(g_tpu_alloc_mutex);
    if (g_tpu_cached_bytes + total_size > TPU_ALLOC_CACHE_CAP) {
        std::free(raw);
        return;
    }
    g_tpu_cached_bytes += total_size;
    g_tpu_free_list.push_back({raw, ptr, total_size});
}

void tpu_hbm_read(void* ptr, size_t size, void* host_ptr, size_t offset) {
    if (!ptr || !host_ptr || size == 0) return;
    const char* src = reinterpret_cast<const char*>(ptr) + offset;
    std::memcpy(host_ptr, src, size);
}

void tpu_hbm_write(void* ptr, size_t size, const void* host_ptr, size_t offset) {
    if (!ptr || !host_ptr || size == 0) return;
    char* dst = reinterpret_cast<char*>(ptr) + offset;
    std::memcpy(dst, host_ptr, size);
}

void tpu_hbm_copy(void* src, void* dst, size_t size, size_t src_offset, size_t dst_offset) {
    if (!src || !dst || size == 0) return;
    const char* s = reinterpret_cast<const char*>(src) + src_offset;
    char* d = reinterpret_cast<char*>(dst) + dst_offset;
    std::memcpy(d, s, size);
}

}
}
