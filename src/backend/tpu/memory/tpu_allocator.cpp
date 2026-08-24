#include "../common/tpu_common.h"
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace litetorch {
namespace tpu_internal {

void* tpu_hbm_allocate(size_t size) {
    if (size == 0) return nullptr;
    size_t alignment = TPU_HBM_ALIGNMENT;
    size_t total_size = size + alignment + sizeof(void*);
    void* raw = std::malloc(total_size);
    if (!raw) return nullptr;
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw) + sizeof(void*);
    uintptr_t aligned_addr = (addr + (alignment - 1)) & ~(alignment - 1);
    void** storage = reinterpret_cast<void**>(aligned_addr - sizeof(void*));
    *storage = raw;
    return reinterpret_cast<void*>(aligned_addr);
}

void tpu_hbm_free(void* ptr) {
    if (!ptr) return;
    void** storage = reinterpret_cast<void**>(reinterpret_cast<uintptr_t>(ptr) - sizeof(void*));
    std::free(*storage);
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
