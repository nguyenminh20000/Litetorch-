#pragma once
#include "litetorch/backend.h"
#include "litetorch/cl_backend.h"
#include "litetorch/allocator.h"
#include <string>
#include <vector>

namespace litetorch {

inline DeviceBackend* native_gpu_backend() {
    auto b = BackendDispatcher::get().get_backend();
    return (b && b->is_available()) ? b.get() : nullptr;
}

inline bool has_native_gpu() {
    return native_gpu_backend() != nullptr;
}

inline bool has_opencl_gpu() {
    return !has_native_gpu() && CLBackend::get().is_available();
}

// Strict backend separation for GPU memory ops.
// - allocate/free go through CachingAllocator on the native path: it is
//   (device,size)-keyed for multi-GPU and delegates to the native backend
//   internally. Calling native->allocate() directly would bypass that cache.
// - read/write/copy call the native backend directly when available;
//   CLBackend is used only in pure-OpenCL mode (no native backend).
// Device pointers are cl_mem bit patterns (void* under native).
inline void* rt_gpu_allocate(size_t bytes) {
    if (has_native_gpu()) return CachingAllocator::get().allocate_gpu(bytes);
    return (void*)CLBackend::get().allocate(bytes);
}

inline void rt_gpu_free(void* gpu_ptr) {
    if (!gpu_ptr) return;
    if (has_native_gpu()) { CachingAllocator::get().free_gpu(gpu_ptr); return; }
    CLBackend::get().free((cl_mem)gpu_ptr);
}

inline void rt_gpu_write(cl_mem dst, size_t bytes, const void* host_src, size_t dev_offset = 0) {
    if (!dst) return;
    if (auto n = native_gpu_backend()) { n->write((void*)dst, bytes, host_src, dev_offset); return; }
    CLBackend::get().write(dst, bytes, host_src, dev_offset);
}

inline void rt_gpu_read(cl_mem src, size_t bytes, void* host_dst, size_t dev_offset = 0) {
    if (!src) return;
    if (auto n = native_gpu_backend()) { n->read((void*)src, bytes, host_dst, dev_offset); return; }
    CLBackend::get().read(src, bytes, host_dst, dev_offset);
}

inline void rt_gpu_write_async(cl_mem dst, size_t bytes, const void* host_src, size_t dev_offset = 0) {
    if (!dst) return;
    if (auto n = native_gpu_backend()) { n->write_async((void*)dst, bytes, host_src, dev_offset); return; }
    CLBackend::get().write_async(dst, bytes, host_src, dev_offset);
}

inline void rt_gpu_read_async(cl_mem src, size_t bytes, void* host_dst, size_t dev_offset = 0) {
    if (!src) return;
    if (auto n = native_gpu_backend()) { n->read_async((void*)src, bytes, host_dst, dev_offset); return; }
    CLBackend::get().read_async(src, bytes, host_dst, dev_offset);
}

inline void rt_gpu_copy(cl_mem src, cl_mem dst, size_t bytes, size_t src_offset = 0, size_t dst_offset = 0) {
    if (!src || !dst) return;
    if (auto n = native_gpu_backend()) { n->copy((void*)src, (void*)dst, bytes, src_offset, dst_offset); return; }
    CLBackend::get().copy(src, dst, bytes, src_offset, dst_offset);
}

inline void* rt_gpu_get_kernel(const std::string& prog, const std::string& src, const std::string& name) {
    if (auto n = native_gpu_backend()) return n->get_kernel(prog, src, name);
    return (void*)CLBackend::get().get_kernel(prog, src, name);
}

inline void* rt_gpu_get_kernel_by_id(KernelID id) {
    if (auto n = native_gpu_backend()) {
        const char* name = CLBackend::kernel_name(id);
        return name ? n->get_kernel("", "", name) : nullptr;
    }
    return (void*)CLBackend::get().get_kernel(id);
}

inline void rt_gpu_launch(void* kernel, const std::vector<size_t>& gws, const std::vector<size_t>& lws,
                          const std::vector<void*>& args, const std::vector<size_t>& arg_sizes) {
    if (!kernel) return;
    if (auto n = native_gpu_backend()) { n->launch(kernel, gws, lws, args, arg_sizes); return; }
    CLBackend::get().launch((cl_kernel)kernel, gws, lws, args, arg_sizes);
}

}
