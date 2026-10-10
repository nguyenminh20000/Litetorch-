#include "gpu_common.h"
#include <unordered_map>

struct ReduceScratch {
    void* ptr = nullptr;
    size_t bytes = 0;
};
static thread_local std::unordered_map<int, ReduceScratch> s_scratch_map;

static inline void* get_scratch_storage(size_t required_bytes) {
    if (required_bytes == 0) return nullptr;
    int dev = current_device();
    ReduceScratch& sc = s_scratch_map[dev];
    if (required_bytes > sc.bytes) {
        if (sc.ptr) {
            GPU_API(StreamSynchronize)(dev_stream(dev));
            int cur = current_device();
            if (cur != dev) GPU_API(SetDevice)(dev);
            GPU_API(Free)(sc.ptr);
            if (cur != dev) GPU_API(SetDevice)(cur);
            sc.ptr = nullptr;
            sc.bytes = 0;
        }
        size_t alloc_bytes = required_bytes + (required_bytes >> 1);
        int cur = current_device();
        if (cur != dev) GPU_API(SetDevice)(dev);
        if (GPU_API(Malloc)(&sc.ptr, alloc_bytes) == GPU_API(Success)) {
            sc.bytes = alloc_bytes;
        } else if (GPU_API(Malloc)(&sc.ptr, required_bytes) == GPU_API(Success)) {
            sc.bytes = required_bytes;
        }
        if (cur != dev) GPU_API(SetDevice)(cur);
    }
    return sc.ptr;
}

extern "C" void gpu_sum_forward(void* A, int a_off, void* B, int b_off, int size) {
    auto_set_device(A);
    if (size <= 0) return;
    float* d_in = (float*)A + a_off;
    float* d_out = (float*)B + b_off;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
#ifndef __HIP_PLATFORM_AMD__
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, size, dev_stream(current_device()));
    d_temp_storage = get_scratch_storage(temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, size, dev_stream(current_device()));
#else
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, 0.0f, size, rocprim::plus<float>(), dev_stream(current_device()));
    d_temp_storage = get_scratch_storage(temp_storage_bytes);
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, 0.0f, size, rocprim::plus<float>(), dev_stream(current_device()));
#endif
}

extern "C" void gpu_max_forward(void* A, int a_off, void* B, int b_off, int size) {
    auto_set_device(A);
    if (size <= 0) return;
    float* d_in = (float*)A + a_off;
    float* d_out = (float*)B + b_off;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
#ifndef __HIP_PLATFORM_AMD__
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, size, dev_stream(current_device()));
    d_temp_storage = get_scratch_storage(temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, size, dev_stream(current_device()));
#else
    float initial_value = -1e38f;
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, initial_value, size, rocprim::maximum<float>(), dev_stream(current_device()));
    d_temp_storage = get_scratch_storage(temp_storage_bytes);
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, initial_value, size, rocprim::maximum<float>(), dev_stream(current_device()));
#endif
}
