#include "gpu_common.h"

static thread_local void* s_temp_storage = nullptr;
static thread_local size_t s_temp_storage_allocated = 0;

static inline void* get_scratch_storage(size_t required_bytes) {
    if (required_bytes == 0) return nullptr;
    if (required_bytes > s_temp_storage_allocated) {
        if (s_temp_storage) {
            GPU_API(Free)(s_temp_storage);
            s_temp_storage = nullptr;
            s_temp_storage_allocated = 0;
        }
        size_t alloc_bytes = required_bytes + (required_bytes >> 1);
        if (GPU_API(Malloc)(&s_temp_storage, alloc_bytes) == GPU_API(Success)) {
            s_temp_storage_allocated = alloc_bytes;
        } else {
            GPU_API(Malloc)(&s_temp_storage, required_bytes);
            s_temp_storage_allocated = required_bytes;
        }
    }
    return s_temp_storage;
}

extern "C" void gpu_sum_forward(void* A, int a_off, void* B, int b_off, int size) {
    if (size <= 0) return;
    float* d_in = (float*)A + a_off;
    float* d_out = (float*)B + b_off;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
#ifndef __HIP_PLATFORM_AMD__
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, size, g_compute_stream);
    d_temp_storage = get_scratch_storage(temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, size, g_compute_stream);
#else
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, 0.0f, size, rocprim::plus<float>(), g_compute_stream);
    d_temp_storage = get_scratch_storage(temp_storage_bytes);
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, 0.0f, size, rocprim::plus<float>(), g_compute_stream);
#endif
}

extern "C" void gpu_max_forward(void* A, int a_off, void* B, int b_off, int size) {
    if (size <= 0) return;
    float* d_in = (float*)A + a_off;
    float* d_out = (float*)B + b_off;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
#ifndef __HIP_PLATFORM_AMD__
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, size, g_compute_stream);
    d_temp_storage = get_scratch_storage(temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, size, g_compute_stream);
#else
    float initial_value = -1e38f;
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, initial_value, size, rocprim::maximum<float>(), g_compute_stream);
    d_temp_storage = get_scratch_storage(temp_storage_bytes);
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, initial_value, size, rocprim::maximum<float>(), g_compute_stream);
#endif
}
