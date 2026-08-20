#include "gpu_common.h"

extern "C" void gpu_sum_forward(void* A, int a_off, void* B, int b_off, int size) {
    if (size <= 0) return;
    float* d_in = (float*)A + a_off;
    float* d_out = (float*)B + b_off;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
#ifndef __HIP_PLATFORM_AMD__
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, size, g_compute_stream);
    GPU_API(Malloc)(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, size, g_compute_stream);
    GPU_API(StreamSynchronize)(g_compute_stream);
    GPU_API(Free)(d_temp_storage);
#else
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, 0.0f, size, rocprim::plus<float>(), g_compute_stream);
    GPU_API(Malloc)(&d_temp_storage, temp_storage_bytes);
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, 0.0f, size, rocprim::plus<float>(), g_compute_stream);
    GPU_API(StreamSynchronize)(g_compute_stream);
    GPU_API(Free)(d_temp_storage);
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
    GPU_API(Malloc)(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, size, g_compute_stream);
    GPU_API(StreamSynchronize)(g_compute_stream);
    GPU_API(Free)(d_temp_storage);
#else
    float initial_value = -1e38f;
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, initial_value, size, rocprim::maximum<float>(), g_compute_stream);
    GPU_API(Malloc)(&d_temp_storage, temp_storage_bytes);
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, initial_value, size, rocprim::maximum<float>(), g_compute_stream);
    GPU_API(StreamSynchronize)(g_compute_stream);
    GPU_API(Free)(d_temp_storage);
#endif
}
