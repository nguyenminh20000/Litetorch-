#include "gpu_common.h"
#include <limits>
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

extern "C" void gpu_sum_forward(void* A, int64_t a_off, void* B, int64_t b_off, int64_t size) {
    auto_set_device(A);
    if (size <= 0) return;
    if (size > 2147483647LL) throw std::runtime_error("[litetorch Error] gpu_sum_forward: size exceeds INT_MAX");
    float* d_in = (float*)A + a_off;
    float* d_out = (float*)B + b_off;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
#ifndef __HIP_PLATFORM_AMD__
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, (int)size, dev_stream(current_device()));
    d_temp_storage = get_scratch_storage(temp_storage_bytes);
    cudaError_t st = cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, (int)size, dev_stream(current_device()));
    if (st != cudaSuccess) throw std::runtime_error(std::string("[litetorch Error] cub::DeviceReduce::Sum failed: ") + cudaGetErrorString(st));
#else
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, 0.0f, (int)size, rocprim::plus<float>(), dev_stream(current_device()));
    d_temp_storage = get_scratch_storage(temp_storage_bytes);
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, 0.0f, (int)size, rocprim::plus<float>(), dev_stream(current_device()));
#endif
}

struct NanAwareMaxOp {
    __host__ __device__ __forceinline__ float operator()(float a, float b) const {
        if (a != a || b != b) return a + b;
        return (a > b) ? a : b;
    }
};

extern "C" void gpu_max_forward(void* A, int64_t a_off, void* B, int64_t b_off, int64_t size) {
    auto_set_device(A);
    if (size <= 0) throw std::runtime_error("[litetorch Error] gpu_max_forward: max() of empty input");
    if (size > 2147483647LL) throw std::runtime_error("[litetorch Error] gpu_max_forward: size exceeds INT_MAX");
    float* d_in = (float*)A + a_off;
    float* d_out = (float*)B + b_off;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    float init = -std::numeric_limits<float>::infinity();
#ifndef __HIP_PLATFORM_AMD__
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, (int)size, NanAwareMaxOp(), init, dev_stream(current_device()));
    d_temp_storage = get_scratch_storage(temp_storage_bytes);
    cudaError_t st = cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, (int)size, NanAwareMaxOp(), init, dev_stream(current_device()));
    if (st != cudaSuccess) throw std::runtime_error(std::string("[litetorch Error] cub::DeviceReduce::Reduce (NaN-aware max) failed: ") + cudaGetErrorString(st));
#else
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, init, (int)size, NanAwareMaxOp(), dev_stream(current_device()));
    d_temp_storage = get_scratch_storage(temp_storage_bytes);
    rocprim::reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, init, (int)size, NanAwareMaxOp(), dev_stream(current_device()));
#endif
}

extern "C" void gpu_reduce_free_scratch() {
    for (auto& kv : s_scratch_map) {
        if (kv.second.ptr) {
            int dev = kv.first;
            int cur = current_device();
            if (cur != dev) GPU_API(SetDevice)(dev);
            GPU_API(Free)(kv.second.ptr);
            if (cur != dev) GPU_API(SetDevice)(cur);
            kv.second.ptr = nullptr;
            kv.second.bytes = 0;
        }
    }
    s_scratch_map.clear();
}
