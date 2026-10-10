#include "common/gpu_common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <mutex>
#include <map>
#include <unordered_map>
#include <vector>
static std::mutex g_dev_stream_mutex;
static std::vector<GPU_API(Stream_t)> g_dev_streams;

static GPU_API(Stream_t) dev_stream(int dev) {
    if (dev < 0) dev = 0;
    std::lock_guard<std::mutex> lock(g_dev_stream_mutex);
    if ((int)g_dev_streams.size() <= dev) g_dev_streams.resize(dev + 1, nullptr);
    if (!g_dev_streams[dev]) {
        int cur = 0;
        GPU_API(GetDevice)(&cur);
        GPU_API(SetDevice)(dev);
        GPU_API(StreamCreate)(&g_dev_streams[dev]);
        GPU_API(SetDevice)(cur);
    }
    return g_dev_streams[dev];
}

static inline int current_device() {
    int dev = 0;
    GPU_API(GetDevice)(&dev);
    return dev;
}

static inline int device_of_ptr_strict(const void* ptr) {
    if (!ptr) return -1;
#ifndef __HIP_PLATFORM_AMD__
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, ptr) == cudaSuccess && attr.type == cudaMemoryTypeDevice)
        return attr.device;
#else
    hipPointerAttribute_t attr;
    if (hipPointerGetAttributes(&attr, ptr) == hipSuccess && attr.memoryType == hipMemoryTypeDevice)
        return attr.device;
#endif
    return -1;
}

static inline int ptr_device(const void* ptr) {
    int dev = device_of_ptr_strict(ptr);
    return dev >= 0 ? dev : current_device();
}

static inline void auto_set_device(const void* ptr) {
    int dev = ptr_device(ptr);
    if (dev != current_device()) GPU_API(SetDevice)(dev);
}

static inline int launch_device(void** args, int arg_count) {
    int n = arg_count < 4 ? arg_count : 4;
    for (int i = 0; i < n; i++) {
        if (!args || !args[i]) continue;
        void* p = *(void**)args[i];
        int dev = device_of_ptr_strict(p);
        if (dev >= 0) return dev;
    }
    return current_device();
}

#include "math/gemm.cu"
#include "math/reduction.cu"
#include "elementwise/elementwise_ops.cu"
#include "optim/optimizers.cu"
#include "nn/conv_kernels.cu"
#include "nn/pool_kernels.cu"
#include "nn/softmax_kernels.cu"
#include "nn/norm_kernels.cu"
#include "nn/misc_kernels.cu"
#include "nn/flash_attention.cu"

GPU_API(Stream_t) g_compute_stream = nullptr;
GPU_API(Stream_t) g_comm_stream = nullptr;
GPU_API(Stream_t) g_h2d_stream = nullptr;
GPU_API(Stream_t) g_d2h_stream = nullptr;

#ifndef __HIP_PLATFORM_AMD__
static bool g_tf32_enabled = true;

extern "C" void gpu_set_tf32_enabled(bool enabled) {
    g_tf32_enabled = enabled;
    cublasHandle_t handle = get_cublas_handle();
    if (handle) {
        cublasMath_t mode = enabled ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
        cublasSetMathMode(handle, mode);
    }
}

extern "C" bool gpu_is_tf32_enabled() {
    return g_tf32_enabled;
}

cublasHandle_t get_cublas_handle(const void* ref) {
    thread_local std::unordered_map<int, cublasHandle_t> handles;
    auto_set_device(ref);
    int dev = current_device();
    auto it = handles.find(dev);
    if (it != handles.end()) return it->second;
    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    cublasSetStream(handle, dev_stream(dev));
    cublasMath_t mode = g_tf32_enabled ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasSetMathMode(handle, mode);
    handles[dev] = handle;
    return handle;
}
cublasLtHandle_t get_cublaslt_handle() {
    thread_local cublasLtHandle_t handle = nullptr;
    if (!handle) {
        cublasLtCreate(&handle);
    }
    return handle;
}
lt_cudnnHandle_t get_cudnn_handle(const void* ref) {
    thread_local std::unordered_map<int, lt_cudnnHandle_t> handles;
    if (!g_cudnn_available) return nullptr;
    auto_set_device(ref);
    int dev = current_device();
    auto it = handles.find(dev);
    if (it != handles.end()) return it->second;
    lt_cudnnHandle_t handle = nullptr;
    g_cudnn.Create(&handle);
    g_cudnn.SetStream(handle, dev_stream(dev));
    handles[dev] = handle;
    return handle;
}
#else
rocblas_handle get_rocblas_handle(const void* ref) {
    thread_local std::unordered_map<int, rocblas_handle> handles;
    auto_set_device(ref);
    int dev = current_device();
    auto it = handles.find(dev);
    if (it != handles.end()) return it->second;
    rocblas_handle handle = nullptr;
    rocblas_create_handle(&handle);
    rocblas_set_stream(handle, dev_stream(dev));
    handles[dev] = handle;
    return handle;
}
#ifdef USE_MIOPEN
miopenHandle_t get_miopen_handle(const void* ref) {
    thread_local std::unordered_map<int, miopenHandle_t> handles;
    auto_set_device(ref);
    int dev = current_device();
    auto it = handles.find(dev);
    if (it != handles.end()) return it->second;
    miopenHandle_t handle = nullptr;
    miopenCreate(&handle);
    miopenSetStream(handle, dev_stream(dev));
    handles[dev] = handle;
    return handle;
}
#endif
extern "C" void gpu_set_tf32_enabled(bool) {}
extern "C" bool gpu_is_tf32_enabled() { return false; }
#endif



static void* g_fa3_handle = nullptr;
fa3_fwd_t g_fa3_fwd_fn = nullptr;
fa3_bwd_t g_fa3_bwd_fn = nullptr;

extern "C" bool gpu_init() {
    GPU_API(Error_t) err = GPU_API(SetDevice)(0);
    if (err != GPU_API(Success)) return false;
    g_compute_stream = dev_stream(0);
    g_comm_stream = dev_stream(0);
    if (!g_h2d_stream) g_h2d_stream = dev_stream(0);
    if (!g_d2h_stream) g_d2h_stream = dev_stream(0);
#ifndef __HIP_PLATFORM_AMD__
    if (cudnn_dyn_init())
        printf("[litetorch] cuDNN runtime: available v%zu (dynamic load)\n", g_cudnn.GetVersion());
#endif
    const char* fa_paths[] = { "libflash_attn.so", "/usr/local/cuda/lib64/libflash_attn.so", "./libflash_attn.so" };
    for (const char* p : fa_paths) {
        g_fa3_handle = dlopen(p, RTLD_NOW | RTLD_GLOBAL);
        if (g_fa3_handle) {
            g_fa3_fwd_fn = (fa3_fwd_t)dlsym(g_fa3_handle, "flash_attn_fwd");
            g_fa3_bwd_fn = (fa3_bwd_t)dlsym(g_fa3_handle, "flash_attn_bwd");
            break;
        }
    }
    return true;
}

extern "C" bool gpu_init_device(int device_id) {
    GPU_API(Error_t) err = GPU_API(SetDevice)(device_id);
    if (err != GPU_API(Success)) return false;
    g_compute_stream = dev_stream(device_id);
    g_comm_stream = dev_stream(device_id);
    if (!g_h2d_stream) g_h2d_stream = dev_stream(device_id);
    if (!g_d2h_stream) g_d2h_stream = dev_stream(device_id);
#ifndef __HIP_PLATFORM_AMD__
    cudnn_dyn_init();
#endif
    const char* fa_paths[] = { "libflash_attn.so", "/usr/local/cuda/lib64/libflash_attn.so", "./libflash_attn.so" };
    for (const char* p : fa_paths) {
        g_fa3_handle = dlopen(p, RTLD_NOW | RTLD_GLOBAL);
        if (g_fa3_handle) {
            g_fa3_fwd_fn = (fa3_fwd_t)dlsym(g_fa3_handle, "flash_attn_fwd");
            g_fa3_bwd_fn = (fa3_bwd_t)dlsym(g_fa3_handle, "flash_attn_bwd");
            break;
        }
    }
    return true;
}

extern "C" void* gpu_get_comm_stream() {
    return (void*)dev_stream(current_device());
}

extern "C" void* gpu_get_compute_stream() {
    return (void*)dev_stream(current_device());
}

extern "C" void gpu_sync_stream(void* stream) {
    if (stream) {
        GPU_API(StreamSynchronize)((GPU_API(Stream_t))stream);
    }
}

extern "C" void* gpu_create_event() {
    GPU_API(Event_t) event;
    GPU_API(EventCreateWithFlags)(&event, GPU_API(EventDisableTiming));
    return (void*)event;
}

extern "C" void gpu_record_event(void* event, void* stream) {
    if (event) {
        GPU_API(EventRecord)((GPU_API(Event_t))event, stream ? (GPU_API(Stream_t))stream : dev_stream(current_device()));
    }
}

extern "C" void gpu_stream_wait_event(void* stream, void* event) {
    if (event) {
        GPU_API(StreamWaitEvent)(stream ? (GPU_API(Stream_t))stream : g_comm_stream, (GPU_API(Event_t))event, 0);
    }
}

extern "C" void gpu_destroy_event(void* event) {
    if (event) {
        GPU_API(EventDestroy)((GPU_API(Event_t))event);
    }
}

extern "C" void gpu_set_device(int device_id) {
    GPU_API(SetDevice)(device_id);
}

extern "C" int gpu_get_device() {
    return current_device();
}

extern "C" void* gpu_start_recording() {
#ifndef __HIP_PLATFORM_AMD__
    cudaStreamBeginCapture(dev_stream(current_device()), cudaStreamCaptureModeGlobal);
#else
    hipStreamBeginCapture(dev_stream(current_device()), hipStreamCaptureModeGlobal);
#endif
    return nullptr;
}

extern "C" void* gpu_stop_recording(void* stream_capture) {
    void* exec_graph = nullptr;
#ifndef __HIP_PLATFORM_AMD__
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    if (cudaStreamEndCapture(dev_stream(current_device()), &graph) == cudaSuccess && graph) {
        if (cudaGraphInstantiate(&instance, graph, NULL, NULL, 0) == cudaSuccess) {
            exec_graph = (void*)instance;
        }
        cudaGraphDestroy(graph);
    } else {
        cudaGetLastError();
    }
#else
    hipGraph_t graph = nullptr;
    hipGraphExec_t instance = nullptr;
    if (hipStreamEndCapture(dev_stream(current_device()), &graph) == hipSuccess && graph) {
        if (hipGraphInstantiate(&instance, graph, NULL, NULL, 0) == hipSuccess) {
            exec_graph = (void*)instance;
        }
        hipGraphDestroy(graph);
    } else {
        hipGetLastError();
    }
#endif
    return exec_graph;
}

extern "C" void gpu_launch_graph(void* graph) {
    if (!graph) return;
#ifndef __HIP_PLATFORM_AMD__
    cudaGraphLaunch((cudaGraphExec_t)graph, dev_stream(current_device()));
#else
    hipGraphLaunch((hipGraphExec_t)graph, dev_stream(current_device()));
#endif
}

extern "C" void gpu_free_graph(void* graph) {
    if (!graph) return;
#ifndef __HIP_PLATFORM_AMD__
    cudaGraphExecDestroy((cudaGraphExec_t)graph);
#else
    hipGraphExecDestroy((hipGraphExec_t)graph);
#endif
}

struct GpuAllocInfo {
    size_t bucket;
    int device;
};
struct GpuMemPool {
    std::mutex mutex_;
    std::map<std::pair<size_t, int>, std::vector<void*>> free_;
    std::unordered_map<void*, GpuAllocInfo> live_;
    size_t cached_bytes_ = 0;
};

static void gpu_free_raw_on_device(void* ptr, int dev) {
    GPU_API(Stream_t) s = dev_stream(dev);
#ifndef __HIP_PLATFORM_AMD__
    cudaFreeAsync(ptr, s);
#else
    int cur = current_device();
    if (cur != dev) GPU_API(SetDevice)(dev);
    GPU_API(Free)(ptr);
    if (cur != dev) GPU_API(SetDevice)(cur);
#endif
}

static GpuMemPool& gpu_mem_pool() {
    static GpuMemPool pool;
    return pool;
}

static size_t gpu_cache_cap_bytes() {
    static const size_t cap = []() {
        const char* e = std::getenv("LITETORCH_GPU_CACHE_MB");
        long mb = e ? std::atol(e) : 1024;
        if (mb < 0) mb = 0;
        return (size_t)mb * 1024ULL * 1024ULL;
    }();
    return cap;
}

static inline size_t gpu_pool_bucket(size_t size) {
    size_t b = 512;
    while (b < size) {
        size_t nb = b + (b >> 1);
        b = nb > b ? nb : b << 1;
    }
    return b;
}

extern "C" void* gpu_allocate(size_t size) {
    if (size == 0) return nullptr;
    int dev = current_device();
    GPU_API(Stream_t) stream = dev_stream(dev);
    size_t bucket = gpu_pool_bucket(size);
    auto key = std::make_pair(bucket, dev);
    auto& pool = gpu_mem_pool();
    {
        std::lock_guard<std::mutex> lock(pool.mutex_);
        auto it = pool.free_.find(key);
        if (it != pool.free_.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            pool.live_[ptr] = {bucket, dev};
            pool.cached_bytes_ -= bucket;
            return ptr;
        }
    }
    void* ptr = nullptr;
#ifndef __HIP_PLATFORM_AMD__
    cudaMallocAsync(&ptr, bucket, stream);
#else
    GPU_API(Malloc)(&ptr, bucket);
#endif
    if (!ptr) {
        std::vector<std::pair<void*, int>> drain;
        {
            std::lock_guard<std::mutex> lock(pool.mutex_);
            for (auto& kv : pool.free_) {
                for (void* p : kv.second) drain.emplace_back(p, kv.first.second);
                kv.second.clear();
            }
            pool.cached_bytes_ = 0;
        }
        for (auto& dp : drain) gpu_free_raw_on_device(dp.first, dp.second);
        ptr = nullptr;
#ifndef __HIP_PLATFORM_AMD__
        cudaMallocAsync(&ptr, bucket, stream);
#else
        GPU_API(Malloc)(&ptr, bucket);
#endif
        if (!ptr) return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(pool.mutex_);
        pool.live_[ptr] = {bucket, dev};
    }
    return ptr;
}

extern "C" void gpu_empty_cache() {
    auto& pool = gpu_mem_pool();
    std::vector<std::pair<void*, int>> drain;
    {
        std::lock_guard<std::mutex> lock(pool.mutex_);
        for (auto& kv : pool.free_) {
            for (void* p : kv.second) drain.emplace_back(p, kv.first.second);
            kv.second.clear();
        }
        pool.cached_bytes_ = 0;
    }
    for (auto& dp : drain) gpu_free_raw_on_device(dp.first, dp.second);
}

extern "C" void gpu_free(void* ptr) {
    if (!ptr) return;
    auto& pool = gpu_mem_pool();
    size_t bucket = 0;
    int alloc_dev = -1;
    {
        std::lock_guard<std::mutex> lock(pool.mutex_);
        auto it = pool.live_.find(ptr);
        if (it == pool.live_.end()) {
            int dev = device_of_ptr_strict(ptr);
            gpu_free_raw_on_device(ptr, dev >= 0 ? dev : current_device());
            return;
        }
        bucket = it->second.bucket;
        alloc_dev = it->second.device;
        pool.live_.erase(it);
    }
    bool cache_it = false;
    {
        std::lock_guard<std::mutex> lock(pool.mutex_);
        if (pool.cached_bytes_ + bucket <= gpu_cache_cap_bytes()) {
            pool.free_[std::make_pair(bucket, alloc_dev)].push_back(ptr);
            pool.cached_bytes_ += bucket;
            cache_it = true;
        }
    }
    if (!cache_it) gpu_free_raw_on_device(ptr, alloc_dev);
}

static std::mutex g_p2p_mutex;
static std::map<std::pair<int, int>, char> g_p2p;

static bool ensure_p2p(int dev, int peer) {
    if (dev == peer) return true;
    if (dev < 0 || peer < 0) return false;
    auto key = std::make_pair(dev, peer);
    {
        std::lock_guard<std::mutex> lock(g_p2p_mutex);
        auto it = g_p2p.find(key);
        if (it != g_p2p.end()) return it->second != 0;
    }
    bool ok = false;
    int can = 0;
    if (GPU_API(DeviceCanAccessPeer)(&can, dev, peer) == GPU_API(Success) && can) {
        int cur = current_device();
        GPU_API(SetDevice)(dev);
        GPU_API(Error_t) e = GPU_API(DeviceEnablePeerAccess)(peer, 0);
        if (e == GPU_API(Success) || e == GPU_API(ErrorPeerAccessAlreadyEnabled)) ok = true;
        GPU_API(SetDevice)(cur);
    }
    {
        std::lock_guard<std::mutex> lock(g_p2p_mutex);
        g_p2p[key] = ok ? 1 : 0;
    }
    return ok;
}

extern "C" void gpu_read(void* ptr, size_t size, void* host_ptr, size_t offset) {
    int dev = ptr_device(ptr);
    GPU_API(SetDevice)(dev);
    GPU_API(Stream_t) s = dev_stream(dev);
    GPU_API(MemcpyAsync)(host_ptr, (char*)ptr + offset, size, GPU_API(MemcpyDeviceToHost), s);
    GPU_API(StreamSynchronize)(s);
}

extern "C" void gpu_write(void* ptr, size_t size, const void* host_ptr, size_t offset) {
    int dev = ptr_device(ptr);
    GPU_API(SetDevice)(dev);
    GPU_API(Stream_t) s = dev_stream(dev);
    GPU_API(MemcpyAsync)((char*)ptr + offset, host_ptr, size, GPU_API(MemcpyHostToDevice), s);
    GPU_API(StreamSynchronize)(s);
}

extern "C" void gpu_copy(void* src, void* dst, size_t size, size_t src_offset, size_t dst_offset) {
    if (!src || !dst || size == 0) return;
    int sdev = ptr_device(src);
    int ddev = ptr_device(dst);
    if (sdev == ddev) {
        GPU_API(SetDevice)(ddev);
        GPU_API(Stream_t) s = dev_stream(ddev);
        GPU_API(MemcpyAsync)((char*)dst + dst_offset, (char*)src + src_offset, size, GPU_API(MemcpyDeviceToDevice), s);
        GPU_API(StreamSynchronize)(s);
        return;
    }
    if (ensure_p2p(ddev, sdev)) {
        GPU_API(StreamSynchronize)(dev_stream(sdev));
        GPU_API(SetDevice)(ddev);
        GPU_API(Stream_t) ds = dev_stream(ddev);
        GPU_API(MemcpyAsync)((char*)dst + dst_offset, (char*)src + src_offset, size, GPU_API(MemcpyDeviceToDevice), ds);
        GPU_API(StreamSynchronize)(ds);
        return;
    }
    void* tmp = malloc(size);
    if (!tmp) return;
    GPU_API(SetDevice)(sdev);
    GPU_API(Stream_t) ss = dev_stream(sdev);
    GPU_API(MemcpyAsync)(tmp, (char*)src + src_offset, size, GPU_API(MemcpyDeviceToHost), ss);
    GPU_API(StreamSynchronize)(ss);
    GPU_API(SetDevice)(ddev);
    GPU_API(Stream_t) ds = dev_stream(ddev);
    GPU_API(MemcpyAsync)((char*)dst + dst_offset, tmp, size, GPU_API(MemcpyHostToDevice), ds);
    GPU_API(StreamSynchronize)(ds);
    free(tmp);
}

extern "C" void gpu_copy_peer(void* src, int sdev, void* dst, int ddev, size_t size) {
    if (!src || !dst || size == 0) return;
    if (sdev < 0) sdev = 0;
    if (ddev < 0) ddev = 0;
    if (sdev == ddev) {
        GPU_API(SetDevice)(ddev);
        GPU_API(Stream_t) s = dev_stream(ddev);
        GPU_API(MemcpyAsync)(dst, src, size, GPU_API(MemcpyDeviceToDevice), s);
        GPU_API(StreamSynchronize)(s);
        return;
    }
    void* tmp = malloc(size);
    if (!tmp) return;
    GPU_API(SetDevice)(sdev);
    GPU_API(Stream_t) ss = dev_stream(sdev);
    GPU_API(Error_t) e1 = GPU_API(MemcpyAsync)(tmp, src, size, GPU_API(MemcpyDeviceToHost), ss);
    GPU_API(StreamSynchronize)(ss);
    GPU_API(SetDevice)(ddev);
    GPU_API(Stream_t) ds = dev_stream(ddev);
    GPU_API(Error_t) e2 = GPU_API(MemcpyAsync)(dst, tmp, size, GPU_API(MemcpyHostToDevice), ds);
    GPU_API(StreamSynchronize)(ds);
    free(tmp);
    (void)e1; (void)e2;
}

extern "C" void gpu_read_async(void* ptr, size_t size, void* host_ptr, size_t offset) {
    int dev = ptr_device(ptr);
    GPU_API(SetDevice)(dev);
    GPU_API(MemcpyAsync)(host_ptr, (char*)ptr + offset, size, GPU_API(MemcpyDeviceToHost), dev_stream(dev));
}

extern "C" void gpu_write_async(void* ptr, size_t size, const void* host_ptr, size_t offset) {
    int dev = ptr_device(ptr);
    GPU_API(SetDevice)(dev);
    GPU_API(MemcpyAsync)((char*)ptr + offset, host_ptr, size, GPU_API(MemcpyHostToDevice), dev_stream(dev));
}

extern "C" void gpu_copy_async(void* src, void* dst, size_t size, size_t src_offset, size_t dst_offset) {
    if (!src || !dst || size == 0) return;
    int ddev = ptr_device(dst);
    GPU_API(SetDevice)(ddev);
    GPU_API(Stream_t) s = dev_stream(ddev);
    GPU_API(MemcpyAsync)((char*)dst + dst_offset, (char*)src + src_offset, size, GPU_API(MemcpyDeviceToDevice), s);
}

extern "C" void gpu_finish() {
    std::lock_guard<std::mutex> lock(g_dev_stream_mutex);
    for (auto s : g_dev_streams) {
        if (s) GPU_API(StreamSynchronize)(s);
    }
}

extern "C" void gpu_launch(void* kernel, int global_x, int global_y, int global_z, void** args, int arg_count) {
    int dev = launch_device(args, arg_count);
    GPU_API(SetDevice)(dev);
    GPU_API(Stream_t) stream = dev_stream(dev);
    dim3 block(256, 1, 1);
    if (global_y > 1) {
        block = dim3(16, 16, 1);
    }
    dim3 grid((global_x + block.x - 1) / block.x, (global_y + block.y - 1) / block.y, global_z);
#ifndef __HIP_PLATFORM_AMD__
    cudaLaunchKernel(kernel, grid, block, args, 0, stream);
#else
    hipLaunchKernel(kernel, grid, block, args, 0, stream);
#endif
}

struct KernelNameHash {
    size_t operator()(const char* s) const {
        size_t h = 1469598103934665603ull;
        for (const unsigned char* p = (const unsigned char*)s; *p; ++p) {
            h ^= *p;
            h *= 1099511628211ull;
        }
        return h;
    }
};

struct KernelNameEq {
    bool operator()(const char* a, const char* b) const {
        return std::strcmp(a, b) == 0;
    }
};

using KernelMap = std::unordered_map<const char*, void*, KernelNameHash, KernelNameEq>;

static const KernelMap& gpu_kernel_map() {
    static const KernelMap m = [] {
        KernelMap m;
        m.reserve(128);
        m["Add"] = (void*)&elementwise_add;
        m["elementwise_add"] = (void*)&elementwise_add;
        m["elementwise_add_inplace"] = (void*)&elementwise_add_inplace;
        m["elementwise_sub"] = (void*)&elementwise_sub;
        m["elementwise_mul"] = (void*)&elementwise_mul;
        m["elementwise_div"] = (void*)&elementwise_div;
        m["relu_forward"] = (void*)&relu_forward;
        m["relu_backward_kernel"] = (void*)&relu_backward_kernel;
        m["sigmoid_forward"] = (void*)&sigmoid_forward;
        m["sigmoid_backward_kernel"] = (void*)&sigmoid_backward_kernel;
        m["tanh_forward"] = (void*)&tanh_forward;
        m["tanh_backward_kernel"] = (void*)&tanh_backward_kernel;
        m["pow_forward"] = (void*)&pow_forward;
        m["sqrt_forward"] = (void*)&sqrt_forward;
        m["exp_forward"] = (void*)&exp_forward;
        m["log_forward"] = (void*)&log_forward;
        m["abs_forward"] = (void*)&abs_forward;
        m["neg_forward"] = (void*)&neg_forward;
        m["leaky_relu_forward"] = (void*)&leaky_relu_forward;
        m["leaky_relu_backward_kernel"] = (void*)&leaky_relu_backward_kernel;
        m["make_contiguous_kernel"] = (void*)&make_contiguous_kernel;
        m["copy_to_strided_kernel"] = (void*)&copy_to_strided_kernel;
        m["conv2d_kernel"] = (void*)&conv2d_kernel;
        m["conv2d_backward_gb"] = (void*)&conv2d_backward_gb;
        m["conv2d_backward_gw"] = (void*)&conv2d_backward_gw;
        m["conv2d_backward_gdx"] = (void*)&conv2d_backward_gdx;
        m["im2col_kernel"] = (void*)&im2col_kernel;
        m["im2col_batched_kernel"] = (void*)&im2col_batched_kernel;
        m["im2col_flat_kernel"] = (void*)&im2col_flat_kernel;
        m["transpose_conv_out_kernel"] = (void*)&transpose_conv_out_kernel;
        m["transpose_conv_out_inv_kernel"] = (void*)&transpose_conv_out_inv_kernel;
        m["transpose_conv_out_bias_relu_kernel"] = (void*)&transpose_conv_out_bias_relu_kernel;
        m["col2im_kernel"] = (void*)&col2im_kernel;
        m["broadcast_batch_kernel"] = (void*)&broadcast_batch_kernel;
        m["add_bias_2d"] = (void*)&add_bias_2d;
        m["conv3d_kernel"] = (void*)&conv3d_kernel;
        m["conv3d_backward_gb"] = (void*)&conv3d_backward_gb;
        m["conv3d_backward_gw"] = (void*)&conv3d_backward_gw;
        m["conv3d_backward_gdx"] = (void*)&conv3d_backward_gdx;
#ifndef __HIP_PLATFORM_AMD__
        m["matmul_ex_cublaslt"] = (void*)&gpu_matmul_ex_lt;
        m["matmul_ex_cublaslt_bias"] = (void*)&gpu_matmul_ex_lt_bias;
#endif
        m["maxpool2d_kernel"] = (void*)&maxpool2d_kernel;
        m["maxpool2d_backward_kernel"] = (void*)&maxpool2d_backward_kernel;
        m["adaptive_avg_pool2d_forward_kernel"] = (void*)&adaptive_avg_pool2d_forward_kernel;
        m["adaptive_avg_pool2d_backward_kernel"] = (void*)&adaptive_avg_pool2d_backward_kernel;
        m["maxpool3d_kernel"] = (void*)&maxpool3d_kernel;
        m["maxpool3d_backward_kernel"] = (void*)&maxpool3d_backward_kernel;
        m["softmax_forward_kernel"] = (void*)&softmax_forward_kernel;
        m["softmax_backward_kernel"] = (void*)&softmax_backward_kernel;
        m["softmax_fast_kernel"] = (void*)&softmax_fast_kernel;
        m["layer_norm_forward_kernel"] = (void*)&layer_norm_forward_kernel;
        m["layer_norm_fast_kernel"] = (void*)&layer_norm_fast_kernel;
        m["fused_add_layer_norm_forward_kernel"] = (void*)&fused_add_layer_norm_forward_kernel;
        m["layer_norm_backward_dx_kernel"] = (void*)&layer_norm_backward_dx_kernel;
        m["layer_norm_backward_dw_kernel"] = (void*)&layer_norm_backward_dw_kernel;
        m["layer_norm_backward_db_kernel"] = (void*)&layer_norm_backward_db_kernel;
        m["layer_norm_backward_fused_kernel"] = (void*)&layer_norm_backward_fused_kernel;
        m["batch_norm2d_forward_stats_kernel"] = (void*)&batch_norm2d_forward_stats_kernel;
        m["batch_norm2d_forward_norm_kernel"] = (void*)&batch_norm2d_forward_norm_kernel;
        m["batch_norm2d_backward_stats_kernel"] = (void*)&batch_norm2d_backward_stats_kernel;
        m["batch_norm2d_backward_dx_kernel"] = (void*)&batch_norm2d_backward_dx_kernel;
        m["mse_loss_forward"] = (void*)&mse_loss_forward;
        m["mse_loss_backward"] = (void*)&mse_loss_backward;
        m["l1_loss_forward"] = (void*)&l1_loss_forward;
        m["l1_loss_backward"] = (void*)&l1_loss_backward;
        m["bce_loss_forward"] = (void*)&bce_loss_forward;
        m["bce_loss_backward"] = (void*)&bce_loss_backward;
        m["cross_entropy_loss_forward"] = (void*)&cross_entropy_loss_forward;
        m["cross_entropy_loss_backward"] = (void*)&cross_entropy_loss_backward;
        m["fill_zero"] = (void*)&fill_zero;
        m["sum_backward"] = (void*)&sum_backward;
        m["fake_quantize_forward"] = (void*)&fake_quantize_forward;
        m["cast_fp32_to_fp16"] = (void*)&cast_fp32_to_fp16;
        m["cast_fp16_to_fp32"] = (void*)&cast_fp16_to_fp32;
        m["cast_fp32_to_bf16"] = (void*)&cast_fp32_to_bf16;
        m["cast_bf16_to_fp32"] = (void*)&cast_bf16_to_fp32;
        m["cast_fp32_to_nf4"] = (void*)&cast_fp32_to_nf4;
        m["cast_nf4_to_fp32"] = (void*)&cast_nf4_to_fp32;
        m["cast_fp32_to_int8"] = (void*)&cast_fp32_to_int8;
        m["cast_int8_to_fp32"] = (void*)&cast_int8_to_fp32;
        m["cast_fp32_to_int4"] = (void*)&cast_fp32_to_int4;
        m["cast_int4_to_fp32"] = (void*)&cast_int4_to_fp32;
        m["cast_fp32_to_fp8_e4m3"] = (void*)&cast_fp32_to_fp8_e4m3;
        m["cast_fp8_e4m3_to_fp32"] = (void*)&cast_fp8_e4m3_to_fp32;
        m["cast_fp32_to_fp8_e5m2"] = (void*)&cast_fp32_to_fp8_e5m2;
        m["cast_fp8_e5m2_to_fp32"] = (void*)&cast_fp8_e5m2_to_fp32;
        m["embedding_forward"] = (void*)&embedding_forward;
        m["embedding_backward"] = (void*)&embedding_backward;
        m["generate_dropout_mask"] = (void*)&generate_dropout_mask;
        m["sgd_step_kernel"] = (void*)&sgd_step_kernel;
        m["rmsprop_step_kernel"] = (void*)&rmsprop_step_kernel;
        m["adam_step_kernel"] = (void*)&adam_step_kernel;
        m["adam_foreach"] = (void*)&gpu_adam_foreach;
        m["adamw_step_kernel"] = (void*)&adamw_step_kernel;
        m["gelu_forward_kernel"] = (void*)&gelu_forward_kernel;
        m["gelu_backward_kernel"] = (void*)&gelu_backward_kernel;
        m["reduce_broadcast_prepended"] = (void*)&reduce_broadcast_prepended;
        m["reduce_broadcast_dim"] = (void*)&reduce_broadcast_dim;
        m["gpu_reduce_broadcast_prepended"] = (void*)&gpu_reduce_broadcast_prepended;
        m["gpu_reduce_broadcast_dim"] = (void*)&gpu_reduce_broadcast_dim;
        m["elementwise_broadcast_add"] = (void*)&elementwise_broadcast_add;
        m["elementwise_broadcast_sub"] = (void*)&elementwise_broadcast_sub;
        m["elementwise_broadcast_mul"] = (void*)&elementwise_broadcast_mul;
        m["elementwise_broadcast_div"] = (void*)&elementwise_broadcast_div;
        m["rope_forward"] = (void*)&rope_forward;
        m["rope_backward"] = (void*)&rope_backward;
        m["paged_attention_forward"] = (void*)&paged_attention_forward;
        m["qkv_split_transpose"] = (void*)&gpu_qkv_split_transpose;
        m["qkv_split_transpose_backward"] = (void*)&gpu_qkv_split_transpose_backward;
        m["qkv_extract"] = (void*)&gpu_qkv_extract;
        m["qkv_extract_backward"] = (void*)&gpu_qkv_extract_backward;
        m["w8a8_matmul_kernel"] = (void*)&w8a8_matmul_kernel;
        m["gpu_set_tf32_enabled"] = (void*)&gpu_set_tf32_enabled;
        m["gpu_is_tf32_enabled"] = (void*)&gpu_is_tf32_enabled;
        return m;
    }();
    return m;
}

extern "C" void* gpu_get_kernel(const char* name) {
    if (!name) return nullptr;
#ifndef __HIP_PLATFORM_AMD__
    if (std::strcmp(name, "conv2d_cudnn") == 0) return g_cudnn_available ? (void*)&gpu_conv2d_cudnn : nullptr;
    if (std::strcmp(name, "conv2d_backward_data_cudnn") == 0) return g_cudnn_available ? (void*)&gpu_conv2d_backward_data_cudnn : nullptr;
    if (std::strcmp(name, "conv2d_backward_filter_cudnn") == 0) return g_cudnn_available ? (void*)&gpu_conv2d_backward_filter_cudnn : nullptr;
    if (std::strcmp(name, "softmax_cudnn") == 0) return g_cudnn_available ? (void*)&gpu_softmax_cudnn : nullptr;
#endif
#ifdef USE_MIOPEN
    if (std::strcmp(name, "conv2d_miopen") == 0) return (void*)&gpu_conv2d_miopen;
    if (std::strcmp(name, "softmax_miopen") == 0) return (void*)&gpu_softmax_miopen;
#endif
    const KernelMap& m = gpu_kernel_map();
    auto it = m.find(name);
    return it != m.end() ? it->second : nullptr;
}

extern "C" void* gpu_compile_kernel(const char* source, const char* name) {
    if (!source || !name) return nullptr;
    const char* tmp_dir = "/tmp";
#ifdef _WIN32
    tmp_dir = getenv("TEMP");
    if (!tmp_dir) tmp_dir = getenv("TMP");
    if (!tmp_dir) tmp_dir = ".";
#endif
    char temp_src[512];
    char temp_so[512];
    snprintf(temp_src, sizeof(temp_src), "%s/litetorch_jit_%s_%d.cu", tmp_dir, name, (int)getpid());
    snprintf(temp_so, sizeof(temp_so), "%s/litetorch_jit_%s_%d.so", tmp_dir, name, (int)getpid());

    std::ofstream ofs(temp_src);
    if (!ofs.is_open()) return nullptr;
    ofs << source;
    ofs.close();

    char cmd[2048];
#ifndef __HIP_PLATFORM_AMD__
    snprintf(cmd, sizeof(cmd), "nvcc -O3 --shared -Xcompiler -fPIC %s -o %s > /dev/null 2>&1", temp_src, temp_so);
#else
    snprintf(cmd, sizeof(cmd), "hipcc -O3 -shared -fPIC -D__HIP_PLATFORM_AMD__ %s -o %s > /dev/null 2>&1", temp_src, temp_so);
#endif

    int ret = system(cmd);
#ifdef _WIN32
    _unlink(temp_src);
#else
    unlink(temp_src);
#endif
    if (ret != 0) return nullptr;

    void* handle = dlopen(temp_so, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) return nullptr;
    void* sym = dlsym(handle, name);
    return sym;
}

extern "C" void gpu_launch_dynamic(void* kernel, int gx, int gy, int gz, void** args, int arg_count) {
    gpu_launch(kernel, gx, gy, gz, args, arg_count);
}
