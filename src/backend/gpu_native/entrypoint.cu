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
#include "math/gemm.cu"
#include "math/reduction.cu"
#include "elementwise/elementwise_ops.cu"
#include "optim/optimizers.cu"
#include "nn/nn_kernels.cu"
#include "nn/flash_attention.cu"

GPU_API(Stream_t) g_compute_stream = nullptr;
GPU_API(Stream_t) g_comm_stream = nullptr;

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

cublasHandle_t get_cublas_handle() {
    thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        cublasSetStream(handle, g_compute_stream);
        cublasMath_t mode = g_tf32_enabled ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
        cublasSetMathMode(handle, mode);
    }
    return handle;
}
cublasLtHandle_t get_cublaslt_handle() {
    thread_local cublasLtHandle_t handle = nullptr;
    if (!handle) {
        cublasLtCreate(&handle);
    }
    return handle;
}
lt_cudnnHandle_t get_cudnn_handle() {
    thread_local lt_cudnnHandle_t handle = nullptr;
    if (!g_cudnn_available) return nullptr;
    if (!handle) {
        g_cudnn.Create(&handle);
        g_cudnn.SetStream(handle, g_compute_stream);
    }
    return handle;
}
#else
rocblas_handle get_rocblas_handle() {
    thread_local rocblas_handle handle = nullptr;
    if (!handle) {
        rocblas_create_handle(&handle);
        rocblas_set_stream(handle, g_compute_stream);
    }
    return handle;
}
#ifdef USE_MIOPEN
miopenHandle_t get_miopen_handle() {
    thread_local miopenHandle_t handle = nullptr;
    if (!handle) {
        miopenCreate(&handle);
        miopenSetStream(handle, g_compute_stream);
    }
    return handle;
}
#endif
extern "C" void gpu_set_tf32_enabled(bool) {}
extern "C" bool gpu_is_tf32_enabled() { return false; }
#endif

static inline void auto_set_device(const void* ptr) {
}

static void* g_fa3_handle = nullptr;
fa3_fwd_t g_fa3_fwd_fn = nullptr;
fa3_bwd_t g_fa3_bwd_fn = nullptr;

extern "C" bool gpu_init() {
    GPU_API(Error_t) err = GPU_API(SetDevice)(0);
    if (err != GPU_API(Success)) return false;
    GPU_API(StreamCreate)(&g_compute_stream);
    GPU_API(StreamCreate)(&g_comm_stream);
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
    GPU_API(StreamCreate)(&g_compute_stream);
    GPU_API(StreamCreate)(&g_comm_stream);
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
    return (void*)g_comm_stream;
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
        GPU_API(EventRecord)((GPU_API(Event_t))event, stream ? (GPU_API(Stream_t))stream : g_compute_stream);
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

extern "C" void* gpu_start_recording() {
#ifndef __HIP_PLATFORM_AMD__
    cudaStreamBeginCapture(g_compute_stream, cudaStreamCaptureModeGlobal);
#else
    hipStreamBeginCapture(g_compute_stream, hipStreamCaptureModeGlobal);
#endif
    return nullptr;
}

extern "C" void* gpu_stop_recording(void* stream_capture) {
    void* exec_graph = nullptr;
#ifndef __HIP_PLATFORM_AMD__
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    if (cudaStreamEndCapture(g_compute_stream, &graph) == cudaSuccess && graph) {
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
    if (hipStreamEndCapture(g_compute_stream, &graph) == hipSuccess && graph) {
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
    cudaGraphLaunch((cudaGraphExec_t)graph, g_compute_stream);
#else
    hipGraphLaunch((hipGraphExec_t)graph, g_compute_stream);
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

struct GpuMemPool {
    std::mutex mutex_;
    std::map<size_t, std::vector<void*>> free_;
    std::unordered_map<void*, size_t> live_;
    size_t cached_bytes_ = 0;
};

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
    size_t bucket = gpu_pool_bucket(size);
    auto& pool = gpu_mem_pool();
    {
        std::lock_guard<std::mutex> lock(pool.mutex_);
        auto it = pool.free_.find(bucket);
        if (it != pool.free_.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            pool.live_[ptr] = bucket;
            pool.cached_bytes_ -= bucket;
            return ptr;
        }
    }
    void* ptr = nullptr;
    GPU_API(Malloc)(&ptr, bucket);
    if (!ptr) {
        std::lock_guard<std::mutex> lock(pool.mutex_);
        for (auto& kv : pool.free_) {
            for (void* p : kv.second) GPU_API(Free)(p);
            kv.second.clear();
        }
        pool.cached_bytes_ = 0;
        ptr = nullptr;
        GPU_API(Malloc)(&ptr, bucket);
        if (!ptr) return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(pool.mutex_);
        pool.live_[ptr] = bucket;
    }
    return ptr;
}

extern "C" void gpu_empty_cache() {
    auto& pool = gpu_mem_pool();
    std::lock_guard<std::mutex> lock(pool.mutex_);
    for (auto& kv : pool.free_) {
        for (void* p : kv.second) GPU_API(Free)(p);
        kv.second.clear();
    }
    pool.cached_bytes_ = 0;
}

extern "C" void gpu_free(void* ptr) {
    if (!ptr) return;
    auto& pool = gpu_mem_pool();
    std::lock_guard<std::mutex> lock(pool.mutex_);
    auto it = pool.live_.find(ptr);
    if (it == pool.live_.end()) {
        GPU_API(Free)(ptr);
        return;
    }
    size_t bucket = it->second;
    pool.live_.erase(it);
    if (pool.cached_bytes_ + bucket > gpu_cache_cap_bytes()) {
        GPU_API(Free)(ptr);
        return;
    }
    pool.free_[bucket].push_back(ptr);
    pool.cached_bytes_ += bucket;
}

extern "C" void gpu_read(void* ptr, size_t size, void* host_ptr, size_t offset) {
    GPU_API(MemcpyAsync)(host_ptr, (char*)ptr + offset, size, GPU_API(MemcpyDeviceToHost), g_compute_stream);
    GPU_API(StreamSynchronize)(g_compute_stream);
}

extern "C" void gpu_write(void* ptr, size_t size, const void* host_ptr, size_t offset) {
    GPU_API(MemcpyAsync)((char*)ptr + offset, host_ptr, size, GPU_API(MemcpyHostToDevice), g_compute_stream);
    GPU_API(StreamSynchronize)(g_compute_stream);
}

extern "C" void gpu_copy(void* src, void* dst, size_t size, size_t src_offset, size_t dst_offset) {
    GPU_API(MemcpyAsync)((char*)dst + dst_offset, (char*)src + src_offset, size, GPU_API(MemcpyDeviceToDevice), g_compute_stream);
    GPU_API(StreamSynchronize)(g_compute_stream);
}

extern "C" void gpu_read_async(void* ptr, size_t size, void* host_ptr, size_t offset) {
    GPU_API(MemcpyAsync)(host_ptr, (char*)ptr + offset, size, GPU_API(MemcpyDeviceToHost), g_compute_stream);
}

extern "C" void gpu_write_async(void* ptr, size_t size, const void* host_ptr, size_t offset) {
    GPU_API(MemcpyAsync)((char*)ptr + offset, host_ptr, size, GPU_API(MemcpyHostToDevice), g_compute_stream);
}

extern "C" void gpu_copy_async(void* src, void* dst, size_t size, size_t src_offset, size_t dst_offset) {
    GPU_API(MemcpyAsync)((char*)dst + dst_offset, (char*)src + src_offset, size, GPU_API(MemcpyDeviceToDevice), g_compute_stream);
}

extern "C" void gpu_finish() {
    GPU_API(StreamSynchronize)(g_compute_stream);
}

extern "C" void gpu_launch(void* kernel, int global_x, int global_y, int global_z, void** args, int arg_count) {
    dim3 block(256, 1, 1);
    if (global_y > 1) {
        block = dim3(16, 16, 1);
    }
    dim3 grid((global_x + block.x - 1) / block.x, (global_y + block.y - 1) / block.y, global_z);
#ifndef __HIP_PLATFORM_AMD__
    cudaLaunchKernel(kernel, grid, block, args, 0, g_compute_stream);
#else
    hipLaunchKernel(kernel, grid, block, args, 0, g_compute_stream);
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
        m["adamw_step_kernel"] = (void*)&adamw_step_kernel;
        m["gelu_forward_kernel"] = (void*)&gelu_forward_kernel;
        m["gelu_backward_kernel"] = (void*)&gelu_backward_kernel;
        m["reduce_broadcast_prepended"] = (void*)&reduce_broadcast_prepended;
        m["reduce_broadcast_dim"] = (void*)&reduce_broadcast_dim;
        m["elementwise_broadcast_add"] = (void*)&elementwise_broadcast_add;
        m["elementwise_broadcast_sub"] = (void*)&elementwise_broadcast_sub;
        m["elementwise_broadcast_mul"] = (void*)&elementwise_broadcast_mul;
        m["elementwise_broadcast_div"] = (void*)&elementwise_broadcast_div;
        m["rope_forward"] = (void*)&rope_forward;
        m["rope_backward"] = (void*)&rope_backward;
        m["paged_attention_forward"] = (void*)&paged_attention_forward;
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
