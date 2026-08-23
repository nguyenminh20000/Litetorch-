#include "common/gpu_common.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unistd.h>
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
#ifdef USE_CUDNN
cudnnHandle_t get_cudnn_handle() {
    thread_local cudnnHandle_t handle = nullptr;
    if (!handle) {
        cudnnCreate(&handle);
        cudnnSetStream(handle, g_compute_stream);
    }
    return handle;
}
#endif
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

extern "C" void* gpu_allocate(size_t size) {
    void* ptr = nullptr;
    GPU_API(Malloc)(&ptr, size);
    return ptr;
}

extern "C" void gpu_free(void* ptr) {
    if (ptr) {
        GPU_API(Free)(ptr);
    }
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

extern "C" void* gpu_get_kernel(const char* name) {
    std::string sname(name);
    if (sname == "Add" || sname == "elementwise_add") return (void*)&elementwise_add;
    if (sname == "elementwise_add_inplace") return (void*)&elementwise_add_inplace;
    if (sname == "elementwise_sub") return (void*)&elementwise_sub;
    if (sname == "elementwise_mul") return (void*)&elementwise_mul;
    if (sname == "elementwise_div") return (void*)&elementwise_div;
    if (sname == "relu_forward") return (void*)&relu_forward;
    if (sname == "relu_backward_kernel") return (void*)&relu_backward_kernel;
    if (sname == "sigmoid_forward") return (void*)&sigmoid_forward;
    if (sname == "sigmoid_backward_kernel") return (void*)&sigmoid_backward_kernel;
    if (sname == "tanh_forward") return (void*)&tanh_forward;
    if (sname == "tanh_backward_kernel") return (void*)&tanh_backward_kernel;
    if (sname == "pow_forward") return (void*)&pow_forward;
    if (sname == "sqrt_forward") return (void*)&sqrt_forward;
    if (sname == "exp_forward") return (void*)&exp_forward;
    if (sname == "log_forward") return (void*)&log_forward;
    if (sname == "abs_forward") return (void*)&abs_forward;
    if (sname == "neg_forward") return (void*)&neg_forward;
    if (sname == "leaky_relu_forward") return (void*)&leaky_relu_forward;
    if (sname == "leaky_relu_backward_kernel") return (void*)&leaky_relu_backward_kernel;
    if (sname == "make_contiguous_kernel") return (void*)&make_contiguous_kernel;
    if (sname == "copy_to_strided_kernel") return (void*)&copy_to_strided_kernel;
    if (sname == "conv2d_kernel") return (void*)&conv2d_kernel;
    if (sname == "conv2d_backward_gb") return (void*)&conv2d_backward_gb;
    if (sname == "conv2d_backward_gw") return (void*)&conv2d_backward_gw;
    if (sname == "conv2d_backward_gdx") return (void*)&conv2d_backward_gdx;
    if (sname == "im2col_kernel") return (void*)&im2col_kernel;
    if (sname == "add_bias_2d") return (void*)&add_bias_2d;
    if (sname == "conv3d_kernel") return (void*)&conv3d_kernel;
    if (sname == "conv3d_backward_gb") return (void*)&conv3d_backward_gb;
    if (sname == "conv3d_backward_gw") return (void*)&conv3d_backward_gw;
    if (sname == "conv3d_backward_gdx") return (void*)&conv3d_backward_gdx;
#ifdef USE_CUDNN
    if (sname == "conv2d_cudnn") return (void*)&gpu_conv2d_cudnn;
    if (sname == "softmax_cudnn") return (void*)&gpu_softmax_cudnn;
#endif
#ifdef USE_MIOPEN
    if (sname == "conv2d_miopen") return (void*)&gpu_conv2d_miopen;
    if (sname == "softmax_miopen") return (void*)&gpu_softmax_miopen;
#endif
    if (sname == "maxpool2d_kernel") return (void*)&maxpool2d_kernel;
    if (sname == "maxpool2d_backward_kernel") return (void*)&maxpool2d_backward_kernel;
    if (sname == "adaptive_avg_pool2d_forward_kernel") return (void*)&adaptive_avg_pool2d_forward_kernel;
    if (sname == "adaptive_avg_pool2d_backward_kernel") return (void*)&adaptive_avg_pool2d_backward_kernel;
    if (sname == "maxpool3d_kernel") return (void*)&maxpool3d_kernel;
    if (sname == "maxpool3d_backward_kernel") return (void*)&maxpool3d_backward_kernel;
    if (sname == "softmax_forward_kernel") return (void*)&softmax_forward_kernel;
    if (sname == "softmax_backward_kernel") return (void*)&softmax_backward_kernel;
    if (sname == "layer_norm_forward_kernel") return (void*)&layer_norm_forward_kernel;
    if (sname == "fused_add_layer_norm_forward_kernel") return (void*)&fused_add_layer_norm_forward_kernel;
    if (sname == "layer_norm_backward_dx_kernel") return (void*)&layer_norm_backward_dx_kernel;
    if (sname == "layer_norm_backward_dw_kernel") return (void*)&layer_norm_backward_dw_kernel;
    if (sname == "layer_norm_backward_db_kernel") return (void*)&layer_norm_backward_db_kernel;
    if (sname == "batch_norm2d_forward_stats_kernel") return (void*)&batch_norm2d_forward_stats_kernel;
    if (sname == "batch_norm2d_forward_norm_kernel") return (void*)&batch_norm2d_forward_norm_kernel;
    if (sname == "batch_norm2d_backward_stats_kernel") return (void*)&batch_norm2d_backward_stats_kernel;
    if (sname == "batch_norm2d_backward_dx_kernel") return (void*)&batch_norm2d_backward_dx_kernel;
    if (sname == "mse_loss_forward") return (void*)&mse_loss_forward;
    if (sname == "mse_loss_backward") return (void*)&mse_loss_backward;
    if (sname == "l1_loss_forward") return (void*)&l1_loss_forward;
    if (sname == "l1_loss_backward") return (void*)&l1_loss_backward;
    if (sname == "bce_loss_forward") return (void*)&bce_loss_forward;
    if (sname == "bce_loss_backward") return (void*)&bce_loss_backward;
    if (sname == "cross_entropy_loss_forward") return (void*)&cross_entropy_loss_forward;
    if (sname == "cross_entropy_loss_backward") return (void*)&cross_entropy_loss_backward;
    if (sname == "fill_zero") return (void*)&fill_zero;
    if (sname == "sum_backward") return (void*)&sum_backward;
    if (sname == "fake_quantize_forward") return (void*)&fake_quantize_forward;
    if (sname == "cast_fp32_to_fp16") return (void*)&cast_fp32_to_fp16;
    if (sname == "cast_fp16_to_fp32") return (void*)&cast_fp16_to_fp32;
    if (sname == "cast_fp32_to_bf16") return (void*)&cast_fp32_to_bf16;
    if (sname == "cast_bf16_to_fp32") return (void*)&cast_bf16_to_fp32;
    if (sname == "cast_fp32_to_nf4") return (void*)&cast_fp32_to_nf4;
    if (sname == "cast_nf4_to_fp32") return (void*)&cast_nf4_to_fp32;
    if (sname == "cast_fp32_to_int8") return (void*)&cast_fp32_to_int8;
    if (sname == "cast_int8_to_fp32") return (void*)&cast_int8_to_fp32;
    if (sname == "cast_fp32_to_int4") return (void*)&cast_fp32_to_int4;
    if (sname == "cast_int4_to_fp32") return (void*)&cast_int4_to_fp32;
    if (sname == "cast_fp32_to_fp8_e4m3") return (void*)&cast_fp32_to_fp8_e4m3;
    if (sname == "cast_fp8_e4m3_to_fp32") return (void*)&cast_fp8_e4m3_to_fp32;
    if (sname == "cast_fp32_to_fp8_e5m2") return (void*)&cast_fp32_to_fp8_e5m2;
    if (sname == "cast_fp8_e5m2_to_fp32") return (void*)&cast_fp8_e5m2_to_fp32;
    if (sname == "embedding_forward") return (void*)&embedding_forward;
    if (sname == "embedding_backward") return (void*)&embedding_backward;
    if (sname == "generate_dropout_mask") return (void*)&generate_dropout_mask;
    if (sname == "sgd_step_kernel") return (void*)&sgd_step_kernel;
    if (sname == "rmsprop_step_kernel") return (void*)&rmsprop_step_kernel;
    if (sname == "adam_step_kernel") return (void*)&adam_step_kernel;
    if (sname == "adamw_step_kernel") return (void*)&adamw_step_kernel;
    if (sname == "gelu_forward_kernel") return (void*)&gelu_forward_kernel;
    if (sname == "gelu_backward_kernel") return (void*)&gelu_backward_kernel;
    if (sname == "reduce_broadcast_prepended") return (void*)&reduce_broadcast_prepended;
    if (sname == "reduce_broadcast_dim") return (void*)&reduce_broadcast_dim;
    if (sname == "elementwise_broadcast_add") return (void*)&elementwise_broadcast_add;
    if (sname == "elementwise_broadcast_sub") return (void*)&elementwise_broadcast_sub;
    if (sname == "elementwise_broadcast_mul") return (void*)&elementwise_broadcast_mul;
    if (sname == "elementwise_broadcast_div") return (void*)&elementwise_broadcast_div;
    if (sname == "rope_forward") return (void*)&rope_forward;
    if (sname == "rope_backward") return (void*)&rope_backward;
    if (sname == "paged_attention_forward") return (void*)&paged_attention_forward;
    if (sname == "w8a8_matmul_kernel") return (void*)&w8a8_matmul_kernel;
    if (sname == "gpu_set_tf32_enabled") return (void*)&gpu_set_tf32_enabled;
    if (sname == "gpu_is_tf32_enabled") return (void*)&gpu_is_tf32_enabled;
    return nullptr;
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

    char cmd[1024];
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
