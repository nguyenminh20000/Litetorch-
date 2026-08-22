#include "litetorch/backend.h"
#include "litetorch/platform.h"
#include <iostream>
#include <unordered_set>

namespace litetorch {

typedef bool (*gpu_init_t)();
typedef void* (*gpu_allocate_t)(size_t);
typedef void (*gpu_free_t)(void*);
typedef void (*gpu_read_t)(void*, size_t, void*, size_t);
typedef void (*gpu_write_t)(void*, size_t, const void*, size_t);
typedef void (*gpu_copy_t)(void*, void*, size_t, size_t, size_t);
typedef void (*gpu_read_async_t)(void*, size_t, void*, size_t);
typedef void (*gpu_write_async_t)(void*, size_t, const void*, size_t);
typedef void (*gpu_copy_async_t)(void*, void*, size_t, size_t, size_t);
typedef void (*gpu_finish_t)();
typedef void* (*gpu_get_kernel_t)(const char*);
typedef void (*gpu_launch_t)(void*, int, int, int, void**, int);
typedef void* (*gpu_compile_kernel_t)(const char*, const char*);
typedef void (*gpu_launch_dynamic_t)(void*, int, int, int, void**, int);
typedef void (*gpu_matmul_t)(void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t);
typedef void (*gpu_matmul_ex_t)(void*, int64_t, bool, int64_t, void*, int64_t, bool, int64_t, void*, int64_t, int64_t, int64_t, int64_t);
typedef void (*gpu_bmm_t)(void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t);
typedef void (*gpu_matmul_half_t)(void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t);
typedef void (*gpu_bmm_half_t)(void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t);
typedef void (*gpu_matmul_fp8_t)(void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, float, float, float);
typedef void (*gpu_matmul_bf16_t)(void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t);
typedef void (*gpu_sum_t)(void*, int64_t, void*, int64_t, int64_t);
typedef void (*gpu_max_t)(void*, int64_t, void*, int64_t, int64_t);
typedef void (*gpu_adamw_step_t)(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, float, float, float, float, float);
typedef void (*gpu_flash_attention_t)(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float);
typedef void (*gpu_flash_attention_half_t)(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float);
typedef void (*gpu_flash_attention_backward_t)(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float);
typedef void (*gpu_flash_attention_backward_half_t)(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float);

typedef void (*gpu_cat_forward_t)(void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
typedef void (*gpu_cat_backward_t)(void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
typedef void (*gpu_moe_gate_t)(void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t);
typedef void (*gpu_moe_gate_backward_t)(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t);
typedef void (*gpu_moe_expert_forward_t)(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
typedef void (*gpu_moe_expert_backward_t)(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

typedef void* (*gpu_start_recording_t)();
typedef void* (*gpu_stop_recording_t)(void*);
typedef void (*gpu_launch_graph_t)(void*);
typedef void (*gpu_free_graph_t)(void*);

typedef void* (*gpu_create_event_t)();
typedef void (*gpu_record_event_t)(void*, void*);
typedef void (*gpu_stream_wait_event_t)(void*, void*);
typedef void (*gpu_destroy_event_t)(void*);

typedef void* (*gpu_get_comm_stream_t)();
typedef void (*gpu_sync_stream_t)(void*);
typedef void (*gpu_set_device_t)(int);
typedef void (*gpu_set_tf32_enabled_t)(bool);
typedef bool (*gpu_is_tf32_enabled_t)();

class NativeGPUBackend : public DeviceBackend {
public:
    void* handle = nullptr;
    gpu_init_t gpu_init_fn = nullptr;
    gpu_set_device_t gpu_set_device_fn = nullptr;
    gpu_allocate_t gpu_allocate_fn = nullptr;
    gpu_free_t gpu_free_fn = nullptr;
    gpu_read_t gpu_read_fn = nullptr;
    gpu_write_t gpu_write_fn = nullptr;
    gpu_copy_t gpu_copy_fn = nullptr;
    gpu_read_async_t gpu_read_async_fn = nullptr;
    gpu_write_async_t gpu_write_async_fn = nullptr;
    gpu_copy_async_t gpu_copy_async_fn = nullptr;
    gpu_finish_t gpu_finish_fn = nullptr;
    gpu_get_kernel_t gpu_get_kernel_fn = nullptr;
    gpu_launch_t gpu_launch_fn = nullptr;
    gpu_compile_kernel_t gpu_compile_kernel_fn = nullptr;
    gpu_launch_dynamic_t gpu_launch_dynamic_fn = nullptr;
    gpu_matmul_t gpu_matmul_fn = nullptr;
    gpu_matmul_ex_t gpu_matmul_ex_fn = nullptr;
    gpu_bmm_t gpu_bmm_fn = nullptr;
    gpu_matmul_half_t gpu_matmul_half_fn = nullptr;
    gpu_bmm_half_t gpu_bmm_half_fn = nullptr;
    gpu_matmul_fp8_t gpu_matmul_fp8_fn = nullptr;
    gpu_matmul_bf16_t gpu_matmul_bf16_fn = nullptr;
    gpu_sum_t gpu_sum_fn = nullptr;
    gpu_max_t gpu_max_fn = nullptr;
    gpu_adamw_step_t gpu_adamw_step_fn = nullptr;
    gpu_flash_attention_t gpu_flash_attention_fn = nullptr;
    gpu_flash_attention_half_t gpu_flash_attention_half_fn = nullptr;
    gpu_flash_attention_backward_t gpu_flash_attention_backward_fn = nullptr;
    gpu_flash_attention_backward_half_t gpu_flash_attention_backward_half_fn = nullptr;
    gpu_cat_forward_t gpu_cat_forward_fn = nullptr;
    gpu_cat_backward_t gpu_cat_backward_fn = nullptr;
    gpu_moe_gate_t gpu_moe_gate_fn = nullptr;
    gpu_moe_gate_backward_t gpu_moe_gate_backward_fn = nullptr;
    gpu_moe_expert_forward_t gpu_moe_expert_forward_fn = nullptr;
    gpu_moe_expert_backward_t gpu_moe_expert_backward_fn = nullptr;
    gpu_start_recording_t gpu_start_recording_fn = nullptr;
    gpu_stop_recording_t gpu_stop_recording_fn = nullptr;
    gpu_launch_graph_t gpu_launch_graph_fn = nullptr;
    gpu_free_graph_t gpu_free_graph_fn = nullptr;
    gpu_create_event_t gpu_create_event_fn = nullptr;
    gpu_record_event_t gpu_record_event_fn = nullptr;
    gpu_stream_wait_event_t gpu_stream_wait_event_fn = nullptr;
    gpu_destroy_event_t gpu_destroy_event_fn = nullptr;
    gpu_get_comm_stream_t gpu_get_comm_stream_fn = nullptr;
    gpu_sync_stream_t gpu_sync_stream_fn = nullptr;
    gpu_set_tf32_enabled_t gpu_set_tf32_enabled_fn = nullptr;
    gpu_is_tf32_enabled_t gpu_is_tf32_enabled_fn = nullptr;
    std::unordered_set<void*> dynamic_kernels;
    bool is_ok = false;

    NativeGPUBackend() {
        if (getenv("LITETORCH_NO_GPU") || getenv("LITETORCH_NO_NATIVE_GPU")) {
            return;
        }
        std::vector<std::string> paths = {
            "./build/liblitetorch_gpu.dll",
            "build/liblitetorch_gpu.dll",
            "liblitetorch_gpu.dll",
            "./build/liblitetorch_gpu.so",
            "build/liblitetorch_gpu.so",
            "liblitetorch_gpu.so",
            "/tmp/liblitetorch_gpu.so",
            "/usr/local/lib/liblitetorch_gpu.so",
            "/usr/lib/liblitetorch_gpu.so"
        };
        const char* temp_env = getenv("TEMP");
        if (!temp_env) temp_env = getenv("TMP");
        if (temp_env) {
            paths.push_back(std::string(temp_env) + "/liblitetorch_gpu.dll");
            paths.push_back(std::string(temp_env) + "\\liblitetorch_gpu.dll");
            paths.push_back(std::string(temp_env) + "/liblitetorch_gpu.so");
        }
#ifndef _WIN32
        Dl_info info;
        if (dladdr((void*)&BackendDispatcher::get, &info) && info.dli_fname) {
            std::string dir = info.dli_fname;
            size_t slash = dir.find_last_of("/\\");
            if (slash != std::string::npos) {
                paths.insert(paths.begin(), dir.substr(0, slash) + "/liblitetorch_gpu.so");
            }
        }
#else
        char module_path[MAX_PATH];
        HMODULE hModule = NULL;
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)&BackendDispatcher::get, &hModule);
        if (hModule && GetModuleFileNameA(hModule, module_path, sizeof(module_path))) {
            std::string dir = module_path;
            size_t slash = dir.find_last_of("/\\");
            if (slash != std::string::npos) {
                paths.insert(paths.begin(), dir.substr(0, slash) + "\\liblitetorch_gpu.dll");
                paths.insert(paths.begin(), dir.substr(0, slash) + "/liblitetorch_gpu.dll");
            }
        }
#endif
        for (const auto& path : paths) {
            handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (handle) break;
        }
        if (!handle) return;

        gpu_init_fn = (gpu_init_t)dlsym(handle, "gpu_init");
        gpu_set_device_fn = (gpu_set_device_t)dlsym(handle, "gpu_set_device");
        gpu_allocate_fn = (gpu_allocate_t)dlsym(handle, "gpu_allocate");
        gpu_free_fn = (gpu_free_t)dlsym(handle, "gpu_free");
        gpu_read_fn = (gpu_read_t)dlsym(handle, "gpu_read");
        gpu_write_fn = (gpu_write_t)dlsym(handle, "gpu_write");
        gpu_copy_fn = (gpu_copy_t)dlsym(handle, "gpu_copy");
        gpu_read_async_fn = (gpu_read_async_t)dlsym(handle, "gpu_read_async");
        gpu_write_async_fn = (gpu_write_async_t)dlsym(handle, "gpu_write_async");
        gpu_copy_async_fn = (gpu_copy_async_t)dlsym(handle, "gpu_copy_async");
        gpu_finish_fn = (gpu_finish_t)dlsym(handle, "gpu_finish");
        gpu_get_kernel_fn = (gpu_get_kernel_t)dlsym(handle, "gpu_get_kernel");
        gpu_launch_fn = (gpu_launch_t)dlsym(handle, "gpu_launch");
        gpu_compile_kernel_fn = (gpu_compile_kernel_t)dlsym(handle, "gpu_compile_kernel");
        gpu_launch_dynamic_fn = (gpu_launch_dynamic_t)dlsym(handle, "gpu_launch_dynamic");
        gpu_matmul_fn = (gpu_matmul_t)dlsym(handle, "gpu_matmul");
        gpu_matmul_ex_fn = (gpu_matmul_ex_t)dlsym(handle, "gpu_matmul_ex");
        gpu_bmm_fn = (gpu_bmm_t)dlsym(handle, "gpu_bmm");
        gpu_matmul_half_fn = (gpu_matmul_half_t)dlsym(handle, "gpu_matmul_half");
        gpu_bmm_half_fn = (gpu_bmm_half_t)dlsym(handle, "gpu_bmm_half");
        gpu_matmul_fp8_fn = (gpu_matmul_fp8_t)dlsym(handle, "gpu_matmul_fp8");
        gpu_matmul_bf16_fn = (gpu_matmul_bf16_t)dlsym(handle, "gpu_matmul_bf16");
        gpu_sum_fn = (gpu_sum_t)dlsym(handle, "gpu_sum");
        gpu_max_fn = (gpu_max_t)dlsym(handle, "gpu_max");
        gpu_adamw_step_fn = (gpu_adamw_step_t)dlsym(handle, "gpu_adamw_step");
        gpu_flash_attention_fn = (gpu_flash_attention_t)dlsym(handle, "gpu_flash_attention");
        gpu_flash_attention_half_fn = (gpu_flash_attention_half_t)dlsym(handle, "gpu_flash_attention_half");
        gpu_flash_attention_backward_fn = (gpu_flash_attention_backward_t)dlsym(handle, "gpu_flash_attention_backward");
        gpu_flash_attention_backward_half_fn = (gpu_flash_attention_backward_half_t)dlsym(handle, "gpu_flash_attention_backward_half");
        gpu_cat_forward_fn = (gpu_cat_forward_t)dlsym(handle, "gpu_cat_forward");
        gpu_cat_backward_fn = (gpu_cat_backward_t)dlsym(handle, "gpu_cat_backward");
        gpu_moe_gate_fn = (gpu_moe_gate_t)dlsym(handle, "gpu_moe_gate");
        gpu_moe_gate_backward_fn = (gpu_moe_gate_backward_t)dlsym(handle, "gpu_moe_gate_backward");
        gpu_moe_expert_forward_fn = (gpu_moe_expert_forward_t)dlsym(handle, "gpu_moe_expert_forward");
        gpu_moe_expert_backward_fn = (gpu_moe_expert_backward_t)dlsym(handle, "gpu_moe_expert_backward");
        gpu_start_recording_fn = (gpu_start_recording_t)dlsym(handle, "gpu_start_recording");
        gpu_stop_recording_fn = (gpu_stop_recording_t)dlsym(handle, "gpu_stop_recording");
        gpu_launch_graph_fn = (gpu_launch_graph_t)dlsym(handle, "gpu_launch_graph");
        gpu_free_graph_fn = (gpu_free_graph_t)dlsym(handle, "gpu_free_graph");
        gpu_create_event_fn = (gpu_create_event_t)dlsym(handle, "gpu_create_event");
        gpu_record_event_fn = (gpu_record_event_t)dlsym(handle, "gpu_record_event");
        gpu_stream_wait_event_fn = (gpu_stream_wait_event_t)dlsym(handle, "gpu_stream_wait_event");
        gpu_destroy_event_fn = (gpu_destroy_event_t)dlsym(handle, "gpu_destroy_event");
        gpu_get_comm_stream_fn = (gpu_get_comm_stream_t)dlsym(handle, "gpu_get_comm_stream");
        gpu_sync_stream_fn = (gpu_sync_stream_t)dlsym(handle, "gpu_sync_stream");
        gpu_set_device_fn = (gpu_set_device_t)dlsym(handle, "gpu_set_device");
        gpu_set_tf32_enabled_fn = (gpu_set_tf32_enabled_t)dlsym(handle, "gpu_set_tf32_enabled");
        gpu_is_tf32_enabled_fn = (gpu_is_tf32_enabled_t)dlsym(handle, "gpu_is_tf32_enabled");

        if (gpu_init_fn && gpu_allocate_fn && gpu_free_fn && gpu_read_fn && 
            gpu_write_fn && gpu_copy_fn && gpu_finish_fn && gpu_get_kernel_fn && gpu_launch_fn) {
            is_ok = gpu_init_fn();
        }
    }

    ~NativeGPUBackend() {
        if (handle) {
            dlclose(handle);
        }
    }

    bool is_available() const override { return is_ok; }
    void* allocate(size_t size) override { return gpu_allocate_fn(size); }
    void free(void* ptr) override { gpu_free_fn(ptr); }
    void read(void* ptr, size_t size, void* host_ptr, size_t offset = 0) override { gpu_read_fn(ptr, size, host_ptr, offset); }
    void write(void* ptr, size_t size, const void* host_ptr, size_t offset = 0) override { gpu_write_fn(ptr, size, host_ptr, offset); }
    void read_async(void* ptr, size_t size, void* host_ptr, size_t offset = 0) override { if (gpu_read_async_fn) gpu_read_async_fn(ptr, size, host_ptr, offset); else gpu_read_fn(ptr, size, host_ptr, offset); }
    void write_async(void* ptr, size_t size, const void* host_ptr, size_t offset = 0) override { if (gpu_write_async_fn) gpu_write_async_fn(ptr, size, host_ptr, offset); else gpu_write_fn(ptr, size, host_ptr, offset); }
    void copy(void* src, void* dst, size_t size, size_t src_offset = 0, size_t dst_offset = 0) override { gpu_copy_fn(src, dst, size, src_offset, dst_offset); }
    void finish() override { gpu_finish_fn(); }
    void sum(void* A, int64_t a_off, void* B, int64_t b_off, int64_t size) override { if (gpu_sum_fn) gpu_sum_fn(A, a_off, B, b_off, size); }
    void max(void* A, int64_t a_off, void* B, int64_t b_off, int64_t size) override { if (gpu_max_fn) gpu_max_fn(A, a_off, B, b_off, size); }
    void matmul(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) override { if (gpu_matmul_fn) gpu_matmul_fn(A, a_off, B, b_off, C, c_off, M, N, K); }
    void matmul_ex(void* A, int64_t a_off, bool trans_a, int64_t lda, void* B, int64_t b_off, bool trans_b, int64_t ldb, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) override {
        if (gpu_matmul_ex_fn) gpu_matmul_ex_fn(A, a_off, trans_a, lda, B, b_off, trans_b, ldb, C, c_off, M, N, K);
        else if (gpu_matmul_fn) gpu_matmul_fn(A, a_off, B, b_off, C, c_off, M, N, K);
    }
    void bmm(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t B_batch, int64_t M, int64_t N, int64_t K) override { if (gpu_bmm_fn) gpu_bmm_fn(A, a_off, B, b_off, C, c_off, B_batch, M, N, K); }
    void matmul_half(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) override { if (gpu_matmul_half_fn) gpu_matmul_half_fn(A, a_off, B, b_off, C, c_off, M, N, K); }
    void bmm_half(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t B_batch, int64_t M, int64_t N, int64_t K) override { if (gpu_bmm_half_fn) gpu_bmm_half_fn(A, a_off, B, b_off, C, c_off, B_batch, M, N, K); }
    void matmul_fp8(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K, float a_scale = 1.0f, float b_scale = 1.0f, float d_scale = 1.0f) override {
        if (gpu_matmul_fp8_fn) {
            gpu_matmul_fp8_fn(A, a_off, B, b_off, C, c_off, M, N, K, a_scale, b_scale, d_scale);
        } else if (gpu_matmul_half_fn) {
            gpu_matmul_half_fn(A, a_off, B, b_off, C, c_off, M, N, K);
        }
    }
    void matmul_bf16(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) override {
        if (gpu_matmul_bf16_fn) {
            gpu_matmul_bf16_fn(A, a_off, B, b_off, C, c_off, M, N, K);
        } else if (gpu_matmul_half_fn) {
            gpu_matmul_half_fn(A, a_off, B, b_off, C, c_off, M, N, K);
        }
    }
    void adamw_step(void* P, int64_t p_off, void* G, int64_t g_off, void* M_state, int64_t m_off, void* V, int64_t v_off, int64_t size, float lr_t, float beta1, float beta2, float eps, float weight_decay) override {
        if (gpu_adamw_step_fn) gpu_adamw_step_fn(P, p_off, G, g_off, M_state, m_off, V, v_off, size, lr_t, beta1, beta2, eps, weight_decay);
    }
    void flash_attention(void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, void* O, int64_t o_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) override {
        if (gpu_flash_attention_fn) gpu_flash_attention_fn(Q, q_off, K, k_off, V, v_off, O, o_off, B, H, H_kv, Tq, Tk, D, scale);
    }
    void flash_attention_half(void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, void* O, int64_t o_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) override {
        if (gpu_flash_attention_half_fn) gpu_flash_attention_half_fn(Q, q_off, K, k_off, V, v_off, O, o_off, B, H, H_kv, Tq, Tk, D, scale);
    }
    void flash_attention_backward(void* dQ, int64_t dq_off, void* dK, int64_t dk_off, void* dV, int64_t dv_off, void* O, int64_t o_off, void* dO, int64_t do_off, void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) override {
        if (gpu_flash_attention_backward_fn) gpu_flash_attention_backward_fn(dQ, dq_off, dK, dk_off, dV, dv_off, O, o_off, dO, do_off, Q, q_off, K, k_off, V, v_off, B, H, H_kv, Tq, Tk, D, scale);
    }
    void flash_attention_backward_half(void* dQ, int64_t dq_off, void* dK, int64_t dk_off, void* dV, int64_t dv_off, void* O, int64_t o_off, void* dO, int64_t do_off, void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) override {
        if (gpu_flash_attention_backward_half_fn) gpu_flash_attention_backward_half_fn(dQ, dq_off, dK, dk_off, dV, dv_off, O, o_off, dO, do_off, Q, q_off, K, k_off, V, v_off, B, H, H_kv, Tq, Tk, D, scale);
    }
    
    void cat_forward(void* input, int64_t in_off, void* output, int64_t out_off, int64_t outer_size, int64_t inner_size, int64_t dim_size, int64_t concat_dim_size, int64_t offset) override {
        if (gpu_cat_forward_fn) gpu_cat_forward_fn(input, in_off, output, out_off, outer_size, inner_size, dim_size, concat_dim_size, offset);
    }
    void cat_backward(void* grad_output, int64_t gout_off, void* grad_input, int64_t gin_off, int64_t outer_size, int64_t inner_size, int64_t dim_size, int64_t concat_dim_size, int64_t offset) override {
        if (gpu_cat_backward_fn) gpu_cat_backward_fn(grad_output, gout_off, grad_input, gin_off, outer_size, inner_size, dim_size, concat_dim_size, offset);
    }
    void moe_gate(void* logits, int64_t l_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, int64_t N, int64_t E, int64_t top_k) override {
        if (gpu_moe_gate_fn) gpu_moe_gate_fn(logits, l_off, probs, p_off, indices, idx_off, N, E, top_k);
    }
    void moe_gate_backward(void* grad_output, int64_t gout_off, void* input, int64_t in_off, void* gate_weight, int64_t gw_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, void* grad_input, int64_t gin_off, void* grad_gate_weight, int64_t ggw_off, int64_t N, int64_t D, int64_t E, int64_t top_k) override {
        if (gpu_moe_gate_backward_fn) gpu_moe_gate_backward_fn(grad_output, gout_off, input, in_off, gate_weight, gw_off, probs, p_off, indices, idx_off, grad_input, gin_off, grad_gate_weight, ggw_off, N, D, E, top_k);
    }
    void moe_expert_forward(void* input, int64_t in_off, void* expert_weight, int64_t ew_off, void* expert_bias, int64_t eb_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, void* output, int64_t out_off, int64_t N, int64_t D, int64_t out_features, int64_t expert_idx, int64_t top_k) override {
        if (gpu_moe_expert_forward_fn) gpu_moe_expert_forward_fn(input, in_off, expert_weight, ew_off, expert_bias, eb_off, probs, p_off, indices, idx_off, output, out_off, N, D, out_features, expert_idx, top_k);
    }
    void moe_expert_backward(void* grad_output, int64_t gout_off, void* input, int64_t in_off, void* expert_weight, int64_t ew_off, void* expert_bias, int64_t eb_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, void* grad_input, int64_t gin_off, void* grad_expert, int64_t ge_off, void* grad_bias, int64_t gb_off, void* grad_probs, int64_t gp_off, int64_t N, int64_t D, int64_t out_features, int64_t expert_idx, int64_t top_k) override {
        if (gpu_moe_expert_backward_fn) gpu_moe_expert_backward_fn(grad_output, gout_off, input, in_off, expert_weight, ew_off, expert_bias, eb_off, probs, p_off, indices, idx_off, grad_input, gin_off, grad_expert, ge_off, grad_bias, gb_off, grad_probs, gp_off, N, D, out_features, expert_idx, top_k);
    }
    
    void* start_recording() override { return gpu_start_recording_fn ? gpu_start_recording_fn() : nullptr; }
    void* stop_recording(void* stream_capture) override { return gpu_stop_recording_fn ? gpu_stop_recording_fn(stream_capture) : nullptr; }
    void launch_graph(void* graph) override { if (gpu_launch_graph_fn) gpu_launch_graph_fn(graph); }
    void free_graph(void* graph) override { if (gpu_free_graph_fn) gpu_free_graph_fn(graph); }
    
    void* create_event() override { return gpu_create_event_fn ? gpu_create_event_fn() : nullptr; }
    void record_event(void* event, void* stream) override { if (gpu_record_event_fn && event) gpu_record_event_fn(event, stream); }
    void stream_wait_event(void* stream, void* event) override { if (gpu_stream_wait_event_fn && event) gpu_stream_wait_event_fn(stream, event); }
    void destroy_event(void* event) override { if (gpu_destroy_event_fn && event) gpu_destroy_event_fn(event); }

    void* get_comm_stream() override { return gpu_get_comm_stream_fn ? gpu_get_comm_stream_fn() : nullptr; }
    void sync_stream(void* stream) override { if (gpu_sync_stream_fn && stream) gpu_sync_stream_fn(stream); }
    void set_device(int device_id) override { if (gpu_set_device_fn) gpu_set_device_fn(device_id); }
    void set_tf32_enabled(bool enabled) override { if (gpu_set_tf32_enabled_fn) gpu_set_tf32_enabled_fn(enabled); }
    bool is_tf32_enabled() const override { return gpu_is_tf32_enabled_fn ? gpu_is_tf32_enabled_fn() : false; }
    void* get_kernel(const std::string& program_name, const std::string& program_source, const std::string& kernel_name) override {
        if (gpu_get_kernel_fn) {
            void* precompiled = gpu_get_kernel_fn(kernel_name.c_str());
            if (precompiled) {
                return precompiled;
            }
        }
        if (!program_source.empty() && gpu_compile_kernel_fn) {
            void* k = gpu_compile_kernel_fn(program_source.c_str(), kernel_name.c_str());
            if (k) {
                dynamic_kernels.insert(k);
            }
            return k;
        }
        return nullptr;
    }
    
    void* get_precompiled_kernel(int kernel_id) override {
        return nullptr; 
    }
    
    void launch(void* kernel, const std::vector<size_t>& global_work_size, const std::vector<size_t>& local_work_size, const std::vector<void*>& args, const std::vector<size_t>& arg_sizes) override {
        int gx = global_work_size.empty() ? 1 : global_work_size[0];
        int gy = global_work_size.size() < 2 ? 1 : global_work_size[1];
        int gz = global_work_size.size() < 3 ? 1 : global_work_size[2];
        if (dynamic_kernels.count(kernel) && gpu_launch_dynamic_fn) {
            gpu_launch_dynamic_fn(kernel, gx, gy, gz, const_cast<void**>(args.data()), args.size());
        } else {
            gpu_launch_fn(kernel, gx, gy, gz, const_cast<void**>(args.data()), args.size());
        }
    }
};

BackendDispatcher& BackendDispatcher::get() {
    static BackendDispatcher instance;
    return instance;
}

BackendDispatcher::BackendDispatcher() {
    auto native = std::make_shared<NativeGPUBackend>();
    if (native->is_available()) {
        std::cout << "[litetorch Info] Native GPU (CUDA/HIP) successfully initialized.\n";
        gpu_backend_ = native;
    } else {
        gpu_backend_ = nullptr;
    }
}

std::shared_ptr<DeviceBackend> BackendDispatcher::get_backend() {
    return gpu_backend_;
}

void BackendDispatcher::set_gpu_backend(std::shared_ptr<DeviceBackend> backend) {
    gpu_backend_ = backend;
}

}
