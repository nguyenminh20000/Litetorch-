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
typedef void (*gpu_matmul_t)(void*, int, void*, int, void*, int, int, int, int);
typedef void (*gpu_bmm_t)(void*, int, void*, int, void*, int, int, int, int, int);
typedef void (*gpu_matmul_half_t)(void*, int, void*, int, void*, int, int, int, int);
typedef void (*gpu_bmm_half_t)(void*, int, void*, int, void*, int, int, int, int, int);
typedef void (*gpu_sum_t)(void*, int, void*, int, int);
typedef void (*gpu_max_t)(void*, int, void*, int, int);
typedef void (*gpu_adamw_step_t)(void*, int, void*, int, void*, int, void*, int, int, float, float, float, float, float);
typedef void (*gpu_flash_attention_t)(void*, int, void*, int, void*, int, void*, int, int, int, int, int, int, int, float);
typedef void (*gpu_flash_attention_half_t)(void*, int, void*, int, void*, int, void*, int, int, int, int, int, int, int, float);
typedef void (*gpu_flash_attention_backward_t)(void*, int, void*, int, void*, int, void*, int, void*, int, void*, int, void*, int, void*, int, int, int, int, int, int, int, float);
typedef void (*gpu_flash_attention_backward_half_t)(void*, int, void*, int, void*, int, void*, int, void*, int, void*, int, void*, int, void*, int, int, int, int, int, int, int, float);

typedef void (*gpu_cat_forward_t)(void*, int, void*, int, int, int, int, int, int);
typedef void (*gpu_cat_backward_t)(void*, int, void*, int, int, int, int, int, int);
typedef void (*gpu_moe_gate_t)(void*, int, void*, int, void*, int, int, int, int);
typedef void (*gpu_moe_gate_backward_t)(void*, int, void*, int, void*, int, void*, int, void*, int, void*, int, void*, int, int, int, int, int);
typedef void (*gpu_moe_expert_forward_t)(void*, int, void*, int, void*, int, void*, int, void*, int, void*, int, int, int, int, int, int);
typedef void (*gpu_moe_expert_backward_t)(void*, int, void*, int, void*, int, void*, int, void*, int, void*, int, void*, int, void*, int, void*, int, void*, int, int, int, int, int, int);

typedef void* (*gpu_start_recording_t)();
typedef void* (*gpu_stop_recording_t)(void*);
typedef void (*gpu_launch_graph_t)(void*);
typedef void (*gpu_free_graph_t)(void*);

typedef void* (*gpu_get_comm_stream_t)();
typedef void (*gpu_sync_stream_t)(void*);
typedef void (*gpu_set_device_t)(int);

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
    gpu_bmm_t gpu_bmm_fn = nullptr;
    gpu_matmul_half_t gpu_matmul_half_fn = nullptr;
    gpu_bmm_half_t gpu_bmm_half_fn = nullptr;
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
    gpu_get_comm_stream_t gpu_get_comm_stream_fn = nullptr;
    gpu_sync_stream_t gpu_sync_stream_fn = nullptr;
    std::unordered_set<void*> dynamic_kernels;
    bool is_ok = false;

    NativeGPUBackend() {
        if (getenv("LITETORCH_NO_GPU") || getenv("LITETORCH_NO_NATIVE_GPU")) {
            return;
        }
        const char* paths[] = { "./build/liblitetorch_gpu.so", "liblitetorch_gpu.so" };
        for (const char* path : paths) {
            handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
            if (handle) break;
        }

        if (!handle) return;

        gpu_init_fn = (gpu_init_t)dlsym(handle, "gpu_init");
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
        gpu_bmm_fn = (gpu_bmm_t)dlsym(handle, "gpu_bmm");
        gpu_matmul_half_fn = (gpu_matmul_half_t)dlsym(handle, "gpu_matmul_half");
        gpu_bmm_half_fn = (gpu_bmm_half_t)dlsym(handle, "gpu_bmm_half");
        gpu_sum_fn = (gpu_sum_t)dlsym(handle, "gpu_sum_forward");
        gpu_max_fn = (gpu_max_t)dlsym(handle, "gpu_max_forward");
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
        gpu_get_comm_stream_fn = (gpu_get_comm_stream_t)dlsym(handle, "gpu_get_comm_stream");
        gpu_sync_stream_fn = (gpu_sync_stream_t)dlsym(handle, "gpu_sync_stream");
        gpu_set_device_fn = (gpu_set_device_t)dlsym(handle, "gpu_set_device");

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
    void sum(void* A, int a_off, void* B, int b_off, int size) override { if (gpu_sum_fn) gpu_sum_fn(A, a_off, B, b_off, size); }
    void max(void* A, int a_off, void* B, int b_off, int size) override { if (gpu_max_fn) gpu_max_fn(A, a_off, B, b_off, size); }
    void matmul(void* A, int a_off, void* B, int b_off, void* C, int c_off, int M, int N, int K) override { if (gpu_matmul_fn) gpu_matmul_fn(A, a_off, B, b_off, C, c_off, M, N, K); }
    void bmm(void* A, int a_off, void* B, int b_off, void* C, int c_off, int B_batch, int M, int N, int K) override { if (gpu_bmm_fn) gpu_bmm_fn(A, a_off, B, b_off, C, c_off, B_batch, M, N, K); }
    void matmul_half(void* A, int a_off, void* B, int b_off, void* C, int c_off, int M, int N, int K) override { if (gpu_matmul_half_fn) gpu_matmul_half_fn(A, a_off, B, b_off, C, c_off, M, N, K); }
    void bmm_half(void* A, int a_off, void* B, int b_off, void* C, int c_off, int B_batch, int M, int N, int K) override { if (gpu_bmm_half_fn) gpu_bmm_half_fn(A, a_off, B, b_off, C, c_off, B_batch, M, N, K); }
    void adamw_step(void* P, int p_off, void* G, int g_off, void* M_state, int m_off, void* V, int v_off, int size, float lr_t, float beta1, float beta2, float eps, float weight_decay) override {
        if (gpu_adamw_step_fn) gpu_adamw_step_fn(P, p_off, G, g_off, M_state, m_off, V, v_off, size, lr_t, beta1, beta2, eps, weight_decay);
    }
    void flash_attention(void* Q, int q_off, void* K, int k_off, void* V, int v_off, void* O, int o_off, int B, int H, int H_kv, int Tq, int Tk, int D, float scale) override {
        if (gpu_flash_attention_fn) gpu_flash_attention_fn(Q, q_off, K, k_off, V, v_off, O, o_off, B, H, H_kv, Tq, Tk, D, scale);
    }
    void flash_attention_half(void* Q, int q_off, void* K, int k_off, void* V, int v_off, void* O, int o_off, int B, int H, int H_kv, int Tq, int Tk, int D, float scale) override {
        if (gpu_flash_attention_half_fn) gpu_flash_attention_half_fn(Q, q_off, K, k_off, V, v_off, O, o_off, B, H, H_kv, Tq, Tk, D, scale);
    }
    void flash_attention_backward(void* dQ, int dq_off, void* dK, int dk_off, void* dV, int dv_off, void* O, int o_off, void* dO, int do_off, void* Q, int q_off, void* K, int k_off, void* V, int v_off, int B, int H, int H_kv, int Tq, int Tk, int D, float scale) override {
        if (gpu_flash_attention_backward_fn) gpu_flash_attention_backward_fn(dQ, dq_off, dK, dk_off, dV, dv_off, O, o_off, dO, do_off, Q, q_off, K, k_off, V, v_off, B, H, H_kv, Tq, Tk, D, scale);
    }
    void flash_attention_backward_half(void* dQ, int dq_off, void* dK, int dk_off, void* dV, int dv_off, void* O, int o_off, void* dO, int do_off, void* Q, int q_off, void* K, int k_off, void* V, int v_off, int B, int H, int H_kv, int Tq, int Tk, int D, float scale) override {
        if (gpu_flash_attention_backward_half_fn) gpu_flash_attention_backward_half_fn(dQ, dq_off, dK, dk_off, dV, dv_off, O, o_off, dO, do_off, Q, q_off, K, k_off, V, v_off, B, H, H_kv, Tq, Tk, D, scale);
    }
    
    void cat_forward(void* input, int in_off, void* output, int out_off, int outer_size, int inner_size, int dim_size, int concat_dim_size, int offset) override {
        if (gpu_cat_forward_fn) gpu_cat_forward_fn(input, in_off, output, out_off, outer_size, inner_size, dim_size, concat_dim_size, offset);
    }
    void cat_backward(void* grad_output, int gout_off, void* grad_input, int gin_off, int outer_size, int inner_size, int dim_size, int concat_dim_size, int offset) override {
        if (gpu_cat_backward_fn) gpu_cat_backward_fn(grad_output, gout_off, grad_input, gin_off, outer_size, inner_size, dim_size, concat_dim_size, offset);
    }
    void moe_gate(void* logits, int l_off, void* probs, int p_off, void* indices, int idx_off, int N, int E, int top_k) override {
        if (gpu_moe_gate_fn) gpu_moe_gate_fn(logits, l_off, probs, p_off, indices, idx_off, N, E, top_k);
    }
    void moe_gate_backward(void* grad_output, int gout_off, void* input, int in_off, void* gate_weight, int gw_off, void* probs, int p_off, void* indices, int idx_off, void* grad_input, int gin_off, void* grad_gate_weight, int ggw_off, int N, int D, int E, int top_k) override {
        if (gpu_moe_gate_backward_fn) gpu_moe_gate_backward_fn(grad_output, gout_off, input, in_off, gate_weight, gw_off, probs, p_off, indices, idx_off, grad_input, gin_off, grad_gate_weight, ggw_off, N, D, E, top_k);
    }
    void moe_expert_forward(void* input, int in_off, void* expert_weight, int ew_off, void* expert_bias, int eb_off, void* probs, int p_off, void* indices, int idx_off, void* output, int out_off, int N, int D, int out_features, int expert_idx, int top_k) override {
        if (gpu_moe_expert_forward_fn) gpu_moe_expert_forward_fn(input, in_off, expert_weight, ew_off, expert_bias, eb_off, probs, p_off, indices, idx_off, output, out_off, N, D, out_features, expert_idx, top_k);
    }
    void moe_expert_backward(void* grad_output, int gout_off, void* input, int in_off, void* expert_weight, int ew_off, void* expert_bias, int eb_off, void* probs, int p_off, void* indices, int idx_off, void* grad_input, int gin_off, void* grad_expert, int ge_off, void* grad_bias, int gb_off, void* grad_probs, int gp_off, int N, int D, int out_features, int expert_idx, int top_k) override {
        if (gpu_moe_expert_backward_fn) gpu_moe_expert_backward_fn(grad_output, gout_off, input, in_off, expert_weight, ew_off, expert_bias, eb_off, probs, p_off, indices, idx_off, grad_input, gin_off, grad_expert, ge_off, grad_bias, gb_off, grad_probs, gp_off, N, D, out_features, expert_idx, top_k);
    }
    
    void* start_recording() override { return gpu_start_recording_fn ? gpu_start_recording_fn() : nullptr; }
    void* stop_recording(void* stream_capture) override { return gpu_stop_recording_fn ? gpu_stop_recording_fn(stream_capture) : nullptr; }
    void launch_graph(void* graph) override { if (gpu_launch_graph_fn) gpu_launch_graph_fn(graph); }
    void free_graph(void* graph) override { if (gpu_free_graph_fn) gpu_free_graph_fn(graph); }
    
    void* get_comm_stream() override { return gpu_get_comm_stream_fn ? gpu_get_comm_stream_fn() : nullptr; }
    void sync_stream(void* stream) override { if (gpu_sync_stream_fn && stream) gpu_sync_stream_fn(stream); }
    void set_device(int device_id) override { if (gpu_set_device_fn) gpu_set_device_fn(device_id); }
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
