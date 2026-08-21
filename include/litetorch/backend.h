#ifndef LITETORCH_BACKEND_H
#define LITETORCH_BACKEND_H

#include "litetorch/device.h"
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace litetorch {

class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;
    virtual bool is_available() const = 0;
    virtual void* allocate(size_t size) = 0;
    virtual void free(void* ptr) = 0;
    virtual void read(void* ptr, size_t size, void* host_ptr, size_t offset = 0) = 0;
    virtual void write(void* ptr, size_t size, const void* host_ptr, size_t offset = 0) = 0;
    virtual void read_async(void* ptr, size_t size, void* host_ptr, size_t offset = 0) = 0;
    virtual void write_async(void* ptr, size_t size, const void* host_ptr, size_t offset = 0) = 0;
    virtual void copy(void* src, void* dst, size_t size, size_t src_offset = 0, size_t dst_offset = 0) = 0;
    virtual void finish() = 0;
    virtual void* get_kernel(const std::string& program_name, const std::string& program_source, const std::string& kernel_name) = 0;
    virtual void* get_precompiled_kernel(int kernel_id) = 0;
    virtual void launch(void* kernel, const std::vector<size_t>& global_work_size, const std::vector<size_t>& local_work_size, const std::vector<void*>& args, const std::vector<size_t>& arg_sizes) = 0;
    virtual void matmul(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) = 0;
    virtual void matmul_ex(void* A, int64_t a_off, bool trans_a, int64_t lda,
                           void* B, int64_t b_off, bool trans_b, int64_t ldb,
                           void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
        matmul(A, a_off, B, b_off, C, c_off, M, N, K);
    }
    virtual void bmm(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t B_batch, int64_t M, int64_t N, int64_t K) = 0;
    virtual void matmul_half(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) = 0;
    virtual void bmm_half(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t B_batch, int64_t M, int64_t N, int64_t K) = 0;
    virtual void matmul_fp8(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K, float a_scale = 1.0f, float b_scale = 1.0f, float d_scale = 1.0f) = 0;
    virtual void matmul_bf16(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) = 0;
    virtual void sum(void* A, int64_t a_off, void* B, int64_t b_off, int64_t size) = 0;
    virtual void max(void* A, int64_t a_off, void* B, int64_t b_off, int64_t size) = 0;
    virtual void adamw_step(void* P, int64_t p_off, void* G, int64_t g_off, void* M, int64_t m_off, void* V, int64_t v_off, int64_t size, float lr_t, float beta1, float beta2, float eps, float weight_decay) = 0;
    virtual void flash_attention(void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, void* O, int64_t o_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) = 0;
    virtual void flash_attention_half(void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, void* O, int64_t o_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) = 0;
    virtual void flash_attention_backward(void* dQ, int64_t dq_off, void* dK, int64_t dk_off, void* dV, int64_t dv_off, void* O, int64_t o_off, void* dO, int64_t do_off, void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) = 0;
    virtual void flash_attention_backward_half(void* dQ, int64_t dq_off, void* dK, int64_t dk_off, void* dV, int64_t dv_off, void* O, int64_t o_off, void* dO, int64_t do_off, void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) = 0;
    
    virtual void cat_forward(void* input, int64_t in_off, void* output, int64_t out_off, int64_t outer_size, int64_t inner_size, int64_t dim_size, int64_t concat_dim_size, int64_t offset) = 0;
    virtual void cat_backward(void* grad_output, int64_t gout_off, void* grad_input, int64_t gin_off, int64_t outer_size, int64_t inner_size, int64_t dim_size, int64_t concat_dim_size, int64_t offset) = 0;
    virtual void moe_gate(void* logits, int64_t l_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, int64_t N, int64_t E, int64_t top_k) = 0;
    virtual void moe_gate_backward(void* grad_output, int64_t gout_off, void* input, int64_t in_off, void* gate_weight, int64_t gw_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, void* grad_input, int64_t gin_off, void* grad_gate_weight, int64_t ggw_off, int64_t N, int64_t D, int64_t E, int64_t top_k) = 0;
    virtual void moe_expert_forward(void* input, int64_t in_off, void* expert_weight, int64_t ew_off, void* expert_bias, int64_t eb_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, void* output, int64_t out_off, int64_t N, int64_t D, int64_t out_features, int64_t expert_idx, int64_t top_k) = 0;
    virtual void moe_expert_backward(void* grad_output, int64_t gout_off, void* input, int64_t in_off, void* expert_weight, int64_t ew_off, void* expert_bias, int64_t eb_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, void* grad_input, int64_t gin_off, void* grad_expert, int64_t ge_off, void* grad_bias, int64_t gb_off, void* grad_probs, int64_t gp_off, int64_t N, int64_t D, int64_t out_features, int64_t expert_idx, int64_t top_k) = 0;
    
    virtual void* start_recording() = 0;
    virtual void* stop_recording(void* stream_capture) = 0;
    virtual void launch_graph(void* graph) = 0;
    virtual void free_graph(void* graph) = 0;

    virtual void* get_comm_stream() = 0;
    virtual void* get_compute_stream() { return nullptr; }
    virtual void* create_stream() { return nullptr; }
    virtual void sync_stream(void* stream) = 0;
    virtual void set_device(int device_id) = 0;

    virtual void* create_event() { return nullptr; }
    virtual void record_event(void* event, void* stream) {}
    virtual void stream_wait_event(void* stream, void* event) {}
    virtual void destroy_event(void* event) {}
};

class BackendDispatcher {
public:
    static BackendDispatcher& get();
    std::shared_ptr<DeviceBackend> get_backend();
    std::shared_ptr<DeviceBackend> get_cpu_backend();
    void set_gpu_backend(std::shared_ptr<DeviceBackend> backend);
private:
    BackendDispatcher();
    std::shared_ptr<DeviceBackend> cpu_backend_;
    std::shared_ptr<DeviceBackend> gpu_backend_;
};

}

#endif
