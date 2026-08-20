#ifndef LITETORCH_BACKEND_H
#define LITETORCH_BACKEND_H

#include "litetorch/device.h"
#include <string>
#include <vector>
#include <memory>

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
    virtual void matmul(void* A, int a_off, void* B, int b_off, void* C, int c_off, int M, int N, int K) = 0;
    virtual void bmm(void* A, int a_off, void* B, int b_off, void* C, int c_off, int B_batch, int M, int N, int K) = 0;
    virtual void matmul_half(void* A, int a_off, void* B, int b_off, void* C, int c_off, int M, int N, int K) = 0;
    virtual void bmm_half(void* A, int a_off, void* B, int b_off, void* C, int c_off, int B_batch, int M, int N, int K) = 0;
    virtual void sum(void* A, int a_off, void* B, int b_off, int size) = 0;
    virtual void max(void* A, int a_off, void* B, int b_off, int size) = 0;
    virtual void adamw_step(void* P, int p_off, void* G, int g_off, void* M, int m_off, void* V, int v_off, int size, float lr_t, float beta1, float beta2, float eps, float weight_decay) = 0;
    virtual void flash_attention(void* Q, int q_off, void* K, int k_off, void* V, int v_off, void* O, int o_off, int B, int H, int H_kv, int Tq, int Tk, int D, float scale) = 0;
    virtual void flash_attention_half(void* Q, int q_off, void* K, int k_off, void* V, int v_off, void* O, int o_off, int B, int H, int H_kv, int Tq, int Tk, int D, float scale) = 0;
    virtual void flash_attention_backward(void* dQ, int dq_off, void* dK, int dk_off, void* dV, int dv_off, void* O, int o_off, void* dO, int do_off, void* Q, int q_off, void* K, int k_off, void* V, int v_off, int B, int H, int H_kv, int Tq, int Tk, int D, float scale) = 0;
    virtual void flash_attention_backward_half(void* dQ, int dq_off, void* dK, int dk_off, void* dV, int dv_off, void* O, int o_off, void* dO, int do_off, void* Q, int q_off, void* K, int k_off, void* V, int v_off, int B, int H, int H_kv, int Tq, int Tk, int D, float scale) = 0;
    
    virtual void cat_forward(void* input, int in_off, void* output, int out_off, int outer_size, int inner_size, int dim_size, int concat_dim_size, int offset) = 0;
    virtual void cat_backward(void* grad_output, int gout_off, void* grad_input, int gin_off, int outer_size, int inner_size, int dim_size, int concat_dim_size, int offset) = 0;
    virtual void moe_gate(void* logits, int l_off, void* probs, int p_off, void* indices, int idx_off, int N, int E, int top_k) = 0;
    virtual void moe_gate_backward(void* grad_output, int gout_off, void* input, int in_off, void* gate_weight, int gw_off, void* probs, int p_off, void* indices, int idx_off, void* grad_input, int gin_off, void* grad_gate_weight, int ggw_off, int N, int D, int E, int top_k) = 0;
    virtual void moe_expert_forward(void* input, int in_off, void* expert_weight, int ew_off, void* expert_bias, int eb_off, void* probs, int p_off, void* indices, int idx_off, void* output, int out_off, int N, int D, int out_features, int expert_idx, int top_k) = 0;
    virtual void moe_expert_backward(void* grad_output, int gout_off, void* input, int in_off, void* expert_weight, int ew_off, void* expert_bias, int eb_off, void* probs, int p_off, void* indices, int idx_off, void* grad_input, int gin_off, void* grad_expert, int ge_off, void* grad_bias, int gb_off, void* grad_probs, int gp_off, int N, int D, int out_features, int expert_idx, int top_k) = 0;
    
    virtual void* start_recording() = 0;
    virtual void* stop_recording(void* stream_capture) = 0;
    virtual void launch_graph(void* graph) = 0;
    virtual void free_graph(void* graph) = 0;

    virtual void* get_comm_stream() = 0;
    virtual void sync_stream(void* stream) = 0;
    virtual void set_device(int device_id) = 0;
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
