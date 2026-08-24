#ifndef LITETORCH_TPU_BACKEND_H
#define LITETORCH_TPU_BACKEND_H

#include "litetorch/backend.h"
#include "litetorch/platform.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace litetorch {

class TPUBackend : public DeviceBackend {
public:
    void* handle = nullptr;
    bool is_ok = false;
    int current_device_id = 0;
    int num_devices = 0;
    std::string device_name = "Google TPU";
    std::mutex tpu_mutex;

    TPUBackend();
    ~TPUBackend() override;

    bool is_available() const override;
    void* allocate(size_t size) override;
    void free(void* ptr) override;
    void read(void* ptr, size_t size, void* host_ptr, size_t offset = 0) override;
    void write(void* ptr, size_t size, const void* host_ptr, size_t offset = 0) override;
    void read_async(void* ptr, size_t size, void* host_ptr, size_t offset = 0) override;
    void write_async(void* ptr, size_t size, const void* host_ptr, size_t offset = 0) override;
    void copy(void* src, void* dst, size_t size, size_t src_offset = 0, size_t dst_offset = 0) override;
    void finish() override;

    void* get_kernel(const std::string& program_name, const std::string& program_source, const std::string& kernel_name) override;
    void* get_precompiled_kernel(int kernel_id) override;
    void launch(void* kernel, const std::vector<size_t>& global_work_size, const std::vector<size_t>& local_work_size, const std::vector<void*>& args, const std::vector<size_t>& arg_sizes) override;

    void matmul(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) override;
    void matmul_ex(void* A, int64_t a_off, bool trans_a, int64_t lda,
                   void* B, int64_t b_off, bool trans_b, int64_t ldb,
                   void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) override;
    void bmm(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t B_batch, int64_t M, int64_t N, int64_t K) override;
    void matmul_half(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) override;
    void bmm_half(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t B_batch, int64_t M, int64_t N, int64_t K) override;
    void matmul_fp8(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K, float a_scale = 1.0f, float b_scale = 1.0f, float d_scale = 1.0f) override;
    void matmul_bf16(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) override;

    void sum(void* A, int64_t a_off, void* B, int64_t b_off, int64_t size) override;
    void max(void* A, int64_t a_off, void* B, int64_t b_off, int64_t size) override;
    void adamw_step(void* P, int64_t p_off, void* G, int64_t g_off, void* M, int64_t m_off, void* V, int64_t v_off, int64_t size, float lr_t, float beta1, float beta2, float eps, float weight_decay) override;
    void flash_attention(void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, void* O, int64_t o_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) override;
    void flash_attention_half(void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, void* O, int64_t o_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) override;
    void flash_attention_backward(void* dQ, int64_t dq_off, void* dK, int64_t dk_off, void* dV, int64_t dv_off, void* O, int64_t o_off, void* dO, int64_t do_off, void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) override;
    void flash_attention_backward_half(void* dQ, int64_t dq_off, void* dK, int64_t dk_off, void* dV, int64_t dv_off, void* O, int64_t o_off, void* dO, int64_t do_off, void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) override;

    void cat_forward(void* input, int64_t in_off, void* output, int64_t out_off, int64_t outer_size, int64_t inner_size, int64_t dim_size, int64_t concat_dim_size, int64_t offset) override;
    void cat_backward(void* grad_output, int64_t gout_off, void* grad_input, int64_t gin_off, int64_t outer_size, int64_t inner_size, int64_t dim_size, int64_t concat_dim_size, int64_t offset) override;
    void moe_gate(void* logits, int64_t l_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, int64_t N, int64_t E, int64_t top_k) override;
    void moe_gate_backward(void* grad_output, int64_t gout_off, void* input, int64_t in_off, void* gate_weight, int64_t gw_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, void* grad_input, int64_t gin_off, void* grad_gate_weight, int64_t ggw_off, int64_t N, int64_t D, int64_t E, int64_t top_k) override;
    void moe_expert_forward(void* input, int64_t in_off, void* expert_weight, int64_t ew_off, void* expert_bias, int64_t eb_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, void* output, int64_t out_off, int64_t N, int64_t D, int64_t out_features, int64_t expert_idx, int64_t top_k) override;
    void moe_expert_backward(void* grad_output, int64_t gout_off, void* input, int64_t in_off, void* expert_weight, int64_t ew_off, void* expert_bias, int64_t eb_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, void* grad_input, int64_t gin_off, void* grad_expert, int64_t ge_off, void* grad_bias, int64_t gb_off, void* grad_probs, int64_t gp_off, int64_t N, int64_t D, int64_t out_features, int64_t expert_idx, int64_t top_k) override;

    void* start_recording() override;
    void* stop_recording(void* stream_capture) override;
    void launch_graph(void* graph) override;
    void free_graph(void* graph) override;

    void* get_comm_stream() override;
    void sync_stream(void* stream) override;
    void set_device(int device_id) override;
    int get_device_count() const;
    std::string get_device_name(int id) const;

private:
    void init_tpu_driver();
    void systolic_gemm_tile(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K);
};

}

#endif
