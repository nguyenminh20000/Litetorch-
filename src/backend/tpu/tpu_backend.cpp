#include "tpu_backend.h"
#include "common/tpu_common.h"
#include "litetorch/tpu.h"
#include <iostream>
#include <cstring>

namespace litetorch {

TPUBackend::TPUBackend() {
    tpu_internal::init_tpu_runtime();
}

TPUBackend::~TPUBackend() {
    tpu_internal::shutdown_tpu_runtime();
}

bool TPUBackend::is_available() const {
    return tpu_internal::get_tpu_driver_state().is_available;
}

void* TPUBackend::allocate(size_t size) {
    return tpu_internal::tpu_hbm_allocate(size);
}

void TPUBackend::free(void* ptr) {
    tpu_internal::tpu_hbm_free(ptr);
}

void TPUBackend::read(void* ptr, size_t size, void* host_ptr, size_t offset) {
    tpu_internal::tpu_hbm_read(ptr, size, host_ptr, offset);
}

void TPUBackend::write(void* ptr, size_t size, const void* host_ptr, size_t offset) {
    tpu_internal::tpu_hbm_write(ptr, size, host_ptr, offset);
}

void TPUBackend::read_async(void* ptr, size_t size, void* host_ptr, size_t offset) {
    read(ptr, size, host_ptr, offset);
}

void TPUBackend::write_async(void* ptr, size_t size, const void* host_ptr, size_t offset) {
    write(ptr, size, host_ptr, offset);
}

void TPUBackend::copy(void* src, void* dst, size_t size, size_t src_offset, size_t dst_offset) {
    tpu_internal::tpu_hbm_copy(src, dst, size, src_offset, dst_offset);
}

void TPUBackend::finish() {
}

void* TPUBackend::get_kernel(const std::string&, const std::string&, const std::string&) {
    return nullptr;
}

void* TPUBackend::get_precompiled_kernel(int) {
    return nullptr;
}

void TPUBackend::launch(void*, const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<void*>&, const std::vector<size_t>&) {
}

void TPUBackend::matmul(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    const float* a_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(A) + a_off);
    const float* b_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(B) + b_off);
    float* c_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(C) + c_off);
    tpu_internal::tpu_systolic_matmul(a_ptr, b_ptr, c_ptr, M, N, K);
}

void TPUBackend::matmul_ex(void* A, int64_t a_off, bool trans_a, int64_t lda,
                           void* B, int64_t b_off, bool trans_b, int64_t ldb,
                           void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    const float* a_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(A) + a_off);
    const float* b_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(B) + b_off);
    float* c_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(C) + c_off);
    tpu_internal::tpu_systolic_matmul_ex(a_ptr, trans_a, lda, b_ptr, trans_b, ldb, c_ptr, M, N, K);
}

void TPUBackend::bmm(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t B_batch, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    const float* a_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(A) + a_off);
    const float* b_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(B) + b_off);
    float* c_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(C) + c_off);
    tpu_internal::tpu_systolic_bmm(a_ptr, b_ptr, c_ptr, B_batch, M, N, K);
}

void TPUBackend::matmul_half(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
    matmul(A, a_off, B, b_off, C, c_off, M, N, K);
}

void TPUBackend::bmm_half(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t B_batch, int64_t M, int64_t N, int64_t K) {
    bmm(A, a_off, B, b_off, C, c_off, B_batch, M, N, K);
}

void TPUBackend::matmul_fp8(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K, float, float, float) {
    matmul(A, a_off, B, b_off, C, c_off, M, N, K);
}

void TPUBackend::matmul_bf16(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
    matmul(A, a_off, B, b_off, C, c_off, M, N, K);
}

void TPUBackend::sum(void* A, int64_t a_off, void* B, int64_t b_off, int64_t size) {
    if (!A || !B || size <= 0) return;
    const float* a_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(A) + a_off);
    float* b_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(B) + b_off);
    float total = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        total += a_ptr[i];
    }
    b_ptr[0] = total;
}

void TPUBackend::max(void* A, int64_t a_off, void* B, int64_t b_off, int64_t size) {
    if (!A || !B || size <= 0) return;
    const float* a_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(A) + a_off);
    float* b_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(B) + b_off);
    float max_val = a_ptr[0];
    for (int64_t i = 0; i < size; ++i) {
        if (a_ptr[i] > max_val) max_val = a_ptr[i];
    }
    b_ptr[0] = max_val;
}

void TPUBackend::adamw_step(void* P, int64_t p_off, void* G, int64_t g_off, void* M, int64_t m_off, void* V, int64_t v_off, int64_t size, float lr, float beta1, float beta2, float eps, float weight_decay, float bias_correction1, float bias_correction2) {
    if (!P || !G || !M || !V || size <= 0) return;
    float* p = reinterpret_cast<float*>(reinterpret_cast<char*>(P) + p_off);
    const float* g = reinterpret_cast<const float*>(reinterpret_cast<const char*>(G) + g_off);
    float* m = reinterpret_cast<float*>(reinterpret_cast<char*>(M) + m_off);
    float* v = reinterpret_cast<float*>(reinterpret_cast<char*>(V) + v_off);
    tpu_internal::tpu_adamw_update(p, g, m, v, size, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
}

void TPUBackend::flash_attention(void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, void* O, int64_t o_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) {
    if (!Q || !K || !V || !O) return;
    const float* q_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(Q) + q_off);
    const float* k_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(K) + k_off);
    const float* v_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(V) + v_off);
    float* o_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(O) + o_off);
    tpu_internal::tpu_flash_attention_forward(q_ptr, k_ptr, v_ptr, o_ptr, B, H, H_kv, Tq, Tk, D, scale);
}

void TPUBackend::flash_attention_half(void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, void* O, int64_t o_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) {
    flash_attention(Q, q_off, K, k_off, V, v_off, O, o_off, B, H, H_kv, Tq, Tk, D, scale);
}

void TPUBackend::flash_attention_backward(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float) {}
void TPUBackend::flash_attention_backward_half(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float) {}

void TPUBackend::cat_forward(void* input, int64_t in_off, void* output, int64_t out_off, int64_t outer_size, int64_t inner_size, int64_t dim_size, int64_t concat_dim_size, int64_t offset) {
    if (!input || !output) return;
    const float* in_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(input) + in_off);
    float* out_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(output) + out_off);
    for (int64_t i = 0; i < outer_size; ++i) {
        const float* src = in_ptr + i * dim_size * inner_size;
        float* dst = out_ptr + (i * concat_dim_size + offset) * inner_size;
        std::memcpy(dst, src, dim_size * inner_size * sizeof(float));
    }
}

void TPUBackend::cat_backward(void* grad_output, int64_t gout_off, void* grad_input, int64_t gin_off, int64_t outer_size, int64_t inner_size, int64_t dim_size, int64_t concat_dim_size, int64_t offset) {
    if (!grad_output || !grad_input) return;
    const float* gout_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(grad_output) + gout_off);
    float* gin_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(grad_input) + gin_off);
    for (int64_t i = 0; i < outer_size; ++i) {
        const float* src = gout_ptr + (i * concat_dim_size + offset) * inner_size;
        float* dst = gin_ptr + i * dim_size * inner_size;
        std::memcpy(dst, src, dim_size * inner_size * sizeof(float));
    }
}

void TPUBackend::moe_gate(void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t) {}
void TPUBackend::moe_gate_backward(void* grad_output, int64_t gout_off, void* input, int64_t in_off, void* gate_weight, int64_t gw_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, void* grad_input, int64_t gin_off, void* grad_gate_weight, int64_t ggw_off, int64_t N, int64_t D, int64_t E, int64_t top_k) {}
void TPUBackend::moe_expert_forward(void* input, int64_t in_off, void* expert_weight, int64_t ew_off, void* expert_bias, int64_t eb_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, void* output, int64_t out_off, int64_t N, int64_t D, int64_t out_features, int64_t expert_idx, int64_t top_k) {}
void TPUBackend::moe_expert_backward(void* grad_output, int64_t gout_off, void* input, int64_t in_off, void* expert_weight, int64_t ew_off, void* expert_bias, int64_t eb_off, void* probs, int64_t p_off, void* indices, int64_t idx_off, void* grad_input, int64_t gin_off, void* grad_expert, int64_t ge_off, void* grad_bias, int64_t gb_off, void* grad_probs, int64_t gp_off, int64_t N, int64_t D, int64_t out_features, int64_t expert_idx, int64_t top_k) {}

void* TPUBackend::start_recording() { return nullptr; }
void* TPUBackend::stop_recording(void*) { return nullptr; }
void TPUBackend::launch_graph(void*) {}
void TPUBackend::free_graph(void*) {}

void* TPUBackend::get_comm_stream() { return nullptr; }
void TPUBackend::sync_stream(void*) {}
void TPUBackend::set_device(int device_id) { current_device_id = device_id; }
int TPUBackend::get_device_count() const { return tpu_internal::get_tpu_driver_state().num_devices; }
std::string TPUBackend::get_device_name(int) const { return tpu_internal::get_tpu_driver_state().device_name; }

namespace tpu {

bool is_available() {
    auto backend = BackendDispatcher::get().get_tpu_backend();
    return backend && backend->is_available();
}

int device_count() {
    auto backend = BackendDispatcher::get().get_tpu_backend();
    if (!backend || !backend->is_available()) return 0;
    auto tpu_be = std::dynamic_pointer_cast<TPUBackend>(backend);
    return tpu_be ? tpu_be->get_device_count() : 0;
}

int current_device() {
    auto backend = BackendDispatcher::get().get_tpu_backend();
    if (!backend) return 0;
    auto tpu_be = std::dynamic_pointer_cast<TPUBackend>(backend);
    return tpu_be ? tpu_be->current_device_id : 0;
}

void set_device(int device_id) {
    auto backend = BackendDispatcher::get().get_tpu_backend();
    if (backend) {
        backend->set_device(device_id);
    }
}

void synchronize() {
    auto backend = BackendDispatcher::get().get_tpu_backend();
    if (backend) {
        backend->finish();
    }
}

std::string get_device_name(int device_id) {
    auto backend = BackendDispatcher::get().get_tpu_backend();
    if (!backend || !backend->is_available()) return "N/A";
    auto tpu_be = std::dynamic_pointer_cast<TPUBackend>(backend);
    return tpu_be ? tpu_be->get_device_name(device_id) : "Google TPU";
}

}

}
