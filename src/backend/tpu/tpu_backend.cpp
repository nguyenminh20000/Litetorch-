#include "tpu_backend.h"
#include "common/tpu_common.h"
#include "litetorch/tpu.h"
#include "litetorch/thread_pool.h"
#ifdef LITETORCH_TPU_PJRT
#include "pjrt/tpu_pjrt_backend.h"
#else
namespace tpu_pjrt {
class TpuPjrtBackend {
public:
    static TpuPjrtBackend& instance() { static TpuPjrtBackend b; return b; }
    bool initialize() { return false; }
    void shutdown() {}
    void* allocate(size_t) { return nullptr; }
    bool owns(void*) { return false; }
    void free_buffer(void*) {}
    size_t buffer_bytes(void*) { return 0; }
    bool read_buffer(void*, void*, size_t) { return false; }
    bool write_buffer(void*, const void*, size_t) { return false; }
    void finish() {}
    bool matmul(const float*, const float*, float*, int64_t, int64_t, int64_t) { return false; }
    bool relu(const float*, float*, int64_t) { return false; }
    bool add(const float*, const float*, float*, int64_t) { return false; }
};
}
#endif
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace litetorch {

TPUBackend::TPUBackend() {
    tpu_internal::init_tpu_runtime();
    if (tpu_pjrt::TpuPjrtBackend::instance().initialize()) {
        use_pjrt = true;
    }
}

TPUBackend::~TPUBackend() {
    if (use_pjrt) tpu_pjrt::TpuPjrtBackend::instance().shutdown();
    tpu_internal::shutdown_tpu_runtime();
}

bool TPUBackend::is_available() const {
    if (use_pjrt) return true;
    return tpu_internal::get_tpu_driver_state().is_available;
}

void* TPUBackend::allocate(size_t size) {
    if (use_pjrt) {
        void* p = tpu_pjrt::TpuPjrtBackend::instance().allocate(size);
        if (p) return p;
    }
    return tpu_internal::tpu_hbm_allocate(size);
}

void TPUBackend::free(void* ptr) {
    if (use_pjrt && tpu_pjrt::TpuPjrtBackend::instance().owns(ptr)) {
        tpu_pjrt::TpuPjrtBackend::instance().free_buffer(ptr);
        return;
    }
    tpu_internal::tpu_hbm_free(ptr);
}

void TPUBackend::read(void* ptr, size_t size, void* host_ptr, size_t offset) {
    if (use_pjrt) {
        auto& pjrt = tpu_pjrt::TpuPjrtBackend::instance();
        size_t total = pjrt.buffer_bytes(ptr);
        if (total > 0 && offset + size <= total && host_ptr) {
            std::vector<char> tmp(total);
            if (pjrt.read_buffer(ptr, tmp.data(), total)) {
                std::memcpy(host_ptr, tmp.data() + offset, size);
                return;
            }
        } else if (total > 0) {
            return;
        }
    }
    tpu_internal::tpu_hbm_read(ptr, size, host_ptr, offset);
}

void TPUBackend::write(void* ptr, size_t size, const void* host_ptr, size_t offset) {
    if (use_pjrt) {
        auto& pjrt = tpu_pjrt::TpuPjrtBackend::instance();
        size_t total = pjrt.buffer_bytes(ptr);
        if (total > 0 && offset + size <= total && host_ptr) {
            if (offset == 0 && size == total) {
                if (pjrt.write_buffer(ptr, host_ptr, size)) return;
            } else {
                std::vector<char> tmp(total);
                if (pjrt.read_buffer(ptr, tmp.data(), total)) {
                    std::memcpy(tmp.data() + offset, host_ptr, size);
                    if (pjrt.write_buffer(ptr, tmp.data(), total)) return;
                }
            }
        } else if (total > 0) {
            return;
        }
    }
    tpu_internal::tpu_hbm_write(ptr, size, host_ptr, offset);
}

void TPUBackend::read_async(void* ptr, size_t size, void* host_ptr, size_t offset) {
    read(ptr, size, host_ptr, offset);
}

void TPUBackend::write_async(void* ptr, size_t size, const void* host_ptr, size_t offset) {
    write(ptr, size, host_ptr, offset);
}

void TPUBackend::copy(void* src, void* dst, size_t size, size_t src_offset, size_t dst_offset) {
    if (use_pjrt) {
        auto& pjrt = tpu_pjrt::TpuPjrtBackend::instance();
        size_t stotal = pjrt.buffer_bytes(src);
        size_t dtotal = pjrt.buffer_bytes(dst);
        if (stotal > 0 && dtotal > 0) {
            if (src_offset + size <= stotal && dst_offset + size <= dtotal) {
                std::vector<char> sbuf(stotal), dbuf(dtotal);
                if (pjrt.read_buffer(src, sbuf.data(), stotal) &&
                    pjrt.read_buffer(dst, dbuf.data(), dtotal)) {
                    std::memcpy(dbuf.data() + dst_offset, sbuf.data() + src_offset, size);
                    if (pjrt.write_buffer(dst, dbuf.data(), dtotal)) return;
                }
            }
            return;
        }
    }
    tpu_internal::tpu_hbm_copy(src, dst, size, src_offset, dst_offset);
}

void TPUBackend::finish() {
    if (use_pjrt) tpu_pjrt::TpuPjrtBackend::instance().finish();
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
    if (use_pjrt) {
        auto& pjrt = tpu_pjrt::TpuPjrtBackend::instance();
        bool owna = pjrt.owns(A), ownb = pjrt.owns(B), ownc = pjrt.owns(C);
        if (owna || ownb || ownc) {
            if (owna && ownb && ownc && M > 0 && N > 0 && K > 0) {
                size_t ab = pjrt.buffer_bytes(A), bb = pjrt.buffer_bytes(B), cb = pjrt.buffer_bytes(C);
                size_t an = ab / sizeof(float), bn = bb / sizeof(float), cn = cb / sizeof(float);
                size_t need_a = (size_t)a_off + (size_t)M * (size_t)K;
                size_t need_b = (size_t)b_off + (size_t)K * (size_t)N;
                size_t need_c = (size_t)c_off + (size_t)M * (size_t)N;
                if (need_a <= an && need_b <= bn && need_c <= cn) {
                    std::vector<float> ha(an), hb(bn), hc(cn), out((size_t)M * (size_t)N);
                    if (pjrt.read_buffer(A, ha.data(), ab) &&
                        pjrt.read_buffer(B, hb.data(), bb) &&
                        pjrt.read_buffer(C, hc.data(), cb)) {
                        const float* ap = ha.data() + a_off;
                        const float* bp = hb.data() + b_off;
                        if (!pjrt.matmul(ap, bp, out.data(), M, N, K))
                            tpu_internal::tpu_systolic_matmul(ap, bp, out.data(), M, N, K);
                        std::memcpy(hc.data() + c_off, out.data(), (size_t)M * (size_t)N * sizeof(float));
                        pjrt.write_buffer(C, hc.data(), cb);
                    }
                }
            }
            return;
        }
    }
    const float* a_ptr = reinterpret_cast<const float*>(A) + a_off;
    const float* b_ptr = reinterpret_cast<const float*>(B) + b_off;
    float* c_ptr = reinterpret_cast<float*>(C) + c_off;
    tpu_internal::tpu_systolic_matmul(a_ptr, b_ptr, c_ptr, M, N, K);
}

void TPUBackend::matmul_ex(void* A, int64_t a_off, bool trans_a, int64_t lda,
                           void* B, int64_t b_off, bool trans_b, int64_t ldb,
                           void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    if (use_pjrt) {
        auto& pjrt = tpu_pjrt::TpuPjrtBackend::instance();
        if (pjrt.owns(A) || pjrt.owns(B) || pjrt.owns(C)) {
            size_t ab = pjrt.buffer_bytes(A), bb = pjrt.buffer_bytes(B), cb = pjrt.buffer_bytes(C);
            if (ab && bb && cb) {
                std::vector<float> ha(ab / sizeof(float)), hb(bb / sizeof(float)), hc(cb / sizeof(float));
                if (pjrt.read_buffer(A, ha.data(), ab) &&
                    pjrt.read_buffer(B, hb.data(), bb) &&
                    pjrt.read_buffer(C, hc.data(), cb)) {
                    tpu_internal::tpu_systolic_matmul_ex(ha.data() + a_off, trans_a, lda,
                                                         hb.data() + b_off, trans_b, ldb,
                                                         hc.data() + c_off, M, N, K);
                    pjrt.write_buffer(C, hc.data(), cb);
                }
            }
            return;
        }
    }
    const float* a_ptr = reinterpret_cast<const float*>(A) + a_off;
    const float* b_ptr = reinterpret_cast<const float*>(B) + b_off;
    float* c_ptr = reinterpret_cast<float*>(C) + c_off;
    tpu_internal::tpu_systolic_matmul_ex(a_ptr, trans_a, lda, b_ptr, trans_b, ldb, c_ptr, M, N, K);
}

void TPUBackend::bmm(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t B_batch, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    if (use_pjrt) {
        auto& pjrt = tpu_pjrt::TpuPjrtBackend::instance();
        if (pjrt.owns(A) || pjrt.owns(B) || pjrt.owns(C)) {
            size_t ab = pjrt.buffer_bytes(A), bb = pjrt.buffer_bytes(B), cb = pjrt.buffer_bytes(C);
            if (ab && bb && cb) {
                std::vector<float> ha(ab / sizeof(float)), hb(bb / sizeof(float)), hc(cb / sizeof(float));
                if (pjrt.read_buffer(A, ha.data(), ab) &&
                    pjrt.read_buffer(B, hb.data(), bb) &&
                    pjrt.read_buffer(C, hc.data(), cb)) {
                    tpu_internal::tpu_systolic_bmm(ha.data() + a_off, hb.data() + b_off,
                                                  hc.data() + c_off, B_batch, M, N, K);
                    pjrt.write_buffer(C, hc.data(), cb);
                }
            }
            return;
        }
    }
    const float* a_ptr = reinterpret_cast<const float*>(A) + a_off;
    const float* b_ptr = reinterpret_cast<const float*>(B) + b_off;
    float* c_ptr = reinterpret_cast<float*>(C) + c_off;
    tpu_internal::tpu_systolic_bmm(a_ptr, b_ptr, c_ptr, B_batch, M, N, K);
}

void TPUBackend::matmul_half(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
    throw std::runtime_error("[litetorch Error] TPUBackend::matmul_half not implemented (FP32-only backend)");
}

void TPUBackend::bmm_half(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t B_batch, int64_t M, int64_t N, int64_t K) {
    throw std::runtime_error("[litetorch Error] TPUBackend::bmm_half not implemented (FP32-only backend)");
}

void TPUBackend::matmul_fp8(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K, float, float, float) {
    throw std::runtime_error("[litetorch Error] TPUBackend::matmul_fp8 not implemented (FP32-only backend)");
}

void TPUBackend::matmul_bf16(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
    throw std::runtime_error("[litetorch Error] TPUBackend::matmul_bf16 not implemented (FP32-only backend)");
}

void TPUBackend::sum(void* A, int64_t a_off, void* B, int64_t b_off, int64_t size) {
    if (!A || !B || size <= 0) return;
    const float* a_ptr = reinterpret_cast<const float*>(A) + a_off;
    float* b_ptr = reinterpret_cast<float*>(B) + b_off;
    if (size < 50000) {
        float total = 0.0f;
        for (int64_t i = 0; i < size; ++i) total += a_ptr[i];
        b_ptr[0] = total;
        return;
    }
    int64_t nchunks = 8;
    std::vector<float> partials(nchunks, 0.0f);
    ThreadPool::get().parallel_for(0, nchunks, [&](int64_t c) {
        int64_t s = (size * c) / nchunks;
        int64_t e = (size * (c + 1)) / nchunks;
        float acc = 0.0f;
        for (int64_t i = s; i < e; ++i) acc += a_ptr[i];
        partials[c] = acc;
    });
    float total = 0.0f;
    for (int64_t c = 0; c < nchunks; ++c) total += partials[c];
    b_ptr[0] = total;
}

void TPUBackend::max(void* A, int64_t a_off, void* B, int64_t b_off, int64_t size) {
    if (!A || !B || size <= 0) return;
    const float* a_ptr = reinterpret_cast<const float*>(A) + a_off;
    float* b_ptr = reinterpret_cast<float*>(B) + b_off;
    if (size < 50000) {
        float max_val = a_ptr[0];
        for (int64_t i = 1; i < size; ++i) {
            if (a_ptr[i] > max_val) max_val = a_ptr[i];
        }
        b_ptr[0] = max_val;
        return;
    }
    int64_t nchunks = 8;
    std::vector<float> partials(nchunks);
    ThreadPool::get().parallel_for(0, nchunks, [&](int64_t c) {
        int64_t s = (size * c) / nchunks;
        int64_t e = (size * (c + 1)) / nchunks;
        float m = a_ptr[s];
        for (int64_t i = s + 1; i < e; ++i) {
            if (a_ptr[i] > m) m = a_ptr[i];
        }
        partials[c] = m;
    });
    float max_val = partials[0];
    for (int64_t c = 1; c < nchunks; ++c) {
        if (partials[c] > max_val) max_val = partials[c];
    }
    b_ptr[0] = max_val;
}

void TPUBackend::adamw_step(void* P, int64_t p_off, void* G, int64_t g_off, void* M, int64_t m_off, void* V, int64_t v_off, int64_t size, float lr, float beta1, float beta2, float eps, float weight_decay, float bias_correction1, float bias_correction2) {
    if (!P || !G || !M || !V || size <= 0) return;
    float* p = reinterpret_cast<float*>(P) + p_off;
    const float* g = reinterpret_cast<const float*>(G) + g_off;
    float* m = reinterpret_cast<float*>(M) + m_off;
    float* v = reinterpret_cast<float*>(V) + v_off;
    tpu_internal::tpu_adamw_update(p, g, m, v, size, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
}

void TPUBackend::flash_attention(void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, void* O, int64_t o_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) {
    if (!Q || !K || !V || !O) return;
    const float* q_ptr = reinterpret_cast<const float*>(Q) + q_off;
    const float* k_ptr = reinterpret_cast<const float*>(K) + k_off;
    const float* v_ptr = reinterpret_cast<const float*>(V) + v_off;
    float* o_ptr = reinterpret_cast<float*>(O) + o_off;
    tpu_internal::tpu_flash_attention_forward(q_ptr, k_ptr, v_ptr, o_ptr, B, H, H_kv, Tq, Tk, D, scale);
}

void TPUBackend::flash_attention_half(void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, void* O, int64_t o_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) {
    throw std::runtime_error("[litetorch Error] TPUBackend::flash_attention_half not implemented (FP32-only backend)");
}

void TPUBackend::flash_attention_backward(void* dQ, int64_t dq_off, void* dK, int64_t dk_off, void* dV, int64_t dv_off,
                                          void* O, int64_t o_off, void* dO, int64_t do_off,
                                          void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off,
                                          int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) {
    if (!dQ || !dK || !dV || !O || !dO || !Q || !K || !V) return;
    float* dq_ptr = reinterpret_cast<float*>(dQ) + dq_off;
    float* dk_ptr = reinterpret_cast<float*>(dK) + dk_off;
    float* dv_ptr = reinterpret_cast<float*>(dV) + dv_off;
    const float* o_ptr = reinterpret_cast<const float*>(O) + o_off;
    const float* do_ptr = reinterpret_cast<const float*>(dO) + do_off;
    const float* q_ptr = reinterpret_cast<const float*>(Q) + q_off;
    const float* k_ptr = reinterpret_cast<const float*>(K) + k_off;
    const float* v_ptr = reinterpret_cast<const float*>(V) + v_off;
    tpu_internal::tpu_flash_attention_backward(q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr, B, H, H_kv, Tq, Tk, D, scale);
}

void TPUBackend::flash_attention_backward_half(void* dQ, int64_t dq_off, void* dK, int64_t dk_off, void* dV, int64_t dv_off,
                                               void* O, int64_t o_off, void* dO, int64_t do_off,
                                               void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off,
                                               int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) {
    throw std::runtime_error("[litetorch Error] TPUBackend::flash_attention_backward_half not implemented (FP32-only backend)");
}

void TPUBackend::cat_forward(void* input, int64_t in_off, void* output, int64_t out_off, int64_t outer_size, int64_t inner_size, int64_t dim_size, int64_t concat_dim_size, int64_t offset) {
    if (!input || !output) return;
    const float* in_ptr = reinterpret_cast<const float*>(input) + in_off;
    float* out_ptr = reinterpret_cast<float*>(output) + out_off;
    for (int64_t i = 0; i < outer_size; ++i) {
        const float* src = in_ptr + i * dim_size * inner_size;
        float* dst = out_ptr + (i * concat_dim_size + offset) * inner_size;
        std::memcpy(dst, src, dim_size * inner_size * sizeof(float));
    }
}

void TPUBackend::cat_backward(void* grad_output, int64_t gout_off, void* grad_input, int64_t gin_off, int64_t outer_size, int64_t inner_size, int64_t dim_size, int64_t concat_dim_size, int64_t offset) {
    if (!grad_output || !grad_input) return;
    const float* gout_ptr = reinterpret_cast<const float*>(grad_output) + gout_off;
    float* gin_ptr = reinterpret_cast<float*>(grad_input) + gin_off;
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
