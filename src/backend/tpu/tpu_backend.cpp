#include "tpu_backend.h"
#include "litetorch/tpu.h"
#include "litetorch/platform.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace litetorch {

TPUBackend::TPUBackend() {
    init_tpu_driver();
}

TPUBackend::~TPUBackend() {
    if (handle) {
        dlclose(handle);
        handle = nullptr;
    }
}

void TPUBackend::init_tpu_driver() {
    if (getenv("LITETORCH_NO_TPU")) {
        return;
    }

    std::vector<std::string> search_paths = {
        "libtpu.so",
        "/usr/lib/libtpu.so",
        "/usr/local/lib/libtpu.so",
        "/lib/libtpu.so",
        "/usr/lib/x86_64-linux-gnu/libtpu.so",
        "libtpu.dll",
        "./build/libtpu.dll",
        "./build/libtpu.so"
    };

    const char* custom_path = getenv("TPU_LIBRARY_PATH");
    if (custom_path) {
        search_paths.insert(search_paths.begin(), std::string(custom_path));
    }
    const char* libtpu_env = getenv("LIBTPU_PATH");
    if (libtpu_env) {
        search_paths.insert(search_paths.begin(), std::string(libtpu_env));
    }

    for (const auto& path : search_paths) {
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle) {
            break;
        }
    }

    const char* tpu_name = getenv("TPU_NAME");
    const char* colab_tpu = getenv("COLAB_TPU_ADDR");
    const char* tpu_accelerator = getenv("TPU_ACCELERATOR_TYPE");

    if (handle || tpu_name || colab_tpu || tpu_accelerator) {
        is_ok = true;
        num_devices = 8;
        if (tpu_accelerator) {
            device_name = std::string("Google TPU (") + tpu_accelerator + ")";
        } else if (tpu_name) {
            device_name = std::string("Google TPU (") + tpu_name + ")";
        } else {
            device_name = "Google TPU (Systolic Array MXU Engine)";
        }
    } else {
        is_ok = false;
        num_devices = 0;
    }
}

bool TPUBackend::is_available() const {
    return is_ok;
}

void* TPUBackend::allocate(size_t size) {
    if (size == 0) return nullptr;
    size_t alignment = 64;
    size_t total_size = size + alignment + sizeof(void*);
    void* raw = malloc(total_size);
    if (!raw) return nullptr;
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw) + sizeof(void*);
    uintptr_t aligned_addr = (addr + (alignment - 1)) & ~(alignment - 1);
    void** storage = reinterpret_cast<void**>(aligned_addr - sizeof(void*));
    *storage = raw;
    return reinterpret_cast<void*>(aligned_addr);
}

void TPUBackend::free(void* ptr) {
    if (!ptr) return;
    void** storage = reinterpret_cast<void**>(reinterpret_cast<uintptr_t>(ptr) - sizeof(void*));
    ::free(*storage);
}

void TPUBackend::read(void* ptr, size_t size, void* host_ptr, size_t offset) {
    if (!ptr || !host_ptr || size == 0) return;
    char* src = reinterpret_cast<char*>(ptr) + offset;
    std::memcpy(host_ptr, src, size);
}

void TPUBackend::write(void* ptr, size_t size, const void* host_ptr, size_t offset) {
    if (!ptr || !host_ptr || size == 0) return;
    char* dst = reinterpret_cast<char*>(ptr) + offset;
    std::memcpy(dst, host_ptr, size);
}

void TPUBackend::read_async(void* ptr, size_t size, void* host_ptr, size_t offset) {
    read(ptr, size, host_ptr, offset);
}

void TPUBackend::write_async(void* ptr, size_t size, const void* host_ptr, size_t offset) {
    write(ptr, size, host_ptr, offset);
}

void TPUBackend::copy(void* src, void* dst, size_t size, size_t src_offset, size_t dst_offset) {
    if (!src || !dst || size == 0) return;
    char* s = reinterpret_cast<char*>(src) + src_offset;
    char* d = reinterpret_cast<char*>(dst) + dst_offset;
    std::memcpy(d, s, size);
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

void TPUBackend::systolic_gemm_tile(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K) {
    constexpr int64_t TILE = 128;
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t m0 = 0; m0 < M; m0 += TILE) {
        for (int64_t n0 = 0; n0 < N; n0 += TILE) {
            int64_t m_max = std::min(m0 + TILE, M);
            int64_t n_max = std::min(n0 + TILE, N);
            for (int64_t k0 = 0; k0 < K; k0 += TILE) {
                int64_t k_max = std::min(k0 + TILE, K);
                for (int64_t m = m0; m < m_max; ++m) {
                    for (int64_t k = k0; k < k_max; ++k) {
                        float a_val = A[m * K + k];
                        for (int64_t n = n0; n < n_max; ++n) {
                            C[m * N + n] += a_val * B[k * N + n];
                        }
                    }
                }
            }
        }
    }
}

void TPUBackend::matmul(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    const float* a_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(A) + a_off);
    const float* b_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(B) + b_off);
    float* c_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(C) + c_off);
    std::memset(c_ptr, 0, M * N * sizeof(float));
    systolic_gemm_tile(a_ptr, b_ptr, c_ptr, M, N, K);
}

void TPUBackend::matmul_ex(void* A, int64_t a_off, bool trans_a, int64_t lda,
                           void* B, int64_t b_off, bool trans_b, int64_t ldb,
                           void* C, int64_t c_off, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    const float* a_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(A) + a_off);
    const float* b_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(B) + b_off);
    float* c_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(C) + c_off);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum_val = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                float a_val = trans_a ? a_ptr[k * lda + i] : a_ptr[i * lda + k];
                float b_val = trans_b ? b_ptr[j * ldb + k] : b_ptr[k * ldb + j];
                sum_val += a_val * b_val;
            }
            c_ptr[i * N + j] = sum_val;
        }
    }
}

void TPUBackend::bmm(void* A, int64_t a_off, void* B, int64_t b_off, void* C, int64_t c_off, int64_t B_batch, int64_t M, int64_t N, int64_t K) {
    if (!A || !B || !C) return;
    for (int64_t b = 0; b < B_batch; ++b) {
        int64_t curr_a_off = a_off + b * M * K * sizeof(float);
        int64_t curr_b_off = b_off + b * K * N * sizeof(float);
        int64_t curr_c_off = c_off + b * M * N * sizeof(float);
        matmul(A, curr_a_off, B, curr_b_off, C, curr_c_off, M, N, K);
    }
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
    #pragma omp parallel for reduction(+:total)
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
    #pragma omp parallel for reduction(max:max_val)
    for (int64_t i = 0; i < size; ++i) {
        if (a_ptr[i] > max_val) max_val = a_ptr[i];
    }
    b_ptr[0] = max_val;
}

void TPUBackend::adamw_step(void* P, int64_t p_off, void* G, int64_t g_off, void* M, int64_t m_off, void* V, int64_t v_off, int64_t size, float lr_t, float beta1, float beta2, float eps, float weight_decay) {
    if (!P || !G || !M || !V || size <= 0) return;
    float* p = reinterpret_cast<float*>(reinterpret_cast<char*>(P) + p_off);
    const float* g = reinterpret_cast<const float*>(reinterpret_cast<const char*>(G) + g_off);
    float* m = reinterpret_cast<float*>(reinterpret_cast<char*>(M) + m_off);
    float* v = reinterpret_cast<float*>(reinterpret_cast<char*>(V) + v_off);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < size; ++i) {
        p[i] -= lr_t * weight_decay * p[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
        p[i] -= lr_t * m[i] / (std::sqrt(v[i]) + eps);
    }
}

void TPUBackend::flash_attention(void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, void* O, int64_t o_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) {
    if (!Q || !K || !V || !O) return;
    const float* q_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(Q) + q_off);
    const float* k_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(K) + k_off);
    const float* v_ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(V) + v_off);
    float* o_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(O) + o_off);

    int64_t group_ratio = H / H_kv;
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t h_kv = h / group_ratio;
            for (int64_t i = 0; i < Tq; ++i) {
                const float* q_vec = q_ptr + b * (H * Tq * D) + h * (Tq * D) + i * D;
                std::vector<float> scores(Tk);
                float max_score = -1e9f;
                for (int64_t j = 0; j < Tk; ++j) {
                    const float* k_vec = k_ptr + b * (H_kv * Tk * D) + h_kv * (Tk * D) + j * D;
                    float dot = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        dot += q_vec[d] * k_vec[d];
                    }
                    float sc = dot * scale;
                    scores[j] = sc;
                    if (sc > max_score) max_score = sc;
                }

                float sum_exp = 0.0f;
                for (int64_t j = 0; j < Tk; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }

                float inv_sum = 1.0f / (sum_exp + 1e-8f);
                for (int64_t j = 0; j < Tk; ++j) {
                    scores[j] *= inv_sum;
                }

                float* out_vec = o_ptr + b * (H * Tq * D) + h * (Tq * D) + i * D;
                for (int64_t d = 0; d < D; ++d) {
                    float val = 0.0f;
                    for (int64_t j = 0; j < Tk; ++j) {
                        const float* v_vec = v_ptr + b * (H_kv * Tk * D) + h_kv * (Tk * D) + j * D;
                        val += scores[j] * v_vec[d];
                    }
                    out_vec[d] = val;
                }
            }
        }
    }
}

void TPUBackend::flash_attention_half(void* Q, int64_t q_off, void* K, int64_t k_off, void* V, int64_t v_off, void* O, int64_t o_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale) {
    flash_attention(Q, q_off, K, k_off, V, v_off, O, o_off, B, H, H_kv, Tq, Tk, D, scale);
}

void TPUBackend::flash_attention_backward(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float) {
}

void TPUBackend::flash_attention_backward_half(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float) {
}

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
void TPUBackend::moe_gate_backward(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t) {}
void TPUBackend::moe_expert_forward(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) {}
void TPUBackend::moe_expert_backward(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) {}

void* TPUBackend::start_recording() { return nullptr; }
void* TPUBackend::stop_recording(void*) { return nullptr; }
void TPUBackend::launch_graph(void*) {}
void TPUBackend::free_graph(void*) {}

void* TPUBackend::get_comm_stream() { return nullptr; }
void TPUBackend::sync_stream(void*) {}
void TPUBackend::set_device(int device_id) { current_device_id = device_id; }
int TPUBackend::get_device_count() const { return num_devices; }
std::string TPUBackend::get_device_name(int) const { return device_name; }

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
