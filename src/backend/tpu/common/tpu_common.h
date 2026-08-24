#ifndef LITETORCH_TPU_COMMON_H
#define LITETORCH_TPU_COMMON_H

#include <string>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <memory>

namespace litetorch {
namespace tpu_internal {

constexpr int64_t TPU_SYSTOLIC_TILE_SIZE = 128;
constexpr size_t TPU_HBM_ALIGNMENT = 64;

struct TPUDriverState {
    void* handle = nullptr;
    bool is_available = false;
    int num_devices = 0;
    int current_device = 0;
    std::string device_name = "Google TPU";
};

TPUDriverState& get_tpu_driver_state();
bool init_tpu_runtime();
void shutdown_tpu_runtime();

void* tpu_hbm_allocate(size_t size);
void tpu_hbm_free(void* ptr);
void tpu_hbm_read(void* ptr, size_t size, void* host_ptr, size_t offset);
void tpu_hbm_write(void* ptr, size_t size, const void* host_ptr, size_t offset);
void tpu_hbm_copy(void* src, void* dst, size_t size, size_t src_offset, size_t dst_offset);

void tpu_systolic_matmul(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K);
void tpu_systolic_matmul_ex(const float* A, bool trans_a, int64_t lda,
                            const float* B, bool trans_b, int64_t ldb,
                            float* C, int64_t M, int64_t N, int64_t K);
void tpu_systolic_bmm(const float* A, const float* B, float* C, int64_t B_batch, int64_t M, int64_t N, int64_t K);

void tpu_flash_attention_forward(const float* Q, const float* K, const float* V, float* O,
                                 int64_t B, int64_t H, int64_t H_kv, int64_t Tq, int64_t Tk, int64_t D, float scale);

void tpu_adamw_update(float* p, const float* g, float* m, float* v, int64_t size,
                      float lr_t, float beta1, float beta2, float eps, float weight_decay);

}
}

#endif
