#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <string>

using namespace litetorch;

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cout << "FAIL: " << msg << std::endl; ++failures; } else { std::cout << "ok: " << msg << std::endl; } } while (0)

static std::shared_ptr<Tensor> iota_fp32(const std::vector<int64_t>& shape, float start = 0.01f) {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    std::vector<float> v;
    v.reserve(n);
    for (int64_t i = 0; i < n; ++i) v.push_back(start + 0.001f * (i % 97));
    return Tensor::from_vector(v, shape, Device(DeviceType::CPU, 0), false, DataType::FP32);
}

void test_paged_attention_int32_cpu() {
    int64_t S = 1, H = 2, D = 8, Bs = 4, NB = 4, MB = 2, KvH = 2;
    auto q = iota_fp32({S, H, D});
    auto k_cache = iota_fp32({NB, KvH, Bs, D});
    auto v_cache = iota_fp32({NB, KvH, Bs, D}, 0.5f);
    auto bt = Tensor::from_vector({0.0f, 1.0f}, {S, MB}, Device(DeviceType::CPU, 0), false, DataType::FP32)->cast(DataType::INT32);
    auto cl = Tensor::from_vector({5.0f}, {S}, Device(DeviceType::CPU, 0), false, DataType::FP32)->cast(DataType::INT32);
    CHECK(bt->dtype == DataType::INT32, "block_tables is INT32 (precondition)");
    auto out = Ops::paged_attention(q, k_cache, v_cache, bt, cl, Bs);
    CHECK(out->shape == std::vector<int64_t>({S, H, D}), "paged_attention output shape");
    float s = Ops::sum(out)->item();
    CHECK(std::isfinite(s) && s != 0.0f, "paged_attention INT32 path produced nonzero finite output (was abort before fix)");
}

void test_flash_qkv_fp16_split_cpu() {
    int64_t B = 1, T = 2, H = 2, D = 4;
    auto qkv = iota_fp32({B, T, 3 * H * D})->cast(DataType::FP16);
    try {
        auto out = Ops::flash_attention_qkv(qkv, H);
        (void)out;
        CHECK(false, "flash_attention_qkv unexpectedly succeeded on CPU (expected GPU-backend throw)");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        bool is_gpu_throw = msg.find("GPU native backend required") != std::string::npos;
        bool is_dataptr_throw = msg.find("data_ptr") != std::string::npos;
        CHECK(is_gpu_throw && !is_dataptr_throw,
              "flash_attention_qkv FP16 got past raw_data_ptr split sites (only pre-existing GPU-backend throw remains)");
    }
}

int main() {
    test_paged_attention_int32_cpu();
    test_flash_qkv_fp16_split_cpu();
    std::cout << (failures == 0 ? "ALL CORE_W8 TESTS PASSED" : "FAILURES PRESENT") << std::endl;
    return failures == 0 ? 0 : 1;
}
