#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <chrono>
#include <iostream>

using namespace litetorch;

int main() {
    Device gpu(DeviceType::GPU, 0);

    auto tgt = Tensor::create({128, 512}, gpu, false);
    auto tgt2 = Tensor::create({128, 512}, gpu, false);
    
    float* p_tgt = tgt->data_ptr();
    float* p_tgt2 = tgt2->data_ptr();
    for (size_t i = 0; i < tgt->numel(); ++i) {
        p_tgt[i] = static_cast<float>(i % 100) / 100.0f;
        p_tgt2[i] = static_cast<float>((i + 50) % 100) / 100.0f;
    }
    tgt = tgt->to(gpu);
    tgt2 = tgt2->to(gpu);

    auto weight = Tensor::create({512}, gpu, false);
    auto bias = Tensor::create({512}, gpu, false);
    float* p_w = weight->data_ptr();
    float* p_b = bias->data_ptr();
    for (size_t i = 0; i < 512; ++i) {
        p_w[i] = 1.0f;
        p_b[i] = 0.0f;
    }
    weight = weight->to(gpu);
    bias = bias->to(gpu);

    for (int i = 0; i < 10; ++i) {
        auto tmp = Ops::add(tgt, tgt2);
        auto out = Ops::layer_norm(tmp, {512}, weight, bias, 1e-5f);
        auto fused = Ops::fused_add_layernorm(tgt, tgt2, {512}, weight, bias, 1e-5f);
    }

    auto start_seq = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        auto tmp = Ops::add(tgt, tgt2);
        auto out = Ops::layer_norm(tmp, {512}, weight, bias, 1e-5f);
    }
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_seq = end_seq - start_seq;

    auto start_fused = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        auto fused = Ops::fused_add_layernorm(tgt, tgt2, {512}, weight, bias, 1e-5f);
    }
    auto end_fused = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_fused = end_fused - start_fused;

    std::cout << "==================================================" << std::endl;
    std::cout << "          KERNEL FUSION BENCHMARK (GPU)           " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Sequential (Add + LayerNorm): " << duration_seq.count() << " ms" << std::endl;
    std::cout << "Fused (Add + LayerNorm):       " << duration_fused.count() << " ms" << std::endl;
    std::cout << "Speedup:                      " << (duration_seq.count() / duration_fused.count()) << "x" << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}
