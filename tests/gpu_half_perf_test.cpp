#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>

using namespace litetorch;

void test_matmul_half() {
    std::cout << "--- Testing MatMul FP16 Correctness & Performance ---" << std::endl;
    int M = 256;
    int K = 256;
    int N = 256;

    std::vector<float> a_data(M * K);
    std::vector<float> b_data(K * N);
    for (int i = 0; i < M * K; ++i) a_data[i] = sin(static_cast<float>(i)) * 0.1f;
    for (int i = 0; i < K * N; ++i) b_data[i] = cos(static_cast<float>(i)) * 0.1f;

    auto a_fp32 = Tensor::from_vector(a_data, {M, K}, Device(DeviceType::GPU, 0), false);
    auto b_fp32 = Tensor::from_vector(b_data, {K, N}, Device(DeviceType::GPU, 0), false);

    auto a_fp16 = a_fp32->cast(DataType::FP16);
    auto b_fp16 = b_fp32->cast(DataType::FP16);

    auto c_fp32 = Ops::matmul(a_fp32, b_fp32);
    auto c_fp16 = Ops::matmul(a_fp16, b_fp16);

    CLBackend::get().finish();

    auto c_fp32_cpu = c_fp32->to_vector();
    auto c_fp16_cpu = c_fp16->cast(DataType::FP32)->to_vector();

    float max_diff = 0.0f;
    for (size_t i = 0; i < c_fp32_cpu.size(); ++i) {
        float diff = std::abs(c_fp32_cpu[i] - c_fp16_cpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "MatMul FP32 vs FP16 Max Difference: " << max_diff << std::endl;
    assert(max_diff < 0.15f);

    int iterations = 100;

    auto start_32 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto out = Ops::matmul(a_fp32, b_fp32);
    }
    CLBackend::get().finish();
    auto end_32 = std::chrono::high_resolution_clock::now();
    double time_32 = std::chrono::duration<double, std::milli>(end_32 - start_32).count();

    auto start_16 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto out = Ops::matmul(a_fp16, b_fp16);
    }
    CLBackend::get().finish();
    auto end_16 = std::chrono::high_resolution_clock::now();
    double time_16 = std::chrono::duration<double, std::milli>(end_16 - start_16).count();

    std::cout << "MatMul FP32 GPU Time: " << time_32 << " ms" << std::endl;
    std::cout << "MatMul FP16 GPU Time: " << time_16 << " ms" << std::endl;
    std::cout << "MatMul Speedup: " << (time_32 / time_16) << "x" << std::endl;
}

void test_bmm_half() {
    std::cout << "\n--- Testing BMM FP16 Correctness & Performance ---" << std::endl;
    int B = 16;
    int M = 128;
    int K = 128;
    int N = 128;

    std::vector<float> a_data(B * M * K);
    std::vector<float> b_data(B * K * N);
    for (int i = 0; i < B * M * K; ++i) a_data[i] = sin(static_cast<float>(i)) * 0.1f;
    for (int i = 0; i < B * K * N; ++i) b_data[i] = cos(static_cast<float>(i)) * 0.1f;

    auto a_fp32 = Tensor::from_vector(a_data, {B, M, K}, Device(DeviceType::GPU, 0), false);
    auto b_fp32 = Tensor::from_vector(b_data, {B, K, N}, Device(DeviceType::GPU, 0), false);

    auto a_fp16 = a_fp32->cast(DataType::FP16);
    auto b_fp16 = b_fp32->cast(DataType::FP16);

    auto c_fp32 = Ops::bmm(a_fp32, b_fp32);
    auto c_fp16 = Ops::bmm(a_fp16, b_fp16);

    CLBackend::get().finish();

    auto c_fp32_cpu = c_fp32->to_vector();
    auto c_fp16_cpu = c_fp16->cast(DataType::FP32)->to_vector();

    float max_diff = 0.0f;
    for (size_t i = 0; i < c_fp32_cpu.size(); ++i) {
        float diff = std::abs(c_fp32_cpu[i] - c_fp16_cpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "BMM FP32 vs FP16 Max Difference: " << max_diff << std::endl;
    assert(max_diff < 0.15f);

    int iterations = 100;

    auto start_32 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto out = Ops::bmm(a_fp32, b_fp32);
    }
    CLBackend::get().finish();
    auto end_32 = std::chrono::high_resolution_clock::now();
    double time_32 = std::chrono::duration<double, std::milli>(end_32 - start_32).count();

    auto start_16 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto out = Ops::bmm(a_fp16, b_fp16);
    }
    CLBackend::get().finish();
    auto end_16 = std::chrono::high_resolution_clock::now();
    double time_16 = std::chrono::duration<double, std::milli>(end_16 - start_16).count();

    std::cout << "BMM FP32 GPU Time: " << time_32 << " ms" << std::endl;
    std::cout << "BMM FP16 GPU Time: " << time_16 << " ms" << std::endl;
    std::cout << "BMM Speedup: " << (time_32 / time_16) << "x" << std::endl;
}

void test_flash_attn_half() {
    std::cout << "\n--- Testing Flash Attention FP16 Correctness & Performance ---" << std::endl;
    int B = 4;
    int H = 8;
    int T = 128;
    int D = 64;

    std::vector<float> q_data(B * H * T * D);
    std::vector<float> k_data(B * H * T * D);
    std::vector<float> v_data(B * H * T * D);
    for (int i = 0; i < B * H * T * D; ++i) {
        q_data[i] = sin(static_cast<float>(i)) * 0.1f;
        k_data[i] = cos(static_cast<float>(i)) * 0.1f;
        v_data[i] = sin(static_cast<float>(i + 1)) * 0.1f;
    }

    auto q_fp32 = Tensor::from_vector(q_data, {B, H, T, D}, Device(DeviceType::GPU, 0), false);
    auto k_fp32 = Tensor::from_vector(k_data, {B, H, T, D}, Device(DeviceType::GPU, 0), false);
    auto v_fp32 = Tensor::from_vector(v_data, {B, H, T, D}, Device(DeviceType::GPU, 0), false);

    auto q_fp16 = q_fp32->cast(DataType::FP16);
    auto k_fp16 = k_fp32->cast(DataType::FP16);
    auto v_fp16 = v_fp32->cast(DataType::FP16);

    auto o_fp32 = Ops::flash_attention(q_fp32, k_fp32, v_fp32);
    auto o_fp16 = Ops::flash_attention(q_fp16, k_fp16, v_fp16);

    CLBackend::get().finish();

    auto o_fp32_cpu = o_fp32->to_vector();
    auto o_fp16_cpu = o_fp16->cast(DataType::FP32)->to_vector();

    float max_diff = 0.0f;
    for (size_t i = 0; i < o_fp32_cpu.size(); ++i) {
        float diff = std::abs(o_fp32_cpu[i] - o_fp16_cpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Flash Attention FP32 vs FP16 Max Difference: " << max_diff << std::endl;
    assert(max_diff < 0.15f);

    int iterations = 100;

    auto start_32 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto out = Ops::flash_attention(q_fp32, k_fp32, v_fp32);
    }
    CLBackend::get().finish();
    auto end_32 = std::chrono::high_resolution_clock::now();
    double time_32 = std::chrono::duration<double, std::milli>(end_32 - start_32).count();

    auto start_16 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto out = Ops::flash_attention(q_fp16, k_fp16, v_fp16);
    }
    CLBackend::get().finish();
    auto end_16 = std::chrono::high_resolution_clock::now();
    double time_16 = std::chrono::duration<double, std::milli>(end_16 - start_16).count();

    std::cout << "Flash Attention FP32 GPU Time: " << time_32 << " ms" << std::endl;
    std::cout << "Flash Attention FP16 GPU Time: " << time_16 << " ms" << std::endl;
    std::cout << "Flash Attention Speedup: " << (time_32 / time_16) << "x" << std::endl;
}

int main() {
    std::cout << "Checking GPU availability..." << std::endl;
    if (!CLBackend::get().is_available()) {
        std::cout << "No GPU available. Skipping test." << std::endl;
        return 0;
    }

    test_matmul_half();
    test_bmm_half();
    test_flash_attn_half();

    std::cout << "\nAll GPU Half Precision correctness and performance tests passed successfully!" << std::endl;
    return 0;
}
