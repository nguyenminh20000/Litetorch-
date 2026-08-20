#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/optim.h"
#include "litetorch/custom_ops.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace litetorch;

void test_adamw_8bit() {
    std::cout << "Running test_adamw_8bit..." << std::endl;

    auto p1 = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {1, 3}, Device(DeviceType::CPU), true);
    auto p2 = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {1, 3}, Device(DeviceType::CPU), true);

    p1->grad = Tensor::from_vector({0.1f, -0.2f, 0.3f}, {1, 3}, Device(DeviceType::CPU));
    p2->grad = Tensor::from_vector({0.1f, -0.2f, 0.3f}, {1, 3}, Device(DeviceType::CPU));

    optim::AdamW opt1({p1}, 0.1f, 0.9f, 0.999f, 1e-8f, 0.01f);
    optim::AdamW8bit opt2({p2}, 0.1f, 0.9f, 0.999f, 1e-8f, 0.01f);

    opt1.step();
    opt2.step();

    auto v1 = p1->to_vector();
    auto v2 = p2->to_vector();

    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::abs(v1[i] - v2[i]) > 0.05f) {
            std::cerr << "AdamW8bit output mismatch from AdamW: " << v1[i] << " vs " << v2[i] << std::endl;
            std::exit(1);
        }
    }

    assert(opt2.m[0]->dtype == DataType::INT8);
    assert(opt2.v[0]->dtype == DataType::INT8);

    std::cout << "SUCCESS: test_adamw_8bit passed!" << std::endl;
}

void test_paged_attention() {
    std::cout << "Running test_paged_attention..." << std::endl;

    int num_seqs = 2;
    int num_heads = 4;
    int num_kv_heads = 2;
    int head_dim = 8;
    int block_size = 4;
    int max_num_blocks_per_seq = 2;

    auto q = Tensor::from_vector(
        std::vector<float>(num_seqs * num_heads * head_dim, 0.1f),
        {num_seqs, num_heads, head_dim},
        Device(DeviceType::CPU)
    );

    auto k_cache = Tensor::from_vector(
        std::vector<float>(4 * num_kv_heads * block_size * head_dim, 0.2f),
        {4, num_kv_heads, block_size, head_dim},
        Device(DeviceType::CPU)
    );

    auto v_cache = Tensor::from_vector(
        std::vector<float>(4 * num_kv_heads * block_size * head_dim, 0.3f),
        {4, num_kv_heads, block_size, head_dim},
        Device(DeviceType::CPU)
    );

    auto block_tables = Tensor::create({num_seqs, max_num_blocks_per_seq}, Device(DeviceType::CPU), false, DataType::FP32);
    int* bt_ptr = reinterpret_cast<int*>(block_tables->data_ptr());
    bt_ptr[0] = 0; bt_ptr[1] = 1; bt_ptr[2] = 2; bt_ptr[3] = 3;

    auto context_lens = Tensor::create({num_seqs}, Device(DeviceType::CPU), false, DataType::FP32);
    int* cl_ptr = reinterpret_cast<int*>(context_lens->data_ptr());
    cl_ptr[0] = 6; cl_ptr[1] = 8;

    auto out_cpu = Ops::paged_attention(q, k_cache, v_cache, block_tables, context_lens, block_size);
    assert(out_cpu->shape[0] == num_seqs);
    assert(out_cpu->shape[1] == num_heads);
    assert(out_cpu->shape[2] == head_dim);

    auto v_out = out_cpu->to_vector();
    for (float val : v_out) {
        if (std::abs(val - 0.3f) > 1e-4f) {
            std::cerr << "PagedAttention value mismatch, expected ~0.3, got: " << val << std::endl;
            std::exit(1);
        }
    }

    if (CLBackend::get().is_available()) {
        auto q_gpu = q->to(Device(DeviceType::GPU, 0));
        auto k_gpu = k_cache->to(Device(DeviceType::GPU, 0));
        auto v_gpu = v_cache->to(Device(DeviceType::GPU, 0));
        auto bt_gpu = block_tables->to(Device(DeviceType::GPU, 0));
        auto cl_gpu = context_lens->to(Device(DeviceType::GPU, 0));

        auto out_gpu = Ops::paged_attention(q_gpu, k_gpu, v_gpu, bt_gpu, cl_gpu, block_size);
        auto v_gpu_out = out_gpu->to_vector();

        for (size_t i = 0; i < v_out.size(); ++i) {
            if (std::abs(v_out[i] - v_gpu_out[i]) > 1e-5f) {
                std::cerr << "GPU PagedAttention mismatch from CPU: " << v_out[i] << " vs " << v_gpu_out[i] << std::endl;
                std::exit(1);
            }
        }
        std::cout << "GPU verification passed!" << std::endl;
    }

    std::cout << "SUCCESS: test_paged_attention passed!" << std::endl;
}

void test_custom_operator() {
    std::cout << "Running test_custom_operator..." << std::endl;

    auto cpu_forward = [](const std::vector<std::shared_ptr<Tensor>>& args) -> std::shared_ptr<Tensor> {
        auto x = args[0];
        auto out = Tensor::create(x->shape, x->device, false);
        float* x_ptr = x->data_ptr();
        float* out_ptr = out->data_ptr();
        for (size_t i = 0; i < x->numel(); ++i) {
            out_ptr[i] = x_ptr[i] * 2.0f;
        }
        return out;
    };

    auto backward_func = [](std::shared_ptr<Tensor> grad_output,
                            const std::vector<std::shared_ptr<Tensor>>& inputs,
                            std::shared_ptr<Tensor> output) -> std::vector<std::shared_ptr<Tensor>> {
        auto grad_in = Tensor::create(inputs[0]->shape, inputs[0]->device, false);
        float* go_ptr = grad_output->data_ptr();
        float* gi_ptr = grad_in->data_ptr();
        for (size_t i = 0; i < grad_in->numel(); ++i) {
            gi_ptr[i] = go_ptr[i] * 2.0f;
        }
        return { grad_in };
    };

    custom_ops::Registry::get().register_op("mul_by_two", cpu_forward, nullptr, backward_func);

    auto x = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {1, 3}, Device(DeviceType::CPU), true);
    auto y = custom_ops::Registry::get().call("mul_by_two", {x});

    auto v_y = y->to_vector();
    assert(std::abs(v_y[0] - 2.0f) < 1e-5f);
    assert(std::abs(v_y[1] - 4.0f) < 1e-5f);
    assert(std::abs(v_y[2] - 6.0f) < 1e-5f);

    y->backward(Tensor::from_vector({1.0f, 1.0f, 1.0f}, {1, 3}, Device(DeviceType::CPU)));

    auto v_dx = x->grad->to_vector();
    assert(std::abs(v_dx[0] - 2.0f) < 1e-5f);
    assert(std::abs(v_dx[1] - 2.0f) < 1e-5f);
    assert(std::abs(v_dx[2] - 2.0f) < 1e-5f);

    std::cout << "SUCCESS: test_custom_operator passed!" << std::endl;
}

int main() {
    test_adamw_8bit();
    test_paged_attention();
    test_custom_operator();
    std::cout << "ALL TESTS PASSED!" << std::endl;
    return 0;
}
