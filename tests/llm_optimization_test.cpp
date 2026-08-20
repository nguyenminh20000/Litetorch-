#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/llm_serving.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace litetorch;

class MockModel : public nn::Module {
public:
    int vocab_size;
    std::vector<std::vector<float>> precalculated_logits;

    MockModel(int vocab_size, std::vector<std::vector<float>> logits)
        : vocab_size(vocab_size), precalculated_logits(logits) {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        int64_t seq_len = input->shape[1];
        auto out = Tensor::create({1, seq_len, vocab_size}, Device(DeviceType::CPU));
        float* out_ptr = out->data_ptr();
        for (int64_t s = 0; s < seq_len; ++s) {
            size_t idx = static_cast<size_t>(s);
            if (idx >= precalculated_logits.size()) idx = precalculated_logits.size() - 1;
            for (int v = 0; v < vocab_size; ++v) {
                out_ptr[s * vocab_size + v] = precalculated_logits[idx][v];
            }
        }
        return out;
    }
};

void test_prefix_cache() {
    std::cout << "Running test_prefix_cache..." << std::endl;

    auto& cache = nn::PrefixCache::get();
    cache.clear();

    std::vector<int64_t> prefix = {101, 102, 103};
    auto dummy_cache_val = Tensor::from_vector({1.0f, 2.0f}, {1, 2});
    std::vector<std::shared_ptr<Tensor>> kv = {dummy_cache_val};

    cache.put(prefix, kv);
    auto retrieved = cache.get_cache(prefix);
    assert(retrieved.size() == 1);
    assert(retrieved[0]->to_vector()[0] == 1.0f);

    std::vector<int64_t> bad_prefix = {101, 102, 104};
    auto not_found = cache.get_cache(bad_prefix);
    assert(not_found.empty());

    std::cout << "SUCCESS: test_prefix_cache passed!" << std::endl;
}

void test_speculative_decoding() {
    std::cout << "Running test_speculative_decoding..." << std::endl;

    std::vector<std::vector<float>> draft_logits = {
        {0.1f, 0.8f, 0.1f},
        {0.1f, 0.1f, 0.8f},
        {0.8f, 0.1f, 0.1f},
        {0.1f, 0.8f, 0.1f},
        {0.1f, 0.1f, 0.8f}
    };

    std::vector<std::vector<float>> target_logits = {
        {0.1f, 0.8f, 0.1f},
        {0.1f, 0.1f, 0.8f},
        {0.8f, 0.1f, 0.1f},
        {0.1f, 0.8f, 0.1f},
        {0.1f, 0.1f, 0.8f}
    };

    auto draft = std::make_shared<MockModel>(3, draft_logits);
    auto target = std::make_shared<MockModel>(3, target_logits);

    nn::SpeculativeEngine engine(draft, target);
    std::vector<int64_t> prompt = {0};
    auto tokens = engine.generate(prompt, 3, 2);

    assert(tokens.size() == 4);
    assert(tokens[1] == 1);
    assert(tokens[2] == 2);

    std::cout << "SUCCESS: test_speculative_decoding passed!" << std::endl;
}

void test_flash_decoding() {
    std::cout << "Running test_flash_decoding..." << std::endl;

    auto q = Tensor::from_vector({0.1f, 0.2f, 0.3f, 0.4f}, {1, 1, 1, 4});
    auto k = Tensor::from_vector({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}, {1, 1, 2, 4});
    auto v = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {1, 1, 2, 4});

    auto out_dec = Ops::flash_decoding(q, k, v, 2);
    auto out_att = Ops::flash_attention(q, k, v);

    auto vec_dec = out_dec->to_vector();
    auto vec_att = out_att->to_vector();

    for (size_t i = 0; i < vec_dec.size(); ++i) {
        assert(std::abs(vec_dec[i] - vec_att[i]) < 1e-4f);
    }

    std::cout << "SUCCESS: test_flash_decoding passed!" << std::endl;
}

void test_nf4_cast() {
    std::cout << "Running test_nf4_cast..." << std::endl;

    auto x = Tensor::from_vector({0.0f, -0.6f, 0.5f, 1.0f, -1.0f}, {1, 5});
    auto x_nf4 = x->cast(DataType::NF4);
    assert(x_nf4->dtype == DataType::NF4);
    assert(x_nf4->storage->element_size() == 1);

    auto x_back = x_nf4->cast(DataType::FP32);
    auto v_back = x_back->to_vector();

    assert(v_back[0] == 0.0f);
    assert(v_back[4] == -1.0f);
    assert(std::abs(v_back[1] - (-0.6961917f)) < 0.25f);

    std::cout << "SUCCESS: test_nf4_cast passed!" << std::endl;
}

void test_qlora_linear() {
    std::cout << "Running test_qlora_linear..." << std::endl;

    nn::QLoRALinear linear(4, 2, 2, 16.0f, true);
    auto input = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4}, Device(DeviceType::CPU), true);

    auto out = linear.forward(input);
    assert(out->shape[0] == 1);
    assert(out->shape[1] == 2);

    auto loss = Ops::sum(out);
    loss->backward();

    assert(linear.lora_A->grad != nullptr);
    assert(linear.lora_B->grad != nullptr);

    std::cout << "SUCCESS: test_qlora_linear passed!" << std::endl;
}

void test_fused_linear_cross_entropy() {
    std::cout << "Running test_fused_linear_cross_entropy..." << std::endl;

    auto x = Tensor::from_vector({1.0f, 2.0f, 0.5f, -0.5f}, {2, 2}, Device(DeviceType::CPU), true);
    auto w = Tensor::from_vector({0.1f, 0.2f, -0.3f, 0.4f}, {2, 2}, Device(DeviceType::CPU), true);
    auto target = Tensor::from_vector({0.0f, 1.0f}, {2}, Device(DeviceType::CPU));

    auto fused_loss = Ops::fused_linear_cross_entropy(x, w, target);
    fused_loss->backward();

    auto val_fused = fused_loss->to_vector()[0];
    auto gx_fused = x->grad->to_vector();

    x->grad = nullptr;
    w->grad = nullptr;

    auto logits = Ops::matmul(x, w->transpose(0, 1));
    auto normal_loss = Ops::cross_entropy_loss(logits, target);
    normal_loss->backward();

    auto val_normal = normal_loss->to_vector()[0];
    auto gx_normal = x->grad->to_vector();

    assert(std::abs(val_fused - val_normal) < 1e-5f);
    for (size_t i = 0; i < gx_fused.size(); ++i) {
        assert(std::abs(gx_fused[i] - gx_normal[i]) < 1e-4f);
    }

    std::cout << "SUCCESS: test_fused_linear_cross_entropy passed!" << std::endl;
}

int main() {
    test_prefix_cache();
    test_speculative_decoding();
    test_flash_decoding();
    test_nf4_cast();
    test_qlora_linear();
    test_fused_linear_cross_entropy();
    std::cout << "ALL TESTS PASSED!" << std::endl;
    return 0;
}
