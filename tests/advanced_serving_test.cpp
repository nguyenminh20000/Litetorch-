#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/llm_serving.h"
#include "litetorch/continuous_batching.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <memory>
#include <cmath>

using namespace litetorch;
using namespace litetorch::nn;

class MockModel : public Module {
public:
    int vocab_size;
    MockModel(int vocab_size) : vocab_size(vocab_size) {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        int64_t batch_size = input->shape[0];
        int64_t seq_len = input->shape[1];
        auto logits = Tensor::create({batch_size, seq_len, static_cast<int64_t>(vocab_size)}, input->device, false);
        float* in_ptr = input->data_ptr();
        float* log_ptr = logits->data_ptr();
        std::fill(log_ptr, log_ptr + logits->numel(), 0.0f);
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                int64_t tok = static_cast<int64_t>(in_ptr[b * seq_len + s]);
                int64_t next_tok = (tok + 1) % vocab_size;
                log_ptr[b * seq_len * vocab_size + s * vocab_size + next_tok] = 10.0f;
            }
        }
        return logits;
    }
};

void test_continuous_batching() {
    std::cout << "Running test_continuous_batching..." << std::endl;

    auto model = std::make_shared<MockModel>(32);
    auto scheduler = std::make_shared<ServingScheduler>(10, 4, 2, 8);

    auto req1 = std::make_shared<Request>(1, std::vector<int64_t>{1, 2, 3, 4, 5, 6}, 4);
    auto req2 = std::make_shared<Request>(2, std::vector<int64_t>{10, 11}, 4);

    scheduler->add_request(req1);
    scheduler->add_request(req2);

    int steps = 0;
    while ((!scheduler->pending_queue.empty() || !scheduler->running_queue.empty()) && steps < 50) {
        auto completed = scheduler->step(model);
        for (auto& req : completed) {
            std::cout << "Request " << req->id << " completed. Tokens: ";
            for (int64_t t : req->generated_tokens) {
                std::cout << t << " ";
            }
            std::cout << std::endl;
        }
        steps++;
    }

    assert(req1->is_finished);
    assert(req2->is_finished);
    assert(req1->generated_tokens.size() == 4);
    assert(req2->generated_tokens.size() == 4);

    assert(req1->generated_tokens[0] == 7);
    assert(req1->generated_tokens[1] == 8);
    assert(req1->generated_tokens[2] == 9);
    assert(req1->generated_tokens[3] == 10);

    assert(req2->generated_tokens[0] == 12);
    assert(req2->generated_tokens[1] == 13);
    assert(req2->generated_tokens[2] == 14);
    assert(req2->generated_tokens[3] == 15);

    assert(scheduler->free_blocks.size() == 10);
    std::cout << "SUCCESS: test_continuous_batching passed!" << std::endl;
}

void test_mixture_of_experts() {
    std::cout << "Running test_mixture_of_experts..." << std::endl;

    auto moe = std::make_shared<MoELinear>(4, 4, 3, 2);
    auto input = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 4}, Device(DeviceType::CPU), true);

    auto out = moe->forward(input);
    assert(out->shape[0] == 2);
    assert(out->shape[1] == 4);

    auto loss = Ops::sum(out);
    loss->backward();

    assert(input->grad != nullptr);
    assert(moe->gate_weight->grad != nullptr);
    
    float* gw_grad = moe->gate_weight->grad->data_ptr();
    bool has_gate_grad = false;
    for (size_t i = 0; i < moe->gate_weight->numel(); ++i) {
        if (std::abs(gw_grad[i]) > 1e-5f) {
            has_gate_grad = true;
            break;
        }
    }
    assert(has_gate_grad);

    for (int e = 0; e < 3; ++e) {
        assert(moe->experts[e]->weight->grad != nullptr);
        if (moe->experts[e]->bias) {
            assert(moe->experts[e]->bias->grad != nullptr);
        }
    }

    std::cout << "SUCCESS: test_mixture_of_experts passed!" << std::endl;
}

void test_medusa() {
    std::cout << "Running test_medusa..." << std::endl;

    auto base = std::make_shared<MockModel>(32);
    std::vector<std::shared_ptr<MedusaHead>> heads;
    for (int h = 0; h < 2; ++h) {
        auto head = std::make_shared<MedusaHead>(32, 32);
        
        float* w_ptr = head->weight->data_ptr();
        std::fill(w_ptr, w_ptr + head->weight->numel(), 0.0f);
        for (int curr = 0; curr < 32; ++curr) {
            int next = (curr + h + 2) % 32;
            w_ptr[next * 32 + curr] = 10.0f;
        }
        heads.push_back(head);
    }

    auto engine = std::make_shared<MedusaEngine>(base, heads);
    std::vector<int64_t> prompt = {1, 2};
    auto tokens = engine->generate(prompt, 6, 2);

    std::cout << "Generated tokens: ";
    for (int64_t t : tokens) {
        std::cout << t << " ";
    }
    std::cout << std::endl;

    assert(tokens.size() >= 8);
    assert(tokens[2] == 3);
    assert(tokens[3] == 4);
    assert(tokens[4] == 5);
    
    std::cout << "SUCCESS: test_medusa passed!" << std::endl;
}

int main() {
    test_continuous_batching();
    test_mixture_of_experts();
    test_medusa();
    std::cout << "ALL ADVANCED SERVING TESTS PASSED!" << std::endl;
    return 0;
}
