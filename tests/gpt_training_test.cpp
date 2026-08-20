#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/quantization.h"
#include "litetorch/jit.h"
#include "litetorch/distributed.h"
#include "litetorch/cl_backend.h"
#include "litetorch/amp.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

using namespace litetorch;

class SimpleGPTDecoder : public nn::Module {
public:
    std::shared_ptr<quantization::QATLinear> c_attn;
    std::shared_ptr<quantization::QATLinear> c_proj;
    std::shared_ptr<quantization::QATLinear> c_fc;
    std::shared_ptr<quantization::QATLinear> c_proj2;

    SimpleGPTDecoder(int embed_dim) {
        auto linear_attn = std::make_shared<nn::Linear>(embed_dim, embed_dim * 3, false);
        c_attn = std::make_shared<quantization::QATLinear>(linear_attn, 0.01f, 0.01f, 8);
        c_attn->qat_enabled = true;

        auto linear_proj = std::make_shared<nn::Linear>(embed_dim, embed_dim, false);
        c_proj = std::make_shared<quantization::QATLinear>(linear_proj, 0.01f, 0.01f, 8);
        c_proj->qat_enabled = true;

        auto linear_fc = std::make_shared<nn::Linear>(embed_dim, embed_dim * 4, false);
        c_fc = std::make_shared<quantization::QATLinear>(linear_fc, 0.01f, 0.01f, 8);
        c_fc->qat_enabled = true;

        auto linear_proj2 = std::make_shared<nn::Linear>(embed_dim * 4, embed_dim, false);
        c_proj2 = std::make_shared<quantization::QATLinear>(linear_proj2, 0.01f, 0.01f, 8);
        c_proj2->qat_enabled = true;
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) {
        auto qkv = c_attn->forward(x);
        auto attn_out = c_proj->forward(x);
        auto h = c_fc->forward(attn_out);
        auto out = c_proj2->forward(h);
        return out;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() override {
        std::vector<std::shared_ptr<Tensor>> params;
        auto p1 = c_attn->parameters();
        params.insert(params.end(), p1.begin(), p1.end());
        auto p2 = c_proj->parameters();
        params.insert(params.end(), p2.begin(), p2.end());
        auto p3 = c_fc->parameters();
        params.insert(params.end(), p3.begin(), p3.end());
        auto p4 = c_proj2->parameters();
        params.insert(params.end(), p4.begin(), p4.end());
        return params;
    }
};

int main() {
    std::cout << "Starting LiteTorch GPT Training Integration Test..." << std::endl;

    auto model = std::make_shared<SimpleGPTDecoder>(8);
    auto input = Tensor::from_vector({
        1.0f, -0.5f, 0.2f, 0.8f, -0.1f, 0.4f, -0.9f, 0.3f,
        -0.2f, 0.5f, 0.6f, -0.7f, 0.1f, -0.3f, 0.8f, 0.2f
    }, {2, 8});
    input->requires_grad = true;

    {
        amp::AutocastGuard autocast_guard(true, DataType::FP16);

        auto output = model->forward(input);
        assert(output->shape[0] == 2);
        assert(output->shape[1] == 8);

        auto loss = Ops::sum(output);
        loss->backward();

        assert(input->grad != nullptr);
        std::cout << "AMP + QAT Forward & Backward completed successfully." << std::endl;
    }

    auto x_var = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
    auto y_var = std::make_shared<JITVar>(JITVar::OpType::INPUT, "y");
    auto jit_expr = JIT::gelu(x_var * y_var);

    auto forward_fn = std::make_shared<JITFunction>("gpt_fwd", jit_expr, std::vector<std::shared_ptr<JITVar>>{x_var, y_var});
    forward_fn->save("gpt_jit_model.txt");

    auto loaded_forward_fn = JITFunction::load("gpt_jit_model.txt");

    auto t1 = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4});
    auto t2 = Tensor::from_vector({0.5f, 0.5f, 0.5f, 0.5f}, {4});

    if (CLBackend::get().is_available()) {
        t1 = t1->to(Device(DeviceType::GPU));
        t2 = t2->to(Device(DeviceType::GPU));
    }

    auto out_jit = (*loaded_forward_fn)({t1, t2});
    std::vector<float> res_jit = out_jit->to_vector();
    assert(res_jit.size() == 4);

    std::cout << "JIT Tracing + Serialization verified in GPT training script." << std::endl;

    auto& pg = distributed::ProcessGroup::get();
    pg.init(0, 1, "127.0.0.1", 12347);
    
    auto t_dist = Tensor::from_vector({2.0f, 4.0f}, {2});
    pg.all_reduce(t_dist);
    std::vector<float> res_dist = t_dist->to_vector();
    assert(res_dist[0] == 2.0f);
    assert(res_dist[1] == 4.0f);
    pg.shutdown();

    std::cout << "Distributed Bridge fallback verified." << std::endl;
    std::cout << "LiteTorch GPT Training Integration Test Passed!" << std::endl;
    return 0;
}
