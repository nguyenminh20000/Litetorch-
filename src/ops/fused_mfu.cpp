#include "litetorch/ops.h"
#include "litetorch/tensor.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include <cmath>
#include <tuple>
#include <vector>
#include <stdexcept>

namespace litetorch {
namespace Ops {

class FusedSwiGLUNode : public Node {
public:
    DataType gate_dtype = DataType::FP32;
    DataType up_dtype = DataType::FP32;
    FusedSwiGLUNode() : Node("FusedSwiGLU") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto gate = saved_tensors[0];
        auto up = saved_tensors[1];
        auto go = (grad_output->dtype == DataType::FP32) ? grad_output : cast(grad_output, DataType::FP32);
        auto s = sigmoid(gate);
        auto one = Tensor::ones(gate->shape, gate->device, false, DataType::FP32);
        auto dsilu = mul(s, add(one, mul(gate, sub(one, s))));
        auto silu_g = mul(gate, s);
        auto grad_gate = mul(mul(go, up), dsilu);
        auto grad_up = mul(go, silu_g);
        if (gate_dtype != DataType::FP32) grad_gate = cast(grad_gate, gate_dtype);
        if (up_dtype != DataType::FP32) grad_up = cast(grad_up, up_dtype);
        return { grad_gate, grad_up };
    }
};

std::shared_ptr<Tensor> fused_swiglu(std::shared_ptr<Tensor> gate, std::shared_ptr<Tensor> up) {
    if (gate->device.type == DeviceType::GPU || up->device.type == DeviceType::GPU) {
        throw std::runtime_error("[litetorch Error] fused_swiglu: GPU kernel not implemented (fail-loud)");
    }
    if (gate->shape != up->shape) {
        throw std::runtime_error("[litetorch Error] fused_swiglu: gate and up must have identical shape");
    }
    if (gate->device != up->device) {
        throw std::runtime_error("[litetorch Error] fused_swiglu: gate and up must be on the same device");
    }
    DataType out_dtype = promote_types(gate->dtype, up->dtype);
    auto gc = gate->is_contiguous() ? gate : gate->contiguous();
    auto uc = up->is_contiguous() ? up : up->contiguous();
    auto gf = (gc->dtype == DataType::FP32) ? gc : cast(gc, DataType::FP32);
    auto uf = (uc->dtype == DataType::FP32) ? uc : cast(uc, DataType::FP32);
    auto gfc = gf->is_contiguous() ? gf : gf->contiguous();
    auto ufc = uf->is_contiguous() ? uf : uf->contiguous();
    std::vector<float> gdata = gfc->to_vector();
    std::vector<float> udata = ufc->to_vector();
    size_t n = gdata.size();
    std::vector<float> out_data(n);
    ThreadPool::get().parallel_for(0, (int64_t)n, [&](int64_t i) {
        size_t j = (size_t)i;
        float g = gdata[j];
        float s = 1.0f / (1.0f + std::exp(-g));
        out_data[j] = (g * s) * udata[j];
    });
    auto out = Tensor::from_vector(out_data, gc->shape, gc->device, false, DataType::FP32);
    if (out_dtype != DataType::FP32) out = cast(out, out_dtype);
    if (Autograd::is_grad_enabled() && (gate->requires_grad || up->requires_grad)) {
        auto node = std::make_shared<FusedSwiGLUNode>();
        node->gate_dtype = gate->dtype;
        node->up_dtype = up->dtype;
        node->inputs = { {gate, gate->requires_grad}, {up, up->requires_grad} };
        node->next_nodes = { gate->creator, up->creator };
        node->saved_tensors = { gfc, ufc };
        node->saved_versions = {(gfc ? gfc->_version() : 0), (ufc ? ufc->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> fused_qkv_proj(
    std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> wq,
    std::shared_ptr<Tensor> wk, std::shared_ptr<Tensor> wv) {
    for (auto w : {wq, wk, wv}) {
        if (w->device.type == DeviceType::GPU) {
            throw std::runtime_error("[litetorch Error] fused_qkv_proj: GPU path not verified (fail-loud)");
        }
        if (w->shape.size() != 2) {
            throw std::runtime_error("[litetorch Error] fused_qkv_proj: weights must be 2D [out, in]");
        }
    }
    int64_t d_in = wq->shape[1];
    if (wk->shape[1] != d_in || wv->shape[1] != d_in) {
        throw std::runtime_error("[litetorch Error] fused_qkv_proj: weight in_features mismatch");
    }
    if (x->shape.back() != d_in) {
        throw std::runtime_error("[litetorch Error] fused_qkv_proj: x last dim must match weight in_features");
    }
    int64_t dq = wq->shape[0];
    int64_t dk = wk->shape[0];
    int64_t dv = wv->shape[0];
    auto w_cat = cat({wq, wk, wv}, 0);
    auto w_t = w_cat->transpose(0, 1);
    auto qkv = matmul(x, w_t);
    auto parts = split(qkv, {dq, dk, dv}, -1);
    return { parts[0], parts[1], parts[2] };
}

}
}
