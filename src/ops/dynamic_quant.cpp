#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/quantization.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace litetorch {
namespace quantization {

class DynamicQuantLinear : public nn::Module {
public:
    std::shared_ptr<Tensor> q_weight;
    std::shared_ptr<Tensor> w_scales;
    std::shared_ptr<Tensor> bias;

    DynamicQuantLinear(std::shared_ptr<nn::Linear> lin) {
        auto W = lin->weight->to(Device(DeviceType::CPU), DataType::FP32);
        int64_t N = W->shape[0];
        int64_t K = W->shape[1];
        float* w_data = (float*)W->storage->get_cpu_ptr();
        w_scales = Tensor::create({N}, Device(DeviceType::CPU), false, DataType::FP32);
        float* s_data = (float*)w_scales->storage->get_cpu_ptr();
        q_weight = Tensor::create({N, K}, Device(DeviceType::CPU), false, DataType::INT8);
        int8_t* q_data = (int8_t*)q_weight->storage->get_cpu_ptr();
        for (int64_t i = 0; i < N; ++i) {
            float max_val = 0.0f;
            for (int64_t j = 0; j < K; ++j)
                max_val = std::max(max_val, std::abs(w_data[i * K + j]));
            float scale = max_val > 0.0f ? max_val / 127.0f : 1.0f;
            s_data[i] = scale;
            for (int64_t j = 0; j < K; ++j) {
                float v = std::round(w_data[i * K + j] / scale);
                q_data[i * K + j] = (int8_t)std::max(-128.0f, std::min(127.0f, v));
            }
        }
        bias = lin->bias;
        if (lin->weight->device.type != DeviceType::CPU) to(lin->weight->device);
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        auto xc = input->device.type == DeviceType::CPU ? input : input->to(Device(DeviceType::CPU), DataType::FP32);
        if (xc->dtype != DataType::FP32) xc = xc->cast(DataType::FP32);
        xc = xc->is_contiguous() ? xc : xc->contiguous();
        float* x_data = (float*)xc->storage->get_cpu_ptr();
        size_t n = xc->numel();
        float x_max = 0.0f;
        for (size_t i = 0; i < n; ++i) x_max = std::max(x_max, std::abs(x_data[i]));
        float x_scale = x_max > 0.0f ? x_max / 127.0f : 1.0f;
        auto x_q = Tensor::create(xc->shape, xc->device, false, DataType::INT8);
        int8_t* xq_data = (int8_t*)x_q->storage->get_cpu_ptr();
        for (size_t i = 0; i < n; ++i) {
            float v = std::round(x_data[i] / x_scale);
            xq_data[i] = (int8_t)std::max(-128.0f, std::min(127.0f, v));
        }
        int64_t M = xc->shape[0];
        int64_t K = xc->shape[1];
        int64_t N = q_weight->shape[0];
        auto x_dq = x_q->cast(DataType::FP32);
        float* xdq = (float*)x_dq->storage->get_cpu_ptr();
        for (size_t i = 0; i < n; ++i) xdq[i] *= x_scale;
        auto w_dq = q_weight->cast(DataType::FP32);
        float* wdq = (float*)w_dq->storage->get_cpu_ptr();
        float* s_data = (float*)w_scales->storage->get_cpu_ptr();
        for (int64_t i = 0; i < N; ++i)
            for (int64_t j = 0; j < K; ++j)
                wdq[i * K + j] *= s_data[i];
        auto out = Ops::matmul(x_dq, w_dq->transpose(0, 1));
        if (bias) out = Ops::add(out, bias);
        if (input->device.type != DeviceType::CPU) out = out->to(input->device);
        return out;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() override {
        if (bias) return {bias};
        return {};
    }

    void to(const Device& dev) override {
        q_weight = q_weight->to(dev);
        w_scales = w_scales->to(dev);
        if (bias) bias = bias->to(dev);
    }
};

namespace {
void quantize_cell_weights(std::shared_ptr<nn::Module> cell) {
    if (auto lstm = std::dynamic_pointer_cast<nn::LSTMCell>(cell)) {
        for (auto w : {lstm->weight_ih, lstm->weight_hh}) {
            auto W = w->to(Device(DeviceType::CPU), DataType::FP32);
            float* data = (float*)W->storage->get_cpu_ptr();
            size_t n = W->numel();
            float max_val = 0.0f;
            for (size_t i = 0; i < n; ++i) max_val = std::max(max_val, std::abs(data[i]));
            float scale = max_val > 0.0f ? max_val / 127.0f : 1.0f;
            for (size_t i = 0; i < n; ++i) {
                float v = std::round(data[i] / scale) * scale;
                data[i] = v;
            }
        }
    } else if (auto gru = std::dynamic_pointer_cast<nn::GRUCell>(cell)) {
        for (auto w : {gru->weight_ih, gru->weight_hh}) {
            auto W = w->to(Device(DeviceType::CPU), DataType::FP32);
            float* data = (float*)W->storage->get_cpu_ptr();
            size_t n = W->numel();
            float max_val = 0.0f;
            for (size_t i = 0; i < n; ++i) max_val = std::max(max_val, std::abs(data[i]));
            float scale = max_val > 0.0f ? max_val / 127.0f : 1.0f;
            for (size_t i = 0; i < n; ++i) {
                float v = std::round(data[i] / scale) * scale;
                data[i] = v;
            }
        }
    }
}
}

std::shared_ptr<nn::Module> quantize_dynamic(std::shared_ptr<nn::Linear> linear) {
    return std::make_shared<DynamicQuantLinear>(linear);
}

std::shared_ptr<nn::Module> quantize_dynamic(std::shared_ptr<nn::LSTM> lstm) {
    for (auto& cell : lstm->cells) quantize_cell_weights(cell);
    return lstm;
}

std::shared_ptr<nn::Module> quantize_dynamic(std::shared_ptr<nn::GRU> gru) {
    for (auto& cell : gru->cells) quantize_cell_weights(cell);
    return gru;
}

}
}
