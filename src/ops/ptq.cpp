#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/quantization.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <functional>

namespace litetorch {
namespace quantization {

class QuantizedLinear : public nn::Module {
public:
    std::shared_ptr<Tensor> q_weight;
    std::shared_ptr<Tensor> scales;
    std::shared_ptr<Tensor> bias;
    int bits = 8;

    QuantizedLinear(std::shared_ptr<Tensor> qw, std::shared_ptr<Tensor> sc,
                    std::shared_ptr<Tensor> b, int bits_ = 8)
        : q_weight(qw), scales(sc), bias(b), bits(bits_) {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        auto xc = input->device.type == DeviceType::CPU ? input : input->to(Device(DeviceType::CPU), DataType::FP32);
        if (xc->dtype != DataType::FP32) xc = xc->cast(DataType::FP32);
        auto w_fp32 = q_weight->cast(DataType::FP32);
        float* w_data = (float*)w_fp32->storage->get_cpu_ptr();
        float* s_data = (float*)scales->storage->get_cpu_ptr();
        int64_t N = q_weight->shape[0];
        int64_t K = q_weight->shape[1];
        for (int64_t i = 0; i < N; ++i)
            for (int64_t j = 0; j < K; ++j)
                w_data[i * K + j] *= s_data[i];
        auto out = Ops::matmul(xc, w_fp32->transpose(0, 1));
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
        scales = scales->to(dev);
        if (bias) bias = bias->to(dev);
    }
};

void PreparedModel::calibrate(std::function<std::shared_ptr<Tensor>()> data_fn, int num_batches) {
    model->eval();
    for (int i = 0; i < num_batches; ++i) {
        auto x = data_fn();
        auto out = model->forward(x);
        calibrator.collect("output", out);
        for (auto& [name, mod] : model->named_modules()) {
            if (auto lin = std::dynamic_pointer_cast<nn::Linear>(mod)) {
                calibrator.collect(name + ".weight", lin->weight);
            }
        }
    }
    model->train();
}

std::shared_ptr<PreparedModel> prepare(std::shared_ptr<nn::Module> model, PTQConfig config) {
    return std::make_shared<PreparedModel>(model, config);
}

namespace {
void convert_module(std::shared_ptr<nn::Module> parent, const PTQConfig& config) {
    for (auto& [name, child] : parent->named_children()) {
        if (auto lin = std::dynamic_pointer_cast<nn::Linear>(child)) {
            auto W = lin->weight->to(Device(DeviceType::CPU), DataType::FP32);
            int64_t N = W->shape[0];
            int64_t K = W->shape[1];
            float* w_data = (float*)W->storage->get_cpu_ptr();
            auto scales = Tensor::create({N}, Device(DeviceType::CPU), false, DataType::FP32);
            float* s_data = (float*)scales->storage->get_cpu_ptr();
            auto q_weight = Tensor::create({N, K}, Device(DeviceType::CPU), false, DataType::INT8);
            int8_t* q_data = (int8_t*)q_weight->storage->get_cpu_ptr();
            float bound = std::pow(2.0f, config.bits - 1) - 1.0f;
            for (int64_t i = 0; i < N; ++i) {
                float max_val = 0.0f;
                for (int64_t j = 0; j < K; ++j)
                    max_val = std::max(max_val, std::abs(w_data[i * K + j]));
                float scale = max_val > 0.0f ? max_val / bound : 1.0f;
                s_data[i] = scale;
                for (int64_t j = 0; j < K; ++j) {
                    float v = std::round(w_data[i * K + j] / scale);
                    v = std::max(-bound, std::min(bound, v));
                    q_data[i * K + j] = (int8_t)v;
                }
            }
            auto qlin = std::make_shared<QuantizedLinear>(q_weight, scales, lin->bias, config.bits);
            parent->add_module(name, qlin);
        } else {
            convert_module(child, config);
        }
    }
}
}

std::shared_ptr<nn::Module> convert(std::shared_ptr<PreparedModel> prepared) {
    convert_module(prepared->model, prepared->config);
    return prepared->model;
}

}
}
