#ifndef LITETORCH_QUANTIZATION_H
#define LITETORCH_QUANTIZATION_H

#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <string>

namespace litetorch {
namespace quantization {

inline void quantize_linear(std::shared_ptr<nn::Linear> layer) {
    auto W = layer->weight;
    W->storage->ensure_cpu();
    float* W_data = W->storage->get_cpu_ptr();
    
    int64_t out_features = W->shape[0];
    int64_t in_features = W->shape[1];
    
    auto scales = Tensor::create({out_features}, W->device);
    float* scales_data = scales->storage->get_cpu_ptr();
    
    auto q_weight = Tensor::create({out_features, in_features}, W->device);
    q_weight->dtype = DataType::INT8;
    q_weight->storage->dtype = DataType::INT8;
    int8_t* q_data = (int8_t*)q_weight->storage->get_cpu_ptr();
    
    for (int64_t i = 0; i < out_features; ++i) {
        float max_val = 0.0f;
        for (int64_t j = 0; j < in_features; ++j) {
            max_val = std::max(max_val, std::abs(W_data[i * in_features + j]));
        }
        float scale = max_val > 0.0f ? max_val / 127.0f : 1.0f;
        scales_data[i] = scale;
        
        for (int64_t j = 0; j < in_features; ++j) {
            q_data[i * in_features + j] = (int8_t)std::round(W_data[i * in_features + j] / scale);
        }
    }
    
    if (W->device.type == DeviceType::GPU) {
        q_weight->to(W->device);
        scales->to(W->device);
    }
    
    layer->weight = q_weight;
    layer->scales = scales;
}

class Calibrator {
public:
    std::unordered_map<std::string, float> min_vals;
    std::unordered_map<std::string, float> max_vals;

    void collect(const std::string& name, std::shared_ptr<Tensor> tensor) {
        tensor->storage->ensure_cpu();
        float* data = tensor->storage->get_cpu_ptr();
        size_t size = tensor->numel();
        float local_min = size > 0 ? data[0] : 0.0f;
        float local_max = size > 0 ? data[0] : 0.0f;
        for (size_t i = 0; i < size; ++i) {
            local_min = std::min(local_min, data[i]);
            local_max = std::max(local_max, data[i]);
        }
        if (min_vals.find(name) == min_vals.end()) {
            min_vals[name] = local_min;
            max_vals[name] = local_max;
        } else {
            min_vals[name] = std::min(min_vals[name], local_min);
            max_vals[name] = std::max(max_vals[name], local_max);
        }
    }

    std::pair<float, float> get_asymmetric_params(const std::string& name, int bits = 8) {
        if (min_vals.find(name) == min_vals.end()) return { 1.0f, 0.0f };
        float min_val = min_vals[name];
        float max_val = max_vals[name];
        float bound = std::pow(2.0f, bits - 1) - 1.0f;
        float qmin = -bound;
        float qmax = bound;
        float scale = (max_val - min_val) / (qmax - qmin);
        if (scale == 0.0f) scale = 1.0f;
        float zero_point = std::round((-min_val / scale) + qmin);
        if (zero_point < qmin) zero_point = qmin;
        if (zero_point > qmax) zero_point = qmax;
        return { scale, zero_point };
    }

    float get_scale(const std::string& name, int bits = 8) {
        if (max_vals.find(name) == max_vals.end()) return 1.0f;
        float max_val = std::max(std::abs(min_vals[name]), std::abs(max_vals[name]));
        float bound = std::pow(2.0f, bits - 1) - 1.0f;
        return max_val > 0.0f ? max_val / bound : 1.0f;
    }
};

class QATLinear : public nn::Module {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    float weight_scale = 1.0f;
    float weight_zero_point = 0.0f;
    float act_scale = 1.0f;
    float act_zero_point = 0.0f;
    int bits = 8;
    bool qat_enabled = true;

    QATLinear(std::shared_ptr<nn::Linear> linear, float w_scale = 0.01f, float w_zp = 0.0f, float a_scale = 0.01f, float a_zp = 0.0f, int b = 8)
        : weight(linear->weight), bias(linear->bias), weight_scale(w_scale), weight_zero_point(w_zp), act_scale(a_scale), act_zero_point(a_zp), bits(b) {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
        if (qat_enabled && training) {
            auto q_weight = Ops::fake_quantize(weight, weight_scale, weight_zero_point, bits);
            auto q_input = Ops::fake_quantize(input, act_scale, act_zero_point, bits);
            auto out = Ops::matmul(q_input, q_weight->transpose(0, 1));
            if (bias) {
                out = Ops::add(out, bias);
            }
            return out;
        } else {
            auto out = Ops::matmul(input, weight->transpose(0, 1));
            if (bias) {
                out = Ops::add(out, bias);
            }
            return out;
        }
    }

    std::vector<std::shared_ptr<Tensor>> parameters() override {
        if (bias) return { weight, bias };
        return { weight };
    }

    void to(const Device& dev) override {
        weight = weight->to(dev);
        if (bias) bias = bias->to(dev);
    }
};

}
}

#endif
