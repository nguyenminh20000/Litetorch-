#include "litetorch/nn.h"
#include "litetorch/tensor.h"
#include "litetorch/autograd.h"
#include <random>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace litetorch {
extern std::mt19937& get_generator();

namespace nn {
namespace init {

static void check_mode(const std::string& mode) {
    if (mode != "fan_in" && mode != "fan_out") {
        throw std::invalid_argument("[litetorch Error] init mode must be 'fan_in' or 'fan_out', got '" + mode + "'");
    }
}

static void calc_fan(const std::shared_ptr<Tensor>& t, int64_t& fan_in, int64_t& fan_out) {
    const auto& s = t->shape;
    if (s.size() < 2) {
        throw std::runtime_error("[litetorch Error] init requires tensor with at least 2 dimensions");
    }
    int64_t receptive_field = 1;
    for (size_t i = 2; i < s.size(); ++i) receptive_field *= s[i];
    fan_in = s[1] * receptive_field;
    fan_out = s[0] * receptive_field;
}

static float calc_gain(const std::string& nonlinearity, float a) {
    if (nonlinearity == "relu" || nonlinearity == "leaky_relu") {
        return std::sqrt(2.0f / (1.0f + a * a));
    }
    return 1.0f;
}

static void write_values(std::shared_ptr<Tensor> t, const std::vector<float>& vals) {
    NoGradGuard guard;
    auto tmp = Tensor::from_vector(vals, t->shape, Device(DeviceType::CPU, 0), false, DataType::FP32);
    t->copy_(tmp);
}

std::shared_ptr<Tensor> kaiming_uniform_(std::shared_ptr<Tensor> tensor, float a, const std::string& mode, const std::string& nonlinearity) {
    check_mode(mode);
    int64_t fan_in, fan_out;
    calc_fan(tensor, fan_in, fan_out);
    int64_t fan = (mode == "fan_out") ? fan_out : fan_in;
    float gain = calc_gain(nonlinearity, a);
    float std = gain / std::sqrt(static_cast<float>(fan));
    float bound = std::sqrt(3.0f) * std;
    std::mt19937& gen = get_generator();
    std::uniform_real_distribution<float> dis(-bound, bound);
    std::vector<float> vals(tensor->numel());
    for (size_t i = 0; i < vals.size(); ++i) vals[i] = dis(gen);
    write_values(tensor, vals);
    return tensor;
}

std::shared_ptr<Tensor> kaiming_normal_(std::shared_ptr<Tensor> tensor, float a, const std::string& mode, const std::string& nonlinearity) {
    check_mode(mode);
    int64_t fan_in, fan_out;
    calc_fan(tensor, fan_in, fan_out);
    int64_t fan = (mode == "fan_out") ? fan_out : fan_in;
    float gain = calc_gain(nonlinearity, a);
    float std = gain / std::sqrt(static_cast<float>(fan));
    std::mt19937& gen = get_generator();
    std::normal_distribution<float> dis(0.0f, std);
    std::vector<float> vals(tensor->numel());
    for (size_t i = 0; i < vals.size(); ++i) vals[i] = dis(gen);
    write_values(tensor, vals);
    return tensor;
}

std::shared_ptr<Tensor> xavier_uniform_(std::shared_ptr<Tensor> tensor, float gain) {
    int64_t fan_in, fan_out;
    calc_fan(tensor, fan_in, fan_out);
    float bound = gain * std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::mt19937& gen = get_generator();
    std::uniform_real_distribution<float> dis(-bound, bound);
    std::vector<float> vals(tensor->numel());
    for (size_t i = 0; i < vals.size(); ++i) vals[i] = dis(gen);
    write_values(tensor, vals);
    return tensor;
}

std::shared_ptr<Tensor> xavier_normal_(std::shared_ptr<Tensor> tensor, float gain) {
    int64_t fan_in, fan_out;
    calc_fan(tensor, fan_in, fan_out);
    float std = gain * std::sqrt(2.0f / static_cast<float>(fan_in + fan_out));
    std::mt19937& gen = get_generator();
    std::normal_distribution<float> dis(0.0f, std);
    std::vector<float> vals(tensor->numel());
    for (size_t i = 0; i < vals.size(); ++i) vals[i] = dis(gen);
    write_values(tensor, vals);
    return tensor;
}

std::shared_ptr<Tensor> normal_(std::shared_ptr<Tensor> tensor, float mean, float std) {
    std::mt19937& gen = get_generator();
    std::normal_distribution<float> dis(mean, std);
    std::vector<float> vals(tensor->numel());
    for (size_t i = 0; i < vals.size(); ++i) vals[i] = dis(gen);
    write_values(tensor, vals);
    return tensor;
}

std::shared_ptr<Tensor> uniform_(std::shared_ptr<Tensor> tensor, float a, float b) {
    std::mt19937& gen = get_generator();
    std::uniform_real_distribution<float> dis(a, b);
    std::vector<float> vals(tensor->numel());
    for (size_t i = 0; i < vals.size(); ++i) vals[i] = dis(gen);
    write_values(tensor, vals);
    return tensor;
}

std::shared_ptr<Tensor> zeros_(std::shared_ptr<Tensor> tensor) {
    write_values(tensor, std::vector<float>(tensor->numel(), 0.0f));
    return tensor;
}

std::shared_ptr<Tensor> ones_(std::shared_ptr<Tensor> tensor) {
    write_values(tensor, std::vector<float>(tensor->numel(), 1.0f));
    return tensor;
}

std::shared_ptr<Tensor> fill_(std::shared_ptr<Tensor> tensor, float value) {
    write_values(tensor, std::vector<float>(tensor->numel(), value));
    return tensor;
}

// === W3-T4 ===
std::shared_ptr<Tensor> trunc_normal_(std::shared_ptr<Tensor> tensor, float mean, float std, float a, float b) {
    if (std <= 0.0f) throw std::invalid_argument("[litetorch Error] trunc_normal_ std must be positive");
    if (a >= b) throw std::invalid_argument("[litetorch Error] trunc_normal_ requires a < b");
    std::mt19937& gen = get_generator();
    std::normal_distribution<float> dis(mean, std);
    std::vector<float> vals(tensor->numel());
    for (size_t i = 0; i < vals.size(); ++i) {
        float v;
        do { v = dis(gen); } while (v < a || v > b);
        vals[i] = v;
    }
    write_values(tensor, vals);
    return tensor;
}

std::shared_ptr<Tensor> orthogonal_(std::shared_ptr<Tensor> tensor, float gain) {
    const auto& s = tensor->shape;
    if (s.size() < 2) throw std::runtime_error("[litetorch Error] orthogonal_ requires tensor with at least 2 dimensions");
    int64_t rows = s[0], cols = 1;
    for (size_t i = 1; i < s.size(); ++i) cols *= s[i];
    bool transposed = rows < cols;
    int64_t m = transposed ? cols : rows;
    int64_t n = transposed ? rows : cols;
    std::mt19937& gen = get_generator();
    std::normal_distribution<float> dis(0.0f, 1.0f);
    std::vector<float> a(m * n);
    for (auto& v : a) v = dis(gen);
    std::vector<float> q(m * n, 0.0f);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < i; ++j) {
            float dot = 0.0f;
            for (int64_t k = 0; k < m; ++k) dot += q[k * n + j] * a[k * n + i];
            for (int64_t k = 0; k < m; ++k) a[k * n + i] -= dot * q[k * n + j];
        }
        float norm = 0.0f;
        for (int64_t k = 0; k < m; ++k) norm += a[k * n + i] * a[k * n + i];
        norm = std::sqrt(norm);
        if (norm < 1e-10f) throw std::runtime_error("[litetorch Error] orthogonal_ QR failed (degenerate random matrix)");
        for (int64_t k = 0; k < m; ++k) q[k * n + i] = a[k * n + i] / norm;
    }
    std::vector<float> vals(rows * cols);
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j)
            vals[i * cols + j] = gain * (transposed ? q[j * rows + i] : q[i * n + j]);
    write_values(tensor, vals);
    return tensor;
}
// === end W3-T4 ===

}
}
}
