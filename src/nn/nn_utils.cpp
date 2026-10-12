#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include <stdexcept>
#include <cmath>
#include <random>

namespace litetorch {
namespace nn {
namespace utils {

void clip_grad_value_(const std::vector<std::shared_ptr<Tensor>>& parameters, float clip_value) {
    if (clip_value < 0.0f) throw std::invalid_argument("[litetorch Error] clip_grad_value_ clip_value must be non-negative");
    NoGradGuard guard;
    for (const auto& p : parameters) {
        if (!p || !p->grad) continue;
        p->grad = Ops::clamp(p->grad, -clip_value, clip_value);
    }
}

std::shared_ptr<Tensor> parameters_to_vector(const std::vector<std::shared_ptr<Tensor>>& parameters) {
    if (parameters.empty()) throw std::invalid_argument("[litetorch Error] parameters_to_vector requires at least one parameter");
    NoGradGuard guard;
    std::vector<std::shared_ptr<Tensor>> flats;
    flats.reserve(parameters.size());
    Device dev = parameters[0]->device;
    for (const auto& p : parameters) {
        if (!p) throw std::invalid_argument("[litetorch Error] parameters_to_vector got null parameter");
        flats.push_back(p->reshape({static_cast<int64_t>(p->numel())})->to(dev));
    }
    return Ops::cat(flats, 0);
}

void vector_to_parameters(std::shared_ptr<Tensor> vec, const std::vector<std::shared_ptr<Tensor>>& parameters) {
    NoGradGuard guard;
    auto flat = vec->reshape({static_cast<int64_t>(vec->numel())});
    size_t offset = 0;
    for (const auto& p : parameters) {
        if (!p) throw std::invalid_argument("[litetorch Error] vector_to_parameters got null parameter");
        size_t n = p->numel();
        if (offset + n > flat->numel()) throw std::runtime_error("[litetorch Error] vector_to_parameters vector too short");
        auto piece = Ops::narrow(flat, 0, static_cast<int64_t>(offset), static_cast<int64_t>(n));
        p->copy_(piece->reshape(p->shape));
        offset += n;
    }
    if (offset != flat->numel()) throw std::runtime_error("[litetorch Error] vector_to_parameters vector too long");
}

std::shared_ptr<Tensor> weight_norm(std::shared_ptr<Tensor> v, std::shared_ptr<Tensor> g, int64_t dim) {
    if (dim != 0) throw std::runtime_error("[litetorch Error] weight_norm only supports dim=0 in this version");
    auto v_cpu = v->to(Device(DeviceType::CPU, 0));
    auto g_cpu = g->to(Device(DeviceType::CPU, 0));
    std::vector<float> vv = v_cpu->to_vector();
    std::vector<float> gv = g_cpu->to_vector();
    int64_t rows = v->shape[0];
    int64_t cols = static_cast<int64_t>(v->numel()) / rows;
    if ((int64_t)gv.size() != rows) throw std::runtime_error("[litetorch Error] weight_norm g must have one value per row (dim=0)");
    std::vector<float> out(v->numel());
    for (int64_t i = 0; i < rows; ++i) {
        float norm = 0.0f;
        for (int64_t j = 0; j < cols; ++j) norm += vv[i * cols + j] * vv[i * cols + j];
        norm = std::sqrt(norm);
        if (norm < 1e-12f) throw std::runtime_error("[litetorch Error] weight_norm zero-norm row");
        float scale = gv[i] / norm;
        for (int64_t j = 0; j < cols; ++j) out[i * cols + j] = vv[i * cols + j] * scale;
    }
    auto w = Tensor::from_vector(out, v->shape, v->device, false, v->dtype);
    return w;
}

std::shared_ptr<Tensor> spectral_norm(std::shared_ptr<Tensor> weight, int n_power_iterations) {
    if (weight->shape.size() != 2) throw std::runtime_error("[litetorch Error] spectral_norm requires a 2D weight in this version");
    if (n_power_iterations < 1) throw std::invalid_argument("[litetorch Error] spectral_norm n_power_iterations must be >= 1");
    auto w_cpu = weight->to(Device(DeviceType::CPU, 0));
    std::vector<float> w = w_cpu->to_vector();
    int64_t rows = weight->shape[0], cols = weight->shape[1];
    std::vector<float> u(rows, 0.0f), v(cols, 0.0f);
    {
        std::mt19937 gen(0);
        std::normal_distribution<float> dis(0.0f, 1.0f);
        for (auto& x : u) x = dis(gen);
        float nrm = 0.0f;
        for (float x : u) nrm += x * x;
        nrm = std::sqrt(nrm);
        for (auto& x : u) x /= nrm;
    }
    for (int it = 0; it < n_power_iterations; ++it) {
        for (int64_t j = 0; j < cols; ++j) {
            float s = 0.0f;
            for (int64_t i = 0; i < rows; ++i) s += w[i * cols + j] * u[i];
            v[j] = s;
        }
        float nrm = 0.0f;
        for (float x : v) nrm += x * x;
        nrm = std::sqrt(nrm) + 1e-12f;
        for (auto& x : v) x /= nrm;
        for (int64_t i = 0; i < rows; ++i) {
            float s = 0.0f;
            for (int64_t j = 0; j < cols; ++j) s += w[i * cols + j] * v[j];
            u[i] = s;
        }
        nrm = 0.0f;
        for (float x : u) nrm += x * x;
        nrm = std::sqrt(nrm) + 1e-12f;
        for (auto& x : u) x /= nrm;
    }
    float sigma = 0.0f;
    for (int64_t i = 0; i < rows; ++i) {
        float wv = 0.0f;
        for (int64_t j = 0; j < cols; ++j) wv += w[i * cols + j] * v[j];
        sigma += u[i] * wv;
    }
    if (sigma < 1e-12f) throw std::runtime_error("[litetorch Error] spectral_norm zero spectral norm");
    std::vector<float> out(w.size());
    for (size_t i = 0; i < w.size(); ++i) out[i] = w[i] / sigma;
    return Tensor::from_vector(out, weight->shape, weight->device, false, weight->dtype);
}

}
}
}
