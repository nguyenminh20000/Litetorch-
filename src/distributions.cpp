#include "litetorch/distributions.h"
#include <random>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace litetorch {
std::mt19937& get_generator();
namespace distributions {

static const double PI = 3.14159265358979323846;
static const double EULER = 0.5772156649015328606;

static double digamma(double x) {
    double result = 0.0;
    while (x < 7.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    double r = 1.0 / x;
    result += std::log(x) - 0.5 * r;
    double r2 = r * r;
    result -= r2 * (1.0/12.0 - r2 * (1.0/120.0 - r2 * (1.0/252.0)));
    return result;
}

std::vector<int64_t> broadcast_shapes(const std::vector<std::shared_ptr<Tensor>>& params) {
    std::vector<int64_t> out;
    for (auto& p : params) {
        if (p->shape.size() > out.size()) out.resize(p->shape.size(), 1);
        for (size_t i = 0; i < p->shape.size(); i++) {
            size_t j = out.size() - p->shape.size() + i;
            if (out[j] == 1) out[j] = p->shape[i];
            else if (p->shape[i] != 1 && p->shape[i] != out[j])
                throw std::runtime_error("distributions: incompatible broadcast shapes");
        }
    }
    while (out.size() > 1 && out[0] == 1) out.erase(out.begin());
    if (out.empty()) out = {1};
    return out;
}

std::shared_ptr<Tensor> broadcast_param(std::shared_ptr<Tensor> p, const std::vector<int64_t>& shape) {
    if (p->shape == shape) return p;
    if (p->numel() == 1) {
        auto t = Tensor::create(shape, Device(DeviceType::CPU, 0), false, DataType::FP32);
        float v = p->data_ptr()[p->offset];
        float* d = t->data_ptr() + t->offset;
        for (size_t i = 0; i < t->numel(); i++) d[i] = v;
        return t;
    }
    throw std::runtime_error("distributions: parameter shape mismatch, only scalar broadcast supported");
}

static std::shared_ptr<Tensor> make_tensor(const std::vector<int64_t>& shape) {
    return Tensor::create(shape, Device(DeviceType::CPU, 0), false, DataType::FP32);
}

static std::vector<int64_t> full_shape(const std::vector<int64_t>& sample_shape, const std::vector<int64_t>& batch) {
    std::vector<int64_t> s = sample_shape;
    s.insert(s.end(), batch.begin(), batch.end());
    return s;
}

static size_t shape_numel(const std::vector<int64_t>& s) {
    size_t n = 1;
    for (auto d : s) n *= (size_t)d;
    return n;
}

Normal::Normal(std::shared_ptr<Tensor> l, std::shared_ptr<Tensor> s) {
    auto bs = broadcast_shapes({l, s});
    loc = broadcast_param(l, bs);
    scale = broadcast_param(s, bs);
    float* sd = scale->data_ptr() + scale->offset;
    for (size_t i = 0; i < scale->numel(); i++)
        if (sd[i] <= 0) throw std::runtime_error("Normal: scale must be positive");
}

std::vector<int64_t> Normal::batch_shape() const { return loc->shape; }

std::shared_ptr<Tensor> Normal::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float* od = out->data_ptr() + out->offset;
    float* ld = loc->data_ptr() + loc->offset;
    float* sd = scale->data_ptr() + scale->offset;
    size_t bn = loc->numel();
    size_t n = shape_numel(shape);
    for (size_t i = 0; i < n; i++)
        od[i] = ld[i % bn] + sd[i % bn] * nd(gen);
    return out;
}

std::shared_ptr<Tensor> Normal::log_prob(std::shared_ptr<Tensor> value) {
    auto bs = broadcast_shapes({loc, scale, value});
    auto l = broadcast_param(loc, bs);
    auto s = broadcast_param(scale, bs);
    auto v = broadcast_param(value, bs);
    auto out = make_tensor(bs);
    float* vd = v->data_ptr() + v->offset;
    float* ld = l->data_ptr() + l->offset;
    float* sd = s->data_ptr() + s->offset;
    float* od = out->data_ptr() + out->offset;
    size_t n = out->numel();
    for (size_t i = 0; i < n; i++) {
        double z = (vd[i] - ld[i]) / sd[i];
        od[i] = (float)(-0.5 * z * z - std::log(sd[i]) - 0.5 * std::log(2.0 * PI));
    }
    return out;
}

std::shared_ptr<Tensor> Normal::entropy() {
    auto out = make_tensor(batch_shape());
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++)
        od[i] = (float)(0.5 * (std::log(2.0 * PI) + 1.0) + std::log(sd[i]));
    return out;
}

std::shared_ptr<Tensor> Normal::mean() { return loc; }

std::shared_ptr<Tensor> Normal::variance() {
    auto out = make_tensor(batch_shape());
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = sd[i] * sd[i];
    return out;
}

Uniform::Uniform(std::shared_ptr<Tensor> l, std::shared_ptr<Tensor> h) {
    auto bs = broadcast_shapes({l, h});
    low = broadcast_param(l, bs);
    high = broadcast_param(h, bs);
    float* ld = low->data_ptr() + low->offset;
    float* hd = high->data_ptr() + high->offset;
    for (size_t i = 0; i < low->numel(); i++)
        if (hd[i] <= ld[i]) throw std::runtime_error("Uniform: high must be greater than low");
}

std::vector<int64_t> Uniform::batch_shape() const { return low->shape; }

std::shared_ptr<Tensor> Uniform::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    float* od = out->data_ptr() + out->offset;
    float* ld = low->data_ptr() + low->offset;
    float* hd = high->data_ptr() + high->offset;
    size_t bn = low->numel();
    for (size_t i = 0; i < shape_numel(shape); i++)
        od[i] = ld[i % bn] + (hd[i % bn] - ld[i % bn]) * ud(gen);
    return out;
}

std::shared_ptr<Tensor> Uniform::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* ld = low->data_ptr() + low->offset;
    float* hd = high->data_ptr() + high->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++)
        od[i] = (vd[i] >= ld[i] && vd[i] <= hd[i]) ? (float)(-std::log(hd[i] - ld[i])) : -INFINITY;
    return out;
}

std::shared_ptr<Tensor> Uniform::entropy() {
    auto out = make_tensor(batch_shape());
    float* ld = low->data_ptr() + low->offset;
    float* hd = high->data_ptr() + high->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = (float)std::log(hd[i] - ld[i]);
    return out;
}

std::shared_ptr<Tensor> Uniform::mean() {
    auto out = make_tensor(batch_shape());
    float* ld = low->data_ptr() + low->offset;
    float* hd = high->data_ptr() + high->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = 0.5f * (ld[i] + hd[i]);
    return out;
}

std::shared_ptr<Tensor> Uniform::variance() {
    auto out = make_tensor(batch_shape());
    float* ld = low->data_ptr() + low->offset;
    float* hd = high->data_ptr() + high->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double w = hd[i] - ld[i];
        od[i] = (float)(w * w / 12.0);
    }
    return out;
}

Exponential::Exponential(std::shared_ptr<Tensor> r) {
    auto bs = broadcast_shapes({r});
    rate = broadcast_param(r, bs);
    float* rd = rate->data_ptr() + rate->offset;
    for (size_t i = 0; i < rate->numel(); i++)
        if (rd[i] <= 0) throw std::runtime_error("Exponential: rate must be positive");
}

std::vector<int64_t> Exponential::batch_shape() const { return rate->shape; }

std::shared_ptr<Tensor> Exponential::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    std::exponential_distribution<float> ed(1.0f);
    float* od = out->data_ptr() + out->offset;
    float* rd = rate->data_ptr() + rate->offset;
    size_t bn = rate->numel();
    for (size_t i = 0; i < shape_numel(shape); i++)
        od[i] = ed(gen) / rd[i % bn];
    return out;
}

std::shared_ptr<Tensor> Exponential::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* rd = rate->data_ptr() + rate->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++)
        od[i] = (vd[i] >= 0) ? (float)(std::log(rd[i]) - rd[i] * vd[i]) : -INFINITY;
    return out;
}

std::shared_ptr<Tensor> Exponential::entropy() {
    auto out = make_tensor(batch_shape());
    float* rd = rate->data_ptr() + rate->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = (float)(1.0 - std::log(rd[i]));
    return out;
}

std::shared_ptr<Tensor> Exponential::mean() {
    auto out = make_tensor(batch_shape());
    float* rd = rate->data_ptr() + rate->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = 1.0f / rd[i];
    return out;
}

std::shared_ptr<Tensor> Exponential::variance() {
    auto out = make_tensor(batch_shape());
    float* rd = rate->data_ptr() + rate->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = 1.0f / (rd[i] * rd[i]);
    return out;
}

Laplace::Laplace(std::shared_ptr<Tensor> l, std::shared_ptr<Tensor> s) {
    auto bs = broadcast_shapes({l, s});
    loc = broadcast_param(l, bs);
    scale = broadcast_param(s, bs);
    float* sd = scale->data_ptr() + scale->offset;
    for (size_t i = 0; i < scale->numel(); i++)
        if (sd[i] <= 0) throw std::runtime_error("Laplace: scale must be positive");
}

std::vector<int64_t> Laplace::batch_shape() const { return loc->shape; }

std::shared_ptr<Tensor> Laplace::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    std::uniform_real_distribution<float> ud(-0.5f, 0.5f);
    float* od = out->data_ptr() + out->offset;
    float* ld = loc->data_ptr() + loc->offset;
    float* sd = scale->data_ptr() + scale->offset;
    size_t bn = loc->numel();
    for (size_t i = 0; i < shape_numel(shape); i++) {
        double u = ud(gen);
        double sgn = (u < 0) ? 1.0 : -1.0;
        od[i] = (float)(ld[i % bn] - sd[i % bn] * sgn * std::log(1.0 - 2.0 * std::abs(u)));
    }
    return out;
}

std::shared_ptr<Tensor> Laplace::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* ld = loc->data_ptr() + loc->offset;
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++)
        od[i] = (float)(-std::log(2.0 * sd[i]) - std::abs(vd[i] - ld[i]) / sd[i]);
    return out;
}

std::shared_ptr<Tensor> Laplace::entropy() {
    auto out = make_tensor(batch_shape());
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++)
        od[i] = (float)(1.0 + std::log(2.0 * sd[i]));
    return out;
}

std::shared_ptr<Tensor> Laplace::mean() { return loc; }

std::shared_ptr<Tensor> Laplace::variance() {
    auto out = make_tensor(batch_shape());
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = 2.0f * sd[i] * sd[i];
    return out;
}

Cauchy::Cauchy(std::shared_ptr<Tensor> l, std::shared_ptr<Tensor> s) {
    auto bs = broadcast_shapes({l, s});
    loc = broadcast_param(l, bs);
    scale = broadcast_param(s, bs);
    float* sd = scale->data_ptr() + scale->offset;
    for (size_t i = 0; i < scale->numel(); i++)
        if (sd[i] <= 0) throw std::runtime_error("Cauchy: scale must be positive");
}

std::vector<int64_t> Cauchy::batch_shape() const { return loc->shape; }

std::shared_ptr<Tensor> Cauchy::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    std::cauchy_distribution<float> cd(0.0f, 1.0f);
    float* od = out->data_ptr() + out->offset;
    float* ld = loc->data_ptr() + loc->offset;
    float* sd = scale->data_ptr() + scale->offset;
    size_t bn = loc->numel();
    for (size_t i = 0; i < shape_numel(shape); i++)
        od[i] = ld[i % bn] + sd[i % bn] * cd(gen);
    return out;
}

std::shared_ptr<Tensor> Cauchy::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* ld = loc->data_ptr() + loc->offset;
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double z = (vd[i] - ld[i]) / sd[i];
        od[i] = (float)(-std::log(PI * sd[i] * (1.0 + z * z)));
    }
    return out;
}

std::shared_ptr<Tensor> Cauchy::entropy() {
    auto out = make_tensor(batch_shape());
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++)
        od[i] = (float)std::log(4.0 * PI * sd[i]);
    return out;
}

std::shared_ptr<Tensor> Cauchy::mean() {
    auto out = make_tensor(batch_shape());
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = std::numeric_limits<float>::quiet_NaN();
    return out;
}

std::shared_ptr<Tensor> Cauchy::variance() {
    auto out = make_tensor(batch_shape());
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = std::numeric_limits<float>::infinity();
    return out;
}

Gamma::Gamma(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    auto bs = broadcast_shapes({a, b});
    alpha = broadcast_param(a, bs);
    beta = broadcast_param(b, bs);
    float* ad = alpha->data_ptr() + alpha->offset;
    float* bd = beta->data_ptr() + beta->offset;
    for (size_t i = 0; i < alpha->numel(); i++)
        if (ad[i] <= 0 || bd[i] <= 0) throw std::runtime_error("Gamma: alpha and beta must be positive");
}

std::vector<int64_t> Gamma::batch_shape() const { return alpha->shape; }

std::shared_ptr<Tensor> Gamma::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    float* od = out->data_ptr() + out->offset;
    float* ad = alpha->data_ptr() + alpha->offset;
    float* bd = beta->data_ptr() + beta->offset;
    size_t bn = alpha->numel();
    for (size_t i = 0; i < shape_numel(shape); i++) {
        std::gamma_distribution<float> gd(ad[i % bn], 1.0f);
        od[i] = gd(gen) / bd[i % bn];
    }
    return out;
}

std::shared_ptr<Tensor> Gamma::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* ad = alpha->data_ptr() + alpha->offset;
    float* bd = beta->data_ptr() + beta->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        if (vd[i] <= 0) { od[i] = -INFINITY; continue; }
        od[i] = (float)(ad[i] * std::log(bd[i]) + (ad[i] - 1.0) * std::log(vd[i]) - bd[i] * vd[i] - std::lgamma(ad[i]));
    }
    return out;
}

std::shared_ptr<Tensor> Gamma::entropy() {
    auto out = make_tensor(batch_shape());
    float* ad = alpha->data_ptr() + alpha->offset;
    float* bd = beta->data_ptr() + beta->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++)
        od[i] = (float)(ad[i] - std::log(bd[i]) + std::lgamma(ad[i]) + (1.0 - ad[i]) * digamma(ad[i]));
    return out;
}

std::shared_ptr<Tensor> Gamma::mean() {
    auto out = make_tensor(batch_shape());
    float* ad = alpha->data_ptr() + alpha->offset;
    float* bd = beta->data_ptr() + beta->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = ad[i] / bd[i];
    return out;
}

std::shared_ptr<Tensor> Gamma::variance() {
    auto out = make_tensor(batch_shape());
    float* ad = alpha->data_ptr() + alpha->offset;
    float* bd = beta->data_ptr() + beta->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = ad[i] / (bd[i] * bd[i]);
    return out;
}

Beta::Beta(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    auto bs = broadcast_shapes({a, b});
    alpha = broadcast_param(a, bs);
    beta = broadcast_param(b, bs);
    float* ad = alpha->data_ptr() + alpha->offset;
    float* bd = beta->data_ptr() + beta->offset;
    for (size_t i = 0; i < alpha->numel(); i++)
        if (ad[i] <= 0 || bd[i] <= 0) throw std::runtime_error("Beta: alpha and beta must be positive");
}

std::vector<int64_t> Beta::batch_shape() const { return alpha->shape; }

std::shared_ptr<Tensor> Beta::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    float* od = out->data_ptr() + out->offset;
    float* ad = alpha->data_ptr() + alpha->offset;
    float* bd = beta->data_ptr() + beta->offset;
    size_t bn = alpha->numel();
    for (size_t i = 0; i < shape_numel(shape); i++) {
        std::gamma_distribution<float> ga(ad[i % bn], 1.0f);
        std::gamma_distribution<float> gb(bd[i % bn], 1.0f);
        float x = ga(gen), y = gb(gen);
        od[i] = x / (x + y);
    }
    return out;
}

std::shared_ptr<Tensor> Beta::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* ad = alpha->data_ptr() + alpha->offset;
    float* bd = beta->data_ptr() + beta->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        if (vd[i] <= 0 || vd[i] >= 1) { od[i] = -INFINITY; continue; }
        od[i] = (float)((ad[i] - 1.0) * std::log(vd[i]) + (bd[i] - 1.0) * std::log(1.0 - vd[i])
            - std::lgamma(ad[i]) - std::lgamma(bd[i]) + std::lgamma(ad[i] + bd[i]));
    }
    return out;
}

std::shared_ptr<Tensor> Beta::entropy() {
    auto out = make_tensor(batch_shape());
    float* ad = alpha->data_ptr() + alpha->offset;
    float* bd = beta->data_ptr() + beta->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double a = ad[i], b = bd[i];
        od[i] = (float)(std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b)
            - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b) + (a + b - 2.0) * digamma(a + b));
    }
    return out;
}

std::shared_ptr<Tensor> Beta::mean() {
    auto out = make_tensor(batch_shape());
    float* ad = alpha->data_ptr() + alpha->offset;
    float* bd = beta->data_ptr() + beta->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = ad[i] / (ad[i] + bd[i]);
    return out;
}

std::shared_ptr<Tensor> Beta::variance() {
    auto out = make_tensor(batch_shape());
    float* ad = alpha->data_ptr() + alpha->offset;
    float* bd = beta->data_ptr() + beta->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double s = ad[i] + bd[i];
        od[i] = (float)(ad[i] * bd[i] / (s * s * (s + 1.0)));
    }
    return out;
}

Dirichlet::Dirichlet(std::shared_ptr<Tensor> c) {
    concentration = c;
    float* cd = concentration->data_ptr() + concentration->offset;
    for (size_t i = 0; i < concentration->numel(); i++)
        if (cd[i] <= 0) throw std::runtime_error("Dirichlet: concentration must be positive");
    if (concentration->shape.empty()) throw std::runtime_error("Dirichlet: concentration must have event dim");
}

std::vector<int64_t> Dirichlet::batch_shape() const {
    auto s = concentration->shape;
    s.pop_back();
    if (s.empty()) return {1};
    return s;
}

std::shared_ptr<Tensor> Dirichlet::sample(const std::vector<int64_t>& sample_shape) {
    auto bs = batch_shape();
    int64_t K = concentration->shape.back();
    auto shape = full_shape(sample_shape, concentration->shape);
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    float* od = out->data_ptr() + out->offset;
    float* cd = concentration->data_ptr() + concentration->offset;
    size_t event_n = shape_numel(bs) * (size_t)K;
    size_t total = shape_numel(shape);
    size_t n_events = total / event_n;
    for (size_t e = 0; e < n_events; e++) {
        double sum = 0.0;
        for (size_t k = 0; k < (size_t)K; k++) {
            size_t ci = (e * event_n + k) % concentration->numel();
            std::gamma_distribution<float> gd(cd[ci], 1.0f);
            float x = gd(gen);
            od[e * event_n + k] = x;
            sum += x;
        }
        for (size_t k = 0; k < (size_t)K; k++)
            od[e * event_n + k] = (float)(od[e * event_n + k] / sum);
    }
    return out;
}

std::shared_ptr<Tensor> Dirichlet::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, concentration->shape);
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* cd = concentration->data_ptr() + concentration->offset;
    float* od = out->data_ptr() + out->offset;
    int64_t K = concentration->shape.back();
    size_t bn = shape_numel(batch_shape());
    for (size_t b = 0; b < bn; b++) {
        double lp = 0.0, csum = 0.0;
        bool valid = true;
        for (int64_t k = 0; k < K; k++) {
            double x = vd[b * K + k];
            double c = cd[(b * K + k) % concentration->numel()];
            if (x <= 0) { valid = false; break; }
            lp += (c - 1.0) * std::log(x);
            csum += c;
        }
        if (!valid) { od[b] = -INFINITY; continue; }
        double lgamma_sum = 0.0;
        for (int64_t k = 0; k < K; k++)
            lgamma_sum += std::lgamma(cd[(b * K + k) % concentration->numel()]);
        od[b] = (float)(lp + std::lgamma(csum) - lgamma_sum);
    }
    return out;
}

std::shared_ptr<Tensor> Dirichlet::entropy() {
    auto out = make_tensor(batch_shape());
    float* cd = concentration->data_ptr() + concentration->offset;
    float* od = out->data_ptr() + out->offset;
    int64_t K = concentration->shape.back();
    size_t bn = shape_numel(batch_shape());
    for (size_t b = 0; b < bn; b++) {
        double csum = 0.0;
        for (int64_t k = 0; k < K; k++) csum += cd[(b * K + k) % concentration->numel()];
        double e = std::lgamma(csum);
        for (int64_t k = 0; k < K; k++) {
            double c = cd[(b * K + k) % concentration->numel()];
            e += (c - 1.0) * (digamma(csum) - digamma(c)) - std::lgamma(c);
        }
        od[b] = (float)e;
    }
    return out;
}

std::shared_ptr<Tensor> Dirichlet::mean() {
    auto out = make_tensor(concentration->shape);
    float* cd = concentration->data_ptr() + concentration->offset;
    float* od = out->data_ptr() + out->offset;
    int64_t K = concentration->shape.back();
    size_t bn = shape_numel(batch_shape());
    for (size_t b = 0; b < bn; b++) {
        double csum = 0.0;
        for (int64_t k = 0; k < K; k++) csum += cd[(b * K + k) % concentration->numel()];
        for (int64_t k = 0; k < K; k++)
            od[b * K + k] = (float)(cd[(b * K + k) % concentration->numel()] / csum);
    }
    return out;
}

std::shared_ptr<Tensor> Dirichlet::variance() {
    auto out = make_tensor(concentration->shape);
    float* cd = concentration->data_ptr() + concentration->offset;
    float* od = out->data_ptr() + out->offset;
    int64_t K = concentration->shape.back();
    size_t bn = shape_numel(batch_shape());
    for (size_t b = 0; b < bn; b++) {
        double csum = 0.0;
        for (int64_t k = 0; k < K; k++) csum += cd[(b * K + k) % concentration->numel()];
        for (int64_t k = 0; k < K; k++) {
            double c = cd[(b * K + k) % concentration->numel()];
            od[b * K + k] = (float)(c * (csum - c) / (csum * csum * (csum + 1.0)));
        }
    }
    return out;
}

Bernoulli::Bernoulli(std::shared_ptr<Tensor> p, bool from_logits) {
    auto bs = broadcast_shapes({p});
    auto pb = broadcast_param(p, bs);
    probs = make_tensor(bs);
    float* pd = pb->data_ptr() + pb->offset;
    float* od = probs->data_ptr() + probs->offset;
    for (size_t i = 0; i < probs->numel(); i++) {
        double v = from_logits ? 1.0 / (1.0 + std::exp(-pd[i])) : pd[i];
        if (v < 0 || v > 1) throw std::runtime_error("Bernoulli: probs must be in [0, 1]");
        od[i] = (float)v;
    }
}

std::vector<int64_t> Bernoulli::batch_shape() const { return probs->shape; }

std::shared_ptr<Tensor> Bernoulli::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    float* od = out->data_ptr() + out->offset;
    float* pd = probs->data_ptr() + probs->offset;
    size_t bn = probs->numel();
    for (size_t i = 0; i < shape_numel(shape); i++)
        od[i] = (ud(gen) < pd[i % bn]) ? 1.0f : 0.0f;
    return out;
}

std::shared_ptr<Tensor> Bernoulli::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double p = pd[i];
        od[i] = (float)(vd[i] * std::log(p + 1e-30) + (1.0 - vd[i]) * std::log(1.0 - p + 1e-30));
    }
    return out;
}

std::shared_ptr<Tensor> Bernoulli::entropy() {
    auto out = make_tensor(batch_shape());
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double p = pd[i];
        od[i] = (float)(-(p * std::log(p + 1e-30) + (1.0 - p) * std::log(1.0 - p + 1e-30)));
    }
    return out;
}

std::shared_ptr<Tensor> Bernoulli::mean() { return probs; }

std::shared_ptr<Tensor> Bernoulli::variance() {
    auto out = make_tensor(batch_shape());
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = pd[i] * (1.0f - pd[i]);
    return out;
}

Categorical::Categorical(std::shared_ptr<Tensor> p, bool from_logits) {
    if (p->shape.empty()) throw std::runtime_error("Categorical: probs must have event dim");
    probs = make_tensor(p->shape);
    float* pd = p->data_ptr() + p->offset;
    float* od = probs->data_ptr() + probs->offset;
    int64_t K = p->shape.back();
    size_t bn = p->numel() / (size_t)K;
    for (size_t b = 0; b < bn; b++) {
        double sum = 0.0;
        for (int64_t k = 0; k < K; k++) {
            double v = from_logits ? std::exp(pd[b * K + k]) : pd[b * K + k];
            od[b * K + k] = (float)v;
            sum += v;
        }
        if (sum <= 0) throw std::runtime_error("Categorical: probs must sum to positive");
        for (int64_t k = 0; k < K; k++) od[b * K + k] = (float)(od[b * K + k] / sum);
    }
}

std::vector<int64_t> Categorical::batch_shape() const {
    auto s = probs->shape;
    s.pop_back();
    if (s.empty()) return {1};
    return s;
}

std::shared_ptr<Tensor> Categorical::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    float* od = out->data_ptr() + out->offset;
    float* pd = probs->data_ptr() + probs->offset;
    int64_t K = probs->shape.back();
    size_t bn = shape_numel(batch_shape());
    size_t total = shape_numel(shape);
    for (size_t i = 0; i < total; i++) {
        double u = ud(gen);
        double cum = 0.0;
        int64_t choice = K - 1;
        size_t b = i % bn;
        for (int64_t k = 0; k < K; k++) {
            cum += pd[b * K + k];
            if (u < cum) { choice = k; break; }
        }
        od[i] = (float)choice;
    }
    return out;
}

std::shared_ptr<Tensor> Categorical::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    int64_t K = probs->shape.back();
    for (size_t i = 0; i < out->numel(); i++) {
        int64_t k = (int64_t)vd[i];
        od[i] = (k >= 0 && k < K) ? (float)std::log(pd[i * K + k] + 1e-30) : -INFINITY;
    }
    return out;
}

std::shared_ptr<Tensor> Categorical::entropy() {
    auto out = make_tensor(batch_shape());
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    int64_t K = probs->shape.back();
    for (size_t i = 0; i < out->numel(); i++) {
        double e = 0.0;
        for (int64_t k = 0; k < K; k++) {
            double p = pd[i * K + k];
            e -= p * std::log(p + 1e-30);
        }
        od[i] = (float)e;
    }
    return out;
}

std::shared_ptr<Tensor> Categorical::mean() {
    auto out = make_tensor(batch_shape());
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = std::numeric_limits<float>::quiet_NaN();
    return out;
}

std::shared_ptr<Tensor> Categorical::variance() {
    auto out = make_tensor(batch_shape());
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = std::numeric_limits<float>::quiet_NaN();
    return out;
}

Poisson::Poisson(std::shared_ptr<Tensor> r) {
    auto bs = broadcast_shapes({r});
    rate = broadcast_param(r, bs);
    float* rd = rate->data_ptr() + rate->offset;
    for (size_t i = 0; i < rate->numel(); i++)
        if (rd[i] <= 0) throw std::runtime_error("Poisson: rate must be positive");
}

std::vector<int64_t> Poisson::batch_shape() const { return rate->shape; }

std::shared_ptr<Tensor> Poisson::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    float* od = out->data_ptr() + out->offset;
    float* rd = rate->data_ptr() + rate->offset;
    size_t bn = rate->numel();
    for (size_t i = 0; i < shape_numel(shape); i++) {
        std::poisson_distribution<int> pd(rd[i % bn]);
        od[i] = (float)pd(gen);
    }
    return out;
}

std::shared_ptr<Tensor> Poisson::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* rd = rate->data_ptr() + rate->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double k = vd[i];
        if (k < 0 || k != std::floor(k)) { od[i] = -INFINITY; continue; }
        od[i] = (float)(k * std::log(rd[i]) - rd[i] - std::lgamma(k + 1.0));
    }
    return out;
}

std::shared_ptr<Tensor> Poisson::entropy() {
    auto out = make_tensor(batch_shape());
    float* rd = rate->data_ptr() + rate->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double l = rd[i];
        od[i] = (float)(0.5 * (std::log(2.0 * PI) + 1.0 + std::log(l)) - 1.0 / (12.0 * l) - 1.0 / (24.0 * l * l));
    }
    return out;
}

std::shared_ptr<Tensor> Poisson::mean() { return rate; }
std::shared_ptr<Tensor> Poisson::variance() { return rate; }

Geometric::Geometric(std::shared_ptr<Tensor> p) {
    auto bs = broadcast_shapes({p});
    probs = broadcast_param(p, bs);
    float* pd = probs->data_ptr() + probs->offset;
    for (size_t i = 0; i < probs->numel(); i++)
        if (pd[i] <= 0 || pd[i] > 1) throw std::runtime_error("Geometric: probs must be in (0, 1]");
}

std::vector<int64_t> Geometric::batch_shape() const { return probs->shape; }

std::shared_ptr<Tensor> Geometric::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    float* od = out->data_ptr() + out->offset;
    float* pd = probs->data_ptr() + probs->offset;
    size_t bn = probs->numel();
    for (size_t i = 0; i < shape_numel(shape); i++) {
        std::geometric_distribution<int> gd(pd[i % bn]);
        od[i] = (float)gd(gen);
    }
    return out;
}

std::shared_ptr<Tensor> Geometric::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double k = vd[i];
        if (k < 0 || k != std::floor(k)) { od[i] = -INFINITY; continue; }
        od[i] = (float)(k * std::log(1.0 - pd[i]) + std::log(pd[i]));
    }
    return out;
}

std::shared_ptr<Tensor> Geometric::entropy() {
    auto out = make_tensor(batch_shape());
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double p = pd[i];
        od[i] = (float)((-(1.0 - p) * std::log(1.0 - p) - p * std::log(p)) / p);
    }
    return out;
}

std::shared_ptr<Tensor> Geometric::mean() {
    auto out = make_tensor(batch_shape());
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = (1.0f - pd[i]) / pd[i];
    return out;
}

std::shared_ptr<Tensor> Geometric::variance() {
    auto out = make_tensor(batch_shape());
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = (1.0f - pd[i]) / (pd[i] * pd[i]);
    return out;
}

Binomial::Binomial(std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> p) {
    auto bs = broadcast_shapes({t, p});
    total_count = broadcast_param(t, bs);
    probs = broadcast_param(p, bs);
    float* td = total_count->data_ptr() + total_count->offset;
    float* pd = probs->data_ptr() + probs->offset;
    for (size_t i = 0; i < total_count->numel(); i++) {
        if (td[i] < 0 || td[i] != std::floor(td[i])) throw std::runtime_error("Binomial: total_count must be non-negative integer");
        if (pd[i] < 0 || pd[i] > 1) throw std::runtime_error("Binomial: probs must be in [0, 1]");
    }
}

std::vector<int64_t> Binomial::batch_shape() const { return total_count->shape; }

std::shared_ptr<Tensor> Binomial::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    float* od = out->data_ptr() + out->offset;
    float* td = total_count->data_ptr() + total_count->offset;
    float* pd = probs->data_ptr() + probs->offset;
    size_t bn = total_count->numel();
    for (size_t i = 0; i < shape_numel(shape); i++) {
        std::binomial_distribution<int> bd((int)td[i % bn], pd[i % bn]);
        od[i] = (float)bd(gen);
    }
    return out;
}

static double log_binom_coeff(double n, double k) {
    return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
}

std::shared_ptr<Tensor> Binomial::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* td = total_count->data_ptr() + total_count->offset;
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double k = vd[i], n = td[i], p = pd[i];
        if (k < 0 || k > n || k != std::floor(k)) { od[i] = -INFINITY; continue; }
        od[i] = (float)(log_binom_coeff(n, k) + k * std::log(p + 1e-30) + (n - k) * std::log(1.0 - p + 1e-30));
    }
    return out;
}

std::shared_ptr<Tensor> Binomial::entropy() {
    auto out = make_tensor(batch_shape());
    float* td = total_count->data_ptr() + total_count->offset;
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double n = td[i], p = pd[i];
        od[i] = (float)(0.5 * std::log(2.0 * PI * EULER * n * p * (1.0 - p)));
    }
    return out;
}

std::shared_ptr<Tensor> Binomial::mean() {
    auto out = make_tensor(batch_shape());
    float* td = total_count->data_ptr() + total_count->offset;
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = td[i] * pd[i];
    return out;
}

std::shared_ptr<Tensor> Binomial::variance() {
    auto out = make_tensor(batch_shape());
    float* td = total_count->data_ptr() + total_count->offset;
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = td[i] * pd[i] * (1.0f - pd[i]);
    return out;
}

NegativeBinomial::NegativeBinomial(std::shared_ptr<Tensor> t, std::shared_ptr<Tensor> p) {
    auto bs = broadcast_shapes({t, p});
    total_count = broadcast_param(t, bs);
    probs = broadcast_param(p, bs);
    float* td = total_count->data_ptr() + total_count->offset;
    float* pd = probs->data_ptr() + probs->offset;
    for (size_t i = 0; i < total_count->numel(); i++) {
        if (td[i] <= 0) throw std::runtime_error("NegativeBinomial: total_count must be positive");
        if (pd[i] <= 0 || pd[i] > 1) throw std::runtime_error("NegativeBinomial: probs must be in (0, 1]");
    }
}

std::vector<int64_t> NegativeBinomial::batch_shape() const { return total_count->shape; }

std::shared_ptr<Tensor> NegativeBinomial::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    float* od = out->data_ptr() + out->offset;
    float* td = total_count->data_ptr() + total_count->offset;
    float* pd = probs->data_ptr() + probs->offset;
    size_t bn = total_count->numel();
    for (size_t i = 0; i < shape_numel(shape); i++) {
        std::negative_binomial_distribution<int> nbd((int)td[i % bn], pd[i % bn]);
        od[i] = (float)nbd(gen);
    }
    return out;
}

std::shared_ptr<Tensor> NegativeBinomial::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* td = total_count->data_ptr() + total_count->offset;
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double k = vd[i], r = td[i], p = pd[i];
        if (k < 0 || k != std::floor(k)) { od[i] = -INFINITY; continue; }
        od[i] = (float)(log_binom_coeff(k + r - 1.0, k) + r * std::log(p) + k * std::log(1.0 - p));
    }
    return out;
}

std::shared_ptr<Tensor> NegativeBinomial::entropy() {
    auto out = make_tensor(batch_shape());
    float* td = total_count->data_ptr() + total_count->offset;
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double r = td[i], p = pd[i];
        double q = 1.0 - p;
        od[i] = (float)((r * q / p) * (std::log(q) - digamma(r)) + r * (digamma(r * q / p + r) - std::log(p)));
    }
    return out;
}

std::shared_ptr<Tensor> NegativeBinomial::mean() {
    auto out = make_tensor(batch_shape());
    float* td = total_count->data_ptr() + total_count->offset;
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = td[i] * (1.0f - pd[i]) / pd[i];
    return out;
}

std::shared_ptr<Tensor> NegativeBinomial::variance() {
    auto out = make_tensor(batch_shape());
    float* td = total_count->data_ptr() + total_count->offset;
    float* pd = probs->data_ptr() + probs->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = td[i] * (1.0f - pd[i]) / (pd[i] * pd[i]);
    return out;
}

StudentT::StudentT(std::shared_ptr<Tensor> d) {
    auto bs = broadcast_shapes({d});
    df = broadcast_param(d, bs);
    float* dd = df->data_ptr() + df->offset;
    for (size_t i = 0; i < df->numel(); i++)
        if (dd[i] <= 0) throw std::runtime_error("StudentT: df must be positive");
}

std::vector<int64_t> StudentT::batch_shape() const { return df->shape; }

std::shared_ptr<Tensor> StudentT::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float* od = out->data_ptr() + out->offset;
    float* dd = df->data_ptr() + df->offset;
    size_t bn = df->numel();
    for (size_t i = 0; i < shape_numel(shape); i++) {
        std::chi_squared_distribution<float> cd(dd[i % bn]);
        od[i] = nd(gen) / std::sqrt(cd(gen) / dd[i % bn]);
    }
    return out;
}

std::shared_ptr<Tensor> StudentT::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* dd = df->data_ptr() + df->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double nu = dd[i], x = vd[i];
        od[i] = (float)(std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0)
            - 0.5 * std::log(nu * PI) - (nu + 1.0) / 2.0 * std::log(1.0 + x * x / nu));
    }
    return out;
}

std::shared_ptr<Tensor> StudentT::entropy() {
    auto out = make_tensor(batch_shape());
    float* dd = df->data_ptr() + df->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double nu = dd[i];
        od[i] = (float)((nu + 1.0) / 2.0 * (digamma((nu + 1.0) / 2.0) - digamma(nu / 2.0))
            + 0.5 * std::log(nu) + std::lgamma(nu / 2.0) - std::lgamma((nu + 1.0) / 2.0)
            + 0.5 * std::log(PI));
    }
    return out;
}

std::shared_ptr<Tensor> StudentT::mean() {
    auto out = make_tensor(batch_shape());
    float* dd = df->data_ptr() + df->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++)
        od[i] = (dd[i] > 1.0) ? 0.0f : std::numeric_limits<float>::quiet_NaN();
    return out;
}

std::shared_ptr<Tensor> StudentT::variance() {
    auto out = make_tensor(batch_shape());
    float* dd = df->data_ptr() + df->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        if (dd[i] > 2.0) od[i] = (float)(dd[i] / (dd[i] - 2.0));
        else if (dd[i] > 1.0) od[i] = std::numeric_limits<float>::infinity();
        else od[i] = std::numeric_limits<float>::quiet_NaN();
    }
    return out;
}

Chi2::Chi2(std::shared_ptr<Tensor> d) {
    auto bs = broadcast_shapes({d});
    df = broadcast_param(d, bs);
    float* dd = df->data_ptr() + df->offset;
    for (size_t i = 0; i < df->numel(); i++)
        if (dd[i] <= 0) throw std::runtime_error("Chi2: df must be positive");
}

std::vector<int64_t> Chi2::batch_shape() const { return df->shape; }

std::shared_ptr<Tensor> Chi2::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    float* od = out->data_ptr() + out->offset;
    float* dd = df->data_ptr() + df->offset;
    size_t bn = df->numel();
    for (size_t i = 0; i < shape_numel(shape); i++) {
        std::chi_squared_distribution<float> cd(dd[i % bn]);
        od[i] = cd(gen);
    }
    return out;
}

std::shared_ptr<Tensor> Chi2::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* dd = df->data_ptr() + df->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double k = dd[i], x = vd[i];
        if (x <= 0) { od[i] = -INFINITY; continue; }
        od[i] = (float)((k / 2.0 - 1.0) * std::log(x) - x / 2.0 - (k / 2.0) * std::log(2.0) - std::lgamma(k / 2.0));
    }
    return out;
}

std::shared_ptr<Tensor> Chi2::entropy() {
    auto out = make_tensor(batch_shape());
    float* dd = df->data_ptr() + df->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double k = dd[i];
        od[i] = (float)(k / 2.0 + std::log(2.0) + std::lgamma(k / 2.0) + (1.0 - k / 2.0) * digamma(k / 2.0));
    }
    return out;
}

std::shared_ptr<Tensor> Chi2::mean() { return df; }

std::shared_ptr<Tensor> Chi2::variance() {
    auto out = make_tensor(batch_shape());
    float* dd = df->data_ptr() + df->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = 2.0f * dd[i];
    return out;
}

LogNormal::LogNormal(std::shared_ptr<Tensor> l, std::shared_ptr<Tensor> s) {
    auto bs = broadcast_shapes({l, s});
    loc = broadcast_param(l, bs);
    scale = broadcast_param(s, bs);
    float* sd = scale->data_ptr() + scale->offset;
    for (size_t i = 0; i < scale->numel(); i++)
        if (sd[i] <= 0) throw std::runtime_error("LogNormal: scale must be positive");
}

std::vector<int64_t> LogNormal::batch_shape() const { return loc->shape; }

std::shared_ptr<Tensor> LogNormal::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float* od = out->data_ptr() + out->offset;
    float* ld = loc->data_ptr() + loc->offset;
    float* sd = scale->data_ptr() + scale->offset;
    size_t bn = loc->numel();
    for (size_t i = 0; i < shape_numel(shape); i++)
        od[i] = std::exp(ld[i % bn] + sd[i % bn] * nd(gen));
    return out;
}

std::shared_ptr<Tensor> LogNormal::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* ld = loc->data_ptr() + loc->offset;
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        if (vd[i] <= 0) { od[i] = -INFINITY; continue; }
        double z = (std::log(vd[i]) - ld[i]) / sd[i];
        od[i] = (float)(-std::log(vd[i]) - 0.5 * z * z - std::log(sd[i]) - 0.5 * std::log(2.0 * PI));
    }
    return out;
}

std::shared_ptr<Tensor> LogNormal::entropy() {
    auto out = make_tensor(batch_shape());
    float* ld = loc->data_ptr() + loc->offset;
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++)
        od[i] = (float)(ld[i] + 0.5 * (std::log(2.0 * PI) + 1.0) + std::log(sd[i]));
    return out;
}

std::shared_ptr<Tensor> LogNormal::mean() {
    auto out = make_tensor(batch_shape());
    float* ld = loc->data_ptr() + loc->offset;
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++)
        od[i] = (float)std::exp(ld[i] + 0.5 * sd[i] * sd[i]);
    return out;
}

std::shared_ptr<Tensor> LogNormal::variance() {
    auto out = make_tensor(batch_shape());
    float* ld = loc->data_ptr() + loc->offset;
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double s2 = sd[i] * sd[i];
        od[i] = (float)((std::exp(s2) - 1.0) * std::exp(2.0 * ld[i] + s2));
    }
    return out;
}

Gumbel::Gumbel(std::shared_ptr<Tensor> l, std::shared_ptr<Tensor> s) {
    auto bs = broadcast_shapes({l, s});
    loc = broadcast_param(l, bs);
    scale = broadcast_param(s, bs);
    float* sd = scale->data_ptr() + scale->offset;
    for (size_t i = 0; i < scale->numel(); i++)
        if (sd[i] <= 0) throw std::runtime_error("Gumbel: scale must be positive");
}

std::vector<int64_t> Gumbel::batch_shape() const { return loc->shape; }

std::shared_ptr<Tensor> Gumbel::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    float* od = out->data_ptr() + out->offset;
    float* ld = loc->data_ptr() + loc->offset;
    float* sd = scale->data_ptr() + scale->offset;
    size_t bn = loc->numel();
    for (size_t i = 0; i < shape_numel(shape); i++) {
        double u = std::max((double)ud(gen), 1e-30);
        od[i] = (float)(ld[i % bn] - sd[i % bn] * std::log(-std::log(u)));
    }
    return out;
}

std::shared_ptr<Tensor> Gumbel::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* ld = loc->data_ptr() + loc->offset;
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double z = (vd[i] - ld[i]) / sd[i];
        od[i] = (float)(-z - std::exp(-z) - std::log(sd[i]));
    }
    return out;
}

std::shared_ptr<Tensor> Gumbel::entropy() {
    auto out = make_tensor(batch_shape());
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++)
        od[i] = (float)(std::log(sd[i]) + 1.0 + EULER);
    return out;
}

std::shared_ptr<Tensor> Gumbel::mean() {
    auto out = make_tensor(batch_shape());
    float* ld = loc->data_ptr() + loc->offset;
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = (float)(ld[i] + sd[i] * EULER);
    return out;
}

std::shared_ptr<Tensor> Gumbel::variance() {
    auto out = make_tensor(batch_shape());
    float* sd = scale->data_ptr() + scale->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) od[i] = (float)(PI * PI / 6.0 * sd[i] * sd[i]);
    return out;
}

Weibull::Weibull(std::shared_ptr<Tensor> s, std::shared_ptr<Tensor> c) {
    auto bs = broadcast_shapes({s, c});
    scale = broadcast_param(s, bs);
    concentration = broadcast_param(c, bs);
    float* sd = scale->data_ptr() + scale->offset;
    float* cd = concentration->data_ptr() + concentration->offset;
    for (size_t i = 0; i < scale->numel(); i++)
        if (sd[i] <= 0 || cd[i] <= 0) throw std::runtime_error("Weibull: scale and concentration must be positive");
}

std::vector<int64_t> Weibull::batch_shape() const { return scale->shape; }

std::shared_ptr<Tensor> Weibull::sample(const std::vector<int64_t>& sample_shape) {
    auto shape = full_shape(sample_shape, batch_shape());
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    float* od = out->data_ptr() + out->offset;
    float* sd = scale->data_ptr() + scale->offset;
    float* cd = concentration->data_ptr() + concentration->offset;
    size_t bn = scale->numel();
    for (size_t i = 0; i < shape_numel(shape); i++) {
        double u = std::max((double)ud(gen), 1e-30);
        od[i] = (float)(sd[i % bn] * std::pow(-std::log(1.0 - u), 1.0 / cd[i % bn]));
    }
    return out;
}

std::shared_ptr<Tensor> Weibull::log_prob(std::shared_ptr<Tensor> value) {
    auto v = broadcast_param(value, batch_shape());
    auto out = make_tensor(batch_shape());
    float* vd = v->data_ptr() + v->offset;
    float* sd = scale->data_ptr() + scale->offset;
    float* cd = concentration->data_ptr() + concentration->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        if (vd[i] <= 0) { od[i] = -INFINITY; continue; }
        double k = cd[i], l = sd[i], x = vd[i];
        double xl = x / l;
        od[i] = (float)(std::log(k) - std::log(l) + (k - 1.0) * std::log(xl) - std::pow(xl, k));
    }
    return out;
}

std::shared_ptr<Tensor> Weibull::entropy() {
    auto out = make_tensor(batch_shape());
    float* sd = scale->data_ptr() + scale->offset;
    float* cd = concentration->data_ptr() + concentration->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++)
        od[i] = (float)(EULER * (1.0 - 1.0 / cd[i]) + std::log(sd[i] / cd[i]) + 1.0);
    return out;
}

std::shared_ptr<Tensor> Weibull::mean() {
    auto out = make_tensor(batch_shape());
    float* sd = scale->data_ptr() + scale->offset;
    float* cd = concentration->data_ptr() + concentration->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++)
        od[i] = (float)(sd[i] * std::tgamma(1.0 + 1.0 / cd[i]));
    return out;
}

std::shared_ptr<Tensor> Weibull::variance() {
    auto out = make_tensor(batch_shape());
    float* sd = scale->data_ptr() + scale->offset;
    float* cd = concentration->data_ptr() + concentration->offset;
    float* od = out->data_ptr() + out->offset;
    for (size_t i = 0; i < out->numel(); i++) {
        double k = cd[i];
        double g1 = std::tgamma(1.0 + 1.0 / k);
        double g2 = std::tgamma(1.0 + 2.0 / k);
        od[i] = (float)(sd[i] * sd[i] * (g2 - g1 * g1));
    }
    return out;
}

MultivariateNormal::MultivariateNormal(std::shared_ptr<Tensor> l, std::shared_ptr<Tensor> c) {
    if (l->shape.empty() || c->shape.size() < 2) throw std::runtime_error("MultivariateNormal: bad shapes");
    int64_t n = l->shape.back();
    if (c->shape[c->shape.size()-2] != n || c->shape.back() != n)
        throw std::runtime_error("MultivariateNormal: covariance_matrix must be (..., n, n)");
    loc = l;
    covariance_matrix = c;
    int64_t rows = c->shape[c->shape.size()-2];
    int64_t cols = c->shape.back();
    size_t bn = c->numel() / ((size_t)rows * (size_t)cols);
    float* cd = c->data_ptr() + c->offset;
    for (size_t b = 0; b < bn; b++) {
        std::vector<double> L((size_t)rows * (size_t)cols, 0.0);
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j <= i; j++) {
                double s = cd[b * rows * cols + i * cols + j];
                for (int64_t k = 0; k < j; k++) s -= L[i * cols + k] * L[j * cols + k];
                if (i == j) {
                    if (s <= 0) throw std::runtime_error("MultivariateNormal: covariance_matrix must be positive definite");
                    L[i * cols + j] = std::sqrt(s);
                } else {
                    L[i * cols + j] = s / L[j * cols + j];
                }
            }
        }
    }
}

std::vector<int64_t> MultivariateNormal::batch_shape() const {
    auto s = loc->shape;
    s.pop_back();
    if (s.empty()) return {1};
    return s;
}

static std::vector<float> cholesky_factor(const float* cd, int64_t n) {
    std::vector<float> L((size_t)n * (size_t)n, 0.0f);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j <= i; j++) {
            double s = cd[i * n + j];
            for (int64_t k = 0; k < j; k++) s -= (double)L[i * n + k] * L[j * n + k];
            if (i == j) L[i * n + j] = (float)std::sqrt(std::max(s, 0.0));
            else L[i * n + j] = (float)(s / L[j * n + j]);
        }
    }
    return L;
}

std::shared_ptr<Tensor> MultivariateNormal::sample(const std::vector<int64_t>& sample_shape) {
    int64_t n = loc->shape.back();
    auto bs = batch_shape();
    std::vector<int64_t> shape = sample_shape;
    shape.insert(shape.end(), bs.begin(), bs.end());
    shape.push_back(n);
    auto out = make_tensor(shape);
    auto& gen = get_generator();
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float* od = out->data_ptr() + out->offset;
    float* ld = loc->data_ptr() + loc->offset;
    float* cd = covariance_matrix->data_ptr() + covariance_matrix->offset;
    size_t bn = shape_numel(bs);
    size_t n_out_events = shape_numel(shape) / ((size_t)bn * (size_t)n);
    std::vector<float> z((size_t)n);
    for (size_t e = 0; e < n_out_events * bn; e++) {
        size_t b = e % bn;
        auto L = cholesky_factor(cd + b * n * n, n);
        for (int64_t i = 0; i < n; i++) z[i] = nd(gen);
        for (int64_t i = 0; i < n; i++) {
            double s = ld[(b * n + i) % loc->numel()];
            for (int64_t j = 0; j <= i; j++) s += L[i * n + j] * z[j];
            od[e * n + i] = (float)s;
        }
    }
    return out;
}

std::shared_ptr<Tensor> MultivariateNormal::log_prob(std::shared_ptr<Tensor> value) {
    int64_t n = loc->shape.back();
    auto bs = batch_shape();
    auto out = make_tensor(bs);
    float* vd = value->data_ptr() + value->offset;
    float* ld = loc->data_ptr() + loc->offset;
    float* cd = covariance_matrix->data_ptr() + covariance_matrix->offset;
    float* od = out->data_ptr() + out->offset;
    size_t bn = shape_numel(bs);
    std::vector<double> d((size_t)n), y((size_t)n);
    for (size_t b = 0; b < bn; b++) {
        auto L = cholesky_factor(cd + b * n * n, n);
        for (int64_t i = 0; i < n; i++)
            d[i] = vd[(b * n + i) % value->numel()] - ld[(b * n + i) % loc->numel()];
        double logdet = 0.0;
        for (int64_t i = 0; i < n; i++) {
            double s = d[i];
            for (int64_t j = 0; j < i; j++) s -= L[i * n + j] * y[j];
            y[i] = s / L[i * n + i];
            logdet += std::log(L[i * n + i]);
        }
        double q = 0.0;
        for (int64_t i = 0; i < n; i++) q += y[i] * y[i];
        od[b] = (float)(-0.5 * (n * std::log(2.0 * PI) + 2.0 * logdet + q));
    }
    return out;
}

std::shared_ptr<Tensor> MultivariateNormal::entropy() {
    int64_t n = loc->shape.back();
    auto bs = batch_shape();
    auto out = make_tensor(bs);
    float* cd = covariance_matrix->data_ptr() + covariance_matrix->offset;
    float* od = out->data_ptr() + out->offset;
    size_t bn = shape_numel(bs);
    for (size_t b = 0; b < bn; b++) {
        auto L = cholesky_factor(cd + b * n * n, n);
        double logdet = 0.0;
        for (int64_t i = 0; i < n; i++) logdet += std::log(std::max((double)L[i * n + i], 1e-30));
        od[b] = (float)(0.5 * (n * (1.0 + std::log(2.0 * PI)) + 2.0 * logdet));
    }
    return out;
}

std::shared_ptr<Tensor> MultivariateNormal::mean() { return loc; }

std::shared_ptr<Tensor> MultivariateNormal::variance() {
    int64_t n = loc->shape.back();
    auto bs = batch_shape();
    std::vector<int64_t> shape = bs;
    shape.push_back(n);
    auto out = make_tensor(shape);
    float* cd = covariance_matrix->data_ptr() + covariance_matrix->offset;
    float* od = out->data_ptr() + out->offset;
    size_t bn = shape_numel(bs);
    for (size_t b = 0; b < bn; b++)
        for (int64_t i = 0; i < n; i++)
            od[b * n + i] = cd[b * n * n + i * n + i];
    return out;
}

std::shared_ptr<Tensor> kl_divergence(std::shared_ptr<Distribution> p, std::shared_ptr<Distribution> q) {
    auto pn = std::dynamic_pointer_cast<Normal>(p);
    auto qn = std::dynamic_pointer_cast<Normal>(q);
    if (pn && qn) {
        auto bs = broadcast_shapes({pn->loc, pn->scale, qn->loc, qn->scale});
        auto pl = broadcast_param(pn->loc, bs), ps = broadcast_param(pn->scale, bs);
        auto ql = broadcast_param(qn->loc, bs), qs = broadcast_param(qn->scale, bs);
        auto out = make_tensor(bs);
        float* pld = pl->data_ptr() + pl->offset;
        float* psd = ps->data_ptr() + ps->offset;
        float* qld = ql->data_ptr() + ql->offset;
        float* qsd = qs->data_ptr() + qs->offset;
        float* od = out->data_ptr() + out->offset;
        for (size_t i = 0; i < out->numel(); i++) {
            double dm = qld[i] - pld[i];
            od[i] = (float)(std::log(qsd[i] / psd[i]) + (psd[i]*psd[i] + dm*dm) / (2.0*qsd[i]*qsd[i]) - 0.5);
        }
        return out;
    }
    auto pe = std::dynamic_pointer_cast<Exponential>(p);
    auto qe = std::dynamic_pointer_cast<Exponential>(q);
    if (pe && qe) {
        auto bs = broadcast_shapes({pe->rate, qe->rate});
        auto pr = broadcast_param(pe->rate, bs), qr = broadcast_param(qe->rate, bs);
        auto out = make_tensor(bs);
        float* prd = pr->data_ptr() + pr->offset;
        float* qrd = qr->data_ptr() + qr->offset;
        float* od = out->data_ptr() + out->offset;
        for (size_t i = 0; i < out->numel(); i++)
            od[i] = (float)(std::log(prd[i] / qrd[i]) + qrd[i] / prd[i] - 1.0);
        return out;
    }
    throw std::runtime_error("kl_divergence: only Normal-Normal and Exponential-Exponential pairs supported");
}

}
}
