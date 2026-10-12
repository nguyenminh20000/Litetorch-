#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include <cmath>
#include <random>
#include <stdexcept>

namespace litetorch {

extern std::mt19937& get_generator();

namespace nn {

static void uniform_fill(const std::shared_ptr<Tensor>& t, float bound) {
    std::mt19937& gen = get_generator();
    std::uniform_real_distribution<float> dis(-bound, bound);
    float* d = t->data_ptr();
    for (size_t i = 0; i < t->numel(); ++i) d[i] = dis(gen);
}

static void cpu_only(const std::shared_ptr<Tensor>& t, const char* who) {
    if (t->device.type != DeviceType::CPU)
        throw std::runtime_error(std::string("[litetorch Error] ") + who + ": only CPU supported");
}

static bool next_index(std::vector<int64_t>& idx, const std::vector<int64_t>& dims) {
    for (int d = (int)dims.size() - 1; d >= 0; --d) {
        if (++idx[d] < dims[d]) return true;
        idx[d] = 0;
    }
    return false;
}

static int64_t prod_dims(const std::vector<int64_t>& v) {
    int64_t p = 1;
    for (auto d : v) p *= d;
    return p;
}

static std::vector<int64_t> row_strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> s(dims.size(), 1);
    for (int d = (int)dims.size() - 2; d >= 0; --d) s[d] = s[d + 1] * dims[d + 1];
    return s;
}

struct CTGeom {
    int nd;
    int64_t N, Cin, Cout, G, CinG, CoutG;
    std::vector<int64_t> S, K, O;
    int stride, padding, out_pad, dilation;
    std::vector<int64_t> sS, sK, sO;
    int64_t flatS, flatK, flatO;
};

static CTGeom ct_geom(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& weight,
                      int stride, int padding, int out_pad, int groups, int dilation) {
    CTGeom g;
    g.nd = (int)input->shape.size() - 2;
    g.N = input->shape[0];
    g.Cin = input->shape[1];
    g.CoutG = weight->shape[1];
    g.G = groups;
    g.CinG = g.Cin / groups;
    g.Cout = g.CoutG * groups;
    g.stride = stride;
    g.padding = padding;
    g.out_pad = out_pad;
    g.dilation = dilation;
    for (int d = 0; d < g.nd; ++d) {
        g.S.push_back(input->shape[2 + d]);
        g.K.push_back(weight->shape[2 + d]);
        g.O.push_back((g.S[d] - 1) * stride - 2 * padding + dilation * (g.K[d] - 1) + 1 + out_pad);
    }
    g.sS = row_strides(g.S);
    g.sK = row_strides(g.K);
    g.sO = row_strides(g.O);
    g.flatS = prod_dims(g.S);
    g.flatK = prod_dims(g.K);
    g.flatO = prod_dims(g.O);
    return g;
}

static std::shared_ptr<Tensor> ct_forward_raw(const std::shared_ptr<Tensor>& input,
                                              const std::shared_ptr<Tensor>& weight,
                                              const std::shared_ptr<Tensor>& bias,
                                              const CTGeom& g) {
    std::vector<int64_t> out_shape = {g.N, g.Cout};
    for (auto d : g.O) out_shape.push_back(d);
    auto out = Tensor::create(out_shape, input->device, false, input->dtype);
    float* od = out->data_ptr();
    for (int64_t i = 0; i < out->numel(); ++i) od[i] = 0.0f;
    const float* xd = input->data_ptr();
    const float* wd = weight->data_ptr();
    const float* bd = bias ? bias->data_ptr() : nullptr;
    std::vector<int64_t> oidx(g.nd, 0), kidx(g.nd, 0);
    for (int64_t n = 0; n < g.N; ++n) {
        for (int64_t gg = 0; gg < g.G; ++gg) {
            for (int64_t oc = 0; oc < g.CoutG; ++oc) {
                std::fill(oidx.begin(), oidx.end(), 0);
                do {
                    int64_t oflat = 0;
                    for (int d = 0; d < g.nd; ++d) oflat += oidx[d] * g.sO[d];
                    float acc = bd ? bd[gg * g.CoutG + oc] : 0.0f;
                    std::fill(kidx.begin(), kidx.end(), 0);
                    do {
                        int64_t sflat = 0, kflat = 0;
                        bool ok = true;
                        for (int d = 0; d < g.nd; ++d) {
                            int64_t sd = oidx[d] + g.padding - kidx[d] * g.dilation;
                            if (sd < 0 || sd % g.stride != 0) { ok = false; break; }
                            sd /= g.stride;
                            if (sd >= g.S[d]) { ok = false; break; }
                            sflat += sd * g.sS[d];
                            kflat += kidx[d] * g.sK[d];
                        }
                        if (!ok) continue;
                        for (int64_t ic = 0; ic < g.CinG; ++ic) {
                            int64_t xoff = (n * g.Cin + gg * g.CinG + ic) * g.flatS + sflat;
                            int64_t woff = ((gg * g.CinG + ic) * g.CoutG + oc) * g.flatK + kflat;
                            acc += xd[xoff] * wd[woff];
                        }
                    } while (next_index(kidx, g.K));
                    od[(n * g.Cout + gg * g.CoutG + oc) * g.flatO + oflat] = acc;
                } while (next_index(oidx, g.O));
            }
        }
    }
    return out;
}

class ConvTransposeNode : public Node {
public:
    CTGeom g;
    bool has_bias;
    ConvTransposeNode(const CTGeom& g, bool has_bias) : Node("ConvTranspose"), g(g), has_bias(has_bias) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto weight = saved_tensors[1];
        auto go = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        const float* gd = go->data_ptr();
        const float* xd = input->data_ptr();
        const float* wd = weight->data_ptr();
        std::shared_ptr<Tensor> grad_input = nullptr, grad_weight = nullptr, grad_bias = nullptr;
        if (inputs[0].requires_grad) {
            grad_input = Tensor::create(input->shape, input->device, false, input->dtype);
            float* id = grad_input->data_ptr();
            for (int64_t i = 0; i < grad_input->numel(); ++i) id[i] = 0.0f;
        }
        if (inputs[1].requires_grad) {
            grad_weight = Tensor::create(weight->shape, weight->device, false, weight->dtype);
            float* wd2 = grad_weight->data_ptr();
            for (int64_t i = 0; i < grad_weight->numel(); ++i) wd2[i] = 0.0f;
        }
        if (has_bias && inputs[2].requires_grad) {
            grad_bias = Tensor::create({g.Cout}, go->device, false, go->dtype);
            float* bd = grad_bias->data_ptr();
            for (int64_t i = 0; i < g.Cout; ++i) bd[i] = 0.0f;
        }
        float* id = grad_input ? grad_input->data_ptr() : nullptr;
        float* wgd = grad_weight ? grad_weight->data_ptr() : nullptr;
        float* bgd = grad_bias ? grad_bias->data_ptr() : nullptr;
        std::vector<int64_t> oidx(g.nd, 0), kidx(g.nd, 0);
        for (int64_t n = 0; n < g.N; ++n) {
            for (int64_t gg = 0; gg < g.G; ++gg) {
                for (int64_t oc = 0; oc < g.CoutG; ++oc) {
                    std::fill(oidx.begin(), oidx.end(), 0);
                    do {
                        int64_t oflat = 0;
                        for (int d = 0; d < g.nd; ++d) oflat += oidx[d] * g.sO[d];
                        float gv = gd[(n * g.Cout + gg * g.CoutG + oc) * g.flatO + oflat];
                        if (bgd) bgd[gg * g.CoutG + oc] += gv;
                        std::fill(kidx.begin(), kidx.end(), 0);
                        do {
                            int64_t sflat = 0, kflat = 0;
                            bool ok = true;
                            for (int d = 0; d < g.nd; ++d) {
                                int64_t sd = oidx[d] + g.padding - kidx[d] * g.dilation;
                                if (sd < 0 || sd % g.stride != 0) { ok = false; break; }
                                sd /= g.stride;
                                if (sd >= g.S[d]) { ok = false; break; }
                                sflat += sd * g.sS[d];
                                kflat += kidx[d] * g.sK[d];
                            }
                            if (!ok) continue;
                            for (int64_t ic = 0; ic < g.CinG; ++ic) {
                                int64_t xoff = (n * g.Cin + gg * g.CinG + ic) * g.flatS + sflat;
                                int64_t woff = ((gg * g.CinG + ic) * g.CoutG + oc) * g.flatK + kflat;
                                if (id) id[xoff] += gv * wd[woff];
                                if (wgd) wgd[woff] += gv * xd[xoff];
                            }
                        } while (next_index(kidx, g.K));
                    } while (next_index(oidx, g.O));
                }
            }
        }
        if (has_bias) return {grad_input, grad_weight, grad_bias};
        return {grad_input, grad_weight};
    }
};

static std::shared_ptr<Tensor> conv_transpose_op(std::shared_ptr<Tensor> input,
                                                 std::shared_ptr<Tensor> weight,
                                                 std::shared_ptr<Tensor> bias,
                                                 int stride, int padding, int out_pad,
                                                 int groups, int dilation) {
    cpu_only(input, "conv_transpose");
    if ((int)input->shape.size() < 3 || (int)weight->shape.size() != (int)input->shape.size())
        throw std::runtime_error("[litetorch Error] conv_transpose: rank mismatch");
    if (input->shape[1] % groups != 0 || weight->shape[1] * groups % groups != 0)
        throw std::runtime_error("[litetorch Error] conv_transpose: channels not divisible by groups");
    if (weight->shape[0] != input->shape[1])
        throw std::runtime_error("[litetorch Error] conv_transpose: weight in_channels mismatch");
    if (bias && bias->shape != std::vector<int64_t>{weight->shape[1] * groups})
        throw std::runtime_error("[litetorch Error] conv_transpose: bias shape mismatch");
    auto in_c = input->is_contiguous() ? input : input->contiguous();
    auto w_c = weight->is_contiguous() ? weight : weight->contiguous();
    CTGeom g = ct_geom(in_c, w_c, stride, padding, out_pad, groups, dilation);
    auto out = ct_forward_raw(in_c, w_c, bias, g);
    bool need_grad = (input->requires_grad || weight->requires_grad || (bias && bias->requires_grad)) &&
                     Autograd::is_grad_enabled();
    if (need_grad) {
        auto node = std::make_shared<ConvTransposeNode>(g, bias != nullptr);
        if (bias) {
            node->inputs = {{in_c, input->requires_grad}, {w_c, weight->requires_grad}, {bias, bias->requires_grad}};
            node->next_nodes = {in_c->creator, w_c->creator, bias->creator};
            node->saved_tensors = {in_c, w_c, bias};
        } else {
            node->inputs = {{in_c, input->requires_grad}, {w_c, weight->requires_grad}};
            node->next_nodes = {in_c->creator, w_c->creator};
            node->saved_tensors = {in_c, w_c};
        }
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

ConvTranspose1d::ConvTranspose1d(int in_channels, int out_channels, int kernel_size, int stride,
                                 int padding, int output_padding, int groups, bool bias, int dilation)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride),
      padding(padding), output_padding(output_padding), groups(groups), dilation(dilation), has_bias(bias) {
    weight = Tensor::create({in_channels, out_channels / groups, kernel_size}, Device(DeviceType::CPU, 0), true);
    uniform_fill(weight, 1.0f / std::sqrt(static_cast<float>(in_channels * kernel_size)));
    register_parameter("weight", weight);
    if (bias) {
        this->bias = Tensor::create({out_channels}, Device(DeviceType::CPU, 0), true);
        uniform_fill(this->bias, 1.0f / std::sqrt(static_cast<float>(in_channels * kernel_size)));
        register_parameter("bias", this->bias);
    }
}

std::shared_ptr<Tensor> ConvTranspose1d::forward(std::shared_ptr<Tensor> input) {
    return conv_transpose_op(input, weight, bias, stride, padding, output_padding, groups, dilation);
}

std::vector<std::shared_ptr<Tensor>> ConvTranspose1d::parameters() { return Module::parameters(); }

void ConvTranspose1d::to(const Device& device) {
    weight = weight->to(device);
    if (bias) bias = bias->to(device);
}

ConvTranspose2d::ConvTranspose2d(int in_channels, int out_channels, int kernel_size, int stride,
                                 int padding, int output_padding, int groups, bool bias, int dilation)
    : in_channels(in_channels), out_channels(out_channels), kernel_h(kernel_size), kernel_w(kernel_size),
      stride(stride), padding(padding), output_padding(output_padding), groups(groups),
      dilation(dilation), has_bias(bias) {
    weight = Tensor::create({in_channels, out_channels / groups, kernel_size, kernel_size},
                            Device(DeviceType::CPU, 0), true);
    uniform_fill(weight, 1.0f / std::sqrt(static_cast<float>(in_channels * kernel_size * kernel_size)));
    register_parameter("weight", weight);
    if (bias) {
        this->bias = Tensor::create({out_channels}, Device(DeviceType::CPU, 0), true);
        uniform_fill(this->bias, 1.0f / std::sqrt(static_cast<float>(in_channels * kernel_size * kernel_size)));
        register_parameter("bias", this->bias);
    }
}

std::shared_ptr<Tensor> ConvTranspose2d::forward(std::shared_ptr<Tensor> input) {
    return conv_transpose_op(input, weight, bias, stride, padding, output_padding, groups, dilation);
}

std::vector<std::shared_ptr<Tensor>> ConvTranspose2d::parameters() { return Module::parameters(); }

void ConvTranspose2d::to(const Device& device) {
    weight = weight->to(device);
    if (bias) bias = bias->to(device);
}

ConvTranspose3d::ConvTranspose3d(int in_channels, int out_channels, int kernel_size, int stride,
                                 int padding, int output_padding, int groups, bool bias, int dilation)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride),
      padding(padding), output_padding(output_padding), groups(groups), dilation(dilation), has_bias(bias) {
    weight = Tensor::create({in_channels, out_channels / groups, kernel_size, kernel_size, kernel_size},
                            Device(DeviceType::CPU, 0), true);
    uniform_fill(weight, 1.0f / std::sqrt(static_cast<float>(in_channels * kernel_size * kernel_size * kernel_size)));
    register_parameter("weight", weight);
    if (bias) {
        this->bias = Tensor::create({out_channels}, Device(DeviceType::CPU, 0), true);
        uniform_fill(this->bias, 1.0f / std::sqrt(static_cast<float>(in_channels * kernel_size * kernel_size * kernel_size)));
        register_parameter("bias", this->bias);
    }
}

std::shared_ptr<Tensor> ConvTranspose3d::forward(std::shared_ptr<Tensor> input) {
    return conv_transpose_op(input, weight, bias, stride, padding, output_padding, groups, dilation);
}

std::vector<std::shared_ptr<Tensor>> ConvTranspose3d::parameters() { return Module::parameters(); }

void ConvTranspose3d::to(const Device& device) {
    weight = weight->to(device);
    if (bias) bias = bias->to(device);
}

static std::shared_ptr<Tensor> mul_scalar(const std::shared_ptr<Tensor>& t, float s) {
    auto st = Tensor::create({1}, t->device, false, t->dtype);
    st->data_ptr()[0] = s;
    return Ops::mul(t, st);
}

static std::shared_ptr<Tensor> expand_channel_stat(std::shared_ptr<Tensor> stat, int64_t G, int64_t C,
                                                   bool instance) {
    if (!instance && G == C) return stat->reshape({G, 1});
    auto idx = Tensor::create({G}, stat->device, false, DataType::INT32);
    int32_t* idp = reinterpret_cast<int32_t*>(idx->data_ptr());
    for (int64_t g = 0; g < G; ++g) idp[g] = static_cast<int32_t>(instance ? (g % C) : g);
    return stat->index_select(0, idx)->reshape({G, 1});
}

static std::shared_ptr<Tensor> channel_norm(std::shared_ptr<Tensor> flat, int64_t G, int64_t M,
                                            int64_t C, bool instance,
                                            std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias,
                                            std::shared_ptr<Tensor> running_mean,
                                            std::shared_ptr<Tensor> running_var,
                                            float eps, float momentum, bool affine,
                                            bool track_running_stats, bool training) {
    auto mu = Ops::mean(flat, 1, true);
    auto diff = Ops::sub(flat, mu);
    auto var = Ops::mean(Ops::mul(diff, diff), 1, true);
    std::shared_ptr<Tensor> use_mu = mu, use_var = var;
    if (training && track_running_stats && running_mean && running_var) {
        NoGradGuard ng;
        std::shared_ptr<Tensor> bmu, bvar;
        if (instance) {
            int64_t N = G / C;
            auto f2 = flat->reshape({N, C, M});
            bmu = Ops::mean(Ops::mean(f2, 2, false), 0, false);
            auto bmu_e = bmu->reshape({1, C, 1});
            auto d2 = Ops::sub(f2, bmu_e);
            bvar = Ops::mean(Ops::mean(Ops::mul(d2, d2), 2, false), 0, false);
            int64_t n = N * M;
            if (n > 1) bvar = mul_scalar(bvar, static_cast<float>(n) / static_cast<float>(n - 1));
        } else {
            bmu = mu->reshape({C});
            bvar = var->reshape({C});
            if (M > 1) bvar = mul_scalar(bvar, static_cast<float>(M) / static_cast<float>(M - 1));
        }
        running_mean->copy_(Ops::add(mul_scalar(running_mean, 1.0f - momentum), mul_scalar(bmu, momentum)));
        running_var->copy_(Ops::add(mul_scalar(running_var, 1.0f - momentum), mul_scalar(bvar, momentum)));
    } else if (!training && track_running_stats && running_mean && running_var) {
        use_mu = expand_channel_stat(running_mean, G, C, instance);
        use_var = expand_channel_stat(running_var, G, C, instance);
    }
    auto eps_t = Tensor::create({1}, flat->device, false, flat->dtype);
    eps_t->data_ptr()[0] = eps;
    auto denom = Ops::sqrt(Ops::add(use_var, eps_t));
    auto xhat = Ops::div(Ops::sub(flat, use_mu), denom);
    if (affine && weight && bias) {
        auto w = expand_channel_stat(weight, G, C, instance);
        auto b = expand_channel_stat(bias, G, C, instance);
        xhat = Ops::add(Ops::mul(xhat, w), b);
    }
    return xhat;
}

static void init_norm_params(std::shared_ptr<Tensor>& weight, std::shared_ptr<Tensor>& bias,
                             std::shared_ptr<Tensor>& running_mean, std::shared_ptr<Tensor>& running_var,
                             int num_features, bool affine, bool track_running_stats) {
    if (affine) {
        weight = Tensor::create({num_features}, Device(DeviceType::CPU, 0), true);
        bias = Tensor::create({num_features}, Device(DeviceType::CPU, 0), true);
        std::fill(weight->data_ptr(), weight->data_ptr() + num_features, 1.0f);
        std::fill(bias->data_ptr(), bias->data_ptr() + num_features, 0.0f);
    }
    if (track_running_stats) {
        running_mean = Tensor::create({num_features}, Device(DeviceType::CPU, 0), false);
        running_var = Tensor::create({num_features}, Device(DeviceType::CPU, 0), false);
        std::fill(running_mean->data_ptr(), running_mean->data_ptr() + num_features, 0.0f);
        std::fill(running_var->data_ptr(), running_var->data_ptr() + num_features, 1.0f);
    }
}

static std::shared_ptr<Tensor> instance_norm_nd(std::shared_ptr<Tensor> input, int num_features,
                                                std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias,
                                                std::shared_ptr<Tensor> running_mean,
                                                std::shared_ptr<Tensor> running_var,
                                                float eps, float momentum, bool affine,
                                                bool track_running_stats, bool training) {
    int64_t N = input->shape[0], C = input->shape[1];
    if (C != num_features)
        throw std::runtime_error("[litetorch Error] InstanceNorm: channel mismatch");
    int64_t M = input->numel() / (N * C);
    auto flat = input->reshape({N * C, M});
    auto out = channel_norm(flat, N * C, M, C, true, weight, bias, running_mean, running_var,
                            eps, momentum, affine, track_running_stats, training);
    return out->reshape(input->shape);
}

InstanceNorm1d::InstanceNorm1d(int num_features, float eps, float momentum, bool affine,
                               bool track_running_stats)
    : num_features(num_features), eps(eps), momentum(momentum), affine(affine),
      track_running_stats(track_running_stats) {
    init_norm_params(weight, bias, running_mean, running_var, num_features, affine, track_running_stats);
    if (weight) register_parameter("weight", weight);
    if (bias) register_parameter("bias", bias);
    if (running_mean) register_buffer("running_mean", running_mean);
    if (running_var) register_buffer("running_var", running_var);
}

std::shared_ptr<Tensor> InstanceNorm1d::forward(std::shared_ptr<Tensor> input) {
    if (input->shape.size() != 3)
        throw std::runtime_error("[litetorch Error] InstanceNorm1d: expected 3D input");
    return instance_norm_nd(input, num_features, weight, bias, running_mean, running_var,
                            eps, momentum, affine, track_running_stats, training);
}

std::vector<std::shared_ptr<Tensor>> InstanceNorm1d::parameters() { return Module::parameters(); }

void InstanceNorm1d::to(const Device& device) {
    if (weight) weight = weight->to(device);
    if (bias) bias = bias->to(device);
    if (running_mean) running_mean = running_mean->to(device);
    if (running_var) running_var = running_var->to(device);
}

InstanceNorm2d::InstanceNorm2d(int num_features, float eps, float momentum, bool affine,
                               bool track_running_stats)
    : num_features(num_features), eps(eps), momentum(momentum), affine(affine),
      track_running_stats(track_running_stats) {
    init_norm_params(weight, bias, running_mean, running_var, num_features, affine, track_running_stats);
    if (weight) register_parameter("weight", weight);
    if (bias) register_parameter("bias", bias);
    if (running_mean) register_buffer("running_mean", running_mean);
    if (running_var) register_buffer("running_var", running_var);
}

std::shared_ptr<Tensor> InstanceNorm2d::forward(std::shared_ptr<Tensor> input) {
    if (input->shape.size() != 4)
        throw std::runtime_error("[litetorch Error] InstanceNorm2d: expected 4D input");
    return instance_norm_nd(input, num_features, weight, bias, running_mean, running_var,
                            eps, momentum, affine, track_running_stats, training);
}

std::vector<std::shared_ptr<Tensor>> InstanceNorm2d::parameters() { return Module::parameters(); }

void InstanceNorm2d::to(const Device& device) {
    if (weight) weight = weight->to(device);
    if (bias) bias = bias->to(device);
    if (running_mean) running_mean = running_mean->to(device);
    if (running_var) running_var = running_var->to(device);
}

InstanceNorm3d::InstanceNorm3d(int num_features, float eps, float momentum, bool affine,
                               bool track_running_stats)
    : num_features(num_features), eps(eps), momentum(momentum), affine(affine),
      track_running_stats(track_running_stats) {
    init_norm_params(weight, bias, running_mean, running_var, num_features, affine, track_running_stats);
    if (weight) register_parameter("weight", weight);
    if (bias) register_parameter("bias", bias);
    if (running_mean) register_buffer("running_mean", running_mean);
    if (running_var) register_buffer("running_var", running_var);
}

std::shared_ptr<Tensor> InstanceNorm3d::forward(std::shared_ptr<Tensor> input) {
    if (input->shape.size() != 5)
        throw std::runtime_error("[litetorch Error] InstanceNorm3d: expected 5D input");
    return instance_norm_nd(input, num_features, weight, bias, running_mean, running_var,
                            eps, momentum, affine, track_running_stats, training);
}

std::vector<std::shared_ptr<Tensor>> InstanceNorm3d::parameters() { return Module::parameters(); }

void InstanceNorm3d::to(const Device& device) {
    if (weight) weight = weight->to(device);
    if (bias) bias = bias->to(device);
    if (running_mean) running_mean = running_mean->to(device);
    if (running_var) running_var = running_var->to(device);
}

static std::shared_ptr<Tensor> batch_channel_norm(std::shared_ptr<Tensor> input, int num_features,
                                                   std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias,
                                                   std::shared_ptr<Tensor> running_mean,
                                                   std::shared_ptr<Tensor> running_var,
                                                   float eps, float momentum, bool affine,
                                                   bool track_running_stats, bool training) {
    int64_t ndim = (int64_t)input->shape.size();
    if (ndim < 3 || input->shape[1] != num_features)
        throw std::runtime_error("[litetorch Error] channel BatchNorm: bad input rank or channels");
    std::vector<int64_t> perm = {1, 0};
    for (int64_t d = 2; d < ndim; ++d) perm.push_back(d);
    std::vector<int64_t> inv(ndim, 0);
    for (int64_t d = 0; d < ndim; ++d) inv[perm[d]] = d;
    int64_t C = input->shape[1];
    int64_t M = input->numel() / C;
    auto flat = Ops::permute(input, perm)->reshape({C, M});
    auto out = channel_norm(flat, C, M, C, false, weight, bias, running_mean, running_var,
                            eps, momentum, affine, track_running_stats, training);
    std::vector<int64_t> ps(ndim, 0);
    for (int64_t d = 0; d < ndim; ++d) ps[d] = input->shape[perm[d]];
    return Ops::permute(out->reshape(ps), inv);
}

BatchNorm3d::BatchNorm3d(int num_features, float eps, float momentum, bool affine,
                         bool track_running_stats)
    : num_features(num_features), eps(eps), momentum(momentum), affine(affine),
      track_running_stats(track_running_stats) {
    init_norm_params(weight, bias, running_mean, running_var, num_features, affine, track_running_stats);
    if (weight) register_parameter("weight", weight);
    if (bias) register_parameter("bias", bias);
    if (running_mean) register_buffer("running_mean", running_mean);
    if (running_var) register_buffer("running_var", running_var);
}

std::shared_ptr<Tensor> BatchNorm3d::forward(std::shared_ptr<Tensor> input) {
    if (input->shape.size() != 5)
        throw std::runtime_error("[litetorch Error] BatchNorm3d: expected 5D input");
    std::vector<int64_t> perm = {1, 0, 2, 3, 4};
    int64_t C = input->shape[1];
    int64_t M = input->numel() / C;
    auto flat = Ops::permute(input, perm)->reshape({C, M});
    auto out = channel_norm(flat, C, M, C, false, weight, bias, running_mean, running_var,
                            eps, momentum, affine, track_running_stats, training);
    std::vector<int64_t> ps = {C, input->shape[0], input->shape[2], input->shape[3], input->shape[4]};
    return Ops::permute(out->reshape(ps), std::vector<int64_t>{1, 0, 2, 3, 4});
}

std::vector<std::shared_ptr<Tensor>> BatchNorm3d::parameters() { return Module::parameters(); }

void BatchNorm3d::to(const Device& device) {
    if (weight) weight = weight->to(device);
    if (bias) bias = bias->to(device);
    if (running_mean) running_mean = running_mean->to(device);
    if (running_var) running_var = running_var->to(device);
}

SyncBatchNorm::SyncBatchNorm(int num_features, float eps, float momentum, bool affine,
                             bool track_running_stats)
    : num_features(num_features), eps(eps), momentum(momentum), affine(affine),
      track_running_stats(track_running_stats) {
    init_norm_params(weight, bias, running_mean, running_var, num_features, affine, track_running_stats);
    if (weight) register_parameter("weight", weight);
    if (bias) register_parameter("bias", bias);
    if (running_mean) register_buffer("running_mean", running_mean);
    if (running_var) register_buffer("running_var", running_var);
}

std::shared_ptr<Tensor> SyncBatchNorm::forward(std::shared_ptr<Tensor> input) {
    return batch_channel_norm(input, num_features, weight, bias, running_mean, running_var,
                              eps, momentum, affine, track_running_stats, training);
}

std::vector<std::shared_ptr<Tensor>> SyncBatchNorm::parameters() { return Module::parameters(); }

void SyncBatchNorm::to(const Device& device) {
    if (weight) weight = weight->to(device);
    if (bias) bias = bias->to(device);
    if (running_mean) running_mean = running_mean->to(device);
    if (running_var) running_var = running_var->to(device);
}

ELU::ELU(float alpha) : alpha(alpha) {}

std::shared_ptr<Tensor> ELU::forward(std::shared_ptr<Tensor> input) {
    auto zero = Tensor::zeros({1}, input->device, false, input->dtype);
    auto cond = Ops::ge(input, zero);
    auto at = Tensor::create({1}, input->device, false, input->dtype);
    at->data_ptr()[0] = alpha;
    auto one = Tensor::create({1}, input->device, false, input->dtype);
    one->data_ptr()[0] = 1.0f;
    auto neg = Ops::mul(at, Ops::sub(Ops::exp(input), one));
    return Ops::where(cond, input, neg);
}

PReLU::PReLU(int num_parameters, float init) : num_parameters(num_parameters) {
    weight = Tensor::create({num_parameters}, Device(DeviceType::CPU, 0), true);
    std::fill(weight->data_ptr(), weight->data_ptr() + num_parameters, init);
    register_parameter("weight", weight);
}

std::shared_ptr<Tensor> PReLU::forward(std::shared_ptr<Tensor> input) {
    int64_t ndim = (int64_t)input->shape.size();
    if (num_parameters != 1 && num_parameters != input->shape[1])
        throw std::runtime_error("[litetorch Error] PReLU: num_parameters must be 1 or num channels");
    std::vector<int64_t> ws(ndim, 1);
    ws[1] = num_parameters;
    auto w = weight->reshape(ws);
    auto pos = Ops::relu(input);
    auto neg = Ops::mul(w, Ops::sub(input, pos));
    return Ops::add(pos, neg);
}

std::vector<std::shared_ptr<Tensor>> PReLU::parameters() { return Module::parameters(); }

void PReLU::to(const Device& device) {
    weight = weight->to(device);
}

GLU::GLU(int64_t dim) : dim(dim) {}

std::shared_ptr<Tensor> GLU::forward(std::shared_ptr<Tensor> input) {
    int64_t ndim = (int64_t)input->shape.size();
    int64_t d = dim < 0 ? dim + ndim : dim;
    if (d < 0 || d >= ndim)
        throw std::runtime_error("[litetorch Error] GLU: dim out of range");
    int64_t half = input->shape[d] / 2;
    if (input->shape[d] % 2 != 0)
        throw std::runtime_error("[litetorch Error] GLU: halving dimension must be even");
    auto a = Ops::narrow(input, d, 0, half);
    auto b = Ops::narrow(input, d, half, half);
    return Ops::mul(a, Ops::sigmoid(b));
}

PixelShuffle::PixelShuffle(int upscale_factor) : upscale_factor(upscale_factor) {
    if (upscale_factor < 1)
        throw std::runtime_error("[litetorch Error] PixelShuffle: upscale_factor must be >= 1");
}

std::shared_ptr<Tensor> PixelShuffle::forward(std::shared_ptr<Tensor> input) {
    if (input->shape.size() != 4)
        throw std::runtime_error("[litetorch Error] PixelShuffle: expected 4D input");
    int64_t r = upscale_factor;
    int64_t N = input->shape[0], C = input->shape[1] / (r * r), H = input->shape[2], W = input->shape[3];
    if (input->shape[1] % (r * r) != 0)
        throw std::runtime_error("[litetorch Error] PixelShuffle: channels not divisible by r^2");
    auto x = input->reshape({N, C, r, r, H, W});
    x = Ops::permute(x, {0, 1, 4, 2, 5, 3});
    return x->reshape({N, C, H * r, W * r});
}

PixelUnshuffle::PixelUnshuffle(int downscale_factor) : downscale_factor(downscale_factor) {
    if (downscale_factor < 1)
        throw std::runtime_error("[litetorch Error] PixelUnshuffle: downscale_factor must be >= 1");
}

std::shared_ptr<Tensor> PixelUnshuffle::forward(std::shared_ptr<Tensor> input) {
    if (input->shape.size() != 4)
        throw std::runtime_error("[litetorch Error] PixelUnshuffle: expected 4D input");
    int64_t r = downscale_factor;
    int64_t N = input->shape[0], C = input->shape[1], H = input->shape[2] / r, W = input->shape[3] / r;
    if (input->shape[2] % r != 0 || input->shape[3] % r != 0)
        throw std::runtime_error("[litetorch Error] PixelUnshuffle: spatial dims not divisible by r");
    auto x = input->reshape({N, C, H, r, W, r});
    x = Ops::permute(x, {0, 1, 3, 5, 2, 4});
    return x->reshape({N, C * r * r, H, W});
}

std::shared_ptr<Tensor> Identity::forward(std::shared_ptr<Tensor> input) {
    return input;
}

Unflatten::Unflatten(int64_t dim, const std::vector<int64_t>& unflattened_size)
    : dim(dim), unflattened_size(unflattened_size) {}

std::shared_ptr<Tensor> Unflatten::forward(std::shared_ptr<Tensor> input) {
    int64_t ndim = (int64_t)input->shape.size();
    int64_t d = dim < 0 ? dim + ndim : dim;
    if (d < 0 || d >= ndim)
        throw std::runtime_error("[litetorch Error] Unflatten: dim out of range");
    int64_t want = prod_dims(unflattened_size);
    if (want != input->shape[d])
        throw std::runtime_error("[litetorch Error] Unflatten: unflattened size mismatch");
    std::vector<int64_t> ns;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == d) {
            for (auto s : unflattened_size) ns.push_back(s);
        } else {
            ns.push_back(input->shape[i]);
        }
    }
    return input->reshape(ns);
}

}
}
