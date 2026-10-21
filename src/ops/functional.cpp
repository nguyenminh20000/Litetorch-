#include "litetorch/functional.h"
#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
#include "litetorch/backend_route.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "litetorch/storage_guard.h"
namespace litetorch {

extern std::mt19937& get_generator();
extern const std::string litetorch_kernels_src;

namespace functional {
namespace {

void require_cpu(const Tensor& t, const char* op) {
    if (t.device.type != DeviceType::CPU) {
        throw std::runtime_error(std::string("[litetorch Error] functional::") + op +
                                 " has no GPU/TPU kernel yet (CPU-only)");
    }
}

std::shared_ptr<Tensor> upcast_fp32(std::shared_ptr<Tensor> t) {
    if (t->dtype == DataType::FP32) return t;
    return Ops::cast(t, DataType::FP32);
}

int64_t norm_dim(int64_t dim, int64_t ndim, const char* op) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error(std::string("[litetorch Error] functional::") + op +
                                 ": dimension out of range");
    }
    return dim;
}

class EluNode : public Node {
public:
    double alpha;
    EluNode(double alpha) : Node("ELU"), alpha(alpha) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto input = saved_tensors[0];
        auto in_c = input->is_contiguous() ? input : input->contiguous();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto grad_input = Tensor::create(in_c->shape, in_c->device, false, in_c->dtype);
        StorageUseGuard guard({in_c->storage, gout_c->storage, grad_input->storage});
        size_t n = grad_input->numel();
        float* in_ptr = in_c->data_ptr();
        float* g_out = gout_c->data_ptr();
        float* g_in = grad_input->data_ptr();
        float a = static_cast<float>(alpha);
        ThreadPool::get().parallel_for(0, static_cast<int64_t>(n), [&](int64_t i) {
            float d = in_ptr[i] > 0.0f ? 1.0f : a * expf(in_ptr[i]);
            g_in[i] = g_out[i] * d;
        });
        return { grad_input };
    }
};

class PadNode : public Node {
public:
    std::vector<int64_t> input_shape;
    std::vector<int64_t> offsets;
    PadNode(const std::vector<int64_t>& s, const std::vector<int64_t>& o)
        : Node("Pad"), input_shape(s), offsets(o) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto grad_input = Tensor::create(input_shape, gout_c->device, false, gout_c->dtype);
        StorageUseGuard guard({gout_c->storage, grad_input->storage});
        int64_t ndim = static_cast<int64_t>(input_shape.size());
        std::vector<int64_t> in_strides = default_strides(input_shape);
        std::vector<int64_t> gout_strides = default_strides(gout_c->shape);
        size_t n = grad_input->numel();
        float* g_out = gout_c->data_ptr();
        float* g_in = grad_input->data_ptr();
        ThreadPool::get().parallel_for(0, static_cast<int64_t>(n), [&](int64_t flat) {
            int64_t rem = flat;
            int64_t gout_flat = 0;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                int64_t c = rem % input_shape[d];
                rem /= input_shape[d];
                gout_flat += (c + offsets[d]) * gout_strides[d];
            }
            g_in[flat] = g_out[gout_flat];
        });
        return { grad_input };
    }
};

class InterpolateNearestNode : public Node {
public:
    std::vector<int64_t> input_shape;
    InterpolateNearestNode(const std::vector<int64_t>& s) : Node("InterpolateNearest"), input_shape(s) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        std::unique_ptr<NoGradGuard> ng;
        if (!Autograd::is_create_graph_) ng = std::make_unique<NoGradGuard>();
        auto gout_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto grad_input = Tensor::create(input_shape, gout_c->device, false, gout_c->dtype);
        StorageUseGuard guard({gout_c->storage, grad_input->storage});
        std::memset(grad_input->data_ptr(), 0, grad_input->numel() * sizeof(float));
        int64_t ndim = static_cast<int64_t>(input_shape.size());
        int64_t ns = ndim - 2;
        const auto& out_shape = gout_c->shape;
        std::vector<int64_t> in_strides = default_strides(input_shape);
        std::vector<int64_t> out_strides = default_strides(out_shape);
        int64_t outer = input_shape[0] * input_shape[1];
        int64_t out_spatial = 1;
        for (int64_t d = 0; d < ns; ++d) out_spatial *= out_shape[2 + d];
        float* g_out = gout_c->data_ptr();
        float* g_in = grad_input->data_ptr();
        for (int64_t o = 0; o < outer; ++o) {
            int64_t in_base = (o / input_shape[1]) * in_strides[0] +
                              (o % input_shape[1]) * in_strides[1];
            int64_t out_base = o * out_spatial;
            for (int64_t f = 0; f < out_spatial; ++f) {
                int64_t rem = f;
                int64_t in_off = 0;
                for (int64_t d = ns - 1; d >= 0; --d) {
                    int64_t oc = rem % out_shape[2 + d];
                    rem /= out_shape[2 + d];
                    int64_t ic = std::min<int64_t>(
                        static_cast<int64_t>(oc * (static_cast<double>(input_shape[2 + d]) /
                                                  static_cast<double>(out_shape[2 + d]))),
                        input_shape[2 + d] - 1);
                    in_off += ic * in_strides[2 + d];
                }
                g_in[in_base + in_off] += g_out[out_base + f];
            }
        }
        return { grad_input };
    }
};

void attach_unary(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> out,
                  std::shared_ptr<Node> node,
                  const std::vector<std::shared_ptr<Tensor>>& saved) {
    if (Autograd::is_grad_enabled() && input->requires_grad) {
        node->inputs = { {input, true} };
        node->next_nodes = { input->creator };
        node->saved_tensors = saved;
        node->saved_versions.clear();
        for (auto& t : saved) node->saved_versions.push_back(t ? t->_version() : 0);
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
}

}

std::shared_ptr<Tensor> dropout(std::shared_ptr<Tensor> x, double p, bool training) {
    if (p < 0.0 || p > 1.0) {
        throw std::runtime_error("[litetorch Error] functional::dropout: p must be in [0, 1]");
    }
    if (!training || p == 0.0) return x;
    if (p == 1.0) {
        auto zeros = Tensor::create(x->shape, Device(DeviceType::CPU, 0), false, x->dtype);
        std::memset(zeros->raw_data_ptr(), 0, zeros->numel() * zeros->storage->element_size());
        if (zeros->device != x->device) zeros = zeros->to(x->device);
        return Ops::mul(x, zeros);
    }
    float scale = 1.0f / (1.0f - static_cast<float>(p));
    std::shared_ptr<Tensor> mask;
    if (x->device.type == DeviceType::GPU) {
        mask = Tensor::create(x->shape, x->device);
        auto kernel = rt_gpu_get_kernel("litetorch_kernels", litetorch_kernels_src,
                                        "generate_dropout_mask");
        cl_mem mask_mem = mask->gpu_data();
        int mask_off = mask->offset;
        float pf = static_cast<float>(p);
        int size = static_cast<int>(mask->numel());
        std::mt19937& gen = get_generator();
        unsigned int seed = gen();
        rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                      {&mask_mem, &mask_off, &pf, &scale, &seed, &size},
                      {sizeof(cl_mem), sizeof(int), sizeof(float),
                       sizeof(float), sizeof(unsigned int), sizeof(int)});
    } else {
        std::mt19937& gen = get_generator();
        std::bernoulli_distribution dis(1.0 - p);
        mask = Tensor::create(x->shape, Device(DeviceType::CPU, 0), false);
        float* m_ptr = mask->data_ptr();
        size_t n = mask->numel();
        for (size_t i = 0; i < n; ++i) {
            m_ptr[i] = dis(gen) ? scale : 0.0f;
        }
    }
    if (mask->dtype != x->dtype) {
        mask = mask->cast(x->dtype);
    }
    return Ops::mul(x, mask);
}

std::shared_ptr<Tensor> linear(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> weight,
                               std::shared_ptr<Tensor> bias) {
    if (weight->shape.size() != 2) {
        throw std::runtime_error("[litetorch Error] functional::linear: weight must be 2D");
    }
    if (x->shape.empty() || x->shape.back() != weight->shape[1]) {
        throw std::runtime_error("[litetorch Error] functional::linear: shape mismatch");
    }
    if (bias && bias->shape.size() == 1 && bias->shape[0] != weight->shape[0]) {
        throw std::runtime_error("[litetorch Error] functional::linear: bias shape mismatch");
    }
    auto wt = Ops::permute(weight, {1, 0});
    if (x->shape.size() == 2) {
        return Ops::matmul_bias(x, wt, bias);
    }
    int64_t in_f = weight->shape[1];
    int64_t out_f = weight->shape[0];
    int64_t flat = 1;
    for (size_t i = 0; i + 1 < x->shape.size(); ++i) flat *= x->shape[i];
    auto x2 = x->reshape({flat, in_f});
    auto out2 = Ops::matmul_bias(x2, wt, bias);
    std::vector<int64_t> out_shape(x->shape.begin(), x->shape.end() - 1);
    out_shape.push_back(out_f);
    return out2->reshape(out_shape);
}

std::shared_ptr<Tensor> pad(std::shared_ptr<Tensor> x, const std::vector<int64_t>& pad,
                            const std::string& mode, double value) {
    if (mode != "constant") {
        throw std::runtime_error("[litetorch Error] functional::pad: only mode='constant' is "
                                 "supported (got '" + mode + "')");
    }
    require_cpu(*x, "pad");
    if (pad.size() % 2 != 0) {
        throw std::runtime_error("[litetorch Error] functional::pad: pad must have even length");
    }
    int64_t ndim = static_cast<int64_t>(x->shape.size());
    int64_t npad = static_cast<int64_t>(pad.size() / 2);
    if (npad > ndim) {
        throw std::runtime_error("[litetorch Error] functional::pad: too many pad values");
    }
    auto x32 = upcast_fp32(x->is_contiguous() ? x : x->contiguous());
    std::vector<int64_t> out_shape = x32->shape;
    std::vector<int64_t> offsets(ndim, 0);
    for (int64_t k = 0; k < npad; ++k) {
        int64_t d = ndim - 1 - k;
        int64_t left = pad[static_cast<size_t>(2 * k)];
        int64_t right = pad[static_cast<size_t>(2 * k + 1)];
        if (left < 0 || right < 0) {
            throw std::runtime_error("[litetorch Error] functional::pad: negative padding not "
                                     "supported");
        }
        out_shape[d] += left + right;
        offsets[d] = left;
    }
    auto out = Tensor::create(out_shape, x32->device, false, DataType::FP32);
    StorageUseGuard guard({x32->storage, out->storage});
    float v = static_cast<float>(value);
    float* dst = out->data_ptr();
    size_t on = out->numel();
    ThreadPool::get().parallel_for(0, static_cast<int64_t>(on), [&](int64_t i) {
        dst[i] = v;
    });
    float* src = x32->data_ptr();
    std::vector<int64_t> in_strides = default_strides(x32->shape);
    std::vector<int64_t> out_strides = default_strides(out_shape);
    size_t n = x32->numel();
    ThreadPool::get().parallel_for(0, static_cast<int64_t>(n), [&](int64_t flat) {
        int64_t rem = flat;
        int64_t oflat = 0;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t c = rem % x32->shape[d];
            rem /= x32->shape[d];
            oflat += (c + offsets[d]) * out_strides[d];
        }
        dst[oflat] = src[flat];
    });
    std::shared_ptr<Tensor> result = (x->dtype == DataType::FP32) ? out : Ops::cast(out, x->dtype);
    attach_unary(x, result, std::make_shared<PadNode>(x->shape, offsets), {x});
    return result;
}

std::shared_ptr<Tensor> interpolate(std::shared_ptr<Tensor> x,
                                    const std::vector<int64_t>& size,
                                    const std::vector<double>& scale_factor,
                                    const std::string& mode, bool align_corners) {
    if (mode != "nearest") {
        throw std::runtime_error("[litetorch Error] functional::interpolate: only mode='nearest' "
                                 "is supported (got '" + mode + "')");
    }
    require_cpu(*x, "interpolate");
    int64_t ndim = static_cast<int64_t>(x->shape.size());
    if (ndim < 3 || ndim > 5) {
        throw std::runtime_error("[litetorch Error] functional::interpolate: expected 3D/4D/5D "
                                 "input");
    }
    bool has_size = !size.empty();
    bool has_scale = !scale_factor.empty();
    if (has_size == has_scale) {
        throw std::runtime_error("[litetorch Error] functional::interpolate: exactly one of size "
                                 "or scale_factor must be given");
    }
    int64_t ns = ndim - 2;
    std::vector<int64_t> out_spatial(ns);
    if (has_size) {
        if (static_cast<int64_t>(size.size()) != ns) {
            throw std::runtime_error("[litetorch Error] functional::interpolate: size length must "
                                     "match spatial dims");
        }
        out_spatial = size;
    } else {
        if (scale_factor.size() != 1 && static_cast<int64_t>(scale_factor.size()) != ns) {
            throw std::runtime_error("[litetorch Error] functional::interpolate: scale_factor must "
                                     "be a scalar or match spatial dims");
        }
        for (int64_t d = 0; d < ns; ++d) {
            double s = scale_factor.size() == 1 ? scale_factor[0] : scale_factor[d];
            out_spatial[d] = static_cast<int64_t>(std::floor(x->shape[2 + d] * s));
            if (out_spatial[d] <= 0) {
                throw std::runtime_error("[litetorch Error] functional::interpolate: output size "
                                         "must be positive");
            }
        }
    }
    (void)align_corners;
    auto x32 = upcast_fp32(x->is_contiguous() ? x : x->contiguous());
    std::vector<int64_t> out_shape = {x32->shape[0], x32->shape[1]};
    out_shape.insert(out_shape.end(), out_spatial.begin(), out_spatial.end());
    auto out = Tensor::create(out_shape, x32->device, false, DataType::FP32);
    StorageUseGuard guard({x32->storage, out->storage});
    std::vector<int64_t> in_strides = default_strides(x32->shape);
    std::vector<int64_t> out_strides = default_strides(out_shape);
    int64_t outer = x32->shape[0] * x32->shape[1];
    int64_t in_spatial = 1;
    int64_t on_spatial = 1;
    for (int64_t d = 0; d < ns; ++d) {
        in_spatial *= x32->shape[2 + d];
        on_spatial *= out_spatial[d];
    }
    float* src = x32->data_ptr();
    float* dst = out->data_ptr();
    ThreadPool::get().parallel_for(0, outer, [&](int64_t o) {
        int64_t in_base = (o / x32->shape[1]) * in_strides[0] + (o % x32->shape[1]) * in_strides[1];
        int64_t out_base = (o / out_shape[1]) * out_strides[0] + (o % out_shape[1]) * out_strides[1];
        for (int64_t f = 0; f < on_spatial; ++f) {
            int64_t rem = f;
            int64_t in_off = 0;
            for (int64_t d = ns - 1; d >= 0; --d) {
                int64_t oc = rem % out_spatial[d];
                rem /= out_spatial[d];
                double scale = static_cast<double>(x32->shape[2 + d]) /
                               static_cast<double>(out_spatial[d]);
                int64_t ic = static_cast<int64_t>(oc * scale);
                if (ic >= x32->shape[2 + d]) ic = x32->shape[2 + d] - 1;
                in_off += ic * in_strides[2 + d];
            }
            dst[out_base + f] = src[in_base + in_off];
        }
    });
    std::shared_ptr<Tensor> result = (x->dtype == DataType::FP32) ? out : Ops::cast(out, x->dtype);
    attach_unary(x, result, std::make_shared<InterpolateNearestNode>(x->shape), {x});
    return result;
}

std::shared_ptr<Tensor> nll_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
                                 std::shared_ptr<Tensor> weight, const std::string& reduction,
                                 int64_t ignore_index) {
    return Ops::nll_loss(input, target, weight, ignore_index, reduction);
}

std::shared_ptr<Tensor> bce_with_logits(std::shared_ptr<Tensor> input,
                                        std::shared_ptr<Tensor> target,
                                        const std::string& reduction) {
    return Ops::bce_with_logits_loss(input, target, nullptr, reduction, nullptr);
}

std::shared_ptr<Tensor> normalize(std::shared_ptr<Tensor> x, double p, int64_t dim,
                               double eps) {
    int64_t ndim = static_cast<int64_t>(x->shape.size());
    dim = norm_dim(dim, ndim, "normalize");
    if (p <= 0.0) {
        throw std::runtime_error("[litetorch Error] functional::normalize: p must be positive");
    }
    float pf = static_cast<float>(p);
    auto denom = Ops::pow(Ops::sum(Ops::pow(Ops::abs(x), pf), dim, true), 1.0f / pf);
    denom = Ops::clamp_min(denom, static_cast<float>(eps));
    return Ops::div(x, denom);
}

std::shared_ptr<Tensor> glu(std::shared_ptr<Tensor> x, int64_t dim) {
    int64_t ndim = static_cast<int64_t>(x->shape.size());
    dim = norm_dim(dim, ndim, "glu");
    if (x->shape[dim] % 2 != 0) {
        throw std::runtime_error("[litetorch Error] functional::glu: dimension size must be even");
    }
    auto parts = Ops::chunk(x, 2, dim);
    return Ops::mul(parts[0], Ops::sigmoid(parts[1]));
}

std::shared_ptr<Tensor> elu(std::shared_ptr<Tensor> x, double alpha) {
    if (x->device.type == DeviceType::GPU) {
        throw std::runtime_error("[litetorch Error] functional::elu has no GPU kernel yet "
                                 "(CPU-only)");
    }
    auto xc = x->is_contiguous() ? x : x->contiguous();
    auto x32 = upcast_fp32(xc);
    auto out = Tensor::create(x32->shape, x32->device, false, DataType::FP32);
    StorageUseGuard guard({x32->storage, out->storage});
    float* src = x32->data_ptr();
    float* dst = out->data_ptr();
    size_t n = out->numel();
    float a = static_cast<float>(alpha);
    ThreadPool::get().parallel_for(0, static_cast<int64_t>(n), [&](int64_t i) {
        dst[i] = src[i] > 0.0f ? src[i] : a * (expf(src[i]) - 1.0f);
    });
    std::shared_ptr<Tensor> result = (x->dtype == DataType::FP32) ? out : Ops::cast(out, x->dtype);
    attach_unary(x, result, std::make_shared<EluNode>(alpha), {x32});
    return result;
}

}
}
