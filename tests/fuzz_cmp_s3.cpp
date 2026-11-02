#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/jit.h"
#include <iostream>
#include <cmath>
#include <random>
#include <vector>

using namespace litetorch;

static bool same_tensor(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, float tol) {
    if (a->shape != b->shape) return false;
    auto va = a->to_vector();
    auto vb = b->to_vector();
    if (va.size() != vb.size()) return false;
    for (size_t i = 0; i < va.size(); ++i) {
        if (std::abs(va[i] - vb[i]) > tol) return false;
    }
    return true;
}

static std::vector<int64_t> derive_shape(std::mt19937& rng, const std::vector<int64_t>& base) {
    std::vector<int64_t> out = base;
    int drops = std::uniform_int_distribution<int>(0, 2)(rng);
    for (int i = 0; i < drops && out.size() > 1; ++i) out.erase(out.begin());
    for (size_t i = 0; i < out.size(); ++i) {
        if (std::uniform_int_distribution<int>(0, 1)(rng) == 0) out[i] = 1;
    }
    return out;
}

static std::vector<float> rand_data(std::mt19937& rng, size_t n) {
    std::vector<float> d(n);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (size_t i = 0; i < n; ++i) d[i] = dist(rng);
    return d;
}

static size_t numel_of(const std::vector<int64_t>& s) {
    size_t n = 1;
    for (auto d : s) n *= static_cast<size_t>(d);
    return n;
}

int main() {
    auto cpu = Device(DeviceType::CPU, 0);
    auto x = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
    auto y = std::make_shared<JITVar>(JITVar::OpType::INPUT, "y");
    auto z = std::make_shared<JITVar>(JITVar::OpType::INPUT, "z");
    std::mt19937 rng(12345);
    int fails = 0;

    std::vector<std::vector<int64_t>> bases = {{2, 3, 4}, {3, 1, 5}, {4, 4}, {2, 3}, {5}};
    int zero_iters = 0;
    for (int iter = 0; iter < 200; ++iter) {
        auto base = bases[iter % bases.size()];
        auto sa = derive_shape(rng, base);
        auto sb = derive_shape(rng, base);
        auto sc = derive_shape(rng, base);
        std::vector<int> zero_pos;
        for (int t = 0; t < 3; ++t) {
            if (std::uniform_int_distribution<int>(0, 3)(rng) == 0) zero_pos.push_back(t);
        }
        auto apply_zeros = [&](std::vector<int64_t>& s) {
            for (size_t j = 0; j < s.size(); ++j) {
                int t = static_cast<int>(s.size()) - 1 - static_cast<int>(j);
                for (size_t k = 0; k < zero_pos.size(); ++k) {
                    if (zero_pos[k] == t) s[j] = 0;
                }
            }
        };
        apply_zeros(sa);
        apply_zeros(sb);
        apply_zeros(sc);
        bool has_zero = false;
        for (auto d : sa) if (d == 0) has_zero = true;
        for (auto d : sb) if (d == 0) has_zero = true;
        for (auto d : sc) if (d == 0) has_zero = true;
        if (has_zero) zero_iters++;
        auto a = Tensor::from_vector(rand_data(rng, numel_of(sa)), sa, cpu);
        auto b = Tensor::from_vector(rand_data(rng, numel_of(sb)), sb, cpu);
        auto c = Tensor::from_vector(rand_data(rng, numel_of(sc)), sc, cpu);
        int op = iter % 4;
        std::shared_ptr<Tensor> out;
        std::shared_ptr<Tensor> ref;
        if (op == 0) {
            JITFunction fn("f", x + y, {x, y});
            out = fn({a, b});
            ref = Ops::add(a, b);
        } else if (op == 1) {
            JITFunction fn("f", x * y, {x, y});
            out = fn({a, b});
            ref = Ops::mul(a, b);
        } else if (op == 2) {
            JITFunction fn("f", x - y, {x, y});
            out = fn({a, b});
            ref = Ops::sub(a, b);
        } else {
            JITFunction fn("f", JIT::relu(x * y + z), {x, y, z});
            out = fn({a, b, c});
            ref = Ops::relu(Ops::add(Ops::mul(a, b), c));
        }
        if (!same_tensor(out, ref, 1e-5f)) {
            std::cout << "FUZZ FAIL iter " << iter << " op " << op << std::endl;
            fails++;
        }
    }
    std::cout << "FUZZ RESULT " << (200 - fails) << "/200 passed" << std::endl;
    std::cout << "ZERO DIM ITERS " << zero_iters << std::endl;
    return fails == 0 ? 0 : 1;
}
