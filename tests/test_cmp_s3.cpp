#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/jit.h"
#include "litetorch/backend.h"
#include <iostream>
#include <cmath>
#include <string>

using namespace litetorch;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const std::string& name) {
    if (ok) {
        std::cout << "PASS " << name << std::endl;
        g_pass++;
    } else {
        std::cout << "FAIL " << name << std::endl;
        g_fail++;
    }
}

static bool near_vec(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

static bool same_tensor(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, float tol) {
    if (a->shape != b->shape) return false;
    return near_vec(a->to_vector(), b->to_vector(), tol);
}

int main() {
    auto cpu = Device(DeviceType::CPU, 0);

    auto x = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
    auto y = std::make_shared<JITVar>(JITVar::OpType::INPUT, "y");
    auto z = std::make_shared<JITVar>(JITVar::OpType::INPUT, "z");

    {
        JITFunction add_fn("add", x + y, {x, y});
        auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, cpu);
        auto b = Tensor::from_vector({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, {2, 3}, cpu);
        auto o1 = add_fn({a, b});
        check(add_fn.shape_cache_size() == 1 && add_fn.shape_cache_hits() == 0, "cache miss creates one entry");
        auto o2 = add_fn({a, b});
        check(add_fn.shape_cache_size() == 1 && add_fn.shape_cache_hits() == 1, "cache hit on repeat signature");
        check(same_tensor(o1, o2, 0.0f), "cache hit returns identical values");
        auto c = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, cpu);
        auto d = Tensor::from_vector({5.0f, 6.0f, 7.0f, 8.0f}, {4}, cpu);
        add_fn({c, d});
        check(add_fn.shape_cache_size() == 2 && add_fn.shape_cache_hits() == 1, "cache new shape adds entry");
        add_fn({a, b});
        check(add_fn.shape_cache_size() == 2 && add_fn.shape_cache_hits() == 2, "cache interleave hit keeps size");
    }

    {
        JITFunction add_fn("add", x + y, {x, y});
        auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, cpu);
        auto b = Tensor::from_vector({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, {2, 3}, cpu);
        add_fn({a, b});
        auto a64 = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, cpu, false, DataType::FP64);
        auto b64 = Tensor::from_vector({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, {2, 3}, cpu, false, DataType::FP64);
        bool threw = false;
        try {
            add_fn({a64, b64});
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "dtype change call throws on non-FP32 eval");
        check(add_fn.shape_cache_size() == 2, "dtype change creates distinct cache entry");
    }

    {
        JITFunction add_fn("add", x + y, {x, y});
        auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, cpu);
        auto b = Tensor::from_vector({10.0f, 20.0f, 30.0f}, {3}, cpu);
        auto out = add_fn({a, b});
        auto ref = Ops::add(a, b);
        check(out->shape == std::vector<int64_t>({2, 3}), "broadcast add output shape");
        check(same_tensor(out, ref, 1e-5f), "broadcast add values match eager");
    }

    {
        JITFunction mul_fn("mul", x * y, {x, y});
        auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 1, 4}, cpu);
        auto b = Tensor::from_vector({2.0f, 3.0f, 4.0f}, {3, 1}, cpu);
        auto out = mul_fn({a, b});
        auto ref = Ops::mul(a, b);
        check(out->shape == std::vector<int64_t>({2, 3, 4}), "broadcast mul output shape");
        check(same_tensor(out, ref, 1e-5f), "broadcast mul values match eager");
    }

    {
        JITFunction add_fn("add", x + y, {x, y});
        std::vector<float> xv(24);
        for (int i = 0; i < 24; ++i) xv[i] = static_cast<float>(i + 1);
        auto a = Tensor::from_vector(xv, {2, 3, 2, 2}, cpu);
        auto b = Tensor::from_vector({0.5f, 1.5f, 2.5f}, {1, 3, 1, 1}, cpu);
        auto out = add_fn({a, b});
        auto ref = Ops::add(a, b);
        check(out->shape == std::vector<int64_t>({2, 3, 2, 2}), "broadcast bias output shape");
        check(same_tensor(out, ref, 1e-5f), "broadcast bias values match eager");
    }

    {
        JITFunction add_fn("add", x + y, {x, y});
        auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, cpu);
        auto s = Tensor::from_vector({5.0f}, {}, cpu);
        auto out = add_fn({a, s});
        auto ref = Ops::add(a, s);
        check(out->shape == std::vector<int64_t>({2, 2}), "broadcast scalar input output shape");
        check(same_tensor(out, ref, 1e-5f), "broadcast scalar input values match eager");
    }

    {
        auto expr = JIT::gelu(x * y + z);
        JITFunction fn("gelu_add", expr, {x, y, z});
        auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, cpu);
        auto b = Tensor::from_vector({0.5f, 1.5f, 2.0f}, {3}, cpu);
        auto c = Tensor::from_vector({1.0f, 0.0f, -1.0f}, {1, 3}, cpu);
        auto out = fn({a, b, c});
        auto ref = Ops::gelu(Ops::add(Ops::mul(a, b), c));
        check(out->shape == std::vector<int64_t>({2, 3}), "broadcast composite output shape");
        check(same_tensor(out, ref, 1e-5f), "broadcast composite values match eager");
    }

    {
        JITFunction add_fn("add", x + y, {x, y});
        auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, cpu);
        auto b = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, cpu);
        bool threw = false;
        try {
            add_fn({a, b});
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find("broadcast") != std::string::npos;
        }
        check(threw, "incompatible shapes throw broadcast error");
        auto c = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2}, cpu);
        threw = false;
        try {
            add_fn({a, c});
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find("broadcast") != std::string::npos;
        }
        check(threw, "trailing mismatch throws broadcast error");
    }

    {
        JITFunction add_fn("add", x + y, {x, y});
        auto base = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2}, cpu);
        auto a = base->transpose(0, 1);
        check(!a->is_contiguous(), "transposed input is non-contiguous");
        auto b = Tensor::from_vector({10.0f, 20.0f, 30.0f}, {3}, cpu);
        auto out = add_fn({a, b});
        auto ref = Ops::add(a, b);
        check(same_tensor(out, ref, 1e-5f), "broadcast non-contiguous input matches eager");
    }

    {
        JITFunction add_fn("add", x + y, {x, y});
        auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, cpu, true);
        auto b = Tensor::from_vector({10.0f, 20.0f, 30.0f}, {3}, cpu, true);
        auto out = add_fn({a, b});
        auto gout = Tensor::ones({2, 3}, cpu);
        out->backward(gout);
        check(a->grad && a->grad->shape == std::vector<int64_t>({2, 3}), "broadcast grad input shape preserved");
        check(b->grad && b->grad->shape == std::vector<int64_t>({3}), "broadcast grad reduced to input shape");
        bool ok = a->grad && near_vec(a->grad->to_vector(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, 1e-5f);
        ok = ok && b->grad && near_vec(b->grad->to_vector(), {2.0f, 2.0f, 2.0f}, 1e-5f);
        check(ok, "broadcast grad values correct");
    }

    {
        auto expr = JIT::gelu(x * y + z);
        JITFunction fn("gelu_add", expr, {x, y, z});
        auto a = Tensor::from_vector({1.0f, 2.0f, -3.0f, 4.0f}, {4}, cpu);
        auto b = Tensor::from_vector({0.5f, 1.5f, 2.0f, -1.0f}, {4}, cpu);
        auto c = Tensor::from_vector({1.0f, -1.0f, 0.0f, 2.0f}, {4}, cpu);
        auto out = fn({a, b, c});
        auto ref = Ops::gelu(Ops::add(Ops::mul(a, b), c));
        check(same_tensor(out, ref, 1e-5f), "identical shapes regression matches eager");
    }

    auto active_backend = BackendDispatcher::get().get_backend();
    if (active_backend && active_backend->is_available()) {
        auto dev_gpu = Device(DeviceType::GPU, 0);
        JITFunction add_fn("add", x + y, {x, y});
        auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, cpu);
        auto b = Tensor::from_vector({10.0f, 20.0f, 30.0f}, {3}, cpu);
        auto ag = a->to(dev_gpu);
        auto bg = b->to(dev_gpu);
        auto out = add_fn({ag, bg});
        auto ref = Ops::add(ag, bg);
        auto ov = out->to_vector();
        auto rv = ref->to_vector();
        check(out->shape == std::vector<int64_t>({2, 3}) && near_vec(ov, rv, 1e-5f), "gpu broadcast add matches eager");
    } else {
        std::cout << "SKIP gpu broadcast add (no GPU backend)" << std::endl;
    }

    std::cout << "RESULT " << g_pass << " passed, " << g_fail << " failed" << std::endl;
    return g_fail == 0 ? 0 : 1;
}
