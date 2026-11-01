#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/jit.h"
#include <iostream>
#include <cmath>
#include <string>

using namespace litetorch;

static int failures = 0;

static void check(bool ok, const std::string& name) {
    std::cout << (ok ? "PASS " : "FAIL ") << name << std::endl;
    if (!ok) failures++;
}

template <typename F>
static bool throws_with(F&& fn, const std::string& needle) {
    try {
        fn();
    } catch (const std::runtime_error& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
}

int main() {
    auto cpu = Device(DeviceType::CPU, 0);

    auto x1 = Tensor::from_vector({2.0f}, {1}, cpu);
    auto x3 = Tensor::from_vector({1.0f, -2.0f, 3.0f}, {3}, cpu);
    auto y3 = Tensor::from_vector({0.5f, 1.0f, 2.0f}, {3}, cpu);

    check(throws_with([&] {
        Tracer::trace({x1}, [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            if (inputs[0]->item() > 0) {
                return Ops::relu(inputs[0]);
            }
            return Ops::mul(inputs[0], inputs[0]);
        }, "branch_on_item");
    }, "concretization"), "1a item() inside trace throws concretization error");

    check(throws_with([&] {
        Tracer::trace({x3}, [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            auto v = inputs[0]->to_vector();
            (void)v;
            return Ops::relu(inputs[0]);
        }, "to_vector_in_trace");
    }, "concretization"), "1a to_vector() inside trace throws concretization error");

    check(throws_with([&] {
        Tracer::trace({x3}, [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            (void)inputs;
            return Tensor::randn({1}, Device(DeviceType::CPU, 0), false);
        }, "scalar_randn");
    }, "RNG"), "1c scalar randn during trace throws RNG error");

    check(throws_with([&] {
        Tracer::trace({x3}, [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            (void)inputs;
            return Tensor::randn({3}, Device(DeviceType::CPU, 0), false);
        }, "vec_randn");
    }, "RNG"), "1c multi-element randn during trace throws RNG error");

    check(throws_with([&] {
        Tracer::trace({x3}, [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            (void)inputs;
            manual_seed(123);
            return Ops::relu(inputs[0]);
        }, "reseed");
    }, "RNG"), "1c manual_seed during trace throws RNG error");

    bool clean_ok = false;
    try {
        auto fn = Tracer::trace({x3, y3}, [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            return Ops::relu(Ops::add(inputs[0], inputs[1]));
        }, "clean");
        auto eager = Ops::relu(Ops::add(x3, y3));
        auto out = (*fn)({x3, y3});
        auto ve = eager->to_vector();
        auto vo = out->to_vector();
        clean_ok = ve.size() == vo.size();
        for (size_t i = 0; i < ve.size() && clean_ok; ++i) {
            if (std::abs(ve[i] - vo[i]) >= 1e-5f) clean_ok = false;
        }
    } catch (...) {
        clean_ok = false;
    }
    check(clean_ok, "clean trace still works and matches eager");

    bool const_read_ok = false;
    try {
        auto c = Tensor::from_vector({2.0f}, {1}, cpu, false);
        auto fn = Tracer::trace({x3}, [c](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            float s = c->item();
            (void)s;
            return Ops::relu(inputs[0]);
        }, "const_read");
        (void)fn;
        const_read_ok = true;
    } catch (...) {
        const_read_ok = false;
    }
    check(const_read_ok, "requires_grad=false constant item() inside trace allowed");

    bool restore_ok = false;
    try {
        auto x = Tensor::from_vector({1.0f}, {1}, cpu, false);
        try {
            Tracer::trace({x}, [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
                (void)inputs[0]->item();
                return Ops::relu(inputs[0]);
            }, "throwing_trace");
        } catch (const std::runtime_error&) {
        }
        auto xg = Tensor::from_vector({1.0f}, {1}, cpu, true);
        auto fn = Tracer::trace({xg}, [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            return Ops::relu(inputs[0]);
        }, "clean_grad");
        (void)fn;
        restore_ok = !x->requires_grad && xg->requires_grad;
    } catch (...) {
        restore_ok = false;
    }
    check(restore_ok, "requires_grad restored after throwing and clean traces");

    bool rng_untouched_ok = false;
    try {
        std::string before = get_rng_state();
        auto fn = Tracer::trace({x3, y3}, [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            return Ops::relu(Ops::add(inputs[0], inputs[1]));
        }, "no_rng");
        (void)fn;
        rng_untouched_ok = (get_rng_state() == before);
    } catch (...) {
        rng_untouched_ok = false;
    }
    check(rng_untouched_ok, "clean trace leaves RNG state unchanged");

    bool const_bake_ok = false;
    try {
        auto k = Tensor::from_vector({2.0f}, {1}, cpu, false);
        auto fn = Tracer::trace({x1}, [k](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            return Ops::mul(inputs[0], k);
        }, "const_bake");
        auto out = (*fn)({x1});
        auto eager = Ops::mul(x1, k);
        const_bake_ok = std::abs(out->to_vector()[0] - eager->to_vector()[0]) < 1e-5f;
    } catch (...) {
        const_bake_ok = false;
    }
    check(const_bake_ok, "numel==1 non-grad leaf still bakes as CONST after trace walk");

    bool nested_ok = false;
    try {
        auto fn = Tracer::trace({x3}, [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            auto inner = Tracer::trace({inputs[0]}, [](const std::vector<std::shared_ptr<Tensor>>& in2) {
                return Ops::relu(in2[0]);
            }, "inner");
            (void)inner;
            return Ops::relu(inputs[0]);
        }, "outer");
        (void)fn;
        nested_ok = true;
    } catch (...) {
        nested_ok = false;
    }
    check(nested_ok, "nested trace does not corrupt tracing flag");

    bool outside_ok = false;
    try {
        float v = x1->item();
        auto vec = x3->to_vector();
        outside_ok = (v == 2.0f) && (vec.size() == 3);
    } catch (...) {
        outside_ok = false;
    }
    check(outside_ok, "item()/to_vector() unaffected outside trace");

    bool walk_throw_restore_ok = false;
    try {
        auto x = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, false);
        bool threw = false;
        try {
            Tracer::trace({x}, [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
                return Ops::softmax(inputs[0], -1);
            }, "unsupported_op");
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find("Unsupported operator") != std::string::npos;
        }
        walk_throw_restore_ok = threw && !x->requires_grad;
    } catch (...) {
        walk_throw_restore_ok = false;
    }
    check(walk_throw_restore_ok, "requires_grad restored after walk-phase throw (unsupported op)");

    if (failures == 0) {
        std::cout << "ALL_PASS" << std::endl;
        return 0;
    }
    std::cout << "FAILURES: " << failures << std::endl;
    return 1;
}
