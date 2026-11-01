#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/jit.h"
#include "litetorch/autograd.h"
#include <iostream>
#include <cmath>
#include <string>
#include <functional>

using namespace litetorch;

static int failures = 0;

static void expect_throw(const std::string& label, const std::function<void()>& fn, const std::string& needle) {
    try {
        fn();
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        if (msg.find(needle) != std::string::npos) {
            std::cout << "PASS " << label << std::endl;
            return;
        }
        std::cout << "FAIL " << label << ": wrong message: " << msg << std::endl;
        ++failures;
        return;
    }
    std::cout << "FAIL " << label << ": no throw" << std::endl;
    ++failures;
}

static void check_close(const std::string& label, const std::vector<float>& got, const std::vector<float>& want) {
    if (got.size() != want.size()) {
        std::cout << "FAIL " << label << ": size mismatch" << std::endl;
        ++failures;
        return;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::abs(got[i] - want[i]) >= 1e-5f) {
            std::cout << "FAIL " << label << ": element " << i << " got " << got[i] << " want " << want[i] << std::endl;
            ++failures;
            return;
        }
    }
    std::cout << "PASS " << label << std::endl;
}

int main() {
    auto dev = Device(DeviceType::CPU);
    auto x = Tensor::from_vector({1.0f, 2.0f, -3.0f, 4.0f}, {4}, dev);
    auto y = Tensor::from_vector({0.5f, 1.5f, 2.0f, -1.0f}, {4}, dev);
    auto delta = Tensor::from_vector({0.25f, 0.25f, 0.25f, 0.25f}, {4}, dev);
    auto two = Tensor::from_vector({2.0f, 2.0f, 2.0f, 2.0f}, {4}, dev);

    {
        auto fn = Tracer::trace({x, y}, [](const std::vector<std::shared_ptr<Tensor>>& ins) {
            return Ops::relu(Ops::add(ins[0], ins[1]));
        });
        auto got = (*fn)({x, y})->to_vector();
        auto want = Ops::relu(Ops::add(x, y))->to_vector();
        check_close("clean trace evaluates", got, want);
    }

    {
        auto xc = Tensor::from_vector({1.0f, 2.0f, -3.0f, 4.0f}, {4}, dev);
        auto yc = Tensor::from_vector({0.5f, 1.5f, 2.0f, -1.0f}, {4}, dev);
        expect_throw("input add_ under NoGradGuard throws", [&]() {
            NoGradGuard guard;
            Tracer::trace({xc, yc}, [&](const std::vector<std::shared_ptr<Tensor>>& ins) {
                ins[0]->add_(delta);
                return Ops::add(ins[0], ins[1]);
            });
        }, "input tensor 0 was mutated in place during tracing");
    }

    {
        auto xc = Tensor::from_vector({1.0f, 2.0f, -3.0f, 4.0f}, {4}, dev);
        auto yc = Tensor::from_vector({0.5f, 1.5f, 2.0f, -1.0f}, {4}, dev);
        expect_throw("input copy_ under NoGradGuard throws with index 1", [&]() {
            NoGradGuard guard;
            Tracer::trace({xc, yc}, [&](const std::vector<std::shared_ptr<Tensor>>& ins) {
                ins[1]->copy_(delta);
                return Ops::add(ins[0], ins[1]);
            });
        }, "input tensor 1 was mutated in place during tracing");
    }

    {
        auto xc = Tensor::from_vector({1.0f, 2.0f, -3.0f, 4.0f}, {4}, dev);
        auto yc = Tensor::from_vector({0.5f, 1.5f, 2.0f, -1.0f}, {4}, dev);
        expect_throw("input add_ with grad enabled throws", [&]() {
            Tracer::trace({xc, yc}, [&](const std::vector<std::shared_ptr<Tensor>>& ins) {
                ins[0]->add_(delta);
                return Ops::add(ins[0], ins[1]);
            });
        }, "in-place operation");
    }

    {
        auto xc = Tensor::from_vector({1.0f, 2.0f, -3.0f, 4.0f}, {4}, dev);
        auto yc = Tensor::from_vector({0.5f, 1.5f, 2.0f, -1.0f}, {4}, dev);
        expect_throw("intermediate add_ before consumption throws", [&]() {
            Tracer::trace({xc, yc, two}, [&](const std::vector<std::shared_ptr<Tensor>>& ins) {
                auto t = Ops::add(ins[0], ins[1]);
                t->add_(delta);
                return Ops::mul(t, ins[2]);
            });
        }, "intermediate tensor was mutated in place during tracing");
    }

    {
        auto xc = Tensor::from_vector({1.0f, 2.0f, -3.0f, 4.0f}, {4}, dev);
        auto yc = Tensor::from_vector({0.5f, 1.5f, 2.0f, -1.0f}, {4}, dev);
        expect_throw("intermediate copy_ before consumption throws", [&]() {
            Tracer::trace({xc, yc, two}, [&](const std::vector<std::shared_ptr<Tensor>>& ins) {
                auto t = Ops::add(ins[0], ins[1]);
                t->copy_(delta);
                return Ops::mul(t, ins[2]);
            });
        }, "intermediate tensor was mutated in place during tracing");
    }

    {
        auto xc = Tensor::from_vector({1.0f, 2.0f, -3.0f, 4.0f}, {4}, dev);
        auto yc = Tensor::from_vector({0.5f, 1.5f, 2.0f, -1.0f}, {4}, dev);
        expect_throw("intermediate add_ after consumption throws", [&]() {
            Tracer::trace({xc, yc, two}, [&](const std::vector<std::shared_ptr<Tensor>>& ins) {
                auto t = Ops::add(ins[0], ins[1]);
                auto out = Ops::mul(t, ins[2]);
                t->add_(delta);
                return out;
            });
        }, "intermediate tensor was mutated in place during tracing");
    }

    {
        auto xc = Tensor::from_vector({1.0f, 2.0f, -3.0f, 4.0f}, {4}, dev);
        auto yc = Tensor::from_vector({0.5f, 1.5f, 2.0f, -1.0f}, {4}, dev);
        auto fn = Tracer::trace({xc, yc, two}, [&](const std::vector<std::shared_ptr<Tensor>>& ins) {
            auto t = Ops::add(ins[0], ins[1]);
            t->add_(delta);
            return Ops::mul(ins[0], ins[2]);
        });
        auto got = (*fn)({xc, two})->to_vector();
        auto want = Ops::mul(xc, two)->to_vector();
        check_close("dead intermediate mutation does not throw", got, want);
    }

    {
        auto xc = Tensor::from_vector({1.0f, 2.0f, -3.0f, 4.0f}, {4}, dev);
        auto yc = Tensor::from_vector({0.5f, 1.5f, 2.0f, -1.0f}, {4}, dev);
        expect_throw("alias view mutation of input throws", [&]() {
            Tracer::trace({xc, yc}, [&](const std::vector<std::shared_ptr<Tensor>>& ins) {
                auto v = ins[0]->view({4});
                v->add_(delta);
                return Ops::add(ins[0], ins[1]);
            });
        }, "input tensor 0 was mutated in place during tracing");
    }

    {
        auto xc = Tensor::from_vector({1.0f, 2.0f, -3.0f, 4.0f}, {4}, dev);
        auto yc = Tensor::from_vector({0.5f, 1.5f, 2.0f, -1.0f}, {4}, dev);
        std::vector<std::shared_ptr<Tensor>> keep;
        auto fn = Tracer::trace({xc, yc}, [&](const std::vector<std::shared_ptr<Tensor>>& ins) {
            auto m = Ops::mul(ins[0], ins[1]);
            keep.push_back(m);
            auto s = Ops::add(m, ins[1]);
            keep.push_back(s);
            return Ops::gelu(s);
        });
        auto got = (*fn)({xc, yc})->to_vector();
        auto want = Ops::gelu(Ops::add(Ops::mul(xc, yc), yc))->to_vector();
        check_close("clean multi-op trace evaluates", got, want);
    }

    if (failures == 0) {
        std::cout << "ALL_PASS" << std::endl;
        return 0;
    }
    std::cout << "FAILURES " << failures << std::endl;
    return 1;
}
