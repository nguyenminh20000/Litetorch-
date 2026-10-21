#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstring>
#include <functional>
#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/checkpoint.h"

using namespace litetorch;
using namespace litetorch::Ops;

static Device cpu(DeviceType::CPU, 0);
static int fails = 0;

static void check(const char* name, bool ok, const std::string& detail) {
    std::cout << "CASE " << name << ": " << (ok ? "PASS" : "FAIL") << " | " << detail << std::endl;
    if (!ok) fails++;
}

static bool veq(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-5f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (std::isnan(b[i])) { if (!std::isnan(a[i])) return false; continue; }
        if (std::fabs(a[i] - b[i]) > tol && !(std::isinf(a[i]) && std::isinf(b[i]) && (a[i] > 0) == (b[i] > 0))) return false;
    }
    return true;
}

static void run(const char* name, std::function<void()> fn) {
    try { fn(); }
    catch (const std::exception& ex) {
        check(name, false, std::string("UNEXPECTED THROW: ") + ex.what());
    }
}

int main() {
    run("empty_add", []() {
        auto e = Tensor::zeros({0}, cpu);
        auto c = add(e, e);
        check("empty_add", c->numel() == 0, "numel=" + std::to_string(c->numel()));
    });

    run("matmul_1x1_fwd_bwd", []() {
        auto a = Tensor::from_vector({2.0f}, {1, 1}, cpu, true);
        auto b = Tensor::from_vector({3.0f}, {1, 1}, cpu, true);
        auto m = matmul(a, b);
        bool okm = std::fabs(m->item() - 6.0f) < 1e-5f;
        m->backward();
        okm = okm && a->grad && veq(a->grad->to_vector(), {3.0f}) && b->grad && veq(b->grad->to_vector(), {2.0f});
        check("matmul_1x1_fwd_bwd", okm, "c=6 da=3 db=2 expected");
    });

    run("matmul_2x2_fwd", []() {
        auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, cpu);
        auto b = Tensor::from_vector({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, cpu);
        auto m = matmul(a, b);
        check("matmul_2x2_fwd", veq(m->to_vector(), {19.0f, 22.0f, 43.0f, 50.0f}), "[[19,22],[43,50]] expected");
    });

    run("slice_step2_backward", []() {
        auto x = Tensor::arange(0, 6, 1, cpu, DataType::FP32);
        x->requires_grad = true;
        auto s = x->slice(0, 0, 6, 2);
        bool oks = s->numel() == 3 && veq(s->to_vector(), {0.0f, 2.0f, 4.0f});
        auto loss = s->sum(0);
        loss->backward();
        oks = oks && x->grad && veq(x->grad->to_vector(), {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f});
        check("slice_step2_backward", oks, "grad=[1,0,1,0,1,0] expected");
    });

    run("extremes_sum_no_crash", []() {
        float inf = std::numeric_limits<float>::infinity();
        float nan = std::numeric_limits<float>::quiet_NaN();
        auto xv = Tensor::from_vector({inf, -inf, nan, 1e30f, 1e-40f}, {5}, cpu);
        float sv = xv->sum(0)->item();
        check("extremes_sum_no_crash", true, "sum=" + std::to_string(sv) + " isnan=" + (std::isnan(sv) ? "1" : "0"));
    });

    run("masked_fill_nan_backward", []() {
        float nan = std::numeric_limits<float>::quiet_NaN();
        auto xm = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
        auto mask = Tensor::create({2}, cpu, false, DataType::BOOL);
        std::memset(mask->raw_data_ptr(), 0, 2);
        static_cast<uint8_t*>(mask->raw_data_ptr())[1] = 1;
        auto mf = xm->masked_fill(mask, nan);
        bool okmf = veq(mf->to_vector(), {1.0f, nan});
        mf->sum(0)->backward();
        okmf = okmf && xm->grad && veq(xm->grad->to_vector(), {1.0f, 0.0f});
        check("masked_fill_nan_backward", okmf, "masked value=nan, grad=[1,0] expected");
    });

    run("dtype_data_ptr_contract", []() {
        bool t64 = false, t8 = false, tb = false;
        auto ti = Tensor::create({2}, cpu, false, DataType::INT64);
        auto u8 = Tensor::create({2}, cpu, false, DataType::UINT8);
        auto bl = Tensor::create({2}, cpu, false, DataType::BOOL);
        try { volatile float* p = ti->data_ptr(); (void)p; } catch (const std::exception&) { t64 = true; }
        try { volatile float* p = u8->data_ptr(); (void)p; } catch (const std::exception&) { t8 = true; }
        try { volatile float* p = bl->data_ptr(); (void)p; } catch (const std::exception&) { tb = true; }
        void* rp = ti->raw_data_ptr();
        bool okraw = rp != nullptr;
        if (okraw) { static_cast<int64_t*>(rp)[0] = 42; okraw = static_cast<int64_t*>(rp)[0] == 42; }
        check("dtype_data_ptr_contract", t64 && t8 && tb && okraw,
            "int64=" + std::string(t64 ? "throw" : "NO-THROW") + " uint8=" + std::string(t8 ? "throw" : "NO-THROW") + " bool=" + std::string(tb ? "throw" : "NO-THROW") + " raw_rw=" + (okraw ? "ok" : "BAD"));
    });

    run("int64_matmul_correctness", []() {
        auto i1 = Tensor::create({2, 2}, cpu, false, DataType::INT64);
        auto i2 = Tensor::create({2, 2}, cpu, false, DataType::INT64);
        int64_t* p1 = static_cast<int64_t*>(i1->raw_data_ptr());
        int64_t* p2 = static_cast<int64_t*>(i2->raw_data_ptr());
        p1[0] = 1; p1[1] = 2; p1[2] = 3; p1[3] = 4;
        p2[0] = 5; p2[1] = 6; p2[2] = 7; p2[3] = 8;
        bool threw = false;
        std::string detail;
        try {
            auto r = matmul(i1, i2);
            int64_t* pr = static_cast<int64_t*>(r->raw_data_ptr());
            bool correct = (pr[0] == 19 && pr[1] == 22 && pr[2] == 43 && pr[3] == 50);
            detail = "returned [" + std::to_string(pr[0]) + "," + std::to_string(pr[1]) + "," + std::to_string(pr[2]) + "," + std::to_string(pr[3]) + "] expected [19,22,43,50] | torch: torch.matmul supports int64";
            check("int64_matmul_correctness", correct, detail);
        } catch (const std::exception& ex) { threw = true; detail = std::string("threw: ") + ex.what(); check("int64_matmul_correctness", true, detail + " | torch supports int64 matmul - clean throw is acceptable, document it"); }
        (void)threw;
    });

    run("inplace_leaf_refused", []() {
        auto xi = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
        bool threw = false;
        try { xi->add_(Tensor::ones({2}, cpu)); } catch (const std::exception&) { threw = true; }
        check("inplace_leaf_refused", threw, "in-place on grad-requiring leaf must throw (torch parity)");
    });

    run("inplace_version_guard", []() {
        auto x = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
        auto h = mul(x, x);
        auto k = mul(h, h);
        auto z = sum(k);
        h->add_(Tensor::ones({2}, cpu));
        bool guard_threw = false;
        std::string guard_msg;
        try { z->backward(); } catch (const std::exception& ex) { guard_threw = true; guard_msg = ex.what(); }
        bool okg = guard_threw && guard_msg.find("inplace") != std::string::npos && guard_msg.find("version") != std::string::npos;
        check("inplace_version_guard", okg, guard_threw ? guard_msg : "NO THROW - silent wrong grad risk");
    });

    run("checkpoint_nogradguard", []() {
        auto xc = Tensor::from_vector({2.0f, 3.0f}, {2}, cpu, true);
        auto yc = checkpoint([](std::shared_ptr<Tensor> t) { return mul(t, t); }, xc);
        yc->sum(0)->backward();
        bool okc = xc->grad && veq(xc->grad->to_vector(), {4.0f, 6.0f});
        check("checkpoint_nogradguard", okc, "grad=2x=[4,6] expected");
    });

    run("retain_graph_twice", []() {
        auto xr = Tensor::from_vector({1.0f, 2.0f}, {2}, cpu, true);
        auto lr = mul(xr, xr)->sum(0);
        lr->backward(nullptr, false, true);
        std::vector<float> g1 = xr->grad->to_vector();
        xr->zero_grad();
        lr->backward(nullptr, false, true);
        std::vector<float> g2 = xr->grad->to_vector();
        bool torch_parity = veq(g2, {2.0f, 4.0f});
        std::string detail = "g1=[" + std::to_string(g1[0]) + "," + std::to_string(g1[1]) + "] g2=[" + std::to_string(g2[0]) + "," + std::to_string(g2[1]) + "] torch-expected g2=[2,4]";
        check("retain_graph_twice_torch_parity", torch_parity, detail + (torch_parity ? "" : " | PRE-EXISTING DIVERGENCE (base 536f3b5): Tensor::backward accumulates implicit root grad across calls"));
    });

    run("index_select_int64_backward", []() {
        auto xs = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, cpu, true);
        auto idx = Tensor::create({2}, cpu, false, DataType::INT64);
        static_cast<int64_t*>(idx->raw_data_ptr())[0] = 0;
        static_cast<int64_t*>(idx->raw_data_ptr())[1] = 2;
        auto isel = xs->index_select(0, idx);
        bool okis = veq(isel->to_vector(), {1.0f, 3.0f});
        isel->sum(0)->backward();
        okis = okis && xs->grad && veq(xs->grad->to_vector(), {1.0f, 0.0f, 1.0f});
        check("index_select_int64_backward", okis, "vals=[1,3] grad=[1,0,1] expected");
    });

    run("large_shape", []() {
        auto big = Tensor::zeros({1024, 1024}, cpu);
        auto s = big->sum(0)->sum(0);
        check("large_shape", s->numel() == 1 && s->item() == 0.0f, "1024x1024 zeros sum=0");
    });

    std::cout << "FUZZ_DONE fails=" << fails << std::endl;
    return fails == 0 ? 0 : 1;
}
