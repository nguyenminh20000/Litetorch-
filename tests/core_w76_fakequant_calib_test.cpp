#include "litetorch/quant_scheme.h"
#include "litetorch/quantization.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace litetorch;
using namespace litetorch::quantization;

static int passed = 0;
static int failed = 0;

static void check(const char* name, bool ok) {
    if (ok) {
        ++passed;
        printf("  PASS %s\n", name);
    } else {
        ++failed;
        printf("  FAIL %s\n", name);
    }
}

static std::shared_ptr<Tensor> fvec(const std::vector<float>& v, const std::vector<int64_t>& shape) {
    return Tensor::from_vector(v, shape, Device(DeviceType::CPU, 0), false, DataType::FP32);
}

static std::vector<float> tovec(std::shared_ptr<Tensor> t) {
    return t->to(Device(DeviceType::CPU, 0), DataType::FP32)->to_vector();
}

static bool near_vec(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

int main() {
    printf("fake-quant calibrate include-0, CPU:0\n");
    QuantScheme scheme = get_scheme("fake-quant");

    auto xc = fvec({0.5f, 0.5f, 0.5f, 0.5f}, {4});
    QParams qpc = scheme.calibrate(xc, nullptr, "", scheme.params);
    check("constant 0.5 scale is 0.5/254", std::fabs(qpc.scale - 0.5f / 254.0f) < 1e-6f);
    check("constant 0.5 zero_point is -127", qpc.zero_point == -127.0f);
    check("constant 0.5 fake_quant preserves 0.5", near_vec(tovec(scheme_fake_quant(scheme, xc, qpc, scheme.params)), {0.5f, 0.5f, 0.5f, 0.5f}, 1e-6f));

    auto xp = fvec({0.1f, 0.4f, 0.7f, 1.0f}, {4});
    QParams qpp = scheme.calibrate(xp, nullptr, "", scheme.params);
    check("all-positive scale is 1.0/254", std::fabs(qpp.scale - 1.0f / 254.0f) < 1e-6f);
    check("all-positive zero_point is -127", qpp.zero_point == -127.0f);
    std::vector<float> fqp = tovec(scheme_fake_quant(scheme, xp, qpp, scheme.params));
    check("all-positive fake_quant max preserved 1.0", std::fabs(fqp[3] - 1.0f) < 1e-6f);
    check("all-positive fake_quant min near 0.1", std::fabs(fqp[0] - 0.1f) < 0.002f);

    auto xn = fvec({-1.0f, -0.7f, -0.4f, -0.1f}, {4});
    QParams qpn = scheme.calibrate(xn, nullptr, "", scheme.params);
    check("all-negative scale is 1.0/254", std::fabs(qpn.scale - 1.0f / 254.0f) < 1e-6f);
    check("all-negative zero_point is 127", qpn.zero_point == 127.0f);
    std::vector<float> fqn = tovec(scheme_fake_quant(scheme, xn, qpn, scheme.params));
    check("all-negative fake_quant min preserved -1.0", std::fabs(fqn[0] - (-1.0f)) < 1e-6f);

    auto xb = fvec({-0.5f, -0.25f, 0.25f, 0.5f}, {4});
    QParams qpb = scheme.calibrate(xb, nullptr, "", scheme.params);
    check("both-signed scale unchanged 1.0/254", std::fabs(qpb.scale - 1.0f / 254.0f) < 1e-6f);
    check("both-signed zero_point unchanged 0", qpb.zero_point == 0.0f);

    auto x0 = fvec({0.0f, 0.0f, 0.0f, 0.0f}, {4});
    bool ok0 = true;
    QParams qp0;
    try {
        qp0 = scheme.calibrate(x0, nullptr, "", scheme.params);
    } catch (...) {
        ok0 = false;
    }
    check("all-zeros calibrate does not throw", ok0);
    check("all-zeros scale falls back to 1.0", ok0 && qp0.scale == 1.0f);
    check("all-zeros fake_quant is 0", ok0 && near_vec(tovec(scheme_fake_quant(scheme, x0, qp0, scheme.params)), {0.0f, 0.0f, 0.0f, 0.0f}, 1e-6f));

    Calibrator cal;
    cal.collect("w", xc);
    QParams qpcc = scheme.calibrate(xc, &cal, "w", scheme.params);
    check("calibrator-branch constant 0.5 scale is 0.5/254", std::fabs(qpcc.scale - 0.5f / 254.0f) < 1e-6f);
    check("calibrator-branch constant 0.5 zero_point is -127", qpcc.zero_point == -127.0f);

    printf("TOTAL %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
