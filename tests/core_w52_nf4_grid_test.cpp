#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace litetorch;

static Device cpu(DeviceType::CPU, 0);
static int fails = 0;
static int checks = 0;

static void check(const char* name, float got, float want, float tol) {
    ++checks;
    if (std::fabs(got - want) <= tol) {
        std::printf("PASS: %s got %.9g want %.9g\n", name, got, want);
    } else {
        std::printf("FAIL: %s got %.9g want %.9g\n", name, got, want);
        ++fails;
    }
}

static float roundtrip(float v) {
    auto t = Tensor::from_vector({v}, {1}, cpu, false);
    auto q = t->cast(DataType::NF4);
    auto b = q->cast(DataType::FP32);
    return b->to_vector()[0];
}

int main() {
    const float std_grid[16] = {
        -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f, -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
        0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f, 0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
    };
    float back[16];
    for (int i = 0; i < 16; ++i) {
        back[i] = roundtrip(std_grid[i]);
        char name[64];
        std::snprintf(name, sizeof(name), "grid roundtrip index %d", i);
        check(name, back[i], std_grid[i], 1e-6f);
    }
    bool ascending = true;
    bool distinct = true;
    for (int i = 1; i < 16; ++i) {
        if (!(back[i] > back[i - 1])) ascending = false;
        if (back[i] == back[i - 1]) distinct = false;
    }
    ++checks;
    if (ascending) {
        std::printf("PASS: grid strictly ascending\n");
    } else {
        std::printf("FAIL: grid strictly ascending\n");
        ++fails;
    }
    ++checks;
    if (distinct) {
        std::printf("PASS: grid 16 distinct levels\n");
    } else {
        std::printf("FAIL: grid 16 distinct levels\n");
        ++fails;
    }
    check("nearest level of 0.85", roundtrip(0.85f), 0.7229568362236023f, 1e-6f);
    auto t = Tensor::from_vector({0.5f}, {1}, cpu, false);
    auto q = t->cast(DataType::NF4);
    ++checks;
    if (q->dtype == DataType::NF4) {
        std::printf("PASS: cast dtype NF4\n");
    } else {
        std::printf("FAIL: cast dtype NF4\n");
        ++fails;
    }
    std::printf("checks=%d fails=%d\n", checks, fails);
    return fails ? 1 : 0;
}
