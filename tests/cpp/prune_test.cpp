#include "litetorch/prune.h"
#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace litetorch;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); ++failures; } } while (0)

static std::shared_ptr<nn::Linear> make_linear() {
    auto lin = std::make_shared<nn::Linear>(4, 2, false);
    std::vector<float> w = {0.1f, -2.0f, 0.3f, -0.05f, 1.5f, -0.7f, 0.9f, -0.2f};
    auto wt = Tensor::from_vector(w, {2, 4}, Device(DeviceType::CPU), true);
    lin->weight->copy_(wt);
    return lin;
}

static std::shared_ptr<Tensor> find_mask(nn::Module& m) {
    for (auto& kv : m.named_buffers()) {
        if (kv.first == "weight_mask") return kv.second;
    }
    return nullptr;
}

static int64_t count_zeros(std::shared_ptr<Tensor> t) {
    auto c = t->to(Device(DeviceType::CPU), DataType::FP32);
    float* d = (float*)c->storage->get_cpu_ptr();
    int64_t z = 0;
    for (size_t i = 0; i < t->numel(); ++i) if (d[i] == 0.0f) ++z;
    return z;
}

int main() {
    {
        auto lin = make_linear();
        CHECK(!prune::is_pruned(*lin, "weight"));
        prune::l1_unstructured(*lin, "weight", 0.5);
        CHECK(prune::is_pruned(*lin, "weight"));
        auto mask = find_mask(*lin);
        CHECK(mask != nullptr);
        CHECK(count_zeros(mask) == 4);
        std::printf("l1 mask: zeros=%lld\n", (long long)count_zeros(mask));
    }
    {
        auto lin = make_linear();
        prune::l1_unstructured(*lin, "weight", 0.5);
        auto x = Tensor::from_vector({1.0f, 1.0f, 1.0f, 1.0f}, {1, 4}, Device(DeviceType::CPU), false);
        auto y1 = (*lin)(x);
        auto w_before = lin->weight->to(Device(DeviceType::CPU), DataType::FP32);
        prune::remove(*lin, "weight");
        CHECK(!prune::is_pruned(*lin, "weight"));
        auto y2 = (*lin)(x);
        auto d1 = (float*)y1->storage->get_cpu_ptr();
        auto d2 = (float*)y2->storage->get_cpu_ptr();
        CHECK(std::fabs(d1[0] - d2[0]) < 1e-6 && std::fabs(d1[1] - d2[1]) < 1e-6);
        CHECK(count_zeros(lin->weight) == 4);
        std::printf("remove: forward equal, baked zeros=%lld\n", (long long)count_zeros(lin->weight));
        (void)w_before;
    }
    {
        auto lin = make_linear();
        prune::l1_unstructured(*lin, "weight", (int64_t)2);
        auto mask = find_mask(*lin);
        CHECK(mask != nullptr && count_zeros(mask) == 2);
        std::printf("int-count prune: zeros=%lld\n", (long long)count_zeros(mask));
    }
    {
        auto lin = make_linear();
        prune::random_unstructured(*lin, "weight", 0.5, 42);
        auto mask = find_mask(*lin);
        CHECK(mask != nullptr);
        auto lin2 = make_linear();
        prune::random_unstructured(*lin2, "weight", 0.5, 42);
        auto m1 = find_mask(*lin);
        auto m2 = find_mask(*lin2);
        auto c1 = m1->to(Device(DeviceType::CPU), DataType::FP32);
        auto c2 = m2->to(Device(DeviceType::CPU), DataType::FP32);
        float* d1 = (float*)c1->storage->get_cpu_ptr();
        float* d2 = (float*)c2->storage->get_cpu_ptr();
        bool same = true;
        for (size_t i = 0; i < m1->numel(); ++i) if (d1[i] != d2[i]) same = false;
        CHECK(same);
        std::printf("random seed determinism: %s\n", same ? "same" : "DIFFERENT");
    }
    {
        auto lin = make_linear();
        bool threw = false;
        try {
            prune::remove(*lin, "weight");
        } catch (const std::exception&) { threw = true; }
        CHECK(threw);
        threw = false;
        try {
            prune::l1_unstructured(*lin, "nope", 0.5);
        } catch (const std::exception&) { threw = true; }
        CHECK(threw);
        std::printf("error paths throw: ok\n");
    }
    {
        auto lin = make_linear();
        prune::l1_unstructured(*lin, "weight", 0.5);
        auto sd = lin->state_dict();
        bool has_mask = false;
        for (auto& kv : sd) {
            if (kv.first == "weight_mask") {
                has_mask = true;
                CHECK(count_zeros(kv.second) == 4);
            }
        }
        CHECK(has_mask);
        auto lin2 = make_linear();
        prune::l1_unstructured(*lin2, "weight", 0.25);
        auto res = lin2->load_state_dict(sd, true);
        CHECK(res.first.empty() && res.second.empty());
        CHECK(prune::is_pruned(*lin2, "weight"));
        auto x = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4}, Device(DeviceType::CPU), false);
        auto y1 = (*lin)(x);
        auto y2 = (*lin2)(x);
        auto d1 = (float*)y1->storage->get_cpu_ptr();
        auto d2 = (float*)y2->storage->get_cpu_ptr();
        CHECK(std::fabs(d1[0] - d2[0]) < 1e-6 && std::fabs(d1[1] - d2[1]) < 1e-6);
        std::printf("state_dict round-trip: mask preserved, forward equal\n");
    }
    if (failures == 0) std::printf("ALL CPP PRUNE TESTS PASS\n");
    return failures ? 1 : 0;
}
