#include "litetorch/tensor.h"
#include "litetorch/autograd.h"
#include "litetorch/nn.h"
#include <cstdio>
using namespace litetorch;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %d: %s\n", __LINE__, #c); ++fails; } } while (0)
int main() {
    Device cpu(DeviceType::CPU, 0);
    {
        nn::Embedding emb(10, 4);
        auto w = Tensor::from_vector({1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16, 17,18,19,20, 21,22,23,24, 25,26,27,28, 29,30,31,32, 33,34,35,36, 37,38,39,40}, {10,4}, cpu, false);
        { NoGradGuard ng; emb.weight->copy_(w); }
        auto idx = Tensor::from_vector({2.0, 0.0, 9.0}, {3}, cpu, false, DataType::INT64);
        auto out = emb.forward(idx);
        auto v = out->to_vector();
        CHECK(v.size() == 12);
        for (int j = 0; j < 4; ++j) CHECK(v[j] == 9.0f + j);
        for (int j = 0; j < 4; ++j) CHECK(v[4+j] == 1.0f + j);
        for (int j = 0; j < 4; ++j) CHECK(v[8+j] == 37.0f + j);
    }
    {
        nn::Embedding emb(8, 4);
        auto idx = Tensor::from_vector({1.0, 3.0}, {2}, cpu, false);
        auto out = emb.forward(idx);
        CHECK(out->shape[0] == 2 && out->shape[1] == 4);
    }
    {
        nn::Embedding emb(10, 4);
        auto idx = Tensor::from_vector({-1.0, 100.0}, {2}, cpu, false, DataType::INT64);
        auto out = emb.forward(idx);
        for (float x : out->to_vector()) CHECK(x == 0.0f);
    }
    if (fails == 0) printf("ALL CORE_W10 EMBEDDING TESTS PASSED\n");
    return fails ? 1 : 0;
}
