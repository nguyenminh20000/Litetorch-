#include "litetorch/tensor.h"
#include "litetorch/autograd.h"
#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <cstdio>
#include <cmath>
#include <stdexcept>
using namespace litetorch;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %d: %s\n", __LINE__, #c); ++fails; } } while (0)
#define CHECK_THROWS(expr) do { bool threw = false; try { expr; } catch (const std::runtime_error&) { threw = true; } if (!threw) { printf("FAIL %d: expected throw: %s\n", __LINE__, #expr); ++fails; } } while (0)
static bool near(float a, float b, float tol = 1e-6f) { return std::fabs(a - b) <= tol; }
int main() {
    Device cpu(DeviceType::CPU, 0);
    auto w = Tensor::from_vector({0.00,0.01,0.02,0.03, 0.04,0.05,0.06,0.07, 0.08,0.09,0.10,0.11, 0.12,0.13,0.14,0.15, 0.16,0.17,0.18,0.19, 0.20,0.21,0.22,0.23, 0.24,0.25,0.26,0.27, 0.28,0.29,0.30,0.31}, {8,4}, cpu, false);
    {
        nn::VocabParallelEmbedding emb(8, 4);
        { NoGradGuard ng; emb.weight->copy_(w); }
        auto idx = Tensor::from_vector({0.0, 2.0, 7.0}, {3}, cpu, false, DataType::INT64);
        auto out = emb.forward(idx);
        auto v = out->to_vector();
        CHECK(v.size() == 12);
        CHECK(near(v[0], 0.0f) && near(v[1], 0.01f) && near(v[2], 0.02f) && near(v[3], 0.03f));
        CHECK(near(v[4], 0.08f) && near(v[5], 0.09f) && near(v[6], 0.10f) && near(v[7], 0.11f));
        CHECK(near(v[8], 0.28f) && near(v[9], 0.29f) && near(v[10], 0.30f) && near(v[11], 0.31f));
    }
    {
        nn::VocabParallelEmbedding emb(8, 4);
        { NoGradGuard ng; emb.weight->copy_(w); }
        auto idx = Tensor::from_vector({0.0, 2.0, 7.0}, {3}, cpu, false, DataType::INT32);
        auto out = emb.forward(idx);
        auto v = out->to_vector();
        CHECK(v.size() == 12);
        CHECK(near(v[0], 0.0f) && near(v[4], 0.08f) && near(v[8], 0.28f));
    }
    {
        nn::VocabParallelEmbedding emb(8, 4);
        { NoGradGuard ng; emb.weight->copy_(w); }
        auto idx2d = Tensor::from_vector({0.0,3.0,4.0,7.0, 1.0,2.0,5.0,6.0}, {2,4}, cpu, false, DataType::INT64);
        auto out = emb.forward(idx2d);
        CHECK(out->shape.size() == 3 && out->shape[0] == 2 && out->shape[1] == 4 && out->shape[2] == 4);
        auto v = out->to_vector();
        CHECK(v.size() == 32);
        CHECK(near(v[0], 0.0f) && near(v[12], 0.28f));
        CHECK(near(v[16], 0.04f) && near(v[28], 0.24f));
    }
    {
        nn::VocabParallelEmbedding emb(8, 4);
        auto idxf = Tensor::from_vector({0.0, 2.0, 7.0}, {3}, cpu, false, DataType::FP32);
        CHECK_THROWS(emb.forward(idxf));
        auto idxb = Tensor::from_vector({1.0, 0.0, 1.0}, {3}, cpu, false, DataType::BOOL);
        CHECK_THROWS(emb.forward(idxb));
    }
    {
        nn::VocabParallelEmbedding emb(8, 4);
        auto idxoob = Tensor::from_vector({8.0}, {1}, cpu, false, DataType::INT64);
        CHECK_THROWS(emb.forward(idxoob));
        auto idxneg = Tensor::from_vector({-1.0}, {1}, cpu, false, DataType::INT64);
        CHECK_THROWS(emb.forward(idxneg));
    }
    {
        nn::VocabParallelEmbedding emb(8, 4, 3);
        auto wv = emb.weight->to_vector();
        CHECK(wv[3 * 4 + 0] == 0.0f && wv[3 * 4 + 1] == 0.0f && wv[3 * 4 + 2] == 0.0f && wv[3 * 4 + 3] == 0.0f);
        auto idxp = Tensor::from_vector({3.0}, {1}, cpu, false, DataType::INT64);
        auto outp = emb.forward(idxp);
        for (float x : outp->to_vector()) CHECK(x == 0.0f);
        auto idx0 = Tensor::from_vector({0.0}, {1}, cpu, false, DataType::INT64);
        auto out0 = emb.forward(idx0);
        auto v0 = out0->to_vector();
        CHECK(near(v0[0], (float)wv[0]) && near(v0[1], (float)wv[1]) && near(v0[2], (float)wv[2]) && near(v0[3], (float)wv[3]));
    }
    {
        nn::VocabParallelEmbedding emb(8, 4);
        { NoGradGuard ng; emb.weight->copy_(w); }
        auto h = Tensor::from_vector({0.0,0.02,0.04,0.06, 0.08,0.10,0.12,0.14, 0.16,0.18,0.20,0.22}, {3,4}, cpu, true);
        auto lsh = emb.logits(h);
        CHECK(lsh->shape.size() == 2 && lsh->shape[0] == 3 && lsh->shape[1] == 8);
        auto lv = lsh->to_vector();
        CHECK(near(lv[0], 0.0028f, 1e-5f) && near(lv[1], 0.0076f, 1e-5f) && near(lv[7], 0.0364f, 1e-5f));
        auto tgt = Tensor::from_vector({0.0, 7.0, 3.0}, {3}, cpu, false, DataType::INT32);
        auto loss = emb.sharded_cross_entropy(lsh, tgt);
        CHECK(near(loss->to_vector()[0], 2.0706742f, 1e-4f));
        auto full = Ops::matmul(h, w->transpose(0, 1));
        auto ref = Ops::cross_entropy_loss(full, tgt);
        CHECK(near(loss->to_vector()[0], ref->to_vector()[0], 1e-5f));
    }
    {
        auto t = Tensor::create({2, 2}, cpu, false, DataType::INT64);
        int64_t* tp = reinterpret_cast<int64_t*>(t->raw_data_ptr());
        tp[0] = 16777217; tp[1] = 5; tp[2] = 16777218; tp[3] = 9;
        auto tr = t->transpose(0, 1);
        CHECK(!tr->is_contiguous());
        auto tc = tr->contiguous();
        const int64_t* cp = reinterpret_cast<const int64_t*>(tc->raw_data_ptr());
        CHECK(cp[0] == 16777217 && cp[1] == 16777218 && cp[2] == 5 && cp[3] == 9);
    }
    {
        const int64_t big_vocab = 16777225;
        nn::VocabParallelEmbedding emb((int)big_vocab, 4);
        auto idx = Tensor::create({1}, cpu, false, DataType::INT64);
        *reinterpret_cast<int64_t*>(idx->raw_data_ptr()) = 16777217;
        auto out = emb.forward(idx);
        CHECK(out->shape.size() == 2 && out->shape[0] == 1 && out->shape[1] == 4);
        auto v = out->to_vector();
        const float* wp = reinterpret_cast<const float*>(emb.weight->raw_data_ptr());
        int64_t row = 16777217;
        CHECK(near(v[0], wp[row * 4 + 0]) && near(v[1], wp[row * 4 + 1]));
        CHECK(near(v[2], wp[row * 4 + 2]) && near(v[3], wp[row * 4 + 3]));
        auto gidx = Tensor::create({1, 1}, cpu, false, DataType::INT64);
        *reinterpret_cast<int64_t*>(gidx->raw_data_ptr()) = 16777217;
        auto gd = Ops::gather(emb.weight, 0, gidx);
        CHECK(near(gd->to_vector()[0], wp[row * 4 + 0]));
    }
    if (fails == 0) printf("ALL CORE_W41 VOCABEMB TESTS PASSED\n");
    return fails ? 1 : 0;
}
