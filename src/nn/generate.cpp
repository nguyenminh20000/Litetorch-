#include "litetorch/generate.h"
#include "litetorch/autograd.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace litetorch {
namespace nn {

static uint64_t xorshift64(uint64_t& s) {
    if (s == 0) s = 0x9E3779B97F4A7C15ULL;
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return s;
}

static double rand_unit(uint64_t& s) {
    return (xorshift64(s) >> 11) * (1.0 / 9007199254740992.0);
}

static std::shared_ptr<Tensor> tokens_to_input(const std::vector<int64_t>& toks) {
    std::vector<float> v(toks.size());
    for (size_t i = 0; i < toks.size(); ++i) v[i] = static_cast<float>(toks[i]);
    return Tensor::from_vector(v, {1, static_cast<int64_t>(toks.size())});
}

static std::vector<float> last_logits(std::shared_ptr<Module> model,
                                      const std::vector<int64_t>& toks) {
    auto out = model->forward(tokens_to_input(toks));
    int64_t V = out->shape[2], T = out->shape[1];
    std::vector<float> flat = out->to_vector();
    return std::vector<float>(flat.begin() + (T - 1) * V, flat.begin() + T * V);
}

std::vector<float> apply_repetition_penalty(const std::vector<float>& logits,
                                           const std::vector<int64_t>& seen,
                                           float penalty) {
    if (penalty == 1.0f) return logits;
    std::vector<float> out = logits;
    for (int64_t id : seen) {
        if (id < 0 || id >= (int64_t)out.size()) continue;
        if (out[id] > 0) out[id] /= penalty; else out[id] *= penalty;
    }
    return out;
}

std::vector<float> apply_top_k(const std::vector<float>& logits, int64_t k) {
    if (k <= 0 || k >= (int64_t)logits.size()) return logits;
    std::vector<float> out(logits.size(), -std::numeric_limits<float>::infinity());
    std::vector<int64_t> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                     [&](int64_t a, int64_t b) { return logits[a] > logits[b]; });
    for (int64_t i = 0; i < k; ++i) out[idx[i]] = logits[idx[i]];
    return out;
}

std::vector<float> apply_top_p(std::vector<float> logits, float p) {
    if (p >= 1.0f) return logits;
    int64_t V = logits.size();
    std::vector<int64_t> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return logits[a] > logits[b]; });
    float mx = logits[idx[0]];
    std::vector<double> probs(V);
    double sum = 0;
    for (int64_t i = 0; i < V; ++i) { probs[i] = std::exp(logits[idx[i]] - mx); sum += probs[i]; }
    double cum = 0;
    std::vector<float> out(V, -std::numeric_limits<float>::infinity());
    for (int64_t i = 0; i < V; ++i) {
        cum += probs[i] / sum;
        out[idx[i]] = logits[idx[i]];
        if (cum >= p) break;
    }
    return out;
}

int64_t sample_token(const std::vector<float>& logits, float temperature, uint64_t& rng_state) {
    int64_t V = logits.size();
    float temp = temperature > 0 ? temperature : 1.0f;
    float mx = -std::numeric_limits<float>::infinity();
    for (float l : logits) if (l > mx) mx = l;
    std::vector<double> probs(V);
    double sum = 0;
    for (int64_t i = 0; i < V; ++i) {
        double p = (logits[i] == -std::numeric_limits<float>::infinity())
            ? 0.0 : std::exp((logits[i] - mx) / temp);
        probs[i] = p; sum += p;
    }
    if (sum <= 0) {
        int64_t best = 0;
        for (int64_t i = 1; i < V; ++i) if (logits[i] > logits[best]) best = i;
        return best;
    }
    double r = rand_unit(rng_state) * sum, cum = 0;
    for (int64_t i = 0; i < V; ++i) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return V - 1;
}

static bool is_stop(const std::vector<int64_t>& gen, const GenerateConfig& cfg) {
    if (gen.empty()) return false;
    int64_t last = gen.back();
    if (cfg.eos_token_id >= 0 && last == cfg.eos_token_id) return true;
    for (int64_t s : cfg.stop_token_ids) if (last == s) return true;
    for (auto& seq : cfg.stop_sequences) {
        if (seq.empty() || seq.size() > gen.size()) continue;
        bool match = true;
        for (size_t i = 0; i < seq.size(); ++i)
            if (gen[gen.size() - seq.size() + i] != seq[i]) { match = false; break; }
        if (match) return true;
    }
    return false;
}

static int64_t pick_token(std::vector<float> logits, const std::vector<int64_t>& seen,
                          const GenerateConfig& cfg, uint64_t& rng) {
    logits = apply_repetition_penalty(logits, seen, cfg.repetition_penalty);
    if (cfg.do_sample) {
        logits = apply_top_k(logits, cfg.top_k);
        logits = apply_top_p(std::move(logits), cfg.top_p);
        return sample_token(logits, cfg.temperature, rng);
    }
    int64_t best = 0;
    for (size_t i = 1; i < logits.size(); ++i) if (logits[i] > logits[best]) best = i;
    return best;
}

static std::vector<int64_t> greedy_or_sample(std::shared_ptr<Module> model,
                                             const std::vector<int64_t>& prompt,
                                             const GenerateConfig& cfg,
                                             TokenCallback on_token) {
    NoGradGuard ng;
    std::vector<int64_t> gen, seen = prompt;
    uint64_t rng = cfg.seed;
    std::vector<int64_t> ctx = prompt;
    for (int64_t step = 0; step < cfg.max_new_tokens; ++step) {
        int64_t tok = pick_token(last_logits(model, ctx), seen, cfg, rng);
        gen.push_back(tok); seen.push_back(tok); ctx.push_back(tok);
        if (on_token && !on_token(tok)) break;
        if (is_stop(gen, cfg)) break;
    }
    return gen;
}

static std::vector<int64_t> beam_search(std::shared_ptr<Module> model,
                                        const std::vector<int64_t>& prompt,
                                        const GenerateConfig& cfg) {
    NoGradGuard ng;
    struct Hyp { std::vector<int64_t> toks; double score = 0; bool done = false; };
    int64_t B = cfg.num_beams;
    std::vector<Hyp> beams = {{prompt, 0.0, false}};
    for (int64_t step = 0; step < cfg.max_new_tokens; ++step) {
        struct Cand { double score; int64_t beam; int64_t tok; std::vector<float> lp; };
        std::vector<Cand> cands;
        bool all_done = true;
        for (size_t b = 0; b < beams.size(); ++b) {
            if (beams[b].done) continue;
            all_done = false;
            auto lg = last_logits(model, beams[b].toks);
            float mx = *std::max_element(lg.begin(), lg.end());
            double z = 0;
            std::vector<float> lp(lg.size());
            for (size_t i = 0; i < lg.size(); ++i) { lp[i] = (lg[i] - mx); z += std::exp(lp[i]); }
            double lz = std::log(z);
            std::vector<int64_t> idx(lg.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::nth_element(idx.begin(), idx.begin() + B, idx.end(),
                             [&](int64_t a, int64_t b2) { return lp[a] > lp[b2]; });
            for (int64_t i = 0; i < B; ++i)
                cands.push_back({beams[b].score + lp[idx[i]] - lz, (int64_t)b, idx[i], {}});
        }
        if (all_done) break;
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& a, const Cand& b) { return a.score > b.score; });
        std::vector<Hyp> next;
        for (size_t i = 0; i < cands.size() && (int64_t)next.size() < B; ++i) {
            Hyp h = beams[cands[i].beam];
            h.toks.push_back(cands[i].tok);
            h.score = cands[i].score;
            std::vector<int64_t> tail(h.toks.end() - std::min<size_t>(1, h.toks.size()), h.toks.end());
            if (is_stop(tail, cfg)) h.done = true;
            next.push_back(std::move(h));
        }
        beams = std::move(next);
    }
    auto best = std::max_element(beams.begin(), beams.end(),
                                 [](const Hyp& a, const Hyp& b) { return a.score < b.score; });
    std::vector<int64_t> out(best->toks.begin() + prompt.size(), best->toks.end());
    return out;
}

std::vector<int64_t> generate(std::shared_ptr<Module> model,
                              const std::vector<int64_t>& prompt_tokens,
                              const GenerateConfig& config) {
    return generate_stream(model, prompt_tokens, config, nullptr);
}

std::vector<int64_t> generate_stream(std::shared_ptr<Module> model,
                                     const std::vector<int64_t>& prompt_tokens,
                                     const GenerateConfig& config,
                                     TokenCallback on_token) {
    if (prompt_tokens.empty())
        throw std::runtime_error("[litetorch Error] generate: empty prompt");
    if (config.num_beams > 1)
        return beam_search(model, prompt_tokens, config);
    return greedy_or_sample(model, prompt_tokens, config, on_token);
}

}
}
