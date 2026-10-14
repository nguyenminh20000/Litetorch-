#include "litetorch/prune.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace litetorch {

extern std::mt19937& get_generator();

namespace prune {

namespace {

struct HookState {
    std::string name;
    std::shared_ptr<Tensor> orig;
    std::shared_ptr<Tensor> saved;
};

std::shared_ptr<Tensor> get_member(nn::Module& m, const std::string& name) {
    if (auto* l = dynamic_cast<nn::Linear*>(&m)) {
        if (name == "weight") return l->weight;
        if (name == "bias") return l->bias;
    } else if (auto* c = dynamic_cast<nn::Conv1d*>(&m)) {
        if (name == "weight") return c->weight;
        if (name == "bias") return c->bias;
    } else if (auto* c = dynamic_cast<nn::Conv2d*>(&m)) {
        if (name == "weight") return c->weight;
        if (name == "bias") return c->bias;
    } else if (auto* c = dynamic_cast<nn::Conv3d*>(&m)) {
        if (name == "weight") return c->weight;
        if (name == "bias") return c->bias;
    } else if (auto* c = dynamic_cast<nn::ConvTranspose1d*>(&m)) {
        if (name == "weight") return c->weight;
        if (name == "bias") return c->bias;
    } else if (auto* c = dynamic_cast<nn::ConvTranspose2d*>(&m)) {
        if (name == "weight") return c->weight;
        if (name == "bias") return c->bias;
    } else if (auto* c = dynamic_cast<nn::ConvTranspose3d*>(&m)) {
        if (name == "weight") return c->weight;
        if (name == "bias") return c->bias;
    } else if (auto* e = dynamic_cast<nn::Embedding*>(&m)) {
        if (name == "weight") return e->weight;
    }
    return nullptr;
}

void set_member(nn::Module& m, const std::string& name, std::shared_ptr<Tensor> t) {
    if (auto* l = dynamic_cast<nn::Linear*>(&m)) {
        if (name == "weight") { l->weight = t; return; }
        if (name == "bias") { l->bias = t; return; }
    } else if (auto* c = dynamic_cast<nn::Conv1d*>(&m)) {
        if (name == "weight") { c->weight = t; return; }
        if (name == "bias") { c->bias = t; return; }
    } else if (auto* c = dynamic_cast<nn::Conv2d*>(&m)) {
        if (name == "weight") { c->weight = t; return; }
        if (name == "bias") { c->bias = t; return; }
    } else if (auto* c = dynamic_cast<nn::Conv3d*>(&m)) {
        if (name == "weight") { c->weight = t; return; }
        if (name == "bias") { c->bias = t; return; }
    } else if (auto* c = dynamic_cast<nn::ConvTranspose1d*>(&m)) {
        if (name == "weight") { c->weight = t; return; }
        if (name == "bias") { c->bias = t; return; }
    } else if (auto* c = dynamic_cast<nn::ConvTranspose2d*>(&m)) {
        if (name == "weight") { c->weight = t; return; }
        if (name == "bias") { c->bias = t; return; }
    } else if (auto* c = dynamic_cast<nn::ConvTranspose3d*>(&m)) {
        if (name == "weight") { c->weight = t; return; }
        if (name == "bias") { c->bias = t; return; }
    } else if (auto* e = dynamic_cast<nn::Embedding*>(&m)) {
        if (name == "weight") { e->weight = t; return; }
    }
    throw std::runtime_error("[litetorch Error] prune: unsupported module/param '" + name + "'");
}

std::string mask_key(const std::string& name) {
    return name + "_mask";
}

std::shared_ptr<Tensor> find_mask(nn::Module& m, const std::string& name) {
    std::string key = mask_key(name);
    for (auto& kv : m.named_buffers()) {
        if (kv.first == key) return kv.second;
    }
    return nullptr;
}

std::vector<int8_t> mask_bytes_of(std::shared_ptr<Tensor> mask) {
    auto c = mask->to(Device(DeviceType::CPU), DataType::INT8);
    int8_t* d = reinterpret_cast<int8_t*>(c->storage->get_cpu_ptr());
    return std::vector<int8_t>(d, d + mask->numel());
}

std::vector<int64_t> unpruned_candidates(nn::Module& module, const std::string& name, int64_t n) {
    std::vector<int64_t> cand;
    if (auto old = find_mask(module, name)) {
        auto bytes = mask_bytes_of(old);
        for (int64_t i = 0; i < n; ++i) {
            if (bytes[(size_t)i]) cand.push_back(i);
        }
        return cand;
    }
    cand.resize((size_t)n);
    std::iota(cand.begin(), cand.end(), 0);
    return cand;
}

std::vector<int64_t> pick_l1(const float* d, const std::vector<int64_t>& cand, int64_t k) {
    std::vector<int64_t> idx = cand;
    if (k <= 0) return {};
    if (k >= (int64_t)idx.size()) return idx;
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                     [&](int64_t a, int64_t b) { return std::fabs(d[a]) < std::fabs(d[b]); });
    idx.resize((size_t)k);
    return idx;
}

std::vector<int64_t> pick_random(const std::vector<int64_t>& cand, int64_t k, int64_t seed) {
    if (k <= 0) return {};
    if (k >= (int64_t)cand.size()) return cand;
    std::mt19937 local_gen;
    std::mt19937* gen = &local_gen;
    if (seed >= 0) {
        local_gen.seed((uint32_t)seed);
    } else {
        gen = &get_generator();
    }
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::vector<double> r(cand.size());
    for (size_t i = 0; i < cand.size(); ++i) r[i] = dis(*gen);
    std::vector<int64_t> idx(cand.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                     [&](int64_t a, int64_t b) { return r[(size_t)a] > r[(size_t)b]; });
    std::vector<int64_t> out;
    for (int64_t i = 0; i < k; ++i) out.push_back(cand[(size_t)idx[(size_t)i]]);
    return out;
}

void apply_zero_idx(nn::Module& module, const std::string& name, const std::vector<int64_t>& zero_idx) {
    auto live = get_member(module, name);
    if (!live) throw std::runtime_error("[litetorch Error] prune: param '" + name + "' not found");
    int64_t n = (int64_t)live->numel();
    std::vector<int8_t> bytes((size_t)n, 1);
    if (auto old = find_mask(module, name)) {
        bytes = mask_bytes_of(old);
    } else {
        auto st = std::make_shared<HookState>();
        st->name = name;
        st->orig = live;
        module.register_forward_pre_hook([st](nn::Module* m, std::shared_ptr<Tensor>) {
            auto mask = find_mask(*m, st->name);
            if (!mask) return;
            auto cur = get_member(*m, st->name);
            if (!cur) return;
            if (cur->device != st->orig->device || cur->dtype != st->orig->dtype) st->orig = cur;
            auto mf = mask;
            if (mf->dtype != st->orig->dtype || mf->device != st->orig->device)
                mf = mf->to(st->orig->device, st->orig->dtype);
            st->saved = st->orig;
            set_member(*m, st->name, Ops::mul(st->orig, mf));
        });
        module.register_forward_hook(
            [st](nn::Module* m, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>) {
                if (!st->saved) return;
                if (!find_mask(*m, st->name)) {
                    st->saved.reset();
                    return;
                }
                set_member(*m, st->name, st->saved);
                st->saved.reset();
            });
    }
    for (auto i : zero_idx) bytes[(size_t)i] = 0;
    auto t = Tensor::create(live->shape, Device(DeviceType::CPU), false, DataType::INT8);
    int8_t* d = reinterpret_cast<int8_t*>(t->storage->get_cpu_ptr());
    for (size_t i = 0; i < bytes.size(); ++i) d[i] = bytes[i];
    if (t->device != live->device) t = t->to(live->device);
    module.register_buffer(mask_key(name), t);
}

void l1_impl(nn::Module& module, const std::string& name, int64_t k) {
    auto live = get_member(module, name);
    if (!live) throw std::runtime_error("[litetorch Error] prune: param '" + name + "' not found");
    int64_t n = (int64_t)live->numel();
    auto cand = unpruned_candidates(module, name, n);
    if (k < 0 || k > (int64_t)cand.size())
        throw std::runtime_error("[litetorch Error] prune: amount out of range");
    auto wf = live->to(Device(DeviceType::CPU), DataType::FP32);
    float* d = (float*)wf->storage->get_cpu_ptr();
    apply_zero_idx(module, name, pick_l1(d, cand, k));
}

void random_impl(nn::Module& module, const std::string& name, int64_t k, int64_t seed) {
    auto live = get_member(module, name);
    if (!live) throw std::runtime_error("[litetorch Error] prune: param '" + name + "' not found");
    int64_t n = (int64_t)live->numel();
    auto cand = unpruned_candidates(module, name, n);
    if (k < 0 || k > (int64_t)cand.size())
        throw std::runtime_error("[litetorch Error] prune: amount out of range");
    apply_zero_idx(module, name, pick_random(cand, k, seed));
}

}

void l1_unstructured(nn::Module& module, const std::string& name, int64_t amount) {
    auto live = get_member(module, name);
    if (!live) throw std::runtime_error("[litetorch Error] prune: param '" + name + "' not found");
    int64_t n = (int64_t)live->numel();
    auto cand = unpruned_candidates(module, name, n);
    if (amount < 0 || amount > (int64_t)cand.size())
        throw std::runtime_error("[litetorch Error] prune: amount out of range");
    l1_impl(module, name, amount);
}

void l1_unstructured(nn::Module& module, const std::string& name, double amount) {
    if (!(amount >= 0.0 && amount <= 1.0))
        throw std::runtime_error("[litetorch Error] prune: amount fraction must be in [0, 1]");
    auto live = get_member(module, name);
    if (!live) throw std::runtime_error("[litetorch Error] prune: param '" + name + "' not found");
    int64_t n = (int64_t)live->numel();
    auto cand = unpruned_candidates(module, name, n);
    l1_impl(module, name, std::llround(amount * (double)cand.size()));
}

void random_unstructured(nn::Module& module, const std::string& name, int64_t amount, int64_t seed) {
    auto live = get_member(module, name);
    if (!live) throw std::runtime_error("[litetorch Error] prune: param '" + name + "' not found");
    int64_t n = (int64_t)live->numel();
    auto cand = unpruned_candidates(module, name, n);
    if (amount < 0 || amount > (int64_t)cand.size())
        throw std::runtime_error("[litetorch Error] prune: amount out of range");
    random_impl(module, name, amount, seed);
}

void random_unstructured(nn::Module& module, const std::string& name, double amount, int64_t seed) {
    if (!(amount >= 0.0 && amount <= 1.0))
        throw std::runtime_error("[litetorch Error] prune: amount fraction must be in [0, 1]");
    auto live = get_member(module, name);
    if (!live) throw std::runtime_error("[litetorch Error] prune: param '" + name + "' not found");
    int64_t n = (int64_t)live->numel();
    auto cand = unpruned_candidates(module, name, n);
    random_impl(module, name, std::llround(amount * (double)cand.size()), seed);
}

void remove(nn::Module& module, const std::string& name) {
    auto mask = find_mask(module, name);
    if (!mask) throw std::runtime_error("[litetorch Error] prune.remove: '" + name + "' is not pruned");
    auto live = get_member(module, name);
    if (!live) throw std::runtime_error("[litetorch Error] prune: param '" + name + "' not found");
    auto mf = mask;
    if (mf->dtype != live->dtype || mf->device != live->device)
        mf = mf->to(live->device, live->dtype);
    set_member(module, name, Ops::mul(live, mf));
    module.unregister_buffer(mask_key(name));
}

bool is_pruned(nn::Module& module, const std::string& name) {
    return find_mask(module, name) != nullptr;
}

}
}
