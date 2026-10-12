#include "litetorch/guided_decoding.h"
#include <regex>
#include <cctype>
#include <mutex>
#include <unordered_map>

namespace litetorch {
namespace nn {

namespace {

std::mutex g_regex_mutex;
std::unordered_map<std::string, std::pair<std::regex, bool>> g_regex_cache;

std::pair<std::regex, bool> get_cached_regex(const std::string& pattern) {
    std::lock_guard<std::mutex> lock(g_regex_mutex);
    auto it = g_regex_cache.find(pattern);
    if (it != g_regex_cache.end()) return it->second;
    if (g_regex_cache.size() > 64) g_regex_cache.clear();
    std::pair<std::regex, bool> entry;
    try {
        entry = {std::regex(pattern), true};
    } catch (...) {
        entry = {std::regex(), false};
    }
    g_regex_cache[pattern] = entry;
    return entry;
}

bool token_allowed(const std::string& candidate, const std::string& pattern, bool is_regex, bool is_prefix, const std::string& target_prefix, const std::pair<std::regex, bool>& re_entry) {
    if (is_regex) {
        if (!re_entry.second) return false;
        return std::regex_match(candidate, re_entry.first) || std::regex_search(candidate, re_entry.first);
    }
    if (pattern == "digits") {
        for (char c : candidate) {
            if (!std::isdigit(static_cast<unsigned char>(c))) return false;
        }
        return true;
    }
    if (is_prefix) {
        return target_prefix.rfind(candidate, 0) == 0 || candidate.rfind(target_prefix, 0) == 0;
    }
    return true;
}

}

GuidedDecoder::GuidedDecoder(const std::vector<std::string>& vocabulary) : vocab(vocabulary) {}

void GuidedDecoder::apply_mask(std::shared_ptr<Tensor> logits, const std::string& prefix, const std::string& target_pattern) {
    int64_t V = logits->shape[logits->shape.size() - 1];
    int64_t rows = logits->numel() / V;

    auto work = logits->to(Device(DeviceType::CPU, 0), DataType::FP32);
    if (!work->is_contiguous()) work = work->contiguous();
    float* work_ptr = work->data_ptr();

    bool is_regex = target_pattern.rfind("regex:", 0) == 0;
    std::string pattern = is_regex ? target_pattern.substr(6) : target_pattern;

    std::pair<std::regex, bool> re_entry;
    if (is_regex) re_entry = get_cached_regex(pattern);

    std::string target_prefix;
    bool is_prefix = false;
    if (!is_regex && pattern.rfind("prefix:", 0) == 0) {
        is_prefix = true;
        target_prefix = pattern.substr(7);
    }

    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t v = 0; v < V; ++v) {
            if (v >= static_cast<int64_t>(vocab.size())) continue;
            const std::string& tok_str = vocab[v];
            std::string candidate = prefix + tok_str;
            bool ok = token_allowed(candidate, pattern, is_regex, is_prefix, target_prefix, re_entry);
            if (!ok) work_ptr[r * V + v] = -1e9f;
        }
    }

    logits->copy_(work);
}

}
}
