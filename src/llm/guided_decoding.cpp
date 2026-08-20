#include "litetorch/guided_decoding.h"
#include <regex>
#include <cctype>

namespace litetorch {
namespace nn {

GuidedDecoder::GuidedDecoder(const std::vector<std::string>& vocabulary) : vocab(vocabulary) {}

void GuidedDecoder::apply_mask(std::shared_ptr<Tensor> logits, const std::string& prefix, const std::string& target_pattern) {
    int64_t vocab_size = logits->shape[logits->shape.size() - 1];
    float* logits_ptr = logits->data_ptr();
    
    bool is_regex = target_pattern.rfind("regex:", 0) == 0;
    std::string pattern = is_regex ? target_pattern.substr(6) : target_pattern;

    std::regex re;
    bool regex_ok = true;
    if (is_regex) {
        try {
            re = std::regex(pattern);
        } catch (...) {
            regex_ok = false;
        }
    }

    std::string target_prefix;
    bool is_prefix = false;
    if (!is_regex && pattern.rfind("prefix:", 0) == 0) {
        is_prefix = true;
        target_prefix = pattern.substr(7);
    }

    for (int64_t v = 0; v < vocab_size; ++v) {
        if (v >= static_cast<int64_t>(vocab.size())) continue;
        const std::string& tok_str = vocab[v];
        std::string candidate = prefix + tok_str;

        bool ok = true;
        if (is_regex) {
            if (!regex_ok) {
                ok = false;
            } else {
                ok = std::regex_match(candidate, re) || std::regex_search(candidate, re);
            }
        } else {
            if (pattern == "digits") {
                for (char c : candidate) {
                    if (!std::isdigit(c)) {
                        ok = false;
                        break;
                    }
                }
            } else if (is_prefix) {
                if (target_prefix.rfind(candidate, 0) != 0 && candidate.rfind(target_prefix, 0) != 0) {
                    ok = false;
                }
            }
        }

        if (!ok) {
            logits_ptr[v] = -1e9f;
        }
    }
}

}
}
