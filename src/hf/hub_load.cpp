#include "litetorch/hf.h"
#include "litetorch/hub.h"
#include "litetorch/serialization.h"
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>

namespace litetorch {
namespace hf {

namespace {

// Minimal JSON helpers: no third-party JSON dependency in the framework, and
// config.json / model.safetensors.index.json only need top-level scalar fields
// plus one string->string map (weight_map).

bool is_local_dir(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("[litetorch Error] hf: cannot read " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct JsonScan {
    const char* p;
    const char* end;
    JsonScan(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}

    void ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }

    std::string str() {
        ws();
        if (p >= end || *p != '"') throw std::runtime_error("[litetorch Error] hf: bad json string");
        ++p;
        std::string out;
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                ++p;
                char e = *p;
                out += (e == 'n' ? '\n' : e == 't' ? '\t' : e);
            } else {
                out += *p;
            }
            ++p;
        }
        if (p < end) ++p;
        return out;
    }

    std::string scalar() {
        ws();
        if (p < end && *p == '"') return str();
        const char* s = p;
        while (p < end && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            ++p;
        return std::string(s, p);
    }

    void skip_value() {
        ws();
        if (p >= end) return;
        if (*p == '"') { str(); return; }
        if (*p == '{' || *p == '[') {
            char open = *p, close = (*p == '{') ? '}' : ']';
            int depth = 0;
            bool in_s = false;
            while (p < end) {
                char c = *p++;
                if (in_s) {
                    if (c == '\\' && p < end) ++p;
                    else if (c == '"') in_s = false;
                } else if (c == '"') {
                    in_s = true;
                } else if (c == open) {
                    ++depth;
                } else if (c == close) {
                    if (--depth == 0) return;
                }
            }
            return;
        }
        scalar();
    }
};

std::map<std::string, std::string> parse_top_scalars(const std::string& text) {
    std::map<std::string, std::string> out;
    JsonScan j(text);
    j.ws();
    if (j.p >= j.end || *j.p != '{') return out;
    ++j.p;
    int depth = 1;
    while (j.p < j.end && depth > 0) {
        j.ws();
        if (j.p >= j.end) break;
        if (*j.p == '}') { --depth; ++j.p; continue; }
        if (*j.p == ',') { ++j.p; continue; }
        if (*j.p != '"') { j.skip_value(); continue; }
        std::string key = j.str();
        j.ws();
        if (j.p < j.end && *j.p == ':') ++j.p;
        if (depth == 1) {
            j.ws();
            if (j.p < j.end && *j.p != '{' && *j.p != '[')
                out[key] = j.scalar();
            else
                j.skip_value();
        } else {
            j.skip_value();
        }
    }
    return out;
}

std::map<std::string, std::string> parse_string_map(const std::string& text,
                                                    const std::string& obj_key) {
    std::map<std::string, std::string> out;
    JsonScan j(text);
    j.ws();
    if (j.p >= j.end || *j.p != '{') return out;
    ++j.p;
    while (j.p < j.end) {
        j.ws();
        if (j.p >= j.end) break;
        if (*j.p == '}') break;
        if (*j.p == ',') { ++j.p; continue; }
        if (*j.p != '"') { j.skip_value(); continue; }
        std::string key = j.str();
        j.ws();
        if (j.p < j.end && *j.p == ':') ++j.p;
        if (key == obj_key) {
            j.ws();
            if (j.p < j.end && *j.p == '{') {
                ++j.p;
                while (j.p < j.end) {
                    j.ws();
                    if (j.p >= j.end || *j.p == '}') break;
                    if (*j.p == ',') { ++j.p; continue; }
                    if (*j.p != '"') { j.skip_value(); continue; }
                    std::string k = j.str();
                    j.ws();
                    if (j.p < j.end && *j.p == ':') ++j.p;
                    out[k] = j.str();
                }
                if (j.p < j.end && *j.p == '}') ++j.p;
            } else {
                j.skip_value();
            }
        } else {
            j.skip_value();
        }
    }
    return out;
}

int64_t to_i64(const std::map<std::string, std::string>& m, const std::string& k) {
    auto it = m.find(k);
    if (it == m.end() || it->second.empty() || it->second == "null") return 0;
    try {
        return std::stoll(it->second);
    } catch (...) {
        return 0;
    }
}

double to_f64(const std::map<std::string, std::string>& m, const std::string& k, double dflt) {
    auto it = m.find(k);
    if (it == m.end() || it->second.empty() || it->second == "null") return dflt;
    try {
        return std::stod(it->second);
    } catch (...) {
        return dflt;
    }
}

std::string to_str(const std::map<std::string, std::string>& m, const std::string& k) {
    auto it = m.find(k);
    if (it == m.end() || it->second == "null") return "";
    return it->second;
}

std::string resolve_file(const std::string& repo_or_dir, const std::string& filename,
                         const std::string& revision) {
    if (is_local_dir(repo_or_dir)) {
        std::string p = repo_or_dir + "/" + filename;
        if (!file_exists(p))
            throw std::runtime_error("[litetorch Error] hf: missing file " + p);
        return p;
    }
    std::string url = "https://huggingface.co/" + repo_or_dir + "/resolve/" + revision + "/" + filename;
    std::string sub = repo_or_dir;
    for (char& c : sub)
        if (c == '/') c = '_';
    std::string dst = hub::get_dir() + "/hf/" + sub + "/" + revision + "/" + filename;
    if (file_exists(dst)) return dst;
    return hub::download_url_to_file(url, dst, false);
}

struct NameRule {
    std::regex pattern;
    std::string replacement;
};

std::vector<NameRule> rules_for(const std::string& model_type) {
    std::vector<NameRule> r;
    if (model_type == "llama" || model_type == "mistral" || model_type == "qwen2") {
        r.push_back({std::regex(R"(^model\.layers\.(\d+)\.self_attn\.([a-z_]+)\.(weight|bias)$)"),
                     "layers.$1.attn.$2.$3"});
        r.push_back({std::regex(R"(^model\.layers\.(\d+)\.mlp\.([a-z_]+)\.(weight|bias)$)"),
                     "layers.$1.mlp.$2.$3"});
        r.push_back({std::regex(R"(^model\.layers\.(\d+)\.input_layernorm\.(weight|bias)$)"),
                     "layers.$1.attn_norm.$2"});
        r.push_back({std::regex(R"(^model\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)$)"),
                     "layers.$1.ffn_norm.$2"});
    } else if (model_type == "gpt2") {
        r.push_back({std::regex(R"(^transformer\.h\.(\d+)\.attn\.([a-z_]+)\.(weight|bias)$)"),
                     "layers.$1.attn.$2.$3"});
        r.push_back({std::regex(R"(^transformer\.h\.(\d+)\.ln_1\.(weight|bias)$)"),
                     "layers.$1.attn_norm.$2"});
        r.push_back({std::regex(R"(^transformer\.h\.(\d+)\.ln_2\.(weight|bias)$)"),
                     "layers.$1.ffn_norm.$2"});
        r.push_back({std::regex(R"(^transformer\.h\.(\d+)\.mlp\.([a-z_]+)\.(weight|bias)$)"),
                     "layers.$1.mlp.$2.$3"});
    } else if (model_type == "bert") {
        r.push_back({std::regex(R"(^bert\.encoder\.layer\.(\d+)\.attention\.self\.(query|key|value)\.(weight|bias)$)"),
                     "layers.$1.attn.$2_proj.$3"});
        r.push_back({std::regex(R"(^bert\.encoder\.layer\.(\d+)\.attention\.output\.dense\.(weight|bias)$)"),
                     "layers.$1.attn.o_proj.$2"});
        r.push_back({std::regex(R"(^bert\.encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.(weight|bias)$)"),
                     "layers.$1.attn_norm.$2"});
        r.push_back({std::regex(R"(^bert\.encoder\.layer\.(\d+)\.intermediate\.dense\.(weight|bias)$)"),
                     "layers.$1.mlp.up_proj.$2"});
        r.push_back({std::regex(R"(^bert\.encoder\.layer\.(\d+)\.output\.dense\.(weight|bias)$)"),
                     "layers.$1.mlp.down_proj.$2"});
        r.push_back({std::regex(R"(^bert\.encoder\.layer\.(\d+)\.output\.LayerNorm\.(weight|bias)$)"),
                     "layers.$1.ffn_norm.$2"});
    }
    return r;
}

}

ModelConfig load_config(const std::string& repo_or_dir, const std::string& revision) {
    std::string path = resolve_file(repo_or_dir, "config.json", revision);
    auto kv = parse_top_scalars(read_file(path));
    ModelConfig c;
    c.model_type = to_str(kv, "model_type");
    c.hidden_size = to_i64(kv, "hidden_size");
    c.num_hidden_layers = to_i64(kv, "num_hidden_layers");
    c.num_attention_heads = to_i64(kv, "num_attention_heads");
    c.num_key_value_heads = to_i64(kv, "num_key_value_heads");
    if (c.num_key_value_heads == 0) c.num_key_value_heads = c.num_attention_heads;
    c.intermediate_size = to_i64(kv, "intermediate_size");
    c.vocab_size = to_i64(kv, "vocab_size");
    c.max_position_embeddings = to_i64(kv, "max_position_embeddings");
    c.hidden_act = to_str(kv, "hidden_act");
    c.rms_norm_eps = to_f64(kv, "rms_norm_eps", 1e-6);
    c.layer_norm_eps = to_f64(kv, "layer_norm_eps", 1e-5);
    c.n_layer = to_i64(kv, "n_layer");
    c.n_head = to_i64(kv, "n_head");
    c.n_embd = to_i64(kv, "n_embd");
    c.n_positions = to_i64(kv, "n_positions");
    return c;
}

PretrainedBundle from_pretrained(const std::string& repo_id, const std::string& revision) {
    PretrainedBundle b;
    b.config = load_config(repo_id, revision);

    std::vector<std::string> files;
    bool sharded = false;
    try {
        std::string index_path = resolve_file(repo_id, "model.safetensors.index.json", revision);
        auto wmap = parse_string_map(read_file(index_path), "weight_map");
        std::map<std::string, bool> seen;
        for (auto& kv : wmap) {
            if (!seen[kv.second]) {
                seen[kv.second] = true;
                files.push_back(kv.second);
            }
        }
        sharded = !files.empty();
    } catch (...) {
    }
    if (!sharded) {
        files.push_back(resolve_file(repo_id, "model.safetensors", revision));
    } else if (!is_local_dir(repo_id)) {
        for (auto& f : files) {
            std::string url = "https://huggingface.co/" + repo_id + "/resolve/" + revision + "/" + f;
            std::string sub = repo_id;
            for (char& c : sub)
                if (c == '/') c = '_';
            std::string dst = hub::get_dir() + "/hf/" + sub + "/" + revision + "/" + f;
            f = file_exists(dst) ? dst : hub::download_url_to_file(url, dst, false);
        }
    } else {
        for (auto& f : files) f = repo_id + "/" + f;
    }

    for (auto& f : files) {
        auto part = load_safetensors(f);
        for (auto& kv : part) b.state_dict[kv.first] = kv.second;
    }
    return b;
}

std::map<std::string, std::string> weight_name_map(const std::string& model_type) {
    std::map<std::string, std::string> m;
    if (model_type == "llama" || model_type == "mistral" || model_type == "qwen2") {
        m["model.embed_tokens.weight"] = "embed.weight";
        m["model.norm.weight"] = "final_norm.weight";
        m["lm_head.weight"] = "lm_head.weight";
    } else if (model_type == "gpt2") {
        m["transformer.wte.weight"] = "embed.weight";
        m["transformer.wpe.weight"] = "pos_embed.weight";
        m["transformer.ln_f.weight"] = "final_norm.weight";
        m["transformer.ln_f.bias"] = "final_norm.bias";
        m["lm_head.weight"] = "lm_head.weight";
    } else if (model_type == "bert") {
        m["bert.embeddings.word_embeddings.weight"] = "embed.word.weight";
        m["bert.embeddings.position_embeddings.weight"] = "embed.pos.weight";
        m["bert.embeddings.token_type_embeddings.weight"] = "embed.token_type.weight";
        m["bert.embeddings.LayerNorm.weight"] = "embed.norm.weight";
        m["bert.embeddings.LayerNorm.bias"] = "embed.norm.bias";
        m["bert.pooler.dense.weight"] = "pooler.weight";
        m["bert.pooler.dense.bias"] = "pooler.bias";
    }
    return m;
}

std::map<std::string, std::shared_ptr<Tensor>> remap_state_dict(
    const std::map<std::string, std::shared_ptr<Tensor>>& sd, const std::string& model_type) {
    auto exact = weight_name_map(model_type);
    auto rules = rules_for(model_type);
    std::map<std::string, std::shared_ptr<Tensor>> out;
    for (auto& kv : sd) {
        std::string name = kv.first;
        auto it = exact.find(name);
        if (it != exact.end()) {
            name = it->second;
        } else {
            for (auto& r : rules) {
                std::string rep = std::regex_replace(name, r.pattern, r.replacement);
                if (rep != name) {
                    name = rep;
                    break;
                }
            }
        }
        out[name] = kv.second;
    }
    return out;
}

}
}
