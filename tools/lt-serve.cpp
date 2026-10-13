// lt-serve: OpenAI-compatible HTTP serving for LiteTorch causal language models.
//
// Endpoints: GET /health, GET /v1/models,
//            POST /v1/completions, POST /v1/chat/completions (SSE streaming via "stream": true)
// Model: built-in GPT2B (Llama-style decoder). --arch gpt2b-tiny for tests,
//        or custom dims + --weights model.safetensors (best-effort name match).
// Tokenizer: --vocab vocab.txt (one token per line, whitespace split) or
//        --tokenizer-mode char (byte-level fallback).
#include "httplib.h"
#include "json.hpp"
#include "litetorch/nn.h"
#include "litetorch/generate.h"
#include "litetorch/serialization.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

using json = nlohmann::json;
namespace lt = litetorch;

struct ServeArgs {
    std::string host = "127.0.0.1";
    int port = 8000;
    std::string arch = "gpt2b-tiny";
    std::string weights;
    std::string vocab;
    std::string tokenizer_mode = "word";
    int vocab_size = 256;
    int d_model = 64;
    int n_layers = 2;
    int n_q_heads = 4;
    int n_kv_heads = 2;
    int d_ff = 128;
    int64_t max_new_tokens = 32;
    int64_t eos = 2;
    int threads = 4;
};

static void usage() {
    std::fprintf(stderr,
        "usage: lt-serve [--host H] [--port P] [--arch gpt2b-tiny|gpt2b]\n"
        "               [--weights model.safetensors] [--vocab vocab.txt]\n"
        "               [--tokenizer-mode word|char] [--vocab-size N] [--d-model N]\n"
        "               [--n-layers N] [--n-q-heads N] [--n-kv-heads N] [--d-ff N]\n"
        "               [--max-new-tokens N] [--eos ID] [--threads N]\n");
}

static bool parse_args(int argc, char** argv, ServeArgs& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](std::string& out) {
            if (i + 1 >= argc) { usage(); return false; }
            out = argv[++i]; return true;
        };
        auto needi = [&](int& out) {
            std::string s; if (!need(s)) return false;
            out = std::atoi(s.c_str()); return true;
        };
        if (k == "--host") { if (!need(a.host)) return false; }
        else if (k == "--port") { if (!needi(a.port)) return false; }
        else if (k == "--arch") { if (!need(a.arch)) return false; }
        else if (k == "--weights") { if (!need(a.weights)) return false; }
        else if (k == "--vocab") { if (!need(a.vocab)) return false; }
        else if (k == "--tokenizer-mode") { if (!need(a.tokenizer_mode)) return false; }
        else if (k == "--vocab-size") { if (!needi(a.vocab_size)) return false; }
        else if (k == "--d-model") { if (!needi(a.d_model)) return false; }
        else if (k == "--n-layers") { if (!needi(a.n_layers)) return false; }
        else if (k == "--n-q-heads") { if (!needi(a.n_q_heads)) return false; }
        else if (k == "--n-kv-heads") { if (!needi(a.n_kv_heads)) return false; }
        else if (k == "--d-ff") { if (!needi(a.d_ff)) return false; }
        else if (k == "--max-new-tokens") { int v; if (!needi(v)) return false; a.max_new_tokens = v; }
        else if (k == "--eos") { int v; if (!needi(v)) return false; a.eos = v; }
        else if (k == "--threads") { if (!needi(a.threads)) return false; }
        else if (k == "--help" || k == "-h") { usage(); return false; }
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); usage(); return false; }
    }
    return true;
}

class SimpleTokenizer {
public:
    bool load(const std::string& path, const std::string& mode, int vocab_size) {
        mode_ = mode;
        vocab_size_ = vocab_size;
        if (mode_ == "word" && !path.empty()) {
            std::ifstream f(path);
            if (!f) return false;
            std::string line;
            while (std::getline(f, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                tok2id_[line] = (int64_t)id2tok_.size();
                id2tok_.push_back(line);
            }
        }
        return true;
    }

    std::vector<int64_t> encode(const std::string& text) const {
        std::vector<int64_t> out;
        if (mode_ == "char" || id2tok_.empty()) {
            for (unsigned char c : text)
                out.push_back((int64_t)(c % (unsigned)vocab_size_));
            return out;
        }
        std::istringstream ss(text);
        std::string w;
        while (ss >> w) {
            auto it = tok2id_.find(w);
            out.push_back(it == tok2id_.end() ? 0 : it->second);
        }
        if (out.empty()) out.push_back(0);
        return out;
    }

    std::string decode(const std::vector<int64_t>& ids) const {
        if (mode_ == "char" || id2tok_.empty()) {
            std::string s;
            for (int64_t id : ids) {
                if (id >= 0 && id < (int64_t)id2tok_.size()) s += id2tok_[id];
            }
            return s;
        }
        std::string s;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) s += " ";
            int64_t id = ids[i];
            s += (id >= 0 && id < (int64_t)id2tok_.size()) ? id2tok_[id] : "<unk>";
        }
        return s;
    }

    std::string decode_one(int64_t id) const {
        if (id >= 0 && id < (int64_t)id2tok_.size()) return id2tok_[id];
        if (mode_ == "char" || id2tok_.empty()) return "";
        return "<unk>";
    }

private:
    std::string mode_ = "word";
    int vocab_size_ = 256;
    std::vector<std::string> id2tok_;
    std::unordered_map<std::string, int64_t> tok2id_;
};

static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; std::snprintf(b, 8, "\\u%04x", c); o += b; }
                else o += c;
        }
    }
    return o;
}

static std::string new_id(const char* prefix) {
    static std::atomic<uint64_t> ctr{0};
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    char b[64];
    std::snprintf(b, 64, "%s%lld-%llu", prefix, (long long)now, (unsigned long long)++ctr);
    return b;
}

struct Server {
    ServeArgs args;
    SimpleTokenizer tok;
    std::shared_ptr<lt::nn::Module> model;
    std::mutex model_mu;
    std::string model_name = "lt-serve";

    bool init() {
        if (!tok.load(args.vocab, args.tokenizer_mode, args.vocab_size)) {
            std::fprintf(stderr, "lt-serve: cannot read vocab file %s\n", args.vocab.c_str());
            return false;
        }
        int vs = args.vocab_size, dm = args.d_model, nl = args.n_layers;
        int nqh = args.n_q_heads, nkh = args.n_kv_heads, dff = args.d_ff;
        if (args.arch == "gpt2b-tiny") { vs = 256; dm = 64; nl = 2; nqh = 4; nkh = 2; dff = 128; }
        else if (args.arch != "gpt2b") {
            std::fprintf(stderr, "lt-serve: unknown arch %s\n", args.arch.c_str());
            return false;
        }
        model = std::make_shared<lt::nn::GPT2B>(vs, dm, nl, nqh, nkh, dff);
        if (!args.weights.empty()) {
            auto ckpt = lt::load_safetensors(args.weights);
            auto sd = model->state_dict();
            int loaded = 0;
            for (auto& kv : sd) {
                auto it = ckpt.find(kv.first);
                if (it == ckpt.end()) continue;
                if (it->second->shape != kv.second->shape) {
                    std::fprintf(stderr, "lt-serve: shape mismatch for %s, skipping\n", kv.first.c_str());
                    continue;
                }
                std::vector<float> v = it->second->to_vector();
                auto dst = kv.second;
                float* p = dst->data_ptr();
                std::copy(v.begin(), v.end(), p);
                ++loaded;
            }
            std::fprintf(stderr, "lt-serve: loaded %d/%zu tensors from %s\n",
                         loaded, sd.size(), args.weights.c_str());
        }
        return true;
    }

    lt::nn::GenerateConfig config_from(const json& body, int64_t fallback_max) {
        lt::nn::GenerateConfig cfg;
        cfg.max_new_tokens = fallback_max;
        if (body.contains("max_tokens") && body["max_tokens"].is_number())
            cfg.max_new_tokens = body["max_tokens"].get<int64_t>();
        if (body.contains("max_completion_tokens") && body["max_completion_tokens"].is_number())
            cfg.max_new_tokens = body["max_completion_tokens"].get<int64_t>();
        if (cfg.max_new_tokens <= 0) cfg.max_new_tokens = fallback_max;
        if (body.contains("temperature") && body["temperature"].is_number()) {
            cfg.temperature = body["temperature"].get<float>();
            cfg.do_sample = cfg.temperature > 0;
        }
        if (body.contains("top_p") && body["top_p"].is_number())
            cfg.top_p = body["top_p"].get<float>();
        if (body.contains("top_k") && body["top_k"].is_number())
            cfg.top_k = body["top_k"].get<int64_t>();
        if (body.contains("seed") && body["seed"].is_number())
            cfg.seed = body["seed"].get<uint64_t>();
        cfg.eos_token_id = args.eos;
        if (body.contains("stop")) {
            if (body["stop"].is_string()) {
                auto ids = tok.encode(body["stop"].get<std::string>());
                cfg.stop_sequences.push_back(ids);
            } else if (body["stop"].is_array()) {
                for (auto& s : body["stop"])
                    if (s.is_string()) cfg.stop_sequences.push_back(tok.encode(s.get<std::string>()));
            }
        }
        return cfg;
    }

    std::vector<int64_t> run_generate(const std::vector<int64_t>& prompt,
                                      const lt::nn::GenerateConfig& cfg,
                                      lt::nn::TokenCallback cb) {
        std::lock_guard<std::mutex> l(model_mu);
        return lt::nn::generate_stream(model, prompt, cfg, cb);
    }

    void handle_completions(const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"error\":\"invalid json\"}", "application/json"); return; }
        if (!body.contains("prompt")) {
            res.status = 400; res.set_content("{\"error\":\"missing prompt\"}", "application/json"); return;
        }
        std::string prompt;
        if (body["prompt"].is_string()) prompt = body["prompt"].get<std::string>();
        else if (body["prompt"].is_array() && !body["prompt"].empty() && body["prompt"][0].is_string())
            prompt = body["prompt"][0].get<std::string>();
        bool stream = body.value("stream", false);
        auto cfg = config_from(body, args.max_new_tokens);
        auto toks = tok.encode(prompt);
        std::string id = new_id("cmpl-");
        long long created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        if (!stream) {
            auto gen = run_generate(toks, cfg, nullptr);
            std::string text = tok.decode(gen);
            std::string finish = gen.size() < (size_t)cfg.max_new_tokens ? "stop" : "length";
            char buf[4096];
            std::snprintf(buf, sizeof(buf),
                "{\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%lld,\"model\":\"%s\","
                "\"choices\":[{\"text\":\"%s\",\"index\":0,\"finish_reason\":\"%s\"}],"
                "\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%zu,\"total_tokens\":%zu}}",
                id.c_str(), created, model_name.c_str(), json_escape(text).c_str(), finish.c_str(),
                toks.size(), gen.size(), toks.size() + gen.size());
            res.set_content(buf, "application/json");
            return;
        }

        struct Q { std::mutex m; std::condition_variable cv; std::queue<std::string> q; bool done = false; };
        auto q = std::make_shared<Q>();
        std::thread gen_thr([this, toks, cfg, q]() {
            run_generate(toks, cfg, [&](int64_t t) {
                std::string piece = tok.decode_one(t);
                std::string chunk = "data: {\"id\":\"x\",\"object\":\"text_completion\",\"choices\":"
                    "[{\"text\":\"" + json_escape(piece) + "\",\"index\":0,\"finish_reason\":null}]}\n\n";
                { std::lock_guard<std::mutex> l(q->m); q->q.push(chunk); }
                q->cv.notify_one();
                return true;
            });
            { std::lock_guard<std::mutex> l(q->m); q->done = true; }
            q->cv.notify_one();
        });
        res.set_chunked_content_provider("text/event-stream",
            [q](size_t, httplib::DataSink& sink) -> bool {
                std::unique_lock<std::mutex> l(q->m);
                q->cv.wait(l, [&] { return !q->q.empty() || q->done; });
                while (!q->q.empty()) {
                    std::string c = q->q.front(); q->q.pop();
                    l.unlock();
                    if (!sink.write(c.data(), c.size())) return false;
                    l.lock();
                }
                if (q->done) { sink.write("data: [DONE]\n\n", 13); return false; }
                return true;
            });
        gen_thr.detach();
    }

    void handle_chat(const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"error\":\"invalid json\"}", "application/json"); return; }
        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400; res.set_content("{\"error\":\"missing messages\"}", "application/json"); return;
        }
        std::string prompt;
        for (auto& m : body["messages"]) {
            std::string role = m.value("role", "user");
            std::string content = m.value("content", "");
            if (!prompt.empty()) prompt += "\n";
            prompt += role + ": " + content;
        }
        bool stream = body.value("stream", false);
        auto cfg = config_from(body, args.max_new_tokens);
        auto toks = tok.encode(prompt);
        std::string id = new_id("chatcmpl-");
        long long created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        if (!stream) {
            auto gen = run_generate(toks, cfg, nullptr);
            std::string text = tok.decode(gen);
            std::string finish = gen.size() < (size_t)cfg.max_new_tokens ? "stop" : "length";
            std::string out = "{\"id\":\"" + id + "\",\"object\":\"chat.completion\",\"created\":" +
                std::to_string(created) + ",\"model\":\"" + model_name + "\",\"choices\":[{\"index\":0,"
                "\"message\":{\"role\":\"assistant\",\"content\":\"" + json_escape(text) +
                "\"},\"finish_reason\":\"" + finish + "\"}],\"usage\":{\"prompt_tokens\":" +
                std::to_string(toks.size()) + ",\"completion_tokens\":" + std::to_string(gen.size()) +
                ",\"total_tokens\":" + std::to_string(toks.size() + gen.size()) + "}}";
            res.set_content(out, "application/json");
            return;
        }

        struct Q { std::mutex m; std::condition_variable cv; std::queue<std::string> q; bool done = false; };
        auto q = std::make_shared<Q>();
        std::string role_chunk = "data: {\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\","
            "\"created\":" + std::to_string(created) + ",\"model\":\"" + model_name +
            "\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n";
        std::thread gen_thr([this, toks, cfg, q, id, created]() {
            run_generate(toks, cfg, [&](int64_t t) {
                std::string piece = tok.decode_one(t);
                std::string chunk = "data: {\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\","
                    "\"created\":" + std::to_string(created) + ",\"model\":\"" + model_name +
                    "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + json_escape(piece) +
                    "\"},\"finish_reason\":null}]}\n\n";
                { std::lock_guard<std::mutex> l(q->m); q->q.push(chunk); }
                q->cv.notify_one();
                return true;
            });
            { std::lock_guard<std::mutex> l(q->m); q->done = true; }
            q->cv.notify_one();
        });
        bool first_sent = false;
        res.set_chunked_content_provider("text/event-stream",
            [q, id, created, role_chunk, first_sent](size_t, httplib::DataSink& sink) mutable -> bool {
                std::unique_lock<std::mutex> l(q->m);
                if (!first_sent) {
                    first_sent = true; l.unlock();
                    if (!sink.write(role_chunk.data(), role_chunk.size())) return false;
                    l.lock();
                }
                q->cv.wait(l, [&] { return !q->q.empty() || q->done; });
                while (!q->q.empty()) {
                    std::string c = q->q.front(); q->q.pop();
                    l.unlock();
                    if (!sink.write(c.data(), c.size())) return false;
                    l.lock();
                }
                if (q->done) { sink.write("data: [DONE]\n\n", 13); return false; }
                return true;
            });
        gen_thr.detach();
    }
};

int main(int argc, char** argv) {
    ServeArgs args;
    if (!parse_args(argc, argv, args)) return 1;
    Server srv;
    srv.args = args;
    if (!srv.init()) return 1;

    httplib::Server svr;
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        std::string b = "{\"object\":\"list\",\"data\":[{\"id\":\"" + srv.model_name +
            "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"litetorch\"}]}";
        res.set_content(b, "application/json");
    });
    svr.Post("/v1/completions", [&](const httplib::Request& rq, httplib::Response& rs) {
        srv.handle_completions(rq, rs);
    });
    svr.Post("/v1/chat/completions", [&](const httplib::Request& rq, httplib::Response& rs) {
        srv.handle_chat(rq, rs);
    });
    std::fprintf(stderr, "lt-serve listening on %s:%d (arch=%s)\n",
                 args.host.c_str(), args.port, args.arch.c_str());
    svr.listen(args.host.c_str(), args.port);
    return 0;
}
