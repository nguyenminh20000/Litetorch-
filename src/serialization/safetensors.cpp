#include "litetorch/serialization.h"
#include "litetorch/nn.h"
#include <fstream>
#include <stdexcept>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <regex>
#include <sys/stat.h>
#include <sys/types.h>

namespace litetorch {

namespace {

std::string dtype_to_safetensors(DataType dtype) {
    if (dtype == DataType::FP32) return "F32";
    if (dtype == DataType::FP16) return "F16";
    if (dtype == DataType::BF16) return "BF16";
    if (dtype == DataType::FP64) return "F64";
    if (dtype == DataType::INT32) return "I32";
    if (dtype == DataType::INT16) return "I16";
    if (dtype == DataType::INT8) return "I8";
    throw std::runtime_error("[litetorch Error] safetensors: unsupported dtype for serialization");
}

DataType safetensors_to_dtype(const std::string& s) {
    if (s == "F32") return DataType::FP32;
    if (s == "F16") return DataType::FP16;
    if (s == "BF16") return DataType::BF16;
    if (s == "F64") return DataType::FP64;
    if (s == "I32") return DataType::INT32;
    if (s == "I16") return DataType::INT16;
    if (s == "I8") return DataType::INT8;
    if (s == "U8") return DataType::INT8;
    throw std::runtime_error("[litetorch Error] safetensors: unsupported dtype string: " + s);
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

struct ParsedTensor {
    std::string dtype_str;
    std::vector<int64_t> shape;
    uint64_t data_start;
    uint64_t data_end;
};

class JsonParser {
public:
    const char* p;
    const char* end;
    JsonParser(const char* data, size_t len) : p(data), end(data + len) {}

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }

    void expect(char c) {
        skip_ws();
        if (p >= end || *p != c) throw std::runtime_error("[litetorch Error] safetensors: malformed JSON header");
        p++;
    }

    std::string parse_string() {
        skip_ws();
        expect('"');
        std::string out;
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                p++;
                if (*p == '"') out += '"';
                else if (*p == '\\') out += '\\';
                else if (*p == 'n') out += '\n';
                else if (*p == 'u' && p + 4 < end) {
                    uint32_t cp = 0;
                    bool ok = true;
                    for (int i = 1; i <= 4; i++) {
                        char c = p[i];
                        cp <<= 4;
                        if (c >= '0' && c <= '9') cp |= static_cast<uint32_t>(c - '0');
                        else if (c >= 'a' && c <= 'f') cp |= static_cast<uint32_t>(c - 'a' + 10);
                        else if (c >= 'A' && c <= 'F') cp |= static_cast<uint32_t>(c - 'A' + 10);
                        else ok = false;
                    }
                    if (ok) {
                        p += 4;
                        if (cp >= 0xD800 && cp <= 0xDBFF && p + 6 < end && p[1] == '\\' && p[2] == 'u') {
                            uint32_t lo = 0;
                            bool lo_ok = true;
                            for (int i = 3; i <= 6; i++) {
                                char c = p[i];
                                lo <<= 4;
                                if (c >= '0' && c <= '9') lo |= static_cast<uint32_t>(c - '0');
                                else if (c >= 'a' && c <= 'f') lo |= static_cast<uint32_t>(c - 'a' + 10);
                                else if (c >= 'A' && c <= 'F') lo |= static_cast<uint32_t>(c - 'A' + 10);
                                else lo_ok = false;
                            }
                            if (lo_ok && lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                p += 6;
                            }
                        }
                        if (cp < 0x80) out += static_cast<char>(cp);
                        else if (cp < 0x800) {
                            out += static_cast<char>(0xC0 | (cp >> 6));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        } else if (cp < 0x10000) {
                            out += static_cast<char>(0xE0 | (cp >> 12));
                            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            out += static_cast<char>(0xF0 | (cp >> 18));
                            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                    } else {
                        out += 'u';
                    }
                } else out += *p;
            } else {
                out += *p;
            }
            p++;
        }
        expect('"');
        return out;
    }

    int64_t parse_int() {
        skip_ws();
        bool neg = false;
        if (p < end && *p == '-') { neg = true; p++; }
        int64_t v = 0;
        while (p < end && *p >= '0' && *p <= '9') {
            v = v * 10 + (*p - '0');
            p++;
        }
        return neg ? -v : v;
    }

    std::vector<int64_t> parse_int_array() {
        std::vector<int64_t> out;
        expect('[');
        skip_ws();
        if (p < end && *p == ']') { p++; return out; }
        while (true) {
            out.push_back(parse_int());
            skip_ws();
            if (p < end && *p == ',') { p++; continue; }
            break;
        }
        expect(']');
        return out;
    }

    void skip_value() {
        skip_ws();
        if (p >= end) return;
        if (*p == '"') { parse_string(); return; }
        if (*p == '{') {
            p++;
            skip_ws();
            if (p < end && *p == '}') { p++; return; }
            while (true) {
                parse_string();
                expect(':');
                skip_value();
                skip_ws();
                if (p < end && *p == ',') { p++; continue; }
                break;
            }
            expect('}');
            return;
        }
        if (*p == '[') {
            p++;
            skip_ws();
            if (p < end && *p == ']') { p++; return; }
            while (true) {
                skip_value();
                skip_ws();
                if (p < end && *p == ',') { p++; continue; }
                break;
            }
            expect(']');
            return;
        }
        while (p < end && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\n') p++;
    }

    ParsedTensor parse_tensor_entry() {
        ParsedTensor t;
        expect('{');
        while (true) {
            std::string key = parse_string();
            expect(':');
            if (key == "dtype") {
                t.dtype_str = parse_string();
            } else if (key == "shape") {
                t.shape = parse_int_array();
            } else if (key == "data_offsets") {
                auto offs = parse_int_array();
                if (offs.size() != 2) throw std::runtime_error("[litetorch Error] safetensors: data_offsets must have 2 elements");
                t.data_start = static_cast<uint64_t>(offs[0]);
                t.data_end = static_cast<uint64_t>(offs[1]);
            } else {
                skip_value();
            }
            skip_ws();
            if (p < end && *p == ',') { p++; continue; }
            break;
        }
        expect('}');
        return t;
    }

    std::map<std::string, ParsedTensor> parse_header() {
        std::map<std::string, ParsedTensor> out;
        expect('{');
        skip_ws();
        if (p < end && *p == '}') { p++; return out; }
        while (true) {
            std::string key = parse_string();
            expect(':');
            if (key == "__metadata__") {
                skip_value();
            } else {
                out[key] = parse_tensor_entry();
            }
            skip_ws();
            if (p < end && *p == ',') { p++; continue; }
            break;
        }
        expect('}');
        return out;
    }
};

const void* tensor_bytes(std::shared_ptr<Tensor> t, std::shared_ptr<Tensor>& holder) {
    if (!t->is_contiguous()) {
        holder = t->contiguous();
        t = holder;
    }
    t->storage->get_cpu_ptr();
    const char* base = static_cast<const char*>(static_cast<const void*>(t->storage->cpu_data));
    return static_cast<const void*>(base + t->offset * t->storage->element_size());
}

}

void save_safetensors(const std::map<std::string, std::shared_ptr<Tensor>>& tensors, const std::string& filepath) {
    std::string header = "{";
    uint64_t offset = 0;
    bool first = true;
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> ordered;
    for (const auto& kv : tensors) {
        if (!kv.second) throw std::runtime_error("[litetorch Error] safetensors: null tensor for key " + kv.first);
        ordered.push_back(kv);
    }
    for (const auto& kv : ordered) {
        auto t = kv.second;
        std::string dt = dtype_to_safetensors(t->dtype);
        size_t nbytes = t->numel() * t->storage->element_size();
        if (!first) header += ",";
        first = false;
        header += "\"" + json_escape(kv.first) + "\":{\"dtype\":\"" + dt + "\",\"shape\":[";
        for (size_t i = 0; i < t->shape.size(); i++) {
            if (i) header += ",";
            header += std::to_string(t->shape[i]);
        }
        header += "],\"data_offsets\":[" + std::to_string(offset) + "," + std::to_string(offset + nbytes) + "]}";
        offset += nbytes;
    }
    header += "}";

    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("[litetorch Error] safetensors: cannot open file for writing: " + filepath);
    uint64_t hlen = header.size();
    unsigned char hlen_le[8];
    for (int i = 0; i < 8; i++) hlen_le[i] = static_cast<unsigned char>((hlen >> (i * 8)) & 0xFF);
    out.write(reinterpret_cast<const char*>(hlen_le), 8);
    out.write(header.data(), header.size());
    for (const auto& kv : ordered) {
        auto t = kv.second;
        std::shared_ptr<Tensor> holder;
        const void* data = tensor_bytes(t, holder);
        size_t nbytes = t->numel() * t->storage->element_size();
        out.write(static_cast<const char*>(data), nbytes);
    }
}

std::map<std::string, std::shared_ptr<Tensor>> load_safetensors(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("[litetorch Error] safetensors: cannot open file for reading: " + filepath);
    in.seekg(0, std::ios::end);
    uint64_t file_size = static_cast<uint64_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (file_size < 8) throw std::runtime_error("[litetorch Error] safetensors: file too small");
    unsigned char hlen_le[8];
    in.read(reinterpret_cast<char*>(hlen_le), 8);
    if (!in) throw std::runtime_error("[litetorch Error] safetensors: failed to read header size");
    uint64_t hlen = 0;
    for (int i = 0; i < 8; i++) hlen |= (static_cast<uint64_t>(hlen_le[i]) << (i * 8));
    if (hlen > file_size - 8) throw std::runtime_error("[litetorch Error] safetensors: header size exceeds file size");
    std::string header(hlen, '\0');
    in.read(&header[0], hlen);
    if (!in) throw std::runtime_error("[litetorch Error] safetensors: failed to read header");

    JsonParser parser(header.data(), header.size());
    auto entries = parser.parse_header();

    uint64_t data_base = 8 + hlen;
    if (file_size < data_base) throw std::runtime_error("[litetorch Error] safetensors: file truncated");

    std::map<std::string, std::shared_ptr<Tensor>> out;
    for (const auto& kv : entries) {
        const ParsedTensor& pt = kv.second;
        DataType dtype = safetensors_to_dtype(pt.dtype_str);
        auto t = Tensor::create(pt.shape, Device(DeviceType::CPU, 0), false, dtype);
        size_t elem_sz = t->storage->element_size();
        size_t expect_bytes = t->numel() * elem_sz;
        size_t actual_bytes = static_cast<size_t>(pt.data_end - pt.data_start);
        if (actual_bytes != expect_bytes) throw std::runtime_error("[litetorch Error] safetensors: data size mismatch for tensor " + kv.first);
        uint64_t abs_off = data_base + pt.data_start;
        if (abs_off + actual_bytes > file_size) throw std::runtime_error("[litetorch Error] safetensors: data offset out of bounds for tensor " + kv.first);
        in.seekg(static_cast<std::streamoff>(abs_off), std::ios::beg);
        t->storage->get_cpu_ptr();
        in.read(static_cast<char*>(static_cast<void*>(t->storage->cpu_data)), actual_bytes);
        if (!in) throw std::runtime_error("[litetorch Error] safetensors: failed to read tensor data for " + kv.first);
        out[kv.first] = t;
    }
    return out;
}

namespace nn {

static void ensure_dir(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

void Module::save_pretrained(const std::string& path, int num_shards) {
    auto params = parameters();
    if (num_shards < 1) num_shards = 1;
    if (num_shards == 1) {
        std::map<std::string, std::shared_ptr<Tensor>> named;
        for (size_t i = 0; i < params.size(); i++) {
            named["param_" + std::to_string(i)] = params[i];
        }
        save_safetensors(named, path);
        return;
    }
    ensure_dir(path);
    size_t per_shard = (params.size() + num_shards - 1) / num_shards;
    for (int s = 0; s < num_shards; s++) {
        std::map<std::string, std::shared_ptr<Tensor>> named;
        size_t begin = s * per_shard;
        size_t end = std::min(begin + per_shard, params.size());
        for (size_t i = begin; i < end; i++) {
            named["param_" + std::to_string(i)] = params[i];
        }
        char fname[512];
        snprintf(fname, sizeof(fname), "%s/model-%05d-of-%05d.safetensors", path.c_str(), s + 1, num_shards);
        save_safetensors(named, fname);
    }
    std::string index_path = path + "/model.safetensors.index.json";
    std::ofstream idx(index_path);
    uint64_t total_size = 0;
    for (auto& p : params) total_size += static_cast<uint64_t>(p->numel()) * p->storage->element_size();
    idx << "{\"metadata\":{\"total_size\":" << total_size << "},\"weight_map\":{";
    bool first = true;
    for (size_t i = 0; i < params.size(); i++) {
        int shard = static_cast<int>(i / per_shard);
        if (!first) idx << ",";
        first = false;
        char fname[128];
        snprintf(fname, sizeof(fname), "model-%05d-of-%05d.safetensors", shard + 1, num_shards);
        idx << "\"param_" << i << "\":\"" << fname << "\"";
    }
    idx << "}}";
}

void Module::load_pretrained(const std::string& path) {
    auto params = parameters();
    std::map<std::string, std::shared_ptr<Tensor>> loaded;
    struct stat st;
    bool is_dir = (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
    if (!is_dir) {
        std::ifstream f(path, std::ios::binary);
        if (f.good()) {
            loaded = load_safetensors(path);
        }
    }
    if (loaded.empty()) {
        std::string index_path = path + "/model.safetensors.index.json";
        std::ifstream idxf(index_path);
        if (idxf.good()) {
            std::string content((std::istreambuf_iterator<char>(idxf)), std::istreambuf_iterator<char>());
            std::set<std::string> shard_files;
            std::regex shard_re(R"(model-\d{5}-of-\d{5}\.safetensors)");
            for (std::sregex_iterator it(content.begin(), content.end(), shard_re), e; it != e; ++it) {
                shard_files.insert(it->str());
            }
            for (auto& fname : shard_files) {
                auto part = load_safetensors(path + "/" + fname);
                for (auto& kv : part) loaded[kv.first] = kv.second;
            }
        } else {
            throw std::runtime_error("[litetorch Error] load_pretrained: no file or index found at " + path);
        }
    }
    for (size_t i = 0; i < params.size(); i++) {
        std::string key = "param_" + std::to_string(i);
        auto it = loaded.find(key);
        if (it == loaded.end()) throw std::runtime_error("[litetorch Error] load_pretrained: missing tensor " + key);
        auto src = it->second;
        if (src->shape != params[i]->shape) throw std::runtime_error("[litetorch Error] load_pretrained: shape mismatch for " + key);
        params[i]->copy_(src);
    }
}

}
}
