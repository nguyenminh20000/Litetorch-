#include "litetorch/quant_scheme.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>

namespace litetorch {
namespace quantization {

namespace {

Device cpu_device() { return Device(DeviceType::CPU); }

std::shared_ptr<Tensor> cpu_f32(std::shared_ptr<Tensor> t) {
    if (!t) throw std::runtime_error("quant_scheme: null tensor");
    return t->to(cpu_device(), DataType::FP32);
}

float* f32_ptr(std::shared_ptr<Tensor> t) {
    return (float*)t->storage->get_cpu_ptr();
}

const float* f32_cptr(std::shared_ptr<Tensor> t) {
    return (const float*)t->storage->get_cpu_ptr();
}

std::string dtype_name(DataType dt) {
    switch (dt) {
        case DataType::FP32: return "FP32";
        case DataType::FP64: return "FP64";
        case DataType::FP16: return "FP16";
        case DataType::BF16: return "BF16";
        case DataType::INT16: return "INT16";
        case DataType::INT8: return "INT8";
        case DataType::INT4: return "INT4";
        case DataType::FP8_E4M3: return "FP8_E4M3";
        case DataType::FP8_E5M2: return "FP8_E5M2";
        case DataType::NF4: return "NF4";
        case DataType::FP4_E2M1: return "FP4_E2M1";
        case DataType::INT32: return "INT32";
        case DataType::INT64: return "INT64";
        case DataType::UINT8: return "UINT8";
        case DataType::BOOL: return "BOOL";
        default: throw std::runtime_error("quant_scheme: unknown DataType value");
    }
}

DataType dtype_from_name(const std::string& s) {
    if (s == "FP32") return DataType::FP32;
    if (s == "FP64") return DataType::FP64;
    if (s == "FP16") return DataType::FP16;
    if (s == "BF16") return DataType::BF16;
    if (s == "INT16") return DataType::INT16;
    if (s == "INT8") return DataType::INT8;
    if (s == "INT4") return DataType::INT4;
    if (s == "FP8_E4M3") return DataType::FP8_E4M3;
    if (s == "FP8_E5M2") return DataType::FP8_E5M2;
    if (s == "NF4") return DataType::NF4;
    if (s == "FP4_E2M1") return DataType::FP4_E2M1;
    if (s == "INT32") return DataType::INT32;
    if (s == "INT64") return DataType::INT64;
    if (s == "UINT8") return DataType::UINT8;
    if (s == "BOOL") return DataType::BOOL;
    throw std::runtime_error("quant_scheme: unknown dtype name '" + s + "' in metadata");
}

int param_int(const std::map<std::string, std::string>& params, const std::string& key, int dflt) {
    auto it = params.find(key);
    if (it == params.end()) return dflt;
    try {
        size_t used = 0;
        int v = std::stoi(it->second, &used);
        if (used != it->second.size()) throw std::invalid_argument("trailing");
        return v;
    } catch (...) {
        throw std::runtime_error("quant_scheme: scheme param '" + key + "' value '" + it->second + "' is not an integer");
    }
}

void mkdir_p(const std::string& path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur.push_back(path[i]);
        if (path[i] == '/' && cur.size() > 1) {
            std::string dir = cur.substr(0, cur.size() - 1);
            if (!dir.empty() && mkdir(dir.c_str(), 0777) != 0 && errno != EEXIST)
                throw std::runtime_error("quant_scheme: cannot create directory '" + dir + "'");
        }
    }
    if (!path.empty() && mkdir(path.c_str(), 0777) != 0 && errno != EEXIST)
        throw std::runtime_error("quant_scheme: cannot create directory '" + path + "'");
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else out.push_back(c);
        }
    }
    return out;
}

struct Json {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;
    const Json* find(const std::string& key) const {
        if (type != Object) return nullptr;
        for (auto& kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

struct JsonParser {
    const char* p;
    explicit JsonParser(const std::string& s) : p(s.c_str()) {}
    void ws() { while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p; }
    Json parse() {
        ws();
        Json v = value();
        ws();
        if (*p != '\0') throw std::runtime_error("quant_scheme: trailing characters in metadata JSON");
        return v;
    }
    Json value() {
        ws();
        if (*p == '{') {
            ++p;
            Json v; v.type = Json::Object;
            ws();
            if (*p == '}') { ++p; return v; }
            while (true) {
                ws();
                if (*p != '"') throw std::runtime_error("quant_scheme: expected string key in metadata JSON");
                std::string k = str();
                ws();
                if (*p != ':') throw std::runtime_error("quant_scheme: expected ':' in metadata JSON");
                ++p;
                Json val = value();
                v.obj.push_back({k, val});
                ws();
                if (*p == ',') { ++p; continue; }
                if (*p == '}') { ++p; return v; }
                throw std::runtime_error("quant_scheme: expected ',' or '}' in metadata JSON");
            }
        }
        if (*p == '[') {
            ++p;
            Json v; v.type = Json::Array;
            ws();
            if (*p == ']') { ++p; return v; }
            while (true) {
                v.arr.push_back(value());
                ws();
                if (*p == ',') { ++p; continue; }
                if (*p == ']') { ++p; return v; }
                throw std::runtime_error("quant_scheme: expected ',' or ']' in metadata JSON");
            }
        }
        if (*p == '"') { Json v; v.type = Json::String; v.str = str(); return v; }
        if (strncmp(p, "true", 4) == 0) { p += 4; Json v; v.type = Json::Bool; v.b = true; return v; }
        if (strncmp(p, "false", 5) == 0) { p += 5; Json v; v.type = Json::Bool; v.b = false; return v; }
        if (strncmp(p, "null", 4) == 0) { p += 4; return Json(); }
        char* end = nullptr;
        double d = strtod(p, &end);
        if (end == p) throw std::runtime_error("quant_scheme: invalid value in metadata JSON");
        p = end;
        Json v; v.type = Json::Number; v.num = d; return v;
    }
    std::string str() {
        ++p;
        std::string out;
        while (*p != '\0' && *p != '"') {
            if (*p == '\\') {
                ++p;
                switch (*p) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        unsigned code = 0;
                        for (int i = 0; i < 4; ++i) {
                            ++p;
                            char c = *p;
                            code <<= 4;
                            if (c >= '0' && c <= '9') code += (unsigned)(c - '0');
                            else if (c >= 'a' && c <= 'f') code += (unsigned)(c - 'a' + 10);
                            else if (c >= 'A' && c <= 'F') code += (unsigned)(c - 'A' + 10);
                            else throw std::runtime_error("quant_scheme: bad unicode escape in metadata JSON");
                        }
                        if (code < 0x80) out.push_back((char)code);
                        else if (code < 0x800) {
                            out.push_back((char)(0xC0 | (code >> 6)));
                            out.push_back((char)(0x80 | (code & 0x3F)));
                        } else {
                            out.push_back((char)(0xE0 | (code >> 12)));
                            out.push_back((char)(0x80 | ((code >> 6) & 0x3F)));
                            out.push_back((char)(0x80 | (code & 0x3F)));
                        }
                        break;
                    }
                    default: throw std::runtime_error("quant_scheme: bad escape in metadata JSON");
                }
                ++p;
            } else {
                out.push_back(*p);
                ++p;
            }
        }
        if (*p != '"') throw std::runtime_error("quant_scheme: unterminated string in metadata JSON");
        ++p;
        return out;
    }
};

std::string read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("quant_scheme: cannot open '" + path + "'");
    std::string out;
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    fclose(f);
    return out;
}

void write_file(const std::string& path, const std::string& content) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("quant_scheme: cannot write '" + path + "'");
    if (!content.empty()) fwrite(content.data(), 1, content.size(), f);
    fclose(f);
}

}
namespace {

std::map<std::string, QuantScheme>& registry() {
    static std::map<std::string, QuantScheme> m;
    return m;
}

QParams int8_per_channel_calibrate(std::shared_ptr<Tensor> t, const Calibrator*, const std::string&, const std::map<std::string, std::string>&) {
    auto W = cpu_f32(t);
    if (W->shape.empty() || W->shape[0] == 0) throw std::runtime_error("quant_scheme: int8-per-channel calibrate needs a non-empty tensor");
    int64_t rows = W->shape[0];
    int64_t cols = (int64_t)W->numel() / rows;
    const float* d = f32_cptr(W);
    QParams qp;
    qp.bits = 8;
    qp.block_size = 0;
    qp.scales = Tensor::create({rows}, cpu_device(), false, DataType::FP32);
    float* s = f32_ptr(qp.scales);
    for (int64_t r = 0; r < rows; ++r) {
        float m = 0.0f;
        for (int64_t c = 0; c < cols; ++c) m = std::max(m, std::fabs(d[r * cols + c]));
        s[r] = (m == 0.0f) ? 1.0f : m / 127.0f;
    }
    return qp;
}

QuantizedTensor int8_per_channel_quantize(std::shared_ptr<Tensor> t, const QParams& qp, const std::map<std::string, std::string>&) {
    if (!qp.scales) throw std::runtime_error("quant_scheme: int8-per-channel quantize needs per-row scales");
    auto W = cpu_f32(t);
    if (W->shape.empty() || W->shape[0] == 0) throw std::runtime_error("quant_scheme: int8-per-channel quantize needs a non-empty tensor");
    int64_t rows = W->shape[0];
    int64_t cols = (int64_t)W->numel() / rows;
    const float* d = f32_cptr(W);
    const float* s = f32_cptr(qp.scales);
    QuantizedTensor rec;
    rec.data = Tensor::create(W->shape, cpu_device(), false, DataType::INT8);
    int8_t* q = (int8_t*)rec.data->storage->get_cpu_ptr();
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            float v = std::round(d[r * cols + c] / s[r]);
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            q[r * cols + c] = (int8_t)v;
        }
    }
    rec.qparams = qp;
    rec.logical_shape = t->shape;
    rec.logical_dtype = t->dtype;
    rec.packed = false;
    return rec;
}

std::shared_ptr<Tensor> int8_per_channel_dequantize(const QuantizedTensor& rec) {
    if (!rec.qparams.scales) throw std::runtime_error("quant_scheme: int8-per-channel dequantize needs per-row scales");
    if (!rec.data) throw std::runtime_error("quant_scheme: int8-per-channel dequantize needs record data");
    if (rec.logical_shape.empty() || rec.logical_shape[0] == 0) throw std::runtime_error("quant_scheme: int8-per-channel dequantize needs a non-empty logical shape");
    auto W = rec.data->cast(DataType::FP32);
    int64_t rows = rec.logical_shape[0];
    int64_t cols = (int64_t)W->numel() / rows;
    float* d = f32_ptr(W);
    const float* s = f32_cptr(rec.qparams.scales);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c) d[r * cols + c] *= s[r];
    return W;
}

QParams fake_quant_calibrate(std::shared_ptr<Tensor> t, const Calibrator* cal, const std::string& name, const std::map<std::string, std::string>& params) {
    int bits = param_int(params, "bits", 8);
    if (bits < 2 || bits > 32) throw std::runtime_error("quant_scheme: fake-quant bits must be in [2, 32]");
    float mn = 0.0f, mx = 0.0f;
    bool have = false;
    if (cal && !name.empty()) {
        auto mi = cal->min_vals.find(name);
        auto ma = cal->max_vals.find(name);
        if (mi != cal->min_vals.end() && ma != cal->max_vals.end()) {
            mn = mi->second;
            mx = ma->second;
            have = true;
        }
    }
    if (!have) {
        auto W = cpu_f32(t);
        if (W->numel() == 0) throw std::runtime_error("quant_scheme: fake-quant calibrate needs a non-empty tensor");
        const float* d = f32_cptr(W);
        mn = d[0];
        mx = d[0];
        for (size_t i = 1; i < W->numel(); ++i) {
            mn = std::min(mn, d[i]);
            mx = std::max(mx, d[i]);
        }
    }
    float bound = (float)((1LL << (bits - 1)) - 1);
    float qmin = -bound, qmax = bound;
    float scale = (mx - mn) / (qmax - qmin);
    if (scale == 0.0f) scale = 1.0f;
    float zp = std::round(qmin - mn / scale);
    if (zp < qmin) zp = qmin;
    if (zp > qmax) zp = qmax;
    QParams qp;
    qp.scale = scale;
    qp.zero_point = zp;
    qp.bits = bits;
    return qp;
}

QuantizedTensor fake_quant_quantize(std::shared_ptr<Tensor> t, const QParams& qp, const std::map<std::string, std::string>&) {
    QuantizedTensor rec;
    rec.data = Ops::fake_quantize(t, qp.scale, qp.zero_point, qp.bits);
    rec.qparams = qp;
    rec.logical_shape = t->shape;
    rec.logical_dtype = t->dtype;
    rec.packed = false;
    return rec;
}

std::shared_ptr<Tensor> fake_quant_dequantize(const QuantizedTensor& rec) {
    return rec.data;
}

QParams nf4_block_calibrate(std::shared_ptr<Tensor> t, const Calibrator*, const std::string&, const std::map<std::string, std::string>& params) {
    int bs = param_int(params, "block_size", 64);
    if (bs <= 0) throw std::runtime_error("quant_scheme: nf4-block64 block_size must be positive");
    auto W = cpu_f32(t);
    int64_t n = (int64_t)W->numel();
    int64_t nblocks = (n + bs - 1) / bs;
    const float* d = f32_cptr(W);
    QParams qp;
    qp.bits = 4;
    qp.block_size = bs;
    qp.scales = Tensor::create({nblocks}, cpu_device(), false, DataType::FP32);
    float* s = f32_ptr(qp.scales);
    for (int64_t b = 0; b < nblocks; ++b) {
        int64_t lo = b * bs, hi = std::min(n, lo + bs);
        float m = 0.0f;
        for (int64_t i = lo; i < hi; ++i) m = std::max(m, std::fabs(d[i]));
        s[b] = (m == 0.0f) ? 1.0f : m;
    }
    return qp;
}

QuantizedTensor nf4_block_quantize(std::shared_ptr<Tensor> t, const QParams& qp, const std::map<std::string, std::string>&) {
    if (!qp.scales || qp.block_size <= 0) throw std::runtime_error("quant_scheme: nf4-block64 quantize needs per-block scales and block_size");
    auto W = cpu_f32(t);
    int64_t n = (int64_t)W->numel();
    const float* d = f32_cptr(W);
    const float* s = f32_cptr(qp.scales);
    auto norm = Tensor::create(W->shape, cpu_device(), false, DataType::FP32);
    float* nd = f32_ptr(norm);
    for (int64_t i = 0; i < n; ++i) nd[i] = d[i] / s[i / qp.block_size];
    QuantizedTensor rec;
    rec.data = norm->cast(DataType::NF4);
    rec.qparams = qp;
    rec.logical_shape = t->shape;
    rec.logical_dtype = t->dtype;
    rec.packed = false;
    return rec;
}

std::shared_ptr<Tensor> nf4_block_dequantize(const QuantizedTensor& rec) {
    if (!rec.qparams.scales || rec.qparams.block_size <= 0) throw std::runtime_error("quant_scheme: nf4-block64 dequantize needs per-block scales and block_size");
    auto W = rec.data->cast(DataType::FP32);
    int64_t n = (int64_t)W->numel();
    float* d = f32_ptr(W);
    const float* s = f32_cptr(rec.qparams.scales);
    for (int64_t i = 0; i < n; ++i) d[i] *= s[i / rec.qparams.block_size];
    return W;
}

void ensure_builtins() {
    static bool done = false;
    if (done) return;
    done = true;
    auto& m = registry();
    if (m.find("int8-per-channel") == m.end()) {
        QuantScheme s;
        s.name = "int8-per-channel";
        s.calibrate = int8_per_channel_calibrate;
        s.quantize = int8_per_channel_quantize;
        s.dequantize = int8_per_channel_dequantize;
        m[s.name] = s;
    }
    if (m.find("dynamic-int8") == m.end()) {
        QuantScheme s;
        s.name = "dynamic-int8";
        s.params["dynamic_activation"] = "1";
        s.calibrate = int8_per_channel_calibrate;
        s.quantize = int8_per_channel_quantize;
        s.dequantize = int8_per_channel_dequantize;
        m[s.name] = s;
    }
    if (m.find("fake-quant") == m.end()) {
        QuantScheme s;
        s.name = "fake-quant";
        s.params["bits"] = "8";
        s.calibrate = fake_quant_calibrate;
        s.quantize = fake_quant_quantize;
        s.dequantize = fake_quant_dequantize;
        m[s.name] = s;
    }
    if (m.find("nf4-block64") == m.end()) {
        QuantScheme s;
        s.name = "nf4-block64";
        s.params["block_size"] = "64";
        s.calibrate = nf4_block_calibrate;
        s.quantize = nf4_block_quantize;
        s.dequantize = nf4_block_dequantize;
        m[s.name] = s;
    }
}

}

void register_scheme(const QuantScheme& scheme) {
    if (scheme.name.empty()) throw std::runtime_error("quant_scheme: scheme name must not be empty");
    if (!scheme.calibrate || !scheme.quantize || !scheme.dequantize)
        throw std::runtime_error("quant_scheme: scheme '" + scheme.name + "' must provide calibrate, quantize and dequantize");
    registry()[scheme.name] = scheme;
}

void unregister_scheme(const std::string& name) {
    ensure_builtins();
    registry().erase(name);
}

bool has_scheme(const std::string& name) {
    ensure_builtins();
    return registry().find(name) != registry().end();
}

QuantScheme get_scheme(const std::string& name) {
    ensure_builtins();
    auto it = registry().find(name);
    if (it == registry().end())
        throw std::runtime_error("quant_scheme: scheme '" + name + "' is not registered; register it with register_scheme before use");
    return it->second;
}

std::vector<std::string> list_schemes() {
    ensure_builtins();
    std::vector<std::string> out;
    for (auto& kv : registry()) out.push_back(kv.first);
    return out;
}

void register_builtin_schemes() {
    ensure_builtins();
}

std::shared_ptr<Tensor> scheme_fake_quant(const QuantScheme& scheme, std::shared_ptr<Tensor> x, const QParams& qp, const std::map<std::string, std::string>& params) {
    if (scheme.fake_quantize) return scheme.fake_quantize(x, qp, params);
    if (!scheme.quantize || !scheme.dequantize)
        throw std::runtime_error("quant_scheme: scheme '" + scheme.name + "' has no fake_quantize override and no quantize/dequantize pair");
    QuantizedTensor rec = scheme.quantize(x, qp, params);
    return scheme.dequantize(rec);
}

}
namespace quantization {

SchemeQuantizedLinear::SchemeQuantizedLinear(const QuantizedTensor& rec, std::shared_ptr<Tensor> b)
    : record(rec), bias(b) {
    auto it = record.scheme_params.find("dynamic_activation");
    dynamic_activation = (it != record.scheme_params.end() && it->second == "1");
    if (record.data) register_buffer("record_data", record.data);
    if (record.qparams.scales) register_buffer("record_scales", record.qparams.scales);
    if (record.qparams.zero_points) register_buffer("record_zero_points", record.qparams.zero_points);
    if (bias) register_parameter("bias", bias);
}

std::shared_ptr<Tensor> SchemeQuantizedLinear::forward(std::shared_ptr<Tensor> input) {
    if (!record.data) throw std::runtime_error("quant_scheme: SchemeQuantizedLinear has no record data");
    QuantScheme scheme = get_scheme(record.scheme_name);
    auto xc = input->to(cpu_device(), DataType::FP32);
    auto xq = xc;
    if (dynamic_activation) {
        const float* d = f32_cptr(xc);
        float m = 0.0f;
        for (size_t i = 0; i < xc->numel(); ++i) m = std::max(m, std::fabs(d[i]));
        float s = (m == 0.0f) ? 1.0f : m / 127.0f;
        xq = Tensor::create(xc->shape, cpu_device(), false, DataType::FP32);
        float* qd = f32_ptr(xq);
        for (size_t i = 0; i < xc->numel(); ++i) {
            float v = std::round(d[i] / s);
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            qd[i] = v * s;
        }
    }
    auto W = [&]() {
        QuantizedTensor cpu_rec = record;
        if (cpu_rec.data) cpu_rec.data = cpu_rec.data->to(cpu_device());
        if (cpu_rec.qparams.scales) cpu_rec.qparams.scales = cpu_rec.qparams.scales->to(cpu_device());
        if (cpu_rec.qparams.zero_points) cpu_rec.qparams.zero_points = cpu_rec.qparams.zero_points->to(cpu_device());
        return scheme.dequantize(cpu_rec);
    }();
    auto out = Ops::matmul(xq, W->transpose(0, 1));
    if (bias) {
        auto bc = bias->to(cpu_device(), DataType::FP32);
        out = Ops::add(out, bc);
    }
    if (input->device.type != DeviceType::CPU) out = out->to(input->device);
    if (input->dtype != DataType::FP32) out = out->to(input->dtype);
    return out;
}

std::vector<std::shared_ptr<Tensor>> SchemeQuantizedLinear::parameters() {
    if (bias) return {bias};
    return {};
}

void SchemeQuantizedLinear::to(const Device& device) {
    if (record.data) record.data = record.data->to(device);
    if (record.qparams.scales) record.qparams.scales = record.qparams.scales->to(device);
    if (record.qparams.zero_points) record.qparams.zero_points = record.qparams.zero_points->to(device);
    if (bias) bias = bias->to(device);
    if (record.data) register_buffer("record_data", record.data);
    if (record.qparams.scales) register_buffer("record_scales", record.qparams.scales);
    if (record.qparams.zero_points) register_buffer("record_zero_points", record.qparams.zero_points);
    if (bias) register_parameter("bias", bias);
}

SchemeQATLinear::SchemeQATLinear(std::shared_ptr<nn::Linear> linear, const std::string& scheme_name_, const std::map<std::string, std::string>& params)
    : weight(linear->weight), bias(linear->bias), scheme_name(scheme_name_) {
    QuantScheme scheme = get_scheme(scheme_name);
    scheme_params = scheme.params;
    for (auto& kv : params) scheme_params[kv.first] = kv.second;
    weight_qparams = scheme.calibrate(weight, nullptr, "", scheme_params);
    if (weight) register_parameter("weight", weight);
    if (bias) register_parameter("bias", bias);
}

std::shared_ptr<Tensor> SchemeQATLinear::forward(std::shared_ptr<Tensor> input) {
    QuantScheme scheme = get_scheme(scheme_name);
    auto xc = input->to(cpu_device(), DataType::FP32);
    if (qat_enabled && training) {
        auto wc = weight->to(cpu_device(), DataType::FP32);
        auto qw = scheme_fake_quant(scheme, wc, weight_qparams, scheme_params);
        if (!act_calibrated) {
            act_qparams = scheme.calibrate(xc, nullptr, "", scheme_params);
            act_calibrated = true;
        }
        auto qi = scheme_fake_quant(scheme, xc, act_qparams, scheme_params);
        auto out = Ops::matmul(qi, qw->transpose(0, 1));
        if (bias) out = Ops::add(out, bias->to(cpu_device(), DataType::FP32));
        if (input->device.type != DeviceType::CPU) out = out->to(input->device);
        if (input->dtype != DataType::FP32) out = out->to(input->dtype);
        return out;
    }
    auto out = Ops::matmul(xc, weight->to(cpu_device(), DataType::FP32)->transpose(0, 1));
    if (bias) out = Ops::add(out, bias->to(cpu_device(), DataType::FP32));
    if (input->device.type != DeviceType::CPU) out = out->to(input->device);
    if (input->dtype != DataType::FP32) out = out->to(input->dtype);
    return out;
}

std::vector<std::shared_ptr<Tensor>> SchemeQATLinear::parameters() {
    std::vector<std::shared_ptr<Tensor>> out;
    if (weight) out.push_back(weight);
    if (bias) out.push_back(bias);
    return out;
}

void SchemeQATLinear::to(const Device& device) {
    if (weight) weight->move_to(device);
    if (bias) bias->move_to(device);
    if (weight) register_parameter("weight", weight);
    if (bias) register_parameter("bias", bias);
}

namespace {

std::string join_dotted(const std::string& prefix, const std::string& name) {
    return prefix.empty() ? name : prefix + "." + name;
}

void fake_quant_cell_weight(std::shared_ptr<Tensor> w, const QuantScheme& scheme) {
    if (!w) return;
    if (w->device.type != DeviceType::CPU || w->dtype != DataType::FP32)
        throw std::runtime_error("quant_scheme: in-place cell fake-quant needs CPU FP32 cell weights");
    if (!w->is_contiguous())
        throw std::runtime_error("quant_scheme: in-place cell fake-quant needs contiguous cell weights");
    auto W = w->to(cpu_device(), DataType::FP32);
    QParams qp = scheme.calibrate(W, nullptr, "", scheme.params);
    auto fq = scheme_fake_quant(scheme, W, qp, scheme.params);
    auto fqc = fq->to(cpu_device(), DataType::FP32);
    float* dst = f32_ptr(W);
    const float* src = f32_cptr(fqc);
    for (size_t i = 0; i < W->numel(); ++i) dst[i] = src[i];
}

void quantize_children(std::shared_ptr<nn::Module> parent, const QuantScheme& scheme, const Calibrator* cal, const std::string& prefix) {
    for (auto& kv : parent->named_children()) {
        const std::string& name = kv.first;
        auto child = kv.second;
        std::string dotted = join_dotted(prefix, name);
        if (auto lin = std::dynamic_pointer_cast<nn::Linear>(child)) {
            QParams qp = scheme.calibrate(lin->weight, cal, dotted + ".weight", scheme.params);
            QuantizedTensor rec = scheme.quantize(lin->weight, qp, scheme.params);
            rec.scheme_name = scheme.name;
            rec.scheme_params = scheme.params;
            auto qlin = std::make_shared<SchemeQuantizedLinear>(rec, lin->bias);
            parent->add_module(name, qlin);
        } else if (auto lstm = std::dynamic_pointer_cast<nn::LSTM>(child)) {
            for (auto& cell : lstm->cells) {
                fake_quant_cell_weight(cell->weight_ih, scheme);
                fake_quant_cell_weight(cell->weight_hh, scheme);
            }
        } else if (auto gru = std::dynamic_pointer_cast<nn::GRU>(child)) {
            for (auto& cell : gru->cells) {
                fake_quant_cell_weight(cell->weight_ih, scheme);
                fake_quant_cell_weight(cell->weight_hh, scheme);
            }
        } else {
            quantize_children(child, scheme, cal, dotted);
        }
    }
}

QuantScheme effective_scheme(const std::string& name, const std::map<std::string, std::string>& params) {
    QuantScheme scheme = get_scheme(name);
    for (auto& kv : params) scheme.params[kv.first] = kv.second;
    return scheme;
}

}

std::shared_ptr<nn::Module> quantize_model(std::shared_ptr<nn::Module> model, const std::string& scheme_name, const std::map<std::string, std::string>& params) {
    QuantScheme scheme = effective_scheme(scheme_name, params);
    if (auto lin = std::dynamic_pointer_cast<nn::Linear>(model)) {
        QParams qp = scheme.calibrate(lin->weight, nullptr, "", scheme.params);
        QuantizedTensor rec = scheme.quantize(lin->weight, qp, scheme.params);
        rec.scheme_name = scheme.name;
        rec.scheme_params = scheme.params;
        return std::make_shared<SchemeQuantizedLinear>(rec, lin->bias);
    }
    if (auto lstm = std::dynamic_pointer_cast<nn::LSTM>(model)) {
        for (auto& cell : lstm->cells) {
            fake_quant_cell_weight(cell->weight_ih, scheme);
            fake_quant_cell_weight(cell->weight_hh, scheme);
        }
        return model;
    }
    if (auto gru = std::dynamic_pointer_cast<nn::GRU>(model)) {
        for (auto& cell : gru->cells) {
            fake_quant_cell_weight(cell->weight_ih, scheme);
            fake_quant_cell_weight(cell->weight_hh, scheme);
        }
        return model;
    }
    quantize_children(model, scheme, nullptr, "");
    return model;
}

std::shared_ptr<nn::Module> quantize_ptq(std::shared_ptr<PreparedModel> prepared, const std::string& scheme_name, const std::map<std::string, std::string>& params) {
    if (!prepared) throw std::runtime_error("quant_scheme: quantize_ptq needs a PreparedModel");
    if (prepared->calibrator.min_vals.empty() && prepared->calibrator.max_vals.empty())
        throw std::runtime_error("quant_scheme: quantize_ptq requires calibration data; run PreparedModel::calibrate first");
    QuantScheme scheme = effective_scheme(scheme_name, params);
    quantize_children(prepared->model, scheme, &prepared->calibrator, "");
    return prepared->model;
}

}
namespace quantization {

namespace {

struct SaveEntry {
    std::string name;
    std::string kind;
    int cell_index = -1;
    std::string field;
    std::string scheme;
    std::map<std::string, std::string> params;
    std::vector<int64_t> logical_shape;
    std::string logical_dtype;
    std::string stored_dtype;
    std::vector<int64_t> data_shape;
    bool packed = false;
    std::string nibble_order;
    float scale = 1.0f;
    float zero_point = 0.0f;
    int bits = 8;
    int block_size = 0;
    std::string data_key;
    std::string scales_key;
    std::string zero_points_key;
    std::string bias_key;
    std::string bias_ih_key;
    std::string bias_hh_key;
};

bool safetensors_save_supported(DataType dt) {
    switch (dt) {
        case DataType::FP32: case DataType::FP16: case DataType::BF16: case DataType::FP64:
        case DataType::INT32: case DataType::INT16: case DataType::INT8:
            return true;
        default:
            return false;
    }
}

std::shared_ptr<Tensor> bridge_for_save(std::shared_ptr<Tensor> t) {
    if (!t) throw std::runtime_error("quant_scheme: save_quantized found a null record data tensor");
    if (safetensors_save_supported(t->dtype)) return t;
    auto cpu = t->to(cpu_device());
    if (!cpu->is_contiguous()) throw std::runtime_error("quant_scheme: save_quantized needs contiguous record data for dtype bridging");
    size_t nbytes = cpu->numel() * cpu->storage->element_size();
    auto bridge = Tensor::create({(int64_t)nbytes}, cpu_device(), false, DataType::INT8);
    std::memcpy(bridge->storage->get_cpu_ptr(), cpu->storage->get_cpu_ptr(), nbytes);
    return bridge;
}

void collect_entries(std::shared_ptr<nn::Module> module, const std::string& path,
                     std::map<std::string, std::shared_ptr<Tensor>>& tensors,
                     std::vector<SaveEntry>& entries) {
    std::string name = path.empty() ? "$root" : path;
    if (auto qlin = std::dynamic_pointer_cast<SchemeQuantizedLinear>(module)) {
        if (!qlin->record.data) throw std::runtime_error("quant_scheme: save_quantized found a SchemeQuantizedLinear with no record data at '" + name + "'");
        SaveEntry e;
        e.name = name;
        e.kind = "linear";
        e.scheme = qlin->record.scheme_name;
        e.params = qlin->record.scheme_params;
        e.logical_shape = qlin->record.logical_shape;
        e.logical_dtype = dtype_name(qlin->record.logical_dtype);
        e.stored_dtype = dtype_name(qlin->record.data->dtype);
        e.data_shape = qlin->record.data->shape;
        e.packed = qlin->record.packed;
        e.nibble_order = qlin->record.nibble_order;
        e.scale = qlin->record.qparams.scale;
        e.zero_point = qlin->record.qparams.zero_point;
        e.bits = qlin->record.qparams.bits;
        e.block_size = qlin->record.qparams.block_size;
        e.data_key = name + ".record_data";
        tensors[e.data_key] = bridge_for_save(qlin->record.data);
        if (qlin->record.qparams.scales) {
            e.scales_key = name + ".record_scales";
            tensors[e.scales_key] = qlin->record.qparams.scales;
        }
        if (qlin->record.qparams.zero_points) {
            e.zero_points_key = name + ".record_zero_points";
            tensors[e.zero_points_key] = qlin->record.qparams.zero_points;
        }
        if (qlin->bias) {
            e.bias_key = name + ".bias";
            tensors[e.bias_key] = qlin->bias;
        }
        entries.push_back(e);
        return;
    }
    if (auto lstm = std::dynamic_pointer_cast<nn::LSTM>(module)) {
        for (size_t i = 0; i < lstm->cells.size(); ++i) {
            for (int f = 0; f < 2; ++f) {
                std::string field = (f == 0) ? "weight_ih" : "weight_hh";
                auto w = (f == 0) ? lstm->cells[i]->weight_ih : lstm->cells[i]->weight_hh;
                if (!w) continue;
                SaveEntry e;
                e.name = name;
                e.kind = "cell";
                e.cell_index = (int)i;
                e.field = field;
                e.logical_shape = w->shape;
                e.logical_dtype = dtype_name(w->dtype);
                e.stored_dtype = dtype_name(w->dtype);
                e.data_key = name + ".cell_" + std::to_string(i) + "." + field;
                tensors[e.data_key] = w;
                auto bih = lstm->cells[i]->bias_ih;
                auto bhh = lstm->cells[i]->bias_hh;
                if (bih) {
                    e.bias_ih_key = name + ".cell_" + std::to_string(i) + ".bias_ih";
                    tensors[e.bias_ih_key] = bih;
                }
                if (bhh) {
                    e.bias_hh_key = name + ".cell_" + std::to_string(i) + ".bias_hh";
                    tensors[e.bias_hh_key] = bhh;
                }
                entries.push_back(e);
            }
        }
    } else if (auto gru = std::dynamic_pointer_cast<nn::GRU>(module)) {
        for (size_t i = 0; i < gru->cells.size(); ++i) {
            for (int f = 0; f < 2; ++f) {
                std::string field = (f == 0) ? "weight_ih" : "weight_hh";
                auto w = (f == 0) ? gru->cells[i]->weight_ih : gru->cells[i]->weight_hh;
                if (!w) continue;
                SaveEntry e;
                e.name = name;
                e.kind = "cell";
                e.cell_index = (int)i;
                e.field = field;
                e.logical_shape = w->shape;
                e.logical_dtype = dtype_name(w->dtype);
                e.stored_dtype = dtype_name(w->dtype);
                e.data_key = name + ".cell_" + std::to_string(i) + "." + field;
                tensors[e.data_key] = w;
                auto bih = gru->cells[i]->bias_ih;
                auto bhh = gru->cells[i]->bias_hh;
                if (bih) {
                    e.bias_ih_key = name + ".cell_" + std::to_string(i) + ".bias_ih";
                    tensors[e.bias_ih_key] = bih;
                }
                if (bhh) {
                    e.bias_hh_key = name + ".cell_" + std::to_string(i) + ".bias_hh";
                    tensors[e.bias_hh_key] = bhh;
                }
                entries.push_back(e);
            }
        }
    }
    for (auto& kv : module->named_children()) {
        std::string child_path = path.empty() ? kv.first : path + "." + kv.first;
        collect_entries(kv.second, child_path, tensors, entries);
    }
}

std::string entry_json(const SaveEntry& e) {
    std::string s = "{";
    s += "\"name\":\"" + json_escape(e.name) + "\"";
    s += ",\"kind\":\"" + json_escape(e.kind) + "\"";
    if (e.kind == "cell") {
        s += ",\"cell_index\":" + std::to_string(e.cell_index);
        s += ",\"field\":\"" + json_escape(e.field) + "\"";
        s += ",\"bias_ih_key\":\"" + json_escape(e.bias_ih_key) + "\"";
        s += ",\"bias_hh_key\":\"" + json_escape(e.bias_hh_key) + "\"";
    } else {
        s += ",\"scheme\":\"" + json_escape(e.scheme) + "\"";
        s += ",\"params\":{";
        bool first = true;
        for (auto& kv : e.params) {
            if (!first) s += ",";
            first = false;
            s += "\"" + json_escape(kv.first) + "\":\"" + json_escape(kv.second) + "\"";
        }
        s += "}";
        s += ",\"packed\":" + std::string(e.packed ? "true" : "false");
        s += ",\"nibble_order\":\"" + json_escape(e.nibble_order) + "\"";
        char buf[64];
        snprintf(buf, sizeof(buf), "%.9g", e.scale);
        s += ",\"scale\":" + std::string(buf);
        snprintf(buf, sizeof(buf), "%.9g", e.zero_point);
        s += ",\"zero_point\":" + std::string(buf);
        s += ",\"bits\":" + std::to_string(e.bits);
        s += ",\"block_size\":" + std::to_string(e.block_size);
        s += ",\"scales_key\":\"" + json_escape(e.scales_key) + "\"";
        s += ",\"zero_points_key\":\"" + json_escape(e.zero_points_key) + "\"";
        s += ",\"bias_key\":\"" + json_escape(e.bias_key) + "\"";
    }
    s += ",\"logical_shape\":[";
    for (size_t i = 0; i < e.logical_shape.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(e.logical_shape[i]);
    }
    s += "]";
    if (e.kind == "linear") {
        s += ",\"data_shape\":[";
        for (size_t i = 0; i < e.data_shape.size(); ++i) {
            if (i) s += ",";
            s += std::to_string(e.data_shape[i]);
        }
        s += "]";
    }
    s += ",\"logical_dtype\":\"" + json_escape(e.logical_dtype) + "\"";
    s += ",\"stored_dtype\":\"" + json_escape(e.stored_dtype) + "\"";
    s += ",\"data_key\":\"" + json_escape(e.data_key) + "\"";
    s += "}";
    return s;
}

std::vector<std::string> split_dotted(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == '.') { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

std::shared_ptr<nn::Module> find_module(std::shared_ptr<nn::Module> root, const std::string& dotted) {
    if (dotted.empty() || dotted == "$root") return root;
    auto parts = split_dotted(dotted);
    auto cur = root;
    for (auto& part : parts) {
        bool found = false;
        for (auto& kv : cur->named_children()) {
            if (kv.first == part) { cur = kv.second; found = true; break; }
        }
        if (!found) return nullptr;
    }
    return cur;
}

std::string req_str(const Json& o, const std::string& key) {
    const Json* v = o.find(key);
    if (!v || v->type != Json::String) throw std::runtime_error("quant_scheme: metadata entry is missing string field '" + key + "'");
    return v->str;
}

std::string opt_str(const Json& o, const std::string& key) {
    const Json* v = o.find(key);
    if (!v || v->type != Json::String) return "";
    return v->str;
}

double req_num(const Json& o, const std::string& key, double dflt, bool required) {
    const Json* v = o.find(key);
    if (!v) {
        if (required) throw std::runtime_error("quant_scheme: metadata entry is missing numeric field '" + key + "'");
        return dflt;
    }
    if (v->type != Json::Number) throw std::runtime_error("quant_scheme: metadata field '" + key + "' is not a number");
    return v->num;
}

std::shared_ptr<Tensor> req_tensor(const std::map<std::string, std::shared_ptr<Tensor>>& tensors, const std::string& key, const std::string& what) {
    if (key.empty()) throw std::runtime_error("quant_scheme: metadata entry is missing tensor key for " + what);
    auto it = tensors.find(key);
    if (it == tensors.end()) throw std::runtime_error("quant_scheme: saved tensor '" + key + "' for " + what + " not found in model.safetensors");
    return it->second;
}

}

void save_quantized(std::shared_ptr<nn::Module> model, const std::string& dir) {
    if (!model) throw std::runtime_error("quant_scheme: save_quantized needs a model");
    mkdir_p(dir);
    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    std::vector<SaveEntry> entries;
    collect_entries(model, "", tensors, entries);
    save_safetensors(tensors, dir + "/model.safetensors");
    std::string js = "{\"format\":\"litetorch-quantized-v1\",\"modules\":[";
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i) js += ",";
        js += entry_json(entries[i]);
    }
    js += "]}";
    write_file(dir + "/quant_metadata.json", js);
}

std::shared_ptr<nn::Module> load_quantized(std::shared_ptr<nn::Module> model_template, const std::string& dir) {
    if (!model_template) throw std::runtime_error("quant_scheme: load_quantized needs a model template");
    std::string meta = read_file(dir + "/quant_metadata.json");
    JsonParser parser(meta);
    Json root = parser.parse();
    const Json* modules = root.find("modules");
    if (!modules || modules->type != Json::Array) throw std::runtime_error("quant_scheme: metadata has no 'modules' array");
    auto tensors = load_safetensors(dir + "/model.safetensors");
    std::shared_ptr<nn::Module> root_replacement = nullptr;
    for (auto& entry : modules->arr) {
        std::string name = req_str(entry, "name");
        std::string kind = req_str(entry, "kind");
        std::string data_key = req_str(entry, "data_key");
        if (kind == "linear") {
            std::string scheme_name = req_str(entry, "scheme");
            get_scheme(scheme_name);
            std::map<std::string, std::string> params;
            const Json* pj = entry.find("params");
            if (pj && pj->type == Json::Object)
                for (auto& kv : pj->obj) {
                    if (kv.second.type != Json::String) throw std::runtime_error("quant_scheme: scheme param values must be strings");
                    params[kv.first] = kv.second.str;
                }
            QuantizedTensor rec;
            auto loaded_data = req_tensor(tensors, data_key, "linear '" + name + "'");
            DataType stored_dtype = dtype_from_name(req_str(entry, "stored_dtype"));
            if (safetensors_save_supported(stored_dtype)) {
                rec.data = loaded_data;
            } else {
                if (loaded_data->dtype != DataType::INT8)
                    throw std::runtime_error("quant_scheme: bridged data for linear '" + name + "' is not INT8 in model.safetensors");
                const Json* ds = entry.find("data_shape");
                if (!ds || ds->type != Json::Array) throw std::runtime_error("quant_scheme: metadata entry '" + name + "' is missing data_shape for bridged data");
                std::vector<int64_t> dshape;
                for (auto& d : ds->arr) {
                    if (d.type != Json::Number) throw std::runtime_error("quant_scheme: data_shape entries must be numbers");
                    dshape.push_back((int64_t)d.num);
                }
                auto restored = Tensor::create(dshape, cpu_device(), false, stored_dtype);
                size_t nbytes = restored->numel() * restored->storage->element_size();
                auto cpu_loaded = loaded_data->to(cpu_device());
                if (!cpu_loaded->is_contiguous()) throw std::runtime_error("quant_scheme: bridged data for linear '" + name + "' is not contiguous");
                if (cpu_loaded->numel() != nbytes) throw std::runtime_error("quant_scheme: bridged data byte count mismatch for linear '" + name + "'");
                std::memcpy(restored->storage->get_cpu_ptr(), cpu_loaded->storage->get_cpu_ptr(), nbytes);
                rec.data = restored;
            }
            std::string scales_key = opt_str(entry, "scales_key");
            if (!scales_key.empty()) rec.qparams.scales = req_tensor(tensors, scales_key, "linear '" + name + "' scales");
            std::string zp_key = opt_str(entry, "zero_points_key");
            if (!zp_key.empty()) rec.qparams.zero_points = req_tensor(tensors, zp_key, "linear '" + name + "' zero_points");
            rec.qparams.scale = (float)req_num(entry, "scale", 1.0, false);
            rec.qparams.zero_point = (float)req_num(entry, "zero_point", 0.0, false);
            rec.qparams.bits = (int)req_num(entry, "bits", 8, false);
            rec.qparams.block_size = (int)req_num(entry, "block_size", 0, false);
            rec.scheme_name = scheme_name;
            rec.scheme_params = params;
            const Json* shape_j = entry.find("logical_shape");
            if (!shape_j || shape_j->type != Json::Array) throw std::runtime_error("quant_scheme: metadata entry '" + name + "' is missing logical_shape");
            for (auto& d : shape_j->arr) {
                if (d.type != Json::Number) throw std::runtime_error("quant_scheme: logical_shape entries must be numbers");
                rec.logical_shape.push_back((int64_t)d.num);
            }
            rec.logical_dtype = dtype_from_name(req_str(entry, "logical_dtype"));
            rec.packed = false;
            const Json* packed_j = entry.find("packed");
            if (packed_j && packed_j->type == Json::Bool) rec.packed = packed_j->b;
            rec.nibble_order = opt_str(entry, "nibble_order");
            std::shared_ptr<Tensor> bias = nullptr;
            std::string bias_key = opt_str(entry, "bias_key");
            if (!bias_key.empty()) bias = req_tensor(tensors, bias_key, "linear '" + name + "' bias");
            auto qlin = std::make_shared<SchemeQuantizedLinear>(rec, bias);
            if (name == "$root") {
                if (!std::dynamic_pointer_cast<nn::Linear>(model_template))
                    throw std::runtime_error("quant_scheme: metadata root is a quantized linear but the template root is not a Linear");
                root_replacement = qlin;
            } else {
                auto parts = split_dotted(name);
                std::string local = parts.back();
                parts.pop_back();
                std::string parent_path;
                for (size_t i = 0; i < parts.size(); ++i) {
                    if (i) parent_path += ".";
                    parent_path += parts[i];
                }
                auto parent = find_module(model_template, parent_path);
                if (!parent) throw std::runtime_error("quant_scheme: template has no module at '" + parent_path + "' for saved linear '" + name + "'");
                auto target = find_module(model_template, name);
                if (!target || !std::dynamic_pointer_cast<nn::Linear>(target))
                    throw std::runtime_error("quant_scheme: template has no Linear at '" + name + "' for saved quantized linear");
                parent->add_module(local, qlin);
            }
        } else if (kind == "cell") {
            int cell_index = (int)req_num(entry, "cell_index", -1, true);
            std::string field = req_str(entry, "field");
            auto w = req_tensor(tensors, data_key, "cell '" + name + "'");
            std::shared_ptr<nn::Module> container = (name == "$root") ? model_template : find_module(model_template, name);
            if (!container) throw std::runtime_error("quant_scheme: template has no module at '" + name + "' for saved cell weights");
            bool set = false;
            if (auto lstm = std::dynamic_pointer_cast<nn::LSTM>(container)) {
                if (cell_index < 0 || cell_index >= (int)lstm->cells.size()) throw std::runtime_error("quant_scheme: cell_index out of range for '" + name + "'");
                auto cell = lstm->cells[cell_index];
                if (field == "weight_ih") { cell->weight_ih = w; cell->register_parameter("weight_ih", w); }
                else if (field == "weight_hh") { cell->weight_hh = w; cell->register_parameter("weight_hh", w); }
                else throw std::runtime_error("quant_scheme: unknown cell field '" + field + "'");
                std::string bih_key = opt_str(entry, "bias_ih_key");
                if (!bih_key.empty()) { auto b = req_tensor(tensors, bih_key, "cell '" + name + "' bias_ih"); cell->bias_ih = b; cell->register_parameter("bias_ih", b); }
                std::string bhh_key = opt_str(entry, "bias_hh_key");
                if (!bhh_key.empty()) { auto b = req_tensor(tensors, bhh_key, "cell '" + name + "' bias_hh"); cell->bias_hh = b; cell->register_parameter("bias_hh", b); }
                set = true;
            }
            if (auto gru = std::dynamic_pointer_cast<nn::GRU>(container)) {
                if (cell_index < 0 || cell_index >= (int)gru->cells.size()) throw std::runtime_error("quant_scheme: cell_index out of range for '" + name + "'");
                auto cell = gru->cells[cell_index];
                if (field == "weight_ih") { cell->weight_ih = w; cell->register_parameter("weight_ih", w); }
                else if (field == "weight_hh") { cell->weight_hh = w; cell->register_parameter("weight_hh", w); }
                else throw std::runtime_error("quant_scheme: unknown cell field '" + field + "'");
                std::string bih_key = opt_str(entry, "bias_ih_key");
                if (!bih_key.empty()) { auto b = req_tensor(tensors, bih_key, "cell '" + name + "' bias_ih"); cell->bias_ih = b; cell->register_parameter("bias_ih", b); }
                std::string bhh_key = opt_str(entry, "bias_hh_key");
                if (!bhh_key.empty()) { auto b = req_tensor(tensors, bhh_key, "cell '" + name + "' bias_hh"); cell->bias_hh = b; cell->register_parameter("bias_hh", b); }
                set = true;
            }
            if (!set) throw std::runtime_error("quant_scheme: template module at '" + name + "' is not an LSTM or GRU");
        } else {
            throw std::runtime_error("quant_scheme: unknown metadata module kind '" + kind + "'");
        }
    }
    return root_replacement ? root_replacement : model_template;
}

}
}
