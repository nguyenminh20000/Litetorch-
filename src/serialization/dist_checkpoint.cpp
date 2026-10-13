#include "litetorch/dist_checkpoint.h"
#include "litetorch/serialization.h"
#include "litetorch/distributed.h"
#include "litetorch/ops.h"
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

namespace litetorch {
namespace dist_checkpoint {

namespace {

int current_rank() {
    auto& pg = distributed::ProcessGroup::get();
    return pg.is_initialized() ? pg.get_rank() : 0;
}

int current_world_size() {
    auto& pg = distributed::ProcessGroup::get();
    return pg.is_initialized() ? pg.get_world_size() : 1;
}

void maybe_barrier() {
    auto& pg = distributed::ProcessGroup::get();
    if (pg.is_initialized() && pg.get_world_size() > 1) pg.barrier();
}

void ensure_dir(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

std::string json_escape_str(const std::string& s) {
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

std::string shape_to_json(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i) s += ",";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

std::string ranges_to_json(const std::vector<std::pair<int64_t,int64_t>>& ranges) {
    std::string s = "[";
    for (size_t i = 0; i < ranges.size(); i++) {
        if (i) s += ",";
        s += "[" + std::to_string(ranges[i].first) + "," + std::to_string(ranges[i].second) + "]";
    }
    s += "]";
    return s;
}

struct JsonReader {
    const char* p;
    const char* end;
    JsonReader(const char* d, size_t n) : p(d), end(d + n) {}
    void ws() { while (p < end && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) p++; }
    void expect(char c) {
        ws();
        if (p >= end || *p != c) throw std::runtime_error("[litetorch Error] dist_checkpoint: malformed metadata JSON");
        p++;
    }
    std::string str() {
        ws(); expect('"');
        std::string o;
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                p++;
                if (*p == '"') o += '"';
                else if (*p == '\\') o += '\\';
                else if (*p == 'n') o += '\n';
                else o += *p;
            } else o += *p;
            p++;
        }
        expect('"');
        return o;
    }
    int64_t num() {
        ws();
        bool neg = false;
        if (p < end && *p == '-') { neg = true; p++; }
        int64_t v = 0;
        while (p < end && *p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
        return neg ? -v : v;
    }
    std::vector<int64_t> int_array() {
        std::vector<int64_t> o;
        expect('[');
        ws();
        if (p < end && *p == ']') { p++; return o; }
        while (true) {
            o.push_back(num());
            ws();
            if (p < end && *p == ',') { p++; continue; }
            break;
        }
        expect(']');
        return o;
    }
    std::vector<std::pair<int64_t,int64_t>> range_array() {
        std::vector<std::pair<int64_t,int64_t>> o;
        expect('[');
        ws();
        if (p < end && *p == ']') { p++; return o; }
        while (true) {
            expect('[');
            int64_t a = num();
            expect(',');
            int64_t b = num();
            expect(']');
            o.emplace_back(a, b);
            ws();
            if (p < end && *p == ',') { p++; continue; }
            break;
        }
        expect(']');
        return o;
    }
    void skip_value() {
        ws();
        if (p >= end) return;
        if (*p == '"') { str(); return; }
        if (*p == '{') {
            p++;
            ws();
            if (p < end && *p == '}') { p++; return; }
            while (true) {
                str(); expect(':'); skip_value();
                ws();
                if (p < end && *p == ',') { p++; continue; }
                break;
            }
            expect('}');
            return;
        }
        if (*p == '[') {
            p++;
            ws();
            if (p < end && *p == ']') { p++; return; }
            while (true) {
                skip_value();
                ws();
                if (p < end && *p == ',') { p++; continue; }
                break;
            }
            expect(']');
            return;
        }
        while (p < end && *p != ',' && *p != '}' && *p != ']') p++;
    }
};

}

std::string shard_filename(const std::string& ckpt_dir, int rank, int world_size) {
    char buf[64];
    snprintf(buf, sizeof(buf), "rank-%05d-of-%05d.safetensors", rank, world_size);
    return ckpt_dir + "/" + buf;
}

std::string metadata_path(const std::string& ckpt_dir) {
    return ckpt_dir + "/.metadata.json";
}

TensorShardInfo make_sharded_spec(
    const std::vector<int64_t>& global_shape,
    const std::string& dtype_str,
    int shard_dim,
    int world_size) {
    TensorShardInfo info;
    info.global_shape = global_shape;
    info.dtype_str = dtype_str;
    info.shard_dim = shard_dim;
    int64_t dim_size = global_shape[shard_dim];
    for (int r = 0; r < world_size; r++) {
        int64_t start = (dim_size * r) / world_size;
        int64_t end = (dim_size * (r + 1)) / world_size;
        info.shard_ranges.emplace_back(start, end);
    }
    return info;
}

TensorShardInfo make_replicated_spec(
    const std::vector<int64_t>& global_shape,
    const std::string& dtype_str) {
    TensorShardInfo info;
    info.global_shape = global_shape;
    info.dtype_str = dtype_str;
    info.shard_dim = -1;
    return info;
}

void write_metadata(const std::string& ckpt_dir, const CheckpointMetadata& meta) {
    std::string json = "{";
    json += "\"version\":" + std::to_string(meta.version) + ",";
    json += "\"world_size\":" + std::to_string(meta.world_size) + ",";
    json += "\"step\":" + std::to_string(meta.step) + ",";
    json += "\"timestamp\":" + std::to_string(meta.timestamp) + ",";
    json += "\"tensors\":{";
    bool first = true;
    for (const auto& kv : meta.tensors) {
        if (!first) json += ",";
        first = false;
        json += "\"" + json_escape_str(kv.first) + "\":{";
        json += "\"global_shape\":" + shape_to_json(kv.second.global_shape) + ",";
        json += "\"dtype\":\"" + json_escape_str(kv.second.dtype_str) + "\",";
        json += "\"shard_dim\":" + std::to_string(kv.second.shard_dim) + ",";
        json += "\"shard_ranges\":" + ranges_to_json(kv.second.shard_ranges);
        json += "}";
    }
    json += "}}";
    std::ofstream out(metadata_path(ckpt_dir), std::ios::binary | std::ios::trunc);
    if (!out.is_open()) throw std::runtime_error("[litetorch Error] dist_checkpoint: cannot write metadata");
    out.write(json.data(), json.size());
}

CheckpointMetadata read_metadata(const std::string& ckpt_dir) {
    std::ifstream in(metadata_path(ckpt_dir), std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("[litetorch Error] dist_checkpoint: cannot open metadata: " + metadata_path(ckpt_dir));
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    JsonReader r(data.data(), data.size());
    CheckpointMetadata meta;
    r.expect('{');
    while (true) {
        r.ws();
        if (r.p < r.end && *r.p == '}') { r.p++; break; }
        std::string key = r.str();
        r.expect(':');
        if (key == "version") meta.version = static_cast<int>(r.num());
        else if (key == "world_size") meta.world_size = static_cast<int>(r.num());
        else if (key == "step") meta.step = r.num();
        else if (key == "timestamp") meta.timestamp = r.num();
        else if (key == "tensors") {
            r.expect('{');
            while (true) {
                r.ws();
                if (r.p < r.end && *r.p == '}') { r.p++; break; }
                std::string tname = r.str();
                r.expect(':');
                r.expect('{');
                TensorShardInfo info;
                while (true) {
                    r.ws();
                    if (r.p < r.end && *r.p == '}') { r.p++; break; }
                    std::string f = r.str();
                    r.expect(':');
                    if (f == "global_shape") info.global_shape = r.int_array();
                    else if (f == "dtype") info.dtype_str = r.str();
                    else if (f == "shard_dim") info.shard_dim = static_cast<int>(r.num());
                    else if (f == "shard_ranges") info.shard_ranges = r.range_array();
                    else r.skip_value();
                    r.ws();
                    if (r.p < r.end && *r.p == ',') { r.p++; continue; }
                }
                meta.tensors[tname] = info;
                r.ws();
                if (r.p < r.end && *r.p == ',') { r.p++; continue; }
            }
        } else r.skip_value();
        r.ws();
        if (r.p < r.end && *r.p == ',') { r.p++; continue; }
    }
    return meta;
}

void save_distributed(
    const std::map<std::string, std::shared_ptr<Tensor>>& local_shards,
    const std::map<std::string, TensorShardInfo>& shard_specs,
    const std::string& ckpt_dir,
    int64_t step) {
    int rank = current_rank();
    int world = current_world_size();
    if (rank == 0) ensure_dir(ckpt_dir);
    maybe_barrier();
    ensure_dir(ckpt_dir);
    std::map<std::string, std::shared_ptr<Tensor>> to_save;
    for (const auto& kv : local_shards) {
        auto it = shard_specs.find(kv.first);
        if (it == shard_specs.end())
            throw std::runtime_error("[litetorch Error] dist_checkpoint: missing shard spec for " + kv.first);
        if (it->second.shard_dim == -1 && rank != 0) continue;
        if (!kv.second) throw std::runtime_error("[litetorch Error] dist_checkpoint: null tensor for " + kv.first);
        to_save[kv.first] = kv.second;
    }
    save_safetensors(to_save, shard_filename(ckpt_dir, rank, world));
    maybe_barrier();
    if (rank == 0) {
        CheckpointMetadata meta;
        meta.version = 1;
        meta.world_size = world;
        meta.step = step;
        meta.timestamp = static_cast<int64_t>(std::time(nullptr));
        meta.tensors = shard_specs;
        write_metadata(ckpt_dir, meta);
    }
    maybe_barrier();
}

namespace {

std::shared_ptr<Tensor> load_shard_slice(
    const std::string& ckpt_dir,
    const CheckpointMetadata& meta,
    const std::string& name,
    const TensorShardInfo& info,
    int64_t want_start,
    int64_t want_end,
    int cur_world) {
    std::vector<std::shared_ptr<Tensor>> pieces;
    if (info.shard_dim == -1) {
        auto m = load_safetensors(shard_filename(ckpt_dir, 0, meta.world_size));
        auto it = m.find(name);
        if (it == m.end()) throw std::runtime_error("[litetorch Error] dist_checkpoint: replicated tensor missing: " + name);
        return it->second;
    }
    int d = info.shard_dim;
    for (int s = 0; s < meta.world_size; s++) {
        int64_t s_start = info.shard_ranges[s].first;
        int64_t s_end = info.shard_ranges[s].second;
        int64_t ov_start = std::max(want_start, s_start);
        int64_t ov_end = std::min(want_end, s_end);
        if (ov_start >= ov_end) continue;
        auto m = load_safetensors(shard_filename(ckpt_dir, s, meta.world_size));
        auto it = m.find(name);
        if (it == m.end()) throw std::runtime_error("[litetorch Error] dist_checkpoint: tensor missing in shard: " + name);
        auto t = it->second;
        int64_t local_start = ov_start - s_start;
        int64_t local_end = ov_end - s_start;
        pieces.push_back(t->slice(d, local_start, local_end));
    }
    if (pieces.empty()) throw std::runtime_error("[litetorch Error] dist_checkpoint: no overlapping shards for " + name);
    if (pieces.size() == 1) return pieces[0]->contiguous();
    return Ops::cat(pieces, d)->contiguous();
}

}

void load_distributed(
    std::map<std::string, std::shared_ptr<Tensor>>& local_shards,
    const std::string& ckpt_dir) {
    int rank = current_rank();
    int world = current_world_size();
    CheckpointMetadata meta = read_metadata(ckpt_dir);
    if (meta.version != 1) throw std::runtime_error("[litetorch Error] dist_checkpoint: unsupported version");
    for (auto& kv : local_shards) {
        const std::string& name = kv.first;
        auto dst = kv.second;
        if (!dst) throw std::runtime_error("[litetorch Error] dist_checkpoint: null destination for " + name);
        auto it = meta.tensors.find(name);
        if (it == meta.tensors.end()) throw std::runtime_error("[litetorch Error] dist_checkpoint: not in checkpoint: " + name);
        const TensorShardInfo& info = it->second;
        std::shared_ptr<Tensor> src;
        if (info.shard_dim == -1) {
            src = load_shard_slice(ckpt_dir, meta, name, info, 0, 0, world);
        } else {
            int d = info.shard_dim;
            int64_t dim_size = info.global_shape[d];
            int64_t want_start = (dim_size * rank) / world;
            int64_t want_end = (dim_size * (rank + 1)) / world;
            if (meta.world_size == world && (int)info.shard_ranges.size() == world) {
                int64_t es = info.shard_ranges[rank].first;
                int64_t ee = info.shard_ranges[rank].second;
                if (es == want_start && ee == want_end) {
                    auto m = load_safetensors(shard_filename(ckpt_dir, rank, meta.world_size));
                    auto fit = m.find(name);
                    if (fit == m.end()) throw std::runtime_error("[litetorch Error] dist_checkpoint: tensor missing: " + name);
                    src = fit->second;
                } else {
                    src = load_shard_slice(ckpt_dir, meta, name, info, want_start, want_end, world);
                }
            } else {
                src = load_shard_slice(ckpt_dir, meta, name, info, want_start, want_end, world);
            }
        }
        if (src->shape != dst->shape)
            throw std::runtime_error("[litetorch Error] dist_checkpoint: shape mismatch for " + name);
        if (src->dtype != dst->dtype)
            throw std::runtime_error("[litetorch Error] dist_checkpoint: dtype mismatch for " + name);
        dst->copy_(src);
    }
    maybe_barrier();
}

}
}
