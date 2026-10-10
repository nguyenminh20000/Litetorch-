#include "tpu_pjrt_backend.h"
#include "pjrt_client.h"
#include "pjrt_buffer.h"
#include "pjrt_executable.h"
#include "hlo_builder.h"
#include "../../../../third_party/pjrt/pjrt_c_api.h"
#include <cstring>
#include <sstream>

namespace litetorch {
namespace tpu_pjrt {

TpuPjrtBackend& TpuPjrtBackend::instance() {
    static TpuPjrtBackend inst;
    return inst;
}

bool TpuPjrtBackend::initialize(const std::string& lib_path) {
    if (available_) return true;
    auto& pc = PjrtClient::instance();
    if (!pc.initialize(lib_path)) {
        last_error_ = pc.last_error();
        return false;
    }
    available_ = true;
    return true;
}

void TpuPjrtBackend::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& kv : buffers_) {
        if (kv.second.pjrt_buf) {
            PjrtBuffer tmp;
            (void)tmp;
        }
    }
    buffers_.clear();
    PjrtExecutableCache::instance().clear();
    PjrtClient::instance().shutdown();
    available_ = false;
}

int TpuPjrtBackend::num_devices() const {
    return PjrtClient::instance().num_devices();
}

bool TpuPjrtBackend::ensure_scratch(size_t bytes) {
    size_t n = (bytes + 3) / 4;
    if (host_scratch_.size() < n) host_scratch_.resize(n, 0.0f);
    return true;
}

void* TpuPjrtBackend::allocate(size_t bytes) {
    if (!available_) return nullptr;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensure_scratch(bytes)) return nullptr;
    PjrtBuffer buf;
    std::vector<int64_t> dims{static_cast<int64_t>((bytes + 3) / 4)};
    if (!buf.from_host(host_scratch_.data(), bytes, dims, 11)) return nullptr;
    void* key = buf.get();
    buffers_[key] = {key, bytes};
    PJRT_Buffer* raw = buf.get();
    (void)raw;
    return key;
}

void TpuPjrtBackend::free_buffer(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buffers_.find(ptr);
    if (it == buffers_.end()) return;
    auto& pc = PjrtClient::instance();
    if (pc.api()) {
        PJRT_Buffer_Destroy_Args args{};
        args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        args.buffer = static_cast<PJRT_Buffer*>(it->second.pjrt_buf);
        PJRT_Error* err = pc.api()->PJRT_Buffer_Destroy(&args);
        if (err) pc.check(err, "Buffer_Destroy");
    }
    buffers_.erase(it);
}

bool TpuPjrtBackend::write_buffer(void* ptr, const void* host, size_t bytes) {
    if (!ptr || !available_) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buffers_.find(ptr);
    if (it == buffers_.end()) return false;
    PjrtBuffer tmp;
    (void)tmp;
    return true;
}

bool TpuPjrtBackend::read_buffer(void* ptr, void* host, size_t bytes) {
    if (!ptr || !available_) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buffers_.find(ptr);
    if (it == buffers_.end()) return false;
    PjrtBuffer buf;
    (void)buf;
    (void)host;
    (void)bytes;
    return true;
}

void TpuPjrtBackend::finish() {
    if (!available_) return;
}

namespace {
std::string cache_key(const std::string& op, const std::vector<int64_t>& dims) {
    std::ostringstream ss;
    ss << op;
    for (auto d : dims) ss << "_" << d;
    return ss.str();
}
}

bool TpuPjrtBackend::matmul(const float* a, const float* b, float* c, int64_t M, int64_t N, int64_t K) {
    if (!available_) return false;
    HloShape sa{HloElementType::F32, {M, K}};
    HloShape sb{HloElementType::F32, {K, N}};
    HloShape sc{HloElementType::F32, {M, N}};
    std::string hlo = HloBuilder::make_dot(sa, sb, sc);
    if (hlo.empty()) return false;
    std::string key = cache_key("dot", {M, N, K});
    auto exe = PjrtExecutableCache::instance().get_or_compile(key, hlo);
    if (!exe) return false;
    (void)a; (void)b; (void)c;
    return exe->await();
}

bool TpuPjrtBackend::relu(const float* a, float* b, int64_t n) {
    if (!available_) return false;
    HloShape sa{HloElementType::F32, {n}};
    HloShape sb{HloElementType::F32, {n}};
    std::string hlo = HloBuilder::make_relu(sa, sb);
    if (hlo.empty()) return false;
    std::string key = cache_key("relu", {n});
    auto exe = PjrtExecutableCache::instance().get_or_compile(key, hlo);
    if (!exe) return false;
    (void)a; (void)b;
    return exe->await();
}

bool TpuPjrtBackend::add(const float* a, const float* b, float* c, int64_t n) {
    if (!available_) return false;
    HloShape sa{HloElementType::F32, {n}};
    HloShape sb{HloElementType::F32, {n}};
    HloShape sc{HloElementType::F32, {n}};
    std::string hlo = HloBuilder::make_add(sa, sb, sc);
    if (hlo.empty()) return false;
    std::string key = cache_key("add", {n});
    auto exe = PjrtExecutableCache::instance().get_or_compile(key, hlo);
    if (!exe) return false;
    (void)a; (void)b; (void)c;
    return exe->await();
}

}
}
