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
    buffers_.clear();
    PjrtExecutableCache::instance().clear();
    PjrtClient::instance().shutdown();
    available_ = false;
}

int TpuPjrtBackend::num_devices() const {
    return PjrtClient::instance().num_devices();
}

void* TpuPjrtBackend::allocate(size_t bytes) {
    if (!available_) return nullptr;
    std::lock_guard<std::mutex> lock(mutex_);
    size_t n = (bytes + 3) / 4;
    if (n == 0) n = 1;
    std::vector<float> zeros(n, 0.0f);
    std::vector<int64_t> dims{static_cast<int64_t>(n)};
    PjrtBuffer buf;
    if (!buf.from_host(zeros.data(), n * sizeof(float), dims, PJRT_Buffer_Type_F32)) return nullptr;
    void* handle = reinterpret_cast<void*>(next_handle_++);
    BufferEntry entry;
    entry.pjrt_buf = std::move(buf);
    entry.bytes = n * sizeof(float);
    buffers_.emplace(handle, std::move(entry));
    return handle;
}

void TpuPjrtBackend::free_buffer(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_.erase(ptr);
}

bool TpuPjrtBackend::owns(void* ptr) {
    if (!ptr) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    return buffers_.find(ptr) != buffers_.end();
}

size_t TpuPjrtBackend::buffer_bytes(void* ptr) {
    if (!ptr) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buffers_.find(ptr);
    return it == buffers_.end() ? 0 : it->second.bytes;
}

bool TpuPjrtBackend::write_buffer(void* ptr, const void* host, size_t bytes) {
    if (!ptr || !host || !available_) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buffers_.find(ptr);
    if (it == buffers_.end()) return false;
    size_t n = (bytes + 3) / 4;
    if (n == 0) n = 1;
    std::vector<int64_t> dims{static_cast<int64_t>(n)};
    PjrtBuffer nb;
    if (!nb.from_host(host, n * sizeof(float), dims, PJRT_Buffer_Type_F32)) return false;
    it->second.pjrt_buf = std::move(nb);
    it->second.bytes = n * sizeof(float);
    return true;
}

bool TpuPjrtBackend::read_buffer(void* ptr, void* host, size_t bytes) {
    if (!ptr || !host || !available_) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buffers_.find(ptr);
    if (it == buffers_.end()) return false;
    size_t nbytes = bytes < it->second.bytes ? bytes : it->second.bytes;
    if (nbytes == 0) return true;
    return it->second.pjrt_buf.to_host(host, nbytes);
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

bool run_elementwise(const std::string& key, const std::string& hlo,
                     const float* a, const float* b, float* c, int64_t n, int num_inputs) {
    auto exe = PjrtExecutableCache::instance().get_or_compile(key, hlo);
    if (!exe) return false;
    PjrtBuffer ba, bb;
    std::vector<int64_t> dims{n};
    if (!ba.from_host(a, (size_t)n * sizeof(float), dims, PJRT_Buffer_Type_F32)) return false;
    std::vector<PJRT_Buffer*> inputs{ba.get()};
    if (num_inputs > 1) {
        if (!b) return false;
        if (!bb.from_host(b, (size_t)n * sizeof(float), dims, PJRT_Buffer_Type_F32)) return false;
        inputs.push_back(bb.get());
    }
    std::vector<PJRT_Buffer*> outputs;
    if (!exe->execute(inputs, outputs)) return false;
    if (!exe->await()) {
        for (auto* o : outputs) PjrtBuffer::destroy_buffer(o);
        return false;
    }
    bool ok = !outputs.empty() && outputs[0] &&
              PjrtBuffer::to_host_buffer(outputs[0], c, (size_t)n * sizeof(float));
    for (auto* o : outputs) PjrtBuffer::destroy_buffer(o);
    return ok;
}
}

bool TpuPjrtBackend::matmul(const float* a, const float* b, float* c, int64_t M, int64_t N, int64_t K) {
    if (!available_ || !a || !b || !c || M <= 0 || N <= 0 || K <= 0) return false;
    HloShape sa{HloElementType::F32, {M, K}};
    HloShape sb{HloElementType::F32, {K, N}};
    HloShape sc{HloElementType::F32, {M, N}};
    std::string hlo = HloBuilder::make_dot(sa, sb, sc);
    if (hlo.empty()) return false;
    std::string key = cache_key("dot", {M, N, K});
    auto exe = PjrtExecutableCache::instance().get_or_compile(key, hlo);
    if (!exe) return false;
    PjrtBuffer ba, bb;
    std::vector<int64_t> da{M, K}, db{K, N};
    if (!ba.from_host(a, (size_t)M * K * sizeof(float), da, PJRT_Buffer_Type_F32)) return false;
    if (!bb.from_host(b, (size_t)K * N * sizeof(float), db, PJRT_Buffer_Type_F32)) return false;
    std::vector<PJRT_Buffer*> inputs{ba.get(), bb.get()};
    std::vector<PJRT_Buffer*> outputs;
    if (!exe->execute(inputs, outputs)) return false;
    if (!exe->await()) {
        for (auto* o : outputs) PjrtBuffer::destroy_buffer(o);
        return false;
    }
    bool ok = !outputs.empty() && outputs[0] &&
              PjrtBuffer::to_host_buffer(outputs[0], c, (size_t)M * N * sizeof(float));
    for (auto* o : outputs) PjrtBuffer::destroy_buffer(o);
    return ok;
}

bool TpuPjrtBackend::relu(const float* a, float* b, int64_t n) {
    if (!available_ || !a || !b || n <= 0) return false;
    HloShape sa{HloElementType::F32, {n}};
    HloShape sb{HloElementType::F32, {n}};
    std::string hlo = HloBuilder::make_relu(sa, sb);
    if (hlo.empty()) return false;
    return run_elementwise(cache_key("relu", {n}), hlo, a, nullptr, b, n, 1);
}

bool TpuPjrtBackend::add(const float* a, const float* b, float* c, int64_t n) {
    if (!available_ || !a || !b || !c || n <= 0) return false;
    HloShape sa{HloElementType::F32, {n}};
    HloShape sb{HloElementType::F32, {n}};
    HloShape sc{HloElementType::F32, {n}};
    std::string hlo = HloBuilder::make_add(sa, sb, sc);
    if (hlo.empty()) return false;
    return run_elementwise(cache_key("add", {n}), hlo, a, b, c, n, 2);
}

}
}
