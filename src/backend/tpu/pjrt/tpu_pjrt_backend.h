#ifndef LITETORCH_TPU_PJRT_BACKEND_H
#define LITETORCH_TPU_PJRT_BACKEND_H

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include <mutex>
#include <memory>

namespace litetorch {
namespace tpu_pjrt {

class TpuPjrtBackend {
public:
    static TpuPjrtBackend& instance();
    bool initialize(const std::string& lib_path = "");
    void shutdown();
    bool is_available() const { return available_; }
    std::string last_error() const { return last_error_; }
    int num_devices() const;

    void* allocate(size_t bytes);
    void free_buffer(void* ptr);
    bool write_buffer(void* ptr, const void* host, size_t bytes);
    bool read_buffer(void* ptr, void* host, size_t bytes);
    void finish();

    bool matmul(const float* a, const float* b, float* c, int64_t M, int64_t N, int64_t K);
    bool relu(const float* a, float* b, int64_t n);
    bool add(const float* a, const float* b, float* c, int64_t n);

private:
    TpuPjrtBackend() = default;
    ~TpuPjrtBackend() { shutdown(); }
    TpuPjrtBackend(const TpuPjrtBackend&) = delete;
    TpuPjrtBackend& operator=(const TpuPjrtBackend&) = delete;

    struct BufferEntry {
        void* pjrt_buf = nullptr;
        size_t bytes = 0;
    };
    bool ensure_scratch(size_t bytes);
    void* to_pjrt_ptr(void* ptr);

    bool available_ = false;
    std::string last_error_;
    std::unordered_map<void*, BufferEntry> buffers_;
    std::mutex mutex_;
    std::vector<float> host_scratch_;
};

}
}

#endif
