#ifndef LITETORCH_TPU_PJRT_BUFFER_H
#define LITETORCH_TPU_PJRT_BUFFER_H

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

struct PJRT_Buffer;
struct PJRT_Device;

namespace litetorch {
namespace tpu_pjrt {

class PjrtBuffer {
public:
    PjrtBuffer() = default;
    ~PjrtBuffer() { destroy(); }
    PjrtBuffer(const PjrtBuffer&) = delete;
    PjrtBuffer& operator=(const PjrtBuffer&) = delete;
    PjrtBuffer(PjrtBuffer&& o) noexcept { move_from(o); }
    PjrtBuffer& operator=(PjrtBuffer&& o) noexcept { destroy(); move_from(o); return *this; }

    bool from_host(const void* data, size_t bytes, const std::vector<int64_t>& dims, int element_type);
    bool to_host(void* out, size_t bytes);
    bool await_ready();
    void destroy();
    bool valid() const { return buf_ != nullptr; }
    PJRT_Buffer* get() const { return buf_; }
    size_t on_device_bytes();
    static bool to_host_buffer(PJRT_Buffer* buf, void* out, size_t bytes);
    static void destroy_buffer(PJRT_Buffer* buf);

private:
    void move_from(PjrtBuffer& o) { buf_ = o.buf_; o.buf_ = nullptr; }
    PJRT_Buffer* buf_ = nullptr;
};

}
}

#endif
