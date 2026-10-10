#include "pjrt_buffer.h"
#include "pjrt_client.h"
#include "../../../../third_party/pjrt/pjrt_c_api.h"
#include <cstring>

namespace litetorch {
namespace tpu_pjrt {

bool PjrtBuffer::from_host(const void* data, size_t bytes, const std::vector<int64_t>& dims, int element_type) {
    auto& pc = PjrtClient::instance();
    if (!pc.is_available()) return false;
    const PJRT_Api* api = pc.api();
    PJRT_Client_BufferFromHostBuffer_Args args{};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = pc.client();
    args.data = data;
    args.type = static_cast<PJRT_Buffer_Type>(element_type);
    args.dims = dims.data();
    args.num_dims = dims.size();
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableZeroCopy;
    args.device = pc.default_device();
    if (!pc.check(api->PJRT_Client_BufferFromHostBuffer(&args), "BufferFromHostBuffer")) return false;
    buf_ = args.buffer;
    (void)bytes;
    return true;
}

bool PjrtBuffer::to_host(void* out, size_t bytes) {
    auto& pc = PjrtClient::instance();
    if (!pc.is_available() || !buf_) return false;
    const PJRT_Api* api = pc.api();
    PJRT_Buffer_ToHostBuffer_Args args{};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = buf_;
    args.dst = out;
    args.dst_size = bytes;
    if (!pc.check(api->PJRT_Buffer_ToHostBuffer(&args), "ToHostBuffer")) return false;
    return true;
}

bool PjrtBuffer::await_ready() {
    auto& pc = PjrtClient::instance();
    if (!pc.is_available() || !buf_) return false;
    const PJRT_Api* api = pc.api();
    PJRT_Buffer_ReadyEvent_Args args{};
    args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    args.buffer = buf_;
    if (!pc.check(api->PJRT_Buffer_ReadyEvent(&args), "ReadyEvent")) return false;
    if (args.event) {
        PJRT_Event_Await_Args wargs{};
        wargs.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        wargs.event = args.event;
        bool ok = pc.check(api->PJRT_Event_Await(&wargs), "Event_Await");
        PJRT_Event_Destroy_Args dargs{};
        dargs.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        dargs.event = args.event;
        api->PJRT_Event_Destroy(&dargs);
        return ok;
    }
    return true;
}

size_t PjrtBuffer::on_device_bytes() {
    auto& pc = PjrtClient::instance();
    if (!pc.is_available() || !buf_) return 0;
    const PJRT_Api* api = pc.api();
    PJRT_Buffer_OnDeviceSizeInBytes_Args args{};
    args.struct_size = PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE;
    args.buffer = buf_;
    if (!pc.check(api->PJRT_Buffer_OnDeviceSizeInBytes(&args), "OnDeviceSizeInBytes")) return 0;
    return args.on_device_size_in_bytes;
}

void PjrtBuffer::destroy() {
    auto& pc = PjrtClient::instance();
    if (buf_ && pc.api()) {
        PJRT_Buffer_Destroy_Args args{};
        args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        args.buffer = buf_;
        PJRT_Error* err = pc.api()->PJRT_Buffer_Destroy(&args);
        if (err) pc.check(err, "Buffer_Destroy");
    }
    buf_ = nullptr;
}

}
}
