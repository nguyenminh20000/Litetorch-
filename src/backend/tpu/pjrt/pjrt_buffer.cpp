#include "pjrt_buffer.h"
#include "pjrt_client.h"
#include "../../../../third_party/pjrt/pjrt_c_api.h"
#include <cstring>

namespace litetorch {
namespace tpu_pjrt {

namespace {

size_t pjrt_element_size(int element_type) {
    switch (static_cast<PJRT_Buffer_Type>(element_type)) {
        case PJRT_Buffer_Type_PRED:
        case PJRT_Buffer_Type_S8:
        case PJRT_Buffer_Type_U8:
        case PJRT_Buffer_Type_F8E5M2:
        case PJRT_Buffer_Type_F8E4M3FN:
        case PJRT_Buffer_Type_F8E4M3B11FNUZ:
        case PJRT_Buffer_Type_F8E5M2FNUZ:
        case PJRT_Buffer_Type_F8E4M3FNUZ:
            return 1;
        case PJRT_Buffer_Type_S16:
        case PJRT_Buffer_Type_U16:
        case PJRT_Buffer_Type_F16:
        case PJRT_Buffer_Type_BF16:
            return 2;
        case PJRT_Buffer_Type_S32:
        case PJRT_Buffer_Type_U32:
        case PJRT_Buffer_Type_F32:
            return 4;
        case PJRT_Buffer_Type_S64:
        case PJRT_Buffer_Type_U64:
        case PJRT_Buffer_Type_F64:
        case PJRT_Buffer_Type_C64:
            return 8;
        case PJRT_Buffer_Type_C128:
            return 16;
        default:
            return 0;
    }
}

}

bool PjrtBuffer::from_host(const void* data, size_t bytes, const std::vector<int64_t>& dims, int element_type) {
    auto& pc = PjrtClient::instance();
    if (!pc.is_available() || !data) return false;
    size_t elem = pjrt_element_size(element_type);
    if (elem == 0) return false;
    size_t expect = elem;
    for (int64_t d : dims) {
        if (d < 0) return false;
        expect *= static_cast<size_t>(d);
    }
    if (bytes != expect) return false;
    const PJRT_Api* api = pc.api();
    PJRT_Client_BufferFromHostBuffer_Args args{};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = pc.client();
    args.data = data;
    args.type = static_cast<PJRT_Buffer_Type>(element_type);
    args.dims = dims.data();
    args.num_dims = dims.size();
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    args.device = pc.default_device();
    if (!pc.check(api->PJRT_Client_BufferFromHostBuffer(&args), "BufferFromHostBuffer")) return false;
    buf_ = args.buffer;
    if (args.done_with_host_buffer) {
        PJRT_Event_Await_Args wargs{};
        wargs.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        wargs.event = args.done_with_host_buffer;
        pc.check(api->PJRT_Event_Await(&wargs), "Event_Await");
        PJRT_Event_Destroy_Args dargs{};
        dargs.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        dargs.event = args.done_with_host_buffer;
        api->PJRT_Event_Destroy(&dargs);
    }
    (void)bytes;
    return true;
}

bool PjrtBuffer::to_host_buffer(PJRT_Buffer* buf, void* out, size_t bytes) {
    auto& pc = PjrtClient::instance();
    if (!pc.is_available() || !buf) return false;
    const PJRT_Api* api = pc.api();
    PJRT_Buffer_ToHostBuffer_Args args{};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = buf;
    args.dst = out;
    args.dst_size = bytes;
    if (!pc.check(api->PJRT_Buffer_ToHostBuffer(&args), "ToHostBuffer")) return false;
    return true;
}

bool PjrtBuffer::to_host(void* out, size_t bytes) {
    return to_host_buffer(buf_, out, bytes);
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
    destroy_buffer(buf_);
    buf_ = nullptr;
}

void PjrtBuffer::destroy_buffer(PJRT_Buffer* buf) {
    auto& pc = PjrtClient::instance();
    if (buf && pc.api()) {
        PJRT_Buffer_Destroy_Args args{};
        args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        args.buffer = buf;
        PJRT_Error* err = pc.api()->PJRT_Buffer_Destroy(&args);
        if (err) pc.check(err, "Buffer_Destroy");
    }
}

}
}
