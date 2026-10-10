#include "pjrt_executable.h"
#include "pjrt_client.h"
#include "../../../../third_party/pjrt/pjrt_c_api.h"

namespace litetorch {
namespace tpu_pjrt {

bool PjrtExecutable::compile(const std::string& hlo_proto_bytes, const std::string& name) {
    auto& pc = PjrtClient::instance();
    if (!pc.is_available()) return false;
    const PJRT_Api* api = pc.api();
    PJRT_Program prog{};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.format = "hlo";
    prog.format_size = 3;
    prog.code = const_cast<char*>(hlo_proto_bytes.data());
    prog.code_size = hlo_proto_bytes.size();
    PJRT_Client_Compile_Args args{};
    args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    args.client = pc.client();
    args.program = &prog;
    if (!pc.check(api->PJRT_Client_Compile(&args), "Client_Compile")) return false;
    exe_ = args.executable;
    (void)name;
    return true;
}

bool PjrtExecutable::execute(const std::vector<PJRT_Buffer*>& inputs, std::vector<PJRT_Buffer*>& outputs) {
    auto& pc = PjrtClient::instance();
    if (!pc.is_available() || !exe_) return false;
    const PJRT_Api* api = pc.api();
    PJRT_Device* device = pc.default_device();
    PJRT_Buffer* const* arg_list = inputs.data();
    PJRT_Buffer* const* const* arg_lists = &arg_list;
    size_t num_outputs = 1;
    PJRT_Buffer*** output_lists = nullptr;
    PJRT_Event** events = nullptr;
    PJRT_LoadedExecutable_Execute_Args args{};
    args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    args.executable = exe_;
    args.argument_lists = arg_lists;
    args.num_devices = 1;
    args.num_args = inputs.size();
    args.output_lists = output_lists;
    args.device_complete_events = events;
    args.execute_device = device;
    if (!pc.check(api->PJRT_LoadedExecutable_Execute(&args), "Execute")) return false;
    (void)num_outputs;
    outputs.clear();
    return true;
}

bool PjrtExecutable::await() {
    auto& pc = PjrtClient::instance();
    if (!execute_event_) return true;
    const PJRT_Api* api = pc.api();
    PJRT_Event* event = static_cast<PJRT_Event*>(execute_event_);
    PJRT_Event_Await_Args args{};
    args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    args.event = event;
    bool ok = pc.check(api->PJRT_Event_Await(&args), "Event_Await");
    PJRT_Event_Destroy_Args dargs{};
    dargs.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    dargs.event = event;
    api->PJRT_Event_Destroy(&dargs);
    execute_event_ = nullptr;
    return ok;
}

void PjrtExecutable::destroy() {
    auto& pc = PjrtClient::instance();
    if (exe_ && pc.api()) {
        PJRT_LoadedExecutable_Destroy_Args args{};
        args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        args.executable = exe_;
        PJRT_Error* err = pc.api()->PJRT_LoadedExecutable_Destroy(&args);
        if (err) pc.check(err, "LoadedExecutable_Destroy");
    }
    exe_ = nullptr;
    execute_event_ = nullptr;
}

PjrtExecutableCache& PjrtExecutableCache::instance() {
    static PjrtExecutableCache inst;
    return inst;
}

std::shared_ptr<PjrtExecutable> PjrtExecutableCache::get_or_compile(const std::string& key, const std::string& hlo_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    auto exe = std::make_shared<PjrtExecutable>();
    if (!exe->compile(hlo_bytes, key)) return nullptr;
    cache_[key] = exe;
    return exe;
}

void PjrtExecutableCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

}
}
