#include "pjrt_client.h"
#include "../../../../third_party/pjrt/pjrt_c_api.h"
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>
#include <glob.h>
#include <unistd.h>

namespace litetorch {
namespace tpu_pjrt {

PjrtClient& PjrtClient::instance() {
    static PjrtClient inst;
    return inst;
}

PjrtClient::~PjrtClient() {
    shutdown();
}

bool PjrtClient::check(PJRT_Error* err, const char* what) {
    if (!err) return true;
    PJRT_Error_Message_Args margs{};
    margs.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    margs.error = err;
    api_->PJRT_Error_Message(&margs);
    last_error_ = std::string(what) + ": " + (margs.message ? std::string(margs.message, margs.message_size) : "unknown");
    PJRT_Error_Destroy_Args dargs{};
    dargs.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    dargs.error = err;
    api_->PJRT_Error_Destroy(&dargs);
    return false;
}

std::string PjrtClient::find_libtpu() {
    const char* env_paths[] = {"LIBTPU_PATH", "TPU_LIBRARY_PATH", nullptr};
    for (int i = 0; env_paths[i]; ++i) {
        const char* p = std::getenv(env_paths[i]);
        if (p && p[0]) return std::string(p);
    }
    const char* candidates[] = {
        "libtpu.so",
        "/lib/libtpu.so",
        "/usr/lib/libtpu.so",
        "/usr/local/lib/libtpu.so",
        nullptr
    };
    for (int i = 0; candidates[i]; ++i) {
        void* h = dlopen(candidates[i], RTLD_NOW | RTLD_NOLOAD);
        if (h) { dlclose(h); return std::string(candidates[i]); }
    }
    const char* home = std::getenv("HOME");
    std::string home_pat = home ? std::string(home) + "/.local/lib/python3.*/site-packages/libtpu/libtpu.so*" : "";
    const char* patterns[] = {
        "/usr/local/lib/python3.*/dist-packages/libtpu/libtpu.so*",
        "/usr/local/lib/python3.*/site-packages/libtpu/libtpu.so*",
        "/usr/lib/python3.*/dist-packages/libtpu/libtpu.so*",
        "/opt/conda/lib/python3.*/site-packages/libtpu/libtpu.so*",
        nullptr
    };
    for (int i = 0; patterns[i]; ++i) {
        glob_t g{};
        if (glob(patterns[i], GLOB_NOSORT, nullptr, &g) != 0) continue;
        std::string hit;
        for (size_t j = 0; j < g.gl_pathc; ++j) {
            if (access(g.gl_pathv[j], R_OK) == 0) { hit = g.gl_pathv[j]; break; }
        }
        globfree(&g);
        if (!hit.empty()) return hit;
    }
    if (!home_pat.empty()) {
        glob_t g{};
        if (glob(home_pat.c_str(), GLOB_NOSORT, nullptr, &g) == 0) {
            std::string hit;
            for (size_t j = 0; j < g.gl_pathc; ++j) {
                if (access(g.gl_pathv[j], R_OK) == 0) { hit = g.gl_pathv[j]; break; }
            }
            globfree(&g);
            if (!hit.empty()) return hit;
        }
    }
    return "";
}

bool PjrtClient::load_library(const std::string& path) {
    std::string lib = path.empty() ? find_libtpu() : path;
    if (lib.empty()) {
        last_error_ = "libtpu.so not found";
        return false;
    }
    handle_ = dlopen(lib.c_str(), RTLD_NOW);
    if (!handle_) {
        last_error_ = std::string("dlopen failed: ") + dlerror();
        return false;
    }
    using GetApiFn = const PJRT_Api*(*)();
    auto get_api = reinterpret_cast<GetApiFn>(dlsym(handle_, "GetPjrtApi"));
    if (!get_api) {
        last_error_ = "GetPjrtApi symbol not found";
        dlclose(handle_);
        handle_ = nullptr;
        return false;
    }
    api_ = get_api();
    if (!api_) {
        last_error_ = "GetPjrtApi returned null";
        dlclose(handle_);
        handle_ = nullptr;
        return false;
    }
    if (api_->pjrt_api_version.major_version != PJRT_API_MAJOR) {
        last_error_ = "PJRT API major version mismatch";
        dlclose(handle_);
        handle_ = nullptr;
        api_ = nullptr;
        return false;
    }
    return true;
}

bool PjrtClient::initialize(const std::string& lib_path) {
    if (client_) return true;
    if (!load_library(lib_path)) return false;
    PJRT_Plugin_Initialize_Args iargs{};
    iargs.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
    if (!check(api_->PJRT_Plugin_Initialize(&iargs), "Plugin_Initialize")) return false;
    PJRT_Client_Create_Args cargs{};
    cargs.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    if (!check(api_->PJRT_Client_Create(&cargs), "Client_Create")) return false;
    client_ = cargs.client;
    PJRT_Client_AddressableDevices_Args dargs{};
    dargs.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    dargs.client = client_;
    if (!check(api_->PJRT_Client_AddressableDevices(&dargs), "AddressableDevices")) {
        client_ = nullptr;
        return false;
    }
    devices_ = const_cast<PJRT_Device**>(dargs.addressable_devices);
    num_devices_ = static_cast<int>(dargs.num_addressable_devices);
    if (num_devices_ == 0) {
        last_error_ = "no addressable TPU devices";
        return false;
    }
    return true;
}

void PjrtClient::shutdown() {
    if (client_ && api_) {
        PJRT_Client_Destroy_Args dargs{};
        dargs.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        dargs.client = client_;
        PJRT_Error* err = api_->PJRT_Client_Destroy(&dargs);
        if (err) check(err, "Client_Destroy");
        client_ = nullptr;
    }
    devices_ = nullptr;
    num_devices_ = 0;
    api_ = nullptr;
    if (handle_) {
        dlclose(handle_);
        handle_ = nullptr;
    }
}

PJRT_Device* PjrtClient::default_device() const {
    if (num_devices_ > 0 && devices_) return devices_[0];
    return nullptr;
}

}
}
