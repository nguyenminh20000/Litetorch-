#ifndef LITETORCH_TPU_PJRT_CLIENT_H
#define LITETORCH_TPU_PJRT_CLIENT_H

#include <string>
#include <vector>
#include <cstdint>

struct PJRT_Api;
struct PJRT_Client;
struct PJRT_Device;
struct PJRT_Buffer;
struct PJRT_LoadedExecutable;
struct PJRT_Event;
struct PJRT_Error;

namespace litetorch {
namespace tpu_pjrt {

class PjrtClient {
public:
    static PjrtClient& instance();
    bool initialize(const std::string& lib_path = "");
    void shutdown();
    bool is_available() const { return client_ != nullptr; }
    PJRT_Client* client() const { return client_; }
    PJRT_Device* default_device() const;
    int num_devices() const { return num_devices_; }
    std::string last_error() const { return last_error_; }
    const PJRT_Api* api() const { return api_; }
    bool check(PJRT_Error* err, const char* what);

private:
    PjrtClient() = default;
    ~PjrtClient();
    PjrtClient(const PjrtClient&) = delete;
    PjrtClient& operator=(const PjrtClient&) = delete;
    bool load_library(const std::string& path);
    std::string find_libtpu();

    void* handle_ = nullptr;
    const PJRT_Api* api_ = nullptr;
    PJRT_Client* client_ = nullptr;
    PJRT_Device** devices_ = nullptr;
    int num_devices_ = 0;
    std::string last_error_;
};

}
}

#endif
