#include "tpu_common.h"
#include "litetorch/platform.h"
#include <vector>
#include <string>
#include <cstdlib>

namespace litetorch {
namespace tpu_internal {

static TPUDriverState g_driver_state;
static std::mutex g_driver_mutex;

TPUDriverState& get_tpu_driver_state() {
    return g_driver_state;
}

bool init_tpu_runtime() {
    std::lock_guard<std::mutex> lock(g_driver_mutex);
    if (g_driver_state.is_available) {
        return true;
    }
    if (getenv("LITETORCH_NO_TPU")) {
        return false;
    }

    std::vector<std::string> search_paths = {
        "libtpu.so",
        "/usr/lib/libtpu.so",
        "/usr/local/lib/libtpu.so",
        "/lib/libtpu.so",
        "/usr/lib/x86_64-linux-gnu/libtpu.so",
        "libtpu.dll",
        "./build/libtpu.dll",
        "./build/libtpu.so"
    };

    const char* custom_path = getenv("TPU_LIBRARY_PATH");
    if (custom_path) {
        search_paths.insert(search_paths.begin(), std::string(custom_path));
    }
    const char* libtpu_env = getenv("LIBTPU_PATH");
    if (libtpu_env) {
        search_paths.insert(search_paths.begin(), std::string(libtpu_env));
    }

    for (const auto& path : search_paths) {
        g_driver_state.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (g_driver_state.handle) {
            break;
        }
    }

    const char* tpu_name = getenv("TPU_NAME");
    const char* tpu_accelerator = getenv("TPU_ACCELERATOR_TYPE");

    if (g_driver_state.handle) {
        g_driver_state.is_available = true;
        g_driver_state.num_devices = 8;
        if (tpu_accelerator) {
            g_driver_state.device_name = std::string("Google TPU (") + tpu_accelerator + ")";
        } else if (tpu_name) {
            g_driver_state.device_name = std::string("Google TPU (") + tpu_name + ")";
        } else {
            g_driver_state.device_name = "Google TPU (libtpu.so / PJRT Engine)";
        }
        return true;
    }

    g_driver_state.is_available = false;
    g_driver_state.num_devices = 0;
    return false;
}

void shutdown_tpu_runtime() {
    std::lock_guard<std::mutex> lock(g_driver_mutex);
    if (g_driver_state.handle) {
        dlclose(g_driver_state.handle);
        g_driver_state.handle = nullptr;
    }
    g_driver_state.is_available = false;
    g_driver_state.num_devices = 0;
}

}
}
