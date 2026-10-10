#ifndef LITETORCH_TPU_PJRT_EXECUTABLE_H
#define LITETORCH_TPU_PJRT_EXECUTABLE_H

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <memory>

struct PJRT_LoadedExecutable;
struct PJRT_Buffer;

namespace litetorch {
namespace tpu_pjrt {

class PjrtExecutable {
public:
    PjrtExecutable() = default;
    ~PjrtExecutable() { destroy(); }
    PjrtExecutable(const PjrtExecutable&) = delete;
    PjrtExecutable& operator=(const PjrtExecutable&) = delete;

    bool compile(const std::string& hlo_proto_bytes, const std::string& name = "");
    bool execute(const std::vector<PJRT_Buffer*>& inputs, std::vector<PJRT_Buffer*>& outputs);
    bool await();
    void destroy();
    bool valid() const { return exe_ != nullptr; }

private:
    PJRT_LoadedExecutable* exe_ = nullptr;
    void* execute_event_ = nullptr;
};

class PjrtExecutableCache {
public:
    static PjrtExecutableCache& instance();
    std::shared_ptr<PjrtExecutable> get_or_compile(const std::string& key, const std::string& hlo_bytes);
    void clear();

private:
    std::unordered_map<std::string, std::shared_ptr<PjrtExecutable>> cache_;
    std::mutex mutex_;
};

}
}

#endif
