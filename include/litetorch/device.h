#ifndef LITETORCH_DEVICE_H
#define LITETORCH_DEVICE_H

#include <string>
#include <iostream>
#include <cstddef>

namespace litetorch {

enum class DeviceType {
    CPU,
    GPU,
    META,
    TPU
};

struct Device {
    DeviceType type;
    int index;

    Device(DeviceType type = DeviceType::CPU, int index = 0)
        : type(type), index(index) {}

    bool operator==(const Device& other) const {
        return type == other.type && index == other.index;
    }

    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        switch (type) {
            case DeviceType::CPU:  return "cpu:" + std::to_string(index);
            case DeviceType::GPU:  return "gpu:" + std::to_string(index);
            case DeviceType::TPU:  return "tpu:" + std::to_string(index);
            case DeviceType::META: return "meta";
            default:               return "unknown";
        }
    }
};

inline std::ostream& operator<<(std::ostream& os, const Device& device) {
    os << device.to_string();
    return os;
}

namespace cuda {

bool is_available();
int device_count();
int current_device();
void set_device(int device_id);
size_t memory_reserved();
void manual_seed_all(unsigned int seed);

}

}

#endif
