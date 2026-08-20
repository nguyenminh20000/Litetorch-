#ifndef LITETORCH_DEVICE_H
#define LITETORCH_DEVICE_H

#include <string>
#include <iostream>

namespace litetorch {

enum class DeviceType {
    CPU,
    GPU,
    META
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
        if (type == DeviceType::META) return "meta";
        return (type == DeviceType::CPU ? "cpu" : "gpu:") + std::to_string(index);
    }
};

inline std::ostream& operator<<(std::ostream& os, const Device& device) {
    os << device.to_string();
    return os;
}

}

#endif
