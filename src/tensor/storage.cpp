#include "litetorch/tensor.h"
#include "litetorch/memory_manager.h"
#include "litetorch/cl_backend.h"
#include "litetorch/allocator.h"
#include "litetorch/backend.h"
#include <cstring>

namespace litetorch {

StorageImpl::StorageImpl(size_t size, const Device& device, DataType dtype) : size(size), device(device), dtype(dtype) {
    if (device.type == DeviceType::META) {
        return;
    }
    std::lock_guard<std::mutex> mem_lock(MemoryManager::get().get_mutex());
    std::lock_guard<std::mutex> storage_lock(storage_mutex_);
    if (device.type == DeviceType::GPU) {
        if (CLBackend::get().is_available()) {
            auto native = BackendDispatcher::get().get_backend();
            if (native && native->is_available()) {
                native->set_device(device.index);
            }
            MemoryManager::get().register_gpu_impl(this);
            gpu_data = CLBackend::get().allocate(size * element_size());
            if (!gpu_data) {
                MemoryManager::get().unregister_gpu_impl(this);
                this->device = Device(DeviceType::CPU, 0);
                cpu_data = (float*)CachingAllocator::get().allocate_cpu(size * element_size());
            }
        } else {
            this->device = Device(DeviceType::CPU, 0);
            cpu_data = (float*)CachingAllocator::get().allocate_cpu(size * element_size());
        }
    } else {
        cpu_data = (float*)CachingAllocator::get().allocate_cpu(size * element_size());
    }
}

StorageImpl::~StorageImpl() {
    if (device.type == DeviceType::META) {
        return;
    }
    std::lock_guard<std::mutex> mem_lock(MemoryManager::get().get_mutex());
    std::lock_guard<std::mutex> storage_lock(storage_mutex_);
    MemoryManager::get().unregister_gpu_impl(this);
    if (gpu_data) {
        CLBackend::get().free(gpu_data);
    }
    if (cpu_data) {
        CachingAllocator::get().free_cpu(cpu_data);
    }
}

float* StorageImpl::get_cpu_ptr() {
    if (device.type == DeviceType::META) {
        throw std::runtime_error("[litetorch Error] Cannot access CPU pointer for a Tensor on META device");
    }
    std::lock_guard<std::mutex> storage_lock(storage_mutex_);
    if (is_swapped) {
        return cpu_data;
    }
    if (device.type == DeviceType::GPU && gpu_data) {
        if (!cpu_data) {
            cpu_data = (float*)CachingAllocator::get().allocate_cpu(size * element_size());
            CLBackend::get().read(gpu_data, size * element_size(), cpu_data);
        }
    }
    return cpu_data;
}

cl_mem StorageImpl::get_gpu_ptr() {
    if (device.type == DeviceType::META) {
        throw std::runtime_error("[litetorch Error] Cannot access GPU pointer for a Tensor on META device");
    }
    std::lock_guard<std::mutex> mem_lock(MemoryManager::get().get_mutex());
    std::lock_guard<std::mutex> storage_lock(storage_mutex_);
    if (device.type == DeviceType::CPU) {
        return nullptr;
    }
    if (is_swapped) {
        swap_in_impl();
    }
    if (device.type == DeviceType::CPU) {
        return nullptr;
    }
    MemoryManager::get().touch_impl(this);
    if (cpu_data) {
        CachingAllocator::get().free_cpu(cpu_data);
        cpu_data = nullptr;
    }
    return gpu_data;
}

void StorageImpl::to(const Device& new_device) {
    std::lock_guard<std::mutex> mem_lock(MemoryManager::get().get_mutex());
    std::lock_guard<std::mutex> storage_lock(storage_mutex_);
    if (device == new_device) return;

    if (new_device.type == DeviceType::GPU) {
        if (!CLBackend::get().is_available()) return;
        
        auto native = BackendDispatcher::get().get_backend();
        if (native && native->is_available()) {
            native->set_device(new_device.index);
        }
        MemoryManager::get().register_gpu_impl(this);
        gpu_data = CLBackend::get().allocate(size * element_size());
        if (!gpu_data) {
            MemoryManager::get().unregister_gpu_impl(this);
            return;
        }

        if (cpu_data) {
            CLBackend::get().write(gpu_data, size * element_size(), cpu_data);
            CachingAllocator::get().free_cpu(cpu_data);
            cpu_data = nullptr;
        }
        device = new_device;
        is_swapped = false;
    } else {
        if (is_swapped) {
            is_swapped = false;
            device = new_device;
            return;
        }

        if (!cpu_data) {
            cpu_data = (float*)CachingAllocator::get().allocate_cpu(size * element_size());
        }
        if (gpu_data) {
            CLBackend::get().read(gpu_data, size * element_size(), cpu_data);
            CLBackend::get().free(gpu_data);
            gpu_data = nullptr;
        }
        MemoryManager::get().unregister_gpu_impl(this);
        device = new_device;
    }
}

void StorageImpl::evict_impl() {
    if (is_swapped || !gpu_data) return;

    if (!cpu_data) {
        cpu_data = (float*)CachingAllocator::get().allocate_cpu(size * element_size());
    }
    CLBackend::get().read(gpu_data, size * element_size(), cpu_data);
    CLBackend::get().free(gpu_data);
    gpu_data = nullptr;
    is_swapped = true;

    MemoryManager::get().unregister_gpu_impl(this);
}

void StorageImpl::evict() {
    std::lock_guard<std::mutex> mem_lock(MemoryManager::get().get_mutex());
    std::lock_guard<std::mutex> storage_lock(storage_mutex_);
    evict_impl();
}

void StorageImpl::discard_gpu_impl() {
    if (is_swapped || !gpu_data) return;
    CLBackend::get().free(gpu_data);
    gpu_data = nullptr;
    is_swapped = true;
    MemoryManager::get().unregister_gpu_impl(this);
}

void StorageImpl::discard_gpu() {
    std::lock_guard<std::mutex> mem_lock(MemoryManager::get().get_mutex());
    std::lock_guard<std::mutex> storage_lock(storage_mutex_);
    discard_gpu_impl();
}

void StorageImpl::swap_in_impl() {
    if (!is_swapped || gpu_data) return;

    auto native = BackendDispatcher::get().get_backend();
    if (native && native->is_available()) {
        native->set_device(device.index);
    }
    MemoryManager::get().register_gpu_impl(this);
    gpu_data = CLBackend::get().allocate(size * element_size());
    if (!gpu_data) {
        MemoryManager::get().unregister_gpu_impl(this);
        is_swapped = false;
        device = Device(DeviceType::CPU, 0);
        return;
    }

    CLBackend::get().write(gpu_data, size * element_size(), cpu_data);
    CachingAllocator::get().free_cpu(cpu_data);
    cpu_data = nullptr;
    is_swapped = false;
}

void StorageImpl::swap_in() {
    std::lock_guard<std::mutex> mem_lock(MemoryManager::get().get_mutex());
    std::lock_guard<std::mutex> storage_lock(storage_mutex_);
    swap_in_impl();
}

void StorageImpl::ensure_cpu() {
    std::lock_guard<std::mutex> storage_lock(storage_mutex_);
    if (is_swapped) {
        return;
    }
    if (device.type == DeviceType::GPU && gpu_data) {
        if (!cpu_data) {
            cpu_data = (float*)CachingAllocator::get().allocate_cpu(size * element_size());
        }
        CLBackend::get().read(gpu_data, size * element_size(), cpu_data);
    }
}

}
