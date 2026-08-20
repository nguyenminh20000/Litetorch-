#include "litetorch/tensor.h"
#include "litetorch/memory_manager.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace litetorch;

int main() {
    auto t_int8 = Tensor::from_vector({10.0f, 20.0f, 30.0f}, {3}, Device(DeviceType::CPU), false, DataType::INT8);
    assert(t_int8->dtype == DataType::INT8);
    auto v_int8 = t_int8->to_vector();
    assert(v_int8.size() == 3);
    assert(std::abs(v_int8[0] - 10.0f) < 1e-5f);
    assert(std::abs(v_int8[1] - 20.0f) < 1e-5f);
    assert(std::abs(v_int8[2] - 30.0f) < 1e-5f);

    auto t_fp16 = Tensor::from_vector({1.5f, 2.5f, 3.5f}, {3}, Device(DeviceType::CPU), false, DataType::FP16);
    assert(t_fp16->dtype == DataType::FP16);
    auto v_fp16 = t_fp16->to_vector();
    assert(v_fp16.size() == 3);
    assert(std::abs(v_fp16[0] - 1.5f) < 1e-3f);
    assert(std::abs(v_fp16[1] - 2.5f) < 1e-3f);
    assert(std::abs(v_fp16[2] - 3.5f) < 1e-3f);

    if (CLBackend::get().is_available()) {
        size_t init_used = MemoryManager::get().get_gpu_used();
        auto storage = std::make_shared<StorageImpl>(1000, Device(DeviceType::GPU), DataType::INT8);
        size_t diff = MemoryManager::get().get_gpu_used() - init_used;
        assert(diff == 1000);
    }

    std::cout << "SUCCESS: MemoryManager and from_vector optimizations verified!" << std::endl;
    return 0;
}
