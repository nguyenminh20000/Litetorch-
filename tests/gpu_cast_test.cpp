#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace litetorch;

void verify_cast(DataType target_dtype, float tolerance) {
    int size = 128;
    std::vector<float> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = sin(static_cast<float>(i)) * 2.0f;
    }

    auto t_cpu = Tensor::from_vector(data, {size}, Device(DeviceType::CPU, 0), false);
    auto t_gpu = Tensor::from_vector(data, {size}, Device(DeviceType::GPU, 0), false);

    auto cast_cpu = t_cpu->cast(target_dtype);
    auto cast_gpu = t_gpu->cast(target_dtype);

    auto back_cpu = cast_cpu->cast(DataType::FP32)->to_vector();
    auto back_gpu = cast_gpu->cast(DataType::FP32)->to_vector();

    for (int i = 0; i < size; ++i) {
        float diff = std::abs(back_cpu[i] - back_gpu[i]);
        if (diff > tolerance) {
            std::cerr << "Mismatch at index " << i << " for target datatype " << static_cast<int>(target_dtype)
                      << ": CPU val=" << back_cpu[i] << ", GPU val=" << back_gpu[i] << ", diff=" << diff << std::endl;
            assert(false);
        }
    }
}

int main() {
    if (!CLBackend::get().is_available()) {
        std::cout << "No GPU available. Skipping test." << std::endl;
        return 0;
    }

    verify_cast(DataType::FP16, 1e-4f);
    verify_cast(DataType::BF16, 1e-4f);
    verify_cast(DataType::NF4, 1e-4f);
    verify_cast(DataType::INT8, 1e-4f);
    verify_cast(DataType::INT4, 1e-4f);
    verify_cast(DataType::FP8_E4M3, 1e-4f);
    verify_cast(DataType::FP8_E5M2, 1e-4f);

    std::cout << "All GPU casting tests passed successfully!" << std::endl;
    return 0;
}
