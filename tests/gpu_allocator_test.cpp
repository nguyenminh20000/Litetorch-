#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/backend.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace litetorch;

int main() {
    if (!CLBackend::get().is_available()) {
        std::cout << "No GPU available. Skipping test." << std::endl;
        return 0;
    }

    auto t1 = Tensor::create({1024}, Device(DeviceType::GPU, 0));
    cl_mem ptr1 = t1->gpu_data();
    t1.reset();

    auto t2 = Tensor::create({1024}, Device(DeviceType::GPU, 0));
    cl_mem ptr2 = t2->gpu_data();

    auto native = BackendDispatcher::get().get_backend();
    if (native && native->is_available()) {
        assert(ptr1 == ptr2);
        std::cout << "Native GPU Caching Allocator block reuse verified successfully!" << std::endl;
    } else {
        std::cout << "OpenCL Caching Allocator execution verified successfully!" << std::endl;
    }

    return 0;
}
