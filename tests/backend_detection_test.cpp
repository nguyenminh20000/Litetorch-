#include "litetorch/tensor.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "Starting LiteTorch Multi-Backend Detection Test...\n";

    auto native_backend = litetorch::BackendDispatcher::get().get_backend();
    if (native_backend && native_backend->is_available()) {
        std::cout << "[Test Info] Active Backend: Native GPU (CUDA/HIP)\n";
    } else if (litetorch::CLBackend::get().is_available()) {
        std::cout << "[Test Info] Active Backend: OpenCL GPU\n";
    } else {
        std::cout << "[Test Info] Active Backend: CPU Fallback\n";
    }

    litetorch::Device device(litetorch::DeviceType::GPU, 0);
    
    std::cout << "Creating input tensors...\n";
    auto t1 = std::make_shared<litetorch::Tensor>(std::vector<int64_t>{4}, device);
    auto t2 = std::make_shared<litetorch::Tensor>(std::vector<int64_t>{4}, device);

    float h_data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_data2[] = {10.0f, 20.0f, 30.0f, 40.0f};

    cl_mem t1_mem = t1->storage->gpu_data;
    cl_mem t2_mem = t2->storage->gpu_data;

    std::cout << "Writing data to GPU storage...\n";
    litetorch::CLBackend::get().write(t1_mem, 4 * sizeof(float), h_data1);
    litetorch::CLBackend::get().write(t2_mem, 4 * sizeof(float), h_data2);

    std::cout << "Launching elementwise Add operation...\n";
    auto t_out = std::make_shared<litetorch::Tensor>(std::vector<int64_t>{4}, device);
    
    auto kernel = litetorch::CLBackend::get().get_kernel("litetorch_kernels", "", "elementwise_add");
    assert(kernel != nullptr);

    cl_mem out_mem = t_out->storage->gpu_data;
    int t1_off = 0;
    int t2_off = 0;
    int out_off = 0;
    int size = 4;

    std::vector<void*> args = {&t1_mem, &t1_off, &t2_mem, &t2_off, &out_mem, &out_off, &size};
    std::vector<size_t> arg_sizes = {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)};

    litetorch::CLBackend::get().launch(kernel, {4}, {}, args, arg_sizes);
    litetorch::CLBackend::get().finish();

    std::cout << "Reading results back from GPU...\n";
    float h_out[4] = {0};
    litetorch::CLBackend::get().read(out_mem, 4 * sizeof(float), h_out);

    std::cout << "Results: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << "\n";

    for (int i = 0; i < 4; ++i) {
        assert(std::abs(h_out[i] - (h_data1[i] + h_data2[i])) < 1e-5);
    }

    std::cout << "LiteTorch Multi-Backend Detection Test Passed!\n";
    return 0;
}
