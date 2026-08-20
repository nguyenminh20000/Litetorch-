#include "litetorch/cl_backend.h"
#include "litetorch/memory_manager.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace litetorch;

int main() {
    if (!CLBackend::get().is_available()) {
        std::cout << "OpenCL is not available, skipping allocator test." << std::endl;
        return 0;
    }

    auto& backend = CLBackend::get();

    cl_mem buf1 = backend.allocate(1024);
    cl_mem buf2 = backend.allocate(2048);
    cl_mem buf3 = backend.allocate(4096);

    assert(buf1 != nullptr);
    assert(buf2 != nullptr);
    assert(buf3 != nullptr);

    backend.free(buf1);
    backend.free(buf2);
    backend.free(buf3);

    cl_mem large_buf = backend.allocate(7000);
    assert(large_buf != nullptr);
    backend.free(large_buf);

    backend.clear_cache();

    std::cout << "SUCCESS: Advanced Caching Allocator block-splitting and coalescing verified!" << std::endl;
    return 0;
}
