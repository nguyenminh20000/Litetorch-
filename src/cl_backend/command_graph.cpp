#include "litetorch/cl_backend.h"
#include "cl_functions.h"
#include <stdexcept>

namespace litetorch {

OpenCLCommandGraph::~OpenCLCommandGraph() {
    stop_recording();
}

void OpenCLCommandGraph::start_recording() {
    commands.clear();
    recording = true;
    CLBackend::get().active_graph = std::shared_ptr<OpenCLCommandGraph>(this, [](OpenCLCommandGraph*){});
}

void OpenCLCommandGraph::stop_recording() {
    recording = false;
    CLBackend::get().active_graph = nullptr;
}

void OpenCLCommandGraph::replay() {
    auto queue = CLBackend::get().get_queue();
    if (!queue) return;

    for (const auto& cmd : commands) {
        auto k_mutex = CLBackend::get().get_kernel_mutex(cmd.kernel);
        std::unique_lock<std::mutex> k_lock;
        if (k_mutex) {
            k_lock = std::unique_lock<std::mutex>(*k_mutex);
        }

        for (cl_uint i = 0; i < cmd.arg_data.size(); ++i) {
            cl_int err = p_clSetKernelArg(cmd.kernel, i, cmd.arg_sizes[i], cmd.arg_data[i].data());
            if (err != CL_SUCCESS) {
                throw std::runtime_error("[litetorch Error] clSetKernelArg failed in graph replay for argument " + std::to_string(i));
            }
        }

        const size_t* local_ws_ptr = cmd.local_work_size.empty() ? nullptr : cmd.local_work_size.data();
        cl_int err = p_clEnqueueNDRangeKernel(queue, cmd.kernel, cmd.global_work_size.size(), nullptr, cmd.global_work_size.data(), local_ws_ptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("[litetorch Error] clEnqueueNDRangeKernel failed in graph replay: " + std::to_string(err));
        }
    }
}

}
