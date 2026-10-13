#include "litetorch/inference.h"
#include "litetorch/device.h"
#include <stdexcept>

extern "C" {
__attribute__((weak)) void* gpu_start_recording();
__attribute__((weak)) void* gpu_stop_recording(void* stream_capture);
__attribute__((weak)) void gpu_launch_graph(void* graph);
__attribute__((weak)) void gpu_free_graph(void* graph);
}

namespace litetorch {
namespace inference {

static void require_gpu_backend() {
    if (!gpu_start_recording) {
        throw std::runtime_error("[litetorch Error] CudaGraph: GPU backend not available (fail-loud)");
    }
}

CudaGraph::CudaGraph() = default;

CudaGraph& CudaGraph::operator=(CudaGraph&& other) noexcept {
    if (this != &other) {
        if (graph_ && gpu_free_graph) gpu_free_graph(graph_);
        graph_ = other.graph_;
        other.graph_ = nullptr;
    }
    return *this;
}

CudaGraph::~CudaGraph() {
    if (graph_ && gpu_free_graph) {
        gpu_free_graph(graph_);
        graph_ = nullptr;
    }
}

void CudaGraph::capture(std::function<void()> fn) {
    require_gpu_backend();
    if (graph_) {
        gpu_free_graph(graph_);
        graph_ = nullptr;
    }
    gpu_start_recording();
    try {
        fn();
    } catch (...) {
        gpu_stop_recording(nullptr);
        throw;
    }
    void* g = gpu_stop_recording(nullptr);
    if (!g) {
        throw std::runtime_error("[litetorch Error] CudaGraph::capture: stream capture failed (ops must run on GPU with static shapes; no CPU sync or allocation allowed during capture)");
    }
    graph_ = g;
}

void CudaGraph::replay() {
    require_gpu_backend();
    if (!graph_) {
        throw std::runtime_error("[litetorch Error] CudaGraph::replay: no captured graph");
    }
    gpu_launch_graph(graph_);
}

CudaGraph capture_cuda_graph(std::shared_ptr<nn::Module> model,
                             std::shared_ptr<Tensor> static_in) {
    if (!model || !static_in) {
        throw std::runtime_error("[litetorch Error] capture_cuda_graph: null model or input");
    }
    if (static_in->device.type != DeviceType::GPU) {
        throw std::runtime_error("[litetorch Error] capture_cuda_graph: input must be on GPU (fail-loud)");
    }
    CudaGraph g;
    g.capture([&]() { model->forward(static_in); });
    return g;
}

} // namespace inference
} // namespace litetorch
