#ifndef LITETORCH_INFERENCE_H
#define LITETORCH_INFERENCE_H

#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace litetorch {
namespace inference {

class CudaGraph {
public:
    CudaGraph();
    ~CudaGraph();
    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;
    CudaGraph(CudaGraph&& other) noexcept : graph_(other.graph_) { other.graph_ = nullptr; }
    CudaGraph& operator=(CudaGraph&& other) noexcept;

    void capture(std::function<void()> fn);
    void replay();
    bool captured() const { return graph_ != nullptr; }

private:
    void* graph_ = nullptr;
};

CudaGraph capture_cuda_graph(std::shared_ptr<nn::Module> model,
                             std::shared_ptr<Tensor> static_in);

struct FusionReport {
    int conv_bn_folded = 0;
    int layernorm_detected = 0;
    bool linear_bias_kernel_fused = false;
    std::vector<std::string> notes;
};

std::shared_ptr<nn::Conv2d> fold_conv_bn(std::shared_ptr<nn::Conv2d> conv,
                                         std::shared_ptr<nn::BatchNorm2d> bn);
FusionReport fuse_conv_bn(std::shared_ptr<nn::Sequential> seq);
FusionReport detect_layernorm_fusion(std::shared_ptr<nn::Module> root);
bool verify_linear_bias_fusion();

} // namespace inference
} // namespace litetorch

#endif
