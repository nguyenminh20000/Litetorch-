#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/amp.h"

namespace litetorch {
namespace amp {
thread_local bool AutocastGuard::enabled_ = false;
thread_local DataType AutocastGuard::dtype_ = DataType::FP16;
}

namespace {

class CastNode : public Node {
public:
    DataType src_dtype;
    CastNode(DataType src_dtype) : Node("Cast"), src_dtype(src_dtype) {}
    
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        return { Ops::cast(grad_output, src_dtype) };
    }
};

}

namespace Ops {

std::shared_ptr<Tensor> cast(std::shared_ptr<Tensor> a, DataType target_dtype) {
    if (a->dtype == target_dtype) {
        return a;
    }
    
    auto out = a->cast(target_dtype);
    
    if (a->requires_grad) {
        auto node = std::make_shared<CastNode>(a->dtype);
        node->inputs = { {a, a->requires_grad} };
        node->next_nodes = { a->creator };
        node->saved_tensors = {};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

}
}
