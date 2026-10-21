#ifndef LITETORCH_JIT_DETAIL_H
#define LITETORCH_JIT_DETAIL_H

#include "litetorch/jit.h"

namespace litetorch {
namespace jit_detail {

inline bool fastpath_binary_ok(const std::shared_ptr<JITVar>& expr,
                               const std::vector<std::shared_ptr<JITVar>>& inputs) {
    return expr && inputs.size() == 2 && expr->left && expr->right
        && expr->left->op == JITVar::OpType::INPUT
        && expr->right->op == JITVar::OpType::INPUT
        && inputs[0].get() == expr->left.get()
        && inputs[1].get() == expr->right.get();
}

inline bool fastpath_unary_ok(const std::shared_ptr<JITVar>& expr,
                              const std::vector<std::shared_ptr<JITVar>>& inputs) {
    return expr && inputs.size() == 1 && expr->left
        && expr->left->op == JITVar::OpType::INPUT
        && inputs[0].get() == expr->left.get();
}

}
}

#endif
