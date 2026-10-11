#ifndef LITETORCH_AMP_H
#define LITETORCH_AMP_H

#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/optim.h"
#include "litetorch/grad_scaler.h"
#include <cmath>
#include <vector>
#include <memory>
#include <stdexcept>

namespace litetorch {
namespace amp {

class GradScaler : public optim::GradScaler {
public:
    using optim::GradScaler::GradScaler;
};

class AutocastGuard {
public:
    AutocastGuard(bool enabled = true, DataType dtype = DataType::FP16) {
        prev_enabled_ = enabled_;
        prev_dtype_ = dtype_;
        enabled_ = enabled;
        dtype_ = dtype;
    }
    ~AutocastGuard() {
        enabled_ = prev_enabled_;
        dtype_ = prev_dtype_;
    }
    static bool is_enabled() { return enabled_; }
    static DataType get_dtype() { return dtype_; }

private:
    static thread_local bool enabled_;
    static thread_local DataType dtype_;
    bool prev_enabled_;
    DataType prev_dtype_;
};

}
}

#endif
