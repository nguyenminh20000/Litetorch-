#ifndef LITETORCH_OPTIM_UTILS_H
#define LITETORCH_OPTIM_UTILS_H

#include "litetorch/tensor.h"
#include <vector>
#include <memory>

namespace litetorch {
extern const std::string litetorch_kernels_src;

namespace optim {
namespace {
struct StorageUseGuard {
    std::vector<std::shared_ptr<StorageImpl>> storages;
    StorageUseGuard(const std::vector<std::shared_ptr<StorageImpl>>& list) : storages(list) {
        for (auto& s : storages) {
            if (s) s->in_use = true;
        }
    }
    ~StorageUseGuard() {
        for (auto& s : storages) {
            if (s) s->in_use = false;
        }
    }
};
}
}
}
#endif
