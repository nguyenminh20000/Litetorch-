#ifndef LITETORCH_STORAGE_GUARD_H
#define LITETORCH_STORAGE_GUARD_H

#include <memory>
#include <vector>
#include "litetorch/tensor.h"

namespace litetorch {

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

#endif
