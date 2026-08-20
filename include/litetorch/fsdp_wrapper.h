#ifndef LITETORCH_FSDP_WRAPPER_H
#define LITETORCH_FSDP_WRAPPER_H

#include "litetorch/nn.h"
#include <memory>

namespace litetorch {
namespace distributed {

class FSDP {
public:
    static void fully_shard(std::shared_ptr<nn::Module> module);
};

}
}

#endif
