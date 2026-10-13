#ifndef LITETORCH_P2P_ASYNC_H
#define LITETORCH_P2P_ASYNC_H

#include <memory>

namespace litetorch {
class Tensor;
namespace distributed {

class Work {
public:
    virtual ~Work() = default;
    virtual void wait() = 0;
    virtual bool is_completed() = 0;
};

std::shared_ptr<Work> isend(std::shared_ptr<Tensor> tensor, int dst);
std::shared_ptr<Work> irecv(std::shared_ptr<Tensor> tensor, int src);

}
}

#endif
