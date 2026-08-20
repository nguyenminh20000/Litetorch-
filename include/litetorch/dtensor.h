#ifndef LITETORCH_DTENSOR_H
#define LITETORCH_DTENSOR_H

#include "litetorch/tensor.h"
#include "litetorch/device_mesh.h"
#include <memory>
#include <vector>

namespace litetorch {
namespace distributed {

enum class PlacementType {
    SHARD,
    REPLICATE,
    PARTIAL
};

struct Placement {
    PlacementType type;
    int dim = -1; 
};

class DTensor : public Tensor {
public:
    std::shared_ptr<DeviceMesh> device_mesh;
    std::vector<Placement> placements;
    std::shared_ptr<Tensor> local_tensor;

    DTensor(std::shared_ptr<Tensor> local_t, std::shared_ptr<DeviceMesh> mesh, const std::vector<Placement>& p)
        : Tensor(local_t->shape, local_t->device, local_t->requires_grad, local_t->dtype),
          device_mesh(mesh), placements(p), local_tensor(local_t) {}
    
    std::shared_ptr<DTensor> redistribute(std::shared_ptr<DeviceMesh> new_mesh, const std::vector<Placement>& new_placements) {
        return nullptr;
    }
};

}
}

#endif
