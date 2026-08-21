#ifndef LITETORCH_DEVICE_MESH_H
#define LITETORCH_DEVICE_MESH_H

#include <vector>
#include <string>

namespace litetorch {
namespace distributed {

class DeviceMesh {
public:
    std::vector<int> mesh_shape;
    std::string mesh_dim_names;
    std::vector<int> mesh_topology;

    DeviceMesh(const std::vector<int>& shape, const std::string& dim_names)
        : mesh_shape(shape), mesh_dim_names(dim_names) {}
        
    int get_rank(const std::vector<int>& coords) const {
        return 0;
    }
};

}
}

#endif
