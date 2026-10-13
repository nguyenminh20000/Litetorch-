#ifndef LITETORCH_DEVICE_MESH_H
#define LITETORCH_DEVICE_MESH_H

#include <stdexcept>
#include <string>
#include <vector>

namespace litetorch {
namespace distributed {

class DeviceMesh {
public:
    std::vector<int> mesh_shape;
    std::string mesh_dim_names;
    std::vector<int> mesh_topology;

    DeviceMesh(const std::vector<int>& shape, const std::string& dim_names)
        : mesh_shape(shape), mesh_dim_names(dim_names) {
        if (mesh_shape.empty())
            throw std::runtime_error("[litetorch Error] DeviceMesh: shape must not be empty");
        for (int d : mesh_shape)
            if (d <= 0)
                throw std::runtime_error("[litetorch Error] DeviceMesh: every dim must be > 0");
    }

    int num_dims() const { return (int)mesh_shape.size(); }

    int numel() const {
        int n = 1;
        for (int d : mesh_shape) n *= d;
        return n;
    }

    int dim_size(int dim) const {
        if (dim < 0 || dim >= num_dims())
            throw std::runtime_error("[litetorch Error] DeviceMesh: dim out of range");
        return mesh_shape[dim];
    }

    int get_rank(const std::vector<int>& coords) const {
        if ((int)coords.size() != num_dims())
            throw std::runtime_error("[litetorch Error] DeviceMesh: coords size mismatch");
        int rank = 0;
        for (int i = 0; i < num_dims(); ++i) {
            if (coords[i] < 0 || coords[i] >= mesh_shape[i])
                throw std::runtime_error("[litetorch Error] DeviceMesh: coord out of range");
            rank = rank * mesh_shape[i] + coords[i];
        }
        return rank;
    }

    std::vector<int> get_coords(int rank) const {
        if (rank < 0 || rank >= numel())
            throw std::runtime_error("[litetorch Error] DeviceMesh: rank out of range");
        std::vector<int> coords(num_dims());
        for (int i = num_dims() - 1; i >= 0; --i) {
            coords[i] = rank % mesh_shape[i];
            rank /= mesh_shape[i];
        }
        return coords;
    }

    std::vector<int> get_dim_group(int dim, const std::vector<int>& coords) const {
        if (dim < 0 || dim >= num_dims())
            throw std::runtime_error("[litetorch Error] DeviceMesh: dim out of range");
        if ((int)coords.size() != num_dims())
            throw std::runtime_error("[litetorch Error] DeviceMesh: coords size mismatch");
        std::vector<int> group;
        group.reserve(mesh_shape[dim]);
        std::vector<int> c = coords;
        for (int i = 0; i < mesh_shape[dim]; ++i) {
            c[dim] = i;
            group.push_back(get_rank(c));
        }
        return group;
    }
};

}
}

#endif
