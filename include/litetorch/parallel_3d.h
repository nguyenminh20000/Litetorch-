#ifndef LITETORCH_PARALLEL_3D_H
#define LITETORCH_PARALLEL_3D_H

#include <memory>
#include <vector>

namespace litetorch {
namespace distributed {

struct Parallel3DConfig {
    int tp_size = 1;
    int pp_size = 1;
    int dp_size = 1;
};

class Parallel3D {
public:
    static Parallel3D& get();

    void init(int rank, int world_size, int tp_size, int pp_size);
    void reset();
    bool is_initialized() const;

    int tp_size() const;
    int pp_size() const;
    int dp_size() const;
    int global_rank() const;
    int world_size() const;

    int tp_rank() const;
    int pp_rank() const;
    int dp_rank() const;

    std::vector<int> coords() const;
    std::vector<int> tp_group() const;
    std::vector<int> pp_group() const;
    std::vector<int> dp_group() const;

    void wire_nccl();

private:
    Parallel3D();
    bool initialized_ = false;
    int rank_ = 0;
    int world_ = 0;
    Parallel3DConfig cfg_;
};

void init_3d_parallel(int tp_size, int pp_size);
void shutdown_3d_parallel();

}
}

#endif
