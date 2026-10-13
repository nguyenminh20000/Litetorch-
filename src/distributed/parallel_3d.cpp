#include "litetorch/parallel_3d.h"

#include "litetorch/distributed.h"

#include <stdexcept>

namespace litetorch {
namespace distributed {

Parallel3D::Parallel3D() {}

Parallel3D& Parallel3D::get() {
    static Parallel3D inst;
    return inst;
}

void Parallel3D::init(int rank, int world_size, int tp_size, int pp_size) {
    if (tp_size <= 0 || pp_size <= 0)
        throw std::runtime_error("[litetorch Error] init_3d_parallel: tp_size and pp_size must be > 0");
    if (world_size <= 0 || rank < 0 || rank >= world_size)
        throw std::runtime_error("[litetorch Error] init_3d_parallel: bad rank/world_size");
    if (world_size % (tp_size * pp_size) != 0)
        throw std::runtime_error("[litetorch Error] init_3d_parallel: world_size must be divisible by tp_size*pp_size");
    rank_ = rank;
    world_ = world_size;
    cfg_.tp_size = tp_size;
    cfg_.pp_size = pp_size;
    cfg_.dp_size = world_size / (tp_size * pp_size);
    initialized_ = true;
}

void Parallel3D::reset() {
    initialized_ = false;
    rank_ = 0;
    world_ = 0;
    cfg_ = Parallel3DConfig();
}

bool Parallel3D::is_initialized() const { return initialized_; }

static void require_init(bool ok) {
    if (!ok)
        throw std::runtime_error("[litetorch Error] Parallel3D: not initialized, call init_3d_parallel first");
}

int Parallel3D::tp_size() const { require_init(initialized_); return cfg_.tp_size; }
int Parallel3D::pp_size() const { require_init(initialized_); return cfg_.pp_size; }
int Parallel3D::dp_size() const { require_init(initialized_); return cfg_.dp_size; }
int Parallel3D::global_rank() const { require_init(initialized_); return rank_; }
int Parallel3D::world_size() const { require_init(initialized_); return world_; }

int Parallel3D::tp_rank() const {
    require_init(initialized_);
    return rank_ % cfg_.tp_size;
}

int Parallel3D::pp_rank() const {
    require_init(initialized_);
    return (rank_ / cfg_.tp_size) % cfg_.pp_size;
}

int Parallel3D::dp_rank() const {
    require_init(initialized_);
    return rank_ / (cfg_.tp_size * cfg_.pp_size);
}

std::vector<int> Parallel3D::coords() const {
    require_init(initialized_);
    return {dp_rank(), pp_rank(), tp_rank()};
}

std::vector<int> Parallel3D::tp_group() const {
    require_init(initialized_);
    std::vector<int> g;
    g.reserve(cfg_.tp_size);
    int base = rank_ - tp_rank();
    for (int i = 0; i < cfg_.tp_size; ++i) g.push_back(base + i);
    return g;
}

std::vector<int> Parallel3D::pp_group() const {
    require_init(initialized_);
    std::vector<int> g;
    g.reserve(cfg_.pp_size);
    for (int p = 0; p < cfg_.pp_size; ++p)
        g.push_back((dp_rank() * cfg_.pp_size + p) * cfg_.tp_size + tp_rank());
    return g;
}

std::vector<int> Parallel3D::dp_group() const {
    require_init(initialized_);
    std::vector<int> g;
    g.reserve(cfg_.dp_size);
    int cell = cfg_.tp_size * cfg_.pp_size;
    int pos = rank_ % cell;
    for (int d = 0; d < cfg_.dp_size; ++d) g.push_back(d * cell + pos);
    return g;
}

void Parallel3D::wire_nccl() {
    require_init(initialized_);
    if (!NCCLBridge::get().is_available())
        throw std::runtime_error("[litetorch Error] Parallel3D::wire_nccl: NCCL not available on this machine");
    NCCLBridge::get().init_groups(rank_, world_, cfg_.tp_size, cfg_.pp_size);
}

void init_3d_parallel(int tp_size, int pp_size) {
    auto& pg = ProcessGroup::get();
    if (!pg.is_initialized())
        throw std::runtime_error("[litetorch Error] init_3d_parallel: init_process_group first");
    Parallel3D::get().init(pg.get_rank(), pg.get_world_size(), tp_size, pp_size);
}

void shutdown_3d_parallel() {
    Parallel3D::get().reset();
}

}
}
