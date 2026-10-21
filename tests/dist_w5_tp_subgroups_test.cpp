#include "litetorch/distributed.h"
#include "litetorch/tp_collectives.h"
#include "litetorch/fsdp_tp.h"
#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <vector>
#include <cmath>
#include <string>

using namespace litetorch;

static int failures = 0;

void check_vec(const std::vector<float>& got, const std::vector<float>& want,
               const std::string& name, int rank, float tol = 1e-4f) {
    if (got.size() != want.size()) {
        std::cerr << "[rank " << rank << "] FAIL " << name << ": size " << got.size()
                  << " != " << want.size() << std::endl;
        ++failures;
        return;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - want[i]) > tol) {
            std::cerr << "[rank " << rank << "] FAIL " << name << ": idx " << i << " got "
                      << got[i] << " want " << want[i] << std::endl;
            ++failures;
            return;
        }
    }
    std::cout << "[rank " << rank << "] ok " << name << std::endl;
}

void check_shape(std::shared_ptr<Tensor> t, const std::vector<int64_t>& want,
                 const std::string& name, int rank) {
    if (t->shape != want) {
        std::cerr << "[rank " << rank << "] FAIL " << name << ": bad shape" << std::endl;
        ++failures;
        return;
    }
    std::cout << "[rank " << rank << "] ok " << name << " shape" << std::endl;
}

void phase_pure_tp(int rank, int world) {
    distributed::init_process_group(rank, world, "127.0.0.1", 29991);

    auto t = Tensor::from_vector({(float)(rank + 1), (float)(rank + 1) * 10.0f}, {2},
                                 Device(DeviceType::CPU));
    distributed::tp_allreduce_inplace(t);
    check_vec(t->to_vector(), {3.0f, 30.0f}, "pure_tp_allreduce_sum", rank);

    auto shard = Tensor::from_vector({(float)rank}, {1}, Device(DeviceType::CPU));
    auto full = Tensor::create({2}, Device(DeviceType::CPU));
    distributed::tp_allgather_into(shard, full);
    check_vec(full->to_vector(), {0.0f, 1.0f}, "pure_tp_allgather", rank);

    auto gshard = Tensor::from_vector({(float)rank * 10.0f, (float)rank * 10.0f + 1.0f}, {2},
                                      Device(DeviceType::CPU));
    auto y = distributed::tp_all_gather_autograd(gshard);
    check_shape(y, {4}, "pure_tp_allgather_autograd_fwd", rank);
    check_vec(y->to_vector(), {0.0f, 1.0f, 10.0f, 11.0f}, "pure_tp_allgather_autograd_fwd", rank);

    auto x = Tensor::from_vector({5.0f}, {1}, Device(DeviceType::CPU), true);
    auto yg = distributed::tp_all_gather_autograd(x);
    auto c = Tensor::from_vector({1.0f, 100.0f}, {2}, Device(DeviceType::CPU));
    auto loss = Ops::sum(Ops::mul(yg, c));
    loss->backward();
    std::vector<float> want_grad = (rank == 0) ? std::vector<float>{2.0f}
                                               : std::vector<float>{200.0f};
    if (!x->grad) { std::cerr << "[rank " << rank << "] FAIL allgather_bwd: null grad" << std::endl; ++failures; }
    else check_vec(x->grad->to_vector(), want_grad, "pure_tp_allgather_autograd_bwd", rank);

    auto xa = Tensor::from_vector({7.0f}, {1}, Device(DeviceType::CPU), true);
    auto ya = distributed::tp_all_reduce_autograd(xa);
    auto lossa = Ops::sum(ya);
    lossa->backward();
    if (!xa->grad) { std::cerr << "[rank " << rank << "] FAIL allreduce_bwd: null grad" << std::endl; ++failures; }
    else check_vec(xa->grad->to_vector(), {1.0f}, "pure_tp_allreduce_autograd_bwd_identity", rank);

    distributed::shutdown();
}

void phase_hybrid(int rank, int world) {
    distributed::init_process_group(rank, world, "127.0.0.1", 29992);
    distributed::FSDPTP::init(2, 2);

    auto t = Tensor::from_vector({(float)rank}, {1}, Device(DeviceType::CPU));
    distributed::tp_allreduce_inplace(t);
    float want_sum = (rank < 2) ? 1.0f : 5.0f;
    check_vec(t->to_vector(), {want_sum}, "hybrid_tp_allreduce_sum", rank);

    auto t2 = Tensor::from_vector({(float)rank}, {1}, Device(DeviceType::CPU));
    distributed::tp_allreduce_inplace(t2);
    check_vec(t2->to_vector(), {want_sum}, "hybrid_tp_allreduce_sum_repeat", rank);

    auto shard = Tensor::from_vector({(float)(rank * 10)}, {1}, Device(DeviceType::CPU));
    auto full = Tensor::create({2}, Device(DeviceType::CPU));
    distributed::tp_allgather_into(shard, full);
    std::vector<float> want_full = (rank < 2) ? std::vector<float>{0.0f, 10.0f}
                                              : std::vector<float>{20.0f, 30.0f};
    check_vec(full->to_vector(), want_full, "hybrid_tp_allgather", rank);

    auto gshard = Tensor::from_vector({(float)rank, (float)rank + 100.0f}, {2},
                                      Device(DeviceType::CPU));
    auto y = distributed::tp_all_gather_autograd(gshard);
    check_shape(y, {4}, "hybrid_allgather_autograd_fwd", rank);
    std::vector<float> want_y = (rank < 2) ? std::vector<float>{0.0f, 100.0f, 1.0f, 101.0f}
                                           : std::vector<float>{2.0f, 102.0f, 3.0f, 103.0f};
    check_vec(y->to_vector(), want_y, "hybrid_allgather_autograd_fwd", rank);

    auto x = Tensor::from_vector({5.0f}, {1}, Device(DeviceType::CPU), true);
    auto yg = distributed::tp_all_gather_autograd(x);
    auto c = Tensor::from_vector({1.0f, 100.0f}, {2}, Device(DeviceType::CPU));
    auto loss = Ops::sum(Ops::mul(yg, c));
    loss->backward();
    int pos = rank % 2;
    std::vector<float> want_grad = (pos == 0) ? std::vector<float>{2.0f}
                                             : std::vector<float>{200.0f};
    if (!x->grad) { std::cerr << "[rank " << rank << "] FAIL hybrid_allgather_bwd: null grad" << std::endl; ++failures; }
    else check_vec(x->grad->to_vector(), want_grad, "hybrid_allgather_autograd_bwd", rank);

    auto w = Tensor::from_vector({(float)rank}, {1}, Device(DeviceType::CPU));
    distributed::all_reduce(w);
    check_vec(w->to_vector(), {6.0f}, "hybrid_world_allreduce_untouched", rank);

    distributed::shutdown();
}

void phase_singleton(int rank, int world) {
    distributed::init_process_group(rank, world, "127.0.0.1", 29993);
    distributed::FSDPTP::init(1, 4);

    auto t = Tensor::from_vector({(float)(rank + 1)}, {1}, Device(DeviceType::CPU));
    distributed::tp_allreduce_inplace(t);
    check_vec(t->to_vector(), {(float)(rank + 1)}, "singleton_tp_allreduce_noop", rank);

    auto shard = Tensor::from_vector({(float)rank * 3.0f}, {1}, Device(DeviceType::CPU));
    auto full = Tensor::create({1}, Device(DeviceType::CPU));
    distributed::tp_allgather_into(shard, full);
    check_vec(full->to_vector(), {(float)rank * 3.0f}, "singleton_tp_allgather_copy", rank);

    distributed::shutdown();
}

void phase_asymmetric(int rank, int world) {
    distributed::init_process_group(rank, world, "127.0.0.1", 29994);
    distributed::FSDPTP::init(2, 3);

    auto t = Tensor::from_vector({(float)rank}, {1}, Device(DeviceType::CPU));
    distributed::tp_allreduce_inplace(t);
    float want_sum = (rank < 2) ? 1.0f : (rank < 4) ? 5.0f : 9.0f;
    check_vec(t->to_vector(), {want_sum}, "asym_tp_allreduce_sum", rank);

    auto shard = Tensor::from_vector({(float)(rank * 10)}, {1}, Device(DeviceType::CPU));
    auto full = Tensor::create({2}, Device(DeviceType::CPU));
    distributed::tp_allgather_into(shard, full);
    std::vector<float> want_full = (rank < 2) ? std::vector<float>{0.0f, 10.0f}
                                 : (rank < 4) ? std::vector<float>{20.0f, 30.0f}
                                              : std::vector<float>{40.0f, 50.0f};
    check_vec(full->to_vector(), want_full, "asym_tp_allgather", rank);

    distributed::shutdown();
}

void phase_dtypes(int rank, int world) {
    distributed::init_process_group(rank, world, "127.0.0.1", 29995);
    distributed::FSDPTP::init(2, 2);

    auto ti = Tensor::from_vector({(float)rank, (float)(rank + 100)}, {2}, Device(DeviceType::CPU));
    auto t32 = ti->cast(DataType::INT32);
    distributed::tp_allreduce_inplace(t32);
    auto back32 = t32->cast(DataType::FP32);
    std::vector<float> want32 = (rank < 2) ? std::vector<float>{1.0f, 201.0f}
                                           : std::vector<float>{5.0f, 205.0f};
    check_vec(back32->to_vector(), want32, "hybrid_tp_allreduce_int32", rank);

    auto th = Tensor::from_vector({(float)rank * 0.5f, 1.25f}, {2}, Device(DeviceType::CPU));
    auto t16 = th->cast(DataType::FP16);
    distributed::tp_allreduce_inplace(t16);
    auto back16 = t16->cast(DataType::FP32);
    std::vector<float> want16 = (rank < 2) ? std::vector<float>{0.5f, 2.5f}
                                          : std::vector<float>{2.5f, 2.5f};
    check_vec(back16->to_vector(), want16, "hybrid_tp_allreduce_fp16", rank, 1e-2f);

    distributed::shutdown();
}

void phase_socket_hybrid(int rank, int world) {
    distributed::init_process_group(rank, world, "127.0.0.2", 29996);
    distributed::FSDPTP::init(2, 2);

    auto t = Tensor::from_vector({(float)rank}, {1}, Device(DeviceType::CPU));
    distributed::tp_allreduce_inplace(t);
    float want_sum = (rank < 2) ? 1.0f : 5.0f;
    check_vec(t->to_vector(), {want_sum}, "socket_hybrid_tp_allreduce_sum", rank);

    auto shard = Tensor::from_vector({(float)(rank * 10)}, {1}, Device(DeviceType::CPU));
    auto full = Tensor::create({2}, Device(DeviceType::CPU));
    distributed::tp_allgather_into(shard, full);
    std::vector<float> want_full = (rank < 2) ? std::vector<float>{0.0f, 10.0f}
                                              : std::vector<float>{20.0f, 30.0f};
    check_vec(full->to_vector(), want_full, "socket_hybrid_tp_allgather", rank);

    distributed::shutdown();
}

void run_phase(int phase, int rank, int world) {
    alarm(60);
    try {
        if (phase == 0) phase_pure_tp(rank, world);
        else if (phase == 1) phase_hybrid(rank, world);
        else if (phase == 2) phase_singleton(rank, world);
        else if (phase == 3) phase_asymmetric(rank, world);
        else if (phase == 4) phase_dtypes(rank, world);
        else phase_socket_hybrid(rank, world);
    } catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] EXCEPTION phase " << phase << ": " << e.what() << std::endl;
        exit(1);
    }
    if (failures > 0) {
        std::cerr << "[rank " << rank << "] phase " << phase << " FAILURES=" << failures << std::endl;
        exit(1);
    }
    std::cout << "[rank " << rank << "] phase " << phase << " PASS" << std::endl;
    exit(0);
}

int main() {
    const int phases[6] = {0, 1, 2, 3, 4, 5};
    const int worlds[6] = {2, 4, 4, 6, 4, 4};
    for (int pi = 0; pi < 6; ++pi) {
        int world = worlds[pi];
        std::vector<pid_t> kids;
        for (int r = 0; r < world; ++r) {
            pid_t p = fork();
            if (p < 0) { std::cerr << "fork failed" << std::endl; return 1; }
            if (p == 0) run_phase(phases[pi], r, world);
            kids.push_back(p);
        }
        for (auto k : kids) {
            int st = 0;
            waitpid(k, &st, 0);
            if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
                std::cerr << "PHASE " << phases[pi] << " FAILED (child)" << std::endl;
                return 1;
            }
        }
        std::cout << "PHASE " << phases[pi] << " (world=" << world << ") ALL RANKS DONE" << std::endl;
    }
    std::cout << "ALL PHASES DONE" << std::endl;
    return 0;
}
