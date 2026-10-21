#include "litetorch/distributed.h"
#include "litetorch/ops.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <sys/wait.h>

using namespace litetorch;

static void check_eq(float actual, float expected, const char* what) {
    if (actual != expected) {
        throw std::runtime_error(std::string("dist-w3 check failed: ") + what +
            " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
    }
}

static void check_true(bool cond, const char* what) {
    if (!cond) {
        throw std::runtime_error(std::string("dist-w3 check failed: ") + what);
    }
}

void run_rank(int rank, int world_size) {
    try {
        const char* master_env = getenv("LT_DIST_W3_MASTER");
        distributed::init_process_group(rank, world_size, master_env ? master_env : "127.0.0.1", 9999);

        auto t_cpu = Tensor::from_vector({(float)rank, (float)(rank * 2)}, {2}, Device(DeviceType::CPU));
        distributed::all_reduce(t_cpu);
        std::vector<float> res_cpu = t_cpu->to_vector();
        check_eq(res_cpu[0], 1.0f, "all_reduce default SUM [0]");
        check_eq(res_cpu[1], 2.0f, "all_reduce default SUM [1]");

        auto t_bc = Tensor::from_vector({(float)(rank == 0 ? 42.0f : 0.0f)}, {1}, Device(DeviceType::CPU));
        distributed::broadcast(t_bc, 0);
        std::vector<float> res_bc = t_bc->to_vector();
        check_eq(res_bc[0], 42.0f, "broadcast [0]");

        if (CLBackend::get().is_available()) {
            auto t_gpu = Tensor::from_vector({(float)rank, (float)(rank * 10)}, {2}, Device(DeviceType::GPU));
            distributed::all_reduce(t_gpu);
            std::vector<float> res_gpu = t_gpu->to_vector();
            check_eq(res_gpu[0], 1.0f, "all_reduce GPU default SUM [0]");
            check_eq(res_gpu[1], 10.0f, "all_reduce GPU default SUM [1]");
        }

        {
            float v0 = rank == 0 ? 1.0f : 3.0f;
            float v1 = rank == 0 ? 2.0f : 4.0f;
            auto check_op = [&](distributed::ReduceOp op, const char* name, float e0, float e1) {
                auto t = Tensor::from_vector({v0, v1}, {2}, Device(DeviceType::CPU));
                distributed::all_reduce(t, op);
                std::vector<float> v = t->to_vector();
                check_eq(v[0], e0, (std::string("all_reduce(") + name + ")[0]").c_str());
                check_eq(v[1], e1, (std::string("all_reduce(") + name + ")[1]").c_str());
            };
            check_op(distributed::ReduceOp::SUM, "SUM", 4.0f, 6.0f);
            check_op(distributed::ReduceOp::AVG, "AVG", 2.0f, 3.0f);
            check_op(distributed::ReduceOp::PRODUCT, "PRODUCT", 3.0f, 8.0f);
            check_op(distributed::ReduceOp::MIN, "MIN", 1.0f, 2.0f);
            check_op(distributed::ReduceOp::MAX, "MAX", 3.0f, 4.0f);
        }

        {
            struct GradProbe : public nn::Module {
                std::shared_ptr<Tensor> w;
                GradProbe() {
                    w = Tensor::create({2}, Device(DeviceType::CPU));
                    register_parameter("w", w);
                }
                std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
                    return w;
                }
            };
            auto probe = std::make_shared<GradProbe>();
            float g0 = rank == 0 ? 1.0f : 3.0f;
            float g1 = rank == 0 ? 2.0f : 4.0f;
            probe->w->grad = Tensor::from_vector({g0, g1}, {2}, Device(DeviceType::CPU));
            distributed::DistributedDataParallel ddp(probe);
            check_true(ddp.reduce_op == distributed::ReduceOp::AVG, "DDP default reduce op is AVG");
            check_true(ddp.reduce_op == distributed::DistributedDataParallel::kDefaultDDPReduceOp,
                       "DDP reduce_op == kDefaultDDPReduceOp");
            ddp.sync_grads();
            std::vector<float> gv = probe->w->grad->to_vector();
            check_eq(gv[0], 2.0f, "DDP sync_grads averaged grad [0]");
            check_eq(gv[1], 3.0f, "DDP sync_grads averaged grad [1]");
        }

        if (rank == 0) {
            auto t_send = Tensor::from_vector({100.0f, 200.0f, 300.0f}, {3});
            distributed::send_tensor(t_send, 1);
        } else {
            auto t_recv = Tensor::create({1}, Device(DeviceType::CPU));
            distributed::recv_tensor(t_recv, 0);
            check_true(t_recv->shape.size() == 1, "recv shape rank");
            check_true(t_recv->shape[0] == 3, "recv shape dim0");
            auto res = t_recv->to_vector();
            check_eq(res[0], 100.0f, "recv [0]");
            check_eq(res[1], 200.0f, "recv [1]");
            check_eq(res[2], 300.0f, "recv [2]");
        }

        struct TimesTwoModule : public nn::Module {
            std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override {
                return Ops::mul(input, Tensor::from_vector({2.0f}, {1}, input->device));
            }
        };

        auto local_sub = std::make_shared<TimesTwoModule>();
        auto pp_module = std::make_shared<distributed::PipelineParallelModule>(local_sub);

        if (rank == 0) {
            auto input_tensor = Tensor::from_vector({5.0f}, {1});
            pp_module->forward(input_tensor);
        } else {
            auto output_tensor = pp_module->forward(nullptr);
            auto res = output_tensor->to_vector();
            check_eq(res[0], 20.0f, "pipeline forward [0]");
        }

        distributed::shutdown();
        if (rank == 0) {
            std::cout << "dist-w3: reduce-op semantics pinned: all_reduce default=SUM; "
                      << "SUM/AVG/PRODUCT/MIN/MAX explicit; DDP default=AVG (gradient average)" << std::endl;
        }
        std::cout << "Rank " << rank << " passed all tests!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " failed: " << e.what() << std::endl;
        exit(1);
    }
    exit(0);
}

int mismatch_rank_main(int rank) {
    const char* master_env = getenv("LT_DIST_W9_MASTER");
    distributed::init_process_group(rank, 2, master_env ? master_env : "127.0.0.1", 19999);
    bool threw = false;
    std::string msg;
    try {
        if (rank == 0) {
            auto t = Tensor::from_vector({1.0f, 2.0f}, {2}, Device(DeviceType::CPU));
            distributed::all_reduce(t);
        } else {
            auto t = Tensor::create({2}, Device(DeviceType::CPU), false, DataType::INT64);
            distributed::all_reduce(t);
        }
    } catch (const std::exception& e) {
        threw = true;
        msg = e.what();
    }
    distributed::shutdown();
    if (!threw) {
        std::cerr << "dist-w9 mismatch test: rank " << rank << " did NOT throw on FP32/INT64 dtype mismatch" << std::endl;
        return 2;
    }
    if (msg.find("mismatch") == std::string::npos) {
        std::cerr << "dist-w9 mismatch test: rank " << rank << " threw without 'mismatch': " << msg << std::endl;
        return 3;
    }
    std::cout << "dist-w9 mismatch test: rank " << rank << " threw as expected: " << msg << std::endl;
    return 0;
}

int run_mismatch_test() {
    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "dist-w9 mismatch test: fork failed" << std::endl;
        return 1;
    }
    if (pid == 0) {
        _exit(mismatch_rank_main(1));
    }
    int rc0 = mismatch_rank_main(0);
    int status = 0;
    waitpid(pid, &status, 0);
    int rc1 = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (rc0 == 0 && rc1 == 0) {
        std::cout << "dist-w9 mismatch test: SUCCESS — both ranks threw 'mismatch' on FP32{2} vs INT64{2} all_reduce" << std::endl;
        return 0;
    }
    std::cerr << "dist-w9 mismatch test: FAILED rc0=" << rc0 << " rc1=" << rc1 << std::endl;
    return 1;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "mismatch") {
        return run_mismatch_test();
    }
    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "Fork failed" << std::endl;
        return 1;
    }

    if (pid == 0) {
        run_rank(1, 2);
    } else {
        run_rank(0, 2);
        
        int status = 0;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            std::cout << "SUCCESS: Distributed multi-process all-reduce & broadcast verified!" << std::endl;
            return 0;
        } else {
            std::cerr << "Child process failed!" << std::endl;
            return 1;
        }
    }
}
