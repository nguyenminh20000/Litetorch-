#include "litetorch/distributed.h"
#include "litetorch/ops.h"
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <cassert>

using namespace litetorch;

void run_rank(int rank, int world_size) {
    try {
        distributed::init_process_group(rank, world_size, "127.0.0.1", 9999);

        auto t_cpu = Tensor::from_vector({(float)rank, (float)(rank * 2)}, {2}, Device(DeviceType::CPU));
        distributed::all_reduce(t_cpu);
        std::vector<float> res_cpu = t_cpu->to_vector();
        assert(res_cpu[0] == 0.5f);
        assert(res_cpu[1] == 1.0f);

        auto t_bc = Tensor::from_vector({(float)(rank == 0 ? 42.0f : 0.0f)}, {1}, Device(DeviceType::CPU));
        distributed::broadcast(t_bc, 0);
        std::vector<float> res_bc = t_bc->to_vector();
        assert(res_bc[0] == 42.0f);

        if (CLBackend::get().is_available()) {
            auto t_gpu = Tensor::from_vector({(float)rank, (float)(rank * 10)}, {2}, Device(DeviceType::GPU));
            distributed::all_reduce(t_gpu);
            std::vector<float> res_gpu = t_gpu->to_vector();
            assert(res_gpu[0] == 0.5f);
            assert(res_gpu[1] == 5.0f);
        }

        if (rank == 0) {
            auto t_send = Tensor::from_vector({100.0f, 200.0f, 300.0f}, {3});
            distributed::send_tensor(t_send, 1);
        } else {
            auto t_recv = Tensor::create({1}, Device(DeviceType::CPU));
            distributed::recv_tensor(t_recv, 0);
            assert(t_recv->shape.size() == 1);
            assert(t_recv->shape[0] == 3);
            auto res = t_recv->to_vector();
            assert(res[0] == 100.0f);
            assert(res[1] == 200.0f);
            assert(res[2] == 300.0f);
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
            assert(res[0] == 20.0f);
        }

        distributed::shutdown();
        std::cout << "Rank " << rank << " passed all tests!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " failed: " << e.what() << std::endl;
        exit(1);
    }
    exit(0);
}

int main() {
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
