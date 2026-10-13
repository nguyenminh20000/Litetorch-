#ifndef LITETORCH_TRAINER_H
#define LITETORCH_TRAINER_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace litetorch {

class Tensor;

namespace nn {
class Module;
}

namespace optim {
class Optimizer;
}

namespace trainer {

struct TrainerConfig {
    std::string ckpt_dir = "checkpoints";
    int64_t save_every_steps = 1000;
    int64_t max_checkpoints = 5;
    bool async_save = true;
    bool elastic = false;
};

class Trainer {
public:
    Trainer(std::shared_ptr<nn::Module> model,
            std::shared_ptr<optim::Optimizer> optimizer,
            TrainerConfig config = TrainerConfig());
    ~Trainer();

    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    void train_step();
    void save_checkpoint(int64_t step);
    int64_t load_latest();
    void garbage_collect();
    int64_t latest_step() const;
    int rank() const;
    int world_size() const;
    bool is_distributed() const;
    void wait_pending_saves();

private:
    std::map<std::string, std::shared_ptr<Tensor>> param_map() const;
    std::vector<int64_t> list_steps() const;
    void ensure_dir(const std::string& path) const;

    std::shared_ptr<nn::Module> model_;
    std::shared_ptr<optim::Optimizer> optimizer_;
    TrainerConfig config_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}

#endif
