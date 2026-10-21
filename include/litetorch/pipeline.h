#ifndef LITETORCH_PIPELINE_H
#define LITETORCH_PIPELINE_H

#include <functional>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "litetorch/nn.h"
#include "litetorch/tensor.h"

namespace litetorch {
namespace distributed {

class Work;
class ProcessGroup;

class PipelineParallelModule : public nn::Module {
public:
    explicit PipelineParallelModule(std::shared_ptr<nn::Module> sub_module);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> forward_microbatches(
        const std::vector<std::shared_ptr<Tensor>>& microbatches);
    std::vector<std::shared_ptr<Tensor>> schedule_1f1b(
        const std::vector<std::shared_ptr<Tensor>>& microbatches);
    std::vector<std::shared_ptr<Tensor>> train_step_1f1b(
        const std::vector<std::shared_ptr<Tensor>>& microbatches,
        std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>, int)> loss_fn = nullptr);
    std::vector<std::shared_ptr<Tensor>> run_1f1b_with_backward(
        const std::vector<std::shared_ptr<Tensor>>& microbatches,
        std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>, int)> loss_fn);

    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;

    void sync();

    std::shared_ptr<nn::Module> local_module;

private:
    int rank_() const;
    int world_size_() const;
    bool active_() const;
    int pp_rank_() const;
    int pp_size_() const;
    int pp_prev_() const;
    int pp_next_() const;
    bool pp_first_() const;
    bool pp_last_() const;
    Device stage_device_(std::shared_ptr<Tensor> ref) const;
    std::shared_ptr<Tensor> recv_activation_(std::shared_ptr<Tensor> ref);
    void validate_microbatches_(const std::vector<std::shared_ptr<Tensor>>& mbs) const;

    std::vector<std::shared_ptr<Work>> pending_sends_;
    void wait_sends_();
};

class InterleavedPipelineModule : public nn::Module {
public:
    explicit InterleavedPipelineModule(std::vector<std::shared_ptr<nn::Module>> chunks);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> schedule_interleaved_1f1b(
        const std::vector<std::shared_ptr<Tensor>>& microbatches);
    std::vector<std::shared_ptr<Tensor>> train_step_interleaved_1f1b(
        const std::vector<std::shared_ptr<Tensor>>& microbatches,
        std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>, int)> loss_fn = nullptr);

    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;

    void sync();
    int num_chunks() const;

    std::vector<std::shared_ptr<nn::Module>> chunks_;

private:
    int rank_() const;
    int world_size_() const;
    bool active_() const;
    int pp_rank_() const;
    int pp_size_() const;
    int pp_stage_global_(int stage) const;
    Device stage_device_(std::shared_ptr<Tensor> ref) const;
    void validate_microbatches_(const std::vector<std::shared_ptr<Tensor>>& mbs) const;
    void validate_chunks_() const;

    std::vector<std::shared_ptr<Work>> pending_sends_;
    void wait_sends_();
};

}
}

#endif
