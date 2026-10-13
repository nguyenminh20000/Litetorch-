#include "litetorch/pipeline.h"
#include "litetorch/distributed.h"
#include "litetorch/p2p_async.h"

namespace litetorch {
namespace distributed {

PipelineParallelModule::PipelineParallelModule(std::shared_ptr<nn::Module> sub_module)
    : local_module(std::move(sub_module)) {
    if (!local_module)
        throw std::runtime_error("[litetorch Error] PipelineParallelModule: null sub_module");
}

int PipelineParallelModule::rank_() const {
    auto& pg = ProcessGroup::get();
    return pg.is_initialized() ? pg.get_rank() : 0;
}

int PipelineParallelModule::world_size_() const {
    auto& pg = ProcessGroup::get();
    return pg.is_initialized() ? pg.get_world_size() : 1;
}

bool PipelineParallelModule::active_() const {
    return world_size_() > 1;
}

Device PipelineParallelModule::stage_device_(std::shared_ptr<Tensor> ref) const {
    if (ref) return ref->device;
    return Device(DeviceType::CPU);
}

std::shared_ptr<Tensor> PipelineParallelModule::recv_activation_(std::shared_ptr<Tensor> ref) {
    auto x = Tensor::create({1}, stage_device_(ref));
    x->requires_grad = true;
    auto w = irecv(x, rank_() - 1);
    w->wait();
    return x;
}

void PipelineParallelModule::validate_microbatches_(
    const std::vector<std::shared_ptr<Tensor>>& mbs) const {
    for (size_t i = 0; i < mbs.size(); ++i) {
        if (!mbs[i])
            throw std::runtime_error("[litetorch Error] PipelineParallelModule: null microbatch at index " +
                                     std::to_string(i));
    }
}

void PipelineParallelModule::wait_sends_() {
    for (auto& w : pending_sends_) {
        if (w) w->wait();
    }
    pending_sends_.clear();
}

void PipelineParallelModule::sync() {
    wait_sends_();
}

std::shared_ptr<Tensor> PipelineParallelModule::forward(std::shared_ptr<Tensor> input) {
    int rank = rank_();
    int world_size = world_size_();

    std::shared_ptr<Tensor> x = input;
    if (rank > 0 && active_()) {
        x = recv_activation_(input);
    }

    std::shared_ptr<Tensor> out = local_module->forward(x);

    if (rank < world_size - 1 && active_()) {
        pending_sends_.push_back(isend(out, rank + 1));
    }
    wait_sends_();
    return out;
}

std::vector<std::shared_ptr<Tensor>> PipelineParallelModule::forward_microbatches(
    const std::vector<std::shared_ptr<Tensor>>& microbatches) {
    return schedule_1f1b(microbatches);
}

std::vector<std::shared_ptr<Tensor>> PipelineParallelModule::schedule_1f1b(
    const std::vector<std::shared_ptr<Tensor>>& microbatches) {
    int rank = rank_();
    int world_size = world_size_();
    int n = static_cast<int>(microbatches.size());
    std::vector<std::shared_ptr<Tensor>> outputs(n);
    if (n == 0) return outputs;
    validate_microbatches_(microbatches);

    if (!active_()) {
        for (int i = 0; i < n; ++i)
            outputs[i] = local_module->forward(microbatches[i]);
        return outputs;
    }

    for (int i = 0; i < n; ++i) {
        std::shared_ptr<Tensor> x = microbatches[i];
        std::shared_ptr<Work> w_recv;
        if (rank > 0) {
            x = Tensor::create({1}, stage_device_(microbatches[i]));
            w_recv = irecv(x, rank - 1);
        }
        if (w_recv) w_recv->wait();
        auto out = local_module->forward(x);
        if (rank < world_size - 1)
            pending_sends_.push_back(isend(out, rank + 1));
        outputs[i] = out;
    }
    wait_sends_();
    return outputs;
}

std::vector<std::shared_ptr<Tensor>> PipelineParallelModule::train_step_1f1b(
    const std::vector<std::shared_ptr<Tensor>>& microbatches,
    std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>, int)> loss_fn) {
    int rank = rank_();
    int world_size = world_size_();
    int n = static_cast<int>(microbatches.size());
    std::vector<std::shared_ptr<Tensor>> losses;
    float grad_scale = n > 0 ? 1.0f / n : 1.0f;

    if (n == 0) return losses;
    validate_microbatches_(microbatches);

    if (!active_()) {
        for (int i = 0; i < n; ++i) {
            auto out = local_module->forward(microbatches[i]);
            if (loss_fn) {
                auto loss = loss_fn(out, i);
                auto scale = Tensor::from_vector({grad_scale}, {}, loss->device, false);
                loss->backward(scale);
                losses.push_back(loss);
            }
        }
        return losses;
    }

    int num_warmup = std::min(n, world_size - rank - 1);
    int num_1f1b = n - num_warmup;
    bool is_last = (rank == world_size - 1);

    struct Frame {
        std::shared_ptr<Tensor> x;
        std::shared_ptr<Tensor> out;
        int mb_idx;
    };
    std::queue<Frame> fifo;
    int fwd_idx = 0;

    auto do_forward = [&](int mb_idx) {
        std::shared_ptr<Tensor> x = microbatches[mb_idx];
        std::shared_ptr<Work> w_recv;
        if (rank > 0) {
            x = Tensor::create({1}, stage_device_(microbatches[mb_idx]));
            x->requires_grad = true;
            w_recv = irecv(x, rank - 1);
        }
        if (w_recv) w_recv->wait();
        auto out = local_module->forward(x);
        if (rank < world_size - 1)
            pending_sends_.push_back(isend(out, rank + 1));
        fifo.push({x, out, mb_idx});
    };

    auto do_backward = [&](std::shared_ptr<Work> w_grad,
                           std::shared_ptr<Tensor> g_buf) {
        Frame fr = fifo.front();
        fifo.pop();
        if (is_last) {
            if (loss_fn) {
                auto loss = loss_fn(fr.out, static_cast<int>(losses.size()));
                losses.push_back(loss);
                auto scale = Tensor::from_vector({grad_scale}, {}, loss->device, false);
                loss->backward(scale);
            } else {
                auto scale = Tensor::from_vector({grad_scale}, {}, fr.out->device, false);
                fr.out->backward(scale);
            }
        } else {
            if (w_grad) w_grad->wait();
            fr.out->backward(g_buf);
        }
        if (rank > 0 && fr.x && fr.x->grad)
            pending_sends_.push_back(isend(fr.x->grad, rank - 1));
    };

    for (int i = 0; i < num_warmup; ++i)
        do_forward(fwd_idx++);

    for (int i = 0; i < num_1f1b; ++i) {
        std::shared_ptr<Work> w_grad;
        std::shared_ptr<Tensor> g_buf;
        if (!is_last && !fifo.empty()) {
            const auto& fr = fifo.front();
            g_buf = Tensor::create(fr.out->shape, fr.out->device);
            w_grad = irecv(g_buf, rank + 1);
        }
        do_forward(fwd_idx++);
        do_backward(w_grad, g_buf);
    }

    while (!fifo.empty()) {
        std::shared_ptr<Work> w_grad;
        std::shared_ptr<Tensor> g_buf;
        if (!is_last) {
            const auto& fr = fifo.front();
            g_buf = Tensor::create(fr.out->shape, fr.out->device);
            w_grad = irecv(g_buf, rank + 1);
            w_grad->wait();
        }
        do_backward(nullptr, g_buf);
    }

    wait_sends_();
    return losses;
}

std::vector<std::shared_ptr<Tensor>> PipelineParallelModule::run_1f1b_with_backward(
    const std::vector<std::shared_ptr<Tensor>>& microbatches,
    std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>, int)> loss_fn) {
    return train_step_1f1b(microbatches, loss_fn);
}

std::vector<std::shared_ptr<Tensor>> PipelineParallelModule::parameters() {
    return local_module->parameters();
}

void PipelineParallelModule::to(const Device& device) {
    local_module->to(device);
}

}
}
