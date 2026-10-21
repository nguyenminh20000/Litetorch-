#include "litetorch/pipeline.h"
#include "litetorch/distributed.h"
#include "litetorch/parallel_3d.h"
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

int PipelineParallelModule::pp_size_() const {
    if (Parallel3D::get().is_initialized()) return Parallel3D::get().pp_size();
    return world_size_();
}

int PipelineParallelModule::pp_rank_() const {
    if (Parallel3D::get().is_initialized()) return Parallel3D::get().pp_rank();
    return rank_();
}

int PipelineParallelModule::pp_prev_() const {
    int pr = pp_rank_();
    if (pr <= 0) return -1;
    if (Parallel3D::get().is_initialized()) return Parallel3D::get().pp_group()[pr - 1];
    return rank_() - 1;
}

int PipelineParallelModule::pp_next_() const {
    int pr = pp_rank_();
    if (pr >= pp_size_() - 1) return -1;
    if (Parallel3D::get().is_initialized()) return Parallel3D::get().pp_group()[pr + 1];
    return rank_() + 1;
}

bool PipelineParallelModule::pp_first_() const {
    return pp_rank_() == 0;
}

bool PipelineParallelModule::pp_last_() const {
    return pp_rank_() == pp_size_() - 1;
}

Device PipelineParallelModule::stage_device_(std::shared_ptr<Tensor> ref) const {
    if (ref) return ref->device;
    return Device(DeviceType::CPU);
}

std::shared_ptr<Tensor> PipelineParallelModule::recv_activation_(std::shared_ptr<Tensor> ref) {
    auto x = Tensor::create({1}, stage_device_(ref));
    x->requires_grad = true;
    auto w = irecv(x, pp_prev_());
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
    std::shared_ptr<Tensor> x = input;
    if (!pp_first_() && active_()) {
        x = recv_activation_(input);
    }

    std::shared_ptr<Tensor> out = local_module->forward(x);

    if (!pp_last_() && active_()) {
        pending_sends_.push_back(isend(out, pp_next_()));
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
        if (!pp_first_()) {
            x = Tensor::create({1}, stage_device_(microbatches[i]));
            w_recv = irecv(x, pp_prev_());
        }
        if (w_recv) w_recv->wait();
        auto out = local_module->forward(x);
        if (!pp_last_())
            pending_sends_.push_back(isend(out, pp_next_()));
        outputs[i] = out;
    }
    wait_sends_();
    return outputs;
}

std::vector<std::shared_ptr<Tensor>> PipelineParallelModule::train_step_1f1b(
    const std::vector<std::shared_ptr<Tensor>>& microbatches,
    std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>, int)> loss_fn) {
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

    int num_warmup = std::min(n, pp_size_() - pp_rank_() - 1);
    int num_1f1b = n - num_warmup;
    bool is_last = pp_last_();

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
        if (!pp_first_()) {
            x = Tensor::create({1}, stage_device_(microbatches[mb_idx]));
            x->requires_grad = true;
            w_recv = irecv(x, pp_prev_());
        }
        if (w_recv) w_recv->wait();
        auto out = local_module->forward(x);
        if (!pp_last_())
            pending_sends_.push_back(isend(out, pp_next_()));
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
        if (!pp_first_() && fr.x && fr.x->grad)
            pending_sends_.push_back(isend(fr.x->grad, pp_prev_()));
    };

    for (int i = 0; i < num_warmup; ++i)
        do_forward(fwd_idx++);

    for (int i = 0; i < num_1f1b; ++i) {
        std::shared_ptr<Work> w_grad;
        std::shared_ptr<Tensor> g_buf;
        if (!is_last && !fifo.empty()) {
            const auto& fr = fifo.front();
            g_buf = Tensor::create(fr.out->shape, fr.out->device);
            w_grad = irecv(g_buf, pp_next_());
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
            w_grad = irecv(g_buf, pp_next_());
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

InterleavedPipelineModule::InterleavedPipelineModule(
    std::vector<std::shared_ptr<nn::Module>> chunks)
    : chunks_(std::move(chunks)) {
    validate_chunks_();
}

void InterleavedPipelineModule::validate_chunks_() const {
    if (chunks_.empty())
        throw std::runtime_error("[litetorch Error] InterleavedPipelineModule: empty chunks");
    for (size_t i = 0; i < chunks_.size(); ++i) {
        if (!chunks_[i])
            throw std::runtime_error("[litetorch Error] InterleavedPipelineModule: null chunk at index " +
                                     std::to_string(i));
    }
}

int InterleavedPipelineModule::rank_() const {
    auto& pg = ProcessGroup::get();
    return pg.is_initialized() ? pg.get_rank() : 0;
}

int InterleavedPipelineModule::world_size_() const {
    auto& pg = ProcessGroup::get();
    return pg.is_initialized() ? pg.get_world_size() : 1;
}

bool InterleavedPipelineModule::active_() const {
    return world_size_() > 1;
}

int InterleavedPipelineModule::pp_size_() const {
    if (Parallel3D::get().is_initialized()) return Parallel3D::get().pp_size();
    return world_size_();
}

int InterleavedPipelineModule::pp_rank_() const {
    if (Parallel3D::get().is_initialized()) return Parallel3D::get().pp_rank();
    return rank_();
}

int InterleavedPipelineModule::pp_stage_global_(int stage) const {
    if (Parallel3D::get().is_initialized()) return Parallel3D::get().pp_group()[stage];
    return stage;
}

Device InterleavedPipelineModule::stage_device_(std::shared_ptr<Tensor> ref) const {
    if (ref) return ref->device;
    return Device(DeviceType::CPU);
}

void InterleavedPipelineModule::validate_microbatches_(
    const std::vector<std::shared_ptr<Tensor>>& mbs) const {
    for (size_t i = 0; i < mbs.size(); ++i) {
        if (!mbs[i])
            throw std::runtime_error("[litetorch Error] InterleavedPipelineModule: null microbatch at index " +
                                     std::to_string(i));
    }
}

void InterleavedPipelineModule::wait_sends_() {
    for (auto& w : pending_sends_) {
        if (w) w->wait();
    }
    pending_sends_.clear();
}

void InterleavedPipelineModule::sync() {
    wait_sends_();
}

int InterleavedPipelineModule::num_chunks() const {
    return static_cast<int>(chunks_.size());
}

std::shared_ptr<Tensor> InterleavedPipelineModule::forward(std::shared_ptr<Tensor> input) {
    auto outs = schedule_interleaved_1f1b({input});
    return outs.empty() ? nullptr : outs[0];
}

std::vector<std::shared_ptr<Tensor>> InterleavedPipelineModule::schedule_interleaved_1f1b(
    const std::vector<std::shared_ptr<Tensor>>& microbatches) {
    int my_stage = pp_rank_();
    int p = pp_size_();
    int v = static_cast<int>(chunks_.size());
    int V = p * v;
    int m = static_cast<int>(microbatches.size());
    std::vector<std::shared_ptr<Tensor>> outputs(m);
    if (m == 0) return outputs;
    validate_microbatches_(microbatches);

    if (!active_()) {
        for (int i = 0; i < m; ++i) {
            auto x = microbatches[i];
            for (int j = 0; j < v; ++j)
                x = chunks_[j]->forward(x);
            outputs[i] = x;
        }
        return outputs;
    }

    auto sync_send = [&](std::shared_ptr<Tensor> t, int dst) {
        auto w = isend(t, dst);
        w->wait();
    };
    auto sync_recv = [&](std::shared_ptr<Tensor> ref, int src) {
        auto x = Tensor::create({1}, stage_device_(ref));
        auto w = irecv(x, src);
        w->wait();
        return x;
    };

    for (int i = 0; i < m; ++i) {
        for (int vs = 0; vs < V; ++vs) {
            int r = vs % p;
            if (my_stage != r) continue;
            int j = (vs - r) / p;
            std::shared_ptr<Tensor> x;
            if (vs == 0) {
                x = microbatches[i];
            } else {
                x = sync_recv(microbatches[i], pp_stage_global_((vs - 1 + p) % p));
            }
            auto out = chunks_[j]->forward(x);
            if (vs < V - 1) {
                sync_send(out, pp_stage_global_((vs + 1) % p));
            } else {
                outputs[i] = out;
            }
        }
    }
    return outputs;
}

std::vector<std::shared_ptr<Tensor>> InterleavedPipelineModule::train_step_interleaved_1f1b(
    const std::vector<std::shared_ptr<Tensor>>& microbatches,
    std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>, int)> loss_fn) {
    int my_stage = pp_rank_();
    int p = pp_size_();
    int v = static_cast<int>(chunks_.size());
    int V = p * v;
    int m = static_cast<int>(microbatches.size());
    std::vector<std::shared_ptr<Tensor>> losses;
    if (m == 0) return losses;
    validate_microbatches_(microbatches);

    float grad_scale = 1.0f / m;

    if (!active_()) {
        for (int i = 0; i < m; ++i) {
            auto x = microbatches[i];
            for (int j = 0; j < v; ++j)
                x = chunks_[j]->forward(x);
            if (loss_fn) {
                auto loss = loss_fn(x, i);
                losses.push_back(loss);
                auto scale = Tensor::from_vector({grad_scale}, {}, loss->device, false);
                loss->backward(scale);
            }
        }
        return losses;
    }

    struct Frame {
        std::shared_ptr<Tensor> x;
        std::shared_ptr<Tensor> out;
        int mb;
    };
    std::vector<std::vector<Frame>> frames(v);

    auto sync_send = [&](std::shared_ptr<Tensor> t, int dst) {
        auto w = isend(t, dst);
        w->wait();
    };

    for (int i = 0; i < m; ++i) {
        for (int vs = 0; vs < V; ++vs) {
            int r = vs % p;
            if (my_stage != r) continue;
            int j = (vs - r) / p;
            std::shared_ptr<Tensor> x;
            if (vs == 0) {
                x = microbatches[i];
            } else {
                int src = pp_stage_global_((vs - 1 + p) % p);
                x = Tensor::create({1}, stage_device_(microbatches[i]));
                x->requires_grad = true;
                auto w = irecv(x, src);
                w->wait();
            }
            auto out = chunks_[j]->forward(x);
            if (vs < V - 1) {
                sync_send(out, pp_stage_global_((vs + 1) % p));
            }
            frames[j].push_back({x, out, i});
        }
    }

    std::vector<std::vector<std::shared_ptr<Tensor>>> mb_losses(m);
    for (int i = m - 1; i >= 0; --i) {
        for (int vs = V - 1; vs >= 0; --vs) {
            int r = vs % p;
            if (my_stage != r) continue;
            int j = (vs - r) / p;
            Frame fr = frames[j][i];
            if (vs == V - 1) {
                if (loss_fn) {
                    auto loss = loss_fn(fr.out, i);
                    mb_losses[i].push_back(loss);
                    auto scale = Tensor::from_vector({grad_scale}, {}, loss->device, false);
                    loss->backward(scale);
                } else {
                    auto scale = Tensor::from_vector({grad_scale}, {}, fr.out->device, false);
                    fr.out->backward(scale);
                }
            } else {
                int src = pp_stage_global_((vs + 1) % p);
                auto g_buf = Tensor::create(fr.out->shape, fr.out->device);
                auto w = irecv(g_buf, src);
                w->wait();
                fr.out->backward(g_buf);
            }
            if (vs > 0 && fr.x && fr.x->grad) {
                sync_send(fr.x->grad, pp_stage_global_((vs - 1 + p) % p));
            }
        }
    }

    for (int i = 0; i < m; ++i) {
        for (auto& l : mb_losses[i])
            losses.push_back(l);
    }
    return losses;
}

std::vector<std::shared_ptr<Tensor>> InterleavedPipelineModule::parameters() {
    std::vector<std::shared_ptr<Tensor>> ps;
    for (auto& c : chunks_) {
        auto cp = c->parameters();
        ps.insert(ps.end(), cp.begin(), cp.end());
    }
    return ps;
}

void InterleavedPipelineModule::to(const Device& device) {
    for (auto& c : chunks_)
        c->to(device);
}

}
}
