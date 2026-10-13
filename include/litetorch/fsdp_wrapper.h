#ifndef LITETORCH_FSDP_WRAPPER_H
#define LITETORCH_FSDP_WRAPPER_H

#include "litetorch/nn.h"
#include "litetorch/flat_param.h"
#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

namespace litetorch {
namespace distributed {

class FSDPUnit : public std::enable_shared_from_this<FSDPUnit> {
public:
    std::shared_ptr<nn::Module> module;
    std::shared_ptr<FlatParameter> flat;
    std::vector<std::shared_ptr<Tensor>> params;

    explicit FSDPUnit(std::shared_ptr<nn::Module> mod);

    void prefetch_gather();
    void ensure_gathered();
    void release_gathered();
    bool is_gathered() const { return gathered_.load(); }
    void copy_flat_to_params();
    void on_param_grad(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad);

    static void set_prefetch_limit(int n);
    static int prefetch_limit();
    static int inflight_count() { return inflight_gathers_.load(); }
    static std::vector<std::shared_ptr<FSDPUnit>>& forward_stack();

private:
    std::future<void> gather_future_;
    std::atomic<bool> gather_issued_{false};
    std::atomic<bool> gathered_{false};
    std::mutex gather_mutex_;

    std::mutex grad_mutex_;
    std::vector<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> pending_grads_;

    void reduce_scatter_grads();

    static std::atomic<int> prefetch_limit_;
    static std::atomic<int> inflight_gathers_;
};

class FSDP {
public:
    static void fully_shard(std::shared_ptr<nn::Module> module);
    static void set_prefetch_limit(int n);
    static std::vector<std::shared_ptr<FSDPUnit>> active_units();
};

}
}

#endif
