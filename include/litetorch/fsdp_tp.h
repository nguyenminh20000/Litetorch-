#ifndef LITETORCH_FSDP_TP_H
#define LITETORCH_FSDP_TP_H

#include "litetorch/nn.h"
#include "litetorch/tensor.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace litetorch {
namespace distributed {

struct TPShardInfo {
    int shard_dim = -1;
    std::vector<int64_t> full_shape;
    bool is_tp_sharded = false;
};

class FSDPTP {
public:
    static void init(int tp_size, int dp_size);
    static bool is_initialized() { return initialized_; }
    static int tp_size() { return tp_size_; }
    static int dp_size() { return dp_size_; }
    static int tp_rank();
    static int dp_rank();

    static void fully_shard_tp(std::shared_ptr<nn::Module> module);

    static std::map<std::string, std::shared_ptr<Tensor>> state_dict(
        std::shared_ptr<nn::Module> module);

    static void load_state_dict(
        std::shared_ptr<nn::Module> module,
        const std::map<std::string, std::shared_ptr<Tensor>>& sd);

    static void register_tp_shard(std::shared_ptr<Tensor> param, int shard_dim,
                                  const std::vector<int64_t>& full_shape);
    static TPShardInfo get_tp_shard_info(std::shared_ptr<Tensor> param);

private:
    static bool initialized_;
    static int tp_size_;
    static int dp_size_;
    static std::map<const Tensor*, TPShardInfo> tp_shard_registry_;
    static std::mutex registry_mutex_;
};

}
}

#endif
