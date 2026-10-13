#ifndef LITETORCH_DIST_CHECKPOINT_H
#define LITETORCH_DIST_CHECKPOINT_H

#include "litetorch/tensor.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <utility>
#include <cstdint>

namespace litetorch {
namespace dist_checkpoint {

struct TensorShardInfo {
    std::vector<int64_t> global_shape;
    std::string dtype_str;
    int shard_dim = -1;
    std::vector<std::pair<int64_t, int64_t>> shard_ranges;
};

struct CheckpointMetadata {
    int version = 1;
    int world_size = 1;
    int64_t step = 0;
    int64_t timestamp = 0;
    std::map<std::string, TensorShardInfo> tensors;
};

std::string shard_filename(const std::string& ckpt_dir, int rank, int world_size);
std::string metadata_path(const std::string& ckpt_dir);

void save_distributed(
    const std::map<std::string, std::shared_ptr<Tensor>>& local_shards,
    const std::map<std::string, TensorShardInfo>& shard_specs,
    const std::string& ckpt_dir,
    int64_t step);

void load_distributed(
    std::map<std::string, std::shared_ptr<Tensor>>& local_shards,
    const std::string& ckpt_dir);

CheckpointMetadata read_metadata(const std::string& ckpt_dir);
void write_metadata(const std::string& ckpt_dir, const CheckpointMetadata& meta);

TensorShardInfo make_sharded_spec(
    const std::vector<int64_t>& global_shape,
    const std::string& dtype_str,
    int shard_dim,
    int world_size);

TensorShardInfo make_replicated_spec(
    const std::vector<int64_t>& global_shape,
    const std::string& dtype_str);

}
}

#endif
