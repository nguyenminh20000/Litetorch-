#ifndef LITETORCH_FLAT_PARAM_H
#define LITETORCH_FLAT_PARAM_H

#include <cstdint>
#include <memory>
#include <vector>
#include "litetorch/tensor.h"

namespace litetorch {
namespace distributed {

struct FlatParamMeta {
    int64_t offset = 0;
    int64_t numel = 0;
    std::vector<int64_t> shape;
};

class FlatParameter {
public:
    explicit FlatParameter(const std::vector<std::shared_ptr<Tensor>>& params);

    int64_t flat_numel() const { return flat_numel_; }
    int64_t shard_numel() const { return shard_numel_; }
    int64_t unpadded_numel() const { return unpadded_numel_; }
    int world_size() const { return world_; }
    int rank() const { return rank_; }
    DataType dtype() const { return dtype_; }
    Device device() const { return device_; }

    std::shared_ptr<Tensor> local_shard() { return shard_; }
    std::shared_ptr<Tensor> full_buffer() { return full_; }
    std::vector<std::shared_ptr<Tensor>> param_views();

    void all_gather_flat();
    std::shared_ptr<Tensor> reduce_scatter_flat(std::shared_ptr<Tensor> full_grad);

    const std::vector<FlatParamMeta>& metadata() const { return metas_; }

private:
    std::vector<FlatParamMeta> metas_;
    std::shared_ptr<Tensor> shard_;
    std::shared_ptr<Tensor> full_;
    int64_t flat_numel_ = 0;
    int64_t shard_numel_ = 0;
    int64_t unpadded_numel_ = 0;
    int world_ = 1;
    int rank_ = 0;
    DataType dtype_ = DataType::FP32;
    Device device_ = Device(DeviceType::CPU, 0);
};

std::shared_ptr<FlatParameter> make_flat_param(const std::vector<std::shared_ptr<Tensor>>& params);

}
}

#endif
