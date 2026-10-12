#ifndef LITETORCH_DATA_H
#define LITETORCH_DATA_H

#include "litetorch/tensor.h"
#include <vector>
#include <memory>
#include <utility>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <map>
#include <functional>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <fstream>
#include <cstring>

namespace litetorch {
namespace data {

class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() = 0;
    virtual std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> get(size_t index) = 0;
    virtual bool is_python_backed() const { return false; }
};

class TensorDataset : public Dataset {
public:
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> y;

    TensorDataset(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y);
    size_t size() override;
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> get(size_t index) override;
};

class Subset : public Dataset {
public:
    Subset(std::shared_ptr<Dataset> dataset, const std::vector<size_t>& indices);
    size_t size() override;
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> get(size_t index) override;

private:
    std::shared_ptr<Dataset> dataset_;
    std::vector<size_t> indices_;
};

std::vector<std::shared_ptr<Subset>> random_split(std::shared_ptr<Dataset> dataset,
                                                  const std::vector<size_t>& lengths,
                                                  uint64_t seed = 0);

class Sampler {
public:
    virtual ~Sampler() = default;
    virtual std::vector<size_t> indices() = 0;
    virtual size_t size() const = 0;
};

class SequentialSampler : public Sampler {
public:
    explicit SequentialSampler(size_t n);
    std::vector<size_t> indices() override;
    size_t size() const override;

private:
    size_t n_;
};

class RandomSampler : public Sampler {
public:
    RandomSampler(size_t n, bool replacement = false, size_t num_samples = 0, uint64_t seed = 0);
    std::vector<size_t> indices() override;
    size_t size() const override;

private:
    size_t n_;
    bool replacement_;
    size_t num_samples_;
    uint64_t seed_;
    uint64_t epoch_ = 0;
};

class BatchSampler : public Sampler {
public:
    BatchSampler(std::shared_ptr<Sampler> sampler, size_t batch_size, bool drop_last = false);
    std::vector<std::vector<size_t>> batches();
    std::vector<size_t> indices() override;
    size_t size() const override;
    size_t batch_size() const { return batch_size_; }
    bool drop_last() const { return drop_last_; }

private:
    std::shared_ptr<Sampler> sampler_;
    size_t batch_size_;
    bool drop_last_;
};

using CollateFn = std::function<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(
    const std::vector<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>&)>;

class DataLoader {
public:
    DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, bool shuffle = true,
               const Device& device = Device(DeviceType::CPU, 0), size_t prefetch_limit = 2,
               size_t num_workers = 0, bool drop_last = false,
               std::shared_ptr<Sampler> sampler = nullptr);
    ~DataLoader();
    void reset();
    bool next(std::shared_ptr<Tensor>& batch_x, std::shared_ptr<Tensor>& batch_y);
    size_t num_batches() const { return num_batches_; }
    void set_collate_fn(CollateFn fn, bool from_python);
    void save_state(const std::string& filepath);
    void load_state(const std::string& filepath);

    std::shared_ptr<Dataset> dataset;
    size_t batch_size;
    bool shuffle;

private:
    struct PrefetchBatch {
        std::shared_ptr<Tensor> x;
        std::shared_ptr<Tensor> y;
    };

    void worker_loop();
    void build_batch(size_t batch_no, size_t gen);
    void build_batch_inner(size_t batch_no, size_t gen, const std::vector<size_t>& batch_indices,
                           const CollateFn& collate);
    void build_epoch_cache();
    void refresh_indices();
    size_t compute_num_batches() const;
    void spawn_workers();
    void stop_workers();

    std::vector<std::thread> workers_;
    size_t num_workers_ = 0;
    bool drop_last_ = false;
    std::shared_ptr<Sampler> sampler_;
    CollateFn collate_fn_;
    bool collate_from_python_ = false;

    std::mutex queue_mutex;
    std::condition_variable cv_cond;
    std::atomic<bool> worker_stop{false};
    std::atomic<size_t> next_job_{0};
    std::atomic<size_t> in_flight_{0};
    std::exception_ptr worker_error_;
    std::map<size_t, PrefetchBatch> out_buffer_;
    size_t out_head_ = 0;
    size_t num_batches_ = 0;
    size_t generation_id = 0;

    std::vector<size_t> indices;
    size_t current_index = 0;
    size_t epoch_id = 0;
    size_t prefetch_limit = 2;
    Device device;
    std::shared_ptr<Tensor> epoch_x_gpu;
    std::shared_ptr<Tensor> epoch_y_gpu;
};

}
}

#endif
