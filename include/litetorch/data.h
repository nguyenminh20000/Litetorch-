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

namespace litetorch {
namespace data {

class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() = 0;
    virtual std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> get(size_t index) = 0;
};

class TensorDataset : public Dataset {
public:
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> y;

    TensorDataset(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y);
    size_t size() override;
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> get(size_t index) override;
};

class DataLoader {
public:
    std::shared_ptr<Dataset> dataset;
    size_t batch_size;
    bool shuffle;
    std::vector<size_t> indices;
    size_t current_index = 0;

    DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, bool shuffle = true, const Device& device = Device(DeviceType::CPU, 0), size_t prefetch_limit = 2);
    ~DataLoader();
    void reset();
    bool next(std::shared_ptr<Tensor>& batch_x, std::shared_ptr<Tensor>& batch_y);
    void save_state(const std::string& filepath);
    void load_state(const std::string& filepath);

private:
    struct PrefetchBatch {
        std::shared_ptr<Tensor> x;
        std::shared_ptr<Tensor> y;
    };
    std::thread worker_thread;
    std::mutex queue_mutex;
    std::condition_variable cv_cond;
    std::atomic<bool> worker_stop;
    std::queue<PrefetchBatch> prefetch_queue;
    size_t worker_index = 0;
    size_t epoch_id = 0;
    size_t generation_id = 0;
    size_t prefetch_limit = 2;
    Device device;
    std::shared_ptr<Tensor> epoch_x_gpu;
    std::shared_ptr<Tensor> epoch_y_gpu;

    void worker_loop();
};

}
}

#endif
