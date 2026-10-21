#include "litetorch/data.h"
#include "litetorch/ops.h"
#include <pybind11/pybind11.h>
#include <random>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace py = pybind11;

namespace litetorch {

namespace {
struct GilHold {
    PyGILState_STATE state;
    bool held = false;
    explicit GilHold(bool want) {
        if (want) {
            state = PyGILState_Ensure();
            held = true;
        }
    }
    ~GilHold() {
        if (held) PyGILState_Release(state);
    }
    GilHold(const GilHold&) = delete;
    GilHold& operator=(const GilHold&) = delete;
};
uint32_t calculate_crc32(const void* data, size_t size, uint32_t initial_crc = 0xFFFFFFFF) {
    static uint32_t table[256];
    static bool table_initialized = false;
    if (!table_initialized) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t crc = i;
            for (uint32_t j = 0; j < 8; ++j) {
                if (crc & 1) {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
            table[i] = crc;
        }
        table_initialized = true;
    }

    uint32_t crc = initial_crc;
    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        crc = (crc >> 8) ^ table[(crc ^ byte_ptr[i]) & 0xFF];
    }
    return crc;
}
}

namespace data {

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, bool shuffle,
                       const Device& device, size_t prefetch_limit, size_t num_workers,
                       bool drop_last, std::shared_ptr<Sampler> sampler)
    : dataset(dataset), batch_size(batch_size), shuffle(shuffle),
      num_workers_(num_workers), drop_last_(drop_last), sampler_(sampler),
      worker_stop(false), next_job_(0), in_flight_(0),
      out_head_(0), num_batches_(0), generation_id(0),
      current_index(0), epoch_id(0), prefetch_limit(prefetch_limit ? prefetch_limit : 1),
      device(device) {
    if (!dataset) {
        throw std::invalid_argument("[litetorch Error] DataLoader requires a non-null dataset");
    }
    if (batch_size == 0) {
        throw std::invalid_argument("[litetorch Error] DataLoader requires batch_size > 0");
    }
    if (sampler_ && shuffle) {
        throw std::invalid_argument("[litetorch Error] DataLoader: sampler and shuffle are mutually exclusive");
    }
    reset();
    spawn_workers();
}

DataLoader::~DataLoader() {
    stop_workers();
    worker_error_ = nullptr;
}

void DataLoader::spawn_workers() {
    size_t nthreads = num_workers_ == 0 ? 1 : num_workers_;
    for (size_t i = 0; i < nthreads; ++i) {
        workers_.emplace_back(&DataLoader::worker_loop, this);
    }
}

void DataLoader::stop_workers() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        worker_stop = true;
    }
    cv_cond.notify_all();
    bool release_gil = Py_IsInitialized() && PyGILState_Check();
    std::unique_ptr<py::gil_scoped_release> rel;
    if (release_gil) rel = std::make_unique<py::gil_scoped_release>();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
}

void DataLoader::refresh_indices() {
    if (sampler_) {
        indices = sampler_->indices();
    } else {
        indices.resize(dataset->size());
        for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
        if (shuffle) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
        }
    }
}

size_t DataLoader::compute_num_batches() const {
    size_t n = indices.size();
    if (n == 0) return 0;
    size_t full = n / batch_size;
    size_t rem = n % batch_size;
    if (rem == 0) return full;
    return drop_last_ ? full : full + 1;
}

void DataLoader::reset() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    generation_id++;
    {
        bool release_gil = Py_IsInitialized() && PyGILState_Check();
        std::unique_ptr<py::gil_scoped_release> rel;
        if (release_gil) rel = std::make_unique<py::gil_scoped_release>();
        cv_cond.wait(lock, [this]() {
            return worker_stop.load() || in_flight_ == 0;
        });
    }
    worker_error_ = nullptr;
    refresh_indices();
    num_batches_ = compute_num_batches();
    current_index = 0;
    out_head_ = 0;
    next_job_ = 0;
    in_flight_ = 0;
    out_buffer_.clear();
    epoch_id++;
    build_epoch_cache();
    cv_cond.notify_all();
}

bool DataLoader::next(std::shared_ptr<Tensor>& batch_x, std::shared_ptr<Tensor>& batch_y) {
    std::unique_lock<std::mutex> lock(queue_mutex);
    cv_cond.wait(lock, [this]() {
        return worker_stop.load() || worker_error_ || out_head_ >= num_batches_ ||
               out_buffer_.count(out_head_) > 0;
    });
    if (worker_error_) {
        std::rethrow_exception(worker_error_);
    }
    if (out_head_ >= num_batches_) {
        return false;
    }
    auto it = out_buffer_.find(out_head_);
    if (it == out_buffer_.end()) {
        return false;
    }
    batch_x = it->second.x;
    batch_y = it->second.y;
    out_buffer_.erase(it);
    out_head_++;
    current_index = std::min(out_head_ * batch_size, indices.size());
    lock.unlock();
    cv_cond.notify_all();
    return true;
}

void DataLoader::set_collate_fn(CollateFn fn, bool from_python) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    collate_fn_ = std::move(fn);
    collate_from_python_ = from_python;
    generation_id++;
    cv_cond.notify_all();
}

void DataLoader::worker_loop() {
    while (!worker_stop.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv_cond.wait(lock, [this]() {
            return worker_stop.load() ||
                   (next_job_ < num_batches_ &&
                    in_flight_ + out_buffer_.size() < prefetch_limit);
        });
        if (worker_stop.load()) break;
        size_t b = next_job_++;
        in_flight_++;
        size_t gen = generation_id;
        lock.unlock();

        try {
            build_batch(b, gen);
        } catch (...) {
            std::lock_guard<std::mutex> elock(queue_mutex);
            if (!worker_error_) worker_error_ = std::current_exception();
        }

        lock.lock();
        in_flight_--;
        cv_cond.notify_all();
    }
}

void DataLoader::build_batch(size_t batch_no, size_t gen) {
    CollateFn collate;
    bool from_python;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        collate = collate_fn_;
        from_python = collate_from_python_;
    }
    GilHold hold(collate || from_python || dataset->is_python_backed());

    size_t start = batch_no * batch_size;
    size_t end = std::min(start + batch_size, indices.size());
    std::vector<size_t> batch_indices(indices.begin() + start, indices.begin() + end);
    try {
        build_batch_inner(batch_no, gen, batch_indices, collate);
    } catch (py::error_already_set&) {
        throw std::runtime_error("Python exception raised in DataLoader worker");
    }
}

void DataLoader::build_batch_inner(size_t batch_no, size_t gen, const std::vector<size_t>& batch_indices,
                                   const CollateFn& collate) {
    PrefetchBatch batch;
    size_t batch_len = batch_indices.size();
    auto tensor_dataset = std::dynamic_pointer_cast<TensorDataset>(dataset);

    if (tensor_dataset && !collate) {
        auto x_tensor = tensor_dataset->x;
        auto y_tensor = tensor_dataset->y;

        int64_t D = 1;
        for (size_t d = 1; d < x_tensor->shape.size(); ++d) D *= x_tensor->shape[d];
        int64_t Dy = 1;
        for (size_t d = 1; d < y_tensor->shape.size(); ++d) Dy *= y_tensor->shape[d];

        std::vector<int64_t> batch_x_shape = x_tensor->shape;
        batch_x_shape[0] = (int64_t)batch_len;
        std::vector<int64_t> batch_y_shape = y_tensor->shape;
        batch_y_shape[0] = (int64_t)batch_len;

        if (device.type != DeviceType::CPU && epoch_x_gpu && epoch_y_gpu) {
            batch.x = std::make_shared<Tensor>(
                epoch_x_gpu->storage, batch_x_shape, default_strides(batch_x_shape),
                epoch_x_gpu->offset + (int64_t)(batch_no * batch_size) * D, device, false);
            batch.y = std::make_shared<Tensor>(
                epoch_y_gpu->storage, batch_y_shape, default_strides(batch_y_shape),
                epoch_y_gpu->offset + (int64_t)(batch_no * batch_size) * Dy, device, false);
        } else {
            auto batch_x_cpu = Tensor::create(batch_x_shape, Device(DeviceType::CPU, 0), false, x_tensor->dtype);
            auto batch_y_cpu = Tensor::create(batch_y_shape, Device(DeviceType::CPU, 0), false, y_tensor->dtype);
            size_t x_elem_sz = x_tensor->storage->element_size();
            size_t y_elem_sz = y_tensor->storage->element_size();
            char* bx_ptr = (char*)batch_x_cpu->raw_data_ptr();
            char* by_ptr = (char*)batch_y_cpu->raw_data_ptr();
            char* x_ptr = (char*)x_tensor->raw_data_ptr();
            char* y_ptr = (char*)y_tensor->raw_data_ptr();
            for (size_t i = 0; i < batch_len; ++i) {
                size_t idx = batch_indices[i];
                std::memcpy(bx_ptr + i * D * x_elem_sz, x_ptr + idx * D * x_elem_sz, D * x_elem_sz);
                std::memcpy(by_ptr + i * Dy * y_elem_sz, y_ptr + idx * Dy * y_elem_sz, Dy * y_elem_sz);
            }
            batch.x = batch_x_cpu;
            batch.y = batch_y_cpu;
            if (device.type != DeviceType::CPU) {
                batch.x = batch.x->to(device);
                batch.y = batch.y->to(device);
            }
        }
    } else {
        std::vector<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> items;
        items.reserve(batch_len);
        for (size_t idx : batch_indices) {
            items.push_back(dataset->get(idx));
        }
        if (collate) {
            auto pr = collate(items);
            batch.x = pr.first;
            batch.y = pr.second;
        } else {
            std::vector<std::shared_ptr<Tensor>> xs;
            std::vector<std::shared_ptr<Tensor>> ys;
            xs.reserve(batch_len);
            ys.reserve(batch_len);
            for (auto& it : items) {
                xs.push_back(Ops::unsqueeze(it.first, 0));
                ys.push_back(Ops::unsqueeze(it.second, 0));
            }
            batch.x = Ops::cat(xs, 0);
            batch.y = Ops::squeeze(Ops::cat(ys, 0), 1);
        }
        if (device.type != DeviceType::CPU) {
            batch.x = batch.x->to(device);
            batch.y = batch.y->to(device);
        }
    }

    std::lock_guard<std::mutex> lock(queue_mutex);
    if (gen == generation_id && !worker_stop.load()) {
        out_buffer_[batch_no] = std::move(batch);
    }
    cv_cond.notify_all();
}

void DataLoader::build_epoch_cache() {
    auto tensor_dataset = std::dynamic_pointer_cast<TensorDataset>(dataset);
    if (tensor_dataset && device.type != DeviceType::CPU && !collate_fn_) {
        size_t n = indices.size();
        auto x_tensor = tensor_dataset->x;
        auto y_tensor = tensor_dataset->y;

        int64_t D = 1;
        for (size_t d = 1; d < x_tensor->shape.size(); ++d) D *= x_tensor->shape[d];
        int64_t Dy = 1;
        for (size_t d = 1; d < y_tensor->shape.size(); ++d) Dy *= y_tensor->shape[d];

        std::vector<int64_t> epoch_x_shape = x_tensor->shape;
        epoch_x_shape[0] = (int64_t)n;
        std::vector<int64_t> epoch_y_shape = y_tensor->shape;
        epoch_y_shape[0] = (int64_t)n;

        auto epoch_x_cpu = Tensor::create(epoch_x_shape, Device(DeviceType::CPU, 0), false, x_tensor->dtype);
        auto epoch_y_cpu = Tensor::create(epoch_y_shape, Device(DeviceType::CPU, 0), false, y_tensor->dtype);

        size_t x_elem_sz = x_tensor->storage->element_size();
        size_t y_elem_sz = y_tensor->storage->element_size();
        char* ex_ptr = (char*)epoch_x_cpu->raw_data_ptr();
        char* ey_ptr = (char*)epoch_y_cpu->raw_data_ptr();
        char* x_ptr = (char*)x_tensor->raw_data_ptr();
        char* y_ptr = (char*)y_tensor->raw_data_ptr();

        for (size_t i = 0; i < n; ++i) {
            size_t idx = indices[i];
            std::memcpy(ex_ptr + i * D * x_elem_sz, x_ptr + idx * D * x_elem_sz, D * x_elem_sz);
            std::memcpy(ey_ptr + i * Dy * y_elem_sz, y_ptr + idx * Dy * y_elem_sz, Dy * y_elem_sz);
        }

        epoch_x_gpu = epoch_x_cpu->to(device);
        epoch_y_gpu = epoch_y_cpu->to(device);
    } else {
        epoch_x_gpu.reset();
        epoch_y_gpu.reset();
    }
}

void DataLoader::save_state(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("[litetorch Error] Failed to open file for saving dataloader state: " + filepath);
    }

    uint32_t running_crc = 0xFFFFFFFF;

    const char magic[4] = {'L', 'T', 'D', 'L'};
    out.write(magic, 4);
    running_crc = calculate_crc32(magic, 4, running_crc);

    uint64_t cur_idx = static_cast<uint64_t>(current_index);
    out.write(reinterpret_cast<const char*>(&cur_idx), sizeof(cur_idx));
    running_crc = calculate_crc32(&cur_idx, sizeof(cur_idx), running_crc);

    uint64_t head = static_cast<uint64_t>(out_head_);
    out.write(reinterpret_cast<const char*>(&head), sizeof(head));
    running_crc = calculate_crc32(&head, sizeof(head), running_crc);

    uint64_t ep_id = static_cast<uint64_t>(epoch_id);
    out.write(reinterpret_cast<const char*>(&ep_id), sizeof(ep_id));
    running_crc = calculate_crc32(&ep_id, sizeof(ep_id), running_crc);

    uint8_t shfl = shuffle ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&shfl), sizeof(shfl));
    running_crc = calculate_crc32(&shfl, sizeof(shfl), running_crc);

    uint64_t num_indices = static_cast<uint64_t>(indices.size());
    out.write(reinterpret_cast<const char*>(&num_indices), sizeof(num_indices));
    running_crc = calculate_crc32(&num_indices, sizeof(num_indices), running_crc);

    for (size_t idx : indices) {
        uint64_t val = static_cast<uint64_t>(idx);
        out.write(reinterpret_cast<const char*>(&val), sizeof(val));
        running_crc = calculate_crc32(&val, sizeof(val), running_crc);
    }

    uint32_t final_crc = running_crc ^ 0xFFFFFFFF;
    out.write(reinterpret_cast<const char*>(&final_crc), sizeof(final_crc));
    out.flush();
    if (out.fail()) {
        throw std::runtime_error("[litetorch Error] Write failure or disk full during saving dataloader state");
    }
    out.close();
}

void DataLoader::load_state(const std::string& filepath) {
    std::unique_lock<std::mutex> lock(queue_mutex);
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("[litetorch Error] Failed to open file for loading dataloader state: " + filepath);
    }

    uint32_t running_crc = 0xFFFFFFFF;

    char magic[4];
    in.read(magic, 4);
    if (magic[0] != 'L' || magic[1] != 'T' || magic[2] != 'D' || magic[3] != 'L') {
        throw std::runtime_error("[litetorch Error] Invalid file format magic in dataloader state file");
    }
    running_crc = calculate_crc32(magic, 4, running_crc);

    uint64_t cur_idx;
    in.read(reinterpret_cast<char*>(&cur_idx), sizeof(cur_idx));
    running_crc = calculate_crc32(&cur_idx, sizeof(cur_idx), running_crc);

    uint64_t head;
    in.read(reinterpret_cast<char*>(&head), sizeof(head));
    running_crc = calculate_crc32(&head, sizeof(head), running_crc);

    uint64_t ep_id;
    in.read(reinterpret_cast<char*>(&ep_id), sizeof(ep_id));
    running_crc = calculate_crc32(&ep_id, sizeof(ep_id), running_crc);

    uint8_t shfl;
    in.read(reinterpret_cast<char*>(&shfl), sizeof(shfl));
    running_crc = calculate_crc32(&shfl, sizeof(shfl), running_crc);

    uint64_t num_indices;
    in.read(reinterpret_cast<char*>(&num_indices), sizeof(num_indices));
    running_crc = calculate_crc32(&num_indices, sizeof(num_indices), running_crc);

    std::vector<size_t> loaded_indices(num_indices);
    for (uint64_t i = 0; i < num_indices; ++i) {
        uint64_t val;
        in.read(reinterpret_cast<char*>(&val), sizeof(val));
        running_crc = calculate_crc32(&val, sizeof(val), running_crc);
        loaded_indices[i] = static_cast<size_t>(val);
    }

    uint32_t stored_crc;
    in.read(reinterpret_cast<char*>(&stored_crc), sizeof(stored_crc));
    if (!in) {
        throw std::runtime_error("[litetorch Error] Unexpected end of file or read error in dataloader state file");
    }

    uint32_t expected_crc = running_crc ^ 0xFFFFFFFF;
    if (stored_crc != expected_crc) {
        throw std::runtime_error("[litetorch Error] Dataloader state file is corrupted (CRC32 checksum mismatch)");
    }

    if (loaded_indices.size() != dataset->size()) {
        throw std::runtime_error("[litetorch Error] Dataloader state indices size mismatch with dataset");
    }
    generation_id++;
    {
        bool release_gil = Py_IsInitialized() && PyGILState_Check();
        std::unique_ptr<py::gil_scoped_release> rel;
        if (release_gil) rel = std::make_unique<py::gil_scoped_release>();
        cv_cond.wait(lock, [this]() {
            return worker_stop.load() || in_flight_ == 0;
        });
    }
    worker_error_ = nullptr;
    indices = std::move(loaded_indices);
    num_batches_ = compute_num_batches();
    shuffle = (shfl != 0);
    epoch_id = static_cast<size_t>(ep_id);
    out_head_ = std::min(static_cast<size_t>(head), num_batches_);
    next_job_ = out_head_;
    in_flight_ = 0;
    out_buffer_.clear();
    current_index = static_cast<size_t>(cur_idx);
    build_epoch_cache();
    cv_cond.notify_all();
}

}
}
