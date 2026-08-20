#include "litetorch/data.h"
#include "litetorch/ops.h"
#include <random>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace litetorch {

namespace {
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

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, bool shuffle, const Device& device, size_t prefetch_limit)
    : dataset(dataset), batch_size(batch_size), shuffle(shuffle), worker_stop(false), worker_index(0), epoch_id(0), generation_id(0), prefetch_limit(prefetch_limit), device(device) {
    indices.resize(dataset->size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    reset();
    worker_thread = std::thread(&DataLoader::worker_loop, this);
}

DataLoader::~DataLoader() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        worker_stop = true;
    }
    cv_cond.notify_all();
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
}

void DataLoader::reset() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    current_index = 0;
    worker_index = 0;
    epoch_id++;
    generation_id++;
    if (shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }

    auto tensor_dataset = std::dynamic_pointer_cast<TensorDataset>(dataset);
    if (tensor_dataset && device.type != DeviceType::CPU) {
        size_t n = indices.size();
        auto x_tensor = tensor_dataset->x;
        auto y_tensor = tensor_dataset->y;

        int64_t D = 1;
        for (size_t d = 1; d < x_tensor->shape.size(); ++d) {
            D *= x_tensor->shape[d];
        }

        std::vector<int64_t> epoch_x_shape = x_tensor->shape;
        epoch_x_shape[0] = n;
        std::vector<int64_t> epoch_y_shape = y_tensor->shape;
        epoch_y_shape[0] = n;

        auto epoch_x_cpu = Tensor::create(epoch_x_shape, Device(DeviceType::CPU, 0), false, x_tensor->dtype);
        auto epoch_y_cpu = Tensor::create(epoch_y_shape, Device(DeviceType::CPU, 0), false, y_tensor->dtype);

        float* ex_ptr = epoch_x_cpu->data_ptr();
        float* ey_ptr = epoch_y_cpu->data_ptr();
        float* x_ptr = x_tensor->data_ptr();
        float* y_ptr = y_tensor->data_ptr();

        for (size_t i = 0; i < n; ++i) {
            size_t idx = indices[i];
            std::memcpy(ex_ptr + i * D, x_ptr + idx * D, D * sizeof(float));
            ey_ptr[i] = y_ptr[idx];
        }

        epoch_x_gpu = epoch_x_cpu->to(device);
        epoch_y_gpu = epoch_y_cpu->to(device);
    }

    while (!prefetch_queue.empty()) {
        prefetch_queue.pop();
    }
    cv_cond.notify_all();
}

bool DataLoader::next(std::shared_ptr<Tensor>& batch_x, std::shared_ptr<Tensor>& batch_y) {
    std::unique_lock<std::mutex> lock(queue_mutex);
    cv_cond.wait(lock, [this]() {
        return !prefetch_queue.empty() || current_index >= indices.size();
    });

    if (prefetch_queue.empty() && current_index >= indices.size()) {
        return false;
    }

    auto batch = prefetch_queue.front();
    prefetch_queue.pop();
    cv_cond.notify_all();

    batch_x = batch.x;
    batch_y = batch.y;
    current_index += batch_size;
    return true;
}

void DataLoader::worker_loop() {
    while (!worker_stop) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv_cond.wait(lock, [this]() {
            return worker_stop || (prefetch_queue.size() < prefetch_limit && worker_index < indices.size());
        });

        if (worker_stop) {
            break;
        }

        size_t start = worker_index;
        size_t end = std::min(start + batch_size, indices.size());
        worker_index = end;
        size_t batch_generation = generation_id;

        std::vector<size_t> batch_indices(end - start);
        std::copy(indices.begin() + start, indices.begin() + end, batch_indices.begin());
        
        auto local_x_gpu = epoch_x_gpu;
        auto local_y_gpu = epoch_y_gpu;

        lock.unlock();

        PrefetchBatch batch;
        auto tensor_dataset = std::dynamic_pointer_cast<TensorDataset>(dataset);
        if (tensor_dataset) {
            size_t batch_len = end - start;
            auto x_tensor = tensor_dataset->x;
            auto y_tensor = tensor_dataset->y;

            int64_t D = 1;
            for (size_t d = 1; d < x_tensor->shape.size(); ++d) {
                D *= x_tensor->shape[d];
            }
            
            std::vector<int64_t> batch_x_shape = x_tensor->shape;
            batch_x_shape[0] = batch_len;
            std::vector<int64_t> batch_y_shape = y_tensor->shape;
            batch_y_shape[0] = batch_len;

            if (device.type != DeviceType::CPU) {
                batch.x = std::make_shared<Tensor>(
                    local_x_gpu->storage, 
                    batch_x_shape, 
                    default_strides(batch_x_shape), 
                    local_x_gpu->offset + start * D, 
                    device, 
                    false
                );
                batch.y = std::make_shared<Tensor>(
                    local_y_gpu->storage, 
                    batch_y_shape, 
                    default_strides(batch_y_shape), 
                    local_y_gpu->offset + start, 
                    device, 
                    false
                );
            } else {
                auto batch_x_cpu = Tensor::create(batch_x_shape, Device(DeviceType::CPU, 0), false, x_tensor->dtype);
                auto batch_y_cpu = Tensor::create(batch_y_shape, Device(DeviceType::CPU, 0), false, y_tensor->dtype);

                float* bx_ptr = batch_x_cpu->data_ptr();
                float* by_ptr = batch_y_cpu->data_ptr();
                float* x_ptr = x_tensor->data_ptr();
                float* y_ptr = y_tensor->data_ptr();

                for (size_t i = 0; i < batch_len; ++i) {
                    size_t idx = batch_indices[i];
                    std::memcpy(bx_ptr + i * D, x_ptr + idx * D, D * sizeof(float));
                    by_ptr[i] = y_ptr[idx];
                }
                batch.x = batch_x_cpu;
                batch.y = batch_y_cpu;
            }
        } else {
            std::vector<std::shared_ptr<Tensor>> xs;
            std::vector<std::shared_ptr<Tensor>> ys;
            for (size_t i = start; i < end; ++i) {
                auto pair = dataset->get(batch_indices[i - start]);
                xs.push_back(Ops::unsqueeze(pair.first, 0));
                ys.push_back(Ops::unsqueeze(pair.second, 0));
            }

            batch.x = Ops::cat(xs, 0);
            batch.y = Ops::squeeze(Ops::cat(ys, 0), 1);

            if (device.type != DeviceType::CPU) {
                batch.x = batch.x->to(device);
                batch.y = batch.y->to(device);
            }
        }

        lock.lock();
        if (batch_generation == generation_id) {
            prefetch_queue.push(batch);
            cv_cond.notify_all();
        }
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

    uint64_t wrk_idx = static_cast<uint64_t>(worker_index);
    out.write(reinterpret_cast<const char*>(&wrk_idx), sizeof(wrk_idx));
    running_crc = calculate_crc32(&wrk_idx, sizeof(wrk_idx), running_crc);

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

    uint64_t wrk_idx;
    in.read(reinterpret_cast<char*>(&wrk_idx), sizeof(wrk_idx));
    running_crc = calculate_crc32(&wrk_idx, sizeof(wrk_idx), running_crc);

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

    current_index = static_cast<size_t>(cur_idx);
    epoch_id = static_cast<size_t>(ep_id);
    generation_id++;
    shuffle = (shfl != 0);
    indices = std::move(loaded_indices);

    while (!prefetch_queue.empty()) {
        prefetch_queue.pop();
    }
    worker_index = current_index;
    cv_cond.notify_all();
}

}
}
