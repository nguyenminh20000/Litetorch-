#include "litetorch/distributed.h"
#include "litetorch/backend.h"
#include "litetorch/platform.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <cstring>
#include <stdexcept>
#include <algorithm>

#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

namespace litetorch {
namespace distributed {

ProcessGroup& ProcessGroup::get() {
    static ProcessGroup instance;
    return instance;
}

ProcessGroup::~ProcessGroup() {
    shutdown();
}

#ifdef _WIN32
void ProcessGroup::init(int rank, int world_size, const std::string&, int) {
    rank_ = rank;
    world_size_ = world_size;
    initialized_ = true;
}
void ProcessGroup::shutdown() { initialized_ = false; }
void ProcessGroup::all_reduce(std::shared_ptr<Tensor>) {}
std::future<void> ProcessGroup::all_reduce_async(std::shared_ptr<Tensor>) {
    return std::async(std::launch::deferred, [](){});
}
void ProcessGroup::broadcast(std::shared_ptr<Tensor>, int) {}
void ProcessGroup::all_gather(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>) {}
void ProcessGroup::reduce_scatter(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>) {}
void ProcessGroup::send_tensor(std::shared_ptr<Tensor>, int) {}
void ProcessGroup::recv_tensor(std::shared_ptr<Tensor>, int) {}
void ProcessGroup::sync_comm() {}
#else
void ProcessGroup::init(int rank, int world_size, const std::string& master_addr, int master_port) {
    if (initialized_) return;

    rank_ = rank;
    world_size_ = world_size;

    if (world_size_ <= 1) {
        initialized_ = true;
        return;
    }

    if (rank_ == 0) {
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ < 0) {
            throw std::runtime_error("Failed to create server socket");
        }

        int opt = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in address;
        std::memset(&address, 0, sizeof(address));
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(master_port);

        if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) < 0) {
            close(server_fd_);
            server_fd_ = -1;
            throw std::runtime_error("Failed to bind server socket to port " + std::to_string(master_port));
        }

        if (listen(server_fd_, world_size_ - 1) < 0) {
            close(server_fd_);
            server_fd_ = -1;
            throw std::runtime_error("Failed to listen on server socket");
        }

        client_fds_.resize(world_size_, -1);
        for (int i = 1; i < world_size_; ++i) {
            struct sockaddr_in client_addr;
            socklen_t addr_len = sizeof(client_addr);
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &addr_len);
            if (client_fd < 0) {
                shutdown();
                throw std::runtime_error("Failed to accept client connection");
            }

            int client_rank = -1;
            if (read(client_fd, &client_rank, sizeof(client_rank)) != sizeof(client_rank)) {
                close(client_fd);
                shutdown();
                throw std::runtime_error("Failed to read client rank");
            }

            client_fds_[client_rank] = client_fd;
        }

        int sync_val = 1;
        for (int i = 1; i < world_size_; ++i) {
            if (write(client_fds_[i], &sync_val, sizeof(sync_val)) != sizeof(sync_val)) {
                shutdown();
                throw std::runtime_error("Failed to write sync confirmation to client");
            }
        }
    } else {
        peer_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (peer_fd_ < 0) {
            throw std::runtime_error("Failed to create client socket");
        }

        struct sockaddr_in serv_addr;
        std::memset(&serv_addr, 0, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(master_port);
        if (inet_pton(AF_INET, master_addr.c_str(), &serv_addr.sin_addr) <= 0) {
            close(peer_fd_);
            peer_fd_ = -1;
            throw std::runtime_error("Invalid master address: " + master_addr);
        }

        int retry = 0;
        while (connect(peer_fd_, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
            usleep(100000);
            if (++retry > 100) {
                close(peer_fd_);
                peer_fd_ = -1;
                throw std::runtime_error("Failed to connect to master at " + master_addr + ":" + std::to_string(master_port));
            }
        }

        if (write(peer_fd_, &rank_, sizeof(rank_)) != sizeof(rank_)) {
            close(peer_fd_);
            peer_fd_ = -1;
            throw std::runtime_error("Failed to send rank to master");
        }

        int sync_val = 0;
        if (read(peer_fd_, &sync_val, sizeof(sync_val)) != sizeof(sync_val)) {
            close(peer_fd_);
            peer_fd_ = -1;
            throw std::runtime_error("Failed to read sync confirmation from master");
        }
    }

    init_shm(master_addr, master_port);

    if (world_size_ > 1) {
        stop_async_worker_ = false;
        async_worker_ = std::thread([this]() {
            while (true) {
                std::pair<std::shared_ptr<Tensor>, std::shared_ptr<std::promise<void>>> task;
                {
                    std::unique_lock<std::mutex> lock(async_mutex_);
                    async_cv_.wait(lock, [this]() {
                        return stop_async_worker_ || !async_queue_.empty();
                    });
                    if (stop_async_worker_ && async_queue_.empty()) {
                        break;
                    }
                    task = async_queue_.front();
                    async_queue_.pop();
                }
                try {
                    all_reduce(task.first);
                    task.second->set_value();
                } catch (...) {
                    task.second->set_exception(std::current_exception());
                }
            }
        });
    }

    initialized_ = true;
}

void ProcessGroup::shutdown() {
    if (!initialized_) return;

    if (world_size_ > 1) {
        {
            std::lock_guard<std::mutex> lock(async_mutex_);
            stop_async_worker_ = true;
        }
        async_cv_.notify_all();
        if (async_worker_.joinable()) {
            async_worker_.join();
        }
    }

    shutdown_shm();

    if (server_fd_ != -1) {
        close(server_fd_);
        server_fd_ = -1;
    }
    for (int fd : client_fds_) {
        if (fd != -1) {
            close(fd);
        }
    }
    client_fds_.clear();

    if (peer_fd_ != -1) {
        close(peer_fd_);
        peer_fd_ = -1;
    }

    initialized_ = false;
}

void ProcessGroup::all_reduce(std::shared_ptr<Tensor> tensor) {
    if (!initialized_ || world_size_ <= 1) return;

    if (NCCLBridge::get().all_reduce(tensor)) {
        return;
    }

    if (all_reduce_shm(tensor)) {
        return;
    }

    size_t N = tensor->numel();
    size_t elem_sz = tensor->storage->element_size();
    std::vector<char> data(N * elem_sz);

    if (tensor->device.type == DeviceType::GPU) {
        CLBackend::get().read(tensor->gpu_data(), N * elem_sz, data.data(), tensor->offset * elem_sz);
    } else {
        std::memcpy(data.data(), tensor->data_ptr(), N * elem_sz);
    }

    if (rank_ == 0) {
        if (tensor->dtype == DataType::FP16) {
            std::vector<float> acc(N, 0.0f);
            uint16_t* data_ptr = reinterpret_cast<uint16_t*>(data.data());
            for (size_t j = 0; j < N; ++j) acc[j] += half_to_float(data_ptr[j]);

            std::vector<char> temp(N * elem_sz);
            uint16_t* temp_ptr = reinterpret_cast<uint16_t*>(temp.data());
            for (int i = 1; i < world_size_; ++i) {
                size_t bytes_received = 0;
                while (bytes_received < N * elem_sz) {
                    ssize_t r = recv(client_fds_[i], temp.data() + bytes_received, N * elem_sz - bytes_received, 0);
                    if (r <= 0) throw std::runtime_error("Socket read error in all_reduce");
                    bytes_received += r;
                }
                for (size_t j = 0; j < N; ++j) acc[j] += half_to_float(temp_ptr[j]);
            }

            for (size_t j = 0; j < N; ++j) acc[j] /= world_size_;
            for (size_t j = 0; j < N; ++j) data_ptr[j] = float_to_half(acc[j]);

            for (int i = 1; i < world_size_; ++i) {
                size_t bytes_sent = 0;
                while (bytes_sent < N * elem_sz) {
                    ssize_t s = send(client_fds_[i], data.data() + bytes_sent, N * elem_sz - bytes_sent, 0);
                    if (s <= 0) throw std::runtime_error("Socket write error in all_reduce");
                    bytes_sent += s;
                }
            }
        } else if (tensor->dtype == DataType::BF16) {
            std::vector<float> acc(N, 0.0f);
            uint16_t* data_ptr = reinterpret_cast<uint16_t*>(data.data());
            for (size_t j = 0; j < N; ++j) {
                uint32_t val = ((uint32_t)data_ptr[j]) << 16;
                acc[j] += *(float*)&val;
            }

            std::vector<char> temp(N * elem_sz);
            uint16_t* temp_ptr = reinterpret_cast<uint16_t*>(temp.data());
            for (int i = 1; i < world_size_; ++i) {
                size_t bytes_received = 0;
                while (bytes_received < N * elem_sz) {
                    ssize_t r = recv(client_fds_[i], temp.data() + bytes_received, N * elem_sz - bytes_received, 0);
                    if (r <= 0) throw std::runtime_error("Socket read error in all_reduce");
                    bytes_received += r;
                }
                for (size_t j = 0; j < N; ++j) {
                    uint32_t val = ((uint32_t)temp_ptr[j]) << 16;
                    acc[j] += *(float*)&val;
                }
            }

            for (size_t j = 0; j < N; ++j) acc[j] /= world_size_;
            for (size_t j = 0; j < N; ++j) {
                float val_f = acc[j];
                uint32_t val_u = *(uint32_t*)&val_f;
                data_ptr[j] = (uint16_t)(val_u >> 16);
            }

            for (int i = 1; i < world_size_; ++i) {
                size_t bytes_sent = 0;
                while (bytes_sent < N * elem_sz) {
                    ssize_t s = send(client_fds_[i], data.data() + bytes_sent, N * elem_sz - bytes_sent, 0);
                    if (s <= 0) throw std::runtime_error("Socket write error in all_reduce");
                    bytes_sent += s;
                }
            }
        } else {
            std::vector<float> acc(N, 0.0f);
            float* data_ptr = reinterpret_cast<float*>(data.data());
            for (size_t j = 0; j < N; ++j) acc[j] += data_ptr[j];

            std::vector<char> temp(N * elem_sz);
            float* temp_ptr = reinterpret_cast<float*>(temp.data());
            for (int i = 1; i < world_size_; ++i) {
                size_t bytes_received = 0;
                while (bytes_received < N * elem_sz) {
                    ssize_t r = recv(client_fds_[i], temp.data() + bytes_received, N * elem_sz - bytes_received, 0);
                    if (r <= 0) throw std::runtime_error("Socket read error in all_reduce");
                    bytes_received += r;
                }
                for (size_t j = 0; j < N; ++j) acc[j] += temp_ptr[j];
            }

            for (size_t j = 0; j < N; ++j) data_ptr[j] = acc[j] / world_size_;

            for (int i = 1; i < world_size_; ++i) {
                size_t bytes_sent = 0;
                while (bytes_sent < N * elem_sz) {
                    ssize_t s = send(client_fds_[i], data.data() + bytes_sent, N * elem_sz - bytes_sent, 0);
                    if (s <= 0) throw std::runtime_error("Socket write error in all_reduce");
                    bytes_sent += s;
                }
            }
        }
    } else {
        size_t bytes_sent = 0;
        while (bytes_sent < N * elem_sz) {
            ssize_t s = send(peer_fd_, data.data() + bytes_sent, N * elem_sz - bytes_sent, 0);
            if (s <= 0) throw std::runtime_error("Socket write error in all_reduce");
            bytes_sent += s;
        }

        size_t bytes_received = 0;
        while (bytes_received < N * elem_sz) {
            ssize_t r = recv(peer_fd_, data.data() + bytes_received, N * elem_sz - bytes_received, 0);
            if (r <= 0) throw std::runtime_error("Socket read error in all_reduce");
            bytes_received += r;
        }
    }

    if (tensor->device.type == DeviceType::GPU) {
        CLBackend::get().write(tensor->gpu_data(), N * elem_sz, data.data(), tensor->offset * elem_sz);
    } else {
        std::memcpy(tensor->data_ptr(), data.data(), N * elem_sz);
    }
}

void ProcessGroup::broadcast(std::shared_ptr<Tensor> tensor, int src) {
    if (!initialized_ || world_size_ <= 1) return;

    if (NCCLBridge::get().broadcast(tensor, src)) {
        return;
    }

    if (broadcast_shm(tensor, src)) {
        return;
    }

    size_t N = tensor->numel();
    size_t elem_sz = tensor->storage->element_size();
    std::vector<char> data(N * elem_sz);

    if (rank_ == src) {
        if (tensor->device.type == DeviceType::GPU) {
            CLBackend::get().read(tensor->gpu_data(), N * elem_sz, data.data(), tensor->offset * elem_sz);
        } else {
            std::memcpy(data.data(), tensor->data_ptr(), N * elem_sz);
        }
    }

    if (src == 0) {
        if (rank_ == 0) {
            for (int i = 1; i < world_size_; ++i) {
                size_t bytes_sent = 0;
                while (bytes_sent < N * elem_sz) {
                    ssize_t s = send(client_fds_[i], data.data() + bytes_sent, N * elem_sz - bytes_sent, 0);
                    if (s <= 0) throw std::runtime_error("Socket write error in broadcast");
                    bytes_sent += s;
                }
            }
        } else {
            size_t bytes_received = 0;
            while (bytes_received < N * elem_sz) {
                ssize_t r = recv(peer_fd_, data.data() + bytes_received, N * elem_sz - bytes_received, 0);
                if (r <= 0) throw std::runtime_error("Socket read error in broadcast");
                bytes_received += r;
            }
        }
    } else {
        if (rank_ == src) {
            size_t bytes_sent = 0;
            while (bytes_sent < N * elem_sz) {
                ssize_t s = send(peer_fd_, data.data() + bytes_sent, N * elem_sz - bytes_sent, 0);
                if (s <= 0) throw std::runtime_error("Socket write error in broadcast");
                bytes_sent += s;
            }
        } else if (rank_ == 0) {
            size_t bytes_received = 0;
            while (bytes_received < N * elem_sz) {
                ssize_t r = recv(client_fds_[src], data.data() + bytes_received, N * elem_sz - bytes_received, 0);
                if (r <= 0) throw std::runtime_error("Socket read error in broadcast");
                bytes_received += r;
            }

            for (int i = 1; i < world_size_; ++i) {
                if (i == src) continue;
                size_t bytes_sent = 0;
                while (bytes_sent < N * elem_sz) {
                    ssize_t s = send(client_fds_[i], data.data() + bytes_sent, N * elem_sz - bytes_sent, 0);
                    if (s <= 0) throw std::runtime_error("Socket write error in broadcast");
                    bytes_sent += s;
                }
            }
        } else {
            size_t bytes_received = 0;
            while (bytes_received < N * elem_sz) {
                ssize_t r = recv(peer_fd_, data.data() + bytes_received, N * elem_sz - bytes_received, 0);
                if (r <= 0) throw std::runtime_error("Socket read error in broadcast");
                bytes_received += r;
            }
        }
    }

    if (rank_ != src) {
        if (tensor->device.type == DeviceType::GPU) {
            CLBackend::get().write(tensor->gpu_data(), N * elem_sz, data.data(), tensor->offset * elem_sz);
        } else {
            std::memcpy(tensor->data_ptr(), data.data(), N * elem_sz);
        }
    }
}

void init_process_group(int rank, int world_size, const std::string& master_addr, int master_port) {
    ProcessGroup::get().init(rank, world_size, master_addr, master_port);
    if (world_size > 1) {
        NCCLBridge::get().init(rank, world_size);
    }
}

void shutdown() {
    NCCLBridge::get().shutdown();
    ProcessGroup::get().shutdown();
}

void all_reduce(std::shared_ptr<Tensor> tensor) {
    ProcessGroup::get().all_reduce(tensor);
}

std::future<void> ProcessGroup::all_reduce_async(std::shared_ptr<Tensor> tensor) {
    auto promise = std::make_shared<std::promise<void>>();
    if (!initialized_ || world_size_ <= 1) {
        promise->set_value();
        return promise->get_future();
    }
    {
        std::lock_guard<std::mutex> lock(async_mutex_);
        async_queue_.push({tensor, promise});
    }
    async_cv_.notify_one();
    return promise->get_future();
}

std::future<void> all_reduce_async(std::shared_ptr<Tensor> tensor) {
    return ProcessGroup::get().all_reduce_async(tensor);
}

void broadcast(std::shared_ptr<Tensor> tensor, int src_rank) {
    ProcessGroup::get().broadcast(tensor, src_rank);
}

void reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    ProcessGroup::get().reduce_scatter(shard, full);
}

void ProcessGroup::all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    if (!initialized_ || world_size_ <= 1) {
        full->copy_(shard);
        return;
    }

    if (NCCLBridge::get().all_gather(shard, full)) {
        return;
    }

    if (all_gather_shm(shard, full)) {
        return;
    }

    size_t S = shard->numel();
    size_t elem_sz = shard->storage->element_size();
    std::vector<char> data(S * elem_sz);

    if (shard->device.type == DeviceType::GPU) {
        CLBackend::get().read(shard->gpu_data(), S * elem_sz, data.data(), shard->offset * elem_sz);
    } else {
        std::memcpy(data.data(), shard->data_ptr(), S * elem_sz);
    }

    std::vector<char> full_cpu(S * world_size_ * elem_sz);

    if (rank_ == 0) {
        std::copy(data.begin(), data.end(), full_cpu.begin());

        for (int r = 1; r < world_size_; ++r) {
            size_t bytes_received = 0;
            while (bytes_received < S * elem_sz) {
                ssize_t n = recv(client_fds_[r], full_cpu.data() + r * S * elem_sz + bytes_received, S * elem_sz - bytes_received, 0);
                if (n <= 0) throw std::runtime_error("Socket read error in all_gather");
                bytes_received += n;
            }
        }

        for (int r = 1; r < world_size_; ++r) {
            size_t bytes_sent = 0;
            while (bytes_sent < S * world_size_ * elem_sz) {
                ssize_t n = send(client_fds_[r], full_cpu.data() + bytes_sent, S * world_size_ * elem_sz - bytes_sent, 0);
                if (n <= 0) throw std::runtime_error("Socket write error in all_gather");
                bytes_sent += n;
            }
        }
    } else {
        size_t bytes_sent = 0;
        while (bytes_sent < S * elem_sz) {
            ssize_t n = send(peer_fd_, data.data() + bytes_sent, S * elem_sz - bytes_sent, 0);
            if (n <= 0) throw std::runtime_error("Socket write error in all_gather");
            bytes_sent += n;
        }

        size_t bytes_received = 0;
        while (bytes_received < S * world_size_ * elem_sz) {
            ssize_t n = recv(peer_fd_, full_cpu.data() + bytes_received, S * world_size_ * elem_sz - bytes_received, 0);
            if (n <= 0) throw std::runtime_error("Socket read error in all_gather");
            bytes_received += n;
        }
    }

    if (full->device.type == DeviceType::GPU) {
        CLBackend::get().write(full->gpu_data(), S * world_size_ * elem_sz, full_cpu.data(), full->offset * elem_sz);
    } else {
        std::memcpy(full->data_ptr(), full_cpu.data(), S * world_size_ * elem_sz);
    }
}

void ProcessGroup::reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    if (!initialized_ || world_size_ <= 1) {
        shard->copy_(full);
        return;
    }

    if (NCCLBridge::get().reduce_scatter(shard, full)) {
        return;
    }

    if (reduce_scatter_shm(shard, full)) {
        return;
    }
    
    throw std::runtime_error("reduce_scatter fallback not implemented for socket");
}

void ProcessGroup::sync_comm() {
    if (!initialized_) return;
    NCCLBridge::get().sync_comm();
}

void ProcessGroup::send_tensor(std::shared_ptr<Tensor> tensor, int dst) {
    if (!initialized_ || world_size_ <= 1) return;
    if (rank_ == dst) return;

    size_t N = tensor->numel();
    size_t elem_sz = tensor->storage->element_size();
    int ndim = tensor->shape.size();
    int dtype_val = static_cast<int>(tensor->dtype);

    if (use_shm_ && shm_ctrl_ && shm_buffers_[rank_]) {
        int cur_step = shm_ctrl_->steps[rank_].load();
        while (shm_ctrl_->steps[dst].load() < cur_step) {
            usleep(1);
        }
        shm_buffers_[rank_][0] = static_cast<float>(ndim);
        shm_buffers_[rank_][1] = static_cast<float>(dtype_val);
        for (int i = 0; i < ndim; ++i) {
            shm_buffers_[rank_][2 + i] = static_cast<float>(tensor->shape[i]);
        }

        if (tensor->device.type == DeviceType::GPU) {
            CLBackend::get().read(tensor->gpu_data(), N * elem_sz, shm_buffers_[rank_] + 2 + ndim, tensor->offset * elem_sz);
        } else {
            std::memcpy(shm_buffers_[rank_] + 2 + ndim, tensor->data_ptr(), N * elem_sz);
        }

        shm_ctrl_->steps[rank_].store(cur_step + 1);
    } else {
        if (rank_ == 0) {
            send(client_fds_[dst], &ndim, sizeof(ndim), 0);
            send(client_fds_[dst], &dtype_val, sizeof(dtype_val), 0);
            send(client_fds_[dst], tensor->shape.data(), ndim * sizeof(int64_t), 0);

            std::vector<char> temp_buf(N * elem_sz);
            if (tensor->device.type == DeviceType::GPU) {
                CLBackend::get().read(tensor->gpu_data(), N * elem_sz, temp_buf.data(), tensor->offset * elem_sz);
            } else {
                std::memcpy(temp_buf.data(), tensor->data_ptr(), N * elem_sz);
            }

            size_t bytes_sent = 0;
            while (bytes_sent < N * elem_sz) {
                ssize_t s = send(client_fds_[dst], temp_buf.data() + bytes_sent, N * elem_sz - bytes_sent, 0);
                if (s <= 0) throw std::runtime_error("Socket write error in send_tensor");
                bytes_sent += s;
            }
        } else if (dst == 0) {
            send(peer_fd_, &ndim, sizeof(ndim), 0);
            send(peer_fd_, &dtype_val, sizeof(dtype_val), 0);
            send(peer_fd_, tensor->shape.data(), ndim * sizeof(int64_t), 0);

            std::vector<char> temp_buf(N * elem_sz);
            if (tensor->device.type == DeviceType::GPU) {
                CLBackend::get().read(tensor->gpu_data(), N * elem_sz, temp_buf.data(), tensor->offset * elem_sz);
            } else {
                std::memcpy(temp_buf.data(), tensor->data_ptr(), N * elem_sz);
            }

            size_t bytes_sent = 0;
            while (bytes_sent < N * elem_sz) {
                ssize_t s = send(peer_fd_, temp_buf.data() + bytes_sent, N * elem_sz - bytes_sent, 0);
                if (s <= 0) throw std::runtime_error("Socket write error in send_tensor");
                bytes_sent += s;
            }
        }
    }
}

void ProcessGroup::recv_tensor(std::shared_ptr<Tensor> tensor, int src) {
    if (!initialized_ || world_size_ <= 1) return;
    if (rank_ == src) return;

    int ndim = 0;
    int dtype_val = 0;
    std::vector<int64_t> shape;

    if (use_shm_ && shm_ctrl_ && shm_buffers_[src]) {
        int cur_step = shm_ctrl_->steps[rank_].load();
        while (shm_ctrl_->steps[src].load() < cur_step + 1) {
            usleep(1);
        }
        ndim = static_cast<int>(shm_buffers_[src][0]);
        dtype_val = static_cast<int>(shm_buffers_[src][1]);
        shape.resize(ndim);
        size_t N = 1;
        for (int i = 0; i < ndim; ++i) {
            shape[i] = static_cast<int64_t>(shm_buffers_[src][2 + i]);
            N *= shape[i];
        }
        DataType dtype = static_cast<DataType>(dtype_val);

        tensor->shape = shape;
        tensor->strides = default_strides(shape);
        tensor->numel_ = N;
        tensor->contiguous_ = true;
        tensor->dtype = dtype;
        tensor->storage = std::make_shared<StorageImpl>(N, tensor->device, dtype);
        size_t elem_sz = tensor->storage->element_size();

        if (tensor->device.type == DeviceType::GPU) {
            CLBackend::get().write(tensor->gpu_data(), N * elem_sz, shm_buffers_[src] + 2 + ndim, 0);
        } else {
            std::memcpy(tensor->data_ptr(), shm_buffers_[src] + 2 + ndim, N * elem_sz);
        }

        shm_ctrl_->steps[rank_].store(cur_step + 1);
    } else {
        if (rank_ == 0) {
            recv(client_fds_[src], &ndim, sizeof(ndim), MSG_WAITALL);
            recv(client_fds_[src], &dtype_val, sizeof(dtype_val), MSG_WAITALL);
            shape.resize(ndim);
            recv(client_fds_[src], shape.data(), ndim * sizeof(int64_t), MSG_WAITALL);
            DataType dtype = static_cast<DataType>(dtype_val);

            size_t N = 1;
            for (int i = 0; i < ndim; ++i) N *= shape[i];

            tensor->shape = shape;
            tensor->strides = default_strides(shape);
            tensor->numel_ = N;
            tensor->contiguous_ = true;
            tensor->dtype = dtype;
            tensor->storage = std::make_shared<StorageImpl>(N, tensor->device, dtype);
            size_t elem_sz = tensor->storage->element_size();

            std::vector<char> temp_buf(N * elem_sz);
            size_t bytes_received = 0;
            while (bytes_received < N * elem_sz) {
                ssize_t r = recv(client_fds_[src], temp_buf.data() + bytes_received, N * elem_sz - bytes_received, 0);
                if (r <= 0) throw std::runtime_error("Socket read error in recv_tensor");
                bytes_received += r;
            }

            if (tensor->device.type == DeviceType::GPU) {
                CLBackend::get().write(tensor->gpu_data(), N * elem_sz, temp_buf.data(), 0);
            } else {
                std::memcpy(tensor->data_ptr(), temp_buf.data(), N * elem_sz);
            }
        } else if (src == 0) {
            recv(peer_fd_, &ndim, sizeof(ndim), MSG_WAITALL);
            recv(peer_fd_, &dtype_val, sizeof(dtype_val), MSG_WAITALL);
            shape.resize(ndim);
            recv(peer_fd_, shape.data(), ndim * sizeof(int64_t), MSG_WAITALL);
            DataType dtype = static_cast<DataType>(dtype_val);

            size_t N = 1;
            for (int i = 0; i < ndim; ++i) N *= shape[i];

            tensor->shape = shape;
            tensor->strides = default_strides(shape);
            tensor->numel_ = N;
            tensor->contiguous_ = true;
            tensor->dtype = dtype;
            tensor->storage = std::make_shared<StorageImpl>(N, tensor->device, dtype);
            size_t elem_sz = tensor->storage->element_size();

            std::vector<char> temp_buf(N * elem_sz);
            size_t bytes_received = 0;
            while (bytes_received < N * elem_sz) {
                ssize_t r = recv(peer_fd_, temp_buf.data() + bytes_received, N * elem_sz - bytes_received, 0);
                if (r <= 0) throw std::runtime_error("Socket read error in recv_tensor");
                bytes_received += r;
            }

            if (tensor->device.type == DeviceType::GPU) {
                CLBackend::get().write(tensor->gpu_data(), N * elem_sz, temp_buf.data(), 0);
            } else {
                std::memcpy(tensor->data_ptr(), temp_buf.data(), N * elem_sz);
            }
        }
    }
}
#endif

void all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    ProcessGroup::get().all_gather(shard, full);
}

void all_reduce_grads(std::shared_ptr<nn::Module> module) {
    std::vector<std::future<void>> futures;
    for (auto& p : module->parameters()) {
        if (p->requires_grad && p->grad) {
            futures.push_back(ProcessGroup::get().all_reduce_async(p->grad));
        }
    }
    for (auto& f : futures) {
        f.wait();
    }
}

void send_tensor(std::shared_ptr<Tensor> tensor, int dst) {
    ProcessGroup::get().send_tensor(tensor, dst);
}

void recv_tensor(std::shared_ptr<Tensor> tensor, int src) {
    ProcessGroup::get().recv_tensor(tensor, src);
}

NCCLBridge& NCCLBridge::get() {
    static NCCLBridge instance;
    return instance;
}

NCCLBridge::NCCLBridge() = default;
NCCLBridge::~NCCLBridge() {
    shutdown();
}

void NCCLBridge::init(int rank, int world_size) {
    if (initialized_) return;

#ifdef _WIN32
    available_ = false;
    initialized_ = true;
    return;
#else
    lib_handle_ = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
    if (!lib_handle_) {
        lib_handle_ = dlopen("librccl.so", RTLD_NOW | RTLD_GLOBAL);
    }

    if (!lib_handle_) {
        available_ = false;
        initialized_ = true;
        return;
    }

    get_unique_id_fn = dlsym(lib_handle_, "ncclGetUniqueId");
    comm_init_rank_fn = dlsym(lib_handle_, "ncclCommInitRank");
    all_reduce_fn = dlsym(lib_handle_, "ncclAllReduce");
    broadcast_fn = dlsym(lib_handle_, "ncclBroadcast");
    all_gather_fn = dlsym(lib_handle_, "ncclAllGather");
    reduce_scatter_fn = dlsym(lib_handle_, "ncclReduceScatter");
    comm_destroy_fn = dlsym(lib_handle_, "ncclCommDestroy");
    group_start_fn = dlsym(lib_handle_, "ncclGroupStart");
    group_end_fn = dlsym(lib_handle_, "ncclGroupEnd");

    if (!get_unique_id_fn || !comm_init_rank_fn || !all_reduce_fn || 
        !broadcast_fn || !all_gather_fn || !reduce_scatter_fn || !comm_destroy_fn) {
        dlclose(lib_handle_);
        lib_handle_ = nullptr;
        available_ = false;
        initialized_ = true;
        return;
    }

    char unique_id[128] = {0};
    const char* rend_file = std::getenv("LITETORCH_RENDEZVOUS_FILE");
    if (!rend_file) {
        rend_file = std::getenv("NCCL_UNIQUE_ID_FILE");
    }

    if (rend_file && std::strlen(rend_file) > 0) {
        if (rank == 0) {
            typedef int (*get_id_t)(void*);
            ((get_id_t)get_unique_id_fn)(unique_id);
            std::ofstream ofs(rend_file, std::ios::binary | std::ios::trunc);
            if (ofs.is_open()) {
                ofs.write(unique_id, 128);
                ofs.close();
            }
        } else {
            bool loaded = false;
            for (int attempt = 0; attempt < 3000; ++attempt) {
                std::ifstream ifs(rend_file, std::ios::binary);
                if (ifs.is_open()) {
                    ifs.seekg(0, std::ios::end);
                    if (ifs.tellg() >= 128) {
                        ifs.seekg(0, std::ios::beg);
                        ifs.read(unique_id, 128);
                        loaded = true;
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (!loaded) {
                dlclose(lib_handle_);
                lib_handle_ = nullptr;
                available_ = false;
                initialized_ = true;
                return;
            }
        }
    } else {
        auto& pg = ProcessGroup::get();
        if (rank == 0) {
            typedef int (*get_id_t)(void*);
            ((get_id_t)get_unique_id_fn)(unique_id);
            for (int i = 1; i < world_size; ++i) {
                if (i < static_cast<int>(pg.client_fds_.size()) && pg.client_fds_[i] >= 0) {
                    size_t sent = 0;
                    while (sent < 128) {
                        ssize_t s = ::send(pg.client_fds_[i], unique_id + sent, 128 - sent, 0);
                        if (s <= 0) break;
                        sent += s;
                    }
                }
            }
        } else {
            if (pg.peer_fd_ >= 0) {
                size_t recvd = 0;
                while (recvd < 128) {
                    ssize_t r = ::recv(pg.peer_fd_, unique_id + recvd, 128 - recvd, MSG_WAITALL);
                    if (r <= 0) break;
                    recvd += r;
                }
            }
        }
    }

    typedef int (*init_rank_t)(void**, int, void*, int);
    int res = ((init_rank_t)comm_init_rank_fn)(&comm_, world_size, unique_id, rank);
    if (res != 0) {
        dlclose(lib_handle_);
        lib_handle_ = nullptr;
        available_ = false;
        initialized_ = true;
        return;
    }

    available_ = true;
    initialized_ = true;
    
    if (BackendDispatcher::get().get_backend() && BackendDispatcher::get().get_backend()->is_available()) {
        comm_stream_ = BackendDispatcher::get().get_backend()->get_comm_stream();
    }
#endif
}

void NCCLBridge::init_groups(int rank, int world_size, int tp_size, int pp_size) {
#ifndef _WIN32
    if (!available_ || !comm_) return;
    comm_split_fn = dlsym(lib_handle_, "ncclCommSplit");
    if (!comm_split_fn) return;

    typedef int (*split_t)(void*, int, int, void**, void*);
    auto split = (split_t)comm_split_fn;

    int tp_color = rank / tp_size;
    int tp_key = rank % tp_size;
    split(comm_, tp_color, tp_key, &tp_comm_, nullptr);

    int dp_size = world_size / (tp_size * pp_size);
    int dp_color = (rank % (tp_size * pp_size)) / tp_size;
    int dp_key = rank / (tp_size * pp_size);
    split(comm_, dp_color, dp_key, &dp_comm_, nullptr);

    int pp_color = rank % (tp_size * dp_size);
    int pp_key = rank / (tp_size * dp_size);
    split(comm_, pp_color, pp_key, &pp_comm_, nullptr);

    auto backend = BackendDispatcher::get().get_backend();
    if (backend && backend->is_available()) {
        tp_stream_ = backend->create_stream();
        dp_stream_ = backend->create_stream();
        pp_stream_ = backend->create_stream();
    }
#endif
}

bool NCCLBridge::all_reduce(std::shared_ptr<Tensor> tensor) {
    if (available_ && comm_ && tensor->device.type == DeviceType::GPU) {
        size_t count = tensor->numel();
        int datatype = map_dtype(tensor->dtype);
        if (datatype < 0) return false;

        void* gpu_ptr = get_gpu_raw_ptr(tensor);
        if (!gpu_ptr) return false;

        typedef int (*all_reduce_t)(const void*, void*, size_t, int, int, void*, void*);
        ((all_reduce_t)all_reduce_fn)(gpu_ptr, gpu_ptr, count, datatype, 0, comm_, comm_stream_);
        return true;
    }
    return false;
}

bool NCCLBridge::tp_all_reduce(std::shared_ptr<Tensor> tensor) {
    void* c = tp_comm_ ? tp_comm_ : comm_;
    void* s = tp_stream_ ? tp_stream_ : comm_stream_;
    if (available_ && c && tensor->device.type == DeviceType::GPU) {
        size_t count = tensor->numel();
        int datatype = map_dtype(tensor->dtype);
        if (datatype < 0) return false;
        void* gpu_ptr = get_gpu_raw_ptr(tensor);
        if (!gpu_ptr) return false;
        typedef int (*all_reduce_t)(const void*, void*, size_t, int, int, void*, void*);
        ((all_reduce_t)all_reduce_fn)(gpu_ptr, gpu_ptr, count, datatype, 0, c, s);
        return true;
    }
    return false;
}

bool NCCLBridge::tp_all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    void* c = tp_comm_ ? tp_comm_ : comm_;
    void* s = tp_stream_ ? tp_stream_ : comm_stream_;
    if (available_ && c && shard->device.type == DeviceType::GPU) {
        size_t sendcount = shard->numel();
        int datatype = map_dtype(shard->dtype);
        if (datatype < 0) return false;
        void* shard_ptr = get_gpu_raw_ptr(shard);
        void* full_ptr = get_gpu_raw_ptr(full);
        if (!shard_ptr || !full_ptr) return false;
        typedef int (*all_gather_t)(const void*, void*, size_t, int, void*, void*);
        ((all_gather_t)all_gather_fn)(shard_ptr, full_ptr, sendcount, datatype, c, s);
        return true;
    }
    return false;
}

bool NCCLBridge::dp_reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    void* c = dp_comm_ ? dp_comm_ : comm_;
    void* s = dp_stream_ ? dp_stream_ : comm_stream_;
    if (available_ && c && reduce_scatter_fn && shard->device.type == DeviceType::GPU) {
        size_t sendcount = shard->numel();
        int datatype = map_dtype(shard->dtype);
        if (datatype < 0) return false;
        void* shard_ptr = get_gpu_raw_ptr(shard);
        void* full_ptr = get_gpu_raw_ptr(full);
        if (!shard_ptr || !full_ptr) return false;
        typedef int (*reduce_scatter_t)(const void*, void*, size_t, int, int, void*, void*);
        ((reduce_scatter_t)reduce_scatter_fn)(full_ptr, shard_ptr, sendcount, datatype, 0, c, s);
        return true;
    }
    return false;
}

bool NCCLBridge::dp_all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    void* c = dp_comm_ ? dp_comm_ : comm_;
    void* s = dp_stream_ ? dp_stream_ : comm_stream_;
    if (available_ && c && shard->device.type == DeviceType::GPU) {
        size_t sendcount = shard->numel();
        int datatype = map_dtype(shard->dtype);
        if (datatype < 0) return false;
        void* shard_ptr = get_gpu_raw_ptr(shard);
        void* full_ptr = get_gpu_raw_ptr(full);
        if (!shard_ptr || !full_ptr) return false;
        typedef int (*all_gather_t)(const void*, void*, size_t, int, void*, void*);
        ((all_gather_t)all_gather_fn)(shard_ptr, full_ptr, sendcount, datatype, c, s);
        return true;
    }
    return false;
}

bool NCCLBridge::broadcast(std::shared_ptr<Tensor> tensor, int src) {
    if (available_ && comm_ && tensor->device.type == DeviceType::GPU) {
        size_t count = tensor->numel();
        int datatype = map_dtype(tensor->dtype);
        if (datatype < 0) return false;

        void* gpu_ptr = get_gpu_raw_ptr(tensor);
        if (!gpu_ptr) return false;

        typedef int (*broadcast_t)(const void*, void*, size_t, int, int, void*, void*);
        ((broadcast_t)broadcast_fn)(gpu_ptr, gpu_ptr, count, datatype, src, comm_, comm_stream_);
        return true;
    }
    return false;
}

bool NCCLBridge::all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    if (available_ && comm_ && shard->device.type == DeviceType::GPU) {
        size_t sendcount = shard->numel();
        int datatype = map_dtype(shard->dtype);
        if (datatype < 0) return false;

        void* shard_ptr = get_gpu_raw_ptr(shard);
        void* full_ptr = get_gpu_raw_ptr(full);
        if (!shard_ptr || !full_ptr) return false;

        typedef int (*all_gather_t)(const void*, void*, size_t, int, void*, void*);
        ((all_gather_t)all_gather_fn)(shard_ptr, full_ptr, sendcount, datatype, comm_, comm_stream_);
        return true;
    }
    return false;
}

bool NCCLBridge::reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    if (available_ && initialized_ && reduce_scatter_fn && shard->device.type == DeviceType::GPU) {
        size_t sendcount = shard->numel();
        int datatype = map_dtype(shard->dtype);
        if (datatype < 0) return false;

        void* shard_ptr = get_gpu_raw_ptr(shard);
        void* full_ptr = get_gpu_raw_ptr(full);
        if (!shard_ptr || !full_ptr) return false;

        typedef int (*reduce_scatter_t)(const void*, void*, size_t, int, int, void*, void*);
        ((reduce_scatter_t)reduce_scatter_fn)(full_ptr, shard_ptr, sendcount, datatype, 0, comm_, comm_stream_);
        return true;
    }
    return false;
}

void NCCLBridge::group_start() {
    if (available_ && initialized_ && group_start_fn) {
        typedef int (*group_start_t)();
        ((group_start_t)group_start_fn)();
    }
}

void NCCLBridge::group_end() {
    if (available_ && initialized_ && group_end_fn) {
        typedef int (*group_end_t)();
        ((group_end_t)group_end_fn)();
    }
}

void NCCLBridge::sync_comm() {
    if (available_ && initialized_ && comm_stream_) {
        if (BackendDispatcher::get().get_backend() && BackendDispatcher::get().get_backend()->is_available()) {
            BackendDispatcher::get().get_backend()->sync_stream(comm_stream_);
        }
    }
}

void NCCLBridge::shutdown() {
    if (comm_ && comm_destroy_fn) {
        typedef int (*destroy_t)(void*);
        ((destroy_t)comm_destroy_fn)(comm_);
        comm_ = nullptr;
    }
    if (tp_comm_ && comm_destroy_fn) {
        typedef int (*destroy_t)(void*);
        ((destroy_t)comm_destroy_fn)(tp_comm_);
        tp_comm_ = nullptr;
    }
    if (dp_comm_ && comm_destroy_fn) {
        typedef int (*destroy_t)(void*);
        ((destroy_t)comm_destroy_fn)(dp_comm_);
        dp_comm_ = nullptr;
    }
    if (pp_comm_ && comm_destroy_fn) {
        typedef int (*destroy_t)(void*);
        ((destroy_t)comm_destroy_fn)(pp_comm_);
        pp_comm_ = nullptr;
    }
    if (lib_handle_) {
        dlclose(lib_handle_);
        lib_handle_ = nullptr;
    }
    available_ = false;
    initialized_ = false;
}

int NCCLBridge::map_dtype(DataType dtype) {
    if (dtype == DataType::FP32) return 7;
    if (dtype == DataType::FP16) return 6;
    if (dtype == DataType::BF16) return 9;
    return -1;
}

void* NCCLBridge::get_gpu_raw_ptr(std::shared_ptr<Tensor> t) {
    if (t->device.type != DeviceType::GPU) return nullptr;
    return (void*)t->gpu_data();
}

OverlappedAllReducer& OverlappedAllReducer::get() {
    static OverlappedAllReducer instance;
    return instance;
}

OverlappedAllReducer::OverlappedAllReducer() {
    auto backend = BackendDispatcher::get().get_backend();
    if (backend && backend->is_available()) {
        ready_event_ = backend->create_event();
        done_event_ = backend->create_event();
    }
}

OverlappedAllReducer::~OverlappedAllReducer() {
    auto backend = BackendDispatcher::get().get_backend();
    if (backend && backend->is_available()) {
        if (ready_event_) backend->destroy_event(ready_event_);
        if (done_event_) backend->destroy_event(done_event_);
    }
}

void OverlappedAllReducer::push_and_all_reduce(std::shared_ptr<Tensor> tensor) {
    auto backend = BackendDispatcher::get().get_backend();
    if (backend && backend->is_available() && NCCLBridge::get().is_available() && tensor->device.type == DeviceType::GPU) {
        void* compute_stream = backend->get_compute_stream();
        void* comm_stream = backend->get_comm_stream();
        backend->record_event(ready_event_, compute_stream);
        backend->stream_wait_event(comm_stream, ready_event_);
        NCCLBridge::get().all_reduce(tensor);
        backend->record_event(done_event_, comm_stream);
    } else {
        ProcessGroup::get().all_reduce(tensor);
    }
}

void OverlappedAllReducer::push_and_reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    auto backend = BackendDispatcher::get().get_backend();
    if (backend && backend->is_available() && NCCLBridge::get().is_available() && shard->device.type == DeviceType::GPU) {
        void* compute_stream = backend->get_compute_stream();
        void* comm_stream = backend->get_comm_stream();
        backend->record_event(ready_event_, compute_stream);
        backend->stream_wait_event(comm_stream, ready_event_);
        NCCLBridge::get().reduce_scatter(shard, full);
        backend->record_event(done_event_, comm_stream);
    } else {
        ProcessGroup::get().reduce_scatter(shard, full);
        ProcessGroup::get().sync_comm();
    }
}

void OverlappedAllReducer::sync() {
    auto backend = BackendDispatcher::get().get_backend();
    if (backend && backend->is_available() && done_event_) {
        void* compute_stream = backend->get_compute_stream();
        backend->stream_wait_event(compute_stream, done_event_);
    }
}

void overlapped_all_reduce(std::shared_ptr<Tensor> tensor) {
    OverlappedAllReducer::get().push_and_all_reduce(tensor);
}

void overlapped_reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    OverlappedAllReducer::get().push_and_reduce_scatter(shard, full);
}

void sync_overlapped_all_reduce() {
    OverlappedAllReducer::get().sync();
}

void all_reduce_bucketed(const std::vector<std::shared_ptr<Tensor>>& tensors, size_t bucket_size_bytes) {
    if (tensors.empty()) return;
    std::vector<std::shared_ptr<Tensor>> current_bucket;
    size_t current_bucket_bytes = 0;

    auto flush_bucket = [](const std::vector<std::shared_ptr<Tensor>>& bucket) {
        if (bucket.empty()) return;
        if (bucket.size() == 1) {
            if (NCCLBridge::get().is_available() && bucket[0]->device.type == DeviceType::GPU) {
                OverlappedAllReducer::get().push_and_all_reduce(bucket[0]);
                OverlappedAllReducer::get().sync();
            } else {
                ProcessGroup::get().all_reduce(bucket[0]);
            }
            return;
        }
        size_t total_elements = 0;
        for (const auto& t : bucket) {
            total_elements += t->numel();
        }
        auto flat_buffer = Tensor::create({static_cast<int64_t>(total_elements)}, bucket[0]->device, false, bucket[0]->dtype);
        size_t offset = 0;
        for (const auto& t : bucket) {
            size_t n = t->numel();
            if (t->device.type == DeviceType::CPU) {
                t->storage->ensure_cpu();
                flat_buffer->storage->ensure_cpu();
                std::memcpy(flat_buffer->data_ptr() + offset, t->data_ptr(), n * t->storage->element_size());
            } else {
                auto backend = BackendDispatcher::get().get_backend();
                if (backend && backend->is_available()) {
                    backend->copy(t->gpu_data(), flat_buffer->gpu_data(), n * t->storage->element_size(), 0, offset * t->storage->element_size());
                } else if (CLBackend::get().is_available()) {
                    CLBackend::get().copy(t->gpu_data(), flat_buffer->gpu_data(), n * t->storage->element_size(), 0, offset * t->storage->element_size());
                }
            }
            offset += n;
        }

        if (NCCLBridge::get().is_available() && flat_buffer->device.type == DeviceType::GPU) {
            OverlappedAllReducer::get().push_and_all_reduce(flat_buffer);
            OverlappedAllReducer::get().sync();
        } else {
            ProcessGroup::get().all_reduce(flat_buffer);
            ProcessGroup::get().sync_comm();
        }

        offset = 0;
        for (const auto& t : bucket) {
            size_t n = t->numel();
            if (t->device.type == DeviceType::CPU) {
                t->storage->ensure_cpu();
                flat_buffer->storage->ensure_cpu();
                std::memcpy(t->data_ptr(), flat_buffer->data_ptr() + offset, n * t->storage->element_size());
            } else {
                auto backend = BackendDispatcher::get().get_backend();
                if (backend && backend->is_available()) {
                    backend->copy(flat_buffer->gpu_data(), t->gpu_data(), n * t->storage->element_size(), offset * t->storage->element_size(), 0);
                } else if (CLBackend::get().is_available()) {
                    CLBackend::get().copy(flat_buffer->gpu_data(), t->gpu_data(), n * t->storage->element_size(), offset * t->storage->element_size(), 0);
                }
            }
            offset += n;
        }
    };

    for (auto& t : tensors) {
        if (!t) continue;
        size_t tensor_bytes = t->numel() * t->storage->element_size();
        if (current_bucket_bytes + tensor_bytes > bucket_size_bytes && !current_bucket.empty()) {
            flush_bucket(current_bucket);
            current_bucket.clear();
            current_bucket_bytes = 0;
        }
        current_bucket.push_back(t);
        current_bucket_bytes += tensor_bytes;
    }
    if (!current_bucket.empty()) {
        flush_bucket(current_bucket);
    }
}

}
}
