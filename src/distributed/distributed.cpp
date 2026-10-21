#include "litetorch/distributed.h"
#include "litetorch/backend.h"
#include "litetorch/backend_route.h"
#include "litetorch/platform.h"
#include "litetorch/ops.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <chrono>
#include <thread>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <tuple>

#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#endif

namespace litetorch {
namespace distributed {

static constexpr int32_t kDirectMagic = 0x4C54444D;
static constexpr int32_t kRelayMagic = 0x4C54524D;
static constexpr int32_t kAgreeMagic = 0x4C544147;

void ProcessGroup::relay_send_all(int fd, const char* buf, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        ssize_t s = send(fd, buf + sent, n - sent, 0);
        if (s <= 0) throw std::runtime_error("Socket write error in relay");
        sent += s;
    }
}

size_t ProcessGroup::relay_dtype_bytes(int32_t dtype_val) {
    switch (static_cast<DataType>(dtype_val)) {
        case DataType::FP64: return 8;
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::BF16: return 2;
        case DataType::INT16: return 2;
        case DataType::INT8: return 1;
        case DataType::INT4: return 1;
        case DataType::FP8_E4M3: return 1;
        case DataType::FP8_E5M2: return 1;
        case DataType::NF4: return 1;
        case DataType::FP4_E2M1: return 1;
        case DataType::INT32: return 4;
        default: return 4;
    }
}

void ProcessGroup::relay_loop() {
    while (!stop_relay_) {
        std::vector<struct pollfd> pfds;
        std::vector<int> pfd_ranks;
        pfds.reserve((size_t)(world_size_ > 1 ? world_size_ - 1 : 0));
        pfd_ranks.reserve((size_t)(world_size_ > 1 ? world_size_ - 1 : 0));
        for (int i = 1; i < world_size_; ++i) {
            int fd = (i < (int)client_fds_.size()) ? client_fds_[i] : -1;
            if (fd >= 0) {
                struct pollfd p;
                p.fd = fd;
                p.events = POLLIN;
                p.revents = 0;
                pfds.push_back(p);
                pfd_ranks.push_back(i);
            }
        }
        if (pfds.empty()) {
            usleep(100000);
            continue;
        }
        int n = poll(pfds.data(), pfds.size(), 100);
        if (n <= 0) continue;
        bool agree_wait = false;
        {
            std::lock_guard<std::mutex> raw_lock(raw_socket_mutex_);
            for (size_t k = 0; k < pfds.size(); ++k) {
                if (!(pfds[k].revents & (POLLIN | POLLERR | POLLHUP))) continue;
                int i = pfd_ranks[k];
                int fd = pfds[k].fd;
                int32_t magic = 0;
                ssize_t pr = recv(fd, &magic, sizeof(magic), MSG_PEEK);
                if (pr != sizeof(magic)) continue;
                if (magic == kAgreeMagic) {
                    agree_wait = true;
                    continue;
                }
                ssize_t r = recv(fd, &magic, sizeof(magic), MSG_WAITALL);
                if (r != sizeof(magic)) continue;
            bool is_relay = (magic == kRelayMagic);
            if (!is_relay && magic != kDirectMagic) continue;
            int32_t dst = 0;
            if (is_relay) {
                if (recv(fd, &dst, sizeof(dst), MSG_WAITALL) != sizeof(dst)) continue;
            }
            RelayMsg msg;
            msg.src = i;
            if (recv(fd, &msg.ndim, sizeof(msg.ndim), MSG_WAITALL) != sizeof(msg.ndim)) continue;
            if (recv(fd, &msg.dtype_val, sizeof(msg.dtype_val), MSG_WAITALL) != sizeof(msg.dtype_val)) continue;
            if (msg.ndim < 0 || msg.ndim > 8) continue;
            msg.shape.resize(msg.ndim);
            if (msg.ndim > 0) {
                if (recv(fd, msg.shape.data(), msg.ndim * sizeof(int64_t), MSG_WAITALL) != (ssize_t)(msg.ndim * sizeof(int64_t))) continue;
            }
            size_t N = 1;
            for (auto s : msg.shape) {
                if (s < 0) { N = 0; break; }
                N *= (size_t)s;
            }
            size_t total = N * relay_dtype_bytes(msg.dtype_val);
            msg.data.resize(total);
            size_t recvd = 0;
            bool ok = true;
            while (recvd < total) {
                ssize_t rr = recv(fd, msg.data.data() + recvd, total - recvd, 0);
                if (rr <= 0) { ok = false; break; }
                recvd += rr;
            }
            if (!ok) continue;
            if (is_relay) {
                if (dst < 1 || dst >= world_size_) continue;
                std::lock_guard<std::mutex> lock(relay_send_mutex_);
                int dfd = (dst < (int)client_fds_.size()) ? client_fds_[dst] : -1;
                if (dfd < 0) continue;
                try {
                    int32_t dm = kDirectMagic;
                    relay_send_all(dfd, (const char*)&dm, sizeof(dm));
                    relay_send_all(dfd, (const char*)&msg.ndim, sizeof(msg.ndim));
                    relay_send_all(dfd, (const char*)&msg.dtype_val, sizeof(msg.dtype_val));
                    if (msg.ndim > 0) relay_send_all(dfd, (const char*)msg.shape.data(), msg.ndim * sizeof(int64_t));
                    if (total > 0) relay_send_all(dfd, msg.data.data(), total);
                } catch (...) {}
            } else {
                std::lock_guard<std::mutex> lock(relay_inbox_mutex_);
                relay_inbox_.push_back(std::move(msg));
                relay_inbox_cv_.notify_one();
            }
        }
        }
        if (agree_wait) usleep(1000);
    }
}

ProcessGroup& ProcessGroup::get() {
    static ProcessGroup instance;
    return instance;
}

ProcessGroup::~ProcessGroup() {
    shutdown();
}

#ifdef _WIN32
void ProcessGroup::init(int rank, int world_size, const std::string&, int) {
    (void)rank; (void)world_size;
    throw std::runtime_error("LiteTorch distributed is not implemented on Windows");
}
void ProcessGroup::shutdown() { initialized_ = false; }
void ProcessGroup::all_reduce(std::shared_ptr<Tensor>) {
    throw std::runtime_error("LiteTorch distributed is not implemented on Windows");
}
std::future<void> ProcessGroup::all_reduce_async(std::shared_ptr<Tensor>) {
    throw std::runtime_error("LiteTorch distributed is not implemented on Windows");
}
void ProcessGroup::broadcast(std::shared_ptr<Tensor>, int) {
    throw std::runtime_error("LiteTorch distributed is not implemented on Windows");
}
void ProcessGroup::all_gather(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>) {
    throw std::runtime_error("LiteTorch distributed is not implemented on Windows");
}
void ProcessGroup::reduce_scatter(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>) {
    throw std::runtime_error("LiteTorch distributed is not implemented on Windows");
}
void ProcessGroup::send_tensor(std::shared_ptr<Tensor>, int) {
    throw std::runtime_error("LiteTorch distributed is not implemented on Windows");
}
void ProcessGroup::recv_tensor(std::shared_ptr<Tensor>, int) {
    throw std::runtime_error("LiteTorch distributed is not implemented on Windows");
}
void ProcessGroup::sync_comm() {
    throw std::runtime_error("LiteTorch distributed is not implemented on Windows");
}
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
            if (client_rank < 1 || client_rank >= world_size_) {
                close(client_fd);
                shutdown();
                throw std::runtime_error("Invalid client rank received");
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
                std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<std::promise<void>>, ReduceOp> task;
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
                    all_reduce(std::get<0>(task), std::get<2>(task));
                    std::get<1>(task)->set_value();
                } catch (...) {
                    std::get<1>(task)->set_exception(std::current_exception());
                }
            }
        });
    }

    if (rank_ == 0 && world_size_ > 1 && !use_shm_) {
        stop_relay_ = false;
        relay_thread_ = std::thread([this]() { relay_loop(); });
    }

    initialized_ = true;
}

void ProcessGroup::shutdown() {
    if (!initialized_) return;

    if (relay_thread_.joinable()) {
        stop_relay_ = true;
        relay_thread_.join();
    }
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

bool ProcessGroup::socket_recv_all(int fd, char* buf, size_t n) {
    size_t recvd = 0;
    while (recvd < n) {
        ssize_t r = recv(fd, buf + recvd, n - recvd, 0);
        if (r <= 0) return false;
        recvd += (size_t)r;
    }
    return true;
}

void ProcessGroup::check_collective_agreement(int32_t dtype_val, const std::vector<int64_t>& shape) {
    if (!initialized_ || world_size_ <= 1) return;
    int32_t ndim = static_cast<int32_t>(shape.size());
    if (rank_ == 0) {
        bool agree = true;
        for (int i = 1; i < world_size_; ++i) {
            int fd = (i < (int)client_fds_.size()) ? client_fds_[i] : -1;
            int32_t magic = 0;
            int32_t hdr[2] = {0, -1};
            bool ok = fd >= 0 && socket_recv_all(fd, reinterpret_cast<char*>(&magic), sizeof(magic));
            ok = ok && magic == kAgreeMagic && socket_recv_all(fd, reinterpret_cast<char*>(hdr), sizeof(hdr));
            std::vector<int64_t> rshape;
            if (ok && hdr[1] >= 0 && hdr[1] <= 8) {
                rshape.resize((size_t)hdr[1]);
                ok = socket_recv_all(fd, reinterpret_cast<char*>(rshape.data()), rshape.size() * sizeof(int64_t));
            } else {
                ok = false;
            }
            if (!ok) { agree = false; break; }
            if (hdr[0] != dtype_val || rshape != shape) agree = false;
        }
        int32_t status = agree ? 0 : 1;
        for (int i = 1; i < world_size_; ++i) {
            int fd = (i < (int)client_fds_.size()) ? client_fds_[i] : -1;
            if (fd < 0) continue;
            try {
                relay_send_all(fd, reinterpret_cast<const char*>(&status), sizeof(status));
            } catch (...) {}
        }
        if (!agree) throw std::runtime_error("collective tensor dtype/shape mismatch across ranks");
    } else {
        int32_t magic = kAgreeMagic;
        int32_t hdr[2] = {dtype_val, ndim};
        relay_send_all(peer_fd_, reinterpret_cast<const char*>(&magic), sizeof(magic));
        relay_send_all(peer_fd_, reinterpret_cast<const char*>(hdr), sizeof(hdr));
        if (ndim > 0) relay_send_all(peer_fd_, reinterpret_cast<const char*>(shape.data()), shape.size() * sizeof(int64_t));
        int32_t status = -1;
        if (!socket_recv_all(peer_fd_, reinterpret_cast<char*>(&status), sizeof(status)) || status != 0)
            throw std::runtime_error("collective tensor dtype/shape mismatch across ranks");
    }
}

void ProcessGroup::all_reduce(std::shared_ptr<Tensor> tensor) {
    if (!initialized_ || world_size_ <= 1) return;
    std::lock_guard<std::mutex> coll_lock(raw_socket_mutex_);

    check_collective_agreement(static_cast<int32_t>(tensor->dtype), tensor->shape);

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
        rt_gpu_read(tensor->gpu_data(), N * elem_sz, data.data(), tensor->offset * elem_sz);
    } else {
        std::memcpy(data.data(), tensor->raw_data_ptr(), N * elem_sz);
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
        } else if (tensor->dtype == DataType::INT32) {
            std::vector<int32_t> acc(N, 0);
            int32_t* data_ptr = reinterpret_cast<int32_t*>(data.data());
            for (size_t j = 0; j < N; ++j) acc[j] += data_ptr[j];

            std::vector<char> temp(N * elem_sz);
            int32_t* temp_ptr = reinterpret_cast<int32_t*>(temp.data());
            for (int i = 1; i < world_size_; ++i) {
                size_t bytes_received = 0;
                while (bytes_received < N * elem_sz) {
                    ssize_t r = recv(client_fds_[i], temp.data() + bytes_received, N * elem_sz - bytes_received, 0);
                    if (r <= 0) throw std::runtime_error("Socket read error in all_reduce");
                    bytes_received += r;
                }
                for (size_t j = 0; j < N; ++j) acc[j] += temp_ptr[j];
            }

            for (size_t j = 0; j < N; ++j) data_ptr[j] = acc[j];

            for (int i = 1; i < world_size_; ++i) {
                size_t bytes_sent = 0;
                while (bytes_sent < N * elem_sz) {
                    ssize_t s = send(client_fds_[i], data.data() + bytes_sent, N * elem_sz - bytes_sent, 0);
                    if (s <= 0) throw std::runtime_error("Socket write error in all_reduce");
                    bytes_sent += s;
                }
            }
        } else if (tensor->dtype == DataType::INT8 || tensor->dtype == DataType::INT16) {
            std::vector<int32_t> acc(N, 0);
            size_t isz = (tensor->dtype == DataType::INT8) ? 1 : 2;
            for (size_t j = 0; j < N; ++j) {
                int32_t v = 0;
                if (isz == 1) v = reinterpret_cast<int8_t*>(data.data())[j];
                else v = reinterpret_cast<int16_t*>(data.data())[j];
                acc[j] += v;
            }

            std::vector<char> temp(N * elem_sz);
            for (int i = 1; i < world_size_; ++i) {
                size_t bytes_received = 0;
                while (bytes_received < N * elem_sz) {
                    ssize_t r = recv(client_fds_[i], temp.data() + bytes_received, N * elem_sz - bytes_received, 0);
                    if (r <= 0) throw std::runtime_error("Socket read error in all_reduce");
                    bytes_received += r;
                }
                for (size_t j = 0; j < N; ++j) {
                    int32_t v = 0;
                    if (isz == 1) v = reinterpret_cast<int8_t*>(temp.data())[j];
                    else v = reinterpret_cast<int16_t*>(temp.data())[j];
                    acc[j] += v;
                }
            }

            for (size_t j = 0; j < N; ++j) {
                if (isz == 1) reinterpret_cast<int8_t*>(data.data())[j] = static_cast<int8_t>(acc[j]);
                else reinterpret_cast<int16_t*>(data.data())[j] = static_cast<int16_t>(acc[j]);
            }

            for (int i = 1; i < world_size_; ++i) {
                size_t bytes_sent = 0;
                while (bytes_sent < N * elem_sz) {
                    ssize_t s = send(client_fds_[i], data.data() + bytes_sent, N * elem_sz - bytes_sent, 0);
                    if (s <= 0) throw std::runtime_error("Socket write error in all_reduce");
                    bytes_sent += s;
                }
            }
        } else if (tensor->dtype == DataType::FP32) {
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

            for (size_t j = 0; j < N; ++j) data_ptr[j] = acc[j];

            for (int i = 1; i < world_size_; ++i) {
                size_t bytes_sent = 0;
                while (bytes_sent < N * elem_sz) {
                    ssize_t s = send(client_fds_[i], data.data() + bytes_sent, N * elem_sz - bytes_sent, 0);
                    if (s <= 0) throw std::runtime_error("Socket write error in all_reduce");
                    bytes_sent += s;
                }
            }
        } else {
            throw std::runtime_error("all_reduce socket fallback: unsupported dtype for CPU reduction");
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
        rt_gpu_write(tensor->gpu_data(), N * elem_sz, data.data(), tensor->offset * elem_sz);
    } else {
        std::memcpy(tensor->raw_data_ptr(), data.data(), N * elem_sz);
    }
}

void ProcessGroup::broadcast(std::shared_ptr<Tensor> tensor, int src) {
    if (!initialized_ || world_size_ <= 1) return;
    std::lock_guard<std::mutex> coll_lock(raw_socket_mutex_);

    check_collective_agreement(static_cast<int32_t>(tensor->dtype), tensor->shape);

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
            rt_gpu_read(tensor->gpu_data(), N * elem_sz, data.data(), tensor->offset * elem_sz);
        } else {
            std::memcpy(data.data(), tensor->raw_data_ptr(), N * elem_sz);
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
            rt_gpu_write(tensor->gpu_data(), N * elem_sz, data.data(), tensor->offset * elem_sz);
        } else {
            std::memcpy(tensor->raw_data_ptr(), data.data(), N * elem_sz);
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
    return all_reduce_async(tensor, ReduceOp::SUM);
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
    std::lock_guard<std::mutex> coll_lock(raw_socket_mutex_);

    check_collective_agreement(static_cast<int32_t>(shard->dtype), shard->shape);

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
        rt_gpu_read(shard->gpu_data(), S * elem_sz, data.data(), shard->offset * elem_sz);
    } else {
        std::memcpy(data.data(), shard->raw_data_ptr(), S * elem_sz);
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
        rt_gpu_write(full->gpu_data(), S * world_size_ * elem_sz, full_cpu.data(), full->offset * elem_sz);
    } else {
        std::memcpy(full->raw_data_ptr(), full_cpu.data(), S * world_size_ * elem_sz);
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
        std::lock_guard<std::mutex> lock(shm_step_mutex_);
        char* base = reinterpret_cast<char*>(shm_buffers_[rank_]);
        size_t hdr_bytes = 8 + 8 * (size_t)ndim;
        if (hdr_bytes + N * elem_sz > shm_buffer_size_) {
            int32_t neg = -1;
            std::memcpy(base, &neg, sizeof(neg));
            shm_ctrl_->p2p_steps[rank_].store(shm_ctrl_->p2p_steps[rank_].load() + 1);
            throw std::runtime_error("send_tensor: tensor exceeds shm buffer [dist-w6]");
        }
        int cur_step = shm_ctrl_->p2p_steps[rank_].load();
        while (shm_ctrl_->p2p_steps[dst].load() < cur_step) {
            usleep(1);
        }
        int32_t ndim32 = (int32_t)ndim;
        int32_t dt32 = (int32_t)dtype_val;
        std::memcpy(base, &ndim32, 4);
        std::memcpy(base + 4, &dt32, 4);
        for (int i = 0; i < ndim; ++i) {
            int64_t d = tensor->shape[i];
            std::memcpy(base + 8 + 8 * i, &d, 8);
        }
        char* dp = base + hdr_bytes;

        if (tensor->device.type == DeviceType::GPU) {
            rt_gpu_read(tensor->gpu_data(), N * elem_sz, dp, tensor->offset * elem_sz);
        } else {
            std::memcpy(dp, tensor->raw_data_ptr(), N * elem_sz);
        }

        shm_ctrl_->p2p_steps[rank_].store(cur_step + 1);
    } else {
        std::vector<char> temp_buf(N * elem_sz);
        if (tensor->device.type == DeviceType::GPU) {
            rt_gpu_read(tensor->gpu_data(), N * elem_sz, temp_buf.data(), tensor->offset * elem_sz);
        } else {
            std::memcpy(temp_buf.data(), tensor->raw_data_ptr(), N * elem_sz);
        }
        int32_t dmagic = kDirectMagic;
        int32_t snd_ndim = ndim;
        int32_t snd_dtype = dtype_val;
        if (rank_ == 0) {
            std::lock_guard<std::mutex> lock(relay_send_mutex_);
            int fd = (dst < (int)client_fds_.size()) ? client_fds_[dst] : -1;
            if (fd < 0) throw std::runtime_error("send_tensor: no socket to rank " + std::to_string(dst));
            relay_send_all(fd, (const char*)&dmagic, sizeof(dmagic));
            relay_send_all(fd, (const char*)&snd_ndim, sizeof(snd_ndim));
            relay_send_all(fd, (const char*)&snd_dtype, sizeof(snd_dtype));
            if (ndim > 0) relay_send_all(fd, (const char*)tensor->shape.data(), ndim * sizeof(int64_t));
            if (N * elem_sz > 0) relay_send_all(fd, temp_buf.data(), N * elem_sz);
        } else if (dst == 0) {
            relay_send_all(peer_fd_, (const char*)&dmagic, sizeof(dmagic));
            relay_send_all(peer_fd_, (const char*)&snd_ndim, sizeof(snd_ndim));
            relay_send_all(peer_fd_, (const char*)&snd_dtype, sizeof(snd_dtype));
            if (ndim > 0) relay_send_all(peer_fd_, (const char*)tensor->shape.data(), ndim * sizeof(int64_t));
            if (N * elem_sz > 0) relay_send_all(peer_fd_, temp_buf.data(), N * elem_sz);
        } else {
            int32_t rmagic = kRelayMagic;
            int32_t rdst = dst;
            relay_send_all(peer_fd_, (const char*)&rmagic, sizeof(rmagic));
            relay_send_all(peer_fd_, (const char*)&rdst, sizeof(rdst));
            relay_send_all(peer_fd_, (const char*)&snd_ndim, sizeof(snd_ndim));
            relay_send_all(peer_fd_, (const char*)&snd_dtype, sizeof(snd_dtype));
            if (ndim > 0) relay_send_all(peer_fd_, (const char*)tensor->shape.data(), ndim * sizeof(int64_t));
            if (N * elem_sz > 0) relay_send_all(peer_fd_, temp_buf.data(), N * elem_sz);
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
        std::lock_guard<std::mutex> lock(shm_step_mutex_);
        int cur_step = shm_ctrl_->p2p_steps[rank_].load();
        while (shm_ctrl_->p2p_steps[src].load() < cur_step + 1) {
            usleep(1);
        }
        const char* base = reinterpret_cast<const char*>(shm_buffers_[src]);
        int32_t ndim32 = 0;
        int32_t dt32 = 0;
        std::memcpy(&ndim32, base, 4);
        std::memcpy(&dt32, base + 4, 4);
        if (ndim32 < 0) throw std::runtime_error("recv_tensor: peer reported send error [dist-w6]");
        if (ndim32 > 32) throw std::runtime_error("recv_tensor: corrupt shm header [dist-w6]");
        ndim = ndim32;
        dtype_val = dt32;
        shape.resize(ndim);
        size_t N = 1;
        for (int i = 0; i < ndim; ++i) {
            int64_t d = 0;
            std::memcpy(&d, base + 8 + 8 * i, 8);
            if (d < 0 || d > 16777216) throw std::runtime_error("recv_tensor: corrupt shm header [dist-w6]");
            shape[i] = d;
            N *= (size_t)d;
        }
        DataType dtype = static_cast<DataType>(dtype_val);

        tensor->shape = shape;
        tensor->strides = default_strides(shape);
        tensor->numel_ = N;
        tensor->contiguous_ = true;
        tensor->dtype = dtype;
        tensor->storage = std::make_shared<StorageImpl>(N, tensor->device, dtype);
        size_t elem_sz = tensor->storage->element_size();
        const char* dp = base + 8 + 8 * (size_t)ndim;

        if (tensor->device.type == DeviceType::GPU) {
            rt_gpu_write(tensor->gpu_data(), N * elem_sz, dp, 0);
        } else {
            std::memcpy(tensor->raw_data_ptr(), dp, N * elem_sz);
        }

        shm_ctrl_->p2p_steps[rank_].store(cur_step + 1);
    } else {
        RelayMsg msg;
        bool have_msg = false;
        if (rank_ == 0) {
            std::unique_lock<std::mutex> lock(relay_inbox_mutex_);
            relay_inbox_cv_.wait(lock, [this, src]() {
                if (stop_relay_) return true;
                for (auto& m : relay_inbox_) {
                    if (m.src == src) return true;
                }
                return false;
            });
            if (stop_relay_) throw std::runtime_error("recv_tensor: shutdown while waiting");
            for (auto it = relay_inbox_.begin(); it != relay_inbox_.end(); ++it) {
                if (it->src == src) {
                    msg = std::move(*it);
                    relay_inbox_.erase(it);
                    have_msg = true;
                    break;
                }
            }
            if (!have_msg) throw std::runtime_error("recv_tensor: inbox lost message");
        } else {
            int32_t magic = 0;
            if (recv(peer_fd_, &magic, sizeof(magic), MSG_WAITALL) != (ssize_t)sizeof(magic))
                throw std::runtime_error("Socket read error in recv_tensor (magic)");
            if (magic != kDirectMagic)
                throw std::runtime_error("recv_tensor: protocol magic mismatch");
            msg.src = src;
            if (recv(peer_fd_, &msg.ndim, sizeof(msg.ndim), MSG_WAITALL) != (ssize_t)sizeof(msg.ndim))
                throw std::runtime_error("Socket read error in recv_tensor");
            if (recv(peer_fd_, &msg.dtype_val, sizeof(msg.dtype_val), MSG_WAITALL) != (ssize_t)sizeof(msg.dtype_val))
                throw std::runtime_error("Socket read error in recv_tensor");
            if (msg.ndim < 0 || msg.ndim > 8) throw std::runtime_error("recv_tensor: bad ndim");
            msg.shape.resize(msg.ndim);
            if (msg.ndim > 0) {
                if (recv(peer_fd_, msg.shape.data(), msg.ndim * sizeof(int64_t), MSG_WAITALL) != (ssize_t)(msg.ndim * sizeof(int64_t)))
                    throw std::runtime_error("Socket read error in recv_tensor");
            }
            size_t N = 1;
            for (auto s : msg.shape) {
                if (s < 0) throw std::runtime_error("recv_tensor: bad shape");
                N *= (size_t)s;
            }
            size_t total = N * relay_dtype_bytes(msg.dtype_val);
            msg.data.resize(total);
            size_t bytes_received = 0;
            while (bytes_received < total) {
                ssize_t r = recv(peer_fd_, msg.data.data() + bytes_received, total - bytes_received, 0);
                if (r <= 0) throw std::runtime_error("Socket read error in recv_tensor");
                bytes_received += r;
            }
            have_msg = true;
        }

        ndim = msg.ndim;
        dtype_val = msg.dtype_val;
        shape = msg.shape;
        DataType dtype = static_cast<DataType>(dtype_val);

        size_t N = 1;
        for (int i = 0; i < ndim; ++i) N *= (size_t)shape[i];

        tensor->shape = shape;
        tensor->strides = default_strides(shape);
        tensor->numel_ = N;
        tensor->contiguous_ = true;
        tensor->dtype = dtype;
        tensor->storage = std::make_shared<StorageImpl>(N, tensor->device, dtype);
        size_t elem_sz = tensor->storage->element_size();

        if (tensor->device.type == DeviceType::GPU) {
            rt_gpu_write(tensor->gpu_data(), N * elem_sz, msg.data.data(), 0);
        } else {
            std::memcpy(tensor->raw_data_ptr(), msg.data.data(), N * elem_sz);
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

struct ncclUniqueId { char internal[128]; };

void NCCLBridge::init(int rank, int world_size) {
    if (initialized_) return;
    world_size_ = world_size;

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
    comm_get_async_error_fn = dlsym(lib_handle_, "ncclCommGetAsyncError");

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

    typedef int (*init_rank_t)(void**, int, ncclUniqueId, int);
    int res = ((init_rank_t)comm_init_rank_fn)(&comm_, world_size, *reinterpret_cast<ncclUniqueId*>(unique_id), rank);
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
        plain_event_ = BackendDispatcher::get().get_backend()->create_event();
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
    int dp_color = rank % (tp_size * pp_size);
    int dp_key = rank / (tp_size * pp_size);
    split(comm_, dp_color, dp_key, &dp_comm_, nullptr);

    int pp_color = (rank / (tp_size * pp_size)) * tp_size + rank % tp_size;
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

static std::shared_ptr<Tensor> nccl_contig_in(std::shared_ptr<Tensor> t);
static void nccl_contig_out(std::shared_ptr<Tensor> orig, std::shared_ptr<Tensor> tmp);

void NCCLBridge::nccl_check(int res, const char* op) {
    if (res != 0) {
        throw std::runtime_error(std::string("nccl ") + op + " failed");
    }
}

void NCCLBridge::order_for_stream(void* stream) {
    auto backend = BackendDispatcher::get().get_backend();
    if (backend && backend->is_available() && plain_event_) {
        backend->record_event(plain_event_, backend->get_compute_stream());
        backend->stream_wait_event(stream ? stream : comm_stream_, plain_event_);
    }
}

bool NCCLBridge::all_reduce(std::shared_ptr<Tensor> tensor) {
    return all_reduce(tensor, ReduceOp::SUM);
}

bool NCCLBridge::all_reduce(std::shared_ptr<Tensor> tensor, ReduceOp op) {
    if (available_ && comm_ && tensor->device.type == DeviceType::GPU) {
        auto work = nccl_contig_in(tensor);
        size_t count = work->numel();
        int datatype = map_dtype(work->dtype);
        if (datatype < 0) return false;

        void* gpu_ptr = get_gpu_raw_ptr(work);
        if (!gpu_ptr) return false;

        order_for_stream(comm_stream_);
        typedef int (*all_reduce_t)(const void*, void*, size_t, int, int, void*, void*);
        nccl_check(((all_reduce_t)all_reduce_fn)(gpu_ptr, gpu_ptr, count, datatype, map_reduce_op(op), comm_, comm_stream_), "all_reduce");
        sync_stream_(comm_stream_);
        if (op == ReduceOp::AVG) {
            if (work->dtype != DataType::FP32 && work->dtype != DataType::FP16 &&
                work->dtype != DataType::BF16 && work->dtype != DataType::FP64) {
                throw std::runtime_error("NCCL all_reduce AVG requires float dtype");
            }
            float inv = 1.0f / (float)world_size_;
            auto denom = Tensor::from_vector(std::vector<float>(count, inv), work->shape, work->device, false, work->dtype);
            auto scaled = Ops::div(work, denom);
            work->copy_(scaled);
        }
        nccl_contig_out(tensor, work);
        return true;
    }
    return false;
}

bool NCCLBridge::tp_all_reduce(std::shared_ptr<Tensor> tensor) {
    void* c = tp_comm_ ? tp_comm_ : comm_;
    void* s = tp_stream_ ? tp_stream_ : comm_stream_;
    if (available_ && c && tensor->device.type == DeviceType::GPU) {
        auto work = nccl_contig_in(tensor);
        size_t count = work->numel();
        int datatype = map_dtype(work->dtype);
        if (datatype < 0) return false;
        void* gpu_ptr = get_gpu_raw_ptr(work);
        if (!gpu_ptr) return false;
        order_for_stream(s);
        typedef int (*all_reduce_t)(const void*, void*, size_t, int, int, void*, void*);
        nccl_check(((all_reduce_t)all_reduce_fn)(gpu_ptr, gpu_ptr, count, datatype, 0, c, s), "tp_all_reduce");
        sync_stream_(s);
        nccl_contig_out(tensor, work);
        return true;
    }
    return false;
}

bool NCCLBridge::tp_all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    void* c = tp_comm_ ? tp_comm_ : comm_;
    void* s = tp_stream_ ? tp_stream_ : comm_stream_;
    if (available_ && c && shard->device.type == DeviceType::GPU) {
        auto shard_c = nccl_contig_in(shard);
        size_t sendcount = shard_c->numel();
        int datatype = map_dtype(shard_c->dtype);
        if (datatype < 0) return false;
        void* shard_ptr = get_gpu_raw_ptr(shard_c);
        if (!shard_ptr) return false;
        auto full_c = nccl_contig_in(full);
        void* full_ptr = get_gpu_raw_ptr(full_c);
        if (!full_ptr) return false;
        order_for_stream(s);
        typedef int (*all_gather_t)(const void*, void*, size_t, int, void*, void*);
        nccl_check(((all_gather_t)all_gather_fn)(shard_ptr, full_ptr, sendcount, datatype, c, s), "tp_all_gather");
        sync_stream_(s);
        nccl_contig_out(full, full_c);
        return true;
    }
    return false;
}

bool NCCLBridge::dp_reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    void* c = dp_comm_ ? dp_comm_ : comm_;
    void* s = dp_stream_ ? dp_stream_ : comm_stream_;
    if (available_ && c && reduce_scatter_fn && shard->device.type == DeviceType::GPU) {
        auto full_c = nccl_contig_in(full);
        size_t sendcount = shard->numel();
        int datatype = map_dtype(shard->dtype);
        if (datatype < 0) return false;
        void* full_ptr = get_gpu_raw_ptr(full_c);
        if (!full_ptr) return false;
        auto shard_c = nccl_contig_in(shard);
        void* shard_ptr = get_gpu_raw_ptr(shard_c);
        if (!shard_ptr) return false;
        order_for_stream(s);
        typedef int (*reduce_scatter_t)(const void*, void*, size_t, int, int, void*, void*);
        nccl_check(((reduce_scatter_t)reduce_scatter_fn)(full_ptr, shard_ptr, sendcount, datatype, 0, c, s), "dp_reduce_scatter");
        sync_stream_(s);
        nccl_contig_out(shard, shard_c);
        return true;
    }
    return false;
}

bool NCCLBridge::dp_all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    void* c = dp_comm_ ? dp_comm_ : comm_;
    void* s = dp_stream_ ? dp_stream_ : comm_stream_;
    if (available_ && c && shard->device.type == DeviceType::GPU) {
        auto shard_c = nccl_contig_in(shard);
        size_t sendcount = shard_c->numel();
        int datatype = map_dtype(shard_c->dtype);
        if (datatype < 0) return false;
        void* shard_ptr = get_gpu_raw_ptr(shard_c);
        if (!shard_ptr) return false;
        auto full_c = nccl_contig_in(full);
        void* full_ptr = get_gpu_raw_ptr(full_c);
        if (!full_ptr) return false;
        order_for_stream(s);
        typedef int (*all_gather_t)(const void*, void*, size_t, int, void*, void*);
        nccl_check(((all_gather_t)all_gather_fn)(shard_ptr, full_ptr, sendcount, datatype, c, s), "dp_all_gather");
        sync_stream_(s);
        nccl_contig_out(full, full_c);
        return true;
    }
    return false;
}

bool NCCLBridge::broadcast(std::shared_ptr<Tensor> tensor, int src) {
    if (available_ && comm_ && tensor->device.type == DeviceType::GPU) {
        auto work = nccl_contig_in(tensor);
        size_t count = work->numel();
        int datatype = map_dtype(work->dtype);
        if (datatype < 0) return false;

        void* gpu_ptr = get_gpu_raw_ptr(work);
        if (!gpu_ptr) return false;

        order_for_stream(comm_stream_);
        typedef int (*broadcast_t)(const void*, void*, size_t, int, int, void*, void*);
        nccl_check(((broadcast_t)broadcast_fn)(gpu_ptr, gpu_ptr, count, datatype, src, comm_, comm_stream_), "broadcast");
        sync_stream_(comm_stream_);
        nccl_contig_out(tensor, work);
        return true;
    }
    return false;
}

bool NCCLBridge::all_gather(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    if (available_ && comm_ && shard->device.type == DeviceType::GPU) {
        auto shard_c = nccl_contig_in(shard);
        size_t sendcount = shard_c->numel();
        int datatype = map_dtype(shard_c->dtype);
        if (datatype < 0) return false;

        void* shard_ptr = get_gpu_raw_ptr(shard_c);
        if (!shard_ptr) return false;
        auto full_c = nccl_contig_in(full);
        void* full_ptr = get_gpu_raw_ptr(full_c);
        if (!full_ptr) return false;

        order_for_stream(comm_stream_);
        typedef int (*all_gather_t)(const void*, void*, size_t, int, void*, void*);
        nccl_check(((all_gather_t)all_gather_fn)(shard_ptr, full_ptr, sendcount, datatype, comm_, comm_stream_), "all_gather");
        sync_stream_(comm_stream_);
        nccl_contig_out(full, full_c);
        return true;
    }
    return false;
}

bool NCCLBridge::reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    return reduce_scatter(shard, full, ReduceOp::SUM);
}

bool NCCLBridge::reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full, ReduceOp op) {
    if (available_ && initialized_ && reduce_scatter_fn && shard->device.type == DeviceType::GPU) {
        auto full_c = nccl_contig_in(full);
        size_t sendcount = shard->numel();
        int datatype = map_dtype(shard->dtype);
        if (datatype < 0) return false;

        void* full_ptr = get_gpu_raw_ptr(full_c);
        if (!full_ptr) return false;
        auto shard_c = nccl_contig_in(shard);
        void* shard_ptr = get_gpu_raw_ptr(shard_c);
        if (!shard_ptr) return false;

        order_for_stream(comm_stream_);
        typedef int (*reduce_scatter_t)(const void*, void*, size_t, int, int, void*, void*);
        nccl_check(((reduce_scatter_t)reduce_scatter_fn)(full_ptr, shard_ptr, sendcount, datatype, map_reduce_op(op), comm_, comm_stream_), "reduce_scatter");
        sync_stream_(comm_stream_);
        if (op == ReduceOp::AVG) {
            if (shard->dtype != DataType::FP32 && shard->dtype != DataType::FP16 &&
                shard->dtype != DataType::BF16 && shard->dtype != DataType::FP64) {
                throw std::runtime_error("NCCL reduce_scatter AVG requires float dtype");
            }
            float inv = 1.0f / (float)world_size_;
            auto shard_c2 = nccl_contig_in(shard);
            auto denom = Tensor::from_vector(std::vector<float>(sendcount, inv), shard_c2->shape, shard_c2->device, false, shard_c2->dtype);
            auto scaled = Ops::div(shard_c2, denom);
            shard_c2->copy_(scaled);
            nccl_contig_out(shard, shard_c2);
            return true;
        }
        nccl_contig_out(shard, shard_c);
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
        if (comm_get_async_error_fn && comm_) {
            typedef int (*async_err_t)(void*, int*);
            int async_err = 0;
            ((async_err_t)comm_get_async_error_fn)(comm_, &async_err);
            if (async_err != 0) throw std::runtime_error("nccl async error detected");
        }
    }
}

void NCCLBridge::shutdown() {
    if (plain_event_) {
        auto backend = BackendDispatcher::get().get_backend();
        if (backend && backend->is_available()) backend->destroy_event(plain_event_);
        plain_event_ = nullptr;
    }
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
    char* base = (char*)t->gpu_data();
    size_t elem_sz = t->storage->element_size();
    return (void*)(base + t->offset * elem_sz);
}

void NCCLBridge::sync_stream_(void* stream) {
    if (available_ && initialized_ && stream) {
        auto backend = BackendDispatcher::get().get_backend();
        if (backend && backend->is_available()) {
            backend->sync_stream(stream);
        }
    }
}

static std::shared_ptr<Tensor> nccl_contig_in(std::shared_ptr<Tensor> t) {
    if (t->is_contiguous()) return t;
    return t->contiguous();
}

static void nccl_contig_out(std::shared_ptr<Tensor> orig, std::shared_ptr<Tensor> tmp) {
    if (tmp.get() != orig.get()) {
        orig->copy_(tmp);
    }
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
    all_reduce_bucketed(tensors, bucket_size_bytes, ReduceOp::SUM);
}

void all_reduce_bucketed(const std::vector<std::shared_ptr<Tensor>>& tensors, size_t bucket_size_bytes,
                         ReduceOp op) {
    if (tensors.empty()) return;
    std::vector<std::shared_ptr<Tensor>> current_bucket;
    size_t current_bucket_bytes = 0;

    auto flush_bucket = [op](const std::vector<std::shared_ptr<Tensor>>& bucket) {
        if (bucket.empty()) return;
        if (bucket.size() == 1) {
            if (op == ReduceOp::SUM && NCCLBridge::get().is_available() &&
                bucket[0]->device.type == DeviceType::GPU) {
                OverlappedAllReducer::get().push_and_all_reduce(bucket[0]);
                OverlappedAllReducer::get().sync();
            } else {
                ProcessGroup::get().all_reduce(bucket[0], op);
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
                std::memcpy(flat_buffer->raw_data_ptr() + offset, t->raw_data_ptr(), n * t->storage->element_size());
            } else {
                rt_gpu_copy(t->gpu_data(), flat_buffer->gpu_data(), n * t->storage->element_size(), 0, offset * t->storage->element_size());
            }
            offset += n;
        }

        if (op == ReduceOp::SUM && NCCLBridge::get().is_available() && flat_buffer->device.type == DeviceType::GPU) {
            OverlappedAllReducer::get().push_and_all_reduce(flat_buffer);
            OverlappedAllReducer::get().sync();
        } else {
            ProcessGroup::get().all_reduce(flat_buffer, op);
            ProcessGroup::get().sync_comm();
        }

        offset = 0;
        for (const auto& t : bucket) {
            size_t n = t->numel();
            if (t->device.type == DeviceType::CPU) {
                t->storage->ensure_cpu();
                flat_buffer->storage->ensure_cpu();
                std::memcpy(t->raw_data_ptr(), flat_buffer->raw_data_ptr() + offset, n * t->storage->element_size());
            } else {
                rt_gpu_copy(flat_buffer->gpu_data(), t->gpu_data(), n * t->storage->element_size(), offset * t->storage->element_size(), 0);
            }
            offset += n;
        }
    };

    for (auto& t : tensors) {
        if (!t) continue;
        size_t tensor_bytes = t->numel() * t->storage->element_size();
        bool same_layout = current_bucket.empty() ||
            (t->dtype == current_bucket[0]->dtype && t->device == current_bucket[0]->device);
        if ((!same_layout || current_bucket_bytes + tensor_bytes > bucket_size_bytes) &&
            !current_bucket.empty()) {
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

static std::shared_ptr<Tensor> dist_contig_in(std::shared_ptr<Tensor> t) {
    if (t->is_contiguous()) return t;
    return t->contiguous();
}

static void dist_contig_out(std::shared_ptr<Tensor> orig, std::shared_ptr<Tensor> tmp) {
    if (tmp.get() != orig.get()) orig->copy_(tmp);
}

static void dist_require_reduce_dtype(DataType dtype) {
    switch (dtype) {
        case DataType::FP64:
        case DataType::FP32:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::INT32:
        case DataType::INT16:
        case DataType::INT8:
            return;
        default:
            throw std::runtime_error("distributed reduce: unsupported dtype");
    }
}

static double dist_decode(DataType dtype, const char* p) {
    switch (dtype) {
        case DataType::FP64:
            return *reinterpret_cast<const double*>(p);
        case DataType::FP32:
            return *reinterpret_cast<const float*>(p);
        case DataType::FP16:
            return half_to_float(*reinterpret_cast<const uint16_t*>(p));
        case DataType::BF16: {
            uint32_t v = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(p)) << 16;
            return *reinterpret_cast<const float*>(&v);
        }
        case DataType::INT32:
            return static_cast<double>(*reinterpret_cast<const int32_t*>(p));
        case DataType::INT16:
            return static_cast<double>(*reinterpret_cast<const int16_t*>(p));
        case DataType::INT8:
            return static_cast<double>(*reinterpret_cast<const int8_t*>(p));
        default:
            throw std::runtime_error("distributed reduce: unsupported dtype");
    }
}

static void dist_encode(DataType dtype, char* p, double v) {
    switch (dtype) {
        case DataType::FP64:
            *reinterpret_cast<double*>(p) = v;
            return;
        case DataType::FP32:
            *reinterpret_cast<float*>(p) = static_cast<float>(v);
            return;
        case DataType::FP16:
            *reinterpret_cast<uint16_t*>(p) = float_to_half(static_cast<float>(v));
            return;
        case DataType::BF16: {
            float f = static_cast<float>(v);
            uint32_t u = *reinterpret_cast<uint32_t*>(&f);
            uint32_t q = u >> 16;
            uint16_t out;
            if ((q & 0x7F80u) == 0x7F80u) {
                out = static_cast<uint16_t>(q);
            } else {
                out = static_cast<uint16_t>((u + 0x7FFFu + (q & 1u)) >> 16);
            }
            *reinterpret_cast<uint16_t*>(p) = out;
            return;
        }
        case DataType::INT32:
            *reinterpret_cast<int32_t*>(p) = static_cast<int32_t>(v);
            return;
        case DataType::INT16:
            *reinterpret_cast<int16_t*>(p) = static_cast<int16_t>(v);
            return;
        case DataType::INT8:
            *reinterpret_cast<int8_t*>(p) = static_cast<int8_t>(v);
            return;
        default:
            throw std::runtime_error("distributed reduce: unsupported dtype");
    }
}

static double dist_op_init(ReduceOp op) {
    switch (op) {
        case ReduceOp::SUM:
        case ReduceOp::AVG:
            return 0.0;
        case ReduceOp::PRODUCT:
            return 1.0;
        case ReduceOp::MIN:
            return std::numeric_limits<double>::infinity();
        case ReduceOp::MAX:
            return -std::numeric_limits<double>::infinity();
    }
    return 0.0;
}

static double dist_op_apply(ReduceOp op, double acc, double v) {
    switch (op) {
        case ReduceOp::SUM:
        case ReduceOp::AVG:
            return acc + v;
        case ReduceOp::PRODUCT:
            return acc * v;
        case ReduceOp::MIN:
            return v < acc ? v : acc;
        case ReduceOp::MAX:
            return v > acc ? v : acc;
    }
    return acc;
}

static double dist_op_finalize(ReduceOp op, double acc, int world_size) {
    if (op == ReduceOp::AVG && world_size > 0) return acc / static_cast<double>(world_size);
    return acc;
}

static void dist_read_bytes(std::shared_ptr<Tensor> t, char* dst, size_t n) {
    size_t esz = t->storage->element_size();
    if (t->device.type == DeviceType::GPU) {
        rt_gpu_read(t->gpu_data(), n, dst, t->offset * esz);
    } else {
        std::memcpy(dst, t->raw_data_ptr(), n);
    }
}

static void dist_write_bytes(std::shared_ptr<Tensor> t, const char* src, size_t n) {
    size_t esz = t->storage->element_size();
    if (t->device.type == DeviceType::GPU) {
        rt_gpu_write(t->gpu_data(), n, src, t->offset * esz);
    } else {
        std::memcpy(t->raw_data_ptr(), src, n);
    }
}

static std::string shm_abort_message(int code, int err_rank, bool self) {
    std::string m = "distributed: shm collective aborted [dist-w6] rank " + std::to_string(err_rank);
    if (code == (int)ShmErr::VALIDATION) m += ": validation failed";
    else if (code == (int)ShmErr::DTYPE) m += ": unsupported dtype";
    else if (code == (int)ShmErr::BUFFER) m += ": tensor exceeds shm buffer";
    else m += ": unknown error";
    if (!self) m += " (propagated from peer)";
    return m;
}

uint64_t ProcessGroup::shm_shape_hash(const std::vector<int64_t>& shape) {
    uint64_t h = 1469598103934665603ull;
    h ^= (uint64_t)shape.size(); h *= 1099511628211ull;
    for (size_t i = 0; i < shape.size(); ++i) {
        h ^= (uint64_t)shape[i] + 0x9e3779b97f4a7c15ull;
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t ProcessGroup::shm_shape_hash2(const std::vector<int64_t>& s1, const std::vector<int64_t>& s2) {
    uint64_t h = shm_shape_hash(s1);
    h ^= shm_shape_hash(s2) + 0x9e3779b97f4a7c15ull;
    h *= 1099511628211ull;
    return h;
}

uint64_t ProcessGroup::shm_make_sig(uint64_t tag, uint64_t sh, uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    uint64_t h = 1469598103934665603ull;
    h ^= tag; h *= 1099511628211ull;
    h ^= sh; h *= 1099511628211ull;
    h ^= a; h *= 1099511628211ull;
    h ^= b; h *= 1099511628211ull;
    h ^= c; h *= 1099511628211ull;
    h ^= d; h *= 1099511628211ull;
    return h;
}

int ProcessGroup::shm_proto_begin() {
    shm_ctrl_->err[rank_].store(0);
    return shm_ctrl_->steps[rank_].load();
}

void ProcessGroup::shm_proto_abort(int code, int cur_step) {
    shm_ctrl_->err[rank_].store(code);
    shm_ctrl_->steps[rank_].store(cur_step + 2);
    throw ShmAbortError(shm_abort_message(code, rank_, true));
}

void ProcessGroup::shm_proto_check_sig(uint64_t sig, int cur_step) {
    for (int r = 0; r < world_size_; ++r) {
        if (shm_ctrl_->sig[r].load() != sig) shm_proto_abort((int)ShmErr::VALIDATION, cur_step);
    }
}

int ProcessGroup::shm_proto_phase1() {
    int cur_step = shm_ctrl_->steps[rank_].load();
    shm_ctrl_->steps[rank_].store(cur_step + 1);
    for (int r = 0; r < world_size_; ++r) {
        while (shm_ctrl_->steps[r].load() < cur_step + 1) {
            int code = shm_ctrl_->err[r].load();
            if (code != 0) {
                shm_ctrl_->steps[rank_].store(cur_step + 2);
                throw ShmAbortError(shm_abort_message(code, r, false));
            }
            usleep(1);
        }
    }
    for (int r = 0; r < world_size_; ++r) {
        int code = shm_ctrl_->err[r].load();
        if (code != 0) {
            shm_ctrl_->steps[rank_].store(cur_step + 2);
            throw ShmAbortError(shm_abort_message(code, r, r == rank_));
        }
    }
    return cur_step;
}

void ProcessGroup::shm_proto_phase2(int cur_step) {
    shm_ctrl_->steps[rank_].store(cur_step + 2);
    for (int r = 0; r < world_size_; ++r) {
        while (shm_ctrl_->steps[r].load() < cur_step + 2) {
            int code = shm_ctrl_->err[r].load();
            if (code != 0) {
                throw ShmAbortError(shm_abort_message(code, r, false));
            }
            usleep(1);
        }
    }
    for (int r = 0; r < world_size_; ++r) {
        int code = shm_ctrl_->err[r].load();
        if (code != 0) {
            throw ShmAbortError(shm_abort_message(code, r, r == rank_));
        }
    }
}

void ProcessGroup::barrier_shm() {
#ifdef _WIN32
    throw std::runtime_error("distributed: shm transport unavailable on Windows [dist-w6]");
#endif
    std::lock_guard<std::mutex> lock(shm_step_mutex_);
    shm_proto_begin();
    int cur_step = shm_proto_phase1();
    shm_proto_phase2(cur_step);
}

void ProcessGroup::barrier_socket() {
    Device cpu(DeviceType::CPU);
    if (rank_ == 0) {
        for (int i = 1; i < world_size_; ++i) {
            auto tmp = Tensor::create({1}, cpu, false, DataType::INT32);
            recv_tensor(tmp, i);
        }
        auto ack = Tensor::create({1}, cpu, false, DataType::INT32);
        for (int i = 1; i < world_size_; ++i) send_tensor(ack, i);
    } else {
        auto sync = Tensor::create({1}, cpu, false, DataType::INT32);
        send_tensor(sync, 0);
        auto ack = Tensor::create({1}, cpu, false, DataType::INT32);
        recv_tensor(ack, 0);
    }
}

void ProcessGroup::barrier() {
    if (!initialized_ || world_size_ <= 1) return;
    if (use_shm_ && shm_ctrl_) {
        barrier_shm();
        return;
    }
    barrier_socket();
}

void ProcessGroup::all_reduce(std::shared_ptr<Tensor> tensor, ReduceOp op) {
#ifdef _WIN32
    throw std::runtime_error("distributed collectives not implemented on Windows");
#endif
    if (!initialized_ || world_size_ <= 1) return;
    if (op == ReduceOp::SUM) {
        all_reduce(tensor);
        return;
    }
    std::lock_guard<std::mutex> coll_lock(raw_socket_mutex_);
    check_collective_agreement(static_cast<int32_t>(tensor->dtype), tensor->shape);
    if (NCCLBridge::get().all_reduce(tensor, op)) return;
    if (all_reduce_shm(tensor, op)) return;
    all_reduce_socket_op(tensor, op);
}

std::future<void> ProcessGroup::all_reduce_async(std::shared_ptr<Tensor> tensor, ReduceOp op) {
#ifdef _WIN32
    throw std::runtime_error("distributed collectives not implemented on Windows");
#endif
    auto promise = std::make_shared<std::promise<void>>();
    if (!initialized_ || world_size_ <= 1) {
        promise->set_value();
        return promise->get_future();
    }
    {
        std::lock_guard<std::mutex> lock(async_mutex_);
        async_queue_.emplace(tensor, promise, op);
    }
    async_cv_.notify_one();
    return promise->get_future();
}

bool ProcessGroup::all_reduce_shm(std::shared_ptr<Tensor> tensor, ReduceOp op) {
#ifdef _WIN32
    throw std::runtime_error("distributed: shm transport unavailable on Windows [dist-w6]");
#endif
    if (op == ReduceOp::SUM) return all_reduce_shm(tensor);
    if (!use_shm_ || !shm_ctrl_ || !shm_buffers_[rank_]) return false;
    std::lock_guard<std::mutex> lock(shm_step_mutex_);
    int cur0 = shm_proto_begin();
    try {
        auto work = dist_contig_in(tensor);
        try {
            dist_require_reduce_dtype(work->dtype);
        } catch (...) {
            shm_proto_abort((int)ShmErr::DTYPE, cur0);
        }
        size_t N = work->numel();
        size_t elem_sz = work->storage->element_size();
        if (N * elem_sz > shm_buffer_size_) shm_proto_abort((int)ShmErr::BUFFER, cur0);
        dist_read_bytes(work, reinterpret_cast<char*>(shm_buffers_[rank_]), N * elem_sz);
        uint64_t sig = shm_make_sig(5, shm_shape_hash(work->shape), (uint64_t)N, (uint64_t)work->dtype, (uint64_t)op, 0);
        shm_ctrl_->sig[rank_].store(sig);
        int cur_step = shm_proto_phase1();
        shm_proto_check_sig(sig, cur_step);
        std::vector<char> out(N * elem_sz);
        for (size_t j = 0; j < N; ++j) {
            double acc = dist_op_init(op);
            for (int r = 0; r < world_size_; ++r) {
                const char* rp = reinterpret_cast<const char*>(shm_buffers_[r]) + j * elem_sz;
                acc = dist_op_apply(op, acc, dist_decode(work->dtype, rp));
            }
            dist_encode(work->dtype, out.data() + j * elem_sz, dist_op_finalize(op, acc, world_size_));
        }
        dist_write_bytes(work, out.data(), N * elem_sz);
        dist_contig_out(tensor, work);
        shm_proto_phase2(cur_step);
    } catch (const ShmAbortError&) {
        throw;
    } catch (...) {
        shm_proto_abort((int)ShmErr::VALIDATION, cur0);
    }
    return true;
}

void ProcessGroup::all_reduce_socket_op(std::shared_ptr<Tensor> tensor, ReduceOp op) {
#ifdef _WIN32
    throw std::runtime_error("distributed collectives not implemented on Windows");
#endif
    dist_require_reduce_dtype(tensor->dtype);
    size_t N = tensor->numel();
    size_t elem_sz = tensor->storage->element_size();
    std::vector<char> data(N * elem_sz);
    dist_read_bytes(tensor, data.data(), N * elem_sz);
    if (rank_ == 0) {
        std::vector<double> acc(N);
        for (size_t j = 0; j < N; ++j) {
            acc[j] = dist_op_apply(op, dist_op_init(op), dist_decode(tensor->dtype, data.data() + j * elem_sz));
        }
        std::vector<char> temp(N * elem_sz);
        for (int i = 1; i < world_size_; ++i) {
            size_t recvd = 0;
            while (recvd < N * elem_sz) {
                ssize_t r = recv(client_fds_[i], temp.data() + recvd, N * elem_sz - recvd, 0);
                if (r <= 0) throw std::runtime_error("Socket read error in all_reduce");
                recvd += r;
            }
            for (size_t j = 0; j < N; ++j) {
                acc[j] = dist_op_apply(op, acc[j], dist_decode(tensor->dtype, temp.data() + j * elem_sz));
            }
        }
        for (size_t j = 0; j < N; ++j) {
            dist_encode(tensor->dtype, data.data() + j * elem_sz, dist_op_finalize(op, acc[j], world_size_));
        }
        for (int i = 1; i < world_size_; ++i) {
            size_t sent = 0;
            while (sent < N * elem_sz) {
                ssize_t s = send(client_fds_[i], data.data() + sent, N * elem_sz - sent, 0);
                if (s <= 0) throw std::runtime_error("Socket write error in all_reduce");
                sent += s;
            }
        }
    } else {
        size_t sent = 0;
        while (sent < N * elem_sz) {
            ssize_t s = send(peer_fd_, data.data() + sent, N * elem_sz - sent, 0);
            if (s <= 0) throw std::runtime_error("Socket write error in all_reduce");
            sent += s;
        }
        size_t recvd = 0;
        while (recvd < N * elem_sz) {
            ssize_t r = recv(peer_fd_, data.data() + recvd, N * elem_sz - recvd, 0);
            if (r <= 0) throw std::runtime_error("Socket read error in all_reduce");
            recvd += r;
        }
    }
    dist_write_bytes(tensor, data.data(), N * elem_sz);
}

void ProcessGroup::all_to_all(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) {
    if (!initialized_ || world_size_ <= 1) {
        output->copy_(input);
        return;
    }
    if (all_to_all_shm(input, output)) return;
    all_to_all_socket(input, output);
}

bool ProcessGroup::all_to_all_shm(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) {
#ifdef _WIN32
    throw std::runtime_error("distributed: shm transport unavailable on Windows [dist-w6]");
#endif
    if (!use_shm_ || !shm_ctrl_ || !shm_buffers_[rank_]) return false;
    std::lock_guard<std::mutex> lock(shm_step_mutex_);
    int cur0 = shm_proto_begin();
    try {
        auto in_work = dist_contig_in(input);
        size_t N = in_work->numel();
        size_t elem_sz = in_work->storage->element_size();
        if (N % (size_t)world_size_ != 0) shm_proto_abort((int)ShmErr::VALIDATION, cur0);
        if (output->numel() != (int64_t)N) shm_proto_abort((int)ShmErr::VALIDATION, cur0);
        size_t S = N / (size_t)world_size_;
        if (N * elem_sz > shm_buffer_size_) shm_proto_abort((int)ShmErr::BUFFER, cur0);
        dist_read_bytes(in_work, reinterpret_cast<char*>(shm_buffers_[rank_]), N * elem_sz);
        uint64_t sig = shm_make_sig(6, shm_shape_hash(in_work->shape), (uint64_t)N, (uint64_t)in_work->dtype, 0, 0);
        shm_ctrl_->sig[rank_].store(sig);
        int cur_step = shm_proto_phase1();
        shm_proto_check_sig(sig, cur_step);
        auto out_work = dist_contig_in(output);
        std::vector<char> out(N * elem_sz);
        for (int r = 0; r < world_size_; ++r) {
            const char* src = reinterpret_cast<const char*>(shm_buffers_[r]) + (size_t)rank_ * S * elem_sz;
            std::memcpy(out.data() + (size_t)r * S * elem_sz, src, S * elem_sz);
        }
        dist_write_bytes(out_work, out.data(), N * elem_sz);
        dist_contig_out(output, out_work);
        shm_proto_phase2(cur_step);
    } catch (const ShmAbortError&) {
        throw;
    } catch (...) {
        shm_proto_abort((int)ShmErr::VALIDATION, cur0);
    }
    return true;
}

void ProcessGroup::all_to_all_socket(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) {
    auto in_work = dist_contig_in(input);
    size_t N = in_work->numel();
    size_t elem_sz = in_work->storage->element_size();
    if (N % (size_t)world_size_ != 0) throw std::runtime_error("all_to_all: numel not divisible by world_size");
    if (output->numel() != (int64_t)N) throw std::runtime_error("all_to_all: output numel mismatch");
    size_t S = N / (size_t)world_size_;
    std::vector<char> in_bytes(N * elem_sz);
    dist_read_bytes(in_work, in_bytes.data(), N * elem_sz);
    std::vector<char> out_bytes(N * elem_sz);
    if (rank_ == 0) {
        std::vector<std::vector<char>> peers(world_size_, std::vector<char>(N * elem_sz));
        peers[0] = in_bytes;
        for (int i = 1; i < world_size_; ++i) {
            auto tmp = Tensor::create({(int64_t)N}, in_work->device, false, in_work->dtype);
            recv_tensor(tmp, i);
            auto c = dist_contig_in(tmp);
            dist_read_bytes(c, peers[i].data(), N * elem_sz);
        }
        for (int i = 1; i < world_size_; ++i) {
            auto ot = Tensor::create({(int64_t)N}, in_work->device, false, in_work->dtype);
            auto oc = dist_contig_in(ot);
            std::vector<char> ob(N * elem_sz);
            for (int r = 0; r < world_size_; ++r) {
                std::memcpy(ob.data() + (size_t)r * S * elem_sz, peers[r].data() + (size_t)i * S * elem_sz, S * elem_sz);
            }
            dist_write_bytes(oc, ob.data(), N * elem_sz);
            dist_contig_out(ot, oc);
            send_tensor(ot, i);
        }
        for (int r = 0; r < world_size_; ++r) {
            std::memcpy(out_bytes.data() + (size_t)r * S * elem_sz, peers[r].data(), S * elem_sz);
        }
    } else {
        send_tensor(in_work, 0);
        auto tmp = Tensor::create({(int64_t)N}, in_work->device, false, in_work->dtype);
        recv_tensor(tmp, 0);
        auto c = dist_contig_in(tmp);
        dist_read_bytes(c, out_bytes.data(), N * elem_sz);
    }
    auto out_work = dist_contig_in(output);
    dist_write_bytes(out_work, out_bytes.data(), N * elem_sz);
    dist_contig_out(output, out_work);
}

void ProcessGroup::scatter(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, int src) {
    if (src < 0 || src >= world_size_) throw std::runtime_error("scatter: src rank out of range");
    if (!initialized_ || world_size_ <= 1) {
        output->copy_(input);
        return;
    }
    if (scatter_shm(output, input, src)) return;
    scatter_socket(output, input, src);
}

bool ProcessGroup::scatter_shm(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, int src) {
#ifdef _WIN32
    throw std::runtime_error("distributed: shm transport unavailable on Windows [dist-w6]");
#endif
    if (!use_shm_ || !shm_ctrl_ || !shm_buffers_[rank_]) return false;
    std::lock_guard<std::mutex> lock(shm_step_mutex_);
    int cur0 = shm_proto_begin();
    try {
        size_t S = output->numel();
        size_t elem_sz = output->storage->element_size();
        size_t N = S * (size_t)world_size_;
        if (N * elem_sz > shm_buffer_size_) shm_proto_abort((int)ShmErr::BUFFER, cur0);
        if (rank_ == src) {
            auto in_work = dist_contig_in(input);
            if (in_work->numel() != (int64_t)N) shm_proto_abort((int)ShmErr::VALIDATION, cur0);
            if (in_work->dtype != output->dtype) shm_proto_abort((int)ShmErr::VALIDATION, cur0);
            dist_read_bytes(in_work, reinterpret_cast<char*>(shm_buffers_[src]), N * elem_sz);
        }
        uint64_t sig = shm_make_sig(7, shm_shape_hash(output->shape), (uint64_t)S, (uint64_t)N, (uint64_t)output->dtype, (uint64_t)src);
        shm_ctrl_->sig[rank_].store(sig);
        int cur_step = shm_proto_phase1();
        shm_proto_check_sig(sig, cur_step);
        auto out_work = dist_contig_in(output);
        const char* srcp = reinterpret_cast<const char*>(shm_buffers_[src]) + (size_t)rank_ * S * elem_sz;
        std::vector<char> out(S * elem_sz);
        std::memcpy(out.data(), srcp, S * elem_sz);
        dist_write_bytes(out_work, out.data(), S * elem_sz);
        dist_contig_out(output, out_work);
        shm_proto_phase2(cur_step);
    } catch (const ShmAbortError&) {
        throw;
    } catch (...) {
        shm_proto_abort((int)ShmErr::VALIDATION, cur0);
    }
    return true;
}

void ProcessGroup::scatter_socket(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, int src) {
    size_t S = output->numel();
    size_t elem_sz = output->storage->element_size();
    if (rank_ == src) {
        auto in_work = dist_contig_in(input);
        size_t N = in_work->numel();
        if (N != S * (size_t)world_size_) throw std::runtime_error("scatter: input numel mismatch");
        std::vector<char> in_bytes(N * elem_sz);
        dist_read_bytes(in_work, in_bytes.data(), N * elem_sz);
        auto out_work = dist_contig_in(output);
        dist_write_bytes(out_work, in_bytes.data() + (size_t)rank_ * S * elem_sz, S * elem_sz);
        dist_contig_out(output, out_work);
        for (int i = 0; i < world_size_; ++i) {
            if (i == src) continue;
            auto chunk = Tensor::create({(int64_t)S}, in_work->device, false, in_work->dtype);
            auto cc = dist_contig_in(chunk);
            dist_write_bytes(cc, in_bytes.data() + (size_t)i * S * elem_sz, S * elem_sz);
            dist_contig_out(chunk, cc);
            send_tensor(chunk, i);
        }
    } else {
        auto tmp = Tensor::create({(int64_t)S}, output->device, false, output->dtype);
        recv_tensor(tmp, src);
        auto c = dist_contig_in(tmp);
        std::vector<char> buf(S * elem_sz);
        dist_read_bytes(c, buf.data(), S * elem_sz);
        auto out_work = dist_contig_in(output);
        dist_write_bytes(out_work, buf.data(), S * elem_sz);
        dist_contig_out(output, out_work);
    }
}

void ProcessGroup::gather(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output, int dst) {
    if (dst < 0 || dst >= world_size_) throw std::runtime_error("gather: dst rank out of range");
    if (!initialized_ || world_size_ <= 1) {
        output->copy_(input);
        return;
    }
    if (gather_shm(input, output, dst)) return;
    gather_socket(input, output, dst);
}

bool ProcessGroup::gather_shm(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output, int dst) {
#ifdef _WIN32
    throw std::runtime_error("distributed: shm transport unavailable on Windows [dist-w6]");
#endif
    if (!use_shm_ || !shm_ctrl_ || !shm_buffers_[rank_]) return false;
    std::lock_guard<std::mutex> lock(shm_step_mutex_);
    int cur0 = shm_proto_begin();
    try {
        auto in_work = dist_contig_in(input);
        size_t S = in_work->numel();
        size_t elem_sz = in_work->storage->element_size();
        size_t N = S * (size_t)world_size_;
        if (N * elem_sz > shm_buffer_size_) shm_proto_abort((int)ShmErr::BUFFER, cur0);
        if (rank_ == dst && output->numel() != (int64_t)N) shm_proto_abort((int)ShmErr::VALIDATION, cur0);
        dist_read_bytes(in_work, reinterpret_cast<char*>(shm_buffers_[rank_]), S * elem_sz);
        uint64_t sig = shm_make_sig(8, shm_shape_hash(in_work->shape), (uint64_t)S, (uint64_t)N, (uint64_t)in_work->dtype, (uint64_t)dst);
        shm_ctrl_->sig[rank_].store(sig);
        int cur_step = shm_proto_phase1();
        shm_proto_check_sig(sig, cur_step);
        if (rank_ == dst) {
            auto out_work = dist_contig_in(output);
            std::vector<char> out(N * elem_sz);
            for (int r = 0; r < world_size_; ++r) {
                std::memcpy(out.data() + (size_t)r * S * elem_sz, shm_buffers_[r], S * elem_sz);
            }
            dist_write_bytes(out_work, out.data(), N * elem_sz);
            dist_contig_out(output, out_work);
        }
        shm_proto_phase2(cur_step);
    } catch (const ShmAbortError&) {
        throw;
    } catch (...) {
        shm_proto_abort((int)ShmErr::VALIDATION, cur0);
    }
    return true;
}

void ProcessGroup::gather_socket(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output, int dst) {
    auto in_work = dist_contig_in(input);
    size_t S = in_work->numel();
    size_t elem_sz = in_work->storage->element_size();
    if (rank_ == dst) {
        size_t N = S * (size_t)world_size_;
        if (output->numel() != (int64_t)N) throw std::runtime_error("gather: output numel mismatch");
        std::vector<char> in_bytes(S * elem_sz);
        dist_read_bytes(in_work, in_bytes.data(), S * elem_sz);
        std::vector<char> out(N * elem_sz);
        std::memcpy(out.data() + (size_t)rank_ * S * elem_sz, in_bytes.data(), S * elem_sz);
        for (int i = 0; i < world_size_; ++i) {
            if (i == dst) continue;
            auto tmp = Tensor::create({(int64_t)S}, in_work->device, false, in_work->dtype);
            recv_tensor(tmp, i);
            auto c = dist_contig_in(tmp);
            dist_read_bytes(c, out.data() + (size_t)i * S * elem_sz, S * elem_sz);
        }
        auto out_work = dist_contig_in(output);
        dist_write_bytes(out_work, out.data(), N * elem_sz);
        dist_contig_out(output, out_work);
    } else {
        send_tensor(in_work, dst);
    }
}

void ProcessGroup::reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full, ReduceOp op) {
#ifdef _WIN32
    throw std::runtime_error("distributed collectives not implemented on Windows");
#endif
    if (!initialized_ || world_size_ <= 1) {
        shard->copy_(full);
        return;
    }
    if (op == ReduceOp::SUM) {
        reduce_scatter(shard, full);
        return;
    }
    if (NCCLBridge::get().reduce_scatter(shard, full, op)) return;
    if (reduce_scatter_shm(shard, full, op)) return;
    throw std::runtime_error("reduce_scatter: non-SUM ReduceOp not implemented for socket fallback");
}

bool ProcessGroup::reduce_scatter_shm(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full, ReduceOp op) {
#ifdef _WIN32
    throw std::runtime_error("distributed: shm transport unavailable on Windows [dist-w6]");
#endif
    if (op == ReduceOp::SUM) return reduce_scatter_shm(shard, full);
    if (!use_shm_ || !shm_ctrl_ || !shm_buffers_[rank_]) return false;
    std::lock_guard<std::mutex> lock(shm_step_mutex_);
    int cur0 = shm_proto_begin();
    try {
        auto full_work = dist_contig_in(full);
        auto shard_work = dist_contig_in(shard);
        try {
            dist_require_reduce_dtype(full_work->dtype);
        } catch (...) {
            shm_proto_abort((int)ShmErr::DTYPE, cur0);
        }
        size_t N = full_work->numel();
        size_t elem_sz = full_work->storage->element_size();
        if (N % (size_t)world_size_ != 0) shm_proto_abort((int)ShmErr::VALIDATION, cur0);
        size_t S = N / (size_t)world_size_;
        if (shard_work->numel() != (int64_t)S) shm_proto_abort((int)ShmErr::VALIDATION, cur0);
        if (N * elem_sz > shm_buffer_size_) shm_proto_abort((int)ShmErr::BUFFER, cur0);
        dist_read_bytes(full_work, reinterpret_cast<char*>(shm_buffers_[rank_]), N * elem_sz);
        uint64_t sig = shm_make_sig(9, shm_shape_hash2(shard_work->shape, full_work->shape), (uint64_t)S, (uint64_t)N, (uint64_t)full_work->dtype, (uint64_t)op);
        shm_ctrl_->sig[rank_].store(sig);
        int cur_step = shm_proto_phase1();
        shm_proto_check_sig(sig, cur_step);
        std::vector<char> out(S * elem_sz);
        for (size_t k = 0; k < S; ++k) {
            size_t j = (size_t)rank_ * S + k;
            double acc = dist_op_init(op);
            for (int r = 0; r < world_size_; ++r) {
                const char* rp = reinterpret_cast<const char*>(shm_buffers_[r]) + j * elem_sz;
                acc = dist_op_apply(op, acc, dist_decode(full_work->dtype, rp));
            }
            dist_encode(shard_work->dtype, out.data() + k * elem_sz, dist_op_finalize(op, acc, world_size_));
        }
        dist_write_bytes(shard_work, out.data(), S * elem_sz);
        dist_contig_out(shard, shard_work);
        shm_proto_phase2(cur_step);
    } catch (const ShmAbortError&) {
        throw;
    } catch (...) {
        shm_proto_abort((int)ShmErr::VALIDATION, cur0);
    }
    return true;
}

int NCCLBridge::map_reduce_op(ReduceOp op) {
    switch (op) {
        case ReduceOp::SUM:
            return 0;
        case ReduceOp::PRODUCT:
            return 1;
        case ReduceOp::MAX:
            return 2;
        case ReduceOp::MIN:
            return 3;
        case ReduceOp::AVG:
            return 0;
    }
    return 0;
}

DistributedDataParallel::DistributedDataParallel(std::shared_ptr<nn::Module> sub_module, ReduceOp op)
    : module(sub_module), reduce_op(op) {}

std::shared_ptr<Tensor> DistributedDataParallel::forward(std::shared_ptr<Tensor> input) {
    return module->forward(input);
}

void DistributedDataParallel::backward(std::shared_ptr<Tensor> loss) {
    loss->backward();
    sync_grads();
}

void DistributedDataParallel::backward(std::shared_ptr<Tensor> loss, std::shared_ptr<Tensor> grad_output) {
    loss->backward(grad_output);
    sync_grads();
}

void DistributedDataParallel::sync_grads() {
    if (no_sync_) return;
    auto& pg = ProcessGroup::get();
    if (!pg.is_initialized() || pg.get_world_size() <= 1) return;
    std::vector<std::shared_ptr<Tensor>> grads;
    size_t missing = 0;
    for (auto& p : module->parameters()) {
        if (!p || !p->requires_grad) continue;
        if (p->grad) grads.push_back(p->grad);
        else ++missing;
    }
    if (missing > 0) {
        if (find_unused_params_)
            throw std::runtime_error("[litetorch Error] DistributedDataParallel: " +
                                     std::to_string(missing) +
                                     " parameter(s) received no gradient; check the model or disable find_unused_parameters");
        std::fprintf(stderr,
                     "[litetorch Warning] DistributedDataParallel: %zu parameter(s) have no grad and were skipped in sync_grads\n",
                     missing);
    }
    all_reduce_bucketed(grads, bucket_bytes_, reduce_op);
}

void DistributedDataParallel::set_no_sync(bool skip) {
    no_sync_ = skip;
}

bool DistributedDataParallel::no_sync() const {
    return no_sync_;
}

void DistributedDataParallel::set_find_unused_parameters(bool enable) {
    find_unused_params_ = enable;
}

bool DistributedDataParallel::find_unused_parameters() const {
    return find_unused_params_;
}

void DistributedDataParallel::set_bucket_size_bytes(size_t bytes) {
    if (bytes == 0)
        throw std::runtime_error("[litetorch Error] DistributedDataParallel: bucket size must be > 0");
    bucket_bytes_ = bytes;
}

size_t DistributedDataParallel::bucket_size_bytes() const {
    return bucket_bytes_;
}

DDPNoSyncGuard::DDPNoSyncGuard(DistributedDataParallel& ddp) : ddp_(ddp), prev_(ddp.no_sync()) {
    ddp_.set_no_sync(true);
}

DDPNoSyncGuard::~DDPNoSyncGuard() {
    ddp_.set_no_sync(prev_);
}

std::vector<std::shared_ptr<Tensor>> DistributedDataParallel::parameters() {
    return module->parameters();
}

std::vector<std::shared_ptr<nn::Module>> DistributedDataParallel::children() {
    return {module};
}

void DistributedDataParallel::to(const Device& device) {
    module->to(device);
}

void DistributedDataParallel::train() {
    module->train();
}

void DistributedDataParallel::eval() {
    module->eval();
}

void barrier() {
    ProcessGroup::get().barrier();
}

void all_reduce(std::shared_ptr<Tensor> tensor, ReduceOp op) {
    ProcessGroup::get().all_reduce(tensor, op);
}

std::future<void> all_reduce_async(std::shared_ptr<Tensor> tensor, ReduceOp op) {
    return ProcessGroup::get().all_reduce_async(tensor, op);
}

void all_to_all(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) {
    ProcessGroup::get().all_to_all(input, output);
}

void scatter(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, int src) {
    ProcessGroup::get().scatter(output, input, src);
}

void gather(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output, int dst) {
    ProcessGroup::get().gather(input, output, dst);
}

void reduce_scatter(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full, ReduceOp op) {
    ProcessGroup::get().reduce_scatter(shard, full, op);
}

void all_reduce_grads(std::shared_ptr<nn::Module> module, ReduceOp op) {
    std::vector<std::future<void>> futures;
    for (auto& p : module->parameters()) {
        if (p && p->requires_grad && p->grad) {
            futures.push_back(ProcessGroup::get().all_reduce_async(p->grad, op));
        }
    }
    for (auto& f : futures) f.wait();
}

}
}
