#include "litetorch/distributed.h"
#include "litetorch/cl_backend.h"
#include "litetorch/platform.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>

#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace litetorch {
namespace distributed {

uint16_t float_to_half(float f) {
    uint32_t x = *(uint32_t*)&f;
    uint32_t sign = (x >> 16) & 0x8000;
    uint16_t exponent = ((x >> 23) & 0xff);
    uint32_t mantissa = x & 0x7fffff;
    if (exponent == 0) {
        return sign;
    } else if (exponent == 255) {
        return sign | 0x7c00 | (mantissa ? 0x200 : 0);
    } else {
        int new_exp = exponent - 127 + 15;
        if (new_exp >= 31) {
            return sign | 0x7c00;
        } else if (new_exp <= 0) {
            return sign;
        }
        return sign | (new_exp << 10) | (mantissa >> 13);
    }
}

float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h & 0x7c00) >> 10;
    uint32_t mantissa = h & 0x03ff;
    uint32_t val = 0;
    if (exponent == 0) {
        if (mantissa != 0) {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03ff;
            val = sign | (exponent << 23) | (mantissa << 13);
        } else {
            val = sign;
        }
    } else if (exponent == 31) {
        val = sign | 0x7f800000 | (mantissa << 13);
    } else {
        val = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    return *(float*)&val;
}

#ifdef _WIN32
void ProcessGroup::init_shm(const std::string&, int) { use_shm_ = false; }
void ProcessGroup::shutdown_shm() {}
bool ProcessGroup::all_reduce_shm(std::shared_ptr<Tensor>) { return false; }
bool ProcessGroup::broadcast_shm(std::shared_ptr<Tensor>, int) { return false; }
bool ProcessGroup::all_gather_shm(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>) { return false; }
bool ProcessGroup::reduce_scatter_shm(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>) { return false; }
#else
void ProcessGroup::init_shm(const std::string& master_addr, int master_port) {
    master_port_ = master_port;
    use_shm_ = (master_addr == "127.0.0.1" || master_addr == "localhost");
    if (!use_shm_) return;

    std::string ctrl_name = "/litetorch_shm_ctrl_" + std::to_string(master_port_);
    shm_buffer_size_ = 64 * 1024 * 1024 * sizeof(float);

    if (rank_ == 0) {
        shm_unlink(ctrl_name.c_str());
        shm_ctrl_fd_ = shm_open(ctrl_name.c_str(), O_CREAT | O_RDWR, 0666);
        if (shm_ctrl_fd_ >= 0) {
            if (ftruncate(shm_ctrl_fd_, sizeof(ShmControl)) == 0) {
                shm_ctrl_ = (ShmControl*)mmap(nullptr, sizeof(ShmControl), PROT_READ | PROT_WRITE, MAP_SHARED, shm_ctrl_fd_, 0);
                if (shm_ctrl_ != MAP_FAILED) {
                    for (int i = 0; i < 128; ++i) {
                        shm_ctrl_->steps[i].store(0);
                    }
                }
            }
        }

        int ctrl_ready = 1;
        for (int i = 1; i < world_size_; ++i) {
            if (write(client_fds_[i], &ctrl_ready, sizeof(ctrl_ready)) != sizeof(ctrl_ready)) {}
        }
    } else {
        int ctrl_ready = 0;
        if (read(peer_fd_, &ctrl_ready, sizeof(ctrl_ready)) != sizeof(ctrl_ready)) {}

        shm_ctrl_fd_ = shm_open(ctrl_name.c_str(), O_RDWR, 0666);
        if (shm_ctrl_fd_ >= 0) {
            shm_ctrl_ = (ShmControl*)mmap(nullptr, sizeof(ShmControl), PROT_READ | PROT_WRITE, MAP_SHARED, shm_ctrl_fd_, 0);
        }
    }

    std::string rank_name = "/litetorch_shm_rank_" + std::to_string(master_port_) + "_" + std::to_string(rank_);
    shm_unlink(rank_name.c_str());
    int rank_fd = shm_open(rank_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (rank_fd >= 0) {
        if (ftruncate(rank_fd, shm_buffer_size_) == 0) {}
        close(rank_fd);
    }

    if (rank_ == 0) {
        int shm_ready = 1;
        for (int i = 1; i < world_size_; ++i) {
            int client_ready = 0;
            if (read(client_fds_[i], &client_ready, sizeof(client_ready)) != sizeof(client_ready)) {}
        }
        for (int i = 1; i < world_size_; ++i) {
            if (write(client_fds_[i], &shm_ready, sizeof(shm_ready)) != sizeof(shm_ready)) {}
        }
    } else {
        int client_ready = 1;
        if (write(peer_fd_, &client_ready, sizeof(client_ready)) != sizeof(client_ready)) {}
        int shm_ready = 0;
        if (read(peer_fd_, &shm_ready, sizeof(shm_ready)) != sizeof(shm_ready)) {}
    }

    shm_buffers_.resize(world_size_, nullptr);
    for (int r = 0; r < world_size_; ++r) {
        std::string r_name = "/litetorch_shm_rank_" + std::to_string(master_port_) + "_" + std::to_string(r);
        int fd = shm_open(r_name.c_str(), O_RDWR, 0666);
        if (fd >= 0) {
            void* ptr = mmap(nullptr, shm_buffer_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (ptr != MAP_FAILED) {
                shm_buffers_[r] = (float*)ptr;
            }
            close(fd);
        }
    }
}

void ProcessGroup::shutdown_shm() {
    if (!use_shm_) return;

    std::string ctrl_name = "/litetorch_shm_ctrl_" + std::to_string(master_port_);
    if (shm_ctrl_ && shm_ctrl_ != MAP_FAILED) {
        munmap(shm_ctrl_, sizeof(ShmControl));
        shm_ctrl_ = nullptr;
    }
    if (shm_ctrl_fd_ != -1) {
        close(shm_ctrl_fd_);
        shm_ctrl_fd_ = -1;
        if (rank_ == 0) {
            shm_unlink(ctrl_name.c_str());
        }
    }

    for (int r = 0; r < world_size_; ++r) {
        if (shm_buffers_[r]) {
            munmap(shm_buffers_[r], shm_buffer_size_);
        }
    }
    shm_buffers_.clear();

    std::string rank_name = "/litetorch_shm_rank_" + std::to_string(master_port_) + "_" + std::to_string(rank_);
    shm_unlink(rank_name.c_str());
}

bool ProcessGroup::all_reduce_shm(std::shared_ptr<Tensor> tensor) {
    if (!use_shm_ || !shm_ctrl_ || !shm_buffers_[rank_]) return false;
    
    size_t N = tensor->numel();
    size_t elem_sz = tensor->storage->element_size();
    if (N * elem_sz > shm_buffer_size_) return false;

    if (tensor->device.type == DeviceType::GPU) {
        CLBackend::get().read(tensor->gpu_data(), N * elem_sz, shm_buffers_[rank_], tensor->offset * elem_sz);
    } else {
        std::memcpy(shm_buffers_[rank_], tensor->data_ptr(), N * elem_sz);
    }

    int cur_step = shm_ctrl_->steps[rank_].load();
    shm_ctrl_->steps[rank_].store(cur_step + 1);

    for (int r = 0; r < world_size_; ++r) {
        while (shm_ctrl_->steps[r].load() < cur_step + 1) {
            usleep(1);
        }
    }

    if (tensor->dtype == DataType::FP16) {
        std::vector<uint16_t> acc(N, 0);
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int r = 0; r < world_size_; ++r) {
                uint16_t* r_buf = reinterpret_cast<uint16_t*>(shm_buffers_[r]);
                sum += half_to_float(r_buf[j]);
            }
            acc[j] = float_to_half(sum / world_size_);
        }
        if (tensor->device.type == DeviceType::GPU) {
            CLBackend::get().write(tensor->gpu_data(), N * elem_sz, acc.data(), tensor->offset * elem_sz);
        } else {
            std::memcpy(tensor->data_ptr(), acc.data(), N * elem_sz);
        }
    } else if (tensor->dtype == DataType::BF16) {
        std::vector<uint16_t> acc(N, 0);
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int r = 0; r < world_size_; ++r) {
                uint16_t* r_buf = reinterpret_cast<uint16_t*>(shm_buffers_[r]);
                uint32_t val = ((uint32_t)r_buf[j]) << 16;
                sum += *(float*)&val;
            }
            float avg = sum / world_size_;
            uint32_t val = *(uint32_t*)&avg;
            acc[j] = (uint16_t)(val >> 16);
        }
        if (tensor->device.type == DeviceType::GPU) {
            CLBackend::get().write(tensor->gpu_data(), N * elem_sz, acc.data(), tensor->offset * elem_sz);
        } else {
            std::memcpy(tensor->data_ptr(), acc.data(), N * elem_sz);
        }
    } else {
        std::vector<float> acc(N, 0.0f);
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int r = 0; r < world_size_; ++r) {
                sum += shm_buffers_[r][j];
            }
            acc[j] = sum / world_size_;
        }
        if (tensor->device.type == DeviceType::GPU) {
            CLBackend::get().write(tensor->gpu_data(), N * elem_sz, acc.data(), tensor->offset * elem_sz);
        } else {
            std::memcpy(tensor->data_ptr(), acc.data(), N * elem_sz);
        }
    }

    shm_ctrl_->steps[rank_].store(cur_step + 2);
    for (int r = 0; r < world_size_; ++r) {
        while (shm_ctrl_->steps[r].load() < cur_step + 2) {
            usleep(1);
        }
    }
    return true;
}

bool ProcessGroup::broadcast_shm(std::shared_ptr<Tensor> tensor, int src) {
    if (!use_shm_ || !shm_ctrl_ || !shm_buffers_[rank_]) return false;

    size_t N = tensor->numel();
    size_t elem_sz = tensor->storage->element_size();
    if (N * elem_sz > shm_buffer_size_) return false;

    if (rank_ == src) {
        if (tensor->device.type == DeviceType::GPU) {
            CLBackend::get().read(tensor->gpu_data(), N * elem_sz, shm_buffers_[rank_], tensor->offset * elem_sz);
        } else {
            std::memcpy(shm_buffers_[rank_], tensor->data_ptr(), N * elem_sz);
        }
    }

    int cur_step = shm_ctrl_->steps[rank_].load();
    shm_ctrl_->steps[rank_].store(cur_step + 1);

    for (int r = 0; r < world_size_; ++r) {
        while (shm_ctrl_->steps[r].load() < cur_step + 1) {
            usleep(1);
        }
    }

    if (rank_ != src) {
        if (tensor->device.type == DeviceType::GPU) {
            CLBackend::get().write(tensor->gpu_data(), N * elem_sz, shm_buffers_[src], tensor->offset * elem_sz);
        } else {
            std::memcpy(tensor->data_ptr(), shm_buffers_[src], N * elem_sz);
        }
    }

    shm_ctrl_->steps[rank_].store(cur_step + 2);
    for (int r = 0; r < world_size_; ++r) {
        while (shm_ctrl_->steps[r].load() < cur_step + 2) {
            usleep(1);
        }
    }
    return true;
}

bool ProcessGroup::all_gather_shm(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    if (!use_shm_ || !shm_ctrl_ || !shm_buffers_[rank_]) return false;

    size_t S = shard->numel();
    size_t elem_sz = shard->storage->element_size();
    size_t total_bytes = S * world_size_ * elem_sz;
    if (total_bytes > shm_buffer_size_) return false;

    if (shard->device.type == DeviceType::GPU) {
        CLBackend::get().read(shard->gpu_data(), S * elem_sz, shm_buffers_[rank_], shard->offset * elem_sz);
    } else {
        std::memcpy(shm_buffers_[rank_], shard->data_ptr(), S * elem_sz);
    }

    int cur_step = shm_ctrl_->steps[rank_].load();
    shm_ctrl_->steps[rank_].store(cur_step + 1);

    for (int r = 0; r < world_size_; ++r) {
        while (shm_ctrl_->steps[r].load() < cur_step + 1) {
            usleep(1);
        }
    }

    if (full->device.type == DeviceType::GPU) {
        for (int r = 0; r < world_size_; ++r) {
            CLBackend::get().write(full->gpu_data(), S * elem_sz, shm_buffers_[r], (full->offset + r * S) * elem_sz);
        }
    } else {
        for (int r = 0; r < world_size_; ++r) {
            std::memcpy((char*)full->data_ptr() + r * S * elem_sz, shm_buffers_[r], S * elem_sz);
        }
    }

    shm_ctrl_->steps[rank_].store(cur_step + 2);
    for (int r = 0; r < world_size_; ++r) {
        while (shm_ctrl_->steps[r].load() < cur_step + 2) {
            usleep(1);
        }
    }
    return true;
}

bool ProcessGroup::reduce_scatter_shm(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    if (!use_shm_ || !shm_ctrl_ || !shm_buffers_[rank_]) return false;

    size_t S = shard->numel();
    size_t elem_sz = shard->storage->element_size();
    size_t total_bytes = S * world_size_ * elem_sz;
    if (total_bytes > shm_buffer_size_) return false;

    if (full->device.type == DeviceType::GPU) {
        CLBackend::get().read(full->gpu_data(), total_bytes, shm_buffers_[rank_], full->offset * elem_sz);
    } else {
        std::memcpy(shm_buffers_[rank_], full->data_ptr(), total_bytes);
    }

    int cur_step = shm_ctrl_->steps[rank_].load();
    shm_ctrl_->steps[rank_].store(cur_step + 1);

    for (int r = 0; r < world_size_; ++r) {
        while (shm_ctrl_->steps[r].load() < cur_step + 1) {
            usleep(1);
        }
    }

    if (shard->dtype == DataType::FP32) {
        std::vector<float> acc(S, 0.0f);
        for (size_t j = 0; j < S; ++j) {
            float sum = 0.0f;
            for (int r = 0; r < world_size_; ++r) {
                sum += shm_buffers_[r][rank_ * S + j];
            }
            acc[j] = sum;
        }
        if (shard->device.type == DeviceType::GPU) {
            CLBackend::get().write(shard->gpu_data(), S * elem_sz, acc.data(), shard->offset * elem_sz);
        } else {
            std::memcpy(shard->data_ptr(), acc.data(), S * elem_sz);
        }
    }

    shm_ctrl_->steps[rank_].store(cur_step + 2);
    for (int r = 0; r < world_size_; ++r) {
        while (shm_ctrl_->steps[r].load() < cur_step + 2) {
            usleep(1);
        }
    }
    return true;
}
#endif

}
}
