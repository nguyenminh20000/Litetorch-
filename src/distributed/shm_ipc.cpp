#include "litetorch/distributed.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend_route.h"
#include "litetorch/platform.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>

#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#endif

namespace litetorch {
namespace distributed {

static std::shared_ptr<Tensor> shm_contig_in(std::shared_ptr<Tensor> t) {
    if (t->is_contiguous()) return t;
    return t->contiguous();
}

static void shm_contig_out(std::shared_ptr<Tensor> orig, std::shared_ptr<Tensor> tmp) {
    if (tmp.get() != orig.get()) orig->copy_(tmp);
}

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
        uint32_t q = mantissa >> 13;
        uint32_t rounded = mantissa + 0xFFFu + (q & 1u);
        uint32_t mq = rounded >> 13;
        if (mq >= 0x400u) {
            return sign | ((new_exp + 1) << 10);
        }
        return sign | (new_exp << 10) | mq;
    }
}

static uint16_t dist_w6_bf16_encode(float f) {
    uint32_t val = *(uint32_t*)&f;
    uint32_t q = val >> 16;
    if ((q & 0x7F80u) == 0x7F80u) {
        return (uint16_t)q;
    }
    return (uint16_t)((val + 0x7FFFu + (q & 1u)) >> 16);
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
bool ProcessGroup::all_reduce_shm(std::shared_ptr<Tensor>) {
    throw std::runtime_error("distributed: shm transport unavailable on Windows [dist-w6]");
}
bool ProcessGroup::broadcast_shm(std::shared_ptr<Tensor>, int) {
    throw std::runtime_error("distributed: shm transport unavailable on Windows [dist-w6]");
}
bool ProcessGroup::all_gather_shm(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>) {
    throw std::runtime_error("distributed: shm transport unavailable on Windows [dist-w6]");
}
bool ProcessGroup::reduce_scatter_shm(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>) {
    throw std::runtime_error("distributed: shm transport unavailable on Windows [dist-w6]");
}
#else
void ProcessGroup::init_shm(const std::string& master_addr, int master_port) {
    master_port_ = master_port;
    use_shm_ = (master_addr == "127.0.0.1" || master_addr == "localhost");
    if (!use_shm_) return;
    if (world_size_ > 4096) throw std::runtime_error("distributed: world_size exceeds shm rank limit [dist-w6]");

    shm_job_tag_ = std::to_string(master_port_) + "_" + std::to_string((long long)getpid());
    std::string ctrl_name = "/litetorch_shm_ctrl_" + shm_job_tag_;
    shm_buffer_size_ = 64 * 1024 * 1024 * sizeof(float);

    if (rank_ == 0) {
        std::string ctrl_prefix = "litetorch_shm_ctrl_" + std::to_string(master_port_) + "_";
        std::string rank_prefix = "litetorch_shm_rank_" + std::to_string(master_port_) + "_";
        DIR* d = opendir("/dev/shm");
        if (d) {
            struct dirent* e;
            while ((e = readdir(d)) != nullptr) {
                std::string n = e->d_name;
                if (n.compare(0, ctrl_prefix.size(), ctrl_prefix) == 0 ||
                    n.compare(0, rank_prefix.size(), rank_prefix) == 0) {
                    shm_unlink(("/" + n).c_str());
                }
            }
            closedir(d);
        }
        shm_unlink(ctrl_name.c_str());
        shm_ctrl_fd_ = shm_open(ctrl_name.c_str(), O_CREAT | O_RDWR, 0666);
        if (shm_ctrl_fd_ >= 0) {
            if (ftruncate(shm_ctrl_fd_, sizeof(ShmControl)) == 0) {
                shm_ctrl_ = (ShmControl*)mmap(nullptr, sizeof(ShmControl), PROT_READ | PROT_WRITE, MAP_SHARED, shm_ctrl_fd_, 0);
                if (shm_ctrl_ != MAP_FAILED) {
                    for (int i = 0; i < 4096; ++i) {
                        shm_ctrl_->steps[i].store(0);
                        shm_ctrl_->err[i].store(0);
                        shm_ctrl_->p2p_steps[i].store(0);
                        shm_ctrl_->sig[i].store(0);
                    }
                }
            }
        }

        int32_t tag_len = (int32_t)shm_job_tag_.size();
        for (int i = 1; i < world_size_; ++i) {
            if (write(client_fds_[i], &tag_len, sizeof(tag_len)) != sizeof(tag_len)) {}
            if (tag_len > 0) {
                if (write(client_fds_[i], shm_job_tag_.data(), (size_t)tag_len) != (ssize_t)tag_len) {}
            }
        }
    } else {
        int32_t tag_len = 0;
        if (read(peer_fd_, &tag_len, sizeof(tag_len)) != sizeof(tag_len)) {}
        if (tag_len > 0 && tag_len < 256) {
            std::string tag((size_t)tag_len, '\0');
            size_t got = 0;
            while (got < (size_t)tag_len) {
                ssize_t r = read(peer_fd_, &tag[got], (size_t)tag_len - got);
                if (r <= 0) break;
                got += (size_t)r;
            }
            shm_job_tag_ = tag;
        }
        ctrl_name = "/litetorch_shm_ctrl_" + shm_job_tag_;

        shm_ctrl_fd_ = shm_open(ctrl_name.c_str(), O_RDWR, 0666);
        if (shm_ctrl_fd_ >= 0) {
            shm_ctrl_ = (ShmControl*)mmap(nullptr, sizeof(ShmControl), PROT_READ | PROT_WRITE, MAP_SHARED, shm_ctrl_fd_, 0);
        }
    }

    std::string rank_name = "/litetorch_shm_rank_" + shm_job_tag_ + "_" + std::to_string(rank_);
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
        std::string r_name = "/litetorch_shm_rank_" + shm_job_tag_ + "_" + std::to_string(r);
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

    std::string ctrl_name = "/litetorch_shm_ctrl_" + shm_job_tag_;
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

    std::string rank_name = "/litetorch_shm_rank_" + shm_job_tag_ + "_" + std::to_string(rank_);
    shm_unlink(rank_name.c_str());
}

bool ProcessGroup::all_reduce_shm(std::shared_ptr<Tensor> tensor) {
    if (!use_shm_ || !shm_ctrl_ || !shm_buffers_[rank_]) return false;
    std::lock_guard<std::mutex> lock(shm_step_mutex_);
    int cur0 = shm_proto_begin();
    try {

    auto work = shm_contig_in(tensor);
    size_t N = work->numel();
    size_t elem_sz = work->storage->element_size();
    if (N * elem_sz > shm_buffer_size_) shm_proto_abort((int)ShmErr::BUFFER, cur0);
    if (work->dtype != DataType::FP16 && work->dtype != DataType::BF16 && work->dtype != DataType::FP32)
        shm_proto_abort((int)ShmErr::DTYPE, cur0);

    if (work->device.type == DeviceType::GPU) {
        rt_gpu_read(work->gpu_data(), N * elem_sz, shm_buffers_[rank_], work->offset * elem_sz);
    } else {
        std::memcpy(shm_buffers_[rank_], work->raw_data_ptr(), N * elem_sz);
    }

    uint64_t sig = shm_make_sig(1, shm_shape_hash(work->shape), (uint64_t)N, (uint64_t)work->dtype, 0, 0);
    shm_ctrl_->sig[rank_].store(sig);
    int cur_step = shm_proto_phase1();
    shm_proto_check_sig(sig, cur_step);

    if (work->dtype == DataType::FP16) {
        std::vector<uint16_t> acc(N, 0);
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int r = 0; r < world_size_; ++r) {
                uint16_t* r_buf = reinterpret_cast<uint16_t*>(shm_buffers_[r]);
                sum += half_to_float(r_buf[j]);
            }
            acc[j] = float_to_half(sum);
        }
        if (work->device.type == DeviceType::GPU) {
            rt_gpu_write(work->gpu_data(), N * elem_sz, acc.data(), work->offset * elem_sz);
        } else {
            std::memcpy(work->raw_data_ptr(), acc.data(), N * elem_sz);
        }
    } else if (work->dtype == DataType::BF16) {
        std::vector<uint16_t> acc(N, 0);
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int r = 0; r < world_size_; ++r) {
                uint16_t* r_buf = reinterpret_cast<uint16_t*>(shm_buffers_[r]);
                uint32_t val = ((uint32_t)r_buf[j]) << 16;
                sum += *(float*)&val;
            }
            acc[j] = dist_w6_bf16_encode(sum);
        }
        if (work->device.type == DeviceType::GPU) {
            rt_gpu_write(work->gpu_data(), N * elem_sz, acc.data(), work->offset * elem_sz);
        } else {
            std::memcpy(work->raw_data_ptr(), acc.data(), N * elem_sz);
        }
    } else {
        std::vector<float> acc(N, 0.0f);
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int r = 0; r < world_size_; ++r) {
                sum += shm_buffers_[r][j];
            }
            acc[j] = sum;
        }
        if (work->device.type == DeviceType::GPU) {
            rt_gpu_write(work->gpu_data(), N * elem_sz, acc.data(), work->offset * elem_sz);
        } else {
            std::memcpy(work->raw_data_ptr(), acc.data(), N * elem_sz);
        }
    }

    shm_contig_out(tensor, work);
    shm_proto_phase2(cur_step);
    } catch (const ShmAbortError&) {
        throw;
    } catch (...) {
        shm_proto_abort((int)ShmErr::VALIDATION, cur0);
    }
    return true;
}

bool ProcessGroup::broadcast_shm(std::shared_ptr<Tensor> tensor, int src) {
    if (!use_shm_ || !shm_ctrl_ || !shm_buffers_[rank_]) return false;
    std::lock_guard<std::mutex> lock(shm_step_mutex_);
    int cur0 = shm_proto_begin();
    try {

    auto work = shm_contig_in(tensor);
    size_t N = work->numel();
    size_t elem_sz = work->storage->element_size();
    if (N * elem_sz > shm_buffer_size_) shm_proto_abort((int)ShmErr::BUFFER, cur0);

    if (rank_ == src) {
        if (work->device.type == DeviceType::GPU) {
            rt_gpu_read(work->gpu_data(), N * elem_sz, shm_buffers_[rank_], work->offset * elem_sz);
        } else {
            std::memcpy(shm_buffers_[rank_], work->raw_data_ptr(), N * elem_sz);
        }
    }

    uint64_t sig = shm_make_sig(2, shm_shape_hash(work->shape), (uint64_t)N, (uint64_t)work->dtype, (uint64_t)src, 0);
    shm_ctrl_->sig[rank_].store(sig);
    int cur_step = shm_proto_phase1();
    shm_proto_check_sig(sig, cur_step);

    if (rank_ != src) {
        if (work->device.type == DeviceType::GPU) {
            rt_gpu_write(work->gpu_data(), N * elem_sz, shm_buffers_[src], work->offset * elem_sz);
        } else {
            std::memcpy(work->raw_data_ptr(), shm_buffers_[src], N * elem_sz);
        }
        shm_contig_out(tensor, work);
    }

    shm_proto_phase2(cur_step);
    } catch (const ShmAbortError&) {
        throw;
    } catch (...) {
        shm_proto_abort((int)ShmErr::VALIDATION, cur0);
    }
    return true;
}

bool ProcessGroup::all_gather_shm(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    if (!use_shm_ || !shm_ctrl_ || !shm_buffers_[rank_]) return false;
    std::lock_guard<std::mutex> lock(shm_step_mutex_);
    int cur0 = shm_proto_begin();
    try {

    auto shard_work = shm_contig_in(shard);
    size_t S = shard_work->numel();
    size_t elem_sz = shard_work->storage->element_size();
    size_t total_bytes = S * world_size_ * elem_sz;
    if (total_bytes > shm_buffer_size_) shm_proto_abort((int)ShmErr::BUFFER, cur0);

    if (shard_work->device.type == DeviceType::GPU) {
        rt_gpu_read(shard_work->gpu_data(), S * elem_sz, shm_buffers_[rank_], shard_work->offset * elem_sz);
    } else {
        std::memcpy(shm_buffers_[rank_], shard_work->raw_data_ptr(), S * elem_sz);
    }

    uint64_t sig = shm_make_sig(3, shm_shape_hash2(shard_work->shape, full->shape), (uint64_t)S, (uint64_t)full->numel(), (uint64_t)shard_work->dtype, 0);
    shm_ctrl_->sig[rank_].store(sig);
    int cur_step = shm_proto_phase1();
    shm_proto_check_sig(sig, cur_step);

    auto full_work = shm_contig_in(full);
    size_t N = full_work->numel();
    bool need_pad = (N != S * (size_t)world_size_);
    auto full_eff = full_work;
    if (need_pad) {
        full_eff = Tensor::create({(int64_t)(S * world_size_)}, full_work->device, false, full_work->dtype);
    }

    if (full_eff->device.type == DeviceType::GPU) {
        for (int r = 0; r < world_size_; ++r) {
            rt_gpu_write(full_eff->gpu_data(), S * elem_sz, shm_buffers_[r], (full_eff->offset + r * S) * elem_sz);
        }
    } else {
        for (int r = 0; r < world_size_; ++r) {
            std::memcpy((char*)full_eff->raw_data_ptr() + r * S * elem_sz, shm_buffers_[r], S * elem_sz);
        }
    }

    if (need_pad) {
        if (full_work->device.type == DeviceType::GPU) {
            std::vector<char> tmp(N * elem_sz);
            rt_gpu_read(full_eff->gpu_data(), N * elem_sz, tmp.data(), full_eff->offset * elem_sz);
            rt_gpu_write(full_work->gpu_data(), N * elem_sz, tmp.data(), full_work->offset * elem_sz);
        } else {
            std::memcpy(full_work->raw_data_ptr(), full_eff->raw_data_ptr(), N * elem_sz);
        }
    }
    shm_contig_out(full, full_work);

    shm_proto_phase2(cur_step);
    } catch (const ShmAbortError&) {
        throw;
    } catch (...) {
        shm_proto_abort((int)ShmErr::VALIDATION, cur0);
    }
    return true;
}

bool ProcessGroup::reduce_scatter_shm(std::shared_ptr<Tensor> shard, std::shared_ptr<Tensor> full) {
    if (!use_shm_ || !shm_ctrl_ || !shm_buffers_[rank_]) return false;
    std::lock_guard<std::mutex> lock(shm_step_mutex_);
    int cur0 = shm_proto_begin();
    try {

    auto shard_work = shm_contig_in(shard);
    size_t S = shard_work->numel();
    size_t elem_sz = shard_work->storage->element_size();
    size_t total_bytes = S * world_size_ * elem_sz;
    if (total_bytes > shm_buffer_size_) shm_proto_abort((int)ShmErr::BUFFER, cur0);
    if (shard_work->dtype != DataType::FP32 && shard_work->dtype != DataType::FP16 && shard_work->dtype != DataType::BF16)
        shm_proto_abort((int)ShmErr::DTYPE, cur0);

    auto full_work = shm_contig_in(full);
    size_t N = full_work->numel();
    auto full_eff = full_work;
    if (N != S * (size_t)world_size_) {
        full_eff = Tensor::create({(int64_t)(S * world_size_)}, full_work->device, false, full_work->dtype);
        if (full_work->device.type == DeviceType::GPU) {
            std::vector<char> tmp(N * elem_sz);
            rt_gpu_read(full_work->gpu_data(), N * elem_sz, tmp.data(), full_work->offset * elem_sz);
            rt_gpu_write(full_eff->gpu_data(), N * elem_sz, tmp.data(), full_eff->offset * elem_sz);
            if (total_bytes > N * elem_sz) {
                std::vector<char> zeros(total_bytes - N * elem_sz, 0);
                rt_gpu_write(full_eff->gpu_data(), zeros.size(), zeros.data(), (full_eff->offset * elem_sz) + N * elem_sz);
            }
        } else {
            std::memcpy(full_eff->raw_data_ptr(), full_work->raw_data_ptr(), N * elem_sz);
            std::memset((char*)full_eff->raw_data_ptr() + N * elem_sz, 0, total_bytes - N * elem_sz);
        }
    }

    if (full_eff->device.type == DeviceType::GPU) {
        rt_gpu_read(full_eff->gpu_data(), total_bytes, shm_buffers_[rank_], full_eff->offset * elem_sz);
    } else {
        std::memcpy(shm_buffers_[rank_], full_eff->raw_data_ptr(), total_bytes);
    }

    uint64_t sig = shm_make_sig(4, shm_shape_hash2(shard_work->shape, full_work->shape), (uint64_t)S, (uint64_t)full_work->numel(), (uint64_t)shard_work->dtype, 0);
    shm_ctrl_->sig[rank_].store(sig);
    int cur_step = shm_proto_phase1();
    shm_proto_check_sig(sig, cur_step);

    if (shard_work->dtype == DataType::FP32) {
        std::vector<float> acc(S, 0.0f);
        for (size_t j = 0; j < S; ++j) {
            float sum = 0.0f;
            for (int r = 0; r < world_size_; ++r) {
                sum += shm_buffers_[r][rank_ * S + j];
            }
            acc[j] = sum;
        }
        if (shard_work->device.type == DeviceType::GPU) {
            rt_gpu_write(shard_work->gpu_data(), S * elem_sz, acc.data(), shard_work->offset * elem_sz);
        } else {
            std::memcpy(shard_work->raw_data_ptr(), acc.data(), S * elem_sz);
        }
    } else if (shard_work->dtype == DataType::FP16) {
        std::vector<uint16_t> acc(S, 0);
        for (size_t j = 0; j < S; ++j) {
            float sum = 0.0f;
            for (int r = 0; r < world_size_; ++r) {
                uint16_t* r_buf = reinterpret_cast<uint16_t*>(shm_buffers_[r]);
                sum += half_to_float(r_buf[rank_ * S + j]);
            }
            acc[j] = float_to_half(sum);
        }
        if (shard_work->device.type == DeviceType::GPU) {
            rt_gpu_write(shard_work->gpu_data(), S * elem_sz, acc.data(), shard_work->offset * elem_sz);
        } else {
            std::memcpy(shard_work->raw_data_ptr(), acc.data(), S * elem_sz);
        }
    } else if (shard_work->dtype == DataType::BF16) {
        std::vector<uint16_t> acc(S, 0);
        for (size_t j = 0; j < S; ++j) {
            float sum = 0.0f;
            for (int r = 0; r < world_size_; ++r) {
                uint16_t* r_buf = reinterpret_cast<uint16_t*>(shm_buffers_[r]);
                uint32_t val = ((uint32_t)r_buf[rank_ * S + j]) << 16;
                sum += *(float*)&val;
            }
            acc[j] = dist_w6_bf16_encode(sum);
        }
        if (shard_work->device.type == DeviceType::GPU) {
            rt_gpu_write(shard_work->gpu_data(), S * elem_sz, acc.data(), shard_work->offset * elem_sz);
        } else {
            std::memcpy(shard_work->raw_data_ptr(), acc.data(), S * elem_sz);
        }
    }

    shm_contig_out(shard, shard_work);
    shm_proto_phase2(cur_step);
    } catch (const ShmAbortError&) {
        throw;
    } catch (...) {
        shm_proto_abort((int)ShmErr::VALIDATION, cur0);
    }
    return true;
}
#endif

}
}
