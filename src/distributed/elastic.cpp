#include "litetorch/elastic.h"
#include "litetorch/rendezvous.h"
#include "litetorch/distributed.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace litetorch {
namespace distributed {
namespace elastic {

namespace {

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool pg_active() {
    auto& pg = ProcessGroup::get();
    return pg.is_initialized() && pg.get_world_size() > 1;
}

void maybe_barrier() {
    if (pg_active()) ProcessGroup::get().barrier();
}

int my_rank() {
    auto& pg = ProcessGroup::get();
    return pg.is_initialized() ? pg.get_rank() : 0;
}

int my_world() {
    auto& pg = ProcessGroup::get();
    return pg.is_initialized() ? pg.get_world_size() : 1;
}

bool file_exists(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool dir_exists(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

void remove_all(const std::string& p) {
    DIR* d = ::opendir(p.c_str());
    if (!d) {
        ::unlink(p.c_str());
        return;
    }
    while (dirent* e = ::readdir(d)) {
        std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        std::string full = p + "/" + name;
        struct stat st{};
        if (::stat(full.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) remove_all(full);
        else ::unlink(full.c_str());
    }
    ::closedir(d);
    ::rmdir(p.c_str());
}

}

HeartbeatAgent::HeartbeatAgent(const std::string& store_addr, int store_port, int rank, int interval_ms)
    : addr_(store_addr), port_(store_port), rank_(rank), interval_ms_(interval_ms) {}

HeartbeatAgent::~HeartbeatAgent() {
    stop();
}

void HeartbeatAgent::start() {
    std::lock_guard<std::mutex> lk(mu_);
    if (running_.load()) return;
    running_.store(true);
    thread_ = std::thread([this]() { loop(); });
}

void HeartbeatAgent::stop() {
    bool was = running_.exchange(false);
    std::thread t;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (thread_.joinable()) t = std::move(thread_);
    }
    if (t.joinable()) t.join();
    (void)was;
}

void HeartbeatAgent::loop() {
    rendezvous::RendezvousClient client(addr_, port_, 10);
    while (running_.load()) {
        try {
            client.heartbeat(rank_);
        } catch (...) {
        }
        for (int i = 0; i < interval_ms_ / 50 && running_.load(); i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

std::vector<int> detect_dead_ranks(const std::string& store_addr, int store_port, int64_t timeout_ms) {
    rendezvous::RendezvousClient client(store_addr, store_port, 10);
    auto beats = client.get_heartbeats();
    int64_t now = now_ms();
    std::vector<int> dead;
    for (auto& kv : beats) {
        if (now - kv.second > timeout_ms) dead.push_back(kv.first);
    }
    return dead;
}

std::string step_dir(const std::string& ckpt_dir, int64_t step) {
    return ckpt_dir + "/step-" + std::to_string(step);
}

std::string tmp_dir(const std::string& ckpt_dir, int64_t step) {
    return ckpt_dir + "/.tmp-step-" + std::to_string(step);
}

void save_consistent(
    const std::map<std::string, std::shared_ptr<Tensor>>& local_shards,
    const std::map<std::string, dist_checkpoint::TensorShardInfo>& shard_specs,
    const std::string& ckpt_dir,
    int64_t step) {
    int rank = my_rank();
    int world = my_world();
    std::string tmp = tmp_dir(ckpt_dir, step);
    std::string fin = step_dir(ckpt_dir, step);

    if (dir_exists(fin))
        throw std::runtime_error("[litetorch Error] elastic: step dir already exists: " + fin);

    if (dir_exists(tmp) || file_exists(tmp)) remove_all(tmp);

    dist_checkpoint::save_distributed(local_shards, shard_specs, tmp, step);

    maybe_barrier();

    if (rank == 0) {
        for (int r = 0; r < world; r++) {
            std::string sf = dist_checkpoint::shard_filename(tmp, r, world);
            if (!file_exists(sf))
                throw std::runtime_error("[litetorch Error] elastic: missing shard file: " + sf);
        }
        auto meta = dist_checkpoint::read_metadata(tmp);
        if (meta.step != step)
            throw std::runtime_error("[litetorch Error] elastic: metadata step mismatch");
        if ((int)meta.tensors.size() == 0 && local_shards.size() > 0)
            throw std::runtime_error("[litetorch Error] elastic: empty tensor map in metadata");
        std::ofstream mark(tmp + "/.validated", std::ios::binary | std::ios::trunc);
        mark << step;
    }

    maybe_barrier();

    if (!file_exists(tmp + "/.validated"))
        throw std::runtime_error("[litetorch Error] elastic: checkpoint validation failed for step " +
                                std::to_string(step));

    if (rank == 0) {
        if (::rename(tmp.c_str(), fin.c_str()) != 0)
            throw std::runtime_error("[litetorch Error] elastic: atomic rename failed: " +
                                    std::string(std::strerror(errno)));
    }

    maybe_barrier();

    if (!dir_exists(fin))
        throw std::runtime_error("[litetorch Error] elastic: committed step dir missing: " + fin);
}

int64_t latest_consistent_step(const std::string& ckpt_dir) {
    int64_t best = -1;
    DIR* d = ::opendir(ckpt_dir.c_str());
    if (!d) return -1;
    while (dirent* e = ::readdir(d)) {
        std::string name = e->d_name;
        if (name.rfind("step-", 0) != 0) continue;
        int64_t step = -1;
        try {
            step = std::stoll(name.substr(5));
        } catch (...) {
            continue;
        }
        std::string dir = ckpt_dir + "/" + name;
        if (!file_exists(dist_checkpoint::metadata_path(dir))) continue;
        try {
            auto meta = dist_checkpoint::read_metadata(dir);
            if (meta.step != step) continue;
        } catch (...) {
            continue;
        }
        if (step > best) best = step;
    }
    ::closedir(d);
    return best;
}

void load_latest(
    std::map<std::string, std::shared_ptr<Tensor>>& local_shards,
    const std::string& ckpt_dir) {
    int64_t step = latest_consistent_step(ckpt_dir);
    if (step < 0)
        throw std::runtime_error("[litetorch Error] elastic: no consistent checkpoint in " + ckpt_dir);
    dist_checkpoint::load_distributed(local_shards, step_dir(ckpt_dir, step));
}

}
}
}
