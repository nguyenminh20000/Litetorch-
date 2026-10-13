#include "litetorch/trainer.h"
#include "litetorch/nn.h"
#include "litetorch/optim.h"
#include "litetorch/serialization.h"
#include "litetorch/async_checkpoint.h"
#include "litetorch/dist_checkpoint.h"
#include "litetorch/elastic.h"
#include "litetorch/distributed.h"
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <stdexcept>

namespace litetorch {
namespace trainer {

namespace {

std::string dtype_str(DataType dtype) {
    if (dtype == DataType::FP32) return "F32";
    if (dtype == DataType::FP16) return "F16";
    if (dtype == DataType::BF16) return "BF16";
    if (dtype == DataType::FP64) return "F64";
    if (dtype == DataType::INT32) return "I32";
    if (dtype == DataType::INT16) return "I16";
    if (dtype == DataType::INT8) return "I8";
    throw std::runtime_error("[litetorch Error] trainer: unsupported dtype for checkpoint");
}

void mkdir_p(const std::string& path) {
    if (path.empty()) return;
    std::string cur;
    size_t i = 0;
    if (path[0] == '/') { cur = "/"; i = 1; }
    while (i <= path.size()) {
        size_t j = path.find('/', i);
        if (j == std::string::npos) j = path.size();
        std::string part = path.substr(i, j - i);
        if (!part.empty()) {
            cur += part;
            mkdir(cur.c_str(), 0755);
            cur += "/";
        }
        i = j + 1;
    }
}

bool is_dir(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

void remove_recursive(const std::string& path) {
    DIR* d = opendir(path.c_str());
    if (!d) {
        unlink(path.c_str());
        return;
    }
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        remove_recursive(path + "/" + n);
    }
    closedir(d);
    rmdir(path.c_str());
}

bool parse_step_file(const std::string& name, int64_t& step) {
    const std::string pre = "step-";
    const std::string suf = ".safetensors";
    if (name.size() <= pre.size() + suf.size()) return false;
    if (name.compare(0, pre.size(), pre) != 0) return false;
    if (name.compare(name.size() - suf.size(), suf.size(), suf) != 0) return false;
    std::string num = name.substr(pre.size(), name.size() - pre.size() - suf.size());
    try {
        step = std::stoll(num);
    } catch (...) {
        return false;
    }
    return true;
}

bool parse_step_dir(const std::string& name, int64_t& step) {
    const std::string pre = "step-";
    if (name.size() <= pre.size()) return false;
    if (name.compare(0, pre.size(), pre) != 0) return false;
    if (name[pre.size()] == '.') return false;
    std::string num = name.substr(pre.size());
    try {
        step = std::stoll(num);
    } catch (...) {
        return false;
    }
    return true;
}

int local_rank() {
    auto& pg = distributed::ProcessGroup::get();
    return pg.is_initialized() ? pg.get_rank() : 0;
}

int local_world_size() {
    auto& pg = distributed::ProcessGroup::get();
    return pg.is_initialized() ? pg.get_world_size() : 1;
}

}

struct Trainer::Impl {
    std::unique_ptr<async_checkpoint::AsyncCheckpointSaver> async_saver;
};

Trainer::Trainer(std::shared_ptr<nn::Module> model,
                 std::shared_ptr<optim::Optimizer> optimizer,
                 TrainerConfig config)
    : model_(model), optimizer_(optimizer), config_(config),
      impl_(new Impl()) {
    if (!model_) throw std::runtime_error("[litetorch Error] trainer: null model");
    if (!optimizer_) throw std::runtime_error("[litetorch Error] trainer: null optimizer");
    if (config_.max_checkpoints < 1) config_.max_checkpoints = 1;
    impl_->async_saver.reset(new async_checkpoint::AsyncCheckpointSaver());
    mkdir_p(config_.ckpt_dir);
}

Trainer::~Trainer() = default;

int Trainer::rank() const { return local_rank(); }

int Trainer::world_size() const { return local_world_size(); }

bool Trainer::is_distributed() const { return world_size() > 1; }

void Trainer::train_step() {
    optimizer_->step();
    optimizer_->zero_grad();
}

void Trainer::wait_pending_saves() {
    impl_->async_saver->wait();
}

std::map<std::string, std::shared_ptr<Tensor>> Trainer::param_map() const {
    std::map<std::string, std::shared_ptr<Tensor>> out;
    for (const auto& kv : model_->state_dict()) out[kv.first] = kv.second;
    return out;
}

std::vector<int64_t> Trainer::list_steps() const {
    std::vector<int64_t> steps;
    DIR* d = opendir(config_.ckpt_dir.c_str());
    if (!d) return steps;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        std::string name = e->d_name;
        int64_t step = -1;
        if (is_distributed()) {
            if (!parse_step_dir(name, step)) continue;
            if (!is_dir(config_.ckpt_dir + "/" + name)) continue;
        } else {
            if (!parse_step_file(name, step)) continue;
        }
        steps.push_back(step);
    }
    closedir(d);
    std::sort(steps.begin(), steps.end());
    return steps;
}

void Trainer::ensure_dir(const std::string& path) const {
    mkdir_p(path);
}

void Trainer::save_checkpoint(int64_t step) {
    ensure_dir(config_.ckpt_dir);
    auto params = param_map();
    if (is_distributed()) {
        std::map<std::string, dist_checkpoint::TensorShardInfo> specs;
        for (const auto& kv : params)
            specs[kv.first] = dist_checkpoint::make_replicated_spec(kv.second->shape, dtype_str(kv.second->dtype));
        if (config_.elastic)
            distributed::elastic::save_consistent(params, specs, config_.ckpt_dir, step);
        else
            dist_checkpoint::save_distributed(params, specs, config_.ckpt_dir, step);
    } else {
        std::string path = config_.ckpt_dir + "/step-" + std::to_string(step) + ".safetensors";
        if (config_.async_save)
            impl_->async_saver->save_async(params, path);
        else
            save_safetensors(params, path);
    }
    garbage_collect();
}

int64_t Trainer::load_latest() {
    if (is_distributed()) {
        int64_t step = distributed::elastic::latest_consistent_step(config_.ckpt_dir);
        if (step < 0) return -1;
        auto params = param_map();
        distributed::elastic::load_latest(params, config_.ckpt_dir);
        std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> vec(params.begin(), params.end());
        model_->load_state_dict(vec, true);
        return step;
    }
    auto steps = list_steps();
    if (steps.empty()) return -1;
    int64_t step = steps.back();
    std::string path = config_.ckpt_dir + "/step-" + std::to_string(step) + ".safetensors";
    auto loaded = load_safetensors(path);
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> vec(loaded.begin(), loaded.end());
    model_->load_state_dict(vec, true);
    return step;
}

void Trainer::garbage_collect() {
    auto steps = list_steps();
    if ((int64_t)steps.size() <= config_.max_checkpoints) return;
    size_t drop = steps.size() - (size_t)config_.max_checkpoints;
    for (size_t i = 0; i < drop; ++i) {
        std::string path;
        if (is_distributed())
            path = config_.ckpt_dir + "/step-" + std::to_string(steps[i]);
        else
            path = config_.ckpt_dir + "/step-" + std::to_string(steps[i]) + ".safetensors";
        if (rank() == 0 || !is_distributed())
            remove_recursive(path);
    }
    if (is_distributed()) {
        auto& pg = distributed::ProcessGroup::get();
        if (pg.is_initialized()) pg.barrier();
    }
}

int64_t Trainer::latest_step() const {
    if (is_distributed())
        return distributed::elastic::latest_consistent_step(config_.ckpt_dir);
    auto steps = list_steps();
    return steps.empty() ? -1 : steps.back();
}

}
}
