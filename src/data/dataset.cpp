#include "litetorch/data.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace litetorch {
namespace data {

TensorDataset::TensorDataset(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y) : x(x), y(y) {}

size_t TensorDataset::size() {
    return x->shape[0];
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> TensorDataset::get(size_t index) {
    if (index >= size()) {
        throw std::out_of_range("Dataset index out of bounds");
    }

    std::vector<int64_t> item_shape_x(x->shape.begin() + 1, x->shape.end());
    if (item_shape_x.empty()) item_shape_x = {1};
    auto item_x = Tensor::create(item_shape_x, Device(DeviceType::CPU, 0), false, x->dtype);
    size_t elem_sz_x = x->storage->element_size();
    char* x_src = (char*)x->raw_data_ptr() + index * x->strides[0] * elem_sz_x;
    std::memcpy(item_x->raw_data_ptr(), x_src, item_x->numel() * elem_sz_x);
    if (x->device.type == DeviceType::GPU) {
        item_x = item_x->to(x->device);
    }

    std::vector<int64_t> item_shape_y(y->shape.begin() + 1, y->shape.end());
    if (item_shape_y.empty()) item_shape_y = {1};
    auto item_y = Tensor::create(item_shape_y, Device(DeviceType::CPU, 0), false, y->dtype);
    size_t elem_sz_y = y->storage->element_size();
    char* y_src = (char*)y->raw_data_ptr() + index * y->strides[0] * elem_sz_y;
    std::memcpy(item_y->raw_data_ptr(), y_src, item_y->numel() * elem_sz_y);
    if (y->device.type == DeviceType::GPU) {
        item_y = item_y->to(y->device);
    }

    return {item_x, item_y};
}

Subset::Subset(std::shared_ptr<Dataset> dataset, const std::vector<size_t>& indices)
    : dataset_(dataset), indices_(indices) {
    if (!dataset_) {
        throw std::invalid_argument("[litetorch Error] Subset requires a non-null dataset");
    }
    size_t n = dataset_->size();
    for (size_t i : indices_) {
        if (i >= n) {
            throw std::out_of_range("[litetorch Error] Subset index out of range of dataset");
        }
    }
}

size_t Subset::size() {
    return indices_.size();
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> Subset::get(size_t index) {
    if (index >= indices_.size()) {
        throw std::out_of_range("[litetorch Error] Subset index out of bounds");
    }
    return dataset_->get(indices_[index]);
}

std::vector<std::shared_ptr<Subset>> random_split(std::shared_ptr<Dataset> dataset,
                                                  const std::vector<size_t>& lengths,
                                                  uint64_t seed) {
    if (!dataset) {
        throw std::invalid_argument("[litetorch Error] random_split requires a non-null dataset");
    }
    if (lengths.empty()) {
        throw std::invalid_argument("[litetorch Error] random_split requires at least one length");
    }
    size_t total = 0;
    for (size_t l : lengths) total += l;
    if (total != dataset->size()) {
        throw std::invalid_argument("[litetorch Error] random_split lengths sum does not match dataset size");
    }
    std::vector<size_t> perm(dataset->size());
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937_64 g(seed ? seed : std::random_device{}());
    std::shuffle(perm.begin(), perm.end(), g);
    std::vector<std::shared_ptr<Subset>> out;
    size_t off = 0;
    for (size_t l : lengths) {
        std::vector<size_t> idx(perm.begin() + off, perm.begin() + off + l);
        out.push_back(std::make_shared<Subset>(dataset, idx));
        off += l;
    }
    return out;
}

SequentialSampler::SequentialSampler(size_t n) : n_(n) {}

std::vector<size_t> SequentialSampler::indices() {
    std::vector<size_t> out(n_);
    std::iota(out.begin(), out.end(), 0);
    return out;
}

size_t SequentialSampler::size() const {
    return n_;
}

RandomSampler::RandomSampler(size_t n, bool replacement, size_t num_samples, uint64_t seed)
    : n_(n), replacement_(replacement), num_samples_(num_samples), seed_(seed) {
    if (n_ == 0) {
        throw std::invalid_argument("[litetorch Error] RandomSampler requires n > 0");
    }
    if (!replacement_ && num_samples_ > n_) {
        throw std::invalid_argument("[litetorch Error] RandomSampler num_samples exceeds population without replacement");
    }
}

std::vector<size_t> RandomSampler::indices() {
    std::mt19937_64 g(seed_ ? (seed_ + epoch_) : std::random_device{}());
    epoch_++;
    std::vector<size_t> out;
    if (!replacement_) {
        out.resize(n_);
        std::iota(out.begin(), out.end(), 0);
        std::shuffle(out.begin(), out.end(), g);
        size_t m = num_samples_ ? num_samples_ : n_;
        if (m < out.size()) out.resize(m);
    } else {
        size_t m = num_samples_ ? num_samples_ : n_;
        std::uniform_int_distribution<size_t> d(0, n_ - 1);
        out.reserve(m);
        for (size_t i = 0; i < m; ++i) out.push_back(d(g));
    }
    return out;
}

size_t RandomSampler::size() const {
    if (replacement_) return num_samples_ ? num_samples_ : n_;
    return num_samples_ ? num_samples_ : n_;
}

BatchSampler::BatchSampler(std::shared_ptr<Sampler> sampler, size_t batch_size, bool drop_last)
    : sampler_(sampler), batch_size_(batch_size), drop_last_(drop_last) {
    if (!sampler_) {
        throw std::invalid_argument("[litetorch Error] BatchSampler requires a non-null sampler");
    }
    if (batch_size_ == 0) {
        throw std::invalid_argument("[litetorch Error] BatchSampler requires batch_size > 0");
    }
}

std::vector<std::vector<size_t>> BatchSampler::batches() {
    std::vector<size_t> idx = sampler_->indices();
    std::vector<std::vector<size_t>> out;
    for (size_t i = 0; i < idx.size(); i += batch_size_) {
        size_t e = std::min(i + batch_size_, idx.size());
        if (drop_last_ && e - i < batch_size_) break;
        out.emplace_back(idx.begin() + i, idx.begin() + e);
    }
    return out;
}

size_t BatchSampler::size() const {
    size_t n = sampler_->size();
    size_t full = n / batch_size_;
    return (n % batch_size_ == 0 || drop_last_) ? full : full + 1;
}

std::vector<size_t> BatchSampler::indices() {
    std::vector<size_t> out;
    for (auto& b : batches()) {
        out.insert(out.end(), b.begin(), b.end());
    }
    return out;
}

}
}
