#ifndef LITETORCH_TENSOR_H
#define LITETORCH_TENSOR_H

#include "litetorch/device.h"
#include "litetorch/cl_backend.h"
#include <vector>
#include <memory>
#include <cstdint>
#include <list>
#include <mutex>
#include <atomic>
#include <functional>
#include <utility>

namespace litetorch {

enum class DataType {
    FP64,
    FP32,
    FP16,
    BF16,
    INT16,
    INT8,
    INT4,
    FP8_E4M3,
    FP8_E5M2,
    NF4,
    FP4_E2M1,
    INT32
};

class Node;
namespace distributed { class ProcessGroup; }
class StorageImpl;

class StorageImpl : public std::enable_shared_from_this<StorageImpl> {
public:
    size_t size;
    Device device;
    DataType dtype = DataType::FP32;
    float* cpu_data = nullptr;
    cl_mem gpu_data = nullptr;
    std::atomic<bool> is_swapped{false};
    mutable std::atomic<bool> in_use{false};
    bool has_lru_iterator = false;
    std::list<StorageImpl*>::iterator lru_iterator;
    mutable std::mutex storage_mutex_;

    friend class MemoryManager;

    StorageImpl(size_t size, const Device& device, DataType dtype = DataType::FP32);
    ~StorageImpl();

    size_t element_size() const {
        if (dtype == DataType::FP64) return 8;
        if (dtype == DataType::FP16) return 2;
        if (dtype == DataType::BF16) return 2;
        if (dtype == DataType::INT16) return 2;
        if (dtype == DataType::INT8) return 1;
        if (dtype == DataType::INT4) return 1;
        if (dtype == DataType::FP8_E4M3) return 1;
        if (dtype == DataType::FP8_E5M2) return 1;
        if (dtype == DataType::NF4) return 1;
        if (dtype == DataType::FP4_E2M1) return 1;
        return 4;
    }

    float* get_cpu_ptr();
    cl_mem get_gpu_ptr();

    void to(const Device& new_device);
    void evict();
    void evict_impl();
    void discard_gpu();
    void discard_gpu_impl();
    void swap_in_impl();
    void swap_in();
    void ensure_cpu();
};

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::shared_ptr<StorageImpl> storage;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int64_t offset = 0;
    Device device;
    DataType dtype = DataType::FP32;
    bool requires_grad = false;
    std::shared_ptr<Tensor> grad = nullptr;
    std::shared_ptr<Tensor> shard = nullptr;
    struct CreatorWrapper {
        std::shared_ptr<Node> ptr = nullptr;
        Tensor* parent = nullptr;
        CreatorWrapper() = default;
        CreatorWrapper(std::nullptr_t) : ptr(nullptr) {}
        CreatorWrapper& operator=(std::nullptr_t) { ptr = nullptr; return *this; }
        CreatorWrapper& operator=(std::shared_ptr<Node> p);
        operator std::shared_ptr<Node>() const { return ptr; }
        std::shared_ptr<Node> operator->() const { return ptr; }
        bool operator==(std::nullptr_t) const { return ptr == nullptr; }
        bool operator!=(std::nullptr_t) const { return ptr != nullptr; }
        bool operator==(const std::shared_ptr<Node>& other) const { return ptr == other; }
        bool operator!=(const std::shared_ptr<Node>& other) const { return ptr != other; }
        bool operator!() const { return ptr == nullptr; }
        explicit operator bool() const { return ptr != nullptr; }
    } creator;
    size_t numel_ = 0;
    bool contiguous_ = true;
    mutable std::mutex grad_mutex;

    friend class distributed::ProcessGroup;

    virtual ~Tensor() = default;
    Tensor(const std::vector<int64_t>& shape, const Device& device = Device(DeviceType::CPU, 0), bool requires_grad = false, DataType dtype = DataType::FP32);
    Tensor(std::shared_ptr<StorageImpl> storage, const std::vector<int64_t>& shape, const std::vector<int64_t>& strides, int64_t offset, const Device& device, bool requires_grad = false);

    static std::shared_ptr<Tensor> create(const std::vector<int64_t>& shape, const Device& device = Device(DeviceType::CPU, 0), bool requires_grad = false, DataType dtype = DataType::FP32);
    static std::shared_ptr<Tensor> from_vector(const std::vector<float>& data, const std::vector<int64_t>& shape, const Device& device = Device(DeviceType::CPU, 0), bool requires_grad = false, DataType dtype = DataType::FP32);
    static std::shared_ptr<Tensor> zeros(const std::vector<int64_t>& shape, const Device& device = Device(DeviceType::CPU, 0), bool requires_grad = false);
    static std::shared_ptr<Tensor> zeros(const std::vector<int64_t>& shape, const Device& device, bool requires_grad, DataType dtype);
    static std::shared_ptr<Tensor> ones(const std::vector<int64_t>& shape, const Device& device = Device(DeviceType::CPU, 0), bool requires_grad = false, DataType dtype = DataType::FP32);
    static std::shared_ptr<Tensor> randn(const std::vector<int64_t>& shape, const Device& device = Device(DeviceType::CPU, 0), bool requires_grad = false, DataType dtype = DataType::FP32);
    static std::shared_ptr<Tensor> arange(int64_t start, int64_t end, int64_t step = 1, const Device& device = Device(DeviceType::CPU, 0), DataType dtype = DataType::FP32);
    std::shared_ptr<Tensor> masked_fill(std::shared_ptr<Tensor> mask, float value);
    std::shared_ptr<Tensor> index_select(int64_t dim, std::shared_ptr<Tensor> indices);
    std::shared_ptr<Tensor> slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1);

    size_t numel() const { return numel_; }
    bool is_contiguous() const { return contiguous_; }
    std::shared_ptr<Tensor> contiguous();
    std::shared_ptr<Tensor> to(const Device& target_device);
    std::shared_ptr<Tensor> to(DataType target_dtype);
    std::shared_ptr<Tensor> to(const Device& target_device, DataType target_dtype);
    std::shared_ptr<Tensor> to_device_async(const Device& target_device);
    std::shared_ptr<Tensor> cast(DataType target_dtype);
    std::shared_ptr<Tensor> clone();
    void copy_(std::shared_ptr<Tensor> src);
    void add_(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> view(const std::vector<int64_t>& new_shape);
    std::shared_ptr<Tensor> as_strided(const std::vector<int64_t>& new_shape, const std::vector<int64_t>& new_strides);
    std::shared_ptr<Tensor> reshape(const std::vector<int64_t>& new_shape);
    std::shared_ptr<Tensor> transpose(int64_t dim0, int64_t dim1);
    std::shared_ptr<Tensor> permute(const std::vector<int64_t>& dims);
    std::shared_ptr<Tensor> detach();
    std::shared_ptr<Tensor> expand(const std::vector<int64_t>& shape);
    std::shared_ptr<Tensor> gather(int64_t dim, std::shared_ptr<Tensor> index);
    std::shared_ptr<Tensor> argmax(int64_t dim = -1, bool keepdim = false);
    std::shared_ptr<Tensor> argmin(int64_t dim = -1, bool keepdim = false);
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> topk(int64_t k, int64_t dim = -1, bool largest = true, bool sorted = true);
    std::shared_ptr<Tensor> eq(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> ne(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> lt(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> gt(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> le(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> ge(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> log_softmax(int64_t dim = -1);
    std::shared_ptr<Tensor> silu();
    std::shared_ptr<Tensor> sum(int64_t dim, bool keepdim = false);
    std::shared_ptr<Tensor> mean(int64_t dim, bool keepdim = false);
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> max(int64_t dim, bool keepdim = false);

    float* data_ptr();
    cl_mem gpu_data();

    void backward(std::shared_ptr<Tensor> gradient = nullptr, bool create_graph = false);
    void zero_grad();

    float item() const;
    std::vector<float> to_vector() const;

    std::vector<std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>)>> backward_hooks;
    void register_hook(std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>)> hook) {
        backward_hooks.push_back(hook);
    }
};

inline std::vector<int64_t> default_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

void manual_seed(unsigned int seed);
std::string get_rng_state();
void set_rng_state(const std::string& state);

}

#endif
