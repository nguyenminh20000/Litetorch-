#include "litetorch/tensor.h"
#include "litetorch/memory_manager.h"
#include "litetorch/autograd.h"
#include "litetorch/ops.h"
#include "litetorch/thread_pool.h"
#include "litetorch/allocator.h"
#include <numeric>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <mutex>
#include <cstring>

namespace litetorch {

struct StorageUseGuard {
    StorageUseGuard(const std::vector<std::shared_ptr<StorageImpl>>& list) : storages(list) {
        for (auto& s : storages) {
            if (s) {
                s->in_use = true;
                MemoryManager::get().ensure_gpu(s.get());
            }
        }
    }
    ~StorageUseGuard() {
        for (auto& s : storages) {
            if (s) s->in_use = false;
        }
    }
private:
    std::vector<std::shared_ptr<StorageImpl>> storages;
};

extern const std::string litetorch_kernels_src;

namespace {

uint8_t float_to_fp8_e4m3(float val) {
    if (std::isnan(val)) return 0x7F;
    union { float f; uint32_t i; } u;
    u.f = val;
    uint32_t sign = (u.i >> 31) & 1;
    uint32_t exp = (u.i >> 23) & 0xFF;
    uint32_t mant = u.i & 0x7FFFFF;
    if (exp == 0) return (sign << 7);
    if (exp == 0xFF) return (sign << 7) | 0x7F;
    int new_exp = static_cast<int>(exp) - 127 + 7;
    if (new_exp <= 0) {
        int shift = 1 - new_exp;
        if (shift > 3) return (sign << 7);
        uint32_t m = (1 << 3) | (mant >> 20);
        m >>= shift;
        return (sign << 7) | m;
    } else if (new_exp >= 15) {
        return (sign << 7) | 0x7E;
    }
    uint32_t m = mant >> 20;
    return (sign << 7) | (new_exp << 3) | m;
}

float fp8_e4m3_to_float(uint8_t val) {
    uint32_t sign = (val >> 7) & 1;
    uint32_t exp = (val >> 3) & 0x0F;
    uint32_t mant = val & 0x07;
    if (exp == 15) return std::numeric_limits<float>::quiet_NaN();
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        return (sign ? -1.0f : 1.0f) * std::pow(2.0f, -6.0f) * (static_cast<float>(mant) / 8.0f);
    }
    return (sign ? -1.0f : 1.0f) * std::pow(2.0f, static_cast<float>(exp) - 7.0f) * (1.0f + static_cast<float>(mant) / 8.0f);
}

uint8_t float_to_fp8_e5m2(float val) {
    if (std::isnan(val)) return 0x7F;
    union { float f; uint32_t i; } u;
    u.f = val;
    uint32_t sign = (u.i >> 31) & 1;
    uint32_t exp = (u.i >> 23) & 0xFF;
    uint32_t mant = u.i & 0x7FFFFF;
    if (exp == 0) return (sign << 7);
    if (exp == 0xFF) return (sign << 7) | 0x7F;
    int new_exp = static_cast<int>(exp) - 127 + 15;
    if (new_exp <= 0) {
        int shift = 1 - new_exp;
        if (shift > 2) return (sign << 7);
        uint32_t m = (1 << 2) | (mant >> 21);
        m >>= shift;
        return (sign << 7) | m;
    } else if (new_exp >= 31) {
        return (sign << 7) | 0x7E;
    }
    uint32_t m = mant >> 21;
    return (sign << 7) | (new_exp << 2) | m;
}

float fp8_e5m2_to_float(uint8_t val) {
    uint32_t sign = (val >> 7) & 1;
    uint32_t exp = (val >> 2) & 0x1F;
    uint32_t mant = val & 0x03;
    if (exp == 31) return std::numeric_limits<float>::quiet_NaN();
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        return (sign ? -1.0f : 1.0f) * std::pow(2.0f, -14.0f) * (static_cast<float>(mant) / 4.0f);
    }
    return (sign ? -1.0f : 1.0f) * std::pow(2.0f, static_cast<float>(exp) - 15.0f) * (1.0f + static_cast<float>(mant) / 4.0f);
}

const float NF4_GRID[16] = {
    -1.0f, -0.6961917f, -0.525073f, -0.3930782f, -0.2753147f, -0.1651313f, -0.0596338f, 0.0f,
    0.0596338f, 0.1651313f, 0.2753147f, 0.3930782f, 0.525073f, 0.6961917f, 1.0f, 1.0f
};

uint8_t float_to_nf4(float val) {
    float min_dist = 1e9f;
    uint8_t best_idx = 0;
    for (uint8_t i = 0; i < 16; ++i) {
        float dist = std::abs(NF4_GRID[i] - val);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

float nf4_to_float(uint8_t idx) {
    if (idx >= 16) idx = 15;
    return NF4_GRID[idx];
}

class ViewNode : public Node {
public:
    std::vector<int64_t> orig_shape;
    ViewNode(const std::vector<int64_t>& orig_shape) : Node("View"), orig_shape(orig_shape) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        return { grad_output->contiguous()->view(orig_shape) };
    }
};

class TransposeNode : public Node {
public:
    int64_t dim0;
    int64_t dim1;
    TransposeNode(int64_t dim0, int64_t dim1) : Node("Transpose"), dim0(dim0), dim1(dim1) {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        return { grad_output->transpose(dim0, dim1) };
    }
};

class ContiguousNode : public Node {
public:
    ContiguousNode() : Node("Contiguous") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        return { grad_output };
    }
};

class CloneNode : public Node {
public:
    CloneNode() : Node("Clone") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        return { grad_output->clone() };
    }
};
}

Tensor::CreatorWrapper& Tensor::CreatorWrapper::operator=(std::shared_ptr<Node> p) {
    ptr = p;
    if (p && parent) {
        Autograd::active_tensors.push_back(parent->shared_from_this());
    }
    return *this;
}

Tensor::Tensor(const std::vector<int64_t>& shape, const Device& device, bool requires_grad, DataType dtype)
    : shape(shape), strides(default_strides(shape)), offset(0), device(device), dtype(dtype), requires_grad(requires_grad) {
    creator.parent = this;
    numel_ = 1;
    for (auto dim : shape) numel_ *= dim;
    if (shape.empty()) numel_ = 0;
    contiguous_ = true;
    storage = std::make_shared<StorageImpl>(numel_, device, dtype);
    this->device = storage->device;
}

Tensor::Tensor(std::shared_ptr<StorageImpl> storage, const std::vector<int64_t>& shape, const std::vector<int64_t>& strides, int64_t offset, const Device& device, bool requires_grad)
    : storage(storage), shape(shape), strides(strides), offset(offset), device(device), requires_grad(requires_grad) {
    creator.parent = this;
    numel_ = 1;
    for (auto dim : shape) numel_ *= dim;
    if (shape.empty()) numel_ = 0;
    contiguous_ = (strides == default_strides(shape));
    if (storage) {
        dtype = storage->dtype;
    }
}

std::shared_ptr<Tensor> Tensor::create(const std::vector<int64_t>& shape, const Device& device, bool requires_grad, DataType dtype) {
    return std::make_shared<Tensor>(shape, device, requires_grad, dtype);
}

std::shared_ptr<Tensor> Tensor::from_vector(const std::vector<float>& data, const std::vector<int64_t>& shape, const Device& device, bool requires_grad, DataType dtype) {
    auto tensor = create(shape, device, requires_grad, DataType::FP32);
    if (tensor->device.type == DeviceType::GPU) {
        cl_mem gpu_ptr = tensor->storage->get_gpu_ptr();
        if (gpu_ptr) {
            CLBackend::get().write(gpu_ptr, tensor->storage->size * sizeof(float), data.data());
        }
    } else {
        float* cpu_ptr = tensor->storage->get_cpu_ptr();
        std::copy(data.begin(), data.end(), cpu_ptr + tensor->offset);
    }
    if (dtype != DataType::FP32) {
        return tensor->cast(dtype);
    }
    return tensor;
}

std::shared_ptr<Tensor> Tensor::zeros(const std::vector<int64_t>& shape, const Device& device, bool requires_grad) {
    auto tensor = create(shape, device, requires_grad);
    if (tensor->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "fill_zero");
        int size_val = static_cast<int>(tensor->numel());
        cl_mem gpu_ptr = tensor->storage->get_gpu_ptr();
        CLBackend::get().launch(kernel, {tensor->numel()}, {}, {&gpu_ptr, &size_val}, {sizeof(cl_mem), sizeof(int)});
    }
    return tensor;
}

std::shared_ptr<Tensor> Tensor::contiguous() {
    if (is_contiguous()) return shared_from_this();

    if (device.type == DeviceType::CPU && dtype != DataType::FP32) {
        auto fp32_self = cast(DataType::FP32);
        auto fp32_cont = fp32_self->contiguous();
        auto res = fp32_cont->cast(dtype);
        res->requires_grad = requires_grad;
        return res;
    }

    auto out = create(shape, device, requires_grad, dtype);
    if (device.type == DeviceType::GPU) {
        size_t num_elements = numel();
        int ndims_val = static_cast<int>(shape.size());
        
        int shape_arr[8] = {0};
        int strides_arr[8] = {0};
        for (size_t i = 0; i < shape.size() && i < 8; ++i) {
            shape_arr[i] = static_cast<int>(shape[i]);
            strides_arr[i] = static_cast<int>(strides[i]);
        }
        
        auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "make_contiguous_kernel");
        
        cl_mem src_mem = gpu_data();
        cl_mem dst_mem = out->gpu_data();
        int src_off = offset;
        int dst_off = out->offset;
        int size_val = static_cast<int>(num_elements);
        
        CLBackend::get().launch(kernel, {num_elements}, {},
            {&src_mem, &src_off, &dst_mem, &dst_off, &ndims_val, &shape_arr, &strides_arr, &size_val},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(shape_arr), sizeof(strides_arr), sizeof(int)});
    } else {
        float* src_ptr = data_ptr();
        float* dst_ptr = out->data_ptr();
        int ndims = shape.size();
        if (ndims == 1) {
            int64_t s0 = strides[0];
            int64_t sh0 = shape[0];
            for (int64_t i0 = 0; i0 < sh0; ++i0) {
                dst_ptr[i0] = src_ptr[i0 * s0];
            }
        } else if (ndims == 2) {
            int64_t s0 = strides[0], s1 = strides[1];
            int64_t sh0 = shape[0], sh1 = shape[1];
            for (int64_t i0 = 0; i0 < sh0; ++i0) {
                int64_t offset0 = i0 * s0;
                int64_t out_offset0 = i0 * sh1;
                for (int64_t i1 = 0; i1 < sh1; ++i1) {
                    dst_ptr[out_offset0 + i1] = src_ptr[offset0 + i1 * s1];
                }
            }
        } else if (ndims == 3) {
            int64_t s0 = strides[0], s1 = strides[1], s2 = strides[2];
            int64_t sh0 = shape[0], sh1 = shape[1], sh2 = shape[2];
            for (int64_t i0 = 0; i0 < sh0; ++i0) {
                int64_t offset0 = i0 * s0;
                int64_t out_offset0 = i0 * sh1 * sh2;
                for (int64_t i1 = 0; i1 < sh1; ++i1) {
                    int64_t offset1 = offset0 + i1 * s1;
                    int64_t out_offset1 = out_offset0 + i1 * sh2;
                    for (int64_t i2 = 0; i2 < sh2; ++i2) {
                        dst_ptr[out_offset1 + i2] = src_ptr[offset1 + i2 * s2];
                    }
                }
            }
        } else if (ndims == 4) {
            int64_t s0 = strides[0], s1 = strides[1], s2 = strides[2], s3 = strides[3];
            int64_t sh0 = shape[0], sh1 = shape[1], sh2 = shape[2], sh3 = shape[3];
            for (int64_t i0 = 0; i0 < sh0; ++i0) {
                int64_t offset0 = i0 * s0;
                int64_t out_offset0 = i0 * sh1 * sh2 * sh3;
                for (int64_t i1 = 0; i1 < sh1; ++i1) {
                    int64_t offset1 = offset0 + i1 * s1;
                    int64_t out_offset1 = out_offset0 + i1 * sh2 * sh3;
                    for (int64_t i2 = 0; i2 < sh2; ++i2) {
                        int64_t offset2 = offset1 + i2 * s2;
                        int64_t out_offset2 = out_offset1 + i2 * sh3;
                        for (int64_t i3 = 0; i3 < sh3; ++i3) {
                            dst_ptr[out_offset2 + i3] = src_ptr[offset2 + i3 * s3];
                        }
                    }
                }
            }
        } else if (ndims == 5) {
            int64_t s0 = strides[0], s1 = strides[1], s2 = strides[2], s3 = strides[3], s4 = strides[4];
            int64_t sh0 = shape[0], sh1 = shape[1], sh2 = shape[2], sh3 = shape[3], sh4 = shape[4];
            for (int64_t i0 = 0; i0 < sh0; ++i0) {
                int64_t offset0 = i0 * s0;
                int64_t out_offset0 = i0 * sh1 * sh2 * sh3 * sh4;
                for (int64_t i1 = 0; i1 < sh1; ++i1) {
                    int64_t offset1 = offset0 + i1 * s1;
                    int64_t out_offset1 = out_offset0 + i1 * sh2 * sh3 * sh4;
                    for (int64_t i2 = 0; i2 < sh2; ++i2) {
                        int64_t offset2 = offset1 + i2 * s2;
                        int64_t out_offset2 = out_offset1 + i2 * sh3 * sh4;
                        for (int64_t i3 = 0; i3 < sh3; ++i3) {
                            int64_t offset3 = offset2 + i3 * s3;
                            int64_t out_offset3 = out_offset2 + i3 * sh4;
                            for (int64_t i4 = 0; i4 < sh4; ++i4) {
                                dst_ptr[out_offset3 + i4] = src_ptr[offset3 + i4 * s4];
                            }
                        }
                    }
                }
            }
        } else {
            size_t num_elements = numel();
            std::vector<int64_t> coords(shape.size(), 0);
            int64_t src_offset = 0;
            for (size_t i = 0; i < num_elements; ++i) {
                dst_ptr[i] = src_ptr[src_offset];

                for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
                    coords[d]++;
                    src_offset += strides[d];
                    if (coords[d] < shape[d]) break;
                    coords[d] = 0;
                    src_offset -= shape[d] * strides[d];
                }
            }
        }
    }

    if (requires_grad) {
        auto node = std::make_shared<ContiguousNode>();
        node->inputs = { {shared_from_this(), true} };
        node->next_nodes = { creator };
        node->output = out;
        out->creator = node;
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::to(const Device& target_device) {
    if (device == target_device) return shared_from_this();
    
    auto new_storage = std::make_shared<StorageImpl>(storage->size, target_device, dtype);
    Device final_device = target_device;
    if (target_device.type == DeviceType::GPU && new_storage->device.type == DeviceType::CPU) {
        final_device = new_storage->device;
    }
    
    size_t elem_sz = storage->element_size();
    if (device.type == DeviceType::CPU && final_device.type == DeviceType::GPU) {
        if (new_storage->get_gpu_ptr()) {
            CLBackend::get().write(new_storage->get_gpu_ptr(), storage->size * elem_sz, storage->get_cpu_ptr());
        }
    } else if (device.type == DeviceType::GPU && final_device.type == DeviceType::CPU) {
        CLBackend::get().read(storage->get_gpu_ptr(), storage->size * elem_sz, new_storage->get_cpu_ptr());
    } else if (device.type == DeviceType::GPU && final_device.type == DeviceType::GPU) {
        if (storage->get_gpu_ptr() && new_storage->get_gpu_ptr()) {
            CLBackend::get().copy(storage->get_gpu_ptr(), new_storage->get_gpu_ptr(), storage->size * elem_sz);
        }
    } else {
        std::memcpy(new_storage->get_cpu_ptr(), storage->get_cpu_ptr(), storage->size * elem_sz);
    }
    
    auto new_tensor = std::make_shared<Tensor>(new_storage, shape, strides, offset, final_device, requires_grad);
    return new_tensor;
}

std::shared_ptr<Tensor> Tensor::to_device_async(const Device& target_device) {
    if (device == target_device) return shared_from_this();
    
    auto new_storage = std::make_shared<StorageImpl>(storage->size, target_device, dtype);
    Device final_device = target_device;
    if (target_device.type == DeviceType::GPU && new_storage->device.type == DeviceType::CPU) {
        final_device = new_storage->device;
    }
    
    size_t elem_sz = storage->element_size();
    if (device.type == DeviceType::CPU && final_device.type == DeviceType::GPU) {
        if (new_storage->get_gpu_ptr()) {
            CLBackend::get().write_async(new_storage->get_gpu_ptr(), storage->size * elem_sz, storage->get_cpu_ptr());
        }
    } else if (device.type == DeviceType::GPU && final_device.type == DeviceType::CPU) {
        CLBackend::get().read_async(storage->get_gpu_ptr(), storage->size * elem_sz, new_storage->get_cpu_ptr());
    } else if (device.type == DeviceType::GPU && final_device.type == DeviceType::GPU) {
        if (storage->get_gpu_ptr() && new_storage->get_gpu_ptr()) {
            CLBackend::get().copy(storage->get_gpu_ptr(), new_storage->get_gpu_ptr(), storage->size * elem_sz);
        }
    } else {
        std::memcpy(new_storage->get_cpu_ptr(), storage->get_cpu_ptr(), storage->size * elem_sz);
    }
    
    auto new_tensor = std::make_shared<Tensor>(new_storage, shape, strides, offset, final_device, requires_grad);
    return new_tensor;
}

std::shared_ptr<Tensor> Tensor::clone() {
    auto cloned = create(shape, device, requires_grad, dtype);
    size_t elem_sz = storage->element_size();
    if (is_contiguous()) {
        if (device.type == DeviceType::CPU) {
            std::memcpy(cloned->data_ptr(), data_ptr(), numel() * elem_sz);
        } else {
            cl_mem src_gpu = gpu_data();
            cl_mem dst_gpu = cloned->gpu_data();
            CLBackend::get().copy(src_gpu, dst_gpu, numel() * elem_sz, offset * elem_sz, 0);
        }
    } else {
        auto cont = contiguous();
        cloned->copy_(cont);
    }

    if (requires_grad) {
        auto node = std::make_shared<CloneNode>();
        node->inputs = { {shared_from_this(), true} };
        node->next_nodes = { creator };
        node->output = cloned;
        cloned->creator = node;
    }

    return cloned;
}

std::shared_ptr<Tensor> Tensor::view(const std::vector<int64_t>& new_shape) {
    int64_t new_numel = 1;
    for (auto d : new_shape) new_numel *= d;
    if (new_numel != static_cast<int64_t>(numel())) {
        throw std::runtime_error("Shape mismatch in view");
    }

    if (!is_contiguous()) {
        throw std::runtime_error("Tensor is not contiguous...");
    }

    auto out = std::make_shared<Tensor>(storage, new_shape, default_strides(new_shape), offset, device, requires_grad);
    if (requires_grad) {
        auto node = std::make_shared<ViewNode>(shape);
        node->inputs = { {shared_from_this(), true} };
        node->next_nodes = { creator };
        node->output = out;
        out->creator = node;
    }
    return out;
}

std::shared_ptr<Tensor> Tensor::transpose(int64_t dim0, int64_t dim1) {
    int64_t ndims = shape.size();
    if (dim0 < 0) dim0 += ndims;
    if (dim1 < 0) dim1 += ndims;

    std::vector<int64_t> new_shape = shape;
    std::vector<int64_t> new_strides = strides;

    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    auto out = std::make_shared<Tensor>(storage, new_shape, new_strides, offset, device, requires_grad);
    if (requires_grad) {
        auto node = std::make_shared<TransposeNode>(dim0, dim1);
        node->inputs = { {shared_from_this(), true} };
        node->next_nodes = { creator };
        node->output = out;
        out->creator = node;
    }
    return out;
}

float* Tensor::data_ptr() {
    return storage->get_cpu_ptr() + offset;
}

cl_mem Tensor::gpu_data() {
    return storage->get_gpu_ptr();
}

void Tensor::copy_(std::shared_ptr<Tensor> src) {
    if (shape != src->shape) {
        throw std::runtime_error("[litetorch Error] Shape mismatch in copy_");
    }
    std::shared_ptr<Tensor> src_cont = src->is_contiguous() ? src : src->contiguous();
    if (src_cont->device != device) {
        src_cont = src_cont->to(device);
    }
    if (src_cont->dtype != dtype) {
        src_cont = src_cont->cast(dtype);
    }
    size_t elem_sz = storage->element_size();
    if (!is_contiguous()) {
        if (device.type == DeviceType::GPU) {
            size_t num_elements = numel();
            int ndims_val = static_cast<int>(shape.size());
            
            int shape_arr[8] = {0};
            int strides_arr[8] = {0};
            for (size_t i = 0; i < shape.size() && i < 8; ++i) {
                shape_arr[i] = static_cast<int>(shape[i]);
                strides_arr[i] = static_cast<int>(strides[i]);
            }

            auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "copy_to_strided_kernel");
            cl_mem src_mem = src_cont->gpu_data();
            cl_mem dst_mem = gpu_data();
            int src_off = src_cont->offset;
            int dst_off = offset;
            int size_val = static_cast<int>(num_elements);

            CLBackend::get().launch(kernel, {num_elements}, {},
                {&src_mem, &src_off, &dst_mem, &dst_off, &ndims_val, &shape_arr, &strides_arr, &size_val},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(shape_arr), sizeof(strides_arr), sizeof(int)});
        } else {
            char* dest_base = (char*)storage->get_cpu_ptr();
            char* src_base = (char*)src_cont->storage->get_cpu_ptr();
            std::vector<int64_t> coords(shape.size(), 0);
            size_t total = numel();
            int64_t dest_offset = offset;
            int64_t src_offset = src_cont->offset;
            for (size_t i = 0; i < total; ++i) {
                memcpy(dest_base + dest_offset * elem_sz, src_base + (src_offset + i) * elem_sz, elem_sz);
                for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
                    coords[d]++;
                    dest_offset += strides[d];
                    if (coords[d] < shape[d]) break;
                    coords[d] = 0;
                    dest_offset -= shape[d] * strides[d];
                }
            }
        }
        return;
    }
    if (device.type == DeviceType::GPU) {
        std::lock_guard<std::mutex> storage_lock(storage->storage_mutex_);
        if (storage->cpu_data) {
            CachingAllocator::get().free_cpu(storage->cpu_data);
            storage->cpu_data = nullptr;
        }
    }
    if (device.type == DeviceType::GPU) {
        if (src_cont->device.type == DeviceType::GPU) {
            CLBackend::get().copy(src_cont->gpu_data(), gpu_data(), numel() * elem_sz, src_cont->offset * elem_sz, offset * elem_sz);
        } else {
            CLBackend::get().write(gpu_data(), numel() * elem_sz, (char*)src_cont->storage->get_cpu_ptr() + src_cont->offset * elem_sz, offset * elem_sz);
        }
    } else {
        if (src_cont->device.type == DeviceType::GPU) {
            CLBackend::get().read(src_cont->gpu_data(), numel() * elem_sz, (char*)storage->get_cpu_ptr() + offset * elem_sz, src_cont->offset * elem_sz);
        } else {
            memcpy((char*)storage->get_cpu_ptr() + offset * elem_sz,
                   (char*)src_cont->storage->get_cpu_ptr() + src_cont->offset * elem_sz,
                   numel() * elem_sz);
        }
    }
}

void Tensor::add_(std::shared_ptr<Tensor> other) {
    if (shape != other->shape) {
        throw std::runtime_error("[litetorch Error] Shape mismatch in add_");
    }
    if (other->device != device) {
        other = other->to(device);
    }
    StorageUseGuard guard({storage, other->storage});
    if (device.type == DeviceType::GPU) {
        if (is_contiguous() && other->is_contiguous()) {
            auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "elementwise_add_inplace");
            int size = numel();
            cl_mem a_mem = gpu_data();
            cl_mem b_mem = other->gpu_data();
            int a_off = offset;
            int b_off = other->offset;
            CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {}, 
                {&a_mem, &a_off, &b_mem, &b_off, &size}, 
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
        } else {
            if (is_contiguous()) {
                auto other_cont = other->contiguous();
                auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "elementwise_add_inplace");
                int size = numel();
                cl_mem a_mem = gpu_data();
                cl_mem b_mem = other_cont->gpu_data();
                int a_off = offset;
                int b_off = other_cont->offset;
                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {}, 
                    {&a_mem, &a_off, &b_mem, &b_off, &size}, 
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
            } else {
                auto this_cont = contiguous();
                auto other_cont = other->is_contiguous() ? other : other->contiguous();
                this_cont->add_(other_cont);
                copy_(this_cont);
            }
        }
    } else {
        if (is_contiguous() && other->is_contiguous()) {
            float* a_ptr = storage->get_cpu_ptr() + offset;
            const float* b_ptr = other->storage->get_cpu_ptr() + other->offset;
            int64_t size = numel();
            ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
                a_ptr[i] += b_ptr[i];
            });
        } else {
            float* a_base = storage->get_cpu_ptr();
            const float* b_base = other->storage->get_cpu_ptr();
            std::vector<int64_t> coords(shape.size(), 0);
            int64_t total = numel();
            int64_t a_offset = offset;
            int64_t b_offset = other->offset;
            for (int64_t i = 0; i < total; ++i) {
                a_base[a_offset] += b_base[b_offset];
                for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
                    coords[d]++;
                    a_offset += strides[d];
                    b_offset += other->strides[d];
                    if (coords[d] < shape[d]) break;
                    coords[d] = 0;
                    a_offset -= shape[d] * strides[d];
                    b_offset -= shape[d] * other->strides[d];
                }
            }
        }
    }
}

void Tensor::backward(std::shared_ptr<Tensor> gradient, bool create_graph) {
    if (!requires_grad) return;
    
    {
        std::lock_guard<std::mutex> lock(grad_mutex);
        if (!gradient) {
            if (numel() != 1) {
                throw std::runtime_error("Grad can only be implicitly created for scalar outputs");
            }
            gradient = from_vector({1.0f}, shape, device);
        }

        if (!grad) {
            grad = gradient;
        } else {
            if (create_graph) {
                grad = Ops::add(grad, gradient);
            } else {
                grad->add_(gradient);
            }
        }
    }

    if (creator) {
        Autograd::backward(shared_from_this(), create_graph);
    }
}

void Tensor::zero_grad() {
    grad = nullptr;
    Autograd::active_tensors.clear();
}

float Tensor::item() const {
    if (numel() != 1) {
        throw std::runtime_error("item() can only be called on 1-element tensors");
    }
    if (dtype == DataType::FP32) {
        return storage->get_cpu_ptr()[offset];
    }
    auto self_non_const = const_cast<Tensor*>(this)->shared_from_this();
    auto fp32_tensor = self_non_const->cast(DataType::FP32);
    return fp32_tensor->storage->get_cpu_ptr()[fp32_tensor->offset];
}

std::vector<float> Tensor::to_vector() const {
    if (dtype != DataType::FP32) {
        auto self_non_const = const_cast<Tensor*>(this)->shared_from_this();
        return self_non_const->cast(DataType::FP32)->to_vector();
    }
    size_t size = numel();
    std::vector<float> vec(size);
    float* src = storage->get_cpu_ptr() + offset;
    if (is_contiguous()) {
        std::copy(src, src + size, vec.begin());
    } else {
        float* dst_ptr = vec.data();
        float* src_ptr = src;
        int ndims = shape.size();
        if (ndims == 1) {
            int64_t s0 = strides[0];
            int64_t sh0 = shape[0];
            for (int64_t i0 = 0; i0 < sh0; ++i0) {
                dst_ptr[i0] = src_ptr[i0 * s0];
            }
        } else if (ndims == 2) {
            int64_t s0 = strides[0], s1 = strides[1];
            int64_t sh0 = shape[0], sh1 = shape[1];
            for (int64_t i0 = 0; i0 < sh0; ++i0) {
                int64_t offset0 = i0 * s0;
                int64_t out_offset0 = i0 * sh1;
                for (int64_t i1 = 0; i1 < sh1; ++i1) {
                    dst_ptr[out_offset0 + i1] = src_ptr[offset0 + i1 * s1];
                }
            }
        } else if (ndims == 3) {
            int64_t s0 = strides[0], s1 = strides[1], s2 = strides[2];
            int64_t sh0 = shape[0], sh1 = shape[1], sh2 = shape[2];
            for (int64_t i0 = 0; i0 < sh0; ++i0) {
                int64_t offset0 = i0 * s0;
                int64_t out_offset0 = i0 * sh1 * sh2;
                for (int64_t i1 = 0; i1 < sh1; ++i1) {
                    int64_t offset1 = offset0 + i1 * s1;
                    int64_t out_offset1 = out_offset0 + i1 * sh2;
                    for (int64_t i2 = 0; i2 < sh2; ++i2) {
                        dst_ptr[out_offset1 + i2] = src_ptr[offset1 + i2 * s2];
                    }
                }
            }
        } else if (ndims == 4) {
            int64_t s0 = strides[0], s1 = strides[1], s2 = strides[2], s3 = strides[3];
            int64_t sh0 = shape[0], sh1 = shape[1], sh2 = shape[2], sh3 = shape[3];
            for (int64_t i0 = 0; i0 < sh0; ++i0) {
                int64_t offset0 = i0 * s0;
                int64_t out_offset0 = i0 * sh1 * sh2 * sh3;
                for (int64_t i1 = 0; i1 < sh1; ++i1) {
                    int64_t offset1 = offset0 + i1 * s1;
                    int64_t out_offset1 = out_offset0 + i1 * sh2 * sh3;
                    for (int64_t i2 = 0; i2 < sh2; ++i2) {
                        int64_t offset2 = offset1 + i2 * s2;
                        int64_t out_offset2 = out_offset1 + i2 * sh3;
                        for (int64_t i3 = 0; i3 < sh3; ++i3) {
                            dst_ptr[out_offset2 + i3] = src_ptr[offset2 + i3 * s3];
                        }
                    }
                }
            }
        } else if (ndims == 5) {
            int64_t s0 = strides[0], s1 = strides[1], s2 = strides[2], s3 = strides[3], s4 = strides[4];
            int64_t sh0 = shape[0], sh1 = shape[1], sh2 = shape[2], sh3 = shape[3], sh4 = shape[4];
            for (int64_t i0 = 0; i0 < sh0; ++i0) {
                int64_t offset0 = i0 * s0;
                int64_t out_offset0 = i0 * sh1 * sh2 * sh3 * sh4;
                for (int64_t i1 = 0; i1 < sh1; ++i1) {
                    int64_t offset1 = offset0 + i1 * s1;
                    int64_t out_offset1 = out_offset0 + i1 * sh2 * sh3 * sh4;
                    for (int64_t i2 = 0; i2 < sh2; ++i2) {
                        int64_t offset2 = offset1 + i2 * s2;
                        int64_t out_offset2 = out_offset1 + i2 * sh3 * sh4;
                        for (int64_t i3 = 0; i3 < sh3; ++i3) {
                            int64_t offset3 = offset2 + i3 * s3;
                            int64_t out_offset3 = out_offset2 + i3 * sh4;
                            for (int64_t i4 = 0; i4 < sh4; ++i4) {
                                dst_ptr[out_offset3 + i4] = src_ptr[offset3 + i4 * s4];
                            }
                        }
                    }
                }
            }
        } else {
            std::vector<int64_t> coords(shape.size(), 0);
            int64_t src_offset = 0;
            for (size_t i = 0; i < size; ++i) {
                dst_ptr[i] = src_ptr[src_offset];

                for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
                    coords[d]++;
                    src_offset += strides[d];
                    if (coords[d] < shape[d]) break;
                    coords[d] = 0;
                    src_offset -= shape[d] * strides[d];
                }
            }
        }
    }
    return vec;
}

thread_local std::mt19937 g_thread_gen;
thread_local unsigned int g_thread_seed_version = 0;
static std::recursive_mutex g_seed_mutex;
static unsigned int g_seed_value = 0;
static unsigned int g_global_seed_version = 1;
static unsigned int g_thread_counter = 0;

std::mt19937& get_generator() {
    if (g_thread_seed_version != g_global_seed_version) {
        std::lock_guard<std::recursive_mutex> lock(g_seed_mutex);
        if (g_global_seed_version > 1) {
            unsigned int thread_id = ++g_thread_counter;
            std::seed_seq seq{g_seed_value, thread_id};
            g_thread_gen.seed(seq);
        } else {
            std::random_device rd;
            g_thread_gen.seed(rd());
        }
        g_thread_seed_version = g_global_seed_version;
    }
    return g_thread_gen;
}

static uint16_t float_to_half(float f) {
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

static float half_to_float(uint16_t h) {
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
        val = sign | (0xff << 23) | (mantissa << 13);
    } else {
        val = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    return *(float*)&val;
}

std::shared_ptr<Tensor> Tensor::cast(DataType target_dtype) {
    if (dtype == target_dtype) {
        return shared_from_this();
    }
    
    size_t num_elements = 1;
    for (auto dim : shape) num_elements *= dim;
    if (shape.empty()) num_elements = 0;

    if (device.type == DeviceType::GPU) {
        KernelID kernel_id;
        bool has_kernel = false;
        
        if (dtype == DataType::FP32 && target_dtype == DataType::FP16) {
            kernel_id = KernelID::CastFP32ToFP16;
            has_kernel = true;
        } else if (dtype == DataType::FP16 && target_dtype == DataType::FP32) {
            kernel_id = KernelID::CastFP16ToFP32;
            has_kernel = true;
        } else if (dtype == DataType::FP32 && target_dtype == DataType::BF16) {
            kernel_id = KernelID::CastFP32ToBF16;
            has_kernel = true;
        } else if (dtype == DataType::BF16 && target_dtype == DataType::FP32) {
            kernel_id = KernelID::CastBF16ToFP32;
            has_kernel = true;
        } else if (dtype == DataType::FP32 && target_dtype == DataType::NF4) {
            kernel_id = KernelID::CastFP32ToNF4;
            has_kernel = true;
        } else if (dtype == DataType::NF4 && target_dtype == DataType::FP32) {
            kernel_id = KernelID::CastNF4ToFP32;
            has_kernel = true;
        } else if (dtype == DataType::FP32 && target_dtype == DataType::INT8) {
            kernel_id = KernelID::CastFP32ToINT8;
            has_kernel = true;
        } else if (dtype == DataType::INT8 && target_dtype == DataType::FP32) {
            kernel_id = KernelID::CastINT8ToFP32;
            has_kernel = true;
        } else if (dtype == DataType::FP32 && target_dtype == DataType::INT4) {
            kernel_id = KernelID::CastFP32ToINT4;
            has_kernel = true;
        } else if (dtype == DataType::INT4 && target_dtype == DataType::FP32) {
            kernel_id = KernelID::CastINT4ToFP32;
            has_kernel = true;
        } else if (dtype == DataType::FP32 && target_dtype == DataType::FP8_E4M3) {
            kernel_id = KernelID::CastFP32ToFP8E4M3;
            has_kernel = true;
        } else if (dtype == DataType::FP8_E4M3 && target_dtype == DataType::FP32) {
            kernel_id = KernelID::CastFP8E4M3ToFP32;
            has_kernel = true;
        } else if (dtype == DataType::FP32 && target_dtype == DataType::FP8_E5M2) {
            kernel_id = KernelID::CastFP32ToFP8E5M2;
            has_kernel = true;
        } else if (dtype == DataType::FP8_E5M2 && target_dtype == DataType::FP32) {
            kernel_id = KernelID::CastFP8E5M2ToFP32;
            has_kernel = true;
        }
        
        if (has_kernel) {
            auto kernel = CLBackend::get().get_kernel(kernel_id);
            if (kernel) {
                auto self_c = is_contiguous() ? shared_from_this() : contiguous();
                auto out = Tensor::create(shape, device, requires_grad, target_dtype);
                cl_mem src_mem = self_c->gpu_data();
                int src_off = self_c->offset;
                cl_mem dst_mem = out->gpu_data();
                int dst_off = out->offset;
                int size = numel();
                CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
                    {&src_mem, &src_off, &dst_mem, &dst_off, &size},
                    {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int)});
                return out;
            }
        }
    }

    auto target_storage = std::make_shared<StorageImpl>(num_elements, Device(DeviceType::CPU, 0), target_dtype);
    auto out = std::make_shared<Tensor>(target_storage, shape, default_strides(shape), 0, Device(DeviceType::CPU, 0), requires_grad);
    
    storage->ensure_cpu();
    void* src_cpu = (void*)storage->get_cpu_ptr();
    void* dst_cpu = (void*)out->storage->get_cpu_ptr();
    
    if (dtype == DataType::FP32 && target_dtype == DataType::FP16) {
        float* src = (float*)src_cpu;
        uint16_t* dst = (uint16_t*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = float_to_half(src[i]);
        }
    } else if (dtype == DataType::FP16 && target_dtype == DataType::FP32) {
        uint16_t* src = (uint16_t*)src_cpu;
        float* dst = (float*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = half_to_float(src[i]);
        }
    } else if (dtype == DataType::FP32 && target_dtype == DataType::BF16) {
        float* src = (float*)src_cpu;
        uint16_t* dst = (uint16_t*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            uint32_t val = *(uint32_t*)&src[i];
            dst[i] = (uint16_t)(val >> 16);
        }
    } else if (dtype == DataType::BF16 && target_dtype == DataType::FP32) {
        uint16_t* src = (uint16_t*)src_cpu;
        float* dst = (float*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            uint32_t val = ((uint32_t)src[i]) << 16;
            dst[i] = *(float*)&val;
        }
    } else if (dtype == DataType::FP32 && target_dtype == DataType::FP64) {
        float* src = (float*)src_cpu;
        double* dst = (double*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = (double)src[i];
        }
    } else if (dtype == DataType::FP64 && target_dtype == DataType::FP32) {
        double* src = (double*)src_cpu;
        float* dst = (float*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = (float)src[i];
        }
    } else if (dtype == DataType::FP32 && target_dtype == DataType::INT16) {
        float* src = (float*)src_cpu;
        int16_t* dst = (int16_t*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = (int16_t)std::max(-32768.0f, std::min(32767.0f, src[i]));
        }
    } else if (dtype == DataType::INT16 && target_dtype == DataType::FP32) {
        int16_t* src = (int16_t*)src_cpu;
        float* dst = (float*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = (float)src[i];
        }
    } else if (dtype == DataType::FP32 && target_dtype == DataType::INT8) {
        float* src = (float*)src_cpu;
        int8_t* dst = (int8_t*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = (int8_t)std::max(-128.0f, std::min(127.0f, src[i]));
        }
    } else if (dtype == DataType::INT8 && target_dtype == DataType::FP32) {
        int8_t* src = (int8_t*)src_cpu;
        float* dst = (float*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = (float)src[i];
        }
    } else if (dtype == DataType::FP32 && target_dtype == DataType::INT4) {
        float* src = (float*)src_cpu;
        int8_t* dst = (int8_t*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = (int8_t)std::max(-8.0f, std::min(7.0f, src[i]));
        }
    } else if (dtype == DataType::INT4 && target_dtype == DataType::FP32) {
        int8_t* src = (int8_t*)src_cpu;
        float* dst = (float*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = (float)src[i];
        }
    } else if (dtype == DataType::FP32 && target_dtype == DataType::FP8_E4M3) {
        float* src = (float*)src_cpu;
        uint8_t* dst = (uint8_t*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = float_to_fp8_e4m3(src[i]);
        }
    } else if (dtype == DataType::FP8_E4M3 && target_dtype == DataType::FP32) {
        uint8_t* src = (uint8_t*)src_cpu;
        float* dst = (float*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = fp8_e4m3_to_float(src[i]);
        }
    } else if (dtype == DataType::FP32 && target_dtype == DataType::FP8_E5M2) {
        float* src = (float*)src_cpu;
        uint8_t* dst = (uint8_t*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = float_to_fp8_e5m2(src[i]);
        }
    } else if (dtype == DataType::FP8_E5M2 && target_dtype == DataType::FP32) {
        uint8_t* src = (uint8_t*)src_cpu;
        float* dst = (float*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = fp8_e5m2_to_float(src[i]);
        }
    } else if (dtype == DataType::FP32 && target_dtype == DataType::NF4) {
        float* src = (float*)src_cpu;
        uint8_t* dst = (uint8_t*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = float_to_nf4(src[i]);
        }
    } else if (dtype == DataType::NF4 && target_dtype == DataType::FP32) {
        uint8_t* src = (uint8_t*)src_cpu;
        float* dst = (float*)dst_cpu;
        for (size_t i = 0; i < numel_; ++i) {
            dst[i] = nf4_to_float(src[i]);
        }
    } else {
        throw std::runtime_error("Unsupported cast operation");
    }
    
    if (device.type == DeviceType::GPU) {
        out = out->to(device);
    }
    return out;
}

void manual_seed(unsigned int seed) {
    std::lock_guard<std::recursive_mutex> lock(g_seed_mutex);
    g_seed_value = seed;
    g_global_seed_version++;
    g_thread_counter = 0;
    
    unsigned int thread_id = ++g_thread_counter;
    std::seed_seq seq{g_seed_value, thread_id};
    g_thread_gen.seed(seq);
    g_thread_seed_version = g_global_seed_version;
}

}
