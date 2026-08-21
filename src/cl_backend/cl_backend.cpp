#include "litetorch/cl_backend.h"
#include "litetorch/memory_manager.h"
#include "litetorch/backend.h"
#include "litetorch/allocator.h"
#include "cl_functions.h"
#include "litetorch/platform.h"
#include <iostream>
#include <stdexcept>
#include <vector>

namespace litetorch {

extern const std::string litetorch_kernels_src;

static const char* g_precompiled_kernel_names[] = {
    "elementwise_add",
    "elementwise_add_inplace",
    "elementwise_sub",
    "elementwise_mul",
    "elementwise_div",
    "pow_forward",
    "sigmoid_forward",
    "tanh_forward",
    "relu_forward",
    "gelu_forward_kernel",
    nullptr,
    nullptr,
    nullptr,
    "exp_forward",
    "log_forward",
    "sqrt_forward",
    "abs_forward",
    "neg_forward",
    "matmul_kernel",
    "bmm_kernel",
    "embedding_forward",
    "embedding_backward",
    "conv2d_kernel",
    "conv2d_backward_gdx",
    "conv2d_backward_gw",
    "conv2d_backward_gb",
    "conv3d_kernel",
    "conv3d_backward_gdx",
    "conv3d_backward_gw",
    "conv3d_backward_gb",
    "maxpool2d_kernel",
    "maxpool2d_backward_kernel",
    "maxpool3d_kernel",
    "maxpool3d_backward_kernel",
    "adaptive_avg_pool2d_forward_kernel",
    "adaptive_avg_pool2d_backward_kernel",
    "fill_zero",
    "make_contiguous_kernel",
    "sgd_step_kernel",
    "adam_step_kernel",
    "rmsprop_step_kernel",
    "relu_backward_kernel",
    "sigmoid_backward_kernel",
    "tanh_backward_kernel",
    "leaky_relu_forward",
    "leaky_relu_backward_kernel",
    "softmax_forward_kernel",
    "softmax_backward_kernel",
    "gelu_forward_kernel",
    "gelu_backward_kernel",
    "layer_norm_forward_kernel",
    "layer_norm_backward_dx_kernel",
    "layer_norm_backward_dw_kernel",
    "layer_norm_backward_db_kernel",
    "batch_norm2d_forward_stats_kernel",
    "batch_norm2d_forward_norm_kernel",
    "batch_norm2d_backward_stats_kernel",
    "batch_norm2d_backward_dx_kernel",
    "mse_loss_forward",
    "mse_loss_backward",
    "l1_loss_forward",
    "l1_loss_backward",
    "bce_loss_forward",
    "bce_loss_backward",
    "cross_entropy_loss_forward",
    "cross_entropy_loss_backward",
    "sum_forward",
    "sum_backward",
    "reduce_broadcast_prepended",
    "reduce_broadcast_dim",
    "im2col_kernel",
    "add_bias_2d",
    "fused_add_layer_norm_forward_kernel",
    "fake_quantize_forward",
    "flash_attention_forward",
    "rope_forward",
    "rope_backward",
    "paged_attention_forward",
    "w8a8_matmul_kernel",
    "cast_fp32_to_fp16",
    "cast_fp16_to_fp32",
    "cast_fp32_to_bf16",
    "cast_bf16_to_fp32",
    "cast_fp32_to_nf4",
    "cast_nf4_to_fp32",
    "cast_fp32_to_int8",
    "cast_int8_to_fp32",
    "cast_fp32_to_int4",
    "cast_int4_to_fp32",
    "cast_fp32_to_fp8_e4m3",
    "cast_fp8_e4m3_to_fp32",
    "cast_fp32_to_fp8_e5m2",
    "cast_fp8_e5m2_to_fp32"
};

CLBackend& CLBackend::get() {
    static CLBackend instance;
    return instance;
}

CLBackend::CLBackend() {
    auto native = BackendDispatcher::get().get_backend();
    if (native && native->is_available()) {
        available_ = true;
        std::cout << "[litetorch Info] GPU acceleration (Native CUDA/HIP) successfully initialized.\n";
        return;
    }
    available_ = init();
    if (!available_) {
        std::cerr << "[litetorch Warning] GPU acceleration (OpenCL) is not available. Falling back to CPU.\n";
    } else {
        std::cout << "[litetorch Info] GPU acceleration (OpenCL) successfully initialized.\n";
    }
}

CLBackend::~CLBackend() {
    shutdown();
}

bool CLBackend::init() {
    const char* libs[] = { "libOpenCL.so.1", "libOpenCL.so" };
    for (const char* lib : libs) {
        lib_handle_ = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
        if (lib_handle_) break;
    }

    if (!lib_handle_) return false;

#define RESOLVE(name) \
    p_##name = (name##_t)dlsym(lib_handle_, #name); \
    if (!p_##name) { shutdown(); return false; }

    RESOLVE(clGetPlatformIDs);
    RESOLVE(clGetDeviceIDs);
    RESOLVE(clCreateContext);
    RESOLVE(clCreateCommandQueue);
    RESOLVE(clCreateBuffer);
    RESOLVE(clReleaseMemObject);
    RESOLVE(clEnqueueReadBuffer);
    RESOLVE(clEnqueueWriteBuffer);
    RESOLVE(clEnqueueCopyBuffer);
    RESOLVE(clCreateProgramWithSource);
    RESOLVE(clBuildProgram);
    RESOLVE(clCreateKernel);
    RESOLVE(clSetKernelArg);
    RESOLVE(clEnqueueNDRangeKernel);
    RESOLVE(clFinish);
    RESOLVE(clReleaseKernel);
    RESOLVE(clReleaseProgram);
    RESOLVE(clReleaseCommandQueue);
    RESOLVE(clReleaseContext);
    RESOLVE(clGetProgramBuildInfo);
#undef RESOLVE
    p_clCreateSubBuffer = (clCreateSubBuffer_t)dlsym(lib_handle_, "clCreateSubBuffer");
    p_clGetDeviceInfo = (clGetDeviceInfo_t)dlsym(lib_handle_, "clGetDeviceInfo");

    cl_uint num_platforms = 0;
    if (p_clGetPlatformIDs(1, &platform_, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        shutdown();
        return false;
    }

    cl_uint num_devices = 0;
    cl_device_id devices[16];
    if (p_clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 16, devices, &num_devices) != CL_SUCCESS || num_devices == 0) {
        shutdown();
        return false;
    }

    int device_idx = 0;
    const char* env_idx = std::getenv("LITETORCH_DEVICE_INDEX");
    if (env_idx) {
        try {
            device_idx = std::stoi(env_idx);
        } catch (...) {
            device_idx = 0;
        }
    }
    if (device_idx < 0 || device_idx >= (int)num_devices) {
        device_idx = 0;
    }
    device_ = devices[device_idx];

    cl_int err = 0;
    context_ = p_clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        shutdown();
        return false;
    }

    queue_ = p_clCreateCommandQueue(context_, device_, 0, &err);
    if (err != CL_SUCCESS) {
        shutdown();
        return false;
    }

    cl_program program = nullptr;
    cl_int build_err = 0;
    const char* src_ptr = litetorch_kernels_src.c_str();
    size_t src_len = litetorch_kernels_src.length();
    program = p_clCreateProgramWithSource(context_, 1, &src_ptr, &src_len, &build_err);
    if (build_err == CL_SUCCESS) {
        build_err = p_clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);
        if (build_err == CL_SUCCESS) {
            programs_["litetorch_kernels"] = program;
            for (size_t i = 0; i < static_cast<size_t>(KernelID::COUNT); ++i) {
                if (g_precompiled_kernel_names[i]) {
                    cl_int k_err = 0;
                    precompiled_kernels_[i] = p_clCreateKernel(program, g_precompiled_kernel_names[i], &k_err);
                    if (k_err == CL_SUCCESS) {
                        std::string key = std::string("litetorch_kernels::") + g_precompiled_kernel_names[i];
                        kernels_[key] = precompiled_kernels_[i];
                        kernel_mutexes_[precompiled_kernels_[i]] = std::make_shared<std::mutex>();
                    } else {
                        precompiled_kernels_[i] = nullptr;
                    }
                } else {
                    precompiled_kernels_[i] = nullptr;
                }
            }
        } else {
            p_clReleaseProgram(program);
            for (size_t i = 0; i < static_cast<size_t>(KernelID::COUNT); ++i) {
                precompiled_kernels_[i] = nullptr;
            }
        }
    } else {
        for (size_t i = 0; i < static_cast<size_t>(KernelID::COUNT); ++i) {
            precompiled_kernels_[i] = nullptr;
        }
    }

    cl_uint align_bits = 1024;
    if (p_clGetDeviceInfo) {
        p_clGetDeviceInfo(device_, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &align_bits, nullptr);
    }
    mem_alignment_bytes_ = align_bits / 8;
    if (mem_alignment_bytes_ == 0) mem_alignment_bytes_ = 128;

    return true;
}

void CLBackend::shutdown() {
    for (auto& pair : kernels_) {
        if (pair.second) p_clReleaseKernel(pair.second);
    }
    kernels_.clear();

    for (auto& pair : programs_) {
        if (pair.second) p_clReleaseProgram(pair.second);
    }
    programs_.clear();

    for (auto& pair : active_sub_buffers_) {
        p_clReleaseMemObject(pair.first);
    }
    active_sub_buffers_.clear();

    for (auto& sb : super_blocks_) {
        if (sb->buffer) {
            p_clReleaseMemObject(sb->buffer);
        }
    }
    super_blocks_.clear();

    for (auto& pair : free_buffers_) {
        p_clReleaseMemObject(pair.second);
    }
    free_buffers_.clear();

    for (auto& pair : allocated_sizes_) {
        p_clReleaseMemObject(pair.first);
    }
    allocated_sizes_.clear();

    if (queue_) { p_clReleaseCommandQueue(queue_); queue_ = nullptr; }
    if (context_) { p_clReleaseContext(context_); context_ = nullptr; }
    if (lib_handle_) { dlclose(lib_handle_); lib_handle_ = nullptr; }
    
    platform_ = nullptr;
    device_ = nullptr;
    available_ = false;
}

cl_mem CLBackend::allocate(size_t size) {
    auto native = BackendDispatcher::get().get_backend();
    if (native && native->is_available()) {
        return (cl_mem)CachingAllocator::get().allocate_gpu(size);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!available_) return nullptr;

    size_t aligned_size = ((size + mem_alignment_bytes_ - 1) / mem_alignment_bytes_) * mem_alignment_bytes_;

    for (auto& sb : super_blocks_) {
        for (auto it = sb->blocks.begin(); it != sb->blocks.end(); ++it) {
            if (it->is_free && it->size >= aligned_size) {
                it->is_free = false;
                if (it->size > aligned_size) {
                    Block remaining;
                    remaining.offset = it->offset + aligned_size;
                    remaining.size = it->size - aligned_size;
                    remaining.is_free = true;
                    remaining.sub_buffer = nullptr;
                    remaining.parent = sb.get();
                    sb->blocks.insert(std::next(it), remaining);
                    it->size = aligned_size;
                }
                cl_buffer_region region = { it->offset, it->size };
                cl_int err = 0;
                cl_mem sub_buf = nullptr;
                if (p_clCreateSubBuffer) {
                    sub_buf = p_clCreateSubBuffer(sb->buffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
                }
                if (err == CL_SUCCESS && sub_buf) {
                    it->sub_buffer = sub_buf;
                    active_sub_buffers_[sub_buf] = &(*it);
                    return sub_buf;
                }
                it->is_free = true;
            }
        }
    }

    size_t sb_size = std::max(size_t(16 * 1024 * 1024), aligned_size);
    cl_int err = 0;
    cl_mem raw_buffer = p_clCreateBuffer(context_, CL_MEM_READ_WRITE, sb_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        for (auto it = super_blocks_.begin(); it != super_blocks_.end(); ) {
            if ((*it)->blocks.size() == 1 && (*it)->blocks.front().is_free) {
                p_clReleaseMemObject((*it)->buffer);
                total_device_allocated_ -= (*it)->size;
                it = super_blocks_.erase(it);
            } else {
                ++it;
            }
        }
        sb_size = aligned_size;
        raw_buffer = p_clCreateBuffer(context_, CL_MEM_READ_WRITE, sb_size, nullptr, &err);
    }

    if (err == CL_SUCCESS && raw_buffer) {
        total_device_allocated_ += sb_size;
        auto sb = std::make_unique<SuperBlock>();
        sb->buffer = raw_buffer;
        sb->size = sb_size;

        Block initial_block;
        initial_block.offset = 0;
        initial_block.size = sb_size;
        initial_block.is_free = true;
        initial_block.sub_buffer = nullptr;
        initial_block.parent = sb.get();
        sb->blocks.push_back(initial_block);

        super_blocks_.push_back(std::move(sb));

        auto& new_sb = super_blocks_.back();
        auto it = new_sb->blocks.begin();
        it->is_free = false;
        if (it->size > aligned_size) {
            Block remaining;
            remaining.offset = aligned_size;
            remaining.size = it->size - aligned_size;
            remaining.is_free = true;
            remaining.sub_buffer = nullptr;
            remaining.parent = new_sb.get();
            new_sb->blocks.insert(std::next(it), remaining);
            it->size = aligned_size;
        }

        cl_buffer_region region = { it->offset, it->size };
        cl_mem sub_buf = nullptr;
        if (p_clCreateSubBuffer) {
            sub_buf = p_clCreateSubBuffer(new_sb->buffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        }
        if (err == CL_SUCCESS && sub_buf) {
            it->sub_buffer = sub_buf;
            active_sub_buffers_[sub_buf] = &(*it);
            return sub_buf;
        }
    }

    cl_int direct_err = 0;
    cl_mem direct_buf = p_clCreateBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, &direct_err);
    if (direct_err == CL_SUCCESS && direct_buf) {
        return direct_buf;
    }
    return nullptr;
}

void CLBackend::free(cl_mem buffer) {
    auto native = BackendDispatcher::get().get_backend();
    if (native && native->is_available()) {
        CachingAllocator::get().free_gpu(buffer);
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!buffer || !available_) return;

    auto it_active = active_sub_buffers_.find(buffer);
    if (it_active != active_sub_buffers_.end()) {
        Block* block = it_active->second;
        active_sub_buffers_.erase(it_active);

        p_clReleaseMemObject(buffer);
        block->sub_buffer = nullptr;
        block->is_free = true;

        SuperBlock* sb = block->parent;
        auto bit = sb->blocks.begin();
        while (bit != sb->blocks.end()) {
            auto next_bit = std::next(bit);
            if (next_bit != sb->blocks.end() && bit->is_free && next_bit->is_free) {
                bit->size += next_bit->size;
                sb->blocks.erase(next_bit);
            } else {
                ++bit;
            }
        }

        evict_cache_to_limit();
    } else {
        p_clReleaseMemObject(buffer);
    }
}

void CLBackend::evict_cache_to_limit() {
    size_t limit = MemoryManager::get().get_gpu_limit();
    for (auto it = super_blocks_.begin(); it != super_blocks_.end(); ) {
        if (total_device_allocated_ <= limit) break;
        if ((*it)->blocks.size() == 1 && (*it)->blocks.front().is_free) {
            p_clReleaseMemObject((*it)->buffer);
            total_device_allocated_ -= (*it)->size;
            it = super_blocks_.erase(it);
        } else {
            ++it;
        }
    }
}

void CLBackend::clear_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = super_blocks_.begin(); it != super_blocks_.end(); ) {
        if ((*it)->blocks.size() == 1 && (*it)->blocks.front().is_free) {
            p_clReleaseMemObject((*it)->buffer);
            total_device_allocated_ -= (*it)->size;
            it = super_blocks_.erase(it);
        } else {
            ++it;
        }
    }
}

void CLBackend::read(cl_mem buffer, size_t size, void* host_ptr, size_t buffer_offset) {
    auto native = BackendDispatcher::get().get_backend();
    if (native && native->is_available()) {
        native->read(buffer, size, host_ptr, buffer_offset);
        return;
    }
    if (!available_ || !buffer) return;
    cl_int err = p_clEnqueueReadBuffer(queue_, buffer, 1, buffer_offset, size, host_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("[litetorch Error] clEnqueueReadBuffer failed: " + std::to_string(err));
    }
}

void CLBackend::write(cl_mem buffer, size_t size, const void* host_ptr, size_t buffer_offset) {
    auto native = BackendDispatcher::get().get_backend();
    if (native && native->is_available()) {
        native->write(buffer, size, host_ptr, buffer_offset);
        return;
    }
    if (!available_ || !buffer) return;
    cl_int err = p_clEnqueueWriteBuffer(queue_, buffer, 1, buffer_offset, size, host_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("[litetorch Error] clEnqueueWriteBuffer failed: " + std::to_string(err));
    }
}

void CLBackend::read_async(cl_mem buffer, size_t size, void* host_ptr, size_t buffer_offset) {
    auto native = BackendDispatcher::get().get_backend();
    if (native && native->is_available()) {
        native->read_async(buffer, size, host_ptr, buffer_offset);
        return;
    }
    if (!available_ || !buffer) return;
    cl_int err = p_clEnqueueReadBuffer(queue_, buffer, 0, buffer_offset, size, host_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("[litetorch Error] clEnqueueReadBuffer (async) failed: " + std::to_string(err));
    }
}

void CLBackend::write_async(cl_mem buffer, size_t size, const void* host_ptr, size_t buffer_offset) {
    auto native = BackendDispatcher::get().get_backend();
    if (native && native->is_available()) {
        native->write_async(buffer, size, host_ptr, buffer_offset);
        return;
    }
    if (!available_ || !buffer) return;
    cl_int err = p_clEnqueueWriteBuffer(queue_, buffer, 0, buffer_offset, size, host_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("[litetorch Error] clEnqueueWriteBuffer (async) failed: " + std::to_string(err));
    }
}

void CLBackend::copy(cl_mem src, cl_mem dst, size_t size, size_t src_offset, size_t dst_offset) {
    auto native = BackendDispatcher::get().get_backend();
    if (native && native->is_available()) {
        native->copy(src, dst, size, src_offset, dst_offset);
        return;
    }
    if (!available_ || !src || !dst) return;
    cl_int err = p_clEnqueueCopyBuffer(queue_, src, dst, src_offset, dst_offset, size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("[litetorch Error] clEnqueueCopyBuffer failed: " + std::to_string(err));
    }
}

cl_kernel CLBackend::get_kernel(KernelID id) {
    auto native = BackendDispatcher::get().get_backend();
    if (native && native->is_available()) {
        std::string name = g_precompiled_kernel_names[static_cast<size_t>(id)];
        return (cl_kernel)native->get_kernel("", "", name);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    cl_kernel k = precompiled_kernels_[static_cast<size_t>(id)];
    if (k && kernel_mutexes_.find(k) == kernel_mutexes_.end()) {
        kernel_mutexes_[k] = std::make_shared<std::mutex>();
    }
    return k;
}

cl_kernel CLBackend::get_kernel(const std::string& program_name, const std::string& program_source, const std::string& kernel_name) {
    auto native = BackendDispatcher::get().get_backend();
    if (native && native->is_available()) {
        return (cl_kernel)native->get_kernel(program_name, program_source, kernel_name);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = program_name + "::" + kernel_name;
    if (kernels_.find(key) != kernels_.end()) {
        return kernels_[key];
    }

    cl_program program = nullptr;
    if (programs_.find(program_name) != programs_.end()) {
        program = programs_[program_name];
    } else {
        if (program_source.empty()) {
            return nullptr;
        }
        cl_int err = 0;
        const char* src_ptr = program_source.c_str();
        size_t src_len = program_source.length();
        program = p_clCreateProgramWithSource(context_, 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS) {
            return nullptr;
        }

        err = p_clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            p_clReleaseProgram(program);
            return nullptr;
        }
        programs_[program_name] = program;
    }

    cl_int err = 0;
    cl_kernel kernel = p_clCreateKernel(program, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS) {
        return nullptr;
    }

    kernels_[key] = kernel;
    kernel_mutexes_[kernel] = std::make_shared<std::mutex>();
    return kernel;
}

void CLBackend::launch(cl_kernel kernel, const std::vector<size_t>& global_work_size, const std::vector<size_t>& local_work_size, const std::vector<void*>& args, const std::vector<size_t>& arg_sizes) {
    auto native = BackendDispatcher::get().get_backend();
    if (native && native->is_available()) {
        native->launch(kernel, global_work_size, local_work_size, args, arg_sizes);
        return;
    }
    if (!available_) return;

    if (active_graph && active_graph->recording) {
        OpenCLCommand cmd;
        cmd.kernel = kernel;
        cmd.global_work_size = global_work_size;
        cmd.local_work_size = local_work_size;
        cmd.arg_sizes = arg_sizes;
        for (size_t i = 0; i < args.size(); ++i) {
            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(args[i]);
            cmd.arg_data.push_back(std::vector<uint8_t>(ptr, ptr + arg_sizes[i]));
        }
        active_graph->commands.push_back(cmd);
    }

    std::shared_ptr<std::mutex> k_mutex;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = kernel_mutexes_.find(kernel);
        if (it != kernel_mutexes_.end()) {
            k_mutex = it->second;
        }
    }

    std::unique_lock<std::mutex> k_lock;
    if (k_mutex) {
        k_lock = std::unique_lock<std::mutex>(*k_mutex);
    }

    for (cl_uint i = 0; i < args.size(); ++i) {
        cl_int err = p_clSetKernelArg(kernel, i, arg_sizes[i], args[i]);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("[litetorch Error] clSetKernelArg failed for argument " + std::to_string(i));
        }
    }

    const size_t* local_ws_ptr = local_work_size.empty() ? nullptr : local_work_size.data();
    cl_int err = p_clEnqueueNDRangeKernel(queue_, kernel, global_work_size.size(), nullptr, global_work_size.data(), local_ws_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("[litetorch Error] clEnqueueNDRangeKernel failed: " + std::to_string(err));
    }
}

std::shared_ptr<std::mutex> CLBackend::get_kernel_mutex(cl_kernel kernel) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = kernel_mutexes_.find(kernel);
    if (it != kernel_mutexes_.end()) {
        return it->second;
    }
    auto mtx = std::make_shared<std::mutex>();
    kernel_mutexes_[kernel] = mtx;
    return mtx;
}

void CLBackend::finish() {
    auto native = BackendDispatcher::get().get_backend();
    if (native && native->is_available()) {
        native->finish();
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!available_ || !queue_) return;
    p_clFinish(queue_);
}

}
