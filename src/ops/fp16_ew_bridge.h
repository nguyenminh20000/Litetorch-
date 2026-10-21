#pragma once
#include "litetorch/autograd.h"
#include "litetorch/backend_route.h"
#include <cstdio>
#include <functional>
#include <string>
#include <unordered_map>
#include <unistd.h>
#ifdef __linux__
#include <dlfcn.h>
#endif

namespace {

const char* fp16_ew_kernel_source() {
    static const char* src = R"CUDA_SRC(
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif
template<typename Op>
__device__ void binary_kernel_fp16(const __half* A, int64_t a_off, const __half* B, int64_t b_off, __half* C, int64_t c_off, int64_t size, Op op) {
    for (int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; id < size; id += (int64_t)blockDim.x * gridDim.x) {
        C[c_off + id] = op(A[a_off + id], B[b_off + id]);
    }
}
template<typename Op>
__device__ void unary_kernel_fp16(const __half* A, int64_t a_off, __half* B, int64_t b_off, int64_t size, Op op) {
    for (int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; id < size; id += (int64_t)blockDim.x * gridDim.x) {
        B[b_off + id] = op(A[a_off + id]);
    }
}
struct AddOpH { __device__ __half operator()(__half a, __half b) const { return __hadd(a, b); } };
struct SubOpH { __device__ __half operator()(__half a, __half b) const { return __hsub(a, b); } };
struct MulOpH { __device__ __half operator()(__half a, __half b) const { return __hmul(a, b); } };
struct DivOpH { __device__ __half operator()(__half a, __half b) const { return __hdiv(a, b); } };
struct ReluOpH { __device__ __half operator()(__half a) const { return __hgt(a, __half(0.0f)) ? a : __half(0.0f); } };
struct SigmoidOpH { __device__ __half operator()(__half a) const { float x = __half2float(a); return __float2half(1.0f / (1.0f + expf(-x))); } };
struct TanhOpH { __device__ __half operator()(__half a) const { float x = __half2float(a); return __float2half(tanhf(x)); } };
struct SqrtOpH { __device__ __half operator()(__half a) const { return hsqrt(a); } };
struct ExpOpH { __device__ __half operator()(__half a) const { return hexp(a); } };
struct LogOpH { __device__ __half operator()(__half a) const { float x = __half2float(a); return __float2half(logf(x)); } };
struct AbsOpH { __device__ __half operator()(__half a) const { return __habs(a); } };
struct NegOpH { __device__ __half operator()(__half a) const { return __hneg(a); } };
struct ReluBackOpH { __device__ __half operator()(__half a, __half g) const { return __hgt(a, __half(0.0f)) ? g : __half(0.0f); } };
struct SigmoidBackOpH { __device__ __half operator()(__half o, __half g) const { float of = __half2float(o); return __float2half(__half2float(g) * of * (1.0f - of)); } };
struct TanhBackOpH { __device__ __half operator()(__half o, __half g) const { float of = __half2float(o); return __float2half(__half2float(g) * (1.0f - of * of)); } };
extern "C" __global__ void elementwise_add_fp16(const __half* A, int64_t a_off, const __half* B, int64_t b_off, __half* C, int64_t c_off, int64_t size) { binary_kernel_fp16(A, a_off, B, b_off, C, c_off, size, AddOpH()); }
extern "C" __global__ void elementwise_sub_fp16(const __half* A, int64_t a_off, const __half* B, int64_t b_off, __half* C, int64_t c_off, int64_t size) { binary_kernel_fp16(A, a_off, B, b_off, C, c_off, size, SubOpH()); }
extern "C" __global__ void elementwise_mul_fp16(const __half* A, int64_t a_off, const __half* B, int64_t b_off, __half* C, int64_t c_off, int64_t size) { binary_kernel_fp16(A, a_off, B, b_off, C, c_off, size, MulOpH()); }
extern "C" __global__ void elementwise_div_fp16(const __half* A, int64_t a_off, const __half* B, int64_t b_off, __half* C, int64_t c_off, int64_t size) { binary_kernel_fp16(A, a_off, B, b_off, C, c_off, size, DivOpH()); }
extern "C" __global__ void relu_forward_fp16(const __half* A, int64_t a_off, __half* B, int64_t b_off, int64_t size) { unary_kernel_fp16(A, a_off, B, b_off, size, ReluOpH()); }
extern "C" __global__ void sigmoid_forward_fp16(const __half* A, int64_t a_off, __half* B, int64_t b_off, int64_t size) { unary_kernel_fp16(A, a_off, B, b_off, size, SigmoidOpH()); }
extern "C" __global__ void tanh_forward_fp16(const __half* A, int64_t a_off, __half* B, int64_t b_off, int64_t size) { unary_kernel_fp16(A, a_off, B, b_off, size, TanhOpH()); }
extern "C" __global__ void sqrt_forward_fp16(const __half* A, int64_t a_off, __half* B, int64_t b_off, int64_t size) { unary_kernel_fp16(A, a_off, B, b_off, size, SqrtOpH()); }
extern "C" __global__ void exp_forward_fp16(const __half* A, int64_t a_off, __half* B, int64_t b_off, int64_t size) { unary_kernel_fp16(A, a_off, B, b_off, size, ExpOpH()); }
extern "C" __global__ void log_forward_fp16(const __half* A, int64_t a_off, __half* B, int64_t b_off, int64_t size) { unary_kernel_fp16(A, a_off, B, b_off, size, LogOpH()); }
extern "C" __global__ void abs_forward_fp16(const __half* A, int64_t a_off, __half* B, int64_t b_off, int64_t size) { unary_kernel_fp16(A, a_off, B, b_off, size, AbsOpH()); }
extern "C" __global__ void neg_forward_fp16(const __half* A, int64_t a_off, __half* B, int64_t b_off, int64_t size) { unary_kernel_fp16(A, a_off, B, b_off, size, NegOpH()); }
extern "C" __global__ void relu_backward_fp16(const __half* A, int a_off, const __half* G, int64_t g_off, __half* GI, int64_t gi_off, int64_t size) { binary_kernel_fp16(A, a_off, G, g_off, GI, gi_off, size, ReluBackOpH()); }
extern "C" __global__ void sigmoid_backward_fp16(const __half* O, int o_off, const __half* G, int64_t g_off, __half* GI, int64_t gi_off, int64_t size) { binary_kernel_fp16(O, o_off, G, g_off, GI, gi_off, size, SigmoidBackOpH()); }
extern "C" __global__ void tanh_backward_fp16(const __half* O, int o_off, const __half* G, int64_t g_off, __half* GI, int64_t gi_off, int64_t size) { binary_kernel_fp16(O, o_off, G, g_off, GI, gi_off, size, TanhBackOpH()); }
)CUDA_SRC";
    return src;
}

void* fp16_ew_kernel(const char* name) {
    static std::unordered_map<std::string, void*> cache;
    static bool jit_done = false;
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    void* k = rt_gpu_get_kernel("", "", name);
    if (k) { cache[name] = k; return k; }
    if (!jit_done) {
        jit_done = true;
        const char* first = "elementwise_add_fp16";
        void* k0 = rt_gpu_get_kernel("", fp16_ew_kernel_source(), first);
        if (k0) {
            cache[first] = k0;
            char path[512];
            snprintf(path, sizeof(path), "/tmp/litetorch_jit_%s_%d.so", first, (int)getpid());
            void* handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
            if (handle) {
                const char* names[] = {
                    "elementwise_add_fp16", "elementwise_sub_fp16", "elementwise_mul_fp16",
                    "elementwise_div_fp16", "relu_forward_fp16", "sigmoid_forward_fp16",
                    "tanh_forward_fp16", "sqrt_forward_fp16", "exp_forward_fp16",
                    "log_forward_fp16", "abs_forward_fp16", "neg_forward_fp16",
                    "relu_backward_fp16", "sigmoid_backward_fp16", "tanh_backward_fp16"
                };
                for (auto n : names) {
                    void* s = dlsym(handle, n);
                    if (s) cache[n] = s;
                }
            }
        }
    }
    it = cache.find(name);
    return it != cache.end() ? it->second : nullptr;
}

struct Fp16UseGuard {
    std::vector<std::shared_ptr<StorageImpl>> storages;
    Fp16UseGuard(std::initializer_list<std::shared_ptr<StorageImpl>> list) : storages(list) {
        for (auto& s : storages) if (s) s->in_use = true;
    }
    ~Fp16UseGuard() {
        for (auto& s : storages) if (s) s->in_use = false;
    }
};

std::shared_ptr<Tensor> try_binary_fp16(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
                                        const char* kname,
                                        std::function<std::shared_ptr<Node>()> make_node,
                                        bool save_inputs) {
    if (a->dtype != DataType::FP16 || b->dtype != DataType::FP16) return nullptr;
    if (a->device.type != DeviceType::GPU || a->device != b->device) return nullptr;
    if (a->shape != b->shape || !a->is_contiguous() || !b->is_contiguous()) return nullptr;
    void* kernel = fp16_ew_kernel(kname);
    if (!kernel) return nullptr;
    auto out = Tensor::create(a->shape, a->device, false, DataType::FP16);
    Fp16UseGuard guard({a->storage, b->storage, out->storage});
    cl_mem a_mem = a->gpu_data();
    cl_mem b_mem = b->gpu_data();
    cl_mem c_mem = out->gpu_data();
    int64_t a_off = a->offset;
    int64_t b_off = b->offset;
    int64_t c_off = out->offset;
    int64_t size = out->numel();
    rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
        {&a_mem, &a_off, &b_mem, &b_off, &c_mem, &c_off, &size},
        {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t)});
    if (Autograd::is_grad_enabled() && (a->requires_grad || b->requires_grad)) {
        auto node = make_node();
        node->inputs = { {a, a->requires_grad}, {b, b->requires_grad} };
        node->next_nodes = { a->creator, b->creator };
        if (save_inputs) { node->saved_tensors = {a, b}; node->saved_versions = {(a ? a->_version() : 0), (b ? b->_version() : 0)}; }
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

std::shared_ptr<Tensor> try_unary_fp16(std::shared_ptr<Tensor> a, const char* kname,
                                       std::function<std::shared_ptr<Node>()> make_node,
                                       bool save_output) {
    if (a->dtype != DataType::FP16) return nullptr;
    if (a->device.type != DeviceType::GPU) return nullptr;
    auto a_c = a->is_contiguous() ? a : a->contiguous();
    void* kernel = fp16_ew_kernel(kname);
    if (!kernel) return nullptr;
    auto out = Tensor::create(a_c->shape, a_c->device, false, DataType::FP16);
    Fp16UseGuard guard({a_c->storage, out->storage});
    cl_mem a_mem = a_c->gpu_data();
    cl_mem b_mem = out->gpu_data();
    int64_t a_off = a_c->offset;
    int64_t b_off = out->offset;
    int64_t size = out->numel();
    rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
        {&a_mem, &a_off, &b_mem, &b_off, &size},
        {sizeof(cl_mem), sizeof(int64_t), sizeof(cl_mem), sizeof(int64_t), sizeof(int64_t)});
    if (Autograd::is_grad_enabled() && a->requires_grad) {
        auto node = make_node();
        node->inputs = { {a_c, a_c->requires_grad} };
        node->next_nodes = { a_c->creator };
        node->saved_tensors = save_output ? std::vector<std::shared_ptr<Tensor>>{out}
                                          : std::vector<std::shared_ptr<Tensor>>{a_c};
        node->saved_versions = save_output ? std::vector<uint64_t>{(out ? out->_version() : 0)}
                                           : std::vector<uint64_t>{(a_c ? a_c->_version() : 0)};
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

}
