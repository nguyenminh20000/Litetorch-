#include "gpu_common.h"
#include <climits>
#include <stdexcept>

template<typename Op>
__device__ void vectorized_binary_kernel(const float* A, int64_t a_off, const float* B, int64_t b_off, float* C, int64_t c_off, int64_t size, Op op) {
    uintptr_t a_addr = reinterpret_cast<uintptr_t>(A + a_off);
    uintptr_t b_addr = reinterpret_cast<uintptr_t>(B + b_off);
    uintptr_t c_addr = reinterpret_cast<uintptr_t>(C + c_off);
    if ((a_addr % 16 == 0) && (b_addr % 16 == 0) && (c_addr % 16 == 0)) {
        int64_t vec_size = size / 4;
        for (int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; id < vec_size; id += (int64_t)blockDim.x * gridDim.x) {
            float4 a_vec = reinterpret_cast<const float4*>(A + a_off)[id];
            float4 b_vec = reinterpret_cast<const float4*>(B + b_off)[id];
            float4 c_vec;
            c_vec.x = op(a_vec.x, b_vec.x);
            c_vec.y = op(a_vec.y, b_vec.y);
            c_vec.z = op(a_vec.z, b_vec.z);
            c_vec.w = op(a_vec.w, b_vec.w);
            reinterpret_cast<float4*>(C + c_off)[id] = c_vec;
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            for (int64_t i = 0; i < size % 4; ++i) {
                int64_t idx = vec_size * 4 + i;
                C[c_off + idx] = op(A[a_off + idx], B[b_off + idx]);
            }
        }
    } else {
        for (int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; id < size; id += (int64_t)blockDim.x * gridDim.x) {
            C[c_off + id] = op(A[a_off + id], B[b_off + id]);
        }
    }
}

template<typename Op>
__device__ void vectorized_unary_kernel(const float* A, int64_t a_off, float* B, int64_t b_off, int64_t size, Op op) {
    uintptr_t a_addr = reinterpret_cast<uintptr_t>(A + a_off);
    uintptr_t b_addr = reinterpret_cast<uintptr_t>(B + b_off);
    if ((a_addr % 16 == 0) && (b_addr % 16 == 0)) {
        int64_t vec_size = size / 4;
        for (int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; id < vec_size; id += (int64_t)blockDim.x * gridDim.x) {
            float4 a_vec = reinterpret_cast<const float4*>(A + a_off)[id];
            float4 b_vec;
            b_vec.x = op(a_vec.x);
            b_vec.y = op(a_vec.y);
            b_vec.z = op(a_vec.z);
            b_vec.w = op(a_vec.w);
            reinterpret_cast<float4*>(B + b_off)[id] = b_vec;
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            for (int64_t i = 0; i < size % 4; ++i) {
                int64_t idx = vec_size * 4 + i;
                B[b_off + idx] = op(A[a_off + idx]);
            }
        }
    } else {
        for (int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; id < size; id += (int64_t)blockDim.x * gridDim.x) {
            B[b_off + id] = op(A[a_off + id]);
        }
    }
}

struct AddOp { __device__ float operator()(float a, float b) const { return a + b; } };
struct SubOp { __device__ float operator()(float a, float b) const { return a - b; } };
struct MulOp { __device__ float operator()(float a, float b) const { return a * b; } };
struct DivOp { __device__ float operator()(float a, float b) const { return a / b; } };
struct ReluOp { __device__ float operator()(float a) const { return a > 0.0f ? a : 0.0f; } };
struct ReluBackOp { __device__ float operator()(float a, float g) const { return a > 0.0f ? g : 0.0f; } };
struct SigmoidOp { __device__ float operator()(float a) const { return 1.0f / (1.0f + expf(-a)); } };
struct TanhOp { __device__ float operator()(float a) const { return tanhf(a); } };
struct PowOp { float p; __device__ float operator()(float a) const { return powf(a, p); } };
struct SqrtOp { __device__ float operator()(float a) const { return sqrtf(a); } };
struct ExpOp { __device__ float operator()(float a) const { return expf(a); } };
struct LogOp { __device__ float operator()(float a) const { return logf(a); } };
struct AbsOp { __device__ float operator()(float a) const { return fabsf(a); } };
struct NegOp { __device__ float operator()(float a) const { return -a; } };
struct LeakyReluOp { float s; __device__ float operator()(float a) const { return a > 0.0f ? a : a * s; } };
struct SigmoidBackOp { __device__ float operator()(float o, float g) const { return g * o * (1.0f - o); } };
struct TanhBackOp { __device__ float operator()(float o, float g) const { return g * (1.0f - o * o); } };
struct LeakyReluBackOp { float s; __device__ float operator()(float x, float g) const { return x > 0.0f ? g : g * s; } };

extern "C" __global__ void elementwise_add(const float* A, int64_t a_off, const float* B, int64_t b_off, float* C, int64_t c_off, int64_t size) { vectorized_binary_kernel(A, a_off, B, b_off, C, c_off, size, AddOp()); }
extern "C" __global__ void elementwise_add_inplace(float* A, int64_t a_off, const float* B, int64_t b_off, int64_t size) { vectorized_binary_kernel(A, a_off, B, b_off, A, a_off, size, AddOp()); }
extern "C" __global__ void elementwise_sub(const float* A, int64_t a_off, const float* B, int64_t b_off, float* C, int64_t c_off, int64_t size) { vectorized_binary_kernel(A, a_off, B, b_off, C, c_off, size, SubOp()); }
extern "C" __global__ void elementwise_mul(const float* A, int64_t a_off, const float* B, int64_t b_off, float* C, int64_t c_off, int64_t size) { vectorized_binary_kernel(A, a_off, B, b_off, C, c_off, size, MulOp()); }
extern "C" __global__ void elementwise_div(const float* A, int64_t a_off, const float* B, int64_t b_off, float* C, int64_t c_off, int64_t size) { vectorized_binary_kernel(A, a_off, B, b_off, C, c_off, size, DivOp()); }
extern "C" __global__ void relu_forward(const float* A, int64_t a_off, float* B, int64_t b_off, int64_t size) { vectorized_unary_kernel(A, a_off, B, b_off, size, ReluOp()); }
extern "C" __global__ void relu_backward_kernel(const float* A, int64_t a_off, const float* grad_output, int64_t gout_off, float* grad_input, int64_t gin_off, int64_t size) { vectorized_binary_kernel(A, a_off, grad_output, gout_off, grad_input, gin_off, size, ReluBackOp()); }
extern "C" __global__ void sigmoid_forward(const float* A, int64_t a_off, float* B, int64_t b_off, int64_t size) { vectorized_unary_kernel(A, a_off, B, b_off, size, SigmoidOp()); }
extern "C" __global__ void tanh_forward(const float* A, int64_t a_off, float* B, int64_t b_off, int64_t size) { vectorized_unary_kernel(A, a_off, B, b_off, size, TanhOp()); }
extern "C" __global__ void pow_forward(const float* A, int64_t a_off, float exponent, float* B, int64_t b_off, int64_t size) { vectorized_unary_kernel(A, a_off, B, b_off, size, PowOp{exponent}); }
extern "C" __global__ void sqrt_forward(const float* A, int64_t a_off, float* B, int64_t b_off, int64_t size) { vectorized_unary_kernel(A, a_off, B, b_off, size, SqrtOp()); }
extern "C" __global__ void exp_forward(const float* A, int64_t a_off, float* B, int64_t b_off, int64_t size) { vectorized_unary_kernel(A, a_off, B, b_off, size, ExpOp()); }
extern "C" __global__ void log_forward(const float* A, int64_t a_off, float* B, int64_t b_off, int64_t size) { vectorized_unary_kernel(A, a_off, B, b_off, size, LogOp()); }
extern "C" __global__ void abs_forward(const float* A, int64_t a_off, float* B, int64_t b_off, int64_t size) { vectorized_unary_kernel(A, a_off, B, b_off, size, AbsOp()); }
extern "C" __global__ void neg_forward(const float* A, int64_t a_off, float* B, int64_t b_off, int64_t size) { vectorized_unary_kernel(A, a_off, B, b_off, size, NegOp()); }
extern "C" __global__ void leaky_relu_forward(const float* A, int64_t a_off, float negative_slope, float* B, int64_t b_off, int64_t size) { vectorized_unary_kernel(A, a_off, B, b_off, size, LeakyReluOp{negative_slope}); }
extern "C" __global__ void sigmoid_backward_kernel(const float* out_data, int64_t out_off, const float* grad_output, int64_t gout_off, float* grad_input, int64_t gin_off, int64_t size) { vectorized_binary_kernel(out_data, out_off, grad_output, gout_off, grad_input, gin_off, size, SigmoidBackOp()); }
extern "C" __global__ void tanh_backward_kernel(const float* out_data, int64_t out_off, const float* grad_output, int64_t gout_off, float* grad_input, int64_t gin_off, int64_t size) { vectorized_binary_kernel(out_data, out_off, grad_output, gout_off, grad_input, gin_off, size, TanhBackOp()); }
extern "C" __global__ void leaky_relu_backward_kernel(const float* A, int64_t a_off, const float* grad_output, int64_t gout_off, float* grad_input, int64_t gin_off, int64_t size, float negative_slope) { vectorized_binary_kernel(A, a_off, grad_output, gout_off, grad_input, gin_off, size, LeakyReluBackOp{negative_slope}); }

extern "C" __global__ void reduce_broadcast_prepended(const float* A, int64_t a_off,
                                         float* B, int64_t b_off,
                                         int64_t prod_prepended, int64_t remaining) {
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j < remaining) {
        float s = 0.0f;
        for (int64_t i = 0; i < prod_prepended; ++i) {
            s += A[a_off + i * remaining + j];
        }
        B[b_off + j] = s;
    }
}

extern "C" __global__ void reduce_broadcast_dim(const float* A, int64_t a_off,
                                   float* B, int64_t b_off,
                                   int64_t outer_size, int64_t dim_size, int64_t inner_size) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * inner_size;
    if (idx < total) {
        int64_t o = idx / inner_size;
        int64_t in = idx % inner_size;
        float s = 0.0f;
        for (int64_t d = 0; d < dim_size; ++d) {
            s += A[a_off + (o * dim_size + d) * inner_size + in];
        }
        B[b_off + o * inner_size + in] = s;
    }
}

extern "C" void gpu_reduce_broadcast_prepended(
    void* A, int64_t a_off, void* B, int64_t b_off, int64_t prod_prepended, int64_t remaining) {
    auto_set_device(A);
    if (remaining <= 0) return;
    int threads = 256;
    int64_t blocks64 = (remaining + threads - 1) / threads;
    if (blocks64 > INT_MAX) throw std::runtime_error("gpu_reduce_broadcast_prepended: grid exceeds INT_MAX blocks");
    int blocks = (int)blocks64;
#ifndef __HIP_PLATFORM_AMD__
    reduce_broadcast_prepended<<<blocks, threads, 0, dev_stream(current_device())>>>(
        (const float*)A, a_off, (float*)B, b_off, prod_prepended, remaining);
#else
    hipLaunchKernelGGL(reduce_broadcast_prepended, dim3(blocks), dim3(threads), 0, dev_stream(current_device()),
        (const float*)A, a_off, (float*)B, b_off, prod_prepended, remaining);
#endif
}

extern "C" void gpu_reduce_broadcast_dim(
    void* A, int64_t a_off, void* B, int64_t b_off, int64_t outer_size, int64_t dim_size, int64_t inner_size) {
    auto_set_device(A);
    int64_t total = outer_size * inner_size;
    if (total <= 0) return;
    int threads = 256;
    int64_t blocks64 = (total + threads - 1) / threads;
    if (blocks64 > INT_MAX) throw std::runtime_error("gpu_reduce_broadcast_dim: grid exceeds INT_MAX blocks");
    int blocks = (int)blocks64;
#ifndef __HIP_PLATFORM_AMD__
    reduce_broadcast_dim<<<blocks, threads, 0, dev_stream(current_device())>>>(
        (const float*)A, a_off, (float*)B, b_off, outer_size, dim_size, inner_size);
#else
    hipLaunchKernelGGL(reduce_broadcast_dim, dim3(blocks), dim3(threads), 0, dev_stream(current_device()),
        (const float*)A, a_off, (float*)B, b_off, outer_size, dim_size, inner_size);
#endif
}

struct Int8Val {
    int s0, s1, s2, s3, s4, s5, s6, s7;
};

extern "C" __global__ void make_contiguous_kernel(
    const float* src, int64_t src_offset,
    float* dst, int64_t dst_offset,
    int ndims,
    Int8Val shape_val,
    Int8Val strides_val,
    int64_t total_elements)
{
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    int shape[8] = { shape_val.s0, shape_val.s1, shape_val.s2, shape_val.s3, shape_val.s4, shape_val.s5, shape_val.s6, shape_val.s7 };
    int strides[8] = { strides_val.s0, strides_val.s1, strides_val.s2, strides_val.s3, strides_val.s4, strides_val.s5, strides_val.s6, strides_val.s7 };

    int64_t src_idx = src_offset;
    int64_t temp = gid;
    for (int d = ndims - 1; d >= 0; --d) {
        int64_t coord = temp % shape[d];
        temp /= shape[d];
        src_idx += coord * strides[d];
    }
    dst[dst_offset + gid] = src[src_idx];
}

extern "C" __global__ void make_contiguous_kernel_fp16(
    const __half* src, int64_t src_offset,
    __half* dst, int64_t dst_offset,
    int ndims,
    Int8Val shape_val,
    Int8Val strides_val,
    int64_t total_elements)
{
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    int shape[8] = { shape_val.s0, shape_val.s1, shape_val.s2, shape_val.s3, shape_val.s4, shape_val.s5, shape_val.s6, shape_val.s7 };
    int strides[8] = { strides_val.s0, strides_val.s1, strides_val.s2, strides_val.s3, strides_val.s4, strides_val.s5, strides_val.s6, strides_val.s7 };

    int64_t src_idx = src_offset;
    int64_t temp = gid;
    for (int d = ndims - 1; d >= 0; --d) {
        int64_t coord = temp % shape[d];
        temp /= shape[d];
        src_idx += coord * strides[d];
    }
    dst[dst_offset + gid] = src[src_idx];
}

extern "C" __global__ void copy_to_strided_kernel(
    const float* src, int64_t src_offset,
    float* dst, int64_t dst_offset,
    int ndims,
    Int8Val shape_val,
    Int8Val strides_val,
    int64_t total_elements)
{
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    int shape[8] = { shape_val.s0, shape_val.s1, shape_val.s2, shape_val.s3, shape_val.s4, shape_val.s5, shape_val.s6, shape_val.s7 };
    int strides[8] = { strides_val.s0, strides_val.s1, strides_val.s2, strides_val.s3, strides_val.s4, strides_val.s5, strides_val.s6, strides_val.s7 };

    int64_t dst_idx = dst_offset;
    int64_t temp = gid;
    for (int d = ndims - 1; d >= 0; --d) {
        int64_t coord = temp % shape[d];
        temp /= shape[d];
        dst_idx += coord * strides[d];
    }
    dst[dst_idx] = src[src_offset + gid];
}

extern "C" __global__ void copy_to_strided_kernel_fp16(
    const __half* src, int64_t src_offset,
    __half* dst, int64_t dst_offset,
    int ndims,
    Int8Val shape_val,
    Int8Val strides_val,
    int64_t total_elements)
{
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    int shape[8] = { shape_val.s0, shape_val.s1, shape_val.s2, shape_val.s3, shape_val.s4, shape_val.s5, shape_val.s6, shape_val.s7 };
    int strides[8] = { strides_val.s0, strides_val.s1, strides_val.s2, strides_val.s3, strides_val.s4, strides_val.s5, strides_val.s6, strides_val.s7 };

    int64_t dst_idx = dst_offset;
    int64_t temp = gid;
    for (int d = ndims - 1; d >= 0; --d) {
        int64_t coord = temp % shape[d];
        temp /= shape[d];
        dst_idx += coord * strides[d];
    }
    dst[dst_idx] = src[src_offset + gid];
}

extern "C" __global__ void elementwise_broadcast_add(
    const float* A, int64_t a_off,
    const float* B, int64_t b_off,
    float* C, int64_t c_off,
    int ndims,
    const int* out_shape,
    int a_ndims, const int* a_shape, const int* a_strides,
    int b_ndims, const int* b_shape, const int* b_strides,
    int64_t total_elements)
{
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    int64_t temp = gid;
    int coords[16];
    for (int d = ndims - 1; d >= 0; --d) {
        coords[d] = (int)(temp % out_shape[d]);
        temp /= out_shape[d];
    }

    int64_t a_idx = a_off;
    int a_diff = ndims - a_ndims;
    for (int i = 0; i < a_ndims; ++i) {
        if (a_shape[i] != 1) {
            a_idx += (int64_t)coords[i + a_diff] * a_strides[i];
        }
    }

    int64_t b_idx = b_off;
    int b_diff = ndims - b_ndims;
    for (int i = 0; i < b_ndims; ++i) {
        if (b_shape[i] != 1) {
            b_idx += (int64_t)coords[i + b_diff] * b_strides[i];
        }
    }

    C[c_off + gid] = A[a_idx] + B[b_idx];
}

extern "C" __global__ void elementwise_broadcast_sub(
    const float* A, int64_t a_off,
    const float* B, int64_t b_off,
    float* C, int64_t c_off,
    int ndims,
    const int* out_shape,
    int a_ndims, const int* a_shape, const int* a_strides,
    int b_ndims, const int* b_shape, const int* b_strides,
    int64_t total_elements)
{
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    int64_t temp = gid;
    int coords[16];
    for (int d = ndims - 1; d >= 0; --d) {
        coords[d] = (int)(temp % out_shape[d]);
        temp /= out_shape[d];
    }

    int64_t a_idx = a_off;
    int a_diff = ndims - a_ndims;
    for (int i = 0; i < a_ndims; ++i) {
        if (a_shape[i] != 1) {
            a_idx += (int64_t)coords[i + a_diff] * a_strides[i];
        }
    }

    int64_t b_idx = b_off;
    int b_diff = ndims - b_ndims;
    for (int i = 0; i < b_ndims; ++i) {
        if (b_shape[i] != 1) {
            b_idx += (int64_t)coords[i + b_diff] * b_strides[i];
        }
    }

    C[c_off + gid] = A[a_idx] - B[b_idx];
}

extern "C" __global__ void elementwise_broadcast_mul(
    const float* A, int64_t a_off,
    const float* B, int64_t b_off,
    float* C, int64_t c_off,
    int ndims,
    const int* out_shape,
    int a_ndims, const int* a_shape, const int* a_strides,
    int b_ndims, const int* b_shape, const int* b_strides,
    int64_t total_elements)
{
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    int64_t temp = gid;
    int coords[16];
    for (int d = ndims - 1; d >= 0; --d) {
        coords[d] = (int)(temp % out_shape[d]);
        temp /= out_shape[d];
    }

    int64_t a_idx = a_off;
    int a_diff = ndims - a_ndims;
    for (int i = 0; i < a_ndims; ++i) {
        if (a_shape[i] != 1) {
            a_idx += (int64_t)coords[i + a_diff] * a_strides[i];
        }
    }

    int64_t b_idx = b_off;
    int b_diff = ndims - b_ndims;
    for (int i = 0; i < b_ndims; ++i) {
        if (b_shape[i] != 1) {
            b_idx += (int64_t)coords[i + b_diff] * b_strides[i];
        }
    }

    C[c_off + gid] = A[a_idx] * B[b_idx];
}

extern "C" __global__ void elementwise_broadcast_div(
    const float* A, int64_t a_off,
    const float* B, int64_t b_off,
    float* C, int64_t c_off,
    int ndims,
    const int* out_shape,
    int a_ndims, const int* a_shape, const int* a_strides,
    int b_ndims, const int* b_shape, const int* b_strides,
    int64_t total_elements)
{
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    int64_t temp = gid;
    int coords[16];
    for (int d = ndims - 1; d >= 0; --d) {
        coords[d] = (int)(temp % out_shape[d]);
        temp /= out_shape[d];
    }

    int64_t a_idx = a_off;
    int a_diff = ndims - a_ndims;
    for (int i = 0; i < a_ndims; ++i) {
        if (a_shape[i] != 1) {
            a_idx += (int64_t)coords[i + a_diff] * a_strides[i];
        }
    }

    int64_t b_idx = b_off;
    int b_diff = ndims - b_ndims;
    for (int i = 0; i < b_ndims; ++i) {
        if (b_shape[i] != 1) {
            b_idx += (int64_t)coords[i + b_diff] * b_strides[i];
        }
    }

    C[c_off + gid] = A[a_idx] / B[b_idx];
}

extern "C" __global__ void fill_zero(float* data, int64_t data_off, int64_t size) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[data_off + idx] = 0.0f;
    }
}

extern "C" __global__ void sum_backward(float* grad_input, int64_t gin_off, const float* grad_output, int64_t gout_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        grad_input[gin_off + id] = grad_output[gout_off];
    }
}

extern "C" __global__ void fake_quantize_forward(const float* A, int64_t a_off, float scale, float zero_point, float clip_min, float clip_max, float* B, int64_t b_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        float val = A[a_off + id] / scale + zero_point;
        float rounded = rintf(val);
        if (rounded < clip_min) rounded = clip_min;
        if (rounded > clip_max) rounded = clip_max;
        B[b_off + id] = (rounded - zero_point) * scale;
    }
}

extern "C" __global__ void cast_fp32_to_fp16(const float* src, int64_t src_off, unsigned short* dst, int64_t dst_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_float_to_half(src[src_off + id]);
}

extern "C" __global__ void cast_fp16_to_fp32(const unsigned short* src, int64_t src_off, float* dst, int64_t dst_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_half_to_float(src[src_off + id]);
}

extern "C" __global__ void cast_fp32_to_bf16(const float* src, int64_t src_off, unsigned short* dst, int64_t dst_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_float_to_bf16(src[src_off + id]);
}

extern "C" __global__ void cast_bf16_to_fp32(const unsigned short* src, int64_t src_off, float* dst, int64_t dst_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_bf16_to_float(src[src_off + id]);
}

extern "C" __global__ void cast_fp32_to_nf4(const float* src, int64_t src_off, unsigned char* dst, int64_t dst_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_float_to_nf4(src[src_off + id]);
}

extern "C" __global__ void cast_nf4_to_fp32(const unsigned char* src, int64_t src_off, float* dst, int64_t dst_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_nf4_to_float(src[src_off + id]);
}

extern "C" __global__ void cast_fp32_to_int8(const float* src, int64_t src_off, signed char* dst, int64_t dst_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = static_cast<signed char>(clamp(src[src_off + id], -128.0f, 127.0f));
}

extern "C" __global__ void cast_int8_to_fp32(const signed char* src, int64_t src_off, float* dst, int64_t dst_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = static_cast<float>(src[src_off + id]);
}

extern "C" __global__ void cast_fp32_to_int4(const float* src, int64_t src_off, signed char* dst, int64_t dst_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = static_cast<signed char>(clamp(src[src_off + id], -8.0f, 7.0f));
}

extern "C" __global__ void cast_int4_to_fp32(const signed char* src, int64_t src_off, float* dst, int64_t dst_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = static_cast<float>(src[src_off + id]);
}

extern "C" __global__ void cast_fp32_to_fp8_e4m3(const float* src, int64_t src_off, unsigned char* dst, int64_t dst_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_float_to_fp8_e4m3(src[src_off + id]);
}

extern "C" __global__ void cast_fp8_e4m3_to_fp32(const unsigned char* src, int64_t src_off, float* dst, int64_t dst_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_fp8_e4m3_to_float(src[src_off + id]);
}

extern "C" __global__ void cast_fp32_to_fp8_e5m2(const float* src, int64_t src_off, unsigned char* dst, int64_t dst_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_float_to_fp8_e5m2(src[src_off + id]);
}

extern "C" __global__ void cast_fp8_e5m2_to_fp32(const unsigned char* src, int64_t src_off, float* dst, int64_t dst_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_fp8_e5m2_to_float(src[src_off + id]);
}

__global__ void cat_forward_kernel(
    const float* input, int64_t in_off,
    float* output, int64_t out_off,
    int64_t outer_size, int64_t inner_size, int64_t dim_size, int64_t concat_dim_size, int64_t offset)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * dim_size * inner_size;
    if (idx >= total) return;

    int64_t i = idx % inner_size;
    int64_t d = (idx / inner_size) % dim_size;
    int64_t o = idx / (inner_size * dim_size);

    int64_t in_index = in_off + (o * dim_size + d) * inner_size + i;
    int64_t out_index = out_off + (o * concat_dim_size + offset + d) * inner_size + i;
    output[out_index] = input[in_index];
}

__global__ void cat_backward_kernel(
    const float* grad_output, int64_t gout_off,
    float* grad_input, int64_t gin_off,
    int64_t outer_size, int64_t inner_size, int64_t dim_size, int64_t concat_dim_size, int64_t offset)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * dim_size * inner_size;
    if (idx >= total) return;

    int64_t i = idx % inner_size;
    int64_t d = (idx / inner_size) % dim_size;
    int64_t o = idx / (inner_size * dim_size);

    int64_t gout_index = gout_off + (o * concat_dim_size + offset + d) * inner_size + i;
    int64_t gin_index = gin_off + (o * dim_size + d) * inner_size + i;
    grad_input[gin_index] = grad_output[gout_index];
}

extern "C" void gpu_cat_forward(
    void* input, int64_t in_off,
    void* output, int64_t out_off,
    int64_t outer_size, int64_t inner_size, int64_t dim_size, int64_t concat_dim_size, int64_t offset)
{
    auto_set_device(input);
    int64_t total = outer_size * dim_size * inner_size;
    if (total <= 0) return;
    int threads = 256;
    int64_t blocks64 = (total + threads - 1) / threads;
    if (blocks64 > INT_MAX) throw std::runtime_error("gpu_cat_forward: grid exceeds INT_MAX blocks");
    int blocks = (int)blocks64;
#ifndef __HIP_PLATFORM_AMD__
    cat_forward_kernel<<<blocks, threads, 0, dev_stream(current_device())>>>(
        (const float*)input, in_off, (float*)output, out_off, outer_size, inner_size, dim_size, concat_dim_size, offset);
#else
    hipLaunchKernelGGL(cat_forward_kernel, dim3(blocks), dim3(threads), 0, dev_stream(current_device()),
        (const float*)input, in_off, (float*)output, out_off, outer_size, inner_size, dim_size, concat_dim_size, offset);
#endif
}

extern "C" void gpu_cat_backward(
    void* grad_output, int64_t gout_off,
    void* grad_input, int64_t gin_off,
    int64_t outer_size, int64_t inner_size, int64_t dim_size, int64_t concat_dim_size, int64_t offset)
{
    auto_set_device(grad_output);
    int64_t total = outer_size * dim_size * inner_size;
    if (total <= 0) return;
    int threads = 256;
    int64_t blocks64 = (total + threads - 1) / threads;
    if (blocks64 > INT_MAX) throw std::runtime_error("gpu_cat_backward: grid exceeds INT_MAX blocks");
    int blocks = (int)blocks64;
#ifndef __HIP_PLATFORM_AMD__
    cat_backward_kernel<<<blocks, threads, 0, dev_stream(current_device())>>>(
        (const float*)grad_output, gout_off, (float*)grad_input, gin_off, outer_size, inner_size, dim_size, concat_dim_size, offset);
#else
    hipLaunchKernelGGL(cat_backward_kernel, dim3(blocks), dim3(threads), 0, dev_stream(current_device()),
        (const float*)grad_output, gout_off, (float*)grad_input, gin_off, outer_size, inner_size, dim_size, concat_dim_size, offset);
#endif
}

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
struct AbsOpH { __device__ __half operator()(__half a) const { return __habs(a); } };
struct NegOpH { __device__ __half operator()(__half a) const { return __hneg(a); } };
struct SigmoidOpH { __device__ __half operator()(__half a) const { float x = __half2float(a); return __float2half(1.0f / (1.0f + expf(-x))); } };
struct TanhOpH { __device__ __half operator()(__half a) const { float x = __half2float(a); return __float2half(tanhf(x)); } };
struct SqrtOpH { __device__ __half operator()(__half a) const { return hsqrt(a); } };
struct ExpOpH { __device__ __half operator()(__half a) const { return hexp(a); } };
struct LogOpH { __device__ __half operator()(__half a) const { float x = __half2float(a); return __float2half(logf(x)); } };
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
extern "C" __global__ void relu_backward_fp16(const __half* A, int64_t a_off, const __half* G, int64_t g_off, __half* GI, int64_t gi_off, int64_t size) { binary_kernel_fp16(A, a_off, G, g_off, GI, gi_off, size, ReluBackOpH()); }
extern "C" __global__ void sigmoid_backward_fp16(const __half* O, int64_t o_off, const __half* G, int64_t g_off, __half* GI, int64_t gi_off, int64_t size) { binary_kernel_fp16(O, o_off, G, g_off, GI, gi_off, size, SigmoidBackOpH()); }
extern "C" __global__ void tanh_backward_fp16(const __half* O, int64_t o_off, const __half* G, int64_t g_off, __half* GI, int64_t gi_off, int64_t size) { binary_kernel_fp16(O, o_off, G, g_off, GI, gi_off, size, TanhBackOpH()); }
extern "C" __global__ void sum_backward_fp16(__half* grad_input, int64_t gin_off, const __half* grad_output, int64_t gout_off, int64_t size) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        grad_input[gin_off + id] = grad_output[gout_off];
    }
}

template<typename T, typename Op>
__device__ void typed_binary_kernel(const T* A, int64_t a_off, const T* B, int64_t b_off,
                                    T* C, int64_t c_off, int64_t size, Op op) {
    for (int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; id < size;
         id += (int64_t)blockDim.x * gridDim.x) {
        C[c_off + id] = op(A[a_off + id], B[b_off + id]);
    }
}

template<typename T, typename Op>
__device__ void typed_unary_kernel(const T* A, int64_t a_off, T* B, int64_t b_off, int64_t size, Op op) {
    for (int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; id < size;
         id += (int64_t)blockDim.x * gridDim.x) {
        B[b_off + id] = op(A[a_off + id]);
    }
}

extern "C" __global__ void elementwise_add_i64(const int64_t* A, int64_t a_off, const int64_t* B, int64_t b_off, int64_t* C, int64_t c_off, int64_t size) {
    typed_binary_kernel(A, a_off, B, b_off, C, c_off, size, [] __device__ (int64_t a, int64_t b) { return (int64_t)((uint64_t)a + (uint64_t)b); });
}
extern "C" __global__ void elementwise_sub_i64(const int64_t* A, int64_t a_off, const int64_t* B, int64_t b_off, int64_t* C, int64_t c_off, int64_t size) {
    typed_binary_kernel(A, a_off, B, b_off, C, c_off, size, [] __device__ (int64_t a, int64_t b) { return (int64_t)((uint64_t)a - (uint64_t)b); });
}
extern "C" __global__ void elementwise_mul_i64(const int64_t* A, int64_t a_off, const int64_t* B, int64_t b_off, int64_t* C, int64_t c_off, int64_t size) {
    typed_binary_kernel(A, a_off, B, b_off, C, c_off, size, [] __device__ (int64_t a, int64_t b) { return (int64_t)((uint64_t)a * (uint64_t)b); });
}
extern "C" __global__ void elementwise_div_i64(const int64_t* A, int64_t a_off, const int64_t* B, int64_t b_off, int64_t* C, int64_t c_off, int64_t size) {
    typed_binary_kernel(A, a_off, B, b_off, C, c_off, size, [] __device__ (int64_t a, int64_t b) { return b == 0 ? 0 : a / b; });
}
extern "C" __global__ void elementwise_neg_i64(const int64_t* A, int64_t a_off, int64_t* B, int64_t b_off, int64_t size) {
    typed_unary_kernel(A, a_off, B, b_off, size, [] __device__ (int64_t a) { return (int64_t)(0 - (uint64_t)a); });
}
extern "C" __global__ void elementwise_abs_i64(const int64_t* A, int64_t a_off, int64_t* B, int64_t b_off, int64_t size) {
    typed_unary_kernel(A, a_off, B, b_off, size, [] __device__ (int64_t a) { return a >= 0 ? a : (int64_t)(0 - (uint64_t)a); });
}

extern "C" __global__ void elementwise_add_u8(const unsigned char* A, int64_t a_off, const unsigned char* B, int64_t b_off, unsigned char* C, int64_t c_off, int64_t size) {
    typed_binary_kernel(A, a_off, B, b_off, C, c_off, size, [] __device__ (unsigned char a, unsigned char b) { return (unsigned char)(a + b); });
}
extern "C" __global__ void elementwise_sub_u8(const unsigned char* A, int64_t a_off, const unsigned char* B, int64_t b_off, unsigned char* C, int64_t c_off, int64_t size) {
    typed_binary_kernel(A, a_off, B, b_off, C, c_off, size, [] __device__ (unsigned char a, unsigned char b) { return (unsigned char)(a - b); });
}
extern "C" __global__ void elementwise_mul_u8(const unsigned char* A, int64_t a_off, const unsigned char* B, int64_t b_off, unsigned char* C, int64_t c_off, int64_t size) {
    typed_binary_kernel(A, a_off, B, b_off, C, c_off, size, [] __device__ (unsigned char a, unsigned char b) { return (unsigned char)(a * b); });
}
extern "C" __global__ void elementwise_div_u8(const unsigned char* A, int64_t a_off, const unsigned char* B, int64_t b_off, unsigned char* C, int64_t c_off, int64_t size) {
    typed_binary_kernel(A, a_off, B, b_off, C, c_off, size, [] __device__ (unsigned char a, unsigned char b) { return b == 0 ? (unsigned char)0 : (unsigned char)(a / b); });
}
extern "C" __global__ void elementwise_neg_u8(const unsigned char* A, int64_t a_off, unsigned char* B, int64_t b_off, int64_t size) {
    typed_unary_kernel(A, a_off, B, b_off, size, [] __device__ (unsigned char a) { return (unsigned char)(-a); });
}
extern "C" __global__ void elementwise_abs_u8(const unsigned char* A, int64_t a_off, unsigned char* B, int64_t b_off, int64_t size) {
    typed_unary_kernel(A, a_off, B, b_off, size, [] __device__ (unsigned char a) { return a; });
}

extern "C" __global__ void elementwise_add_b(const bool* A, int64_t a_off, const bool* B, int64_t b_off, bool* C, int64_t c_off, int64_t size) {
    typed_binary_kernel(A, a_off, B, b_off, C, c_off, size, [] __device__ (bool a, bool b) { return (bool)(a || b); });
}
extern "C" __global__ void elementwise_mul_b(const bool* A, int64_t a_off, const bool* B, int64_t b_off, bool* C, int64_t c_off, int64_t size) {
    typed_binary_kernel(A, a_off, B, b_off, C, c_off, size, [] __device__ (bool a, bool b) { return (bool)(a && b); });
}
extern "C" __global__ void elementwise_not_b(const bool* A, int64_t a_off, bool* B, int64_t b_off, int64_t size) {
    typed_unary_kernel(A, a_off, B, b_off, size, [] __device__ (bool a) { return (bool)(!a); });
}
