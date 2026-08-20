#include "gpu_common.h"

template<typename Op>
__device__ void vectorized_binary_kernel(const float* A, int a_off, const float* B, int b_off, float* C, int c_off, int size, Op op) {
    uintptr_t a_addr = reinterpret_cast<uintptr_t>(A + a_off);
    uintptr_t b_addr = reinterpret_cast<uintptr_t>(B + b_off);
    uintptr_t c_addr = reinterpret_cast<uintptr_t>(C + c_off);
    if ((a_addr % 16 == 0) && (b_addr % 16 == 0) && (c_addr % 16 == 0)) {
        int vec_size = size / 4;
        for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < vec_size; id += blockDim.x * gridDim.x) {
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
            for (int i = 0; i < size % 4; ++i) {
                int idx = vec_size * 4 + i;
                C[c_off + idx] = op(A[a_off + idx], B[b_off + idx]);
            }
        }
    } else {
        for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < size; id += blockDim.x * gridDim.x) {
            C[c_off + id] = op(A[a_off + id], B[b_off + id]);
        }
    }
}

template<typename Op>
__device__ void vectorized_unary_kernel(const float* A, int a_off, float* B, int b_off, int size, Op op) {
    uintptr_t a_addr = reinterpret_cast<uintptr_t>(A + a_off);
    uintptr_t b_addr = reinterpret_cast<uintptr_t>(B + b_off);
    if ((a_addr % 16 == 0) && (b_addr % 16 == 0)) {
        int vec_size = size / 4;
        for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < vec_size; id += blockDim.x * gridDim.x) {
            float4 a_vec = reinterpret_cast<const float4*>(A + a_off)[id];
            float4 b_vec;
            b_vec.x = op(a_vec.x);
            b_vec.y = op(a_vec.y);
            b_vec.z = op(a_vec.z);
            b_vec.w = op(a_vec.w);
            reinterpret_cast<float4*>(B + b_off)[id] = b_vec;
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            for (int i = 0; i < size % 4; ++i) {
                int idx = vec_size * 4 + i;
                B[b_off + idx] = op(A[a_off + idx]);
            }
        }
    } else {
        for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < size; id += blockDim.x * gridDim.x) {
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

extern "C" __global__ void elementwise_add(const float* A, int a_off, const float* B, int b_off, float* C, int c_off, int size) { vectorized_binary_kernel(A, a_off, B, b_off, C, c_off, size, AddOp()); }
extern "C" __global__ void elementwise_add_inplace(float* A, int a_off, const float* B, int b_off, int size) { vectorized_binary_kernel(A, a_off, B, b_off, A, a_off, size, AddOp()); }
extern "C" __global__ void elementwise_sub(const float* A, int a_off, const float* B, int b_off, float* C, int c_off, int size) { vectorized_binary_kernel(A, a_off, B, b_off, C, c_off, size, SubOp()); }
extern "C" __global__ void elementwise_mul(const float* A, int a_off, const float* B, int b_off, float* C, int c_off, int size) { vectorized_binary_kernel(A, a_off, B, b_off, C, c_off, size, MulOp()); }
extern "C" __global__ void elementwise_div(const float* A, int a_off, const float* B, int b_off, float* C, int c_off, int size) { vectorized_binary_kernel(A, a_off, B, b_off, C, c_off, size, DivOp()); }
extern "C" __global__ void relu_forward(const float* A, int a_off, float* B, int b_off, int size) { vectorized_unary_kernel(A, a_off, B, b_off, size, ReluOp()); }
extern "C" __global__ void relu_backward_kernel(const float* A, int a_off, const float* grad_output, int gout_off, float* grad_input, int gin_off, int size) { vectorized_binary_kernel(A, a_off, grad_output, gout_off, grad_input, gin_off, size, ReluBackOp()); }
extern "C" __global__ void sigmoid_forward(const float* A, int a_off, float* B, int b_off, int size) { vectorized_unary_kernel(A, a_off, B, b_off, size, SigmoidOp()); }
extern "C" __global__ void tanh_forward(const float* A, int a_off, float* B, int b_off, int size) { vectorized_unary_kernel(A, a_off, B, b_off, size, TanhOp()); }
extern "C" __global__ void pow_forward(const float* A, int a_off, float exponent, float* B, int b_off, int size) { vectorized_unary_kernel(A, a_off, B, b_off, size, PowOp{exponent}); }
extern "C" __global__ void sqrt_forward(const float* A, int a_off, float* B, int b_off, int size) { vectorized_unary_kernel(A, a_off, B, b_off, size, SqrtOp()); }
extern "C" __global__ void exp_forward(const float* A, int a_off, float* B, int b_off, int size) { vectorized_unary_kernel(A, a_off, B, b_off, size, ExpOp()); }
extern "C" __global__ void log_forward(const float* A, int a_off, float* B, int b_off, int size) { vectorized_unary_kernel(A, a_off, B, b_off, size, LogOp()); }
extern "C" __global__ void abs_forward(const float* A, int a_off, float* B, int b_off, int size) { vectorized_unary_kernel(A, a_off, B, b_off, size, AbsOp()); }
extern "C" __global__ void neg_forward(const float* A, int a_off, float* B, int b_off, int size) { vectorized_unary_kernel(A, a_off, B, b_off, size, NegOp()); }
extern "C" __global__ void leaky_relu_forward(const float* A, int a_off, float negative_slope, float* B, int b_off, int size) { vectorized_unary_kernel(A, a_off, B, b_off, size, LeakyReluOp{negative_slope}); }
extern "C" __global__ void sigmoid_backward_kernel(const float* out_data, int out_off, const float* grad_output, int gout_off, float* grad_input, int gin_off, int size) { vectorized_binary_kernel(out_data, out_off, grad_output, gout_off, grad_input, gin_off, size, SigmoidBackOp()); }
extern "C" __global__ void tanh_backward_kernel(const float* out_data, int out_off, const float* grad_output, int gout_off, float* grad_input, int gin_off, int size) { vectorized_binary_kernel(out_data, out_off, grad_output, gout_off, grad_input, gin_off, size, TanhBackOp()); }
extern "C" __global__ void leaky_relu_backward_kernel(const float* A, int a_off, const float* grad_output, int gout_off, float* grad_input, int gin_off, int size, float negative_slope) { vectorized_binary_kernel(A, a_off, grad_output, gout_off, grad_input, gin_off, size, LeakyReluBackOp{negative_slope}); }

extern "C" __global__ void reduce_broadcast_prepended(const float* A, int a_off,
                                         float* B, int b_off,
                                         int prod_prepended, int remaining) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < remaining) {
        float s = 0.0f;
        for (int i = 0; i < prod_prepended; ++i) {
            s += A[a_off + i * remaining + j];
        }
        B[b_off + j] = s;
    }
}

extern "C" __global__ void reduce_broadcast_dim(const float* A, int a_off,
                                   float* B, int b_off,
                                   int outer_size, int dim_size, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_size * inner_size;
    if (idx < total) {
        int o = idx / inner_size;
        int in = idx % inner_size;
        float s = 0.0f;
        for (int d = 0; d < dim_size; ++d) {
            s += A[a_off + (o * dim_size + d) * inner_size + in];
        }
        B[b_off + o * inner_size + in] = s;
    }
}

struct Int8Val {
    int s0, s1, s2, s3, s4, s5, s6, s7;
};

extern "C" __global__ void make_contiguous_kernel(
    const float* src, int src_offset,
    float* dst, int dst_offset,
    int ndims,
    Int8Val shape_val,
    Int8Val strides_val,
    int total_elements)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    int shape[8] = { shape_val.s0, shape_val.s1, shape_val.s2, shape_val.s3, shape_val.s4, shape_val.s5, shape_val.s6, shape_val.s7 };
    int strides[8] = { strides_val.s0, strides_val.s1, strides_val.s2, strides_val.s3, strides_val.s4, strides_val.s5, strides_val.s6, strides_val.s7 };

    int src_idx = src_offset;
    int temp = gid;
    for (int d = ndims - 1; d >= 0; --d) {
        int coord = temp % shape[d];
        temp /= shape[d];
        src_idx += coord * strides[d];
    }
    dst[dst_offset + gid] = src[src_idx];
}

extern "C" __global__ void copy_to_strided_kernel(
    const float* src, int src_offset,
    float* dst, int dst_offset,
    int ndims,
    Int8Val shape_val,
    Int8Val strides_val,
    int total_elements)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    int shape[8] = { shape_val.s0, shape_val.s1, shape_val.s2, shape_val.s3, shape_val.s4, shape_val.s5, shape_val.s6, shape_val.s7 };
    int strides[8] = { strides_val.s0, strides_val.s1, strides_val.s2, strides_val.s3, strides_val.s4, strides_val.s5, strides_val.s6, strides_val.s7 };

    int dst_idx = dst_offset;
    int temp = gid;
    for (int d = ndims - 1; d >= 0; --d) {
        int coord = temp % shape[d];
        temp /= shape[d];
        dst_idx += coord * strides[d];
    }
    dst[dst_idx] = src[src_offset + gid];
}

extern "C" __global__ void elementwise_broadcast_add(
    const float* A, int a_off,
    const float* B, int b_off,
    float* C, int c_off,
    int ndims,
    const int* out_shape,
    int a_ndims, const int* a_shape, const int* a_strides,
    int b_ndims, const int* b_shape, const int* b_strides,
    int total_elements)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    int temp = gid;
    int coords[16];
    for (int d = ndims - 1; d >= 0; --d) {
        coords[d] = temp % out_shape[d];
        temp /= out_shape[d];
    }

    int a_idx = a_off;
    int a_diff = ndims - a_ndims;
    for (int i = 0; i < a_ndims; ++i) {
        if (a_shape[i] != 1) {
            a_idx += coords[i + a_diff] * a_strides[i];
        }
    }

    int b_idx = b_off;
    int b_diff = ndims - b_ndims;
    for (int i = 0; i < b_ndims; ++i) {
        if (b_shape[i] != 1) {
            b_idx += coords[i + b_diff] * b_strides[i];
        }
    }

    C[c_off + gid] = A[a_idx] + B[b_idx];
}

extern "C" __global__ void elementwise_broadcast_sub(
    const float* A, int a_off,
    const float* B, int b_off,
    float* C, int c_off,
    int ndims,
    const int* out_shape,
    int a_ndims, const int* a_shape, const int* a_strides,
    int b_ndims, const int* b_shape, const int* b_strides,
    int total_elements)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    int temp = gid;
    int coords[16];
    for (int d = ndims - 1; d >= 0; --d) {
        coords[d] = temp % out_shape[d];
        temp /= out_shape[d];
    }

    int a_idx = a_off;
    int a_diff = ndims - a_ndims;
    for (int i = 0; i < a_ndims; ++i) {
        if (a_shape[i] != 1) {
            a_idx += coords[i + a_diff] * a_strides[i];
        }
    }

    int b_idx = b_off;
    int b_diff = ndims - b_ndims;
    for (int i = 0; i < b_ndims; ++i) {
        if (b_shape[i] != 1) {
            b_idx += coords[i + b_diff] * b_strides[i];
        }
    }

    C[c_off + gid] = A[a_idx] - B[b_idx];
}

extern "C" __global__ void elementwise_broadcast_mul(
    const float* A, int a_off,
    const float* B, int b_off,
    float* C, int c_off,
    int ndims,
    const int* out_shape,
    int a_ndims, const int* a_shape, const int* a_strides,
    int b_ndims, const int* b_shape, const int* b_strides,
    int total_elements)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    int temp = gid;
    int coords[16];
    for (int d = ndims - 1; d >= 0; --d) {
        coords[d] = temp % out_shape[d];
        temp /= out_shape[d];
    }

    int a_idx = a_off;
    int a_diff = ndims - a_ndims;
    for (int i = 0; i < a_ndims; ++i) {
        if (a_shape[i] != 1) {
            a_idx += coords[i + a_diff] * a_strides[i];
        }
    }

    int b_idx = b_off;
    int b_diff = ndims - b_ndims;
    for (int i = 0; i < b_ndims; ++i) {
        if (b_shape[i] != 1) {
            b_idx += coords[i + b_diff] * b_strides[i];
        }
    }

    C[c_off + gid] = A[a_idx] * B[b_idx];
}

extern "C" __global__ void elementwise_broadcast_div(
    const float* A, int a_off,
    const float* B, int b_off,
    float* C, int c_off,
    int ndims,
    const int* out_shape,
    int a_ndims, const int* a_shape, const int* a_strides,
    int b_ndims, const int* b_shape, const int* b_strides,
    int total_elements)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    int temp = gid;
    int coords[16];
    for (int d = ndims - 1; d >= 0; --d) {
        coords[d] = temp % out_shape[d];
        temp /= out_shape[d];
    }

    int a_idx = a_off;
    int a_diff = ndims - a_ndims;
    for (int i = 0; i < a_ndims; ++i) {
        if (a_shape[i] != 1) {
            a_idx += coords[i + a_diff] * a_strides[i];
        }
    }

    int b_idx = b_off;
    int b_diff = ndims - b_ndims;
    for (int i = 0; i < b_ndims; ++i) {
        if (b_shape[i] != 1) {
            b_idx += coords[i + b_diff] * b_strides[i];
        }
    }

    C[c_off + gid] = A[a_idx] / B[b_idx];
}

extern "C" __global__ void fill_zero(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0.0f;
    }
}

extern "C" __global__ void sum_backward(float* grad_input, int gin_off, const float* grad_output, int gout_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        grad_input[gin_off + id] = grad_output[gout_off];
    }
}

extern "C" __global__ void fake_quantize_forward(const float* A, int a_off, float scale, float zero_point, float clip_min, float clip_max, float* B, int b_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        float val = A[a_off + id] / scale + zero_point;
        float rounded = rintf(val);
        if (rounded < clip_min) rounded = clip_min;
        if (rounded > clip_max) rounded = clip_max;
        B[b_off + id] = (rounded - zero_point) * scale;
    }
}

extern "C" __global__ void cast_fp32_to_fp16(const float* src, int src_off, unsigned short* dst, int dst_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_float_to_half(src[src_off + id]);
}

extern "C" __global__ void cast_fp16_to_fp32(const unsigned short* src, int src_off, float* dst, int dst_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_half_to_float(src[src_off + id]);
}

extern "C" __global__ void cast_fp32_to_bf16(const float* src, int src_off, unsigned short* dst, int dst_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_float_to_bf16(src[src_off + id]);
}

extern "C" __global__ void cast_bf16_to_fp32(const unsigned short* src, int src_off, float* dst, int dst_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_bf16_to_float(src[src_off + id]);
}

extern "C" __global__ void cast_fp32_to_nf4(const float* src, int src_off, unsigned char* dst, int dst_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_float_to_nf4(src[src_off + id]);
}

extern "C" __global__ void cast_nf4_to_fp32(const unsigned char* src, int src_off, float* dst, int dst_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_nf4_to_float(src[src_off + id]);
}

extern "C" __global__ void cast_fp32_to_int8(const float* src, int src_off, signed char* dst, int dst_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = static_cast<signed char>(clamp(src[src_off + id], -128.0f, 127.0f));
}

extern "C" __global__ void cast_int8_to_fp32(const signed char* src, int src_off, float* dst, int dst_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = static_cast<float>(src[src_off + id]);
}

extern "C" __global__ void cast_fp32_to_int4(const float* src, int src_off, signed char* dst, int dst_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = static_cast<signed char>(clamp(src[src_off + id], -8.0f, 7.0f));
}

extern "C" __global__ void cast_int4_to_fp32(const signed char* src, int src_off, float* dst, int dst_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = static_cast<float>(src[src_off + id]);
}

extern "C" __global__ void cast_fp32_to_fp8_e4m3(const float* src, int src_off, unsigned char* dst, int dst_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_float_to_fp8_e4m3(src[src_off + id]);
}

extern "C" __global__ void cast_fp8_e4m3_to_fp32(const unsigned char* src, int src_off, float* dst, int dst_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_fp8_e4m3_to_float(src[src_off + id]);
}

extern "C" __global__ void cast_fp32_to_fp8_e5m2(const float* src, int src_off, unsigned char* dst, int dst_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_float_to_fp8_e5m2(src[src_off + id]);
}

extern "C" __global__ void cast_fp8_e5m2_to_fp32(const unsigned char* src, int src_off, float* dst, int dst_off, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) dst[dst_off + id] = dev_fp8_e5m2_to_float(src[src_off + id]);
}

__global__ void cat_forward_kernel(
    const float* input, int in_off,
    float* output, int out_off,
    int outer_size, int inner_size, int dim_size, int concat_dim_size, int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_size * dim_size * inner_size;
    if (idx >= total) return;

    int i = idx % inner_size;
    int d = (idx / inner_size) % dim_size;
    int o = idx / (inner_size * dim_size);

    int in_index = in_off + (o * dim_size + d) * inner_size + i;
    int out_index = out_off + (o * concat_dim_size + offset + d) * inner_size + i;
    output[out_index] = input[in_index];
}

__global__ void cat_backward_kernel(
    const float* grad_output, int gout_off,
    float* grad_input, int gin_off,
    int outer_size, int inner_size, int dim_size, int concat_dim_size, int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_size * dim_size * inner_size;
    if (idx >= total) return;

    int i = idx % inner_size;
    int d = (idx / inner_size) % dim_size;
    int o = idx / (inner_size * dim_size);

    int gout_index = gout_off + (o * concat_dim_size + offset + d) * inner_size + i;
    int gin_index = gin_off + (o * dim_size + d) * inner_size + i;
    grad_input[gin_index] = grad_output[gout_index];
}

extern "C" void gpu_cat_forward(
    void* input, int in_off,
    void* output, int out_off,
    int outer_size, int inner_size, int dim_size, int concat_dim_size, int offset)
{
    int total = outer_size * dim_size * inner_size;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
#ifndef __HIP_PLATFORM_AMD__
    cat_forward_kernel<<<blocks, threads, 0, g_compute_stream>>>(
        (const float*)input, in_off, (float*)output, out_off, outer_size, inner_size, dim_size, concat_dim_size, offset);
#else
    hipLaunchKernelGGL(cat_forward_kernel, dim3(blocks), dim3(threads), 0, g_compute_stream,
        (const float*)input, in_off, (float*)output, out_off, outer_size, inner_size, dim_size, concat_dim_size, offset);
#endif
}

extern "C" void gpu_cat_backward(
    void* grad_output, int gout_off,
    void* grad_input, int gin_off,
    int outer_size, int inner_size, int dim_size, int concat_dim_size, int offset)
{
    int total = outer_size * dim_size * inner_size;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
#ifndef __HIP_PLATFORM_AMD__
    cat_backward_kernel<<<blocks, threads, 0, g_compute_stream>>>(
        (const float*)grad_output, gout_off, (float*)grad_input, gin_off, outer_size, inner_size, dim_size, concat_dim_size, offset);
#else
    hipLaunchKernelGGL(cat_backward_kernel, dim3(blocks), dim3(threads), 0, g_compute_stream,
        (const float*)grad_output, gout_off, (float*)grad_input, gin_off, outer_size, inner_size, dim_size, concat_dim_size, offset);
#endif
}
