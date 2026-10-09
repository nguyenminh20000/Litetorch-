#include "gpu_common.h"
#include <unordered_map>
#include <mutex>

#ifndef __HIP_PLATFORM_AMD__

struct CudnnConvKey {
    int N, C_in, H_in, W_in, C_out, H_out, W_out, kh, kw, stride, padding;
    bool operator==(const CudnnConvKey& o) const {
        return N == o.N && C_in == o.C_in && H_in == o.H_in && W_in == o.W_in &&
               C_out == o.C_out && H_out == o.H_out && W_out == o.W_out &&
               kh == o.kh && kw == o.kw && stride == o.stride && padding == o.padding;
    }
};

struct CudnnConvKeyHash {
    size_t operator()(const CudnnConvKey& k) const {
        size_t h = 1469598103934665603ULL;
        auto mix = [&](int v) { h ^= (size_t)v; h *= 1099511628211ULL; };
        mix(k.N); mix(k.C_in); mix(k.H_in); mix(k.W_in);
        mix(k.C_out); mix(k.H_out); mix(k.W_out);
        mix(k.kh); mix(k.kw); mix(k.stride); mix(k.padding);
        return h;
    }
};

struct CudnnConvDescs {
    lt_cudnnTensorDescriptor_t xDesc, yDesc, bDesc;
    lt_cudnnFilterDescriptor_t wDesc;
    lt_cudnnConvolutionDescriptor_t convDesc;
};

static std::unordered_map<CudnnConvKey, CudnnConvDescs, CudnnConvKeyHash> cudnn_conv_cache;
static std::mutex cudnn_conv_cache_mutex;

static CudnnConvDescs& get_cudnn_conv_descs(const CudnnConvKey& key) {
    std::lock_guard<std::mutex> lock(cudnn_conv_cache_mutex);
    auto it = cudnn_conv_cache.find(key);
    if (it != cudnn_conv_cache.end()) return it->second;
    CudnnConvDescs d;
    g_cudnn.CreateTensorDescriptor(&d.xDesc);
    g_cudnn.CreateTensorDescriptor(&d.yDesc);
    g_cudnn.CreateTensorDescriptor(&d.bDesc);
    g_cudnn.CreateFilterDescriptor(&d.wDesc);
    g_cudnn.CreateConvolutionDescriptor(&d.convDesc);
    g_cudnn.SetTensor4dDescriptor(d.xDesc, LT_CUDNN_TENSOR_NCHW, LT_CUDNN_DATA_FLOAT, key.N, key.C_in, key.H_in, key.W_in);
    g_cudnn.SetFilter4dDescriptor(d.wDesc, LT_CUDNN_DATA_FLOAT, LT_CUDNN_TENSOR_NCHW, key.C_out, key.C_in, key.kh, key.kw);
    g_cudnn.SetConvolution2dDescriptor(d.convDesc, key.padding, key.padding, key.stride, key.stride, 1, 1, LT_CUDNN_CROSS_CORRELATION, LT_CUDNN_DATA_FLOAT);
    if (g_cudnn.SetConvolutionMathType) g_cudnn.SetConvolutionMathType(d.convDesc, LT_CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
    g_cudnn.SetTensor4dDescriptor(d.yDesc, LT_CUDNN_TENSOR_NCHW, LT_CUDNN_DATA_FLOAT, key.N, key.C_out, key.H_out, key.W_out);
    g_cudnn.SetTensor4dDescriptor(d.bDesc, LT_CUDNN_TENSOR_NCHW, LT_CUDNN_DATA_FLOAT, 1, key.C_out, 1, 1);
    auto inserted = cudnn_conv_cache.emplace(key, d);
    return inserted.first->second;
}

static lt_cudnnActivationDescriptor_t relu_act_desc = nullptr;
static std::mutex relu_act_desc_mutex;

static lt_cudnnActivationDescriptor_t get_relu_act_desc() {
    std::lock_guard<std::mutex> lock(relu_act_desc_mutex);
    if (!relu_act_desc) {
        lt_cudnnActivationDescriptor_t d = nullptr;
        if (g_cudnn.CreateActivationDescriptor && g_cudnn.SetActivationDescriptor &&
            g_cudnn.CreateActivationDescriptor(&d) == 0 &&
            g_cudnn.SetActivationDescriptor(d, LT_CUDNN_ACTIVATION_RELU, LT_CUDNN_NOT_PROPAGATE_NAN, 0.0) == 0) {
            relu_act_desc = d;
        } else if (d && g_cudnn.DestroyActivationDescriptor) {
            g_cudnn.DestroyActivationDescriptor(d);
        }
    }
    return relu_act_desc;
}
struct CudnnWorkspace {
    void* ptr = nullptr;
    size_t bytes = 0;
};

static void* cudnn_workspace(size_t need) {
    thread_local CudnnWorkspace ws;
    if (need == 0) return nullptr;
    if (need > ws.bytes) {
        if (ws.ptr) cudaFree(ws.ptr);
        ws.ptr = nullptr;
        ws.bytes = 0;
        if (cudaMalloc(&ws.ptr, need) != cudaSuccess) return nullptr;
        ws.bytes = need;
    }
    return ws.ptr;
}

static std::unordered_map<CudnnConvKey, int, CudnnConvKeyHash> cudnn_fwd_algo_cache;
static std::mutex cudnn_fwd_algo_cache_mutex;

static const size_t CUDNN_FWD_ALGO_WS_LIMIT = 256ULL * 1024ULL * 1024ULL;

static int cudnn_pick_fwd_algo(LtCudnnFwdAlgoPerf* perfs, int count) {
    for (int i = 0; i < count; ++i) {
        if (perfs[i].status == 0 && perfs[i].memory <= CUDNN_FWD_ALGO_WS_LIMIT) return perfs[i].algo;
    }
    return 0;
}

static int cudnn_fwd_algo(lt_cudnnHandle_t handle, const CudnnConvKey& key, CudnnConvDescs& d) {
    {
        std::lock_guard<std::mutex> lock(cudnn_fwd_algo_cache_mutex);
        auto it = cudnn_fwd_algo_cache.find(key);
        if (it != cudnn_fwd_algo_cache.end()) return it->second;
    }
    int algo = 0;
    if (g_cudnn.GetConvolutionForwardAlgorithm_v7) {
        LtCudnnFwdAlgoPerf perfs[8];
        int returned = 0;
        if (g_cudnn.GetConvolutionForwardAlgorithm_v7(handle, d.xDesc, d.wDesc, d.convDesc, d.yDesc, 8, &returned, perfs) == 0 && returned > 0) {
            algo = cudnn_pick_fwd_algo(perfs, returned);
        }
    }
    {
        std::lock_guard<std::mutex> lock(cudnn_fwd_algo_cache_mutex);
        cudnn_fwd_algo_cache.emplace(key, algo);
    }
    return algo;
}

extern "C" void gpu_conv2d_cudnn(
    const float* input, int in_off,
    const float* weight, int w_off,
    const float* bias, int b_off, int has_bias,
    float* output, int out_off,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kh, int kw, int stride, int padding, int apply_relu) {
    lt_cudnnHandle_t handle = get_cudnn_handle();
    if (!handle) return;
    CudnnConvKey key{N, C_in, H_in, W_in, C_out, H_out, W_out, kh, kw, stride, padding};
    CudnnConvDescs& d = get_cudnn_conv_descs(key);

    float alpha = 1.0f, beta = 0.0f;
    int fwd_algo = cudnn_fwd_algo(handle, key, d);
    size_t fwd_ws_bytes = 0;
    void* fwd_ws = nullptr;
    if (g_cudnn.GetConvolutionForwardWorkspaceSize &&
        g_cudnn.GetConvolutionForwardWorkspaceSize(handle, d.xDesc, d.wDesc, d.convDesc, d.yDesc, fwd_algo, &fwd_ws_bytes) == 0) {
        fwd_ws = cudnn_workspace(fwd_ws_bytes);
    }
    lt_cudnnActivationDescriptor_t reluDesc = nullptr;
    if (apply_relu && has_bias && bias && g_cudnn.ConvolutionBiasActivationForward) {
        reluDesc = get_relu_act_desc();
    }
    if (reluDesc) {
        int st = g_cudnn.ConvolutionBiasActivationForward(handle, &alpha, d.xDesc, input + in_off, d.wDesc, weight + w_off, d.convDesc, fwd_algo, fwd_ws, fwd_ws ? fwd_ws_bytes : 0, &beta, nullptr, nullptr, d.bDesc, bias + b_off, reluDesc, d.yDesc, output + out_off);
        if (st == 0) return;
    }

    g_cudnn.ConvolutionForward(handle, &alpha, d.xDesc, input + in_off, d.wDesc, weight + w_off, d.convDesc, fwd_algo, fwd_ws, fwd_ws ? fwd_ws_bytes : 0, &beta, d.yDesc, output + out_off);

    if (has_bias && bias) {
        g_cudnn.AddTensor(handle, &alpha, d.bDesc, bias + b_off, &alpha, d.yDesc, output + out_off);
    }

    if (apply_relu) {
        int64_t total = (int64_t)N * C_out * H_out * W_out;
        int threads = 256;
        int64_t blocks = (total + threads - 1) / threads;
        relu_inplace_kernel<<<(unsigned)blocks, threads>>>(output + out_off, total);
    }
}

struct CudnnBwdAlgos {
    int data_algo;
    int filter_algo;
};

static std::unordered_map<CudnnConvKey, CudnnBwdAlgos, CudnnConvKeyHash> cudnn_bwd_algo_cache;
static std::mutex cudnn_bwd_algo_cache_mutex;

static const size_t CUDNN_BWD_ALGO_WS_LIMIT = 256ULL * 1024ULL * 1024ULL;

static int cudnn_pick_bwd_algo(LtCudnnBwdAlgoPerf* perfs, int count) {
    for (int i = 0; i < count; ++i) {
        if (perfs[i].status == 0 && perfs[i].memory <= CUDNN_BWD_ALGO_WS_LIMIT) return perfs[i].algo;
    }
    return 0;
}

static CudnnBwdAlgos cudnn_bwd_algos(lt_cudnnHandle_t handle, const CudnnConvKey& key, CudnnConvDescs& d) {
    {
        std::lock_guard<std::mutex> lock(cudnn_bwd_algo_cache_mutex);
        auto it = cudnn_bwd_algo_cache.find(key);
        if (it != cudnn_bwd_algo_cache.end()) return it->second;
    }
    CudnnBwdAlgos a{0, 0};
    if (g_cudnn.GetConvolutionBackwardDataAlgorithm_v7) {
        LtCudnnBwdAlgoPerf perfs[8];
        int returned = 0;
        if (g_cudnn.GetConvolutionBackwardDataAlgorithm_v7(handle, d.wDesc, d.yDesc, d.convDesc, d.xDesc, 8, &returned, perfs) == 0 && returned > 0) {
            a.data_algo = cudnn_pick_bwd_algo(perfs, returned);
        }
    }
    if (g_cudnn.GetConvolutionBackwardFilterAlgorithm_v7) {
        LtCudnnBwdAlgoPerf perfs[8];
        int returned = 0;
        if (g_cudnn.GetConvolutionBackwardFilterAlgorithm_v7(handle, d.xDesc, d.yDesc, d.convDesc, d.wDesc, 8, &returned, perfs) == 0 && returned > 0) {
            a.filter_algo = cudnn_pick_bwd_algo(perfs, returned);
        }
    }
    {
        std::lock_guard<std::mutex> lock(cudnn_bwd_algo_cache_mutex);
        cudnn_bwd_algo_cache.emplace(key, a);
    }
    return a;
}

extern "C" void gpu_conv2d_backward_data_cudnn(
    const float* gout, int gout_off,
    const float* weight, int w_off,
    float* gdx, int gdx_off,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kh, int kw, int stride, int padding) {
    lt_cudnnHandle_t handle = get_cudnn_handle();
    if (!handle) return;
    CudnnConvKey key{N, C_in, H_in, W_in, C_out, H_out, W_out, kh, kw, stride, padding};
    CudnnConvDescs& d = get_cudnn_conv_descs(key);
    float alpha = 1.0f, beta = 0.0f;
    int algo = cudnn_bwd_algos(handle, key, d).data_algo;
    size_t ws_bytes = 0;
    void* ws = nullptr;
    if (g_cudnn.GetConvolutionBackwardDataWorkspaceSize &&
        g_cudnn.GetConvolutionBackwardDataWorkspaceSize(handle, d.wDesc, d.yDesc, d.convDesc, d.xDesc, algo, &ws_bytes) == 0) {
        ws = cudnn_workspace(ws_bytes);
    }
    g_cudnn.ConvolutionBackwardData(handle, &alpha, d.wDesc, weight + w_off, d.yDesc, gout + gout_off, d.convDesc, algo, ws, ws ? ws_bytes : 0, &beta, d.xDesc, gdx + gdx_off);
}

extern "C" void gpu_conv2d_backward_filter_cudnn(
    const float* gout, int gout_off,
    const float* input, int in_off,
    float* gw, int gw_off,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kh, int kw, int stride, int padding) {
    lt_cudnnHandle_t handle = get_cudnn_handle();
    if (!handle) return;
    CudnnConvKey key{N, C_in, H_in, W_in, C_out, H_out, W_out, kh, kw, stride, padding};
    CudnnConvDescs& d = get_cudnn_conv_descs(key);
    float alpha = 1.0f, beta = 0.0f;
    int algo = cudnn_bwd_algos(handle, key, d).filter_algo;
    size_t ws_bytes = 0;
    void* ws = nullptr;
    if (g_cudnn.GetConvolutionBackwardFilterWorkspaceSize &&
        g_cudnn.GetConvolutionBackwardFilterWorkspaceSize(handle, d.xDesc, d.yDesc, d.convDesc, d.wDesc, algo, &ws_bytes) == 0) {
        ws = cudnn_workspace(ws_bytes);
    }
    g_cudnn.ConvolutionBackwardFilter(handle, &alpha, d.xDesc, input + in_off, d.yDesc, gout + gout_off, d.convDesc, algo, ws, ws ? ws_bytes : 0, &beta, d.wDesc, gw + gw_off);
}
#endif

#ifdef USE_MIOPEN
struct MiopenConvKey {
    int N, C_in, H_in, W_in, C_out, H_out, W_out, kh, kw, stride, padding;
    bool operator==(const MiopenConvKey& o) const {
        return N == o.N && C_in == o.C_in && H_in == o.H_in && W_in == o.W_in &&
               C_out == o.C_out && H_out == o.H_out && W_out == o.W_out &&
               kh == o.kh && kw == o.kw && stride == o.stride && padding == o.padding;
    }
};

struct MiopenConvKeyHash {
    size_t operator()(const MiopenConvKey& k) const {
        size_t h = 1469598103934665603ULL;
        auto mix = [&](int v) { h ^= (size_t)v; h *= 1099511628211ULL; };
        mix(k.N); mix(k.C_in); mix(k.H_in); mix(k.W_in);
        mix(k.C_out); mix(k.H_out); mix(k.W_out);
        mix(k.kh); mix(k.kw); mix(k.stride); mix(k.padding);
        return h;
    }
};

struct MiopenConvDescs {
    miopenTensorDescriptor_t xDesc, yDesc, bDesc, wDesc;
    miopenConvolutionDescriptor_t convDesc;
};

static std::unordered_map<MiopenConvKey, MiopenConvDescs, MiopenConvKeyHash> miopen_conv_cache;
static std::mutex miopen_conv_cache_mutex;

static MiopenConvDescs& get_miopen_conv_descs(const MiopenConvKey& key) {
    std::lock_guard<std::mutex> lock(miopen_conv_cache_mutex);
    auto it = miopen_conv_cache.find(key);
    if (it != miopen_conv_cache.end()) return it->second;
    MiopenConvDescs d;
    miopenCreateTensorDescriptor(&d.xDesc);
    miopenCreateTensorDescriptor(&d.yDesc);
    miopenCreateTensorDescriptor(&d.bDesc);
    miopenCreateTensorDescriptor(&d.wDesc);
    miopenCreateConvolutionDescriptor(&d.convDesc);
    miopenSet4dTensorDescriptor(d.xDesc, miopenFloat, key.N, key.C_in, key.H_in, key.W_in);
    miopenSet4dTensorDescriptor(d.wDesc, miopenFloat, key.C_out, key.C_in, key.kh, key.kw);
    miopenInitConvolutionDescriptor(d.convDesc, miopenConvolution, key.padding, key.padding, key.stride, key.stride, 1, 1);
    miopenSet4dTensorDescriptor(d.yDesc, miopenFloat, key.N, key.C_out, key.H_out, key.W_out);
    miopenSet4dTensorDescriptor(d.bDesc, miopenFloat, 1, key.C_out, 1, 1);
    auto inserted = miopen_conv_cache.emplace(key, d);
    return inserted.first->second;
}

extern "C" void gpu_conv2d_miopen(
    const float* input, int in_off,
    const float* weight, int w_off,
    const float* bias, int b_off, int has_bias,
    float* output, int out_off,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kh, int kw, int stride, int padding) {
    miopenHandle_t handle = get_miopen_handle();
    MiopenConvKey key{N, C_in, H_in, W_in, C_out, H_out, W_out, kh, kw, stride, padding};
    MiopenConvDescs& d = get_miopen_conv_descs(key);

    float alpha = 1.0f, beta = 0.0f;
    miopenConvolutionForward(handle, &alpha, d.xDesc, input + in_off, d.wDesc, weight + w_off, d.convDesc, miopenConvolutionFwdAlgoGEMM, &beta, d.yDesc, output + out_off, nullptr, 0);

    if (has_bias && bias) {
        miopenOpTensor(handle, miopenTensorOpAdd, &alpha, d.yDesc, output + out_off, &alpha, d.bDesc, bias + b_off, &beta, d.yDesc, output + out_off);
    }
}
#endif

extern "C" __global__ void conv2d_kernel(const float* input, int in_off,
                            const float* weight, int w_off,
                            const float* bias, int b_off, int has_bias,
                            float* output, int out_off,
                            int batch_size, int in_channels, int in_h, int in_w,
                            int out_channels, int out_h, int out_w,
                            int kh, int kw, int stride, int padding) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    int total_threads = batch_size * out_channels * out_h * out_w;
    if (idx >= total_threads) return;

    int w_out = idx % out_w;
    int h_out = (idx / out_w) % out_h;
    int c_out = (idx / (out_w * out_h)) % out_channels;
    int b = idx / (out_w * out_h * out_channels);

    float val = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int ky = 0; ky < kh; ++ky) {
            int y = h_out * stride - padding + ky;
            for (int kx = 0; kx < kw; ++kx) {
                int x = w_out * stride - padding + kx;
                if (y >= 0 && y < in_h && x >= 0 && x < in_w) {
                    int input_idx = in_off + ((b * in_channels + c_in) * in_h + y) * in_w + x;
                    int weight_idx = w_off + ((c_out * in_channels + c_in) * kh + ky) * kw + kx;
                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    if (has_bias) {
        val += bias[b_off + c_out];
    }
    output[out_off + idx] = val;
}

extern "C" __global__ void conv2d_backward_gb(
    const float* grad_output, int gout_off,
    float* grad_bias, int gb_off,
    int N, int C_out, int H_out, int W_out)
{
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    if (co >= C_out) return;

    float sum_val = 0.0f;
    for (int b = 0; b < N; ++b) {
        for (int ho = 0; ho < H_out; ++ho) {
            for (int wo = 0; wo < W_out; ++wo) {
                sum_val += grad_output[gout_off + ((b * C_out + co) * H_out + ho) * W_out + wo];
            }
        }
    }
    grad_bias[gb_off + co] = sum_val;
}

extern "C" __global__ void conv2d_backward_gw(
    const float* input, int in_off,
    const float* grad_output, int gout_off,
    float* grad_weight, int gw_off,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int KH, int KW, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * KH * KW;
    if (idx >= total) return;

    int kw = idx % KW;
    int kh = (idx / KW) % KH;
    int ci = (idx / (KW * KH)) % C_in;
    int co = idx / (KW * KH * C_in);

    float sum_val = 0.0f;
    for (int b = 0; b < N; ++b) {
        for (int ho = 0; ho < H_out; ++ho) {
            int y = ho * stride - padding + kh;
            if (y >= 0 && y < H_in) {
                for (int wo = 0; wo < W_out; ++wo) {
                    int x = wo * stride - padding + kw;
                    if (x >= 0 && x < W_in) {
                        int gout_idx = ((b * C_out + co) * H_out + ho) * W_out + wo;
                        int in_idx = ((b * C_in + ci) * H_in + y) * W_in + x;
                        sum_val += grad_output[gout_off + gout_idx] * input[in_off + in_idx];
                    }
                }
            }
        }
    }
    grad_weight[gw_off + idx] = sum_val;
}

extern "C" __global__ void conv2d_backward_gdx(
    const float* grad_output, int gout_off,
    const float* weight, int w_off,
    float* grad_input, int gin_off,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int KH, int KW, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * H_in * W_in;
    if (idx >= total) return;

    int x = idx % W_in;
    int y = (idx / W_in) % H_in;
    int ci = (idx / (W_in * H_in)) % C_in;
    int b = idx / (W_in * H_in * C_in);

    float sum_val = 0.0f;
    for (int co = 0; co < C_out; ++co) {
        for (int kh = 0; kh < KH; ++kh) {
            int ho_temp = y + padding - kh;
            if (ho_temp % stride == 0) {
                int ho = ho_temp / stride;
                if (ho >= 0 && ho < H_out) {
                    for (int kw = 0; kw < KW; ++kw) {
                        int wo_temp = x + padding - kw;
                        if (wo_temp % stride == 0) {
                            int wo = wo_temp / stride;
                            if (wo >= 0 && wo < W_out) {
                                int gout_idx = ((b * C_out + co) * H_out + ho) * W_out + wo;
                                int w_idx = ((co * C_in + ci) * KH + kh) * KW + kw;
                                sum_val += grad_output[gout_off + gout_idx] * weight[w_off + w_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    grad_input[gin_off + idx] = sum_val;
}

extern "C" __global__ void im2col_kernel(
    const float* im, int im_off,
    int C, int H, int W,
    int KH, int KW, int padding, int stride,
    int H_out, int W_out,
    float* col, int col_off)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * KH * KW * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int kw = (idx / (W_out * H_out)) % KW;
    int kh = (idx / (W_out * H_out * KW)) % KH;
    int c_im = idx / (W_out * H_out * KW * KH);

    int im_row = h_out * stride - padding + kh;
    int im_col = w_out * stride - padding + kw;

    int c_col = c_im * KH * KW + kh * KW + kw;
    int col_idx = (c_col * H_out + h_out) * W_out + w_out;

    if (im_row >= 0 && im_row < H && im_col >= 0 && im_col < W) {
        col[col_off + col_idx] = im[im_off + (c_im * H + im_row) * W + im_col];
    } else {
        col[col_off + col_idx] = 0.0f;
    }
}

extern "C" __global__ void im2col_batched_kernel(
    const float* im, int im_off,
    int N, int C, int H, int W,
    int KH, int KW, int padding, int stride,
    int H_out, int W_out,
    float* col, int col_off)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t HW_out = (int64_t)H_out * W_out;
    int64_t K = (int64_t)C * KH * KW;
    int64_t total = (int64_t)N * K * HW_out;
    if (idx >= total) return;

    int64_t tmp = idx;
    int w_out = tmp % W_out; tmp /= W_out;
    int h_out = tmp % H_out; tmp /= H_out;
    int k = tmp % K; tmp /= K;
    int n = tmp;
    int kw = k % KW;
    int kh = (k / KW) % KH;
    int c = k / (KH * KW);

    int im_row = h_out * stride - padding + kh;
    int im_col = w_out * stride - padding + kw;

    float v = 0.0f;
    if (im_row >= 0 && im_row < H && im_col >= 0 && im_col < W) {
        v = im[im_off + ((int64_t)n * C + c) * H * W + im_row * W + im_col];
    }
    col[col_off + (int64_t)n * K * HW_out + k * HW_out + h_out * W_out + w_out] = v;
}

extern "C" __global__ void im2col_flat_kernel(
    const float* im, int im_off,
    int N, int C, int H, int W,
    int KH, int KW, int padding, int stride,
    int H_out, int W_out,
    float* col, int col_off)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t HW_out = (int64_t)H_out * W_out;
    int64_t NHW = (int64_t)N * HW_out;
    int64_t K = (int64_t)C * KH * KW;
    int64_t total = K * NHW;
    if (idx >= total) return;

    int k = (int)(idx / NHW);
    int j = (int)(idx - (int64_t)k * NHW);
    int hw_out = (int)HW_out;
    int n = j / hw_out;
    int hw = j - n * hw_out;
    int h_out = hw / W_out;
    int w_out = hw - h_out * W_out;
    int kw = k % KW;
    int khw = k / KW;
    int kh = khw % KH;
    int c = khw / KH;

    int im_row = h_out * stride - padding + kh;
    int im_col = w_out * stride - padding + kw;

    float v = 0.0f;
    if (im_row >= 0 && im_row < H && im_col >= 0 && im_col < W) {
        v = im[im_off + ((int64_t)n * C + c) * H * W + (int64_t)im_row * W + im_col];
    }
    col[col_off + (int64_t)k * NHW + j] = v;
}

extern "C" __global__ void transpose_conv_out_kernel(
    const float* src, int s_off,
    float* dst, int d_off,
    int N, int C, int H, int W)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t HW = (int64_t)H * W;
    int64_t total = (int64_t)N * C * HW;
    if (idx >= total) return;
    int64_t tmp = idx;
    int hw = tmp % HW; tmp /= HW;
    int c = tmp % C; tmp /= C;
    int n = tmp;
    dst[d_off + idx] = src[s_off + (int64_t)c * N * HW + (int64_t)n * HW + hw];
}

extern "C" __global__ void broadcast_batch_kernel(
    const float* src, int s_off,
    float* dst, int d_off,
    int batch, int rows, int cols)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t inner = (int64_t)rows * cols;
    int64_t total = (int64_t)batch * inner;
    if (idx >= total) return;
    dst[d_off + idx] = src[s_off + idx % inner];
}

extern "C" __global__ void add_bias_2d(
    float* out, int out_off,
    const float* bias, int b_off,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx < total) {
        int c = (idx / (H * W)) % C;
        out[out_off + idx] += bias[b_off + c];
    }
}
extern "C" __global__ void conv3d_kernel(
    const float* input, int in_off,
    const float* weight, int w_off,
    const float* bias, int b_off, int has_bias,
    float* output, int out_off,
    int batch_size, int in_channels, int in_d, int in_h, int in_w,
    int out_channels, int out_d, int out_h, int out_w,
    int kd, int kh, int kw, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * out_d * out_h * out_w;
    if (idx >= total_threads) return;

    int w_out = idx % out_w;
    int h_out = (idx / out_w) % out_h;
    int d_out = (idx / (out_w * out_h)) % out_d;
    int c_out = (idx / (out_w * out_h * out_d)) % out_channels;
    int b = idx / (out_w * out_h * out_d * out_channels);

    float val = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kz = 0; kz < kd; ++kz) {
            int z = d_out * stride - padding + kz;
            if (z >= 0 && z < in_d) {
                for (int ky = 0; ky < kh; ++ky) {
                    int y = h_out * stride - padding + ky;
                    if (y >= 0 && y < in_h) {
                        for (int kx = 0; kx < kw; ++kx) {
                            int x = w_out * stride - padding + kx;
                            if (x >= 0 && x < in_w) {
                                int input_idx = in_off + ((((b * in_channels + c_in) * in_d + z) * in_h + y) * in_w + x);
                                int weight_idx = w_off + ((((c_out * in_channels + c_in) * kd + kz) * kh + ky) * kw + kx);
                                val += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    if (has_bias && bias) {
        val += bias[b_off + c_out];
    }
    output[out_off + idx] = val;
}

extern "C" __global__ void conv3d_backward_gb(
    const float* grad_output, int gout_off,
    float* grad_bias, int gb_off,
    int N, int C_out, int D_out, int H_out, int W_out)
{
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    if (co >= C_out) return;

    float sum_val = 0.0f;
    for (int b = 0; b < N; ++b) {
        for (int do_ = 0; do_ < D_out; ++do_) {
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    int gout_idx = (((b * C_out + co) * D_out + do_) * H_out + ho) * W_out + wo;
                    sum_val += grad_output[gout_off + gout_idx];
                }
            }
        }
    }
    grad_bias[gb_off + co] = sum_val;
}

extern "C" __global__ void conv3d_backward_gw(
    const float* input, int in_off,
    const float* grad_output, int gout_off,
    float* grad_weight, int gw_off,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int KD, int KH, int KW, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * KD * KH * KW;
    if (idx >= total) return;

    int kw = idx % KW;
    int kh = (idx / KW) % KH;
    int kd = (idx / (KW * KH)) % KD;
    int ci = (idx / (KW * KH * KD)) % C_in;
    int co = idx / (KW * KH * KD * C_in);

    float sum_val = 0.0f;
    for (int b = 0; b < N; ++b) {
        for (int do_ = 0; do_ < D_out; ++do_) {
            int z = do_ * stride - padding + kd;
            if (z >= 0 && z < D_in) {
                for (int ho = 0; ho < H_out; ++ho) {
                    int y = ho * stride - padding + kh;
                    if (y >= 0 && y < H_in) {
                        for (int wo = 0; wo < W_out; ++wo) {
                            int x = wo * stride - padding + kw;
                            if (x >= 0 && x < W_in) {
                                int gout_idx = (((b * C_out + co) * D_out + do_) * H_out + ho) * W_out + wo;
                                int in_idx = (((b * C_in + ci) * D_in + z) * H_in + y) * W_in + x;
                                sum_val += grad_output[gout_off + gout_idx] * input[in_off + in_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    grad_weight[gw_off + idx] = sum_val;
}

extern "C" __global__ void conv3d_backward_gdx(
    const float* grad_output, int gout_off,
    const float* weight, int w_off,
    float* grad_input, int gin_off,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int KD, int KH, int KW, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * D_in * H_in * W_in;
    if (idx >= total) return;

    int x = idx % W_in;
    int y = (idx / W_in) % H_in;
    int z = (idx / (W_in * H_in)) % D_in;
    int ci = (idx / (W_in * H_in * D_in)) % C_in;
    int b = idx / (W_in * H_in * D_in * C_in);

    float sum_val = 0.0f;
    for (int co = 0; co < C_out; ++co) {
        for (int kd = 0; kd < KD; ++kd) {
            int do_temp = z + padding - kd;
            if (do_temp % stride == 0) {
                int do_ = do_temp / stride;
                if (do_ >= 0 && do_ < D_out) {
                    for (int kh = 0; kh < KH; ++kh) {
                        int ho_temp = y + padding - kh;
                        if (ho_temp % stride == 0) {
                            int ho = ho_temp / stride;
                            if (ho >= 0 && ho < H_out) {
                                for (int kw = 0; kw < KW; ++kw) {
                                    int wo_temp = x + padding - kw;
                                    if (wo_temp % stride == 0) {
                                        int wo = wo_temp / stride;
                                        if (wo >= 0 && wo < W_out) {
                                            int gout_idx = (((b * C_out + co) * D_out + do_) * H_out + ho) * W_out + wo;
                                            int w_idx = ((((co * C_in + ci) * KD + kd) * KH + kh) * KW + kw);
                                            sum_val += grad_output[gout_off + gout_idx] * weight[w_off + w_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    grad_input[gin_off + idx] = sum_val;
}
extern "C" __global__ void transpose_conv_out_inv_kernel(
    const float* src, int s_off,
    float* dst, int d_off,
    int N, int C, int H, int W)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t HW = (int64_t)H * W;
    int64_t NHW = (int64_t)N * HW;
    int64_t total = (int64_t)C * NHW;
    if (idx >= total) return;
    int64_t tmp = idx;
    int64_t hw = tmp % HW; tmp /= HW;
    int64_t n = tmp % N; tmp /= N;
    int64_t c = tmp;
    int64_t src_idx = ((n * C + c) * H + hw / W) * W + hw % W;
    dst[d_off + idx] = src[s_off + src_idx];
}

extern "C" __global__ void col2im_kernel(
    const float* col, int col_off,
    float* im, int im_off,
    int N, int C, int H, int W,
    int KH, int KW, int pad, int stride,
    int H_out, int W_out)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t K = (int64_t)C * KH * KW;
    int64_t HW_out = (int64_t)H_out * W_out;
    int64_t NHW = (int64_t)N * HW_out;
    int64_t total = K * NHW;
    if (idx >= total) return;
    int64_t tmp = idx;
    int64_t hw_out = tmp % HW_out; tmp /= HW_out;
    int64_t n = tmp % N; tmp /= N;
    int64_t k = tmp;
    int64_t kw = k % KW; tmp = k / KW;
    int64_t kh = tmp % KH; tmp /= KH;
    int64_t c = tmp;
    int64_t ho = hw_out / W_out;
    int64_t wo = hw_out % W_out;
    int64_t h = ho * stride - pad + kh;
    int64_t w = wo * stride - pad + kw;
    if (h >= 0 && h < H && w >= 0 && w < W) {
        int64_t im_idx = ((n * C + c) * H + h) * W + w;
        atomicAdd(&im[im_off + im_idx], col[col_off + idx]);
    }
}

extern "C" __global__ void transpose_conv_out_bias_relu_kernel(
    const float* src, int s_off,
    const float* bias, int b_off, int has_bias,
    float* dst, int d_off, int apply_relu,
    int N, int C, int H, int W)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t HW = (int64_t)H * W;
    int64_t NHW = (int64_t)N * HW;
    int64_t total = (int64_t)C * NHW;
    if (idx >= total) return;
    int64_t hw = idx % HW;
    int64_t c = (idx / HW) % C;
    int64_t n = idx / (HW * (int64_t)C);
    float v = src[s_off + c * NHW + n * HW + hw];
    if (has_bias) {
        v += bias[b_off + c];
    }
    if (apply_relu && v < 0.0f) {
        v = 0.0f;
    }
    dst[d_off + idx] = v;
}
