#ifndef LITETORCH_CUDNN_DYN_H
#define LITETORCH_CUDNN_DYN_H

#ifndef __HIP_PLATFORM_AMD__

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <dlfcn.h>
#include <string>

typedef void* lt_cudnnHandle_t;
typedef void* lt_cudnnTensorDescriptor_t;
typedef void* lt_cudnnFilterDescriptor_t;
typedef void* lt_cudnnConvolutionDescriptor_t;

enum {
    LT_CUDNN_TENSOR_NCHW = 0,
    LT_CUDNN_DATA_FLOAT = 0,
    LT_CUDNN_CROSS_CORRELATION = 1,
    LT_CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    LT_CUDNN_SOFTMAX_ACCURATE = 1,
    LT_CUDNN_SOFTMAX_MODE_CHANNEL = 1
};

typedef int (*lt_cudnnCreate_fn)(lt_cudnnHandle_t*);
typedef int (*lt_cudnnSetStream_fn)(lt_cudnnHandle_t, void*);
typedef int (*lt_cudnnCreateTensorDescriptor_fn)(lt_cudnnTensorDescriptor_t*);
typedef int (*lt_cudnnSetTensor4dDescriptor_fn)(lt_cudnnTensorDescriptor_t, int, int, int, int, int, int);
typedef int (*lt_cudnnCreateFilterDescriptor_fn)(lt_cudnnFilterDescriptor_t*);
typedef int (*lt_cudnnSetFilter4dDescriptor_fn)(lt_cudnnFilterDescriptor_t, int, int, int, int, int, int);
typedef int (*lt_cudnnCreateConvolutionDescriptor_fn)(lt_cudnnConvolutionDescriptor_t*);
typedef int (*lt_cudnnSetConvolution2dDescriptor_fn)(lt_cudnnConvolutionDescriptor_t, int, int, int, int, int, int, int, int);
typedef int (*lt_cudnnConvolutionForward_fn)(lt_cudnnHandle_t, const void*, lt_cudnnTensorDescriptor_t, const void*, lt_cudnnFilterDescriptor_t, const void*, lt_cudnnConvolutionDescriptor_t, int, void*, size_t, const void*, lt_cudnnTensorDescriptor_t, void*);
typedef int (*lt_cudnnAddTensor_fn)(lt_cudnnHandle_t, const void*, lt_cudnnTensorDescriptor_t, const void*, const void*, lt_cudnnTensorDescriptor_t, void*);
typedef int (*lt_cudnnSoftmaxForward_fn)(lt_cudnnHandle_t, int, int, const void*, lt_cudnnTensorDescriptor_t, const void*, const void*, lt_cudnnTensorDescriptor_t, void*);
typedef int (*lt_cudnnDestroyTensorDescriptor_fn)(lt_cudnnTensorDescriptor_t);
typedef int (*lt_cudnnDestroyFilterDescriptor_fn)(lt_cudnnFilterDescriptor_t);
typedef int (*lt_cudnnDestroyConvolutionDescriptor_fn)(lt_cudnnConvolutionDescriptor_t);
typedef size_t (*lt_cudnnGetVersion_fn)();

struct LtCudnnApi {
    lt_cudnnCreate_fn Create;
    lt_cudnnSetStream_fn SetStream;
    lt_cudnnCreateTensorDescriptor_fn CreateTensorDescriptor;
    lt_cudnnSetTensor4dDescriptor_fn SetTensor4dDescriptor;
    lt_cudnnCreateFilterDescriptor_fn CreateFilterDescriptor;
    lt_cudnnSetFilter4dDescriptor_fn SetFilter4dDescriptor;
    lt_cudnnCreateConvolutionDescriptor_fn CreateConvolutionDescriptor;
    lt_cudnnSetConvolution2dDescriptor_fn SetConvolution2dDescriptor;
    lt_cudnnConvolutionForward_fn ConvolutionForward;
    lt_cudnnAddTensor_fn AddTensor;
    lt_cudnnSoftmaxForward_fn SoftmaxForward;
    lt_cudnnDestroyTensorDescriptor_fn DestroyTensorDescriptor;
    lt_cudnnDestroyFilterDescriptor_fn DestroyFilterDescriptor;
    lt_cudnnDestroyConvolutionDescriptor_fn DestroyConvolutionDescriptor;
    lt_cudnnGetVersion_fn GetVersion;
};

inline LtCudnnApi g_cudnn{};
inline bool g_cudnn_available = false;
inline void* g_cudnn_lib = nullptr;

inline bool cudnn_dyn_bind(void* lib) {
    LtCudnnApi api{};
    bool ok = true;
#define LT_CUDNN_LOAD(sym) \
    api.sym = reinterpret_cast<lt_cudnn##sym##_fn>(dlsym(lib, "cudnn" #sym)); \
    ok = ok && (api.sym != nullptr)
    LT_CUDNN_LOAD(Create);
    LT_CUDNN_LOAD(SetStream);
    LT_CUDNN_LOAD(CreateTensorDescriptor);
    LT_CUDNN_LOAD(SetTensor4dDescriptor);
    LT_CUDNN_LOAD(CreateFilterDescriptor);
    LT_CUDNN_LOAD(SetFilter4dDescriptor);
    LT_CUDNN_LOAD(CreateConvolutionDescriptor);
    LT_CUDNN_LOAD(SetConvolution2dDescriptor);
    LT_CUDNN_LOAD(ConvolutionForward);
    LT_CUDNN_LOAD(AddTensor);
    LT_CUDNN_LOAD(SoftmaxForward);
    LT_CUDNN_LOAD(DestroyTensorDescriptor);
    LT_CUDNN_LOAD(DestroyFilterDescriptor);
    LT_CUDNN_LOAD(DestroyConvolutionDescriptor);
    LT_CUDNN_LOAD(GetVersion);
#undef LT_CUDNN_LOAD
    if (!ok) return false;
    g_cudnn = api;
    g_cudnn_lib = lib;
    return true;
}

inline bool cudnn_dyn_try(const char* path) {
    if (!path || !path[0]) return false;
    void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) return false;
    if (!cudnn_dyn_bind(lib)) {
        dlclose(lib);
        return false;
    }
    return true;
}

inline bool cudnn_dyn_init() {
    static bool done = false;
    if (done) return g_cudnn_available;
    done = true;
    const char* off = std::getenv("LITETORCH_USE_CUDNN");
    if (off && off[0] == '0') return false;
    if (cudnn_dyn_try(std::getenv("LITETORCH_CUDNN_LIB"))) {
        g_cudnn_available = true;
        return true;
    }
    const char* bare[] = {"libcudnn.so.9", "libcudnn.so"};
    for (const char* b : bare) {
        if (cudnn_dyn_try(b)) {
            g_cudnn_available = true;
            return true;
        }
    }
    typedef const wchar_t* (*py_getpath_fn)();
    py_getpath_fn py_getpath =
        reinterpret_cast<py_getpath_fn>(dlsym(RTLD_DEFAULT, "Py_GetPath"));
    if (py_getpath) {
        const wchar_t* wpath = py_getpath();
        if (wpath) {
            char npath[16384];
            std::size_t n = std::wcstombs(npath, wpath, sizeof(npath) - 1);
            if (n != static_cast<std::size_t>(-1)) {
                npath[n] = '\0';
                std::string sp(npath);
                std::size_t start = 0;
                const char* tails[] = {
                    "/nvidia/cudnn/lib/libcudnn.so.9",
                    "/nvidia/cudnn/lib64/libcudnn.so.9",
                    "/libcudnn.so.9"
                };
                while (start <= sp.size()) {
                    std::size_t end = sp.find(':', start);
                    std::string entry = sp.substr(start, end == std::string::npos ? end : end - start);
                    if (!entry.empty()) {
                        for (const char* t : tails) {
                            if (cudnn_dyn_try((entry + t).c_str())) {
                                g_cudnn_available = true;
                                return true;
                            }
                        }
                    }
                    if (end == std::string::npos) break;
                    start = end + 1;
                }
            }
        }
    }
    const char* cuda_home = std::getenv("CUDA_HOME");
    if (cuda_home && cuda_home[0]) {
        if (cudnn_dyn_try((std::string(cuda_home) + "/lib64/libcudnn.so.9").c_str())) {
            g_cudnn_available = true;
            return true;
        }
    }
    if (cudnn_dyn_try("/usr/local/cuda/lib64/libcudnn.so.9")) {
        g_cudnn_available = true;
        return true;
    }
    return false;
}

lt_cudnnHandle_t get_cudnn_handle();

#endif

#endif
