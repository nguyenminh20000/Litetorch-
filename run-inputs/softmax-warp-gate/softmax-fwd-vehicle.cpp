#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

typedef bool (*gpu_init_t)();
typedef void (*fa_bwd_t)(void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t,
                         void*, int64_t, void*, int64_t, void*, int64_t, void*, int64_t,
                         int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float, int);

static uint64_t rng_state = 0;

static uint64_t rng_next() {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1DULL;
}

static float rng_uniform() {
    double u = (double)(rng_next() >> 11) * (1.0 / 9007199254740992.0);
    return (float)(u * 2.0 - 1.0);
}

static int fail_cuda(cudaError_t err, const char* what) {
    fprintf(stderr, "FWD-VEHICLE CUDA-ERROR %s: %s\n", what, cudaGetErrorString(err));
    return 2;
}

static int write_bytes(const char* path, const void* data, size_t bytes) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "FWD-VEHICLE IO-ERROR cannot open %s\n", path);
        return 1;
    }
    if (bytes > 0 && fwrite(data, 1, bytes, f) != bytes) {
        fprintf(stderr, "FWD-VEHICLE IO-ERROR short write %s\n", path);
        fclose(f);
        return 1;
    }
    fclose(f);
    return 0;
}

static bool file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        fprintf(stderr, "usage: %s Tk causal half outdir [liblitetorch_gpu.so]\n", argv[0]);
        fprintf(stderr, "  Tk: key length (1,4,16,32,33,64); causal: 0/1; half: 0/1\n");
        return 1;
    }
    int Tk = atoi(argv[1]);
    int causal = atoi(argv[2]);
    int half = atoi(argv[3]);
    const char* outdir = argv[4];
    if (Tk < 1 || Tk > 4096 || (causal != 0 && causal != 1) || (half != 0 && half != 1)) {
        fprintf(stderr, "FWD-VEHICLE usage error: bad Tk/causal/half\n");
        return 1;
    }
    const int B = 1;
    const int H = 2;
    const int H_kv = 2;
    const int Tq = Tk;
    const int D = 8;
    const float scale = 1.0f / (float)sqrt((double)D);
    const int64_t n_q = (int64_t)B * H * Tq * D;
    const int64_t n_kv = (int64_t)B * H_kv * Tk * D;

    const char* fa_paths[3] = {"libflash_attn.so", "/usr/local/cuda/lib64/libflash_attn.so", "./libflash_attn.so"};
    for (int i = 0; i < 3; i++) {
        if (file_exists(fa_paths[i])) {
            printf("FWD-VEHICLE FA3-PRESENT %s: native backward is bypassed, FWD warp kernels NOT_COVERED\n", fa_paths[i]);
            return 4;
        }
    }

    char libpath[4096];
    if (argc == 6) {
        snprintf(libpath, sizeof(libpath), "%s", argv[5]);
    } else if (getenv("LITETORCH_GPU_SO")) {
        snprintf(libpath, sizeof(libpath), "%s", getenv("LITETORCH_GPU_SO"));
    } else if (getenv("LITETORCH_TREE")) {
        snprintf(libpath, sizeof(libpath), "%s/python/liblitetorch_gpu.so", getenv("LITETORCH_TREE"));
    } else {
        snprintf(libpath, sizeof(libpath), "./liblitetorch_gpu.so");
    }

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) return fail_cuda(err, "cudaSetDevice(0)");
    int dev_count = 0;
    err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess) return fail_cuda(err, "cudaGetDeviceCount");
    if (dev_count < 1) {
        fprintf(stderr, "FWD-VEHICLE CUDA-ERROR no CUDA devices\n");
        return 2;
    }

    void* handle = dlopen(libpath, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        fprintf(stderr, "FWD-VEHICLE LOAD-ERROR dlopen %s: %s\n", libpath, dlerror());
        return 3;
    }
    dlerror();
    gpu_init_t gpu_init_fn = (gpu_init_t)dlsym(handle, "gpu_init");
    const char* sym_err = dlerror();
    if (sym_err || !gpu_init_fn) {
        fprintf(stderr, "FWD-VEHICLE LOAD-ERROR dlsym gpu_init: %s\n", sym_err ? sym_err : "null");
        return 3;
    }
    fa_bwd_t bwd_fn = (fa_bwd_t)dlsym(handle, "gpu_flash_attention_backward");
    sym_err = dlerror();
    if (sym_err || !bwd_fn) {
        fprintf(stderr, "FWD-VEHICLE LOAD-ERROR dlsym gpu_flash_attention_backward: %s\n", sym_err ? sym_err : "null");
        return 3;
    }
    fa_bwd_t bwd_half_fn = (fa_bwd_t)dlsym(handle, "gpu_flash_attention_backward_half");
    sym_err = dlerror();
    if (sym_err || !bwd_half_fn) {
        fprintf(stderr, "FWD-VEHICLE LOAD-ERROR dlsym gpu_flash_attention_backward_half: %s\n", sym_err ? sym_err : "null");
        return 3;
    }
    if (!gpu_init_fn()) {
        fprintf(stderr, "FWD-VEHICLE LOAD-ERROR gpu_init() returned false\n");
        return 3;
    }

    mkdir(outdir, 0755);

    float* h_q = (float*)malloc((size_t)n_q * sizeof(float));
    float* h_k = (float*)malloc((size_t)n_kv * sizeof(float));
    float* h_v = (float*)malloc((size_t)n_kv * sizeof(float));
    float* h_do = (float*)malloc((size_t)n_q * sizeof(float));
    if (!h_q || !h_k || !h_v || !h_do) {
        fprintf(stderr, "FWD-VEHICLE IO-ERROR host malloc failed\n");
        return 1;
    }
    rng_state = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)(1000 + Tk) * 0xBF58476D1CE4E5B9ULL);
    for (int i = 0; i < 16; i++) rng_next();
    for (int64_t i = 0; i < n_q; i++) h_q[i] = rng_uniform();
    for (int64_t i = 0; i < n_kv; i++) h_k[i] = rng_uniform();
    for (int64_t i = 0; i < n_kv; i++) h_v[i] = rng_uniform();
    rng_state = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)(2000 + Tk) * 0xBF58476D1CE4E5B9ULL);
    for (int i = 0; i < 16; i++) rng_next();
    for (int64_t i = 0; i < n_q; i++) h_do[i] = rng_uniform();

    __half* h_qh = nullptr;
    __half* h_kh = nullptr;
    __half* h_vh = nullptr;
    __half* h_doh = nullptr;
    if (half) {
        h_qh = (__half*)malloc((size_t)n_q * sizeof(__half));
        h_kh = (__half*)malloc((size_t)n_kv * sizeof(__half));
        h_vh = (__half*)malloc((size_t)n_kv * sizeof(__half));
        h_doh = (__half*)malloc((size_t)n_q * sizeof(__half));
        if (!h_qh || !h_kh || !h_vh || !h_doh) {
            fprintf(stderr, "FWD-VEHICLE IO-ERROR host half malloc failed\n");
            return 1;
        }
        for (int64_t i = 0; i < n_q; i++) h_qh[i] = __float2half(h_q[i]);
        for (int64_t i = 0; i < n_kv; i++) h_kh[i] = __float2half(h_k[i]);
        for (int64_t i = 0; i < n_kv; i++) h_vh[i] = __float2half(h_v[i]);
        for (int64_t i = 0; i < n_q; i++) h_doh[i] = __float2half(h_do[i]);
    }

    size_t bytes_q = (size_t)n_q * (half ? sizeof(__half) : sizeof(float));
    size_t bytes_kv = (size_t)n_kv * (half ? sizeof(__half) : sizeof(float));
    void *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    void *d_do = nullptr, *d_dq = nullptr, *d_dk = nullptr, *d_dv = nullptr;
    err = cudaMalloc(&d_q, bytes_q);
    if (err != cudaSuccess) return fail_cuda(err, "cudaMalloc d_q");
    err = cudaMalloc(&d_k, bytes_kv);
    if (err != cudaSuccess) return fail_cuda(err, "cudaMalloc d_k");
    err = cudaMalloc(&d_v, bytes_kv);
    if (err != cudaSuccess) return fail_cuda(err, "cudaMalloc d_v");
    err = cudaMalloc(&d_o, bytes_q);
    if (err != cudaSuccess) return fail_cuda(err, "cudaMalloc d_o");
    err = cudaMalloc(&d_do, bytes_q);
    if (err != cudaSuccess) return fail_cuda(err, "cudaMalloc d_do");
    err = cudaMalloc(&d_dq, bytes_q);
    if (err != cudaSuccess) return fail_cuda(err, "cudaMalloc d_dq");
    err = cudaMalloc(&d_dk, bytes_kv);
    if (err != cudaSuccess) return fail_cuda(err, "cudaMalloc d_dk");
    err = cudaMalloc(&d_dv, bytes_kv);
    if (err != cudaSuccess) return fail_cuda(err, "cudaMalloc d_dv");

    err = cudaMemcpy(d_q, half ? (const void*)h_qh : (const void*)h_q, bytes_q, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail_cuda(err, "H2D d_q");
    err = cudaMemcpy(d_k, half ? (const void*)h_kh : (const void*)h_k, bytes_kv, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail_cuda(err, "H2D d_k");
    err = cudaMemcpy(d_v, half ? (const void*)h_vh : (const void*)h_v, bytes_kv, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail_cuda(err, "H2D d_v");
    err = cudaMemset(d_o, 0, bytes_q);
    if (err != cudaSuccess) return fail_cuda(err, "memset d_o");
    err = cudaMemcpy(d_do, half ? (const void*)h_doh : (const void*)h_do, bytes_q, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail_cuda(err, "H2D d_do");
    err = cudaMemset(d_dq, 0, bytes_q);
    if (err != cudaSuccess) return fail_cuda(err, "memset d_dq");
    err = cudaMemset(d_dk, 0, bytes_kv);
    if (err != cudaSuccess) return fail_cuda(err, "memset d_dk");
    err = cudaMemset(d_dv, 0, bytes_kv);
    if (err != cudaSuccess) return fail_cuda(err, "memset d_dv");

    if (half) {
        bwd_half_fn(d_dq, 0, d_dk, 0, d_dv, 0, d_o, 0, d_do, 0,
                    d_q, 0, d_k, 0, d_v, 0,
                    B, H, H_kv, Tq, Tk, D, scale, causal);
    } else {
        bwd_fn(d_dq, 0, d_dk, 0, d_dv, 0, d_o, 0, d_do, 0,
               d_q, 0, d_k, 0, d_v, 0,
               B, H, H_kv, Tq, Tk, D, scale, causal);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return fail_cuda(err, "cudaDeviceSynchronize after backward");
    err = cudaGetLastError();
    if (err != cudaSuccess) return fail_cuda(err, "cudaGetLastError after backward");

    char path[4400];
    int rc = 0;
    if (half) {
        __half* h_dqh = (__half*)malloc(bytes_q);
        __half* h_dkh = (__half*)malloc(bytes_kv);
        __half* h_dvh = (__half*)malloc(bytes_kv);
        if (!h_dqh || !h_dkh || !h_dvh) {
            fprintf(stderr, "FWD-VEHICLE IO-ERROR host grad malloc failed\n");
            return 1;
        }
        err = cudaMemcpy(h_dqh, d_dq, bytes_q, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return fail_cuda(err, "D2H d_dq");
        err = cudaMemcpy(h_dkh, d_dk, bytes_kv, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return fail_cuda(err, "D2H d_dk");
        err = cudaMemcpy(h_dvh, d_dv, bytes_kv, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return fail_cuda(err, "D2H d_dv");
        snprintf(path, sizeof(path), "%s/q.bin", outdir);
        rc = write_bytes(path, h_qh, bytes_q);
        if (rc) return rc;
        snprintf(path, sizeof(path), "%s/k.bin", outdir);
        rc = write_bytes(path, h_kh, bytes_kv);
        if (rc) return rc;
        snprintf(path, sizeof(path), "%s/v.bin", outdir);
        rc = write_bytes(path, h_vh, bytes_kv);
        if (rc) return rc;
        snprintf(path, sizeof(path), "%s/do.bin", outdir);
        rc = write_bytes(path, h_doh, bytes_q);
        if (rc) return rc;
        snprintf(path, sizeof(path), "%s/dq.bin", outdir);
        rc = write_bytes(path, h_dqh, bytes_q);
        if (rc) return rc;
        snprintf(path, sizeof(path), "%s/dk.bin", outdir);
        rc = write_bytes(path, h_dkh, bytes_kv);
        if (rc) return rc;
        snprintf(path, sizeof(path), "%s/dv.bin", outdir);
        rc = write_bytes(path, h_dvh, bytes_kv);
        if (rc) return rc;
    } else {
        float* h_dq = (float*)malloc(bytes_q);
        float* h_dk = (float*)malloc(bytes_kv);
        float* h_dv = (float*)malloc(bytes_kv);
        if (!h_dq || !h_dk || !h_dv) {
            fprintf(stderr, "FWD-VEHICLE IO-ERROR host grad malloc failed\n");
            return 1;
        }
        err = cudaMemcpy(h_dq, d_dq, bytes_q, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return fail_cuda(err, "D2H d_dq");
        err = cudaMemcpy(h_dk, d_dk, bytes_kv, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return fail_cuda(err, "D2H d_dk");
        err = cudaMemcpy(h_dv, d_dv, bytes_kv, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return fail_cuda(err, "D2H d_dv");
        snprintf(path, sizeof(path), "%s/q.bin", outdir);
        rc = write_bytes(path, h_q, bytes_q);
        if (rc) return rc;
        snprintf(path, sizeof(path), "%s/k.bin", outdir);
        rc = write_bytes(path, h_k, bytes_kv);
        if (rc) return rc;
        snprintf(path, sizeof(path), "%s/v.bin", outdir);
        rc = write_bytes(path, h_v, bytes_kv);
        if (rc) return rc;
        snprintf(path, sizeof(path), "%s/do.bin", outdir);
        rc = write_bytes(path, h_do, bytes_q);
        if (rc) return rc;
        snprintf(path, sizeof(path), "%s/dq.bin", outdir);
        rc = write_bytes(path, h_dq, bytes_q);
        if (rc) return rc;
        snprintf(path, sizeof(path), "%s/dk.bin", outdir);
        rc = write_bytes(path, h_dk, bytes_kv);
        if (rc) return rc;
        snprintf(path, sizeof(path), "%s/dv.bin", outdir);
        rc = write_bytes(path, h_dv, bytes_kv);
        if (rc) return rc;
    }

    snprintf(path, sizeof(path), "%s/meta.txt", outdir);
    FILE* mf = fopen(path, "w");
    if (!mf) {
        fprintf(stderr, "FWD-VEHICLE IO-ERROR cannot open %s\n", path);
        return 1;
    }
    fprintf(mf, "B=%d\nH=%d\nH_kv=%d\nTq=%d\nTk=%d\nD=%d\nscale=%.10f\ncausal=%d\nhalf=%d\nn_q=%lld\nn_kv=%lld\nseed_q=%d\nseed_do=%d\nlib=%s\n",
            B, H, H_kv, Tq, Tk, D, (double)scale, causal, half,
            (long long)n_q, (long long)n_kv, 1000 + Tk, 2000 + Tk, libpath);
    fclose(mf);

    printf("FWD-VEHICLE DONE Tk=%d causal=%d half=%d outdir=%s lib=%s\n", Tk, causal, half, outdir, libpath);
    return 0;
}
