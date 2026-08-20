#include "litetorch/ops.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>

namespace litetorch {
namespace {
struct StorageUseGuard {
    std::vector<std::shared_ptr<StorageImpl>> storages;
    StorageUseGuard(const std::vector<std::shared_ptr<StorageImpl>>& list) : storages(list) {
        for (auto& s : storages) {
            if (s) s->in_use = true;
        }
    }
    ~StorageUseGuard() {
        for (auto& s : storages) {
            if (s) s->in_use = false;
        }
    }
};
}

namespace Ops {

std::shared_ptr<Tensor> w8a8_matmul(
    std::shared_ptr<Tensor> x,
    std::shared_ptr<Tensor> w,
    float x_scale,
    float w_scale
) {
    if (x->shape.size() != 2 || w->shape.size() != 2) {
        throw std::runtime_error("[litetorch Error] w8a8_matmul expects 2D inputs");
    }

    int64_t M = x->shape[0];
    int64_t K = x->shape[1];
    int64_t N = w->shape[0];
    
    if (w->shape[1] != K) {
        throw std::runtime_error("[litetorch Error] Inner dimension mismatch in w8a8_matmul");
    }

    auto x_c = x->is_contiguous() ? x : x->contiguous();
    auto w_c = w->is_contiguous() ? w : w->contiguous();
    auto out = Tensor::create({M, N}, x_c->device, false);
    StorageUseGuard guard({x_c->storage, w_c->storage, out->storage});

    if (x_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::W8A8MatMul);
        cl_mem x_mem = x_c->gpu_data();
        cl_mem w_mem = w_c->gpu_data();
        cl_mem out_mem = out->gpu_data();
        int x_off = x_c->offset;
        int w_off = w_c->offset;
        int out_off = out->offset;
        int M_val = static_cast<int>(M);
        int K_val = static_cast<int>(K);
        int N_val = static_cast<int>(N);

        size_t local_sz[2] = {16, 16};
        size_t global_sz[2] = {
            static_cast<size_t>((M + 15) / 16 * 16),
            static_cast<size_t>((N + 15) / 16 * 16)
        };

        CLBackend::get().launch(kernel, {global_sz[0], global_sz[1]}, {local_sz[0], local_sz[1]},
                                {&x_mem, &x_off, &w_mem, &w_off, &out_mem, &out_off, &M_val, &K_val, &N_val, &x_scale, &w_scale},
                                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(float), sizeof(float)});
    } else {
        float* x_ptr = x_c->data_ptr();
        float* w_ptr = w_c->data_ptr();
        float* out_ptr = out->data_ptr();

        std::vector<int8_t> qx(M * K);
        std::vector<int8_t> qw(N * K);

        for (int64_t i = 0; i < M * K; ++i) {
            float val = std::round(x_ptr[i] / x_scale);
            qx[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, val)));
        }

        for (int64_t i = 0; i < N * K; ++i) {
            float val = std::round(w_ptr[i] / w_scale);
            qw[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, val)));
        }

        float dequant_scale = x_scale * w_scale;

        ThreadPool::get().parallel_for(0, M, [&](int64_t i) {
            float* out_row = out_ptr + i * N;
            const int8_t* qx_row = qx.data() + i * K;
            for (int64_t j = 0; j < N; ++j) {
                int32_t acc = 0;
                const int8_t* qw_row = qw.data() + j * K;
                for (int64_t k = 0; k < K; ++k) {
                    acc += static_cast<int32_t>(qx_row[k]) * static_cast<int32_t>(qw_row[k]);
                }
                out_row[j] = static_cast<float>(acc) * dequant_scale;
            }
        });
    }

    return out;
}

}
}
