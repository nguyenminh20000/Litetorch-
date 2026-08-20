#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include <cmath>

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

class RopeNode : public Node {
public:
    RopeNode() : Node("Rope") {}
    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        auto cos = saved_tensors[0];
        auto sin = saved_tensors[1];
        auto x_shape = saved_tensors[2]->shape;
        auto device = grad_output->device;

        auto grad_x = Tensor::create(x_shape, device, false);

        int64_t B = x_shape[0];
        int64_t H = x_shape[1];
        int64_t T = x_shape[2];
        int64_t D = x_shape[3];

        auto grad_output_c = grad_output->is_contiguous() ? grad_output : grad_output->contiguous();
        auto cos_c = cos->is_contiguous() ? cos : cos->contiguous();
        auto sin_c = sin->is_contiguous() ? sin : sin->contiguous();

        if (device.type == DeviceType::GPU) {
            StorageUseGuard guard({grad_output_c->storage, cos_c->storage, sin_c->storage, grad_x->storage});
            cl_mem dy_mem = grad_output_c->gpu_data();
            int dy_off = grad_output_c->offset;
            cl_mem cos_mem = cos_c->gpu_data();
            int cos_off = cos_c->offset;
            cl_mem sin_mem = sin_c->gpu_data();
            int sin_off = sin_c->offset;
            cl_mem dx_mem = grad_x->gpu_data();
            int dx_off = grad_x->offset;

            int b_val = B, h_val = H, t_val = T, d_val = D;

            auto kernel = CLBackend::get().get_kernel(KernelID::RopeBackward);
            CLBackend::get().launch(kernel, {static_cast<size_t>(B * H * T * D)}, {},
                {&dy_mem, &dy_off, &cos_mem, &cos_off, &sin_mem, &sin_off, &dx_mem, &dx_off, &b_val, &h_val, &t_val, &d_val},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
        } else {
            float* dy_ptr = grad_output_c->data_ptr();
            float* cos_ptr = cos_c->data_ptr();
            float* sin_ptr = sin_c->data_ptr();
            float* dx_ptr = grad_x->data_ptr();

            int64_t half_d = D / 2;

            ThreadPool::get().parallel_for(0, B * H * T, [&](int64_t idx) {
                int64_t t = idx % T;
                int64_t offset = idx * D;
                for (int64_t i = 0; i < half_d; ++i) {
                    float cos_val = cos_ptr[t * half_d + i];
                    float sin_val = sin_ptr[t * half_d + i];
                    dx_ptr[offset + 2 * i] = dy_ptr[offset + 2 * i] * cos_val + dy_ptr[offset + 2 * i + 1] * sin_val;
                    dx_ptr[offset + 2 * i + 1] = dy_ptr[offset + 2 * i + 1] * cos_val - dy_ptr[offset + 2 * i] * sin_val;
                }
            });
        }

        return {grad_x, nullptr, nullptr};
    }
};

namespace Ops {

std::shared_ptr<Tensor> rope(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> cos, std::shared_ptr<Tensor> sin) {
    if (x->shape.size() != 4) {
        throw std::runtime_error("[litetorch Error] RoPE input must be a 4D tensor [B, H, T, D]");
    }

    int64_t B = x->shape[0];
    int64_t H = x->shape[1];
    int64_t T = x->shape[2];
    int64_t D = x->shape[3];

    auto x_c = x->is_contiguous() ? x : x->contiguous();
    auto cos_c = cos->is_contiguous() ? cos : cos->contiguous();
    auto sin_c = sin->is_contiguous() ? sin : sin->contiguous();

    bool requires_grad = x->requires_grad || cos->requires_grad || sin->requires_grad;
    auto y = Tensor::create(x->shape, x->device, requires_grad);

    if (x->device.type == DeviceType::GPU) {
        StorageUseGuard guard({x_c->storage, cos_c->storage, sin_c->storage, y->storage});
        cl_mem x_mem = x_c->gpu_data();
        int x_off = x_c->offset;
        cl_mem cos_mem = cos_c->gpu_data();
        int cos_off = cos_c->offset;
        cl_mem sin_mem = sin_c->gpu_data();
        int sin_off = sin_c->offset;
        cl_mem y_mem = y->gpu_data();
        int y_off = y->offset;

        int b_val = B, h_val = H, t_val = T, d_val = D;

        auto kernel = CLBackend::get().get_kernel(KernelID::RopeForward);
        CLBackend::get().launch(kernel, {static_cast<size_t>(B * H * T * D)}, {},
            {&x_mem, &x_off, &cos_mem, &cos_off, &sin_mem, &sin_off, &y_mem, &y_off, &b_val, &h_val, &t_val, &d_val},
            {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)});
    } else {
        float* x_ptr = x_c->data_ptr();
        float* cos_ptr = cos_c->data_ptr();
        float* sin_ptr = sin_c->data_ptr();
        float* y_ptr = y->data_ptr();

        int64_t half_d = D / 2;

        ThreadPool::get().parallel_for(0, B * H * T, [&](int64_t idx) {
            int64_t t = idx % T;
            int64_t offset = idx * D;
            for (int64_t i = 0; i < half_d; ++i) {
                float cos_val = cos_ptr[t * half_d + i];
                float sin_val = sin_ptr[t * half_d + i];
                y_ptr[offset + 2 * i] = x_ptr[offset + 2 * i] * cos_val - x_ptr[offset + 2 * i + 1] * sin_val;
                y_ptr[offset + 2 * i + 1] = x_ptr[offset + 2 * i + 1] * cos_val + x_ptr[offset + 2 * i] * sin_val;
            }
        });
    }

    if (Autograd::active_tensors.size() > 0 || requires_grad) {
        auto node = std::make_shared<RopeNode>();
        node->next_nodes = {x->creator, cos->creator, sin->creator};
        node->inputs = {{x, x->requires_grad}, {cos, cos->requires_grad}, {sin, sin->requires_grad}};
        node->saved_tensors = {cos, sin, x};
        y->creator = node;
        node->output = y;
    }

    return y;
}

}
}
