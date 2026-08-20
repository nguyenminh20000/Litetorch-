#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/autograd.h"
#include "litetorch/cl_backend.h"
#include "litetorch/thread_pool.h"
#include <cmath>
#include <algorithm>

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

class FakeQuantizeNode : public Node {
public:
    float scale;
    float zero_point;
    int bits;
    FakeQuantizeNode(float scale, float zero_point, int bits)
        : Node("FakeQuantize"), scale(scale), zero_point(zero_point), bits(bits) {}

    std::vector<std::shared_ptr<Tensor>> backward(std::shared_ptr<Tensor> grad_output) override {
        return { grad_output };
    }
};

}

namespace Ops {

std::shared_ptr<Tensor> fake_quantize(std::shared_ptr<Tensor> input, float scale, float zero_point, int bits) {
    if (input->dtype != DataType::FP32) {
        auto input_fp32 = input->cast(DataType::FP32);
        auto out_fp32 = fake_quantize(input_fp32, scale, zero_point, bits);
        auto out = out_fp32->cast(input->dtype);
        if (input->requires_grad) {
            auto node = std::make_shared<FakeQuantizeNode>(scale, zero_point, bits);
            node->inputs = { {input, true} };
            node->next_nodes = { input->creator };
            node->output = out;
            out->creator = node;
            out->requires_grad = true;
        }
        return out;
    }
    auto input_c = input->is_contiguous() ? input : input->contiguous();
    auto out = Tensor::create(input_c->shape, input_c->device);
    StorageUseGuard guard({input_c->storage, out->storage});

    float bound = std::pow(2.0f, bits - 1) - 1.0f;
    float clip_min = -bound;
    float clip_max = bound;

    if (input_c->device.type == DeviceType::GPU) {
        auto kernel = CLBackend::get().get_kernel(KernelID::FakeQuantizeForward);
        int size = out->numel();
        cl_mem a_mem = input_c->gpu_data();
        cl_mem b_mem = out->gpu_data();
        int a_off = input_c->offset;
        int b_off = out->offset;
        CLBackend::get().launch(
            kernel,
            {static_cast<size_t>(size)},
            {},
            {&a_mem, &a_off, &scale, &zero_point, &clip_min, &clip_max, &b_mem, &b_off, &size},
            {sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(cl_mem), sizeof(int), sizeof(int)}
        );
    } else {
        float* src = input_c->data_ptr();
        float* dst = out->data_ptr();
        size_t size = out->numel();
        ThreadPool::get().parallel_for(0, size, [&](int64_t i) {
            float val = src[i] / scale + zero_point;
            float rounded = std::round(val);
            if (rounded < clip_min) rounded = clip_min;
            if (rounded > clip_max) rounded = clip_max;
            dst[i] = (rounded - zero_point) * scale;
        });
    }

    if (input->requires_grad) {
        auto node = std::make_shared<FakeQuantizeNode>(scale, zero_point, bits);
        node->inputs = { {input, true} };
        node->next_nodes = { input->creator };
        node->output = out;
        out->creator = node;
        out->requires_grad = true;
    }
    return out;
}

}
}
