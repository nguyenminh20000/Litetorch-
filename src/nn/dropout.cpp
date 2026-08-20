#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/cl_backend.h"
#include <random>

namespace litetorch {
extern std::mt19937& get_generator();
extern const std::string litetorch_kernels_src;

namespace nn {

Dropout::Dropout(float p) : p(p) {}

std::shared_ptr<Tensor> Dropout::forward(std::shared_ptr<Tensor> input) {
    if (!training || p == 0.0f) return input;

    if (input->device.type == DeviceType::GPU) {
        auto mask = Tensor::create(input->shape, input->device);
        auto kernel = CLBackend::get().get_kernel("litetorch_kernels", litetorch_kernels_src, "generate_dropout_mask");
        cl_mem mask_mem = mask->gpu_data();
        int mask_off = mask->offset;
        float scale = 1.0f / (1.0f - p);
        int size = mask->numel();
        std::mt19937& gen = get_generator();
        unsigned int seed = gen();

        CLBackend::get().launch(kernel, {static_cast<size_t>(size)}, {},
            {&mask_mem, &mask_off, &p, &scale, &seed, &size},
            {sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(float), sizeof(unsigned int), sizeof(int)});

        return Ops::mul(input, mask);
    }

    std::mt19937& gen = get_generator();
    std::bernoulli_distribution dis(1.0f - p);

    auto mask = Tensor::create(input->shape, Device(DeviceType::CPU, 0), false);
    float* m_ptr = mask->data_ptr();
    float scale = 1.0f / (1.0f - p);
    for (size_t i = 0; i < mask->numel(); ++i) {
        m_ptr[i] = dis(gen) ? scale : 0.0f;
    }
    return Ops::mul(input, mask);
}

}
}
