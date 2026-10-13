#include "litetorch/ops.h"
#include "litetorch/tensor.h"
#include <stdexcept>

namespace litetorch {

namespace {
std::shared_ptr<Tensor> sc(float v, std::shared_ptr<Tensor> ref) {
    return Tensor::from_vector({v}, {1}, ref->device, false, ref->dtype);
}
}

std::shared_ptr<Tensor> Ops::bilinear(std::shared_ptr<Tensor> input1,
                                       std::shared_ptr<Tensor> input2,
                                       std::shared_ptr<Tensor> weight,
                                       std::shared_ptr<Tensor> bias) {
    auto wshape = weight->shape;
    int64_t out_features = wshape[0];
    int64_t in1_features = wshape[1];
    int64_t in2_features = wshape[2];
    auto x1shape = input1->shape;
    auto x2shape = input2->shape;
    if (x1shape.back() != in1_features || x2shape.back() != in2_features) {
        throw std::invalid_argument("[litetorch Error] bilinear feature mismatch");
    }
    auto x1e = Ops::unsqueeze(input1, -1);
    auto x2e = Ops::unsqueeze(Ops::unsqueeze(input2, 1), -1);
    auto w = weight;
    auto x1w = Ops::matmul(x1e, Ops::permute(w, {0, 2, 1}));
    auto out = Ops::matmul(x1w, x2e);
    out = Ops::squeeze(out, -1);
    out = Ops::squeeze(out, -1);
    if (bias) {
        out = Ops::add(out, bias);
    }
    return out;
}

std::shared_ptr<Tensor> Ops::fold(std::shared_ptr<Tensor> input,
                                   const std::vector<int64_t>& output_size,
                                   const std::vector<int64_t>& kernel_size,
                                   const std::vector<int64_t>& dilation,
                                   const std::vector<int64_t>& padding,
                                   const std::vector<int64_t>& stride) {
    auto ishape = input->shape;
    if (ishape.size() != 3) {
        throw std::invalid_argument("[litetorch Error] fold expects 3D input (N, C*kH*kW, L)");
    }
    int64_t N = ishape[0];
    int64_t H = output_size[0];
    int64_t W = output_size[1];
    int64_t kH = kernel_size[0];
    int64_t kW = kernel_size[1];
    int64_t dH = dilation[0];
    int64_t dW = dilation[1];
    int64_t pH = padding[0];
    int64_t pW = padding[1];
    int64_t sH = stride[0];
    int64_t sW = stride[1];
    int64_t H_out = (H + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    int64_t W_out = (W + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    int64_t C = ishape[1] / (kH * kW);
    auto out = Tensor::from_vector(std::vector<float>((size_t)(N * C * H * W), 0.0f),
                                   {N, C, H, W}, input->device, false, input->dtype);
    auto in_data = input->data_ptr();
    auto out_data = out->data_ptr();
    for (int64_t n = 0; n < N; n++) {
        for (int64_t l = 0; l < H_out * W_out; l++) {
            int64_t oh = l / W_out;
            int64_t ow = l % W_out;
            for (int64_t c = 0; c < C; c++) {
                for (int64_t kh = 0; kh < kH; kh++) {
                    for (int64_t kw = 0; kw < kW; kw++) {
                        int64_t h = oh * sH - pH + kh * dH;
                        int64_t w = ow * sW - pW + kw * dW;
                        if (h >= 0 && h < H && w >= 0 && w < W) {
                            int64_t in_idx = ((n * ishape[1] + c * kH * kW + kh * kW + kw) * (H_out * W_out)) + l;
                            int64_t out_idx = ((n * C + c) * H + h) * W + w;
                            out_data[out_idx] += in_data[in_idx];
                        }
                    }
                }
            }
        }
    }
    return out;
}

std::shared_ptr<Tensor> Ops::unfold(std::shared_ptr<Tensor> input,
                                     const std::vector<int64_t>& kernel_size,
                                     const std::vector<int64_t>& dilation,
                                     const std::vector<int64_t>& padding,
                                     const std::vector<int64_t>& stride) {
    auto ishape = input->shape;
    if (ishape.size() != 4) {
        throw std::invalid_argument("[litetorch Error] unfold expects 4D input (N, C, H, W)");
    }
    int64_t N = ishape[0];
    int64_t C = ishape[1];
    int64_t H = ishape[2];
    int64_t W = ishape[3];
    int64_t kH = kernel_size[0];
    int64_t kW = kernel_size[1];
    int64_t dH = dilation[0];
    int64_t dW = dilation[1];
    int64_t pH = padding[0];
    int64_t pW = padding[1];
    int64_t sH = stride[0];
    int64_t sW = stride[1];
    int64_t H_out = (H + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    int64_t W_out = (W + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    int64_t L = H_out * W_out;
    auto out = Tensor::from_vector(std::vector<float>((size_t)(N * C * kH * kW * L), 0.0f),
                                   {N, C * kH * kW, L}, input->device, false, input->dtype);
    auto in_data = input->data_ptr();
    auto out_data = out->data_ptr();
    for (int64_t n = 0; n < N; n++) {
        for (int64_t oh = 0; oh < H_out; oh++) {
            for (int64_t ow = 0; ow < W_out; ow++) {
                int64_t l = oh * W_out + ow;
                for (int64_t c = 0; c < C; c++) {
                    for (int64_t kh = 0; kh < kH; kh++) {
                        for (int64_t kw = 0; kw < kW; kw++) {
                            int64_t h = oh * sH - pH + kh * dH;
                            int64_t w = ow * sW - pW + kw * dW;
                            float val = 0.0f;
                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                int64_t in_idx = ((n * C + c) * H + h) * W + w;
                                val = in_data[in_idx];
                            }
                            int64_t out_idx = ((n * (C * kH * kW) + c * kH * kW + kh * kW + kw) * L) + l;
                            out_data[out_idx] = val;
                        }
                    }
                }
            }
        }
    }
    return out;
}

}
