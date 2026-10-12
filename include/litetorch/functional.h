#ifndef LITETORCH_FUNCTIONAL_H
#define LITETORCH_FUNCTIONAL_H

#include "litetorch/tensor.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace litetorch {
namespace functional {

std::shared_ptr<Tensor> dropout(std::shared_ptr<Tensor> x, double p = 0.5, bool training = true);

std::shared_ptr<Tensor> linear(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> weight,
                               std::shared_ptr<Tensor> bias = nullptr);

std::shared_ptr<Tensor> pad(std::shared_ptr<Tensor> x, const std::vector<int64_t>& pad,
                            const std::string& mode = "constant", double value = 0.0);

std::shared_ptr<Tensor> interpolate(std::shared_ptr<Tensor> x,
                                    const std::vector<int64_t>& size,
                                    const std::vector<double>& scale_factor,
                                    const std::string& mode = "nearest",
                                    bool align_corners = false);

std::shared_ptr<Tensor> nll_loss(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
                                 std::shared_ptr<Tensor> weight = nullptr,
                                 const std::string& reduction = "mean",
                                 int64_t ignore_index = -100);

std::shared_ptr<Tensor> bce_with_logits(std::shared_ptr<Tensor> input,
                                        std::shared_ptr<Tensor> target,
                                        const std::string& reduction = "mean");

std::shared_ptr<Tensor> normalize(std::shared_ptr<Tensor> x, double p = 2.0, int64_t dim = 1);

std::shared_ptr<Tensor> glu(std::shared_ptr<Tensor> x, int64_t dim = -1);

std::shared_ptr<Tensor> elu(std::shared_ptr<Tensor> x, double alpha = 1.0);

}
}

#endif
