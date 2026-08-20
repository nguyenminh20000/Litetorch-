#ifndef LITETORCH_GUIDED_DECODING_H
#define LITETORCH_GUIDED_DECODING_H

#include "litetorch/tensor.h"
#include <string>
#include <vector>
#include <memory>

namespace litetorch {
namespace nn {

class GuidedDecoder {
public:
    std::vector<std::string> vocab;

    GuidedDecoder(const std::vector<std::string>& vocabulary);

    void apply_mask(std::shared_ptr<Tensor> logits, const std::string& prefix, const std::string& target_pattern);
};

}
}

#endif
