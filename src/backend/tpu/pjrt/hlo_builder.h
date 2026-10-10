#ifndef LITETORCH_TPU_HLO_BUILDER_H
#define LITETORCH_TPU_HLO_BUILDER_H

#include <string>
#include <vector>
#include <cstdint>

namespace litetorch {
namespace tpu_pjrt {

enum class HloElementType {
    F32 = 0,
    BF16 = 1,
    F16 = 2,
    S32 = 3,
};

struct HloShape {
    HloElementType dtype;
    std::vector<int64_t> dims;
};

class HloBuilder {
public:
    static std::string make_dot(const HloShape& a, const HloShape& b, const HloShape& out);
    static std::string make_add(const HloShape& a, const HloShape& b, const HloShape& out);
    static std::string make_multiply(const HloShape& a, const HloShape& b, const HloShape& out);
    static std::string make_relu(const HloShape& a, const HloShape& out);
    static std::string make_reduce_sum(const HloShape& a, const HloShape& out, const std::vector<int64_t>& axes);
    static std::string make_conv2d(const HloShape& input, const HloShape& kernel, const HloShape& out,
                                   int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w);
    static std::string make_reshape(const HloShape& a, const HloShape& out);
    static std::string make_transpose(const HloShape& a, const HloShape& out, const std::vector<int64_t>& perm);
    static std::string make_broadcast_zeros(const HloShape& out);
};

}
}

#endif
