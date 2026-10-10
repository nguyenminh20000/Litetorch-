#include "hlo_builder.h"

namespace litetorch {
namespace tpu_pjrt {

namespace {

std::string build_hlo_module_stub(const std::string& /*op_name*/) {
    return "";
}

}

std::string HloBuilder::make_dot(const HloShape& a, const HloShape& b, const HloShape& out) {
    (void)a; (void)b; (void)out;
    return build_hlo_module_stub("dot");
}

std::string HloBuilder::make_add(const HloShape& a, const HloShape& b, const HloShape& out) {
    (void)a; (void)b; (void)out;
    return build_hlo_module_stub("add");
}

std::string HloBuilder::make_multiply(const HloShape& a, const HloShape& b, const HloShape& out) {
    (void)a; (void)b; (void)out;
    return build_hlo_module_stub("multiply");
}

std::string HloBuilder::make_relu(const HloShape& a, const HloShape& out) {
    (void)a; (void)out;
    return build_hlo_module_stub("relu");
}

std::string HloBuilder::make_reduce_sum(const HloShape& a, const HloShape& out, const std::vector<int64_t>& axes) {
    (void)a; (void)out; (void)axes;
    return build_hlo_module_stub("reduce_sum");
}

std::string HloBuilder::make_conv2d(const HloShape& input, const HloShape& kernel, const HloShape& out,
                                     int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w) {
    (void)input; (void)kernel; (void)out;
    (void)stride_h; (void)stride_w; (void)pad_h; (void)pad_w;
    return build_hlo_module_stub("conv2d");
}

std::string HloBuilder::make_reshape(const HloShape& a, const HloShape& out) {
    (void)a; (void)out;
    return build_hlo_module_stub("reshape");
}

std::string HloBuilder::make_transpose(const HloShape& a, const HloShape& out, const std::vector<int64_t>& perm) {
    (void)a; (void)out; (void)perm;
    return build_hlo_module_stub("transpose");
}

std::string HloBuilder::make_broadcast_zeros(const HloShape& out) {
    (void)out;
    return build_hlo_module_stub("zeros");
}

}
}
