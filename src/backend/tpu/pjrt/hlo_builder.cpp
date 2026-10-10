#include "hlo_builder.h"
#include "xla/service/hlo.pb.h"
#include "xla/xla_data.pb.h"

namespace litetorch {
namespace tpu_pjrt {

namespace {

xla::PrimitiveType to_primitive(HloElementType t) {
    switch (t) {
        case HloElementType::F32: return xla::F32;
        case HloElementType::BF16: return xla::BF16;
        case HloElementType::F16: return xla::F16;
        case HloElementType::S32: return xla::S32;
    }
    return xla::F32;
}

xla::ShapeProto shape_proto(HloElementType dtype, const std::vector<int64_t>& dims) {
    xla::ShapeProto s;
    s.set_element_type(to_primitive(dtype));
    for (int64_t d : dims) s.add_dimensions(d);
    return s;
}

xla::ShapeProto shape_proto(const HloShape& hs) {
    return shape_proto(hs.dtype, hs.dims);
}

struct ModuleBuilder {
    xla::HloModuleProto module;
    int64_t next_instr_id = 1;
    int64_t next_comp_id = 1;

    xla::HloComputationProto* new_computation(const std::string& name) {
        auto* c = module.add_computations();
        c->set_name(name);
        c->set_id(next_comp_id++);
        return c;
    }

    xla::HloInstructionProto* add_instr(xla::HloComputationProto* c,
                                       const std::string& name,
                                       const std::string& opcode,
                                       const xla::ShapeProto& shape) {
        auto* inst = c->add_instructions();
        inst->set_name(name);
        inst->set_opcode(opcode);
        *inst->mutable_shape() = shape;
        inst->set_id(next_instr_id++);
        return inst;
    }

    xla::HloInstructionProto* add_parameter(xla::HloComputationProto* c,
                                            int64_t param_no,
                                            const HloShape& hs) {
        auto* inst = add_instr(c, "param." + std::to_string(param_no), "parameter",
                               shape_proto(hs));
        inst->set_parameter_number(param_no);
        auto* ps = c->mutable_program_shape();
        while (ps->parameters_size() <= param_no) ps->add_parameters();
        *ps->mutable_parameters(param_no) = shape_proto(hs);
        return inst;
    }

    xla::HloInstructionProto* add_constant_scalar(xla::HloComputationProto* c,
                                                  const std::string& name,
                                                  HloElementType dtype,
                                                  float value) {
        xla::ShapeProto s = shape_proto(dtype, {});
        auto* inst = add_instr(c, name, "constant", s);
        auto* lit = inst->mutable_literal();
        *lit->mutable_shape() = s;
        if (dtype == HloElementType::BF16 || dtype == HloElementType::F16) {
            uint16_t bits = 0;
            if (dtype == HloElementType::BF16) {
                uint32_t f;
                __builtin_memcpy(&f, &value, 4);
                bits = static_cast<uint16_t>(f >> 16);
            } else {
                uint32_t f;
                __builtin_memcpy(&f, &value, 4);
                uint32_t sign = (f >> 16) & 0x8000;
                int32_t exp = ((f >> 23) & 0xff) - 112;
                uint32_t mant = (f >> 13) & 0x3ff;
                if (exp <= 0) { bits = static_cast<uint16_t>(sign); }
                else if (exp >= 31) { bits = static_cast<uint16_t>(sign | 0x7bff); }
                else { bits = static_cast<uint16_t>(sign | (exp << 10) | mant); }
            }
            std::string bytes;
            bytes.push_back(static_cast<char>(bits & 0xff));
            bytes.push_back(static_cast<char>((bits >> 8) & 0xff));
            if (dtype == HloElementType::BF16) lit->set_bf16s(bytes);
            else lit->set_f16s(bytes);
        } else if (dtype == HloElementType::S32) {
            lit->add_s32s(static_cast<int32_t>(value));
        } else {
            lit->add_f32s(value);
        }
        return inst;
    }

    void finish_computation(xla::HloComputationProto* c,
                            xla::HloInstructionProto* root) {
        c->set_root_id(root->id());
        auto* ps = c->mutable_program_shape();
        *ps->mutable_result() = root->shape();
    }

    std::string serialize(const std::string& name,
                          xla::HloComputationProto* entry) {
        module.set_name(name);
        module.set_entry_computation_name(entry->name());
        module.set_entry_computation_id(entry->id());
        *module.mutable_host_program_shape() = entry->program_shape();
        std::string out;
        module.SerializeToString(&out);
        return out;
    }
};

int64_t num_elements(const std::vector<int64_t>& dims) {
    int64_t n = 1;
    for (int64_t d : dims) n *= d;
    return n;
}

}

std::string HloBuilder::make_dot(const HloShape& a, const HloShape& b, const HloShape& out) {
    ModuleBuilder mb;
    auto* c = mb.new_computation("dot_computation");
    auto* pa = mb.add_parameter(c, 0, a);
    auto* pb = mb.add_parameter(c, 1, b);
    auto* dot = mb.add_instr(c, "dot", "dot", shape_proto(out));
    dot->add_operand_ids(pa->id());
    dot->add_operand_ids(pb->id());
    auto* dnums = dot->mutable_dot_dimension_numbers();
    int64_t a_rank = static_cast<int64_t>(a.dims.size());
    int64_t b_rank = static_cast<int64_t>(b.dims.size());
    for (int64_t i = 0; i < a_rank - 2; ++i) {
        dnums->add_lhs_batch_dimensions(i);
        dnums->add_rhs_batch_dimensions(i);
    }
    dnums->add_lhs_contracting_dimensions(a_rank - 1);
    dnums->add_rhs_contracting_dimensions(b_rank - 2);
    mb.finish_computation(c, dot);
    return mb.serialize("dot_module", c);
}

std::string HloBuilder::make_add(const HloShape& a, const HloShape& b, const HloShape& out) {
    ModuleBuilder mb;
    auto* c = mb.new_computation("add_computation");
    auto* pa = mb.add_parameter(c, 0, a);
    auto* pb = mb.add_parameter(c, 1, b);
    auto* add = mb.add_instr(c, "add", "add", shape_proto(out));
    add->add_operand_ids(pa->id());
    add->add_operand_ids(pb->id());
    mb.finish_computation(c, add);
    return mb.serialize("add_module", c);
}

std::string HloBuilder::make_multiply(const HloShape& a, const HloShape& b, const HloShape& out) {
    ModuleBuilder mb;
    auto* c = mb.new_computation("multiply_computation");
    auto* pa = mb.add_parameter(c, 0, a);
    auto* pb = mb.add_parameter(c, 1, b);
    auto* mul = mb.add_instr(c, "multiply", "multiply", shape_proto(out));
    mul->add_operand_ids(pa->id());
    mul->add_operand_ids(pb->id());
    mb.finish_computation(c, mul);
    return mb.serialize("multiply_module", c);
}

std::string HloBuilder::make_relu(const HloShape& a, const HloShape& out) {
    ModuleBuilder mb;
    auto* c = mb.new_computation("relu_computation");
    auto* pa = mb.add_parameter(c, 0, a);
    auto* cz = mb.add_constant_scalar(c, "zero", a.dtype, 0.0f);
    auto* bc = mb.add_instr(c, "broadcast", "broadcast", shape_proto(out));
    bc->add_operand_ids(cz->id());
    for (int64_t i = 0; i < static_cast<int64_t>(out.dims.size()); ++i) {
        bc->add_dimensions(i);
    }
    auto* mx = mb.add_instr(c, "maximum", "maximum", shape_proto(out));
    mx->add_operand_ids(pa->id());
    mx->add_operand_ids(bc->id());
    mb.finish_computation(c, mx);
    return mb.serialize("relu_module", c);
}

std::string HloBuilder::make_reduce_sum(const HloShape& a, const HloShape& out,
                                        const std::vector<int64_t>& axes) {
    ModuleBuilder mb;
    auto* addc = mb.new_computation("add_scalars");
    xla::ShapeProto scalar = shape_proto(a.dtype, {});
    auto* sa = mb.add_instr(addc, "p0", "parameter", scalar);
    sa->set_parameter_number(0);
    auto* sb = mb.add_instr(addc, "p1", "parameter", scalar);
    sb->set_parameter_number(1);
    auto* sadd = mb.add_instr(addc, "add", "add", scalar);
    sadd->add_operand_ids(sa->id());
    sadd->add_operand_ids(sb->id());
    {
        auto* ps = addc->mutable_program_shape();
        *ps->add_parameters() = scalar;
        *ps->add_parameters() = scalar;
        *ps->mutable_result() = scalar;
    }
    addc->set_root_id(sadd->id());

    auto* c = mb.new_computation("reduce_computation");
    auto* pa = mb.add_parameter(c, 0, a);
    auto* cz = mb.add_constant_scalar(c, "init", a.dtype, 0.0f);
    auto* rd = mb.add_instr(c, "reduce", "reduce", shape_proto(out));
    rd->add_operand_ids(pa->id());
    rd->add_operand_ids(cz->id());
    for (int64_t ax : axes) rd->add_dimensions(ax);
    rd->add_called_computation_ids(addc->id());
    mb.finish_computation(c, rd);
    return mb.serialize("reduce_module", c);
}

std::string HloBuilder::make_conv2d(const HloShape& input, const HloShape& kernel,
                                    const HloShape& out,
                                    int64_t stride_h, int64_t stride_w,
                                    int64_t pad_h, int64_t pad_w) {
    ModuleBuilder mb;
    auto* c = mb.new_computation("conv_computation");
    auto* pi = mb.add_parameter(c, 0, input);
    auto* pk = mb.add_parameter(c, 1, kernel);
    auto* conv = mb.add_instr(c, "convolution", "convolution", shape_proto(out));
    conv->add_operand_ids(pi->id());
    conv->add_operand_ids(pk->id());
    auto* win = conv->mutable_window();
    auto* dh = win->add_dimensions();
    dh->set_size(kernel.dims[2]);
    dh->set_stride(stride_h);
    dh->set_padding_low(pad_h);
    dh->set_padding_high(pad_h);
    auto* dw = win->add_dimensions();
    dw->set_size(kernel.dims[3]);
    dw->set_stride(stride_w);
    dw->set_padding_low(pad_w);
    dw->set_padding_high(pad_w);
    auto* dnums = conv->mutable_convolution_dimension_numbers();
    dnums->set_input_batch_dimension(0);
    dnums->set_input_feature_dimension(1);
    dnums->add_input_spatial_dimensions(2);
    dnums->add_input_spatial_dimensions(3);
    dnums->set_kernel_input_feature_dimension(1);
    dnums->set_kernel_output_feature_dimension(0);
    dnums->add_kernel_spatial_dimensions(2);
    dnums->add_kernel_spatial_dimensions(3);
    dnums->set_output_batch_dimension(0);
    dnums->set_output_feature_dimension(1);
    dnums->add_output_spatial_dimensions(2);
    dnums->add_output_spatial_dimensions(3);
    mb.finish_computation(c, conv);
    return mb.serialize("conv_module", c);
}

std::string HloBuilder::make_reshape(const HloShape& a, const HloShape& out) {
    (void)a;
    ModuleBuilder mb;
    auto* c = mb.new_computation("reshape_computation");
    auto* pa = mb.add_parameter(c, 0, a);
    auto* rs = mb.add_instr(c, "reshape", "reshape", shape_proto(out));
    rs->add_operand_ids(pa->id());
    mb.finish_computation(c, rs);
    return mb.serialize("reshape_module", c);
}

std::string HloBuilder::make_transpose(const HloShape& a, const HloShape& out,
                                       const std::vector<int64_t>& perm) {
    (void)a;
    ModuleBuilder mb;
    auto* c = mb.new_computation("transpose_computation");
    auto* pa = mb.add_parameter(c, 0, a);
    auto* tr = mb.add_instr(c, "transpose", "transpose", shape_proto(out));
    tr->add_operand_ids(pa->id());
    for (int64_t p : perm) tr->add_dimensions(p);
    mb.finish_computation(c, tr);
    return mb.serialize("transpose_module", c);
}

std::string HloBuilder::make_broadcast_zeros(const HloShape& out) {
    ModuleBuilder mb;
    auto* c = mb.new_computation("zeros_computation");
    auto* cz = mb.add_constant_scalar(c, "zero", out.dtype, 0.0f);
    auto* bc = mb.add_instr(c, "broadcast", "broadcast", shape_proto(out));
    bc->add_operand_ids(cz->id());
    for (int64_t i = 0; i < static_cast<int64_t>(out.dims.size()); ++i) {
        bc->add_dimensions(i);
    }
    mb.finish_computation(c, bc);
    return mb.serialize("zeros_module", c);
}

}
}
