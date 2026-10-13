#include "bindings_common.h"

void bind_jit(py::module_& m) {
        auto jit_mod = m.def_submodule("jit", "LiteTorch JIT Compilation");

        py::enum_<JITVar::OpType>(jit_mod, "OpType")
            .value("INPUT", JITVar::OpType::INPUT)
            .value("CONST", JITVar::OpType::CONST)
            .value("ADD", JITVar::OpType::ADD)
            .value("SUB", JITVar::OpType::SUB)
            .value("MUL", JITVar::OpType::MUL)
            .value("DIV", JITVar::OpType::DIV)
            .value("RELU", JITVar::OpType::RELU)
            .value("GELU", JITVar::OpType::GELU)
            .value("SIGMOID", JITVar::OpType::SIGMOID)
            .value("TANH", JITVar::OpType::TANH)
            .value("SQRT", JITVar::OpType::SQRT)
            .value("EXP", JITVar::OpType::EXP)
            .value("LOG", JITVar::OpType::LOG)
            .value("ABS", JITVar::OpType::ABS)
            .value("NEG", JITVar::OpType::NEG)
            .export_values();

        py::class_<JITVar, std::shared_ptr<JITVar>>(jit_mod, "JITVar")
            .def(py::init<JITVar::OpType, const std::string&>(), py::arg("op"), py::arg("name") = "")
            .def(py::init<float>(), py::arg("val"))
            .def_readwrite("op", &JITVar::op)
            .def_readwrite("name", &JITVar::name)
            .def_readwrite("val", &JITVar::val)
            .def("__add__", [](std::shared_ptr<JITVar> a, std::shared_ptr<JITVar> b) { return a + b; })
            .def("__add__", [](std::shared_ptr<JITVar> a, float b) { return a + b; })
            .def("__radd__", [](std::shared_ptr<JITVar> a, float b) { return a + b; })
            .def("__sub__", [](std::shared_ptr<JITVar> a, std::shared_ptr<JITVar> b) { return a - b; })
            .def("__sub__", [](std::shared_ptr<JITVar> a, float b) { return a - b; })
            .def("__rsub__", [](std::shared_ptr<JITVar> a, float b) { return std::make_shared<JITVar>(b) - a; })
            .def("__mul__", [](std::shared_ptr<JITVar> a, std::shared_ptr<JITVar> b) { return a * b; })
            .def("__mul__", [](std::shared_ptr<JITVar> a, float b) { return a * b; })
            .def("__rmul__", [](std::shared_ptr<JITVar> a, float b) { return a * b; })
            .def("__truediv__", [](std::shared_ptr<JITVar> a, std::shared_ptr<JITVar> b) { return a / b; })
            .def("__truediv__", [](std::shared_ptr<JITVar> a, float b) { return a / b; })
            .def("__neg__", [](std::shared_ptr<JITVar> a) { return -a; });

        jit_mod.def("relu", &JIT::relu, py::arg("a"));
        jit_mod.def("gelu", &JIT::gelu, py::arg("a"));
        jit_mod.def("sigmoid", &JIT::sigmoid, py::arg("a"));
        jit_mod.def("tanh", &JIT::tanh, py::arg("a"));
        jit_mod.def("sqrt", &JIT::sqrt, py::arg("a"));
        jit_mod.def("exp", &JIT::exp, py::arg("a"));
        jit_mod.def("log", &JIT::log, py::arg("a"));
        jit_mod.def("abs", &JIT::abs, py::arg("a"));
        jit_mod.def("fused_bias_relu", [](std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> bias) {
            auto v_x = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
            auto v_b = std::make_shared<JITVar>(JITVar::OpType::INPUT, "b");
            auto expr = JIT::relu(v_x + v_b);
            JITFunction fn("fused_bias_relu", expr, {v_x, v_b});
            return fn({x, bias});
        }, py::arg("x"), py::arg("bias"));
        jit_mod.def("fused_bias_gelu", [](std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> bias) {
            auto v_x = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
            auto v_b = std::make_shared<JITVar>(JITVar::OpType::INPUT, "b");
            auto expr = JIT::gelu(v_x + v_b);
            JITFunction fn("fused_bias_gelu", expr, {v_x, v_b});
            return fn({x, bias});
        }, py::arg("x"), py::arg("bias"));
        jit_mod.def("fused_residual_add", [](std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> residual) {
            auto v_x = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
            auto v_r = std::make_shared<JITVar>(JITVar::OpType::INPUT, "r");
            auto expr = v_x + v_r;
            JITFunction fn("fused_residual_add", expr, {v_x, v_r});
            return fn({x, residual});
        }, py::arg("x"), py::arg("residual"));
        jit_mod.def("fused_residual_gelu", [](std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> residual) {
            auto v_x = std::make_shared<JITVar>(JITVar::OpType::INPUT, "x");
            auto v_r = std::make_shared<JITVar>(JITVar::OpType::INPUT, "r");
            auto expr = JIT::gelu(v_x + v_r);
            JITFunction fn("fused_residual_gelu", expr, {v_x, v_r});
            return fn({x, residual});
        }, py::arg("x"), py::arg("residual"));

        py::class_<JITFunction, std::shared_ptr<JITFunction>>(jit_mod, "JITFunction")
            .def(py::init<const std::string&, std::shared_ptr<JITVar>, const std::vector<std::shared_ptr<JITVar>>&>(),
                 py::arg("name"), py::arg("expr"), py::arg("inputs"))
            .def("__call__", &JITFunction::operator(), py::arg("args"))
            .def("forward", &JITFunction::operator(), py::arg("args"))
            .def("save", &JITFunction::save, py::arg("filepath"))
            .def_static("load", &JITFunction::load, py::arg("filepath"));

        py::class_<Tracer>(jit_mod, "Tracer")
            .def_static("trace", &Tracer::trace, py::arg("inputs"), py::arg("func"), py::arg("name") = "traced_fn")
            .def_static("derivative", &Tracer::derivative, py::arg("var"), py::arg("wrt"))
            .def_static("serialize_var", &Tracer::serialize_var, py::arg("var"))
            .def_static("deserialize_var", [](const std::string& str) {
                size_t pos = 0;
                return Tracer::deserialize_var(str, pos);
            }, py::arg("str"));
}
