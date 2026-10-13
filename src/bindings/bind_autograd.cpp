#include "bindings_common.h"

void bind_autograd(py::module_& m) {
        // === W2-T4 autograd ===
        auto autograd_mod = m.def_submodule("autograd", "LiteTorch Autograd");
        py::class_<FunctionContext, std::shared_ptr<FunctionContext>>(autograd_mod, "FunctionContext", py::dynamic_attr())
            .def(py::init<>())
            .def("save_for_backward", [](FunctionContext& self, py::args args) {
                std::vector<std::shared_ptr<Tensor>> ts;
                for (auto a : args) ts.push_back(a.cast<std::shared_ptr<Tensor>>());
                self.save_for_backward(ts);
            })
            .def_property_readonly("saved_tensors", [](FunctionContext& self) {
                const auto& st = self.get_saved_tensors();
                py::tuple t(st.size());
                for (size_t i = 0; i < st.size(); ++i) t[i] = py::cast(st[i]);
                return t;
            })
            .def_property_readonly("needs_input_grad", [](FunctionContext& self) {
                py::list l;
                for (bool b : self.needs_input_grad()) l.append(b);
                return l;
            });
        py::class_<Function, PyFunction, std::shared_ptr<Function>> pyFn(autograd_mod, "Function");
        pyFn.def(py::init<>());
        py::object apply_fn = py::cpp_function([](py::object cls, py::args args) -> py::object {
            py::object inst = cls();
            auto fn = inst.cast<std::shared_ptr<Function>>();
            auto pyfn = std::dynamic_pointer_cast<PyFunction>(fn);
            if (!pyfn)
                throw std::runtime_error("[litetorch Error] autograd.Function.apply: could not access Python subclass state");
            std::vector<std::shared_ptr<Tensor>> tensor_inputs;
            py::tuple tmpl(args.size());
            for (size_t i = 0; i < (size_t)args.size(); ++i) {
                bool is_tensor = false;
                try {
                    tensor_inputs.push_back(args[i].cast<std::shared_ptr<Tensor>>());
                    is_tensor = true;
                } catch (py::cast_error&) {}
                if (is_tensor) tmpl[i] = py::none();
                else tmpl[i] = args[i];
            }
            pyfn->arg_template = tmpl;
            pyfn->py_self = inst;
            std::vector<std::shared_ptr<Tensor>> outputs = Function::apply(fn, tensor_inputs);
            if (outputs.size() == 1) return py::cast(outputs[0]);
            py::tuple out(outputs.size());
            for (size_t i = 0; i < outputs.size(); ++i) out[i] = py::cast(outputs[i]);
            return out;
        }, py::arg("cls"));
        pyFn.attr("apply") = py::module::import("builtins").attr("classmethod")(apply_fn);
        autograd_mod.def("grad", [](py::object outputs, py::object inputs, py::object grad_outputs,
                bool retain_graph, bool create_graph, bool allow_unused) -> py::object {
            auto outs = w2t4_unpack(outputs, false);
            auto ins = w2t4_unpack(inputs, false);
            std::vector<std::shared_ptr<Tensor>> gos;
            if (!grad_outputs.is_none()) gos = w2t4_unpack(grad_outputs, false);
            std::vector<std::shared_ptr<Tensor>> res =
                Autograd::grad(outs, ins, gos, retain_graph, create_graph, allow_unused);
            py::tuple t(res.size());
            for (size_t i = 0; i < res.size(); ++i) {
                if (res[i]) t[i] = py::cast(res[i]);
                else t[i] = py::none();
            }
            return t;
        }, py::arg("outputs"), py::arg("inputs"), py::arg("grad_outputs") = py::none(),
           py::arg("retain_graph") = false, py::arg("create_graph") = false, py::arg("allow_unused") = false);
        autograd_mod.def("gradcheck", [](py::function func, py::object inputs, double eps, double atol,
                double rtol, bool raise_exception) -> bool {
            auto ins = w2t4_unpack(inputs, false);
            AutogradFunc f = [&](const std::vector<std::shared_ptr<Tensor>>& xs)
                -> std::vector<std::shared_ptr<Tensor>> {
                py::gil_scoped_acquire gil;
                py::object res = func(*py::cast(xs));
                return w2t4_unpack(res, false);
            };
            std::string err;
            return autograd_gradcheck(f, ins, eps, atol, rtol, raise_exception, err);
        }, py::arg("func"), py::arg("inputs"), py::arg("eps") = 1e-6, py::arg("atol") = 1e-5,
           py::arg("rtol") = 1e-3, py::arg("raise_exception") = true);
        autograd_mod.def("gradgradcheck", [](py::function func, py::object inputs, double eps, double atol,
                double rtol, bool raise_exception) -> bool {
            auto ins = w2t4_unpack(inputs, false);
            AutogradFunc f = [&](const std::vector<std::shared_ptr<Tensor>>& xs)
                -> std::vector<std::shared_ptr<Tensor>> {
                py::gil_scoped_acquire gil;
                py::object res = func(*py::cast(xs));
                return w2t4_unpack(res, false);
            };
            std::string err;
            return autograd_gradgradcheck(f, ins, eps, atol, rtol, raise_exception, err);
        }, py::arg("func"), py::arg("inputs"), py::arg("eps") = 1e-6, py::arg("atol") = 1e-5,
           py::arg("rtol") = 1e-3, py::arg("raise_exception") = true);
        autograd_mod.def("retain_grad", [](std::shared_ptr<Tensor> t) { Autograd::retain_grad(t); },
            py::arg("tensor"));
        // === end W2-T4 autograd ===
}
