#include "bindings_common.h"

void bind_amp(py::module_& m) {
        auto amp_mod = m.def_submodule("amp", "LiteTorch Automatic Mixed Precision");

        py::class_<amp::GradScaler, optim::GradScaler>(amp_mod, "GradScaler")
            .def(py::init<float, float, float, int, bool>(), py::arg("init_scale") = 65536.0f, py::arg("growth_factor") = 2.0f, py::arg("backoff_factor") = 0.5f, py::arg("growth_interval") = 2000, py::arg("enabled") = true)
            .def("scale", &amp::GradScaler::scale, py::arg("loss"))
            .def("unscale_", &amp::GradScaler::unscale_, py::arg("optimizer"))
            .def("step", &amp::GradScaler::step, py::arg("optimizer"))
            .def("update", &amp::GradScaler::update)
            .def("get_scale", &amp::GradScaler::get_scale)
            .def("set_scale", &amp::GradScaler::set_scale, py::arg("scale"))
            .def("is_enabled", &amp::GradScaler::is_enabled);

        py::class_<amp::AutocastGuard>(amp_mod, "AutocastGuard")
            .def(py::init<bool, DataType>(), py::arg("enabled") = true, py::arg("dtype") = DataType::FP16)
            .def_static("is_enabled", &amp::AutocastGuard::is_enabled)
            .def_static("get_dtype", &amp::AutocastGuard::get_dtype);

        py::class_<PyAutocast>(amp_mod, "autocast")
            .def(py::init([](const std::string& device_type, py::object dtype_obj, bool enabled) {
                PyAutocast s;
                s.device_type = device_type;
                s.dtype = parse_autocast_dtype(dtype_obj, device_type);
                s.enabled = enabled;
                return s;
            }), py::arg("device_type") = "cuda", py::arg("dtype") = py::none(), py::arg("enabled") = true)
            .def("__enter__", [](PyAutocast& self) -> PyAutocast& { self.enter(); return self; })
            .def("__exit__", [](PyAutocast& self, py::object, py::object, py::object) { self.exit(); })
            .def_property_readonly("device_type", [](const PyAutocast& self) { return self.device_type; })
            .def_property_readonly("dtype", [](const PyAutocast& self) { return self.dtype; })
            .def_property_readonly("enabled", [](const PyAutocast& self) { return self.enabled; });

        amp_mod.def("is_autocast_enabled", &amp::AutocastGuard::is_enabled);
        amp_mod.def("get_autocast_dtype", &amp::AutocastGuard::get_dtype);
}
