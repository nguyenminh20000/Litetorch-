#include "bindings_common.h"

void bind_core_ops(py::module_& m) {
        // === W1-T1 core-ops ===
        auto w1_opt_device = [](py::object device) -> Device {
            return device.is_none() ? Device(DeviceType::CPU, 0) : device.cast<Device>();
        };
        auto w1_opt_dtype = [](py::object dtype, DataType dflt) -> DataType {
            return dtype.is_none() ? dflt : dtype.cast<DataType>();
        };
        auto w1_shape_arg = [](py::object shape) {
            std::vector<int64_t> s;
            if (py::isinstance<py::int_>(shape)) {
                s.push_back(shape.cast<int64_t>());
            } else {
                for (auto item : shape.cast<py::sequence>()) s.push_back(item.cast<int64_t>());
            }
            return s;
        };
        m.def("tensor", [w1_opt_device, w1_opt_dtype](py::object data, py::object device, bool requires_grad, py::object dtype) {
            std::vector<float> flat;
            std::vector<int64_t> shape;
            std::function<void(py::handle, size_t)> rec = [&](py::handle obj, size_t depth) {
                if (py::isinstance<py::sequence>(obj) && !py::isinstance<py::str>(obj)) {
                    auto seq = py::reinterpret_borrow<py::sequence>(obj);
                    size_t n = seq.size();
                    if (depth == shape.size()) shape.push_back((int64_t)n);
                    else if (shape[depth] != (int64_t)n)
                        throw std::runtime_error("[litetorch Error] tensor(): inconsistent nested sequence sizes");
                    for (auto item : seq) rec(item, depth + 1);
                } else {
                    double v = py::isinstance<py::bool_>(obj) ? (obj.cast<bool>() ? 1.0 : 0.0) : obj.cast<double>();
                    flat.push_back((float)v);
                }
            };
            rec(data, 0);
            if (shape.empty()) shape.push_back(1);
            return Ops::from_data(flat, shape, w1_opt_device(device), requires_grad, w1_opt_dtype(dtype, DataType::FP32));
        }, py::arg("data"), py::arg("device") = py::none(), py::arg("requires_grad") = false, py::arg("dtype") = py::none());
        m.def("empty", [w1_opt_device, w1_opt_dtype, w1_shape_arg](py::object shape, py::object device, bool requires_grad, py::object dtype) {
            return Ops::empty(w1_shape_arg(shape), w1_opt_device(device), requires_grad, w1_opt_dtype(dtype, DataType::FP32));
        }, py::arg("shape"), py::arg("device") = py::none(), py::arg("requires_grad") = false, py::arg("dtype") = py::none());
        m.def("full", [w1_opt_device, w1_opt_dtype, w1_shape_arg](py::object shape, double fill_value, py::object device, bool requires_grad, py::object dtype) {
            return Ops::full(w1_shape_arg(shape), (float)fill_value, w1_opt_device(device), requires_grad, w1_opt_dtype(dtype, DataType::FP32));
        }, py::arg("shape"), py::arg("fill_value"), py::arg("device") = py::none(), py::arg("requires_grad") = false, py::arg("dtype") = py::none());
        m.def("linspace", [w1_opt_device, w1_opt_dtype](double start, double end, int64_t steps, py::object device, py::object dtype) {
            return Ops::linspace((float)start, (float)end, steps, w1_opt_device(device), w1_opt_dtype(dtype, DataType::FP32));
        }, py::arg("start"), py::arg("end"), py::arg("steps"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
        m.def("logspace", [w1_opt_device, w1_opt_dtype](double start, double end, int64_t steps, double base, py::object device, py::object dtype) {
            return Ops::logspace((float)start, (float)end, steps, (float)base, w1_opt_device(device), w1_opt_dtype(dtype, DataType::FP32));
        }, py::arg("start"), py::arg("end"), py::arg("steps"), py::arg("base") = 10.0, py::arg("device") = py::none(), py::arg("dtype") = py::none());
        m.def("eye", [w1_opt_device, w1_opt_dtype](int64_t n, py::object m_arg, py::object device, py::object dtype) {
            int64_t m = m_arg.is_none() ? -1 : m_arg.cast<int64_t>();
            return Ops::eye(n, m, w1_opt_device(device), w1_opt_dtype(dtype, DataType::FP32));
        }, py::arg("n"), py::arg("m") = py::none(), py::arg("device") = py::none(), py::arg("dtype") = py::none());
        m.def("rand", [w1_opt_device, w1_opt_dtype, w1_shape_arg](py::object shape, py::object device, bool requires_grad, py::object dtype) {
            return Ops::rand(w1_shape_arg(shape), w1_opt_device(device), requires_grad, w1_opt_dtype(dtype, DataType::FP32));
        }, py::arg("shape"), py::arg("device") = py::none(), py::arg("requires_grad") = false, py::arg("dtype") = py::none());
        m.def("randint", [w1_opt_device, w1_opt_dtype, w1_shape_arg](int64_t low, int64_t high, py::object shape, py::object device, py::object dtype) {
            return Ops::randint(low, high, w1_shape_arg(shape), w1_opt_device(device), w1_opt_dtype(dtype, DataType::INT32));
        }, py::arg("low"), py::arg("high"), py::arg("shape"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
        m.def("randint", [w1_opt_device, w1_opt_dtype, w1_shape_arg](int64_t high, py::object shape, py::object device, py::object dtype) {
            return Ops::randint(0, high, w1_shape_arg(shape), w1_opt_device(device), w1_opt_dtype(dtype, DataType::INT32));
        }, py::arg("high"), py::arg("shape"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
        m.def("randperm", [w1_opt_device, w1_opt_dtype](int64_t n, py::object device, py::object dtype) {
            return Ops::randperm(n, w1_opt_device(device), w1_opt_dtype(dtype, DataType::INT32));
        }, py::arg("n"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
        m.def("zeros_like", [w1_opt_dtype](std::shared_ptr<Tensor> a, py::object device, py::object dtype) {
            Device d = device.is_none() ? a->device : device.cast<Device>();
            return Ops::zeros_like(a, d, w1_opt_dtype(dtype, a->dtype));
        }, py::arg("a"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
        m.def("ones_like", [w1_opt_dtype](std::shared_ptr<Tensor> a, py::object device, py::object dtype) {
            Device d = device.is_none() ? a->device : device.cast<Device>();
            return Ops::ones_like(a, d, w1_opt_dtype(dtype, a->dtype));
        }, py::arg("a"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
        m.def("empty_like", [w1_opt_dtype](std::shared_ptr<Tensor> a, py::object device, py::object dtype) {
            Device d = device.is_none() ? a->device : device.cast<Device>();
            return Ops::empty_like(a, d, w1_opt_dtype(dtype, a->dtype));
        }, py::arg("a"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
        m.def("full_like", [w1_opt_dtype](std::shared_ptr<Tensor> a, double fill_value, py::object device, py::object dtype) {
            Device d = device.is_none() ? a->device : device.cast<Device>();
            return Ops::full_like(a, (float)fill_value, d, w1_opt_dtype(dtype, a->dtype));
        }, py::arg("a"), py::arg("fill_value"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
        m.def("rand_like", [w1_opt_dtype](std::shared_ptr<Tensor> a, py::object device, py::object dtype) {
            Device d = device.is_none() ? a->device : device.cast<Device>();
            return Ops::rand_like(a, d, w1_opt_dtype(dtype, a->dtype));
        }, py::arg("a"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
        m.def("randn_like", [w1_opt_dtype](std::shared_ptr<Tensor> a, py::object device, py::object dtype) {
            Device d = device.is_none() ? a->device : device.cast<Device>();
            return Ops::randn_like(a, d, w1_opt_dtype(dtype, a->dtype));
        }, py::arg("a"), py::arg("device") = py::none(), py::arg("dtype") = py::none());
        // === end W1-T1 core-ops ===

        m.def("empty_cache", []() {
            CachingAllocator::get().empty_cache();
            auto backend = litetorch::BackendDispatcher::get().get_backend();
            if (backend && backend->is_available()) {
                backend->empty_cache();
            }
        });

        m.def("set_max_cpu_cache_size", [](size_t bytes) {
            CachingAllocator::get().set_max_cpu_cache_size(bytes);
        }, py::arg("bytes"));

        m.def("get_cached_cpu_bytes", []() -> size_t {
            return CachingAllocator::get().get_cached_cpu_bytes();
        });

        m.def("get_cached_gpu_bytes", []() -> size_t {
            return CachingAllocator::get().get_cached_gpu_bytes();
        });

        py::class_<NoGradContext>(m, "no_grad")
            .def(py::init<>())
            .def("__enter__", [](NoGradContext& self) -> NoGradContext& { self.enter(); return self; })
            .def("__exit__", [](NoGradContext& self, py::object, py::object, py::object) { self.exit(); });

        py::class_<EnableGradContext>(m, "enable_grad")
            .def(py::init<>())
            .def("__enter__", [](EnableGradContext& self) -> EnableGradContext& { self.enter(); return self; })
            .def("__exit__", [](EnableGradContext& self, py::object, py::object, py::object) { self.exit(); });

        py::class_<InferenceModeContext>(m, "inference_mode")
            .def(py::init<>())
            .def("__enter__", [](InferenceModeContext& self) -> InferenceModeContext& { self.enter(); return self; })
            .def("__exit__", [](InferenceModeContext& self, py::object, py::object, py::object) { self.exit(); });

        m.def("is_inference_mode", []() -> bool {
            return inference_mode_flag();
        });

        m.def("is_grad_enabled", []() -> bool {
            return Autograd::is_grad_enabled();
        });
        m.def("set_grad_enabled", [](bool enabled) {
            Autograd::set_grad_enabled(enabled);
        }, py::arg("enabled"));
}
