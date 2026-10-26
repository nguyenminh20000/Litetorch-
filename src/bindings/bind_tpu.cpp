#include "bindings_common.h"

void bind_tpu(py::module_& m) {
        auto tpu_mod = m.def_submodule("tpu");
        tpu_mod.def("is_available", &litetorch::tpu::is_available);
        tpu_mod.def("device_count", &litetorch::tpu::device_count);
        tpu_mod.def("current_device", &litetorch::tpu::current_device);
        tpu_mod.def("set_device", &litetorch::tpu::set_device, py::arg("device_id") = 0);
        tpu_mod.def("synchronize", &litetorch::tpu::synchronize);
        tpu_mod.def("get_device_name", &litetorch::tpu::get_device_name, py::arg("device_id") = 0);

        py::enum_<DataType>(m, "DataType")
            .value("FP64", DataType::FP64)
            .value("FP32", DataType::FP32)
            .value("FP16", DataType::FP16)
            .value("BF16", DataType::BF16)
            .value("INT16", DataType::INT16)
            .value("INT8", DataType::INT8)
            .value("INT4", DataType::INT4)
            .value("FP8_E4M3", DataType::FP8_E4M3)
            .value("FP8_E5M2", DataType::FP8_E5M2)
            .value("NF4", DataType::NF4)
            .value("FP4_E2M1", DataType::FP4_E2M1)
            .value("FP4", DataType::FP4_E2M1)
            .value("INT32", DataType::INT32)
            .value("INT64", DataType::INT64)
            .value("UINT8", DataType::UINT8)
            .value("BOOL", DataType::BOOL)
            .export_values();

        py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
            .def_static("create", &Tensor::create, py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32)
            .def_static("from_vector", &Tensor::from_vector, py::arg("data"), py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32)
            .def_static("zeros", static_cast<std::shared_ptr<Tensor>(*)(const std::vector<int64_t>&, const Device&, bool)>(&Tensor::zeros), py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false)
            .def_static("zeros", static_cast<std::shared_ptr<Tensor>(*)(const std::vector<int64_t>&, const Device&, bool, DataType)>(&Tensor::zeros), py::arg("shape"), py::arg("device"), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32)
            .def_static("ones", &Tensor::ones, py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32)
            .def_static("randn", &Tensor::randn, py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32)
            .def_static("arange", &Tensor::arange, py::arg("start"), py::arg("end"), py::arg("step") = 1, py::arg("device") = Device(DeviceType::CPU, 0), py::arg("dtype") = DataType::FP32)
            .def("masked_fill", &Tensor::masked_fill, py::arg("mask"), py::arg("value"))
            .def("index_select", &Tensor::index_select, py::arg("dim"), py::arg("indices"))
            .def("slice", &Tensor::slice, py::arg("dim"), py::arg("start"), py::arg("end"), py::arg("step") = 1)
            .def("__getitem__", [](Tensor& self, py::object key) -> std::shared_ptr<Tensor> {
                if (py::isinstance<py::int_>(key)) {
                    int64_t idx = key.cast<int64_t>();
                    auto s = self.slice(0, idx, idx + 1);
                    return Ops::squeeze(s, 0);
                } else if (py::isinstance<py::slice>(key)) {
                    py::slice sl = key.cast<py::slice>();
                    int64_t dim_size = self.shape[0];
                    int64_t start, stop, step, slicelength;
                    if (!sl.compute(dim_size, &start, &stop, &step, &slicelength)) {
                        throw py::index_error("invalid slice");
                    }
                    return self.slice(0, start, stop, step);
                }
                throw py::type_error("unsupported index type");
            })
            .def_readwrite("shape", &Tensor::shape)
            .def_readwrite("strides", &Tensor::strides)
            .def_readwrite("dtype", &Tensor::dtype)
            .def_readwrite("device", &Tensor::device)
            .def_readwrite("requires_grad", &Tensor::requires_grad)
            .def_readwrite("grad", &Tensor::grad)
            .def_readwrite("shard", &Tensor::shard)
            .def("numel", &Tensor::numel)
            .def("item", &Tensor::item)
            .def("to_vector", &Tensor::to_vector)
            .def("backward", [](Tensor& self, std::shared_ptr<Tensor> grad, bool create_graph) {
                self.backward(grad, create_graph);
            }, py::arg("gradient") = nullptr, py::arg("create_graph") = false)
            .def("zero_grad", &Tensor::zero_grad)
            .def("retain_grad", [](std::shared_ptr<Tensor> self) { Autograd::retain_grad(self); })
            .def("to", py::overload_cast<const Device&>(&Tensor::to), py::arg("device"))
            .def("to", py::overload_cast<DataType>(&Tensor::to), py::arg("dtype"))
            .def("to", py::overload_cast<const Device&, DataType>(&Tensor::to), py::arg("device"), py::arg("dtype"))
            .def("view", [](Tensor& self, py::args args) {
                std::vector<int64_t> shape;
                if (args.size() == 1) {
                    py::handle h = args[0];
                    if (py::isinstance<py::list>(h) || py::isinstance<py::tuple>(h)) {
                        for (auto o : h.cast<py::sequence>()) shape.push_back(o.cast<int64_t>());
                        return self.view(shape);
                    }
                }
                for (auto o : args) shape.push_back(o.cast<int64_t>());
                return self.view(shape);
            })
            .def("reshape", &Tensor::reshape)
            .def("as_strided", &Tensor::as_strided, py::arg("shape"), py::arg("strides"))
            .def("permute", &Tensor::permute, py::arg("dims"))
            .def("detach", &Tensor::detach)
            .def("expand", &Tensor::expand, py::arg("shape"))
            .def("gather", &Tensor::gather, py::arg("dim"), py::arg("index"))
            .def("argmax", &Tensor::argmax, py::arg("dim") = -1, py::arg("keepdim") = false)
            .def("argmin", &Tensor::argmin, py::arg("dim") = -1, py::arg("keepdim") = false)
            .def("topk", &Tensor::topk, py::arg("k"), py::arg("dim") = -1, py::arg("largest") = true, py::arg("sorted") = true)
            .def("eq", &Tensor::eq, py::arg("other"))
            .def("ne", &Tensor::ne, py::arg("other"))
            .def("lt", &Tensor::lt, py::arg("other"))
            .def("gt", &Tensor::gt, py::arg("other"))
            .def("le", &Tensor::le, py::arg("other"))
            .def("ge", &Tensor::ge, py::arg("other"))
            .def("log_softmax", &Tensor::log_softmax, py::arg("dim") = -1)
            .def("silu", &Tensor::silu)
            .def("sum", [](Tensor& self, py::object dim, bool keepdim) -> std::shared_ptr<Tensor> {
                auto s = self.shared_from_this();
                if (dim.is_none()) return Ops::sum(s);
                return s->sum(dim.cast<int64_t>(), keepdim);
            }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
            .def("mean", [](Tensor& self, py::object dim, bool keepdim) -> std::shared_ptr<Tensor> {
                auto s = self.shared_from_this();
                if (dim.is_none()) return Ops::mean(s);
                return s->mean(dim.cast<int64_t>(), keepdim);
            }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
            .def("max", [](Tensor& self, py::object dim, bool keepdim) -> py::object {
                auto s = self.shared_from_this();
                if (dim.is_none()) return py::cast(Ops::max(s));
                return py::cast(s->max(dim.cast<int64_t>(), keepdim));
            }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
            .def("__add__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return Ops::add(a, b); })
            .def("__add__", [](std::shared_ptr<Tensor> a, double b) { return Ops::add(a, Tensor::from_vector({(float)b}, {1}, a->device)); })
            .def("__radd__", [](std::shared_ptr<Tensor> a, double b) { return Ops::add(a, Tensor::from_vector({(float)b}, {1}, a->device)); })
            .def("__sub__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return Ops::sub(a, b); })
            .def("__sub__", [](std::shared_ptr<Tensor> a, double b) { return Ops::sub(a, Tensor::from_vector({(float)b}, {1}, a->device)); })
            .def("__rsub__", [](std::shared_ptr<Tensor> a, double b) { return Ops::sub(Tensor::from_vector({(float)b}, {1}, a->device), a); })
            .def("__mul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return Ops::mul(a, b); })
            .def("__mul__", [](std::shared_ptr<Tensor> a, double b) { return Ops::mul(a, Tensor::from_vector({(float)b}, {1}, a->device)); })
            .def("__rmul__", [](std::shared_ptr<Tensor> a, double b) { return Ops::mul(a, Tensor::from_vector({(float)b}, {1}, a->device)); })
            .def("__truediv__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return Ops::div(a, b); })
            .def("__truediv__", [](std::shared_ptr<Tensor> a, double b) { return Ops::div(a, Tensor::from_vector({(float)b}, {1}, a->device)); })
            .def("__rtruediv__", [](std::shared_ptr<Tensor> a, double b) { return Ops::div(Tensor::from_vector({(float)b}, {1}, a->device), a); })
            .def("__neg__", [](std::shared_ptr<Tensor> a) { return Ops::neg(a); })
            .def("__matmul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return Ops::matmul(a, b); })
            .def("__pow__", [](std::shared_ptr<Tensor> a, double b) { return Ops::pow(a, (float)b); })
            .def("transpose", &Tensor::transpose)
            .def("contiguous", &Tensor::contiguous)
            .def("is_contiguous", &Tensor::is_contiguous)
            .def("clone", &Tensor::clone)
            .def("to_device_async", &Tensor::to_device_async, py::arg("target_device"))
            .def("cast", &Tensor::cast, py::arg("target_dtype"))
            // === W1-T2 tensor-api ===
            .def("cuda", [](Tensor& self, py::object device) {
                int idx = -1;
                if (!device.is_none()) {
                    idx = device.cast<int>();
                }
                return self.cuda(idx);
            }, py::arg("device") = py::none())
            .def("cpu", &Tensor::cpu)
            .def("dim", &Tensor::dim)
            .def_property_readonly("ndim", &Tensor::dim)
            .def("half", &Tensor::half)
            .def("float", &Tensor::to_float)
            .def("double", &Tensor::to_double)
            .def("long", &Tensor::to_long)
            .def("int", &Tensor::to_int)
            .def("type_as", &Tensor::type_as, py::arg("other"))
            .def("numpy", [](Tensor& self) -> py::object {
                if (self.device.type != DeviceType::CPU) {
                    throw std::runtime_error("[litetorch Error] numpy(): cannot convert non-CPU tensor to numpy; call .cpu() first");
                }
                size_t esz = self.storage->element_size();
                if (self.dtype == DataType::FP32) {
                    std::vector<ssize_t> shape(self.shape.begin(), self.shape.end());
                    std::vector<ssize_t> strides;
                    for (auto s : self.strides) {
                        strides.push_back(static_cast<ssize_t>(s * esz));
                    }
                    float* ptr = reinterpret_cast<float*>(
                        reinterpret_cast<char*>(self.storage->get_cpu_ptr()) + self.offset * esz);
                    auto* holder = new std::shared_ptr<StorageImpl>(self.storage);
                    py::capsule base(holder, [](void* p) {
                        delete static_cast<std::shared_ptr<StorageImpl>*>(p);
                    });
                    return py::array_t<float>(shape, strides, ptr, base);
                }
                std::shared_ptr<Tensor> t = self.shared_from_this();
                if (t->dtype == DataType::BF16) {
                    t = t->cast(DataType::FP32);
                }
                if (!t->is_contiguous()) {
                    t = t->contiguous();
                }
                std::vector<ssize_t> shape(t->shape.begin(), t->shape.end());
                size_t n = t->numel();
                size_t tsz = t->storage->element_size();
                const char* src = reinterpret_cast<const char*>(t->storage->get_cpu_ptr()) + t->offset * tsz;
                const char* fmt = nullptr;
                switch (t->dtype) {
                    case DataType::FP64: fmt = "float64"; break;
                    case DataType::FP32: fmt = "float32"; break;
                    case DataType::FP16: fmt = "float16"; break;
                    case DataType::INT32: fmt = "int32"; break;
                    case DataType::INT16: fmt = "int16"; break;
                    case DataType::INT8: fmt = "int8"; break;
                    default: throw std::runtime_error("[litetorch Error] numpy(): unsupported dtype for copy conversion");
                }
                py::array arr(py::dtype(fmt), shape);
                std::memcpy(arr.mutable_data(), src, n * tsz);
                return arr;
            })
            .def("tolist", [](Tensor& self) -> py::object {
                std::shared_ptr<Tensor> t = self.shared_from_this();
                if (t->device.type != DeviceType::CPU) {
                    t = t->to(Device(DeviceType::CPU, 0));
                }
                if (!t->is_contiguous()) {
                    t = t->contiguous();
                }
                bool as_int = (t->dtype == DataType::INT32 || t->dtype == DataType::INT16 ||
                               t->dtype == DataType::INT8);
                std::vector<double> vals = t->flat_doubles();
                std::function<py::object(size_t, size_t)> build = [&](size_t dim, size_t base) -> py::object {
                    if (dim == t->shape.size()) {
                        if (as_int) {
                            return py::int_(static_cast<int64_t>(vals[base]));
                        }
                        return py::float_(vals[base]);
                    }
                    py::list lst;
                    size_t stride = 1;
                    for (size_t d = dim + 1; d < t->shape.size(); ++d) {
                        stride *= static_cast<size_t>(t->shape[d]);
                    }
                    for (int64_t i = 0; i < t->shape[dim]; ++i) {
                        lst.append(build(dim + 1, base + static_cast<size_t>(i) * stride));
                    }
                    return lst;
                };
                return build(0, 0);
            })
            // === W1-T2 tensor-api ===
            .def("copy_", &Tensor::copy_)
            .def("add_", &Tensor::add_)
            .def("register_hook", &Tensor::register_hook)
            .def("__repr__", [](const Tensor& self) {
                return "<litetorch.Tensor shape=" + std::to_string(self.numel()) + " device=" + self.device.to_string() + ">";
            });
}
