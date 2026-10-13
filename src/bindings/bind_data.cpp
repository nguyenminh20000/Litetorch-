#include "bindings_common.h"

void bind_data(py::module_& m) {
        auto data_mod = m.def_submodule("data", "LiteTorch Data Loading and Datasets");

        py::class_<data::Dataset, PyDataset, std::shared_ptr<data::Dataset>>(data_mod, "Dataset")
            .def(py::init<>())
            .def("size", &data::Dataset::size)
            .def("get", &data::Dataset::get, py::arg("index"))
            .def("__len__", &data::Dataset::size)
            .def("__getitem__", &data::Dataset::get, py::arg("index"));

        py::class_<data::TensorDataset, data::Dataset, std::shared_ptr<data::TensorDataset>>(data_mod, "TensorDataset")
            .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("x"), py::arg("y"))
            .def("size", &data::TensorDataset::size)
            .def("get", &data::TensorDataset::get, py::arg("index"))
            .def_readwrite("x", &data::TensorDataset::x)
            .def_readwrite("y", &data::TensorDataset::y);

        py::class_<data::Sampler, std::shared_ptr<data::Sampler>>(data_mod, "Sampler")
            .def("indices", &data::Sampler::indices)
            .def("__len__", &data::Sampler::size);

        py::class_<data::SequentialSampler, data::Sampler, std::shared_ptr<data::SequentialSampler>>(data_mod, "SequentialSampler")
            .def(py::init<size_t>(), py::arg("n"))
            .def("indices", &data::SequentialSampler::indices)
            .def("__len__", &data::SequentialSampler::size);

        py::class_<data::RandomSampler, data::Sampler, std::shared_ptr<data::RandomSampler>>(data_mod, "RandomSampler")
            .def(py::init<size_t, bool, size_t, uint64_t>(),
                 py::arg("n"), py::arg("replacement") = false, py::arg("num_samples") = 0, py::arg("seed") = 0)
            .def("indices", &data::RandomSampler::indices)
            .def("__len__", &data::RandomSampler::size);

        py::class_<data::BatchSampler, data::Sampler, std::shared_ptr<data::BatchSampler>>(data_mod, "BatchSampler")
            .def(py::init<std::shared_ptr<data::Sampler>, size_t, bool>(),
                 py::arg("sampler"), py::arg("batch_size"), py::arg("drop_last") = false)
            .def("batches", &data::BatchSampler::batches)
            .def("__len__", &data::BatchSampler::size)
            .def_property_readonly("batch_size", &data::BatchSampler::batch_size)
            .def_property_readonly("drop_last", &data::BatchSampler::drop_last);

        py::class_<data::Subset, data::Dataset, std::shared_ptr<data::Subset>>(data_mod, "Subset")
            .def(py::init<std::shared_ptr<data::Dataset>, std::vector<size_t>>(),
                 py::arg("dataset"), py::arg("indices"))
            .def("size", &data::Subset::size)
            .def("get", &data::Subset::get, py::arg("index"))
            .def("__len__", &data::Subset::size)
            .def("__getitem__", &data::Subset::get, py::arg("index"));

        data_mod.def("random_split", &data::random_split,
                     py::arg("dataset"), py::arg("lengths"), py::arg("seed") = 0);

        py::class_<data::DataLoader, std::shared_ptr<data::DataLoader>>(data_mod, "DataLoader")
            .def(py::init([](std::shared_ptr<data::Dataset> dataset, size_t batch_size, bool shuffle,
                             py::object device_obj, size_t prefetch_limit, size_t num_workers,
                             bool drop_last, py::object sampler_obj, py::object collate_obj) {
                Device dev = device_obj.is_none() ? Device(DeviceType::CPU, 0) : device_obj.cast<Device>();
                std::shared_ptr<data::Sampler> sampler;
                if (!sampler_obj.is_none()) {
                    sampler = sampler_obj.cast<std::shared_ptr<data::Sampler>>();
                }
                auto dl = std::make_shared<data::DataLoader>(
                    dataset, batch_size, shuffle, dev, prefetch_limit, num_workers, drop_last, sampler);
                if (!collate_obj.is_none()) {
                    py::function f = collate_obj.cast<py::function>();
                    dl->set_collate_fn([f](const std::vector<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>& items) {
                        py::list lst;
                        for (auto& it : items) {
                            lst.append(py::make_tuple(it.first, it.second));
                        }
                        py::tuple res = f(lst).cast<py::tuple>();
                        return std::make_pair(res[0].cast<std::shared_ptr<Tensor>>(),
                                              res[1].cast<std::shared_ptr<Tensor>>());
                    }, true);
                }
                return dl;
            }), py::arg("dataset"), py::arg("batch_size"), py::arg("shuffle") = true,
                py::arg("device") = py::none(), py::arg("prefetch_limit") = 2,
                py::arg("num_workers") = 0, py::arg("drop_last") = false,
                py::arg("sampler") = py::none(), py::arg("collate_fn") = py::none())
            .def("reset", &data::DataLoader::reset)
            .def("next", [](data::DataLoader& self) -> py::object {
                std::shared_ptr<Tensor> bx;
                std::shared_ptr<Tensor> by;
                bool ok;
                { py::gil_scoped_release rel; ok = self.next(bx, by); }
                if (ok) {
                    return py::make_tuple(bx, by);
                }
                return py::none();
            })
            .def("__iter__", [](data::DataLoader& self) -> data::DataLoader& {
                self.reset();
                return self;
            })
            .def("__next__", [](data::DataLoader& self) -> py::object {
                std::shared_ptr<Tensor> bx;
                std::shared_ptr<Tensor> by;
                bool ok;
                { py::gil_scoped_release rel; ok = self.next(bx, by); }
                if (!ok) {
                    throw py::stop_iteration();
                }
                return py::make_tuple(bx, by);
            })
            .def("__len__", &data::DataLoader::num_batches)
            .def("num_batches", &data::DataLoader::num_batches)
            .def("save_state", &data::DataLoader::save_state, py::arg("filepath"))
            .def("load_state", &data::DataLoader::load_state, py::arg("filepath"));

        m.def("save_parameters", &litetorch::save_parameters, py::arg("params"), py::arg("filepath"));
        m.def("load_parameters", &litetorch::load_parameters, py::arg("params"), py::arg("filepath"));
        m.def("save_safetensors", [](py::dict tensors, const std::string& filepath) {
            std::map<std::string, std::shared_ptr<Tensor>> m;
            for (auto kv : tensors) {
                m[kv.first.cast<std::string>()] = kv.second.cast<std::shared_ptr<Tensor>>();
            }
            litetorch::save_safetensors(m, filepath);
        }, py::arg("tensors"), py::arg("filepath"));
        m.def("load_safetensors", [](const std::string& filepath) {
            auto m = litetorch::load_safetensors(filepath);
            py::dict d;
            for (auto& kv : m) d[py::str(kv.first)] = kv.second;
            return d;
        }, py::arg("filepath"));
        m.def("save_optimizer_state", &litetorch::save_optimizer_state, py::arg("optimizer"), py::arg("filepath"));
        m.def("load_optimizer_state", &litetorch::load_optimizer_state, py::arg("optimizer"), py::arg("filepath"));

        m.def("save", [](py::object obj, const std::string& filepath) {
            if (py::isinstance<nn::Module>(obj)) {
                auto mod = obj.cast<nn::Module*>();
                litetorch::save_parameters(mod->parameters(), filepath);
            } else if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
                auto params = obj.cast<std::vector<std::shared_ptr<Tensor>>>();
                litetorch::save_parameters(params, filepath);
            } else if (py::isinstance<optim::Optimizer>(obj)) {
                auto opt = obj.cast<optim::Optimizer*>();
                litetorch::save_optimizer_state(opt, filepath);
            } else {
                throw std::runtime_error("save expects a Module, list of Tensor parameters, or Optimizer");
            }
        }, py::arg("obj"), py::arg("filepath"));

        m.def("load", [](py::object obj, const std::string& filepath) {
            if (py::isinstance<nn::Module>(obj)) {
                auto mod = obj.cast<nn::Module*>();
                litetorch::load_parameters(mod->parameters(), filepath);
            } else if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
                auto params = obj.cast<std::vector<std::shared_ptr<Tensor>>>();
                litetorch::load_parameters(params, filepath);
            } else if (py::isinstance<optim::Optimizer>(obj)) {
                auto opt = obj.cast<optim::Optimizer*>();
                litetorch::load_optimizer_state(opt, filepath);
            } else {
                throw std::runtime_error("load expects a Module, list of Tensor parameters, or Optimizer");
            }
        }, py::arg("obj"), py::arg("filepath"));

        m.def("manual_seed", &litetorch::manual_seed, py::arg("seed"));
        m.def("get_rng_state", &litetorch::get_rng_state);
        m.def("set_rng_state", &litetorch::set_rng_state, py::arg("state"));

        m.def("randn", [](const std::vector<int64_t>& shape, const Device& device, bool requires_grad, DataType dtype) {
            return Tensor::randn(shape, device, requires_grad, dtype);
        }, py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32);
        m.def("ones", [](const std::vector<int64_t>& shape, const Device& device, bool requires_grad, DataType dtype) {
            return Tensor::ones(shape, device, requires_grad, dtype);
        }, py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32);
        m.def("zeros", [](const std::vector<int64_t>& shape, const Device& device, bool requires_grad, DataType dtype) {
            return Tensor::zeros(shape, device, requires_grad, dtype);
        }, py::arg("shape"), py::arg("device") = Device(DeviceType::CPU, 0), py::arg("requires_grad") = false, py::arg("dtype") = DataType::FP32);
        m.def("arange", [](int64_t start, int64_t end, int64_t step, const Device& device, DataType dtype) {
            return Tensor::arange(start, end, step, device, dtype);
        }, py::arg("start"), py::arg("end"), py::arg("step") = 1, py::arg("device") = Device(DeviceType::CPU, 0), py::arg("dtype") = DataType::FP32);
}
