#include "bindings_common.h"
#include "litetorch/onnx.h"
#include "litetorch/profiler.h"
#include "litetorch/hub.h"

struct PyProfileGuard {
    std::string name;
    std::unique_ptr<litetorch::profiler::ScopedTimer> timer;
    void enter() { timer = std::make_unique<litetorch::profiler::ScopedTimer>(name); }
    void exit() { timer.reset(); }
};

void bind_onnx(py::module_& m) {
    auto onnx_mod = m.def_submodule("onnx", "LiteTorch ONNX export (minimal)");
    py::class_<litetorch::onnx::ValueInfo>(onnx_mod, "ValueInfo")
        .def(py::init<>())
        .def_readwrite("name", &litetorch::onnx::ValueInfo::name)
        .def_readwrite("shape", &litetorch::onnx::ValueInfo::shape)
        .def_readwrite("dtype", &litetorch::onnx::ValueInfo::dtype);
    py::class_<litetorch::onnx::Node>(onnx_mod, "Node")
        .def(py::init<>())
        .def_readwrite("op_type", &litetorch::onnx::Node::op_type)
        .def_readwrite("name", &litetorch::onnx::Node::name)
        .def_readwrite("inputs", &litetorch::onnx::Node::inputs)
        .def_readwrite("outputs", &litetorch::onnx::Node::outputs)
        .def_readwrite("attrs", &litetorch::onnx::Node::attrs)
        .def_readwrite("domain", &litetorch::onnx::Node::domain);
    py::class_<litetorch::onnx::Graph>(onnx_mod, "Graph")
        .def(py::init<>())
        .def_readwrite("name", &litetorch::onnx::Graph::name)
        .def("add_input", &litetorch::onnx::Graph::add_input,
             py::arg("name"), py::arg("shape"), py::arg("dtype") = litetorch::DataType::FP32)
        .def("add_output", &litetorch::onnx::Graph::add_output,
             py::arg("name"), py::arg("shape"), py::arg("dtype") = litetorch::DataType::FP32)
        .def("add_node", &litetorch::onnx::Graph::add_node,
             py::arg("op_type"), py::arg("inputs"), py::arg("outputs"),
             py::arg("attrs") = std::map<std::string, std::string>(),
             py::arg("domain") = "")
        .def("add_initializer", &litetorch::onnx::Graph::add_initializer,
             py::arg("name"), py::arg("tensor"))
        .def("num_nodes", [](const litetorch::onnx::Graph& g) { return g.nodes.size(); })
        .def("node_op_types", [](const litetorch::onnx::Graph& g) {
            std::vector<std::string> out;
            for (auto& n : g.nodes) out.push_back(n.op_type);
            return out;
        });
    onnx_mod.def("export_onnx_text", &litetorch::onnx::export_onnx_text, py::arg("graph"));
    onnx_mod.def("export_onnx", &litetorch::onnx::export_onnx,
                 py::arg("graph"), py::arg("path"));
    onnx_mod.def("trace_module", &litetorch::onnx::trace_module,
                 py::arg("model"), py::arg("example_input"));
    onnx_mod.def("export", &litetorch::onnx::export_module,
                 py::arg("model"), py::arg("args"), py::arg("path"));

    auto prof_mod = m.def_submodule("profiler", "LiteTorch CPU profiler (minimal)");
    prof_mod.def("enable", &litetorch::profiler::enable);
    prof_mod.def("disable", &litetorch::profiler::disable);
    prof_mod.def("is_enabled", &litetorch::profiler::is_enabled);
    prof_mod.def("reset", &litetorch::profiler::reset);
    prof_mod.def("report", &litetorch::profiler::report);
    prof_mod.def("record", &litetorch::profiler::record, py::arg("name"), py::arg("ms"));
    prof_mod.def("events", []() {
        std::vector<std::map<std::string, double>> out;
        for (auto& e : litetorch::profiler::events())
            out.push_back({{"calls", (double)e.calls}, {"total_ms", e.total_ms},
                           {"avg_ms", e.avg_ms()}, {"max_ms", e.max_ms}});
        return out;
    });
    py::class_<PyProfileGuard>(prof_mod, "profile")
        .def(py::init([](const std::string& name) {
            auto g = std::make_unique<PyProfileGuard>();
            g->name = name;
            return g;
        }), py::arg("name") = "block")
        .def("__enter__", [](PyProfileGuard& self) -> PyProfileGuard& { self.enter(); return self; })
        .def("__exit__", [](PyProfileGuard& self, py::object, py::object, py::object) {
            self.exit();
        });

    auto hub_mod = m.def_submodule("hub", "LiteTorch Hub (minimal)");
    hub_mod.def("get_dir", &litetorch::hub::get_dir);
    hub_mod.def("set_dir", &litetorch::hub::set_dir, py::arg("dir"));
    hub_mod.def("is_url", &litetorch::hub::is_url, py::arg("path"));
    hub_mod.def("download_url_to_file", &litetorch::hub::download_url_to_file,
               py::arg("url"), py::arg("dst") = "", py::arg("show_progress") = true);
    hub_mod.def("load_state_dict_from_url", &litetorch::hub::load_state_dict_from_url,
               py::arg("url"), py::arg("force_download") = false);
}
