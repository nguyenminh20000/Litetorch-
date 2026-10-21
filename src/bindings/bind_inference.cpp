#include "bindings_common.h"
#include "litetorch/inference.h"

void bind_inference(py::module_& m) {
    auto inf = m.def_submodule("inference", "LiteTorch Inference");

    py::class_<litetorch::inference::CudaGraph>(inf, "CudaGraph")
        .def(py::init<>())
        .def("capture", [](litetorch::inference::CudaGraph& self, py::function fn) {
            self.capture([fn]() { fn(); });
        }, py::arg("fn"))
        .def("replay", &litetorch::inference::CudaGraph::replay)
        .def("captured", &litetorch::inference::CudaGraph::captured);

    inf.def("capture_cuda_graph", &litetorch::inference::capture_cuda_graph,
            py::arg("model"), py::arg("static_in"));

    py::class_<litetorch::inference::FusionReport>(inf, "FusionReport")
        .def_readonly("conv_bn_folded", &litetorch::inference::FusionReport::conv_bn_folded)
        .def_readonly("layernorm_detected", &litetorch::inference::FusionReport::layernorm_detected)
        .def_readonly("linear_bias_kernel_fused", &litetorch::inference::FusionReport::linear_bias_kernel_fused)
        .def_readonly("notes", &litetorch::inference::FusionReport::notes);

    inf.def("fold_conv_bn", &litetorch::inference::fold_conv_bn,
            py::arg("conv"), py::arg("bn"));
    inf.def("fuse_conv_bn", &litetorch::inference::fuse_conv_bn, py::arg("seq"));
    inf.def("detect_layernorm_candidates", &litetorch::inference::detect_layernorm_candidates,
            py::arg("root"));
    inf.def("detect_layernorm_fusion", &litetorch::inference::detect_layernorm_fusion,
            py::arg("root"));
    inf.def("verify_linear_bias_fusion", &litetorch::inference::verify_linear_bias_fusion);
}
