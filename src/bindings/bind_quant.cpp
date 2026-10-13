#include "bindings_common.h"

void bind_quant(py::module_& m) {
        auto quant_mod = m.def_submodule("quantization", "LiteTorch Quantization");
        quant_mod.def("quantize_linear", &quantization::quantize_linear, py::arg("layer"));

        py::class_<quantization::Calibrator>(quant_mod, "Calibrator")
            .def(py::init<>())
            .def("collect", &quantization::Calibrator::collect, py::arg("name"), py::arg("tensor"))
            .def("get_asymmetric_params", &quantization::Calibrator::get_asymmetric_params, py::arg("name"), py::arg("bits") = 8)
            .def("get_scale", &quantization::Calibrator::get_scale, py::arg("name"), py::arg("bits") = 8);

        py::class_<quantization::QATLinear, nn::Module, std::shared_ptr<quantization::QATLinear>>(quant_mod, "QATLinear")
            .def(py::init<std::shared_ptr<nn::Linear>, float, float, float, float, int>(),
                 py::arg("linear"), py::arg("w_scale") = 0.01f, py::arg("w_zp") = 0.0f, py::arg("a_scale") = 0.01f, py::arg("a_zp") = 0.0f, py::arg("b") = 8)
            .def("forward", &quantization::QATLinear::forward)
            .def("parameters", &quantization::QATLinear::parameters)
            .def("to", &quantization::QATLinear::to)
            .def_readwrite("weight", &quantization::QATLinear::weight)
            .def_readwrite("bias", &quantization::QATLinear::bias)
            .def_readwrite("weight_scale", &quantization::QATLinear::weight_scale)
            .def_readwrite("weight_zero_point", &quantization::QATLinear::weight_zero_point)
            .def_readwrite("act_scale", &quantization::QATLinear::act_scale)
            .def_readwrite("act_zero_point", &quantization::QATLinear::act_zero_point)
            .def_readwrite("bits", &quantization::QATLinear::bits)
            .def_readwrite("qat_enabled", &quantization::QATLinear::qat_enabled);

        py::class_<quantization::PTQConfig>(quant_mod, "PTQConfig")
            .def(py::init<>())
            .def_readwrite("bits", &quantization::PTQConfig::bits)
            .def_readwrite("per_channel", &quantization::PTQConfig::per_channel)
            .def_readwrite("symmetric", &quantization::PTQConfig::symmetric);

        py::class_<quantization::PreparedModel, std::shared_ptr<quantization::PreparedModel>>(quant_mod, "PreparedModel")
            .def("calibrate", &quantization::PreparedModel::calibrate, py::arg("data_fn"), py::arg("num_batches") = 10)
            .def_readwrite("model", &quantization::PreparedModel::model);

        quant_mod.def("prepare", &quantization::prepare, py::arg("model"), py::arg("config") = quantization::PTQConfig());
        quant_mod.def("convert", &quantization::convert, py::arg("prepared"));
        quant_mod.def("quantize_dynamic_linear", [](std::shared_ptr<nn::Linear> l) {
            return quantization::quantize_dynamic(l);
        }, py::arg("linear"));
        quant_mod.def("quantize_dynamic_lstm", [](std::shared_ptr<nn::LSTM> l) {
            return quantization::quantize_dynamic(l);
        }, py::arg("lstm"));
        quant_mod.def("quantize_dynamic_gru", [](std::shared_ptr<nn::GRU> l) {
            return quantization::quantize_dynamic(l);
        }, py::arg("gru"));
}
