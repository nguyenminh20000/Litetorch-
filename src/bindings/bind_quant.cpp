#include "bindings_common.h"
#include "litetorch/quant_scheme.h"

namespace {

std::string qs_dtype_name(DataType dt) {
    switch (dt) {
        case DataType::FP32: return "FP32";
        case DataType::FP64: return "FP64";
        case DataType::FP16: return "FP16";
        case DataType::BF16: return "BF16";
        case DataType::INT16: return "INT16";
        case DataType::INT8: return "INT8";
        case DataType::INT4: return "INT4";
        case DataType::FP8_E4M3: return "FP8_E4M3";
        case DataType::FP8_E5M2: return "FP8_E5M2";
        case DataType::NF4: return "NF4";
        case DataType::FP4_E2M1: return "FP4_E2M1";
        case DataType::INT32: return "INT32";
        case DataType::INT64: return "INT64";
        case DataType::UINT8: return "UINT8";
        case DataType::BOOL: return "BOOL";
        default: throw std::runtime_error("quant_scheme: unknown DataType value");
    }
}

DataType qs_dtype_from_name(const std::string& s) {
    if (s == "FP32") return DataType::FP32;
    if (s == "FP64") return DataType::FP64;
    if (s == "FP16") return DataType::FP16;
    if (s == "BF16") return DataType::BF16;
    if (s == "INT16") return DataType::INT16;
    if (s == "INT8") return DataType::INT8;
    if (s == "INT4") return DataType::INT4;
    if (s == "FP8_E4M3") return DataType::FP8_E4M3;
    if (s == "FP8_E5M2") return DataType::FP8_E5M2;
    if (s == "NF4") return DataType::NF4;
    if (s == "FP4_E2M1") return DataType::FP4_E2M1;
    if (s == "INT32") return DataType::INT32;
    if (s == "INT64") return DataType::INT64;
    if (s == "UINT8") return DataType::UINT8;
    if (s == "BOOL") return DataType::BOOL;
    throw std::runtime_error("quant_scheme: unknown dtype name '" + s + "'");
}

std::map<std::string, std::string> qs_params_from_py(py::dict d) {
    std::map<std::string, std::string> out;
    for (auto kv : d) out[py::cast<std::string>(kv.first)] = py::cast<std::string>(kv.second);
    return out;
}

py::dict qs_params_to_py(const std::map<std::string, std::string>& m) {
    py::dict d;
    for (auto& kv : m) d[py::cast(kv.first)] = py::cast(kv.second);
    return d;
}

py::object qs_cal_dict(const quantization::Calibrator* cal) {
    if (!cal) return py::none();
    py::dict d;
    for (auto& kv : cal->min_vals) {
        auto it = cal->max_vals.find(kv.first);
        if (it != cal->max_vals.end()) d[py::cast(kv.first)] = py::make_tuple(kv.second, it->second);
    }
    return std::move(d);
}

}

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

        py::class_<quantization::QParams>(quant_mod, "QParams")
            .def(py::init<>())
            .def_readwrite("scales", &quantization::QParams::scales)
            .def_readwrite("zero_points", &quantization::QParams::zero_points)
            .def_readwrite("scale", &quantization::QParams::scale)
            .def_readwrite("zero_point", &quantization::QParams::zero_point)
            .def_readwrite("bits", &quantization::QParams::bits)
            .def_readwrite("block_size", &quantization::QParams::block_size);

        py::class_<quantization::QuantizedTensor>(quant_mod, "QuantizedTensor")
            .def(py::init<>())
            .def_readwrite("data", &quantization::QuantizedTensor::data)
            .def_readwrite("qparams", &quantization::QuantizedTensor::qparams)
            .def_readwrite("scheme_name", &quantization::QuantizedTensor::scheme_name)
            .def_readwrite("scheme_params", &quantization::QuantizedTensor::scheme_params)
            .def_readwrite("logical_shape", &quantization::QuantizedTensor::logical_shape)
            .def_property("logical_dtype",
                [](const quantization::QuantizedTensor& r) { return qs_dtype_name(r.logical_dtype); },
                [](quantization::QuantizedTensor& r, const std::string& s) { r.logical_dtype = qs_dtype_from_name(s); })
            .def_readwrite("packed", &quantization::QuantizedTensor::packed)
            .def_readwrite("nibble_order", &quantization::QuantizedTensor::nibble_order);

        py::class_<quantization::SchemeQuantizedLinear, nn::Module, std::shared_ptr<quantization::SchemeQuantizedLinear>>(quant_mod, "SchemeQuantizedLinear")
            .def("forward", &quantization::SchemeQuantizedLinear::forward)
            .def("parameters", &quantization::SchemeQuantizedLinear::parameters)
            .def("to", &quantization::SchemeQuantizedLinear::to)
            .def_readwrite("bias", &quantization::SchemeQuantizedLinear::bias)
            .def_property_readonly("record", [](const quantization::SchemeQuantizedLinear& m) { return m.record; })
            .def_property_readonly("scheme_name", [](const quantization::SchemeQuantizedLinear& m) { return m.record.scheme_name; });

        py::class_<quantization::SchemeQATLinear, nn::Module, std::shared_ptr<quantization::SchemeQATLinear>>(quant_mod, "SchemeQATLinear")
            .def(py::init<std::shared_ptr<nn::Linear>, std::string, std::map<std::string, std::string>>(),
                 py::arg("linear"), py::arg("scheme_name"), py::arg("params") = std::map<std::string, std::string>())
            .def("forward", &quantization::SchemeQATLinear::forward)
            .def("parameters", &quantization::SchemeQATLinear::parameters)
            .def("to", &quantization::SchemeQATLinear::to)
            .def_readwrite("weight", &quantization::SchemeQATLinear::weight)
            .def_readwrite("bias", &quantization::SchemeQATLinear::bias)
            .def_readwrite("scheme_name", &quantization::SchemeQATLinear::scheme_name)
            .def_readwrite("qat_enabled", &quantization::SchemeQATLinear::qat_enabled);

        quant_mod.def("register_scheme", [](const std::string& name, py::dict params, py::function calibrate, py::function quantize_fn, py::function dequantize_fn, py::object fake_quantize_fn) {
            quantization::QuantScheme scheme;
            scheme.name = name;
            scheme.params = qs_params_from_py(params);
            scheme.calibrate = [calibrate](std::shared_ptr<Tensor> t, const quantization::Calibrator* cal, const std::string& cname, const std::map<std::string, std::string>& p) {
                py::gil_scoped_acquire gil;
                py::object res = calibrate(t, qs_cal_dict(cal), cname, qs_params_to_py(p));
                return py::cast<quantization::QParams>(res);
            };
            scheme.quantize = [quantize_fn](std::shared_ptr<Tensor> t, const quantization::QParams& qp, const std::map<std::string, std::string>& p) {
                py::gil_scoped_acquire gil;
                py::object res = quantize_fn(t, qp, qs_params_to_py(p));
                return py::cast<quantization::QuantizedTensor>(res);
            };
            scheme.dequantize = [dequantize_fn](const quantization::QuantizedTensor& rec) {
                py::gil_scoped_acquire gil;
                py::object res = dequantize_fn(rec);
                return py::cast<std::shared_ptr<Tensor>>(res);
            };
            if (!fake_quantize_fn.is_none()) {
                py::function fq = py::cast<py::function>(fake_quantize_fn);
                scheme.fake_quantize = [fq](std::shared_ptr<Tensor> t, const quantization::QParams& qp, const std::map<std::string, std::string>& p) {
                    py::gil_scoped_acquire gil;
                    py::object res = fq(t, qp, qs_params_to_py(p));
                    return py::cast<std::shared_ptr<Tensor>>(res);
                };
            }
            quantization::register_scheme(scheme);
        }, py::arg("name"), py::arg("params") = py::dict(), py::arg("calibrate"), py::arg("quantize"), py::arg("dequantize"), py::arg("fake_quantize") = py::none());
        quant_mod.def("unregister_scheme", &quantization::unregister_scheme, py::arg("name"));
        quant_mod.def("has_scheme", &quantization::has_scheme, py::arg("name"));
        quant_mod.def("list_schemes", &quantization::list_schemes);
        quant_mod.def("quantize", &quantization::quantize_model, py::arg("model"), py::arg("scheme_name"), py::arg("params") = std::map<std::string, std::string>());
        quant_mod.def("quantize_ptq", &quantization::quantize_ptq, py::arg("prepared"), py::arg("scheme_name"), py::arg("params") = std::map<std::string, std::string>());
        quant_mod.def("save_quantized", &quantization::save_quantized, py::arg("model"), py::arg("dir"));
        quant_mod.def("load_quantized", &quantization::load_quantized, py::arg("model_template"), py::arg("dir"));
}
