#include "bindings_common.h"

void bind_core(py::module_& m);
void bind_cuda(py::module_& m);
void bind_tpu(py::module_& m);
void bind_ops(py::module_& m);
void bind_nn(py::module_& m);
void bind_linalg(py::module_& m);
void bind_optim(py::module_& m);
void bind_amp(py::module_& m);
void bind_quant(py::module_& m);
void bind_distributed(py::module_& m);
void bind_jit(py::module_& m);
void bind_data(py::module_& m);
void bind_core_ops(py::module_& m);
void bind_autograd(py::module_& m);
void bind_distributions(py::module_& m);
void bind_fft(py::module_& m);
void bind_onnx(py::module_& m);
void bind_inference(py::module_& m);

PYBIND11_MODULE(litetorch, m) {
    m.doc() = "LiteTorch Python Bindings";
    bind_core(m);
    bind_cuda(m);
    bind_tpu(m);
    bind_ops(m);
    bind_nn(m);
    bind_linalg(m);
    bind_optim(m);
    bind_amp(m);
    bind_quant(m);
    bind_distributed(m);
    bind_jit(m);
    bind_data(m);
    bind_core_ops(m);
    bind_autograd(m);
    bind_distributions(m);
    bind_fft(m);
    bind_onnx(m);
    bind_inference(m);
}
