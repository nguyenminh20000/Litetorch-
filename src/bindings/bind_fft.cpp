#include "bindings_common.h"
#include "litetorch/fft.h"

void bind_fft(py::module_& m) {
    auto fft_mod = m.def_submodule("fft", "LiteTorch FFT");
    fft_mod.def("fft", &fft::fft, py::arg("x"), py::arg("n") = -1, py::arg("dim") = -1,
               py::arg("norm") = "backward");
    fft_mod.def("ifft", &fft::ifft, py::arg("x"), py::arg("n") = -1, py::arg("dim") = -1,
               py::arg("norm") = "backward");
    fft_mod.def("rfft", &fft::rfft, py::arg("x"), py::arg("n") = -1, py::arg("dim") = -1,
               py::arg("norm") = "backward");
    fft_mod.def("irfft", &fft::irfft, py::arg("x"), py::arg("n") = -1, py::arg("dim") = -1,
                py::arg("norm") = "backward");
    fft_mod.def("fftn", &fft::fftn, py::arg("x"), py::arg("s") = std::vector<int64_t>(),
                py::arg("dims") = std::vector<int64_t>(), py::arg("norm") = "backward");
    fft_mod.def("ifftn", &fft::ifftn, py::arg("x"), py::arg("s") = std::vector<int64_t>(),
                py::arg("dims") = std::vector<int64_t>(), py::arg("norm") = "backward");
    fft_mod.def("fftshift", &fft::fftshift, py::arg("x"),
                py::arg("dims") = std::vector<int64_t>());
    fft_mod.def("ifftshift", &fft::ifftshift, py::arg("x"),
                py::arg("dims") = std::vector<int64_t>());
    fft_mod.def("fftfreq", &fft::fftfreq, py::arg("n"), py::arg("d") = 1.0);
}
