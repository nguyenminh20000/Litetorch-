#include "bindings_common.h"

void bind_nn_base(py::module_& m);
void bind_nn_layers(py::module_& m);
void bind_nn_recurrent(py::module_& m);
void bind_nn_utils(py::module_& m);

void bind_nn(py::module_& m) {
    auto nn_mod = m.def_submodule("nn", "LiteTorch Neural Network Modules");
    bind_nn_base(m);
    bind_nn_layers(m);
    bind_nn_recurrent(m);
    bind_nn_utils(m);
}
