#include "bindings_common.h"
#include "litetorch/prune.h"

void bind_prune(py::module_& m) {
    auto prune_mod = m.def_submodule("prune", "LiteTorch pruning (torch.nn.utils.prune parity)");
    prune_mod.def("l1_unstructured",
                  [](std::shared_ptr<nn::Module> module, const std::string& name, int64_t amount) {
                      prune::l1_unstructured(*module, name, amount);
                  },
                  py::arg("module"), py::arg("name"), py::arg("amount"));
    prune_mod.def("l1_unstructured",
                  [](std::shared_ptr<nn::Module> module, const std::string& name, double amount) {
                      prune::l1_unstructured(*module, name, amount);
                  },
                  py::arg("module"), py::arg("name"), py::arg("amount"));
    prune_mod.def("random_unstructured",
                  [](std::shared_ptr<nn::Module> module, const std::string& name, int64_t amount,
                     int64_t seed) { prune::random_unstructured(*module, name, amount, seed); },
                  py::arg("module"), py::arg("name"), py::arg("amount"), py::arg("seed") = -1);
    prune_mod.def("random_unstructured",
                  [](std::shared_ptr<nn::Module> module, const std::string& name, double amount,
                     int64_t seed) { prune::random_unstructured(*module, name, amount, seed); },
                  py::arg("module"), py::arg("name"), py::arg("amount"), py::arg("seed") = -1);
    prune_mod.def("remove",
                  [](std::shared_ptr<nn::Module> module, const std::string& name) {
                      prune::remove(*module, name);
                  },
                  py::arg("module"), py::arg("name"));
    prune_mod.def("is_pruned",
                  [](std::shared_ptr<nn::Module> module, const std::string& name) {
                      return prune::is_pruned(*module, name);
                  },
                  py::arg("module"), py::arg("name"));
}
