#include "bindings_common.h"

void bind_distributions(py::module_& m) {
        // === W3-T2 distributions ===
        auto distr_mod = m.def_submodule("distributions", "LiteTorch Probability Distributions");
        {
            using namespace litetorch::distributions;
            py::class_<Distribution, std::shared_ptr<Distribution>>(distr_mod, "Distribution")
                .def("sample", [](std::shared_ptr<Distribution> self, py::object sample_shape) {
                    std::vector<int64_t> ss;
                    if (!sample_shape.is_none())
                        ss = py::cast<std::vector<int64_t>>(sample_shape);
                    return self->sample(ss);
                }, py::arg("sample_shape") = py::none())
                .def("log_prob", [](std::shared_ptr<Distribution> self, std::shared_ptr<Tensor> v) {
                    return self->log_prob(v);
                }, py::arg("value"))
                .def("entropy", [](std::shared_ptr<Distribution> self) { return self->entropy(); })
                .def("mean", [](std::shared_ptr<Distribution> self) { return self->mean(); })
                .def("variance", [](std::shared_ptr<Distribution> self) { return self->variance(); })
                .def_property_readonly("batch_shape", [](std::shared_ptr<Distribution> self) {
                    return self->batch_shape();
                });
            py::class_<Normal, Distribution, std::shared_ptr<Normal>>(distr_mod, "Normal")
                .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("loc"), py::arg("scale"));
            py::class_<Uniform, Distribution, std::shared_ptr<Uniform>>(distr_mod, "Uniform")
                .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("low"), py::arg("high"));
            py::class_<Exponential, Distribution, std::shared_ptr<Exponential>>(distr_mod, "Exponential")
                .def(py::init<std::shared_ptr<Tensor>>(), py::arg("rate"));
            py::class_<Laplace, Distribution, std::shared_ptr<Laplace>>(distr_mod, "Laplace")
                .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("loc"), py::arg("scale"));
            py::class_<Cauchy, Distribution, std::shared_ptr<Cauchy>>(distr_mod, "Cauchy")
                .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("loc"), py::arg("scale"));
            py::class_<Beta, Distribution, std::shared_ptr<Beta>>(distr_mod, "Beta")
                .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("alpha"), py::arg("beta"));
            py::class_<Gamma, Distribution, std::shared_ptr<Gamma>>(distr_mod, "Gamma")
                .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("alpha"), py::arg("beta"));
            py::class_<Dirichlet, Distribution, std::shared_ptr<Dirichlet>>(distr_mod, "Dirichlet")
                .def(py::init<std::shared_ptr<Tensor>>(), py::arg("concentration"));
            py::class_<Bernoulli, Distribution, std::shared_ptr<Bernoulli>>(distr_mod, "Bernoulli")
                .def(py::init<std::shared_ptr<Tensor>, bool>(), py::arg("probs"), py::arg("from_logits") = false);
            py::class_<Categorical, Distribution, std::shared_ptr<Categorical>>(distr_mod, "Categorical")
                .def(py::init<std::shared_ptr<Tensor>, bool>(), py::arg("probs"), py::arg("from_logits") = false);
            py::class_<Poisson, Distribution, std::shared_ptr<Poisson>>(distr_mod, "Poisson")
                .def(py::init<std::shared_ptr<Tensor>>(), py::arg("rate"));
            py::class_<Geometric, Distribution, std::shared_ptr<Geometric>>(distr_mod, "Geometric")
                .def(py::init<std::shared_ptr<Tensor>>(), py::arg("probs"));
            py::class_<Binomial, Distribution, std::shared_ptr<Binomial>>(distr_mod, "Binomial")
                .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("total_count"), py::arg("probs"));
            py::class_<NegativeBinomial, Distribution, std::shared_ptr<NegativeBinomial>>(distr_mod, "NegativeBinomial")
                .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("total_count"), py::arg("probs"));
            py::class_<StudentT, Distribution, std::shared_ptr<StudentT>>(distr_mod, "StudentT")
                .def(py::init<std::shared_ptr<Tensor>>(), py::arg("df"));
            py::class_<Chi2, Distribution, std::shared_ptr<Chi2>>(distr_mod, "Chi2")
                .def(py::init<std::shared_ptr<Tensor>>(), py::arg("df"));
            py::class_<LogNormal, Distribution, std::shared_ptr<LogNormal>>(distr_mod, "LogNormal")
                .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("loc"), py::arg("scale"));
            py::class_<Gumbel, Distribution, std::shared_ptr<Gumbel>>(distr_mod, "Gumbel")
                .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("loc"), py::arg("scale"));
            py::class_<Weibull, Distribution, std::shared_ptr<Weibull>>(distr_mod, "Weibull")
                .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("scale"), py::arg("concentration"));
            py::class_<MultivariateNormal, Distribution, std::shared_ptr<MultivariateNormal>>(distr_mod, "MultivariateNormal")
                .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("loc"), py::arg("covariance_matrix"));
            distr_mod.def("kl_divergence", [](std::shared_ptr<Distribution> p, std::shared_ptr<Distribution> q) {
                return kl_divergence(p, q);
            }, py::arg("p"), py::arg("q"));
        }
        // === end W3-T2 distributions ===
}
