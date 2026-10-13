#include "bindings_common.h"

void bind_linalg(py::module_& m) {
        // === W3-T1 linalg ===
        auto linalg_mod = m.def_submodule("linalg", "LiteTorch Linear Algebra");
        linalg_mod.def("inv", &linalg::inv, py::arg("A"));
        linalg_mod.def("solve", &linalg::solve, py::arg("A"), py::arg("B"));
        linalg_mod.def("cholesky", &linalg::cholesky, py::arg("A"), py::arg("upper") = false);
        linalg_mod.def("cholesky_inverse", &linalg::cholesky_inverse, py::arg("A"),
                      py::arg("upper") = false);
        linalg_mod.def("qr",
                       [](std::shared_ptr<Tensor> a, const std::string& mode) {
                           auto r = linalg::qr(a, mode);
                           if (mode == "r") return py::cast(r.second);
                           return py::cast(std::make_tuple(r.first, r.second));
                       },
                       py::arg("A"), py::arg("mode") = "reduced");
        linalg_mod.def("svd",
                       [](std::shared_ptr<Tensor> a, bool full_matrices) {
                           auto r = linalg::svd(a, full_matrices);
                           return std::make_tuple(std::get<0>(r), std::get<1>(r), std::get<2>(r));
                       },
                       py::arg("A"), py::arg("full_matrices") = true);
        linalg_mod.def("eigh",
                       [](std::shared_ptr<Tensor> a, const std::string& UPLO) {
                           auto r = linalg::eigh(a, UPLO);
                           return std::make_tuple(r.first, r.second);
                       },
                       py::arg("A"), py::arg("UPLO") = "L");
        linalg_mod.def("eigvalsh", &linalg::eigvalsh, py::arg("A"), py::arg("UPLO") = "L");
        linalg_mod.def("det", &linalg::det, py::arg("A"));
        linalg_mod.def("slogdet",
                       [](std::shared_ptr<Tensor> a) {
                           auto r = linalg::slogdet(a);
                           return std::make_tuple(r.first, r.second);
                       },
                       py::arg("A"));
        linalg_mod.def("vector_norm",
                       [](std::shared_ptr<Tensor> x, py::object ord) {
                           double o = 2.0;
                           if (!ord.is_none()) o = ord.cast<double>();
                           return linalg::vector_norm(x, o);
                       },
                       py::arg("x"), py::arg("ord") = py::none());
        linalg_mod.def("matrix_norm", &linalg::matrix_norm, py::arg("A"), py::arg("ord") = "fro");
        linalg_mod.def("norm",
                       [](std::shared_ptr<Tensor> x, py::object ord, py::object dim) {
                           if (dim.is_none()) {
                               double o = 2.0;
                               if (!ord.is_none()) {
                                   if (py::isinstance<py::str>(ord))
                                       throw std::runtime_error(
                                           "[litetorch Error] linalg.norm: string ord needs dim");
                                   o = ord.cast<double>();
                               }
                               return linalg::vector_norm(x, o);
                           }
                           if (py::isinstance<py::int_>(dim)) {
                               double o = 2.0;
                               if (!ord.is_none()) o = ord.cast<double>();
                               auto xd = x->flat_doubles();
                               int64_t ndim = (int64_t)x->shape.size();
                               int64_t d = dim.cast<int64_t>();
                               d = d < 0 ? d + ndim : d;
                               int64_t dd = x->shape[(size_t)d];
                               int64_t stride = 1;
                               for (int64_t i = d + 1; i < ndim; ++i) stride *= x->shape[(size_t)i];
                               int64_t outer = (int64_t)xd.size() / (dd * stride);
                               std::vector<int64_t> os;
                               for (int64_t i = 0; i < ndim; ++i)
                                   if (i != d) os.push_back(x->shape[(size_t)i]);
                               std::vector<double> out((size_t)(outer * stride));
                               for (int64_t o2 = 0; o2 < outer; ++o2)
                                   for (int64_t s = 0; s < stride; ++s) {
                                       std::vector<double> v((size_t)dd);
                                       int64_t base = o2 * dd * stride + s;
                                       for (int64_t i = 0; i < dd; ++i)
                                           v[(size_t)i] = xd[(size_t)(base + i * stride)];
                                       double nv;
                                       if (o == 0.0) {
                                           nv = 0.0;
                                           for (auto e : v)
                                               if (e != 0.0) nv += 1.0;
                                       } else if (std::isinf(o)) {
                                           nv = 0.0;
                                           for (auto e : v) nv = std::max(nv, std::fabs(e));
                                       } else {
                                           nv = 0.0;
                                           for (auto e : v) nv += std::pow(std::fabs(e), o);
                                           nv = std::pow(nv, 1.0 / o);
                                       }
                                       out[(size_t)(o2 * stride + s)] = nv;
                                   }
                               auto t = Tensor::create(os, Device(DeviceType::CPU, 0), false, x->dtype);
                               char* bp = reinterpret_cast<char*>(t->storage->get_cpu_ptr());
                               if (x->dtype == DataType::FP64) {
                                   double* p = reinterpret_cast<double*>(bp);
                                   for (size_t i = 0; i < out.size(); ++i) p[i] = out[i];
                               } else {
                                   float* p = reinterpret_cast<float*>(bp);
                                   for (size_t i = 0; i < out.size(); ++i)
                                       p[i] = static_cast<float>(out[i]);
                               }
                               return t;
                           }
                           std::string o = "fro";
                           if (!ord.is_none()) o = ord.cast<std::string>();
                           return linalg::matrix_norm(x, o);
                       },
                       py::arg("x"), py::arg("ord") = py::none(), py::arg("dim") = py::none());
        linalg_mod.def("matrix_rank", &linalg::matrix_rank, py::arg("A"), py::arg("tol") = -1.0,
                       py::arg("hermitian") = false);
        linalg_mod.def("pinv", &linalg::pinv, py::arg("A"), py::arg("rcond") = 1e-15);
        linalg_mod.def("lstsq",
                       [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, double rcond) {
                           auto r = linalg::lstsq(a, b, rcond);
                           return std::make_tuple(std::get<0>(r), std::get<1>(r), std::get<2>(r),
                                                  std::get<3>(r));
                       },
                       py::arg("A"), py::arg("B"), py::arg("rcond") = -1.0);
        linalg_mod.def("lu",
                       [](std::shared_ptr<Tensor> a) {
                           auto r = linalg::lu(a);
                           return std::make_tuple(std::get<0>(r), std::get<1>(r), std::get<2>(r));
                       },
                       py::arg("A"));
        linalg_mod.def("cross", &linalg::cross, py::arg("a"), py::arg("b"), py::arg("dim") = -1);
        linalg_mod.def("vecdot", &linalg::vecdot, py::arg("x"), py::arg("y"), py::arg("dim") = -1);
        linalg_mod.def("multi_dot", &linalg::multi_dot, py::arg("tensors"));
        linalg_mod.def("diagonal", &linalg::diagonal, py::arg("A"), py::arg("offset") = 0,
                       py::arg("dim1") = -2, py::arg("dim2") = -1);
        linalg_mod.def("trace", &linalg::trace, py::arg("A"));
        linalg_mod.def("matrix_power", &linalg::matrix_power, py::arg("A"), py::arg("n"));
        linalg_mod.def("cond", &linalg::cond, py::arg("A"), py::arg("p") = "2");
        linalg_mod.def("outer", &linalg::outer, py::arg("a"), py::arg("b"));
        linalg_mod.def("inner", &linalg::inner, py::arg("a"), py::arg("b"));
        linalg_mod.def("kron", &linalg::kron, py::arg("a"), py::arg("b"));
        linalg_mod.def("vdot", &linalg::vdot, py::arg("a"), py::arg("b"));
        // === end W3-T1 linalg ===
}
