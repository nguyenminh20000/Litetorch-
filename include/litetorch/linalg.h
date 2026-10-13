#ifndef LITETORCH_LINALG_H
#define LITETORCH_LINALG_H

#include "litetorch/tensor.h"
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace litetorch {
namespace linalg {

std::shared_ptr<Tensor> inv(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> solve(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> cholesky(std::shared_ptr<Tensor> a, bool upper = false);
std::shared_ptr<Tensor> cholesky_inverse(std::shared_ptr<Tensor> a, bool upper = false);
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> qr(std::shared_ptr<Tensor> a,
                                                               const std::string& mode = "reduced");
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> svd(
    std::shared_ptr<Tensor> a, bool full_matrices = true);
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> eigh(std::shared_ptr<Tensor> a,
                                                                 const std::string& UPLO = "L");
std::shared_ptr<Tensor> eigvalsh(std::shared_ptr<Tensor> a, const std::string& UPLO = "L");
std::shared_ptr<Tensor> det(std::shared_ptr<Tensor> a);
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> slogdet(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> vector_norm(std::shared_ptr<Tensor> x, double ord = 2.0);
std::shared_ptr<Tensor> matrix_norm(std::shared_ptr<Tensor> a, const std::string& ord = "fro");
std::shared_ptr<Tensor> matrix_rank(std::shared_ptr<Tensor> a, double rcond = -1.0,
                                    bool hermitian = false);
std::shared_ptr<Tensor> pinv(std::shared_ptr<Tensor> a, double rcond = 1e-15);
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
           std::shared_ptr<Tensor>>
lstsq(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, double rcond = -1.0);
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> lu(
    std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> cross(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
                              int64_t dim = -1);
std::shared_ptr<Tensor> vecdot(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y,
                               int64_t dim = -1);
std::shared_ptr<Tensor> multi_dot(const std::vector<std::shared_ptr<Tensor>>& mats);
std::shared_ptr<Tensor> diagonal(std::shared_ptr<Tensor> a, int64_t offset = 0,
                                 int64_t dim1 = -2, int64_t dim2 = -1);
std::shared_ptr<Tensor> trace(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> matrix_power(std::shared_ptr<Tensor> a, int64_t n);
std::shared_ptr<Tensor> cond(std::shared_ptr<Tensor> a, const std::string& p = "2");
std::shared_ptr<Tensor> outer(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> inner(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> kron(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> vdot(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

}
}

#endif
