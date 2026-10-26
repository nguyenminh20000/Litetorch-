#include "litetorch/linalg.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace litetorch {
namespace linalg {
namespace {

void require_cpu(const Tensor& t, const char* op) {
    if (t.device.type != DeviceType::CPU)
        throw std::runtime_error(std::string("[litetorch Error] linalg::") + op +
                                 ": only CPU tensors supported");
}

std::vector<double> to_doubles(const std::shared_ptr<Tensor>& t, const char* op) {
    require_cpu(*t, op);
    auto c = t->is_contiguous() ? t : t->contiguous();
    return c->flat_doubles();
}

uint16_t float_to_half_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t exp = (u >> 23) & 0xffu;
    uint32_t mant = u & 0x7fffffu;
    if (exp == 0xffu) return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0));
    int32_t e = (int32_t)exp - 112;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - e);
        uint32_t half = mant >> shift;
        uint32_t rest = mant & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1);
        if (rest > halfway || (rest == halfway && (half & 1u))) half += 1u;
        return (uint16_t)(sign | half);
    }
    uint32_t half = ((uint32_t)e << 10) | (mant >> 13);
    uint32_t rest = mant & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (half & 1u))) half += 1u;
    return (uint16_t)(sign | half);
}

uint16_t float_to_bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

int64_t double_to_int64(double d) {
    if (!(d < 9223372036854775808.0)) return std::numeric_limits<int64_t>::max();
    if (d <= -9223372036854775808.0) return std::numeric_limits<int64_t>::min();
    return (int64_t)std::llround(d);
}

std::shared_ptr<Tensor> from_doubles(const std::vector<double>& d,
                                     const std::vector<int64_t>& shape, DataType dtype) {
    Device dev(DeviceType::CPU, 0);
    auto out = Tensor::create(shape, dev, false, dtype);
    char* base = reinterpret_cast<char*>(out->storage->get_cpu_ptr());
    switch (dtype) {
        case DataType::FP64: {
            double* p = reinterpret_cast<double*>(base);
            for (size_t i = 0; i < d.size(); ++i) p[i] = d[i];
            break;
        }
        case DataType::FP32: {
            float* p = reinterpret_cast<float*>(base);
            for (size_t i = 0; i < d.size(); ++i) p[i] = static_cast<float>(d[i]);
            break;
        }
        case DataType::FP16: {
            uint16_t* p = reinterpret_cast<uint16_t*>(base);
            for (size_t i = 0; i < d.size(); ++i) p[i] = float_to_half_bits(static_cast<float>(d[i]));
            break;
        }
        case DataType::BF16: {
            uint16_t* p = reinterpret_cast<uint16_t*>(base);
            for (size_t i = 0; i < d.size(); ++i) p[i] = float_to_bf16_bits(static_cast<float>(d[i]));
            break;
        }
        case DataType::INT64: {
            int64_t* p = reinterpret_cast<int64_t*>(base);
            for (size_t i = 0; i < d.size(); ++i) p[i] = double_to_int64(d[i]);
            break;
        }
        case DataType::INT32: {
            int32_t* p = reinterpret_cast<int32_t*>(base);
            for (size_t i = 0; i < d.size(); ++i) p[i] = (int32_t)double_to_int64(d[i]);
            break;
        }
        case DataType::INT16: {
            int16_t* p = reinterpret_cast<int16_t*>(base);
            for (size_t i = 0; i < d.size(); ++i) p[i] = (int16_t)double_to_int64(d[i]);
            break;
        }
        case DataType::INT8: {
            int8_t* p = reinterpret_cast<int8_t*>(base);
            for (size_t i = 0; i < d.size(); ++i) p[i] = (int8_t)double_to_int64(d[i]);
            break;
        }
        case DataType::UINT8: {
            uint8_t* p = reinterpret_cast<uint8_t*>(base);
            for (size_t i = 0; i < d.size(); ++i) p[i] = (uint8_t)double_to_int64(d[i]);
            break;
        }
        case DataType::BOOL: {
            uint8_t* p = reinterpret_cast<uint8_t*>(base);
            for (size_t i = 0; i < d.size(); ++i) p[i] = d[i] != 0.0 ? 1 : 0;
            break;
        }
        default:
            throw std::runtime_error("[litetorch Error] linalg: from_doubles: unsupported dtype");
    }
    return out;
}

int64_t mul_wrap(int64_t a, int64_t b) {
    return (int64_t)((uint64_t)a * (uint64_t)b);
}

int64_t sub_wrap(int64_t a, int64_t b) {
    return (int64_t)((uint64_t)a - (uint64_t)b);
}

DataType promote_dtype(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return promote_types(a->dtype, b->dtype);
}

DataType same_dtype(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b, const char* op) {
    if (a->dtype != b->dtype)
        throw std::runtime_error(std::string("[litetorch Error] linalg::") + op +
                                 ": inputs must have the same dtype");
    if (a->dtype == DataType::BOOL)
        throw std::runtime_error(std::string("[litetorch Error] linalg::") + op +
                                 ": bool inputs not supported");
    return a->dtype;
}

DataType float_dtype(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b, const char* op,
                     bool allow_half) {
    if (a->dtype != b->dtype)
        throw std::runtime_error(std::string("[litetorch Error] linalg::") + op +
                                 ": inputs must have the same dtype");
    if (!is_float_dtype(a->dtype))
        throw std::runtime_error(std::string("[litetorch Error] linalg::") + op +
                                 ": expected a floating point tensor as input");
    if (!allow_half && a->dtype != DataType::FP32 && a->dtype != DataType::FP64)
        throw std::runtime_error(std::string("[litetorch Error] linalg::") + op +
                                 ": expected float32 or float64 tensor as input");
    return a->dtype;
}

std::vector<int64_t> to_int64s(const std::shared_ptr<Tensor>& t, const char* op) {
    require_cpu(*t, op);
    auto c = t->is_contiguous() ? t : t->contiguous();
    const char* base = reinterpret_cast<const char*>(c->storage->get_cpu_ptr()) +
                       c->offset * (int64_t)c->storage->element_size();
    std::vector<int64_t> out(c->numel());
    switch (c->dtype) {
        case DataType::INT64: {
            const int64_t* p = reinterpret_cast<const int64_t*>(base);
            for (size_t i = 0; i < out.size(); ++i) out[i] = p[i];
            break;
        }
        case DataType::INT32: {
            const int32_t* p = reinterpret_cast<const int32_t*>(base);
            for (size_t i = 0; i < out.size(); ++i) out[i] = p[i];
            break;
        }
        case DataType::INT16: {
            const int16_t* p = reinterpret_cast<const int16_t*>(base);
            for (size_t i = 0; i < out.size(); ++i) out[i] = p[i];
            break;
        }
        case DataType::INT8: {
            const int8_t* p = reinterpret_cast<const int8_t*>(base);
            for (size_t i = 0; i < out.size(); ++i) out[i] = p[i];
            break;
        }
        case DataType::UINT8:
        case DataType::BOOL: {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(base);
            for (size_t i = 0; i < out.size(); ++i) out[i] = p[i];
            break;
        }
        default:
            throw std::runtime_error(std::string("[litetorch Error] linalg::") + op +
                                     ": integer read of unsupported dtype");
    }
    return out;
}

std::shared_ptr<Tensor> from_int64s(const std::vector<int64_t>& v,
                                    const std::vector<int64_t>& shape, DataType dtype) {
    Device dev(DeviceType::CPU, 0);
    auto out = Tensor::create(shape, dev, false, dtype);
    char* base = reinterpret_cast<char*>(out->storage->get_cpu_ptr());
    switch (dtype) {
        case DataType::INT64: {
            int64_t* p = reinterpret_cast<int64_t*>(base);
            for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
            break;
        }
        case DataType::INT32: {
            int32_t* p = reinterpret_cast<int32_t*>(base);
            for (size_t i = 0; i < v.size(); ++i) p[i] = (int32_t)v[i];
            break;
        }
        case DataType::INT16: {
            int16_t* p = reinterpret_cast<int16_t*>(base);
            for (size_t i = 0; i < v.size(); ++i) p[i] = (int16_t)v[i];
            break;
        }
        case DataType::INT8: {
            int8_t* p = reinterpret_cast<int8_t*>(base);
            for (size_t i = 0; i < v.size(); ++i) p[i] = (int8_t)v[i];
            break;
        }
        case DataType::UINT8: {
            uint8_t* p = reinterpret_cast<uint8_t*>(base);
            for (size_t i = 0; i < v.size(); ++i) p[i] = (uint8_t)v[i];
            break;
        }
        case DataType::BOOL: {
            uint8_t* p = reinterpret_cast<uint8_t*>(base);
            for (size_t i = 0; i < v.size(); ++i) p[i] = v[i] ? 1 : 0;
            break;
        }
        default:
            throw std::runtime_error("[litetorch Error] linalg: from_int64s: unsupported dtype");
    }
    return out;
}

std::vector<int64_t> bcast_batch(const std::vector<int64_t>& s1,
                                 const std::vector<int64_t>& s2) {
    size_t n1 = s1.size() >= 2 ? s1.size() - 2 : 0;
    size_t n2 = s2.size() >= 2 ? s2.size() - 2 : 0;
    size_t n = std::max(n1, n2);
    std::vector<int64_t> b(n, 1);
    for (size_t i = 0; i < n; ++i) {
        int64_t d1 = 1, d2 = 1;
        if (i < n1) d1 = s1[n1 - 1 - i];
        if (i < n2) d2 = s2[n2 - 1 - i];
        if (d1 != d2 && d1 != 1 && d2 != 1)
            throw std::runtime_error("[litetorch Error] linalg: batch dimensions not broadcastable");
        b[n - 1 - i] = std::max(d1, d2);
    }
    return b;
}

int64_t shape_prod(const std::vector<int64_t>& s) {
    int64_t p = 1;
    for (auto d : s) p *= d;
    return p;
}

void fill_bcast(const std::vector<double>& src, const std::vector<int64_t>& src_shape,
                std::vector<double>& dst, const std::vector<int64_t>& bshape,
                int64_t rows, int64_t cols) {
    size_t nb = bshape.size();
    size_t ns = src_shape.size() >= 2 ? src_shape.size() - 2 : 0;
    int64_t count = shape_prod(bshape);
    int64_t mat = rows * cols;
    std::vector<int64_t> idx(nb, 0);
    for (int64_t b = 0; b < count; ++b) {
        int64_t tmp = b;
        for (size_t i = nb; i-- > 0;) {
            idx[i] = tmp % bshape[i];
            tmp /= bshape[i];
        }
        int64_t sboff = 0;
        int64_t stride = 1;
        for (size_t i = ns; i-- > 0;) {
            size_t si = nb - ns + i;
            int64_t d = src_shape[i];
            int64_t k = (d == 1) ? 0 : idx[si];
            sboff += k * stride;
            stride *= d;
        }
        const double* sp = src.data() + sboff * mat;
        double* dp = dst.data() + b * mat;
        std::memcpy(dp, sp, (size_t)mat * sizeof(double));
    }
}

int lu_decompose(double* A, int n, std::vector<int>& piv) {
    piv.resize((size_t)n);
    for (int i = 0; i < n; ++i) piv[(size_t)i] = i;
    int sign = 1;
    for (int k = 0; k < n; ++k) {
        int p = k;
        double mx = std::fabs(A[(size_t)k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            double v = std::fabs(A[(size_t)i * n + k]);
            if (v > mx) {
                mx = v;
                p = i;
            }
        }
        if (mx == 0.0) throw std::runtime_error("[litetorch Error] linalg: singular matrix");
        if (p != k) {
            for (int j = 0; j < n; ++j)
                std::swap(A[(size_t)k * n + j], A[(size_t)p * n + j]);
            std::swap(piv[(size_t)k], piv[(size_t)p]);
            sign = -sign;
        }
        double inv = 1.0 / A[(size_t)k * n + k];
        for (int i = k + 1; i < n; ++i) {
            A[(size_t)i * n + k] *= inv;
            double aik = A[(size_t)i * n + k];
            for (int j = k + 1; j < n; ++j) A[(size_t)i * n + j] -= aik * A[(size_t)k * n + j];
        }
    }
    return sign;
}

void lu_solve_vec(const double* LU, const std::vector<int>& piv, double* b, int n) {
    std::vector<double> y((size_t)n);
    for (int i = 0; i < n; ++i) y[(size_t)i] = b[(size_t)piv[(size_t)i]];
    for (int i = 0; i < n; ++i) {
        double s = y[(size_t)i];
        for (int j = 0; j < i; ++j) s -= LU[(size_t)i * n + j] * y[(size_t)j];
        y[(size_t)i] = s;
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = y[(size_t)i];
        for (int j = i + 1; j < n; ++j) s -= LU[(size_t)i * n + j] * b[(size_t)j];
        b[(size_t)i] = s / LU[(size_t)i * n + i];
    }
}

void mat_inv_buf(const double* A, double* Ai, int n) {
    std::vector<double> LU((size_t)n * n);
    std::memcpy(LU.data(), A, (size_t)n * n * sizeof(double));
    std::vector<int> piv;
    lu_decompose(LU.data(), n, piv);
    std::vector<double> col((size_t)n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) col[(size_t)i] = (i == j) ? 1.0 : 0.0;
        lu_solve_vec(LU.data(), piv, col.data(), n);
        for (int i = 0; i < n; ++i) Ai[(size_t)i * n + j] = col[(size_t)i];
    }
}

void cholesky_lower_buf(const double* A, double* L, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[(size_t)i * n + j];
            for (int k = 0; k < j; ++k) s -= L[(size_t)i * n + k] * L[(size_t)j * n + k];
            if (i == j) {
                if (s <= 0.0)
                    throw std::runtime_error("[litetorch Error] linalg::cholesky: not positive-definite");
                L[(size_t)i * n + j] = std::sqrt(s);
            } else {
                L[(size_t)i * n + j] = s / L[(size_t)j * n + j];
            }
        }
        for (int j = i + 1; j < n; ++j) L[(size_t)i * n + j] = 0.0;
    }
}

void qr_mgs_buf(const double* A, double* Q, double* R, int m, int n) {
    int k = std::min(m, n);
    std::vector<double> V((size_t)m * n);
    std::memcpy(V.data(), A, (size_t)m * n * sizeof(double));
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < j; ++i) {
            double s = 0.0;
            for (int r = 0; r < m; ++r) s += Q[(size_t)r * k + i] * V[(size_t)r * n + j];
            R[(size_t)i * n + j] = s;
            for (int r = 0; r < m; ++r) V[(size_t)r * n + j] -= s * Q[(size_t)r * k + i];
        }
        double nr = 0.0;
        for (int r = 0; r < m; ++r) nr += V[(size_t)r * n + j] * V[(size_t)r * n + j];
        nr = std::sqrt(nr);
        R[(size_t)j * n + j] = nr;
        if (nr > 0) {
            for (int r = 0; r < m; ++r) Q[(size_t)r * k + j] = V[(size_t)r * n + j] / nr;
        } else {
            for (int r = 0; r < m; ++r) Q[(size_t)r * k + j] = 0.0;
        }
        for (int c = j + 1; c < n; ++c) R[(size_t)j * n + c] = 0.0;
        for (int i = j + 1; i < k; ++i) R[(size_t)i * n + j] = 0.0;
    }
}

void svd_jacobi_buf(const double* A, double* U, double* S, double* V, int m, int n) {
    int k = std::min(m, n);
    std::vector<double> B((size_t)m * n);
    std::memcpy(B.data(), A, (size_t)m * n * sizeof(double));
    std::vector<double> Vf((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) Vf[(size_t)i * n + i] = 1.0;
    const double eps = std::numeric_limits<double>::epsilon();
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int i = 0; i < m; ++i) {
                    double bp = B[(size_t)i * n + p], bq = B[(size_t)i * n + q];
                    alpha += bp * bp;
                    beta += bq * bq;
                    gamma += bp * bq;
                }
                off += gamma * gamma;
                if (std::fabs(gamma) <= eps * std::sqrt(alpha * beta)) continue;
                double zeta = (beta - alpha) / (2.0 * gamma);
                double t = (zeta >= 0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = c * t;
                for (int i = 0; i < m; ++i) {
                    double bp = B[(size_t)i * n + p], bq = B[(size_t)i * n + q];
                    B[(size_t)i * n + p] = c * bp - s * bq;
                    B[(size_t)i * n + q] = s * bp + c * bq;
                }
                for (int i = 0; i < n; ++i) {
                    double vp = Vf[(size_t)i * n + p], vq = Vf[(size_t)i * n + q];
                    Vf[(size_t)i * n + p] = c * vp - s * vq;
                    Vf[(size_t)i * n + q] = s * vp + c * vq;
                }
            }
        }
        if (off == 0.0) break;
    }
    std::vector<int> idx((size_t)n);
    std::vector<double> Sall((size_t)n);
    for (int j = 0; j < n; ++j) {
        double nr = 0.0;
        for (int i = 0; i < m; ++i) nr += B[(size_t)i * n + j] * B[(size_t)i * n + j];
        Sall[(size_t)j] = std::sqrt(nr);
        idx[(size_t)j] = j;
    }
    std::sort(idx.begin(), idx.end(), [&](int x, int y) { return Sall[(size_t)x] > Sall[(size_t)y]; });
    for (int j = 0; j < k; ++j) S[j] = Sall[(size_t)idx[(size_t)j]];
    for (int j = 0; j < k; ++j) {
        int src = idx[(size_t)j];
        double sj = S[(size_t)j];
        for (int i = 0; i < m; ++i)
            U[(size_t)i * k + j] = (sj > 0) ? B[(size_t)i * n + src] / sj : 0.0;
        for (int i = 0; i < n; ++i) V[(size_t)i * k + j] = Vf[(size_t)i * n + src];
    }
}

void eigh_jacobi_buf(const double* A, double* w, double* V, int n, bool upper) {
    std::vector<double> M((size_t)n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double aij = A[(size_t)i * n + j], aji = A[(size_t)j * n + i];
            M[(size_t)i * n + j] = upper ? (j >= i ? aij : aji) : (j <= i ? aij : aji);
        }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) V[(size_t)i * n + j] = (i == j) ? 1.0 : 0.0;
    const double eps = std::numeric_limits<double>::epsilon();
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n - 1; ++p)
            for (int q = p + 1; q < n; ++q) {
                double apq = M[(size_t)p * n + q];
                off += apq * apq;
                if (std::fabs(apq) <= eps * 10) continue;
                double app = M[(size_t)p * n + p], aqq = M[(size_t)q * n + q];
                double theta = (aqq - app) / (2.0 * apq);
                double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(1.0 + theta * theta));
                double c = 1.0 / std::sqrt(1.0 + t * t), s = c * t;
                for (int r = 0; r < n; ++r) {
                    double mrp = M[(size_t)r * n + p], mrq = M[(size_t)r * n + q];
                    M[(size_t)r * n + p] = c * mrp - s * mrq;
                    M[(size_t)r * n + q] = s * mrp + c * mrq;
                }
                for (int r = 0; r < n; ++r) {
                    double mpr = M[(size_t)p * n + r], mqr = M[(size_t)q * n + r];
                    M[(size_t)p * n + r] = c * mpr - s * mqr;
                    M[(size_t)q * n + r] = s * mpr + c * mqr;
                }
                for (int r = 0; r < n; ++r) {
                    double vrp = V[(size_t)r * n + p], vrq = V[(size_t)r * n + q];
                    V[(size_t)r * n + p] = c * vrp - s * vrq;
                    V[(size_t)r * n + q] = s * vrp + c * vrq;
                }
            }
        if (off == 0.0) break;
    }
    std::vector<int> idx((size_t)n);
    for (int i = 0; i < n; ++i) {
        w[i] = M[(size_t)i * n + i];
        idx[(size_t)i] = i;
    }
    std::sort(idx.begin(), idx.end(), [&](int x, int y) { return w[x] < w[y]; });
    std::vector<double> ws((size_t)n), Vs((size_t)n * n);
    for (int i = 0; i < n; ++i) {
        ws[(size_t)i] = w[(size_t)idx[(size_t)i]];
        for (int r = 0; r < n; ++r) Vs[(size_t)r * n + i] = V[(size_t)r * n + idx[(size_t)i]];
    }
    for (int i = 0; i < n; ++i) w[i] = ws[(size_t)i];
    std::memcpy(V, Vs.data(), (size_t)n * n * sizeof(double));
}

void matmul_buf(const double* A, const double* B, double* C, int m, int k, int n) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int p = 0; p < k; ++p) s += A[(size_t)i * k + p] * B[(size_t)p * n + j];
            C[(size_t)i * n + j] = s;
        }
}

void complete_basis(const double* Qthin, double* Qfull, int m, int k) {
    std::vector<double> Q((size_t)m * m, 0.0);
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < m; ++i) Q[(size_t)i * m + j] = Qthin[(size_t)i * k + j];
    int col = k;
    std::vector<double> v((size_t)m);
    for (int e = 0; e < m && col < m; ++e) {
        for (int i = 0; i < m; ++i) v[(size_t)i] = (i == e) ? 1.0 : 0.0;
        for (int j = 0; j < col; ++j) {
            double s = 0.0;
            for (int i = 0; i < m; ++i) s += Q[(size_t)i * m + j] * v[(size_t)i];
            for (int i = 0; i < m; ++i) v[(size_t)i] -= s * Q[(size_t)i * m + j];
        }
        double nr = 0.0;
        for (int i = 0; i < m; ++i) nr += v[(size_t)i] * v[(size_t)i];
        nr = std::sqrt(nr);
        if (nr > 1e-10) {
            for (int i = 0; i < m; ++i) Q[(size_t)i * m + col] = v[(size_t)i] / nr;
            ++col;
        }
    }
    std::memcpy(Qfull, Q.data(), (size_t)m * m * sizeof(double));
}

}

std::shared_ptr<Tensor> inv(std::shared_ptr<Tensor> a) {
    if (a->shape.size() < 2) throw std::runtime_error("[litetorch Error] linalg::inv: expected >=2D");
    int64_t n = a->shape.back();
    if (a->shape[a->shape.size() - 2] != n)
        throw std::runtime_error("[litetorch Error] linalg::inv: expected square matrix");
    auto ad = to_doubles(a, "inv");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    std::vector<double> out((size_t)count * n * n);
    int ni = (int)n;
    for (int64_t b = 0; b < count; ++b)
        mat_inv_buf(ad.data() + b * n * n, out.data() + b * n * n, ni);
    std::vector<int64_t> os = bs;
    os.push_back(n);
    os.push_back(n);
    return from_doubles(out, os, a->dtype);
}

std::shared_ptr<Tensor> solve(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape.size() < 2 || b->shape.size() < 1)
        throw std::runtime_error("[litetorch Error] linalg::solve: bad shapes");
    int64_t n = a->shape.back();
    if (a->shape[a->shape.size() - 2] != n)
        throw std::runtime_error("[litetorch Error] linalg::solve: A must be square");
    bool bvec = (b->shape.size() == a->shape.size() - 1);
    int64_t nrhs;
    if (bvec) {
        nrhs = 1;
    } else {
        if (b->shape.size() != a->shape.size())
            throw std::runtime_error("[litetorch Error] linalg::solve: B ndim must match A or be vector");
        if (b->shape[b->shape.size() - 2] != n)
            throw std::runtime_error("[litetorch Error] linalg::solve: B rows must match A");
        nrhs = b->shape.back();
    }
    DataType dt = float_dtype(a, b, "solve", false);
    auto bs = bcast_batch(a->shape, bvec ? b->shape : b->shape);
    auto ad = to_doubles(a, "solve");
    auto bd = to_doubles(b, "solve");
    std::vector<double> abuf((size_t)shape_prod(bs) * n * n);
    std::vector<double> bbuf((size_t)shape_prod(bs) * n * nrhs);
    fill_bcast(ad, a->shape, abuf, bs, n, n);
    {
        std::vector<int64_t> bsh = b->shape;
        if (bvec) bsh.push_back(1);
        fill_bcast(bd, bsh, bbuf, bs, n, nrhs);
    }
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    int ni = (int)n;
    std::vector<int> piv;
    std::vector<double> LU((size_t)n * n);
    std::vector<double> col((size_t)n);
    for (int64_t bidx = 0; bidx < count; ++bidx) {
        std::memcpy(LU.data(), abuf.data() + bidx * n * n, (size_t)n * n * sizeof(double));
        lu_decompose(LU.data(), ni, piv);
        for (int64_t c = 0; c < nrhs; ++c) {
            for (int i = 0; i < n; ++i) col[(size_t)i] = bbuf[(size_t)(bidx * n * nrhs + i * nrhs + c)];
            lu_solve_vec(LU.data(), piv, col.data(), ni);
            for (int i = 0; i < n; ++i) bbuf[(size_t)(bidx * n * nrhs + i * nrhs + c)] = col[(size_t)i];
        }
    }
    std::vector<int64_t> os = bs;
    if (bvec) {
        os.insert(os.end(), b->shape.end() - 1, b->shape.end());
    } else {
        os.push_back(n);
        os.push_back(nrhs);
    }
    return from_doubles(bbuf, os, dt);
}

std::shared_ptr<Tensor> cholesky(std::shared_ptr<Tensor> a, bool upper) {
    if (a->shape.size() < 2) throw std::runtime_error("[litetorch Error] linalg::cholesky: expected >=2D");
    int64_t n = a->shape.back();
    if (a->shape[a->shape.size() - 2] != n)
        throw std::runtime_error("[litetorch Error] linalg::cholesky: expected square matrix");
    auto ad = to_doubles(a, "cholesky");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    std::vector<double> out((size_t)count * n * n);
    int ni = (int)n;
    for (int64_t b = 0; b < count; ++b) {
        cholesky_lower_buf(ad.data() + b * n * n, out.data() + b * n * n, ni);
        if (upper) {
            double* M = out.data() + b * n * n;
            for (int i = 0; i < ni; ++i)
                for (int j = i + 1; j < ni; ++j) M[(size_t)i * ni + j] = M[(size_t)j * ni + i];
            for (int i = 0; i < ni; ++i)
                for (int j = 0; j < i; ++j) M[(size_t)i * ni + j] = 0.0;
        }
    }
    std::vector<int64_t> os = bs;
    os.push_back(n);
    os.push_back(n);
    return from_doubles(out, os, a->dtype);
}

std::shared_ptr<Tensor> cholesky_inverse(std::shared_ptr<Tensor> a, bool upper) {
    if (a->shape.size() < 2) throw std::runtime_error("[litetorch Error] linalg::cholesky_inverse: expected >=2D");
    int64_t n = a->shape.back();
    auto L = cholesky(a, false);
    auto Ld = to_doubles(L, "cholesky_inverse");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    int ni = (int)n;
    std::vector<double> Linv((size_t)count * n * n), Ainv((size_t)count * n * n);
    std::vector<double> col((size_t)n);
    for (int64_t b = 0; b < count; ++b) {
        const double* Lp = Ld.data() + b * n * n;
        double* Ip = Linv.data() + b * n * n;
        for (int j = 0; j < ni; ++j) {
            for (int i = 0; i < ni; ++i) col[(size_t)i] = (i == j) ? 1.0 : 0.0;
            for (int i = 0; i < ni; ++i) {
                double s = col[(size_t)i];
                for (int k = 0; k < i; ++k) s -= Lp[(size_t)i * ni + k] * col[(size_t)k];
                col[(size_t)i] = s / Lp[(size_t)i * ni + i];
            }
            for (int i = 0; i < ni; ++i) Ip[(size_t)i * ni + j] = col[(size_t)i];
        }
        double* Ap = Ainv.data() + b * n * n;
        for (int i = 0; i < ni; ++i)
            for (int j = 0; j < ni; ++j) {
                double s = 0.0;
                for (int k = 0; k < ni; ++k) s += Ip[(size_t)k * ni + i] * Ip[(size_t)k * ni + j];
                Ap[(size_t)i * ni + j] = s;
            }
    }
    (void)upper;
    std::vector<int64_t> os = bs;
    os.push_back(n);
    os.push_back(n);
    return from_doubles(Ainv, os, a->dtype);
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> qr(std::shared_ptr<Tensor> a,
                                                               const std::string& mode) {
    if (a->shape.size() < 2) throw std::runtime_error("[litetorch Error] linalg::qr: expected >=2D");
    int64_t m = a->shape[a->shape.size() - 2], n = a->shape.back();
    if (mode != "reduced" && mode != "complete" && mode != "r")
        throw std::runtime_error("[litetorch Error] linalg::qr: mode must be reduced/complete/r");
    auto ad = to_doubles(a, "qr");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    int mi = (int)m, ni = (int)n;
    int k = std::min(mi, ni);
    std::vector<double> Qv((size_t)count * m * k), Rv((size_t)count * k * n);
    for (int64_t b = 0; b < count; ++b)
        qr_mgs_buf(ad.data() + b * m * n, Qv.data() + b * m * k, Rv.data() + b * k * n, mi, ni);
    std::vector<int64_t> qs = bs, rs = bs;
    if (mode == "complete") {
        std::vector<double> Qf((size_t)count * m * m);
        for (int64_t b = 0; b < count; ++b)
            complete_basis(Qv.data() + b * m * k, Qf.data() + b * m * m, mi, k);
        qs.push_back(m);
        qs.push_back(m);
        rs.push_back(m);
        rs.push_back(n);
        std::vector<double> Rf((size_t)count * m * n, 0.0);
        for (int64_t b = 0; b < count; ++b)
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < ni; ++j)
                    Rf[(size_t)(b * m * n + i * n + j)] = Rv[(size_t)(b * k * n + i * n + j)];
        auto Q = from_doubles(Qf, qs, a->dtype);
        auto R = from_doubles(Rf, rs, a->dtype);
        return {Q, R};
    }
    qs.push_back(m);
    qs.push_back(k);
    rs.push_back(k);
    rs.push_back(n);
    if (mode == "r") {
        auto R = from_doubles(Rv, rs, a->dtype);
        return {nullptr, R};
    }
    auto Q = from_doubles(Qv, qs, a->dtype);
    auto R = from_doubles(Rv, rs, a->dtype);
    return {Q, R};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> svd(
    std::shared_ptr<Tensor> a, bool full_matrices) {
    if (a->shape.size() < 2) throw std::runtime_error("[litetorch Error] linalg::svd: expected >=2D");
    int64_t m = a->shape[a->shape.size() - 2], n = a->shape.back();
    auto ad = to_doubles(a, "svd");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    int mi = (int)m, ni = (int)n;
    int k = std::min(mi, ni);
    std::vector<double> Uv((size_t)count * m * k), Sv((size_t)count * k), Vv((size_t)count * n * k);
    for (int64_t b = 0; b < count; ++b)
        svd_jacobi_buf(ad.data() + b * m * n, Uv.data() + b * m * k, Sv.data() + b * k,
                       Vv.data() + b * n * k, mi, ni);
    std::vector<int64_t> us = bs, ss = bs, vs = bs;
    std::vector<double> Uo, Vo;
    if (full_matrices) {
        Uo.assign((size_t)count * m * m, 0.0);
        Vo.assign((size_t)count * n * n, 0.0);
        for (int64_t b = 0; b < count; ++b) {
            complete_basis(Uv.data() + b * m * k, Uo.data() + b * m * m, mi, k);
            complete_basis(Vv.data() + b * n * k, Vo.data() + b * n * n, ni, k);
        }
        us.push_back(m);
        us.push_back(m);
        vs.push_back(n);
        vs.push_back(n);
    } else {
        Uo = Uv;
        Vo = Vv;
        us.push_back(m);
        us.push_back(k);
        vs.push_back(n);
        vs.push_back(k);
    }
    ss.push_back(k);
    auto U = from_doubles(Uo, us, a->dtype);
    auto S = from_doubles(Sv, ss, a->dtype);
    std::vector<double> Vh((size_t)count * (full_matrices ? n * n : n * k));
    int vn = full_matrices ? ni : k;
    for (int64_t b = 0; b < count; ++b) {
        const double* Vp = Vo.data() + b * n * vn;
        double* Hp = Vh.data() + b * vn * n;
        for (int64_t i = 0; i < vn; ++i)
            for (int j = 0; j < ni; ++j) Hp[(size_t)i * ni + j] = Vp[(size_t)j * vn + i];
    }
    std::vector<int64_t> hs = bs;
    hs.push_back(vn);
    hs.push_back(n);
    auto Vh_t = from_doubles(Vh, hs, a->dtype);
    return {U, S, Vh_t};
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> eigh(std::shared_ptr<Tensor> a,
                                                                 const std::string& UPLO) {
    if (a->shape.size() < 2) throw std::runtime_error("[litetorch Error] linalg::eigh: expected >=2D");
    int64_t n = a->shape.back();
    if (a->shape[a->shape.size() - 2] != n)
        throw std::runtime_error("[litetorch Error] linalg::eigh: expected square matrix");
    bool upper = (UPLO == "U" || UPLO == "u");
    if (!upper && UPLO != "L" && UPLO != "l")
        throw std::runtime_error("[litetorch Error] linalg::eigh: UPLO must be L or U");
    auto ad = to_doubles(a, "eigh");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    int ni = (int)n;
    std::vector<double> wv((size_t)count * n), Vv((size_t)count * n * n);
    for (int64_t b = 0; b < count; ++b)
        eigh_jacobi_buf(ad.data() + b * n * n, wv.data() + b * n, Vv.data() + b * n * n, ni,
                        upper);
    std::vector<int64_t> ws = bs, vs = bs;
    ws.push_back(n);
    vs.push_back(n);
    vs.push_back(n);
    return {from_doubles(wv, ws, a->dtype), from_doubles(Vv, vs, a->dtype)};
}

std::shared_ptr<Tensor> eigvalsh(std::shared_ptr<Tensor> a, const std::string& UPLO) {
    return eigh(a, UPLO).first;
}

std::shared_ptr<Tensor> det(std::shared_ptr<Tensor> a) {
    if (a->shape.size() < 2) throw std::runtime_error("[litetorch Error] linalg::det: expected >=2D");
    int64_t n = a->shape.back();
    if (a->shape[a->shape.size() - 2] != n)
        throw std::runtime_error("[litetorch Error] linalg::det: expected square matrix");
    auto ad = to_doubles(a, "det");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    std::vector<double> out((size_t)count);
    int ni = (int)n;
    std::vector<double> LU((size_t)n * n);
    std::vector<int> piv;
    for (int64_t b = 0; b < count; ++b) {
        std::memcpy(LU.data(), ad.data() + b * n * n, (size_t)n * n * sizeof(double));
        int sign = 1;
        try {
            sign = lu_decompose(LU.data(), ni, piv);
        } catch (const std::runtime_error&) {
            out[(size_t)b] = 0.0;
            continue;
        }
        double d = (double)sign;
        for (int i = 0; i < ni; ++i) d *= LU[(size_t)i * ni + i];
        out[(size_t)b] = d;
    }
    return from_doubles(out, bs, a->dtype);
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> slogdet(std::shared_ptr<Tensor> a) {
    if (a->shape.size() < 2) throw std::runtime_error("[litetorch Error] linalg::slogdet: expected >=2D");
    int64_t n = a->shape.back();
    if (a->shape[a->shape.size() - 2] != n)
        throw std::runtime_error("[litetorch Error] linalg::slogdet: expected square matrix");
    auto ad = to_doubles(a, "slogdet");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    std::vector<double> signv((size_t)count), logv((size_t)count);
    int ni = (int)n;
    std::vector<double> LU((size_t)n * n);
    std::vector<int> piv;
    for (int64_t b = 0; b < count; ++b) {
        std::memcpy(LU.data(), ad.data() + b * n * n, (size_t)n * n * sizeof(double));
        int sign = 1;
        bool singular = false;
        try {
            sign = lu_decompose(LU.data(), ni, piv);
        } catch (const std::runtime_error&) {
            singular = true;
        }
        if (singular) {
            signv[(size_t)b] = 0.0;
            logv[(size_t)b] = -std::numeric_limits<double>::infinity();
            continue;
        }
        double s = (double)sign, l = 0.0;
        for (int i = 0; i < ni; ++i) {
            double d = LU[(size_t)i * ni + i];
            if (d < 0) {
                s = -s;
                d = -d;
            }
            l += std::log(d);
        }
        signv[(size_t)b] = s;
        logv[(size_t)b] = l;
    }
    return {from_doubles(signv, bs, a->dtype), from_doubles(logv, bs, a->dtype)};
}

std::shared_ptr<Tensor> vector_norm(std::shared_ptr<Tensor> x, double ord) {
    auto xd = to_doubles(x, "vector_norm");
    int64_t n = (int64_t)xd.size();
    double v = 0.0;
    if (std::isinf(ord)) {
        double mx = 0.0;
        for (auto e : xd) mx = std::max(mx, std::fabs(e));
        v = (ord > 0) ? mx : *std::min_element(xd.begin(), xd.end(), [](double p, double q) {
            return std::fabs(p) < std::fabs(q);
        });
        if (ord < 0) v = std::fabs(v);
    } else if (ord == 0.0) {
        for (auto e : xd)
            if (e != 0.0) v += 1.0;
    } else if (ord == 1.0) {
        for (auto e : xd) v += std::fabs(e);
    } else {
        for (auto e : xd) v += std::pow(std::fabs(e), ord);
        v = std::pow(v, 1.0 / ord);
    }
    (void)n;
    return from_doubles({v}, {}, x->dtype);
}

std::shared_ptr<Tensor> matrix_norm(std::shared_ptr<Tensor> a, const std::string& ord) {
    if (a->shape.size() < 2) throw std::runtime_error("[litetorch Error] linalg::matrix_norm: expected >=2D");
    int64_t m = a->shape[a->shape.size() - 2], n = a->shape.back();
    auto ad = to_doubles(a, "matrix_norm");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    std::vector<double> out((size_t)count);
    int mi = (int)m, ni = (int)n;
    int k = std::min(mi, ni);
    for (int64_t b = 0; b < count; ++b) {
        const double* M = ad.data() + b * m * n;
        double v = 0.0;
        if (ord == "fro" || ord == "F") {
            for (int64_t i = 0; i < m * n; ++i) v += M[i] * M[i];
            v = std::sqrt(v);
        } else if (ord == "nuc") {
            std::vector<double> U((size_t)m * k), S((size_t)k), V((size_t)n * k);
            svd_jacobi_buf(M, U.data(), S.data(), V.data(), mi, ni);
            for (int i = 0; i < k; ++i) v += S[(size_t)i];
        } else if (ord == "inf") {
            for (int i = 0; i < mi; ++i) {
                double s = 0.0;
                for (int j = 0; j < ni; ++j) s += std::fabs(M[(size_t)i * ni + j]);
                v = std::max(v, s);
            }
        } else if (ord == "-inf") {
            v = std::numeric_limits<double>::infinity();
            for (int i = 0; i < mi; ++i) {
                double s = 0.0;
                for (int j = 0; j < ni; ++j) s += std::fabs(M[(size_t)i * ni + j]);
                v = std::min(v, s);
            }
        } else if (ord == "1") {
            for (int j = 0; j < ni; ++j) {
                double s = 0.0;
                for (int i = 0; i < mi; ++i) s += std::fabs(M[(size_t)i * ni + j]);
                v = std::max(v, s);
            }
        } else if (ord == "-1") {
            v = std::numeric_limits<double>::infinity();
            for (int j = 0; j < ni; ++j) {
                double s = 0.0;
                for (int i = 0; i < mi; ++i) s += std::fabs(M[(size_t)i * ni + j]);
                v = std::min(v, s);
            }
        } else if (ord == "2") {
            std::vector<double> U((size_t)m * k), S((size_t)k), V((size_t)n * k);
            svd_jacobi_buf(M, U.data(), S.data(), V.data(), mi, ni);
            v = S[0];
        } else if (ord == "-2") {
            std::vector<double> U((size_t)m * k), S((size_t)k), V((size_t)n * k);
            svd_jacobi_buf(M, U.data(), S.data(), V.data(), mi, ni);
            v = S[(size_t)k - 1];
        } else {
            throw std::runtime_error("[litetorch Error] linalg::matrix_norm: unsupported ord");
        }
        out[(size_t)b] = v;
    }
    return from_doubles(out, bs, a->dtype);
}

std::shared_ptr<Tensor> matrix_rank(std::shared_ptr<Tensor> a, double rcond, bool hermitian) {
    if (a->shape.size() < 2) throw std::runtime_error("[litetorch Error] linalg::matrix_rank: expected >=2D");
    int64_t m = a->shape[a->shape.size() - 2], n = a->shape.back();
    auto ad = to_doubles(a, "matrix_rank");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    std::vector<double> out((size_t)count);
    int mi = (int)m, ni = (int)n;
    int k = std::min(mi, ni);
    const double eps = std::numeric_limits<double>::epsilon();
    if (hermitian && m != n)
        throw std::runtime_error("[litetorch Error] linalg::matrix_rank: hermitian needs square");
    for (int64_t b = 0; b < count; ++b) {
        const double* M = ad.data() + b * m * n;
        std::vector<double> S((size_t)k);
        if (hermitian) {
            std::vector<double> w((size_t)n), V((size_t)n * n);
            eigh_jacobi_buf(M, w.data(), V.data(), ni, false);
            for (int i = 0; i < k; ++i) S[(size_t)i] = std::fabs(w[(size_t)i]);
            std::sort(S.begin(), S.end(), std::greater<double>());
        } else {
            std::vector<double> U((size_t)m * k), V((size_t)n * k);
            svd_jacobi_buf(M, U.data(), S.data(), V.data(), mi, ni);
        }
        double tol = (rcond < 0) ? (double)std::max(mi, ni) * eps * S[0] : rcond * S[0];
        double r = 0;
        for (int i = 0; i < k; ++i)
            if (S[(size_t)i] > tol) r += 1.0;
        out[(size_t)b] = r;
    }
    return from_doubles(out, bs, a->dtype);
}

std::shared_ptr<Tensor> pinv(std::shared_ptr<Tensor> a, double rcond) {
    if (a->shape.size() < 2) throw std::runtime_error("[litetorch Error] linalg::pinv: expected >=2D");
    int64_t m = a->shape[a->shape.size() - 2], n = a->shape.back();
    auto ad = to_doubles(a, "pinv");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    int mi = (int)m, ni = (int)n;
    int k = std::min(mi, ni);
    std::vector<double> out((size_t)count * n * m);
    std::vector<double> U((size_t)m * k), S((size_t)k), V((size_t)n * k);
    std::vector<double> VS((size_t)n * k);
    for (int64_t b = 0; b < count; ++b) {
        svd_jacobi_buf(ad.data() + b * m * n, U.data(), S.data(), V.data(), mi, ni);
        double cutoff = rcond * S[0];
        for (int i = 0; i < ni; ++i)
            for (int j = 0; j < k; ++j) {
                double s = S[(size_t)j];
                VS[(size_t)i * k + j] = (s > cutoff) ? V[(size_t)i * k + j] / s : 0.0;
            }
        for (int i = 0; i < ni; ++i)
            for (int j = 0; j < mi; ++j) {
                double s = 0.0;
                for (int p = 0; p < k; ++p) s += VS[(size_t)i * k + p] * U[(size_t)j * k + p];
                out[(size_t)(b * n * m + i * m + j)] = s;
            }
    }
    std::vector<int64_t> os = bs;
    os.push_back(n);
    os.push_back(m);
    return from_doubles(out, os, a->dtype);
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
           std::shared_ptr<Tensor>>
lstsq(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, double rcond) {
    if (a->shape.size() < 2 || b->shape.size() < 1)
        throw std::runtime_error("[litetorch Error] linalg::lstsq: bad shapes");
    int64_t m = a->shape[a->shape.size() - 2], n = a->shape.back();
    bool bvec = (b->shape.size() == a->shape.size() - 1);
    int64_t nrhs;
    if (bvec) {
        nrhs = 1;
    } else {
        if (b->shape[b->shape.size() - 2] != m)
            throw std::runtime_error("[litetorch Error] linalg::lstsq: B rows must match A rows");
        nrhs = b->shape.back();
    }
    DataType dt = float_dtype(a, b, "lstsq", false);
    auto bs = bcast_batch(a->shape, b->shape);
    auto ad = to_doubles(a, "lstsq");
    auto bd = to_doubles(b, "lstsq");
    std::vector<double> abuf((size_t)shape_prod(bs) * m * n);
    std::vector<double> bbuf((size_t)shape_prod(bs) * m * nrhs);
    fill_bcast(ad, a->shape, abuf, bs, m, n);
    {
        std::vector<int64_t> bsh = b->shape;
        if (bvec) bsh.push_back(1);
        fill_bcast(bd, bsh, bbuf, bs, m, nrhs);
    }
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    int mi = (int)m, ni = (int)n;
    int k = std::min(mi, ni);
    const double eps = std::numeric_limits<double>::epsilon();
    std::vector<double> X((size_t)count * n * nrhs), Rv((size_t)count * nrhs),
        Rkv((size_t)count), Sv((size_t)count * k);
    std::vector<double> U((size_t)m * k), S((size_t)k), V((size_t)n * k);
    std::vector<double> Utb((size_t)k * nrhs), VS((size_t)n * k);
    for (int64_t bidx = 0; bidx < count; ++bidx) {
        const double* Ap = abuf.data() + bidx * m * n;
        const double* Bp = bbuf.data() + bidx * m * nrhs;
        double* Xp = X.data() + bidx * n * nrhs;
        svd_jacobi_buf(Ap, U.data(), S.data(), V.data(), mi, ni);
        double rc = (rcond < 0) ? (double)std::max(mi, ni) * eps : rcond;
        double cutoff = rc * S[0];
        int64_t rank = 0;
        for (int i = 0; i < k; ++i)
            if (S[(size_t)i] > cutoff) ++rank;
        for (int i = 0; i < ni; ++i)
            for (int j = 0; j < k; ++j)
                VS[(size_t)i * k + j] = (S[(size_t)j] > cutoff) ? V[(size_t)i * k + j] / S[(size_t)j] : 0.0;
        for (int i = 0; i < k; ++i)
            for (int c = 0; c < nrhs; ++c) {
                double s = 0.0;
                for (int r = 0; r < mi; ++r) s += U[(size_t)r * k + i] * Bp[(size_t)r * nrhs + c];
                Utb[(size_t)i * nrhs + c] = s;
            }
        for (int i = 0; i < ni; ++i)
            for (int c = 0; c < nrhs; ++c) {
                double s = 0.0;
                for (int j = 0; j < k; ++j) s += VS[(size_t)i * k + j] * Utb[(size_t)j * nrhs + c];
                Xp[(size_t)i * nrhs + c] = s;
            }
        for (int c = 0; c < nrhs; ++c) {
            double rs = 0.0;
            if (m > n && rank == n) {
                for (int i = 0; i < mi; ++i) {
                    double ax = 0.0;
                    for (int j = 0; j < ni; ++j) ax += Ap[(size_t)i * ni + j] * Xp[(size_t)j * nrhs + c];
                    double d = Bp[(size_t)i * nrhs + c] - ax;
                    rs += d * d;
                }
            }
            Rv[(size_t)(bidx * nrhs + c)] = rs;
        }
        Rkv[(size_t)bidx] = (double)rank;
        for (int i = 0; i < k; ++i) Sv[(size_t)(bidx * k + i)] = S[(size_t)i];
    }
    std::vector<int64_t> xs = bs, rs = bs, ks = bs, svs = bs;
    if (bvec) {
        xs.push_back(n);
    } else {
        xs.push_back(n);
        xs.push_back(nrhs);
        rs.push_back(nrhs);
    }
    if (bvec) rs = bs;
    ks = bs;
    svs.push_back(k);
    std::vector<int64_t> rk(Rkv.size());
    for (size_t i = 0; i < rk.size(); ++i) rk[i] = (int64_t)Rkv[i];
    return {from_doubles(X, xs, dt), from_doubles(Rv, rs, dt), from_int64s(rk, ks, DataType::INT64),
            from_doubles(Sv, svs, dt)};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> lu(
    std::shared_ptr<Tensor> a) {
    if (a->shape.size() < 2) throw std::runtime_error("[litetorch Error] linalg::lu: expected >=2D");
    int64_t m = a->shape[a->shape.size() - 2], n = a->shape.back();
    auto ad = to_doubles(a, "lu");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    int mi = (int)m, ni = (int)n;
    int k = std::min(mi, ni);
    std::vector<double> Pv((size_t)count * m * m, 0.0), Lv((size_t)count * m * k, 0.0),
        Uv((size_t)count * k * n, 0.0);
    std::vector<double> LU((size_t)m * n);
    std::vector<int> piv;
    for (int64_t b = 0; b < count; ++b) {
        std::memcpy(LU.data(), ad.data() + b * m * n, (size_t)m * n * sizeof(double));
        int sign = 1;
        if (mi == ni) {
            sign = lu_decompose(LU.data(), ni, piv);
        } else {
            piv.resize((size_t)mi);
            for (int i = 0; i < mi; ++i) piv[(size_t)i] = i;
            for (int kk = 0; kk < k; ++kk) {
                int p = kk;
                double mx = std::fabs(LU[(size_t)kk * ni + kk]);
                for (int i = kk + 1; i < mi; ++i) {
                    double v = std::fabs(LU[(size_t)i * ni + kk]);
                    if (v > mx) {
                        mx = v;
                        p = i;
                    }
                }
                if (p != kk) {
                    for (int j = 0; j < ni; ++j)
                        std::swap(LU[(size_t)kk * ni + j], LU[(size_t)p * ni + j]);
                    std::swap(piv[(size_t)kk], piv[(size_t)p]);
                    sign = -sign;
                }
                if (LU[(size_t)kk * ni + kk] != 0.0) {
                    double inv = 1.0 / LU[(size_t)kk * ni + kk];
                    for (int i = kk + 1; i < mi; ++i) {
                        LU[(size_t)i * ni + kk] *= inv;
                        double aik = LU[(size_t)i * ni + kk];
                        for (int j = kk + 1; j < ni; ++j)
                            LU[(size_t)i * ni + j] -= aik * LU[(size_t)kk * ni + j];
                    }
                }
            }
        }
        (void)sign;
        double* Pp = Pv.data() + b * m * m;
        double* Lp = Lv.data() + b * m * k;
        double* Up = Uv.data() + b * k * n;
        for (int i = 0; i < mi; ++i) Pp[(size_t)piv[(size_t)i] * mi + i] = 1.0;
        for (int i = 0; i < mi; ++i) {
            for (int j = 0; j < k; ++j) {
                if (i == j) Lp[(size_t)i * k + j] = 1.0;
                else if (i > j) Lp[(size_t)i * k + j] = LU[(size_t)i * ni + j];
            }
        }
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < ni; ++j)
                Up[(size_t)i * ni + j] = (j >= i) ? LU[(size_t)i * ni + j] : 0.0;
    }
    std::vector<int64_t> ps = bs, ls = bs, us = bs;
    ps.push_back(m);
    ps.push_back(m);
    ls.push_back(m);
    ls.push_back(k);
    us.push_back(k);
    us.push_back(n);
    return {from_doubles(Pv, ps, a->dtype), from_doubles(Lv, ls, a->dtype),
            from_doubles(Uv, us, a->dtype)};
}

std::shared_ptr<Tensor> cross(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, int64_t dim) {
    if (a->shape != b->shape)
        throw std::runtime_error("[litetorch Error] linalg::cross: shapes must match");
    int64_t ndim = (int64_t)a->shape.size();
    if (ndim == 0) throw std::runtime_error("[litetorch Error] linalg::cross: expected >=1D");
    int64_t d = dim < 0 ? dim + ndim : dim;
    if (d < 0 || d >= ndim) throw std::runtime_error("[litetorch Error] linalg::cross: bad dim");
    if (a->shape[(size_t)d] != 3)
        throw std::runtime_error("[litetorch Error] linalg::cross: dim must have size 3");
    DataType dt = same_dtype(a, b, "cross");
    if (is_integer_dtype(dt)) {
        auto ai = to_int64s(a, "cross");
        auto bi = to_int64s(b, "cross");
        int64_t n = (int64_t)ai.size();
        int64_t stride = 1;
        for (int64_t i = d + 1; i < ndim; ++i) stride *= a->shape[(size_t)i];
        int64_t outer = n / (3 * stride);
        std::vector<int64_t> out((size_t)n);
        for (int64_t o = 0; o < outer; ++o)
            for (int64_t s = 0; s < stride; ++s) {
                int64_t base = (o * 3 * stride) + s;
                int64_t a0 = ai[(size_t)base], a1 = ai[(size_t)(base + stride)],
                        a2 = ai[(size_t)(base + 2 * stride)];
                int64_t b0 = bi[(size_t)base], b1 = bi[(size_t)(base + stride)],
                        b2 = bi[(size_t)(base + 2 * stride)];
                out[(size_t)base] = sub_wrap(mul_wrap(a1, b2), mul_wrap(a2, b1));
                out[(size_t)(base + stride)] = sub_wrap(mul_wrap(a2, b0), mul_wrap(a0, b2));
                out[(size_t)(base + 2 * stride)] = sub_wrap(mul_wrap(a0, b1), mul_wrap(a1, b0));
            }
        return from_int64s(out, a->shape, dt);
    }
    auto ad = to_doubles(a, "cross");
    auto bd = to_doubles(b, "cross");
    int64_t n = (int64_t)ad.size();
    int64_t stride = 1;
    for (int64_t i = d + 1; i < ndim; ++i) stride *= a->shape[(size_t)i];
    int64_t outer = n / (3 * stride);
    std::vector<double> out(n);
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t s = 0; s < stride; ++s) {
            int64_t base = (o * 3 * stride) + s;
            double a0 = ad[(size_t)base], a1 = ad[(size_t)(base + stride)],
                   a2 = ad[(size_t)(base + 2 * stride)];
            double b0 = bd[(size_t)base], b1 = bd[(size_t)(base + stride)],
                   b2 = bd[(size_t)(base + 2 * stride)];
            out[(size_t)base] = a1 * b2 - a2 * b1;
            out[(size_t)(base + stride)] = a2 * b0 - a0 * b2;
            out[(size_t)(base + 2 * stride)] = a0 * b1 - a1 * b0;
        }
    return from_doubles(out, a->shape, dt);
}

std::shared_ptr<Tensor> vecdot(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y, int64_t dim) {
    if (x->shape != y->shape)
        throw std::runtime_error("[litetorch Error] linalg::vecdot: shapes must match");
    int64_t ndim = (int64_t)x->shape.size();
    if (ndim == 0) throw std::runtime_error("[litetorch Error] linalg::vecdot: expected >=1D");
    int64_t d = dim < 0 ? dim + ndim : dim;
    DataType dt = float_dtype(x, y, "vecdot", true);
    auto xd = to_doubles(x, "vecdot");
    auto yd = to_doubles(y, "vecdot");
    int64_t n = (int64_t)xd.size();
    int64_t dd = x->shape[(size_t)d];
    int64_t stride = 1;
    for (int64_t i = d + 1; i < ndim; ++i) stride *= x->shape[(size_t)i];
    int64_t outer = n / (dd * stride);
    std::vector<int64_t> os;
    for (int64_t i = 0; i < ndim; ++i)
        if (i != d) os.push_back(x->shape[(size_t)i]);
    std::vector<double> out((size_t)(outer * stride), 0.0);
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t s = 0; s < stride; ++s) {
            double acc = 0.0;
            int64_t base = o * dd * stride + s;
            for (int64_t i = 0; i < dd; ++i)
                acc += xd[(size_t)(base + i * stride)] * yd[(size_t)(base + i * stride)];
            out[(size_t)(o * stride + s)] = acc;
        }
    return from_doubles(out, os, dt);
}

std::shared_ptr<Tensor> multi_dot(const std::vector<std::shared_ptr<Tensor>>& mats) {
    size_t L = mats.size();
    if (L == 0) throw std::runtime_error("[litetorch Error] linalg::multi_dot: empty list");
    if (L == 1) {
        auto c = mats[0]->is_contiguous() ? mats[0] : mats[0]->contiguous();
        return c->clone();
    }
    for (auto& t : mats) {
        require_cpu(*t, "multi_dot");
        if (t->shape.size() != 2)
            throw std::runtime_error("[litetorch Error] linalg::multi_dot: only 2D supported");
    }
    std::vector<int64_t> dims(L + 1);
    dims[0] = mats[0]->shape[0];
    for (size_t i = 0; i < L; ++i) {
        if (mats[i]->shape[0] != dims[i])
            throw std::runtime_error("[litetorch Error] linalg::multi_dot: incompatible shapes");
        dims[i + 1] = mats[i]->shape[1];
    }
    std::vector<std::vector<int64_t>> cost(L, std::vector<int64_t>(L, 0));
    std::vector<std::vector<int>> split(L, std::vector<int>(L, 0));
    for (size_t len = 2; len <= L; ++len)
        for (size_t i = 0; i + len <= L; ++i) {
            size_t j = i + len - 1;
            cost[i][j] = std::numeric_limits<int64_t>::max();
            for (size_t k = i; k < j; ++k) {
                int64_t c = cost[i][k] + cost[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1];
                if (c < cost[i][j]) {
                    cost[i][j] = c;
                    split[i][j] = (int)k;
                }
            }
        }
    std::vector<std::vector<double>> bufs(L);
    std::vector<std::pair<int, int>> bufshape(L);
    for (size_t i = 0; i < L; ++i) {
        bufs[i] = to_doubles(mats[i], "multi_dot");
        bufshape[i] = {(int)mats[i]->shape[0], (int)mats[i]->shape[1]};
    }
    std::function<std::vector<double>(size_t, size_t, int&, int&)> solve_range =
        [&](size_t i, size_t j, int& r, int& c) -> std::vector<double> {
        if (i == j) {
            r = bufshape[i].first;
            c = bufshape[i].second;
            return bufs[i];
        }
        int k = split[i][j];
        int r1, c1, r2, c2;
        auto A = solve_range(i, (size_t)k, r1, c1);
        auto B = solve_range((size_t)k + 1, j, r2, c2);
        r = r1;
        c = c2;
        std::vector<double> C((size_t)r * c);
        matmul_buf(A.data(), B.data(), C.data(), r1, c1, c2);
        return C;
    };
    int r, c;
    auto out = solve_range(0, L - 1, r, c);
    DataType dt = mats[0]->dtype;
    return from_doubles(out, {(int64_t)r, (int64_t)c}, dt);
}

std::shared_ptr<Tensor> diagonal(std::shared_ptr<Tensor> a, int64_t offset, int64_t dim1,
                                 int64_t dim2) {
    int64_t ndim = (int64_t)a->shape.size();
    if (ndim < 2) throw std::runtime_error("[litetorch Error] linalg::diagonal: expected >=2D");
    int64_t d1 = dim1 < 0 ? dim1 + ndim : dim1;
    int64_t d2 = dim2 < 0 ? dim2 + ndim : dim2;
    if (d1 == d2) throw std::runtime_error("[litetorch Error] linalg::diagonal: dims must differ");
    auto ad = to_doubles(a, "diagonal");
    int64_t n1 = a->shape[(size_t)d1], n2 = a->shape[(size_t)d2];
    int64_t s1 = 1, s2 = 1;
    for (int64_t i = d1 + 1; i < ndim; ++i) s1 *= a->shape[(size_t)i];
    for (int64_t i = d2 + 1; i < ndim; ++i) s2 *= a->shape[(size_t)i];
    int64_t len;
    if (offset >= 0)
        len = std::min(n1, n2 - offset);
    else
        len = std::min(n1 + offset, n2);
    if (len < 0) len = 0;
    std::vector<int64_t> os;
    for (int64_t i = 0; i < ndim; ++i)
        if (i != d1 && i != d2) os.push_back(a->shape[(size_t)i]);
    os.push_back(len);
    int64_t total_rest = 1;
    for (size_t i = 0; i + 1 < os.size(); ++i) total_rest *= os[i];
    std::vector<double> out((size_t)(total_rest * len));
    for (int64_t rr = 0; rr < total_rest; ++rr) {
        int64_t tmp = rr;
        int64_t base = 0;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            if (i == d1 || i == d2) continue;
            int64_t dd = a->shape[(size_t)i];
            int64_t k = tmp % dd;
            tmp /= dd;
            int64_t st = 1;
            for (int64_t j = i + 1; j < ndim; ++j) st *= a->shape[(size_t)j];
            base += k * st;
        }
        for (int64_t i = 0; i < len; ++i) {
            int64_t i1 = offset >= 0 ? i : i - offset;
            int64_t i2 = offset >= 0 ? i + offset : i;
            out[(size_t)(rr * len + i)] = ad[(size_t)(base + i1 * s1 + i2 * s2)];
        }
    }
    return from_doubles(out, os, a->dtype);
}

std::shared_ptr<Tensor> trace(std::shared_ptr<Tensor> a) {
    auto dd = to_doubles(diagonal(a, 0, -2, -1), "trace");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    std::vector<double> out((size_t)count, 0.0);
    int64_t n = a->shape.back();
    for (int64_t b = 0; b < count; ++b) {
        double acc = 0.0;
        for (int64_t i = 0; i < n; ++i) acc += dd[(size_t)(b * n + i)];
        out[(size_t)b] = acc;
    }
    return from_doubles(out, bs, a->dtype);
}

std::shared_ptr<Tensor> matrix_power(std::shared_ptr<Tensor> a, int64_t n) {
    if (a->shape.size() < 2) throw std::runtime_error("[litetorch Error] linalg::matrix_power: expected >=2D");
    int64_t sz = a->shape.back();
    if (a->shape[a->shape.size() - 2] != sz)
        throw std::runtime_error("[litetorch Error] linalg::matrix_power: expected square");
    auto ad = to_doubles(a, "matrix_power");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    int ni = (int)sz;
    std::vector<double> out((size_t)count * sz * sz);
    std::vector<double> base((size_t)sz * sz), acc((size_t)sz * sz), tmp((size_t)sz * sz);
    for (int64_t b = 0; b < count; ++b) {
        const double* Ap = ad.data() + b * sz * sz;
        if (n < 0) {
            mat_inv_buf(Ap, base.data(), ni);
        } else {
            std::memcpy(base.data(), Ap, (size_t)sz * sz * sizeof(double));
        }
        for (int i = 0; i < ni; ++i)
            for (int j = 0; j < ni; ++j) acc[(size_t)i * ni + j] = (i == j) ? 1.0 : 0.0;
        int64_t e = n < 0 ? -n : n;
        while (e > 0) {
            if (e & 1) {
                matmul_buf(acc.data(), base.data(), tmp.data(), ni, ni, ni);
                acc.swap(tmp);
            }
            matmul_buf(base.data(), base.data(), tmp.data(), ni, ni, ni);
            base.swap(tmp);
            e >>= 1;
        }
        std::memcpy(out.data() + b * sz * sz, acc.data(), (size_t)sz * sz * sizeof(double));
    }
    std::vector<int64_t> os = bs;
    os.push_back(sz);
    os.push_back(sz);
    return from_doubles(out, os, a->dtype);
}

std::shared_ptr<Tensor> cond(std::shared_ptr<Tensor> a, const std::string& p) {
    if (a->shape.size() < 2) throw std::runtime_error("[litetorch Error] linalg::cond: expected >=2D");
    int64_t m = a->shape[a->shape.size() - 2], n = a->shape.back();
    auto ad = to_doubles(a, "cond");
    std::vector<int64_t> bs(a->shape.begin(), a->shape.end() - 2);
    int64_t count = std::max<int64_t>(1, shape_prod(bs));
    std::vector<double> out((size_t)count);
    int mi = (int)m, ni = (int)n;
    int k = std::min(mi, ni);
    for (int64_t b = 0; b < count; ++b) {
        const double* M = ad.data() + b * m * n;
        double v;
        if (p == "2" || p == "") {
            std::vector<double> U((size_t)m * k), S((size_t)k), V((size_t)n * k);
            svd_jacobi_buf(M, U.data(), S.data(), V.data(), mi, ni);
            v = (S[(size_t)k - 1] == 0) ? std::numeric_limits<double>::infinity()
                                        : S[0] / S[(size_t)k - 1];
        } else if (p == "fro" || p == "nuc") {
            auto Mt = from_doubles(std::vector<double>(M, M + m * n), {m, n}, DataType::FP64);
            auto Pi = pinv(Mt, 1e-15);
            auto Pd = to_doubles(Pi, "cond");
            auto N1 = matrix_norm(Mt, p == "fro" ? "fro" : "nuc");
            auto N2 = matrix_norm(Pi, p == "fro" ? "fro" : "nuc");
            v = to_doubles(N1, "cond")[0] * to_doubles(N2, "cond")[0];
            (void)Pd;
        } else if (p == "inf" || p == "-inf" || p == "1" || p == "-1") {
            if (m != n) throw std::runtime_error("[litetorch Error] linalg::cond: p-norm cond needs square");
            auto Mt = from_doubles(std::vector<double>(M, M + m * n), {m, n}, DataType::FP64);
            auto Ai = inv(Mt);
            auto N1 = matrix_norm(Mt, p);
            auto N2 = matrix_norm(Ai, p);
            v = to_doubles(N1, "cond")[0] * to_doubles(N2, "cond")[0];
        } else {
            throw std::runtime_error("[litetorch Error] linalg::cond: unsupported p");
        }
        out[(size_t)b] = v;
    }
    return from_doubles(out, bs, a->dtype);
}

std::shared_ptr<Tensor> outer(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape.size() != 1 || b->shape.size() != 1)
        throw std::runtime_error("[litetorch Error] linalg::outer: expected 1D inputs");
    DataType dt = promote_dtype(a, b);
    if (is_integer_dtype(dt)) {
        auto ai = to_int64s(a, "outer");
        auto bi = to_int64s(b, "outer");
        int64_t m = a->shape[0], n = b->shape[0];
        std::vector<int64_t> out((size_t)(m * n));
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < n; ++j) out[(size_t)(i * n + j)] = mul_wrap(ai[(size_t)i], bi[(size_t)j]);
        return from_int64s(out, {m, n}, dt);
    }
    auto ad = to_doubles(a, "outer");
    auto bd = to_doubles(b, "outer");
    int64_t m = a->shape[0], n = b->shape[0];
    std::vector<double> out((size_t)(m * n));
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j) out[(size_t)(i * n + j)] = ad[(size_t)i] * bd[(size_t)j];
    return from_doubles(out, {m, n}, dt);
}

std::shared_ptr<Tensor> inner(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    DataType dt = same_dtype(a, b, "inner");
    if (is_integer_dtype(dt)) {
        auto ai = to_int64s(a, "inner");
        auto bi = to_int64s(b, "inner");
        if (a->shape.size() == 1 && b->shape.size() == 1) {
            if (a->shape[0] != b->shape[0])
                throw std::runtime_error("[litetorch Error] linalg::inner: size mismatch");
            uint64_t acc = 0;
            for (size_t i = 0; i < ai.size(); ++i) acc += (uint64_t)ai[i] * (uint64_t)bi[i];
            return from_int64s({(int64_t)acc}, {}, dt);
        }
        if (a->shape.empty() || b->shape.empty())
            throw std::runtime_error("[litetorch Error] linalg::inner: bad shapes");
        int64_t na = a->shape.back(), nb = b->shape.back();
        if (na != nb) throw std::runtime_error("[litetorch Error] linalg::inner: last dim mismatch");
        std::vector<int64_t> sa(a->shape.begin(), a->shape.end() - 1);
        std::vector<int64_t> sb(b->shape.begin(), b->shape.end() - 1);
        std::vector<int64_t> os = sa;
        os.insert(os.end(), sb.begin(), sb.end());
        int64_t ra = std::max<int64_t>(1, shape_prod(sa)), rb = std::max<int64_t>(1, shape_prod(sb));
        std::vector<int64_t> out((size_t)(ra * rb), 0);
        for (int64_t i = 0; i < ra; ++i)
            for (int64_t j = 0; j < rb; ++j) {
                uint64_t acc = 0;
                for (int64_t k = 0; k < na; ++k)
                    acc += (uint64_t)ai[(size_t)(i * na + k)] * (uint64_t)bi[(size_t)(j * na + k)];
                out[(size_t)(i * rb + j)] = (int64_t)acc;
            }
        return from_int64s(out, os, dt);
    }
    auto ad = to_doubles(a, "inner");
    auto bd = to_doubles(b, "inner");
    if (a->shape.size() == 1 && b->shape.size() == 1) {
        if (a->shape[0] != b->shape[0])
            throw std::runtime_error("[litetorch Error] linalg::inner: size mismatch");
        double s = 0.0;
        for (size_t i = 0; i < ad.size(); ++i) s += ad[i] * bd[i];
        return from_doubles({s}, {}, dt);
    }
    if (a->shape.empty() || b->shape.empty())
        throw std::runtime_error("[litetorch Error] linalg::inner: bad shapes");
    int64_t na = a->shape.back(), nb = b->shape.back();
    if (na != nb) throw std::runtime_error("[litetorch Error] linalg::inner: last dim mismatch");
    std::vector<int64_t> sa(a->shape.begin(), a->shape.end() - 1);
    std::vector<int64_t> sb(b->shape.begin(), b->shape.end() - 1);
    std::vector<int64_t> os = sa;
    os.insert(os.end(), sb.begin(), sb.end());
    int64_t ra = std::max<int64_t>(1, shape_prod(sa)), rb = std::max<int64_t>(1, shape_prod(sb));
    std::vector<double> out((size_t)(ra * rb), 0.0);
    for (int64_t i = 0; i < ra; ++i)
        for (int64_t j = 0; j < rb; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < na; ++k)
                s += ad[(size_t)(i * na + k)] * bd[(size_t)(j * na + k)];
            out[(size_t)(i * rb + j)] = s;
        }
    return from_doubles(out, os, dt);
}

std::shared_ptr<Tensor> kron(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    DataType dt = promote_dtype(a, b);
    if (is_integer_dtype(dt)) {
        auto ai = to_int64s(a, "kron");
        auto bi = to_int64s(b, "kron");
        std::vector<int64_t> os;
        size_t nd = std::max(a->shape.size(), b->shape.size());
        std::vector<int64_t> pa(nd, 1), pb(nd, 1);
        for (size_t i = 0; i < a->shape.size(); ++i) pa[nd - a->shape.size() + i] = a->shape[i];
        for (size_t i = 0; i < b->shape.size(); ++i) pb[nd - b->shape.size() + i] = b->shape[i];
        for (size_t i = 0; i < nd; ++i) os.push_back(pa[i] * pb[i]);
        int64_t total = shape_prod(os);
        std::vector<int64_t> out((size_t)total);
        std::vector<int64_t> idx(nd, 0);
        for (int64_t t = 0; t < total; ++t) {
            int64_t tmp = t;
            for (size_t i = nd; i-- > 0;) {
                idx[i] = tmp % os[i];
                tmp /= os[i];
            }
            int64_t ia = 0, ib = 0;
            for (size_t i = 0; i < nd; ++i) {
                int64_t qa = idx[i] / pb[i], qb = idx[i] % pb[i];
                ia = ia * pa[i] + qa;
                ib = ib * pb[i] + qb;
            }
            out[(size_t)t] = mul_wrap(ai[(size_t)ia], bi[(size_t)ib]);
        }
        return from_int64s(out, os, dt);
    }
    auto ad = to_doubles(a, "kron");
    auto bd = to_doubles(b, "kron");
    std::vector<int64_t> os;
    size_t nd = std::max(a->shape.size(), b->shape.size());
    std::vector<int64_t> pa(nd, 1), pb(nd, 1);
    for (size_t i = 0; i < a->shape.size(); ++i) pa[nd - a->shape.size() + i] = a->shape[i];
    for (size_t i = 0; i < b->shape.size(); ++i) pb[nd - b->shape.size() + i] = b->shape[i];
    for (size_t i = 0; i < nd; ++i) os.push_back(pa[i] * pb[i]);
    int64_t total = shape_prod(os);
    std::vector<double> out((size_t)total);
    int64_t na = (int64_t)ad.size(), nb = (int64_t)bd.size();
    std::vector<int64_t> idx(nd, 0);
    for (int64_t t = 0; t < total; ++t) {
        int64_t tmp = t;
        for (size_t i = nd; i-- > 0;) {
            idx[i] = tmp % os[i];
            tmp /= os[i];
        }
        int64_t ia = 0, ib = 0;
        for (size_t i = 0; i < nd; ++i) {
            int64_t qa = idx[i] / pb[i], qb = idx[i] % pb[i];
            ia = ia * pa[i] + qa;
            ib = ib * pb[i] + qb;
        }
        out[(size_t)t] = ad[(size_t)ia] * bd[(size_t)ib];
    }
    (void)na;
    (void)nb;
    return from_doubles(out, os, dt);
}

std::shared_ptr<Tensor> vdot(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    DataType dt = same_dtype(a, b, "vdot");
    if (is_integer_dtype(dt)) {
        auto ai = to_int64s(a, "vdot");
        auto bi = to_int64s(b, "vdot");
        if (ai.size() != bi.size())
            throw std::runtime_error("[litetorch Error] linalg::vdot: size mismatch");
        uint64_t acc = 0;
        for (size_t i = 0; i < ai.size(); ++i) acc += (uint64_t)ai[i] * (uint64_t)bi[i];
        return from_int64s({(int64_t)acc}, {}, dt);
    }
    auto ad = to_doubles(a, "vdot");
    auto bd = to_doubles(b, "vdot");
    if (ad.size() != bd.size())
        throw std::runtime_error("[litetorch Error] linalg::vdot: size mismatch");
    double s = 0.0;
    for (size_t i = 0; i < ad.size(); ++i) s += ad[i] * bd[i];
    return from_doubles({s}, {}, dt);
}

}
}
