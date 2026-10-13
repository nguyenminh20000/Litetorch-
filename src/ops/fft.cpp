#include "litetorch/fft.h"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace litetorch {
namespace fft {
namespace {

using cd = std::complex<double>;
const double PI = 3.14159265358979323846;

void require_cpu(const Tensor& t, const char* op) {
    if (t.device.type != DeviceType::CPU)
        throw std::runtime_error(std::string("[litetorch Error] fft::") + op +
                                 ": only CPU tensors supported");
}

void check_norm(const std::string& norm, const char* op) {
    if (norm != "backward" && norm != "forward" && norm != "ortho")
        throw std::runtime_error(std::string("[litetorch Error] fft::") + op +
                                 ": norm must be 'backward', 'forward' or 'ortho'");
}

int64_t norm_dim(int64_t dim, int64_t ndim, const char* op) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::runtime_error(std::string("[litetorch Error] fft::") + op + ": dim out of range");
    return dim;
}

bool is_pow2(int64_t n) { return n > 0 && (n & (n - 1)) == 0; }

int64_t next_pow2(int64_t n) {
    int64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

void fft_radix2(std::vector<cd>& a, bool inverse) {
    int64_t n = (int64_t)a.size();
    for (int64_t i = 1, j = 0; i < n; ++i) {
        int64_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[(size_t)i], a[(size_t)j]);
    }
    for (int64_t len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * PI / (double)len * (inverse ? 1.0 : -1.0);
        cd wlen(std::cos(ang), std::sin(ang));
        for (int64_t i = 0; i < n; i += len) {
            cd w(1.0, 0.0);
            for (int64_t j = 0; j < len / 2; ++j) {
                cd u = a[(size_t)(i + j)];
                cd v = a[(size_t)(i + j + len / 2)] * w;
                a[(size_t)(i + j)] = u + v;
                a[(size_t)(i + j + len / 2)] = u - v;
                w *= wlen;
            }
        }
    }
}

std::vector<cd> dft_bluestein(const std::vector<cd>& x, bool inverse) {
    int64_t n = (int64_t)x.size();
    if (n <= 1) return x;
    int64_t m = next_pow2(2 * n - 1);
    double sign = inverse ? 1.0 : -1.0;
    std::vector<cd> chirp((size_t)n);
    for (int64_t k = 0; k < n; ++k) {
        double ang = PI * (double)k * (double)k / (double)n * sign;
        chirp[(size_t)k] = cd(std::cos(ang), std::sin(ang));
    }
    std::vector<cd> a((size_t)m, cd(0.0, 0.0)), b((size_t)m, cd(0.0, 0.0));
    for (int64_t k = 0; k < n; ++k) a[(size_t)k] = x[(size_t)k] * chirp[(size_t)k];
    b[0] = cd(1.0, 0.0);
    for (int64_t k = 1; k < n; ++k) {
        b[(size_t)k] = std::conj(chirp[(size_t)k]);
        b[(size_t)(m - k)] = std::conj(chirp[(size_t)k]);
    }
    fft_radix2(a, false);
    fft_radix2(b, false);
    for (int64_t k = 0; k < m; ++k) a[(size_t)k] *= b[(size_t)k];
    fft_radix2(a, true);
    double mscale = 1.0 / (double)m;
    std::vector<cd> out((size_t)n);
    for (int64_t k = 0; k < n; ++k) out[(size_t)k] = a[(size_t)k] * mscale * chirp[(size_t)k];
    return out;
}

std::vector<cd> dft_1d(const std::vector<cd>& x, bool inverse) {
    if (x.empty()) return x;
    std::vector<cd> a = x;
    if (is_pow2((int64_t)a.size())) {
        fft_radix2(a, inverse);
        return a;
    }
    return dft_bluestein(x, inverse);
}

double norm_scale(const std::string& norm, int64_t n, bool inverse) {
    if (norm == "ortho") return 1.0 / std::sqrt((double)n);
    if (norm == "backward") return inverse ? 1.0 / (double)n : 1.0;
    return inverse ? 1.0 : 1.0 / (double)n;
}

std::vector<double> to_doubles(const std::shared_ptr<Tensor>& t, const char* op) {
    require_cpu(*t, op);
    auto c = t->is_contiguous() ? t : t->contiguous();
    return c->flat_doubles();
}

std::shared_ptr<Tensor> make_tensor(const std::vector<int64_t>& shape, DataType dtype) {
    return Tensor::create(shape, Device(DeviceType::CPU, 0), false, dtype);
}

void write_doubles(const std::shared_ptr<Tensor>& t, const std::vector<double>& d) {
    char* base = reinterpret_cast<char*>(t->storage->get_cpu_ptr());
    if (t->dtype == DataType::FP64) {
        double* p = reinterpret_cast<double*>(base);
        for (size_t i = 0; i < d.size(); ++i) p[i] = d[i];
    } else {
        float* p = reinterpret_cast<float*>(base);
        for (size_t i = 0; i < d.size(); ++i) p[i] = static_cast<float>(d[i]);
    }
}

DataType out_dtype(const std::shared_ptr<Tensor>& t) {
    return t->dtype == DataType::FP64 ? DataType::FP64 : DataType::FP32;
}

int64_t shape_prod(const std::vector<int64_t>& s, int64_t from, int64_t to) {
    int64_t p = 1;
    for (int64_t i = from; i < to; ++i) p *= s[(size_t)i];
    return p;
}

std::vector<double> complex_fft_along_dim(const std::vector<double>& in,
                                          const std::vector<int64_t>& cshape, int64_t dim,
                                          int64_t n, bool inverse, double scale,
                                          int64_t n_out) {
    int64_t ndim = (int64_t)cshape.size();
    int64_t n_in = cshape[(size_t)dim];
    int64_t outer = shape_prod(cshape, 0, dim);
    int64_t inner = shape_prod(cshape, dim + 1, ndim);
    std::vector<double> out((size_t)(outer * n_out * inner * 2));
    std::vector<cd> buf((size_t)n);
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                if (j < n_in) {
                    size_t si = (size_t)(((o * n_in + j) * inner + i) * 2);
                    buf[(size_t)j] = cd(in[si], in[si + 1]);
                } else {
                    buf[(size_t)j] = cd(0.0, 0.0);
                }
            }
            std::vector<cd> r = dft_1d(buf, inverse);
            for (int64_t k = 0; k < n_out; ++k) {
                size_t di = (size_t)(((o * n_out + k) * inner + i) * 2);
                out[di] = r[(size_t)k].real() * scale;
                out[di + 1] = r[(size_t)k].imag() * scale;
            }
        }
    }
    return out;
}

}

std::shared_ptr<Tensor> fft(std::shared_ptr<Tensor> x, int64_t n, int64_t dim,
                            const std::string& norm) {
    check_norm(norm, "fft");
    std::vector<double> in = to_doubles(x, "fft");
    std::vector<int64_t> shape = x->shape;
    int64_t ndim = (int64_t)shape.size();
    if (ndim == 0)
        throw std::runtime_error("[litetorch Error] fft::fft: input must have at least 1 dim");
    dim = norm_dim(dim, ndim, "fft");
    int64_t n_in = shape[(size_t)dim];
    if (n == -1) n = n_in;
    if (n <= 0)
        throw std::runtime_error("[litetorch Error] fft::fft: n must be positive");
    std::vector<double> cin(in.size() * 2);
    for (size_t i = 0; i < in.size(); ++i) {
        cin[2 * i] = in[i];
        cin[2 * i + 1] = 0.0;
    }
    double scale = norm_scale(norm, n, false);
    std::vector<double> cout = complex_fft_along_dim(cin, shape, dim, n, false, scale, n);
    std::vector<int64_t> oshape = shape;
    oshape[(size_t)dim] = n;
    oshape.push_back(2);
    auto out = make_tensor(oshape, out_dtype(x));
    write_doubles(out, cout);
    return out;
}

std::shared_ptr<Tensor> ifft(std::shared_ptr<Tensor> x, int64_t n, int64_t dim,
                             const std::string& norm) {
    check_norm(norm, "ifft");
    std::vector<int64_t> shape = x->shape;
    int64_t ndim = (int64_t)shape.size();
    if (ndim < 1 || shape.back() != 2)
        throw std::runtime_error("[litetorch Error] fft::ifft: input must be complex (trailing dim 2)");
    std::vector<int64_t> cshape(shape.begin(), shape.end() - 1);
    int64_t cndim = (int64_t)cshape.size();
    if (cndim == 0)
        throw std::runtime_error("[litetorch Error] fft::ifft: input must have at least 1 complex dim");
    dim = norm_dim(dim, cndim, "ifft");
    int64_t n_in = cshape[(size_t)dim];
    if (n == -1) n = n_in;
    if (n <= 0)
        throw std::runtime_error("[litetorch Error] fft::ifft: n must be positive");
    std::vector<double> in = to_doubles(x, "ifft");
    double scale = norm_scale(norm, n, true);
    std::vector<double> cout = complex_fft_along_dim(in, cshape, dim, n, true, scale, n);
    std::vector<double> rout(cout.size() / 2);
    for (size_t i = 0; i < rout.size(); ++i) rout[i] = cout[2 * i];
    std::vector<int64_t> oshape = cshape;
    oshape[(size_t)dim] = n;
    auto out = make_tensor(oshape, out_dtype(x));
    write_doubles(out, rout);
    return out;
}

std::shared_ptr<Tensor> rfft(std::shared_ptr<Tensor> x, int64_t n, int64_t dim,
                             const std::string& norm) {
    check_norm(norm, "rfft");
    std::vector<double> in = to_doubles(x, "rfft");
    std::vector<int64_t> shape = x->shape;
    int64_t ndim = (int64_t)shape.size();
    if (ndim == 0)
        throw std::runtime_error("[litetorch Error] fft::rfft: input must have at least 1 dim");
    dim = norm_dim(dim, ndim, "rfft");
    int64_t n_in = shape[(size_t)dim];
    if (n == -1) n = n_in;
    if (n <= 0)
        throw std::runtime_error("[litetorch Error] fft::rfft: n must be positive");
    std::vector<double> cin(in.size() * 2);
    for (size_t i = 0; i < in.size(); ++i) {
        cin[2 * i] = in[i];
        cin[2 * i + 1] = 0.0;
    }
    double scale = norm_scale(norm, n, false);
    int64_t m = n / 2 + 1;
    std::vector<double> cout = complex_fft_along_dim(cin, shape, dim, n, false, scale, m);
    std::vector<int64_t> oshape = shape;
    oshape[(size_t)dim] = m;
    oshape.push_back(2);
    auto out = make_tensor(oshape, out_dtype(x));
    write_doubles(out, cout);
    return out;
}

std::shared_ptr<Tensor> irfft(std::shared_ptr<Tensor> x, int64_t n, int64_t dim,
                              const std::string& norm) {
    check_norm(norm, "irfft");
    std::vector<int64_t> shape = x->shape;
    int64_t ndim = (int64_t)shape.size();
    if (ndim < 1 || shape.back() != 2)
        throw std::runtime_error("[litetorch Error] fft::irfft: input must be complex (trailing dim 2)");
    std::vector<int64_t> cshape(shape.begin(), shape.end() - 1);
    int64_t cndim = (int64_t)cshape.size();
    if (cndim == 0)
        throw std::runtime_error("[litetorch Error] fft::irfft: input must have at least 1 complex dim");
    dim = norm_dim(dim, cndim, "irfft");
    int64_t m_in = cshape[(size_t)dim];
    int64_t n_out = (n == -1) ? 2 * (m_in - 1) : n;
    if (n_out <= 0)
        throw std::runtime_error("[litetorch Error] fft::irfft: n must be positive");
    int64_t m_use = std::min(m_in, n_out / 2 + 1);
    std::vector<double> in = to_doubles(x, "irfft");
    int64_t outer = shape_prod(cshape, 0, dim);
    int64_t inner = shape_prod(cshape, dim + 1, cndim);
    std::vector<double> full((size_t)(outer * n_out * inner * 2), 0.0);
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            for (int64_t k = 0; k < m_use; ++k) {
                size_t si = (size_t)(((o * m_in + k) * inner + i) * 2);
                size_t di = (size_t)(((o * n_out + k) * inner + i) * 2);
                full[di] = in[si];
                full[di + 1] = in[si + 1];
            }
            for (int64_t k = 1; k < n_out - m_use + 1; ++k) {
                size_t si = (size_t)(((o * m_in + k) * inner + i) * 2);
                size_t di = (size_t)(((o * n_out + (n_out - k)) * inner + i) * 2);
                full[di] = in[si];
                full[di + 1] = -in[si + 1];
            }
        }
    }
    std::vector<int64_t> fshape = cshape;
    fshape[(size_t)dim] = n_out;
    double scale = norm_scale(norm, n_out, true);
    std::vector<double> cout = complex_fft_along_dim(full, fshape, dim, n_out, true, scale, n_out);
    std::vector<double> rout(cout.size() / 2);
    for (size_t i = 0; i < rout.size(); ++i) rout[i] = cout[2 * i];
    std::vector<int64_t> oshape = cshape;
    oshape[(size_t)dim] = n_out;
    auto out = make_tensor(oshape, out_dtype(x));
    write_doubles(out, rout);
    return out;
}

std::shared_ptr<Tensor> fftn(std::shared_ptr<Tensor> x, const std::vector<int64_t>& s,
                             const std::vector<int64_t>& dims, const std::string& norm) {
    check_norm(norm, "fftn");
    std::vector<double> in = to_doubles(x, "fftn");
    std::vector<int64_t> cshape = x->shape;
    int64_t ndim = (int64_t)cshape.size();
    if (ndim == 0)
        throw std::runtime_error("[litetorch Error] fft::fftn: input must have at least 1 dim");
    std::vector<int64_t> adims = dims;
    if (adims.empty()) {
        for (int64_t d = 0; d < ndim; ++d) adims.push_back(d);
    } else {
        for (size_t i = 0; i < adims.size(); ++i) adims[i] = norm_dim(adims[i], ndim, "fftn");
    }
    if (!s.empty() && s.size() != adims.size())
        throw std::runtime_error("[litetorch Error] fft::fftn: s and dims must match in length");
    std::vector<double> cur(in.size() * 2);
    for (size_t i = 0; i < in.size(); ++i) {
        cur[2 * i] = in[i];
        cur[2 * i + 1] = 0.0;
    }
    for (size_t di = 0; di < adims.size(); ++di) {
        int64_t d = adims[di];
        int64_t n = s.empty() ? cshape[(size_t)d] : s[di];
        if (n <= 0)
            throw std::runtime_error("[litetorch Error] fft::fftn: n must be positive");
        double scale = norm_scale(norm, n, false);
        cur = complex_fft_along_dim(cur, cshape, d, n, false, scale, n);
        cshape[(size_t)d] = n;
    }
    std::vector<int64_t> oshape = cshape;
    oshape.push_back(2);
    auto out = make_tensor(oshape, out_dtype(x));
    write_doubles(out, cur);
    return out;
}

std::shared_ptr<Tensor> ifftn(std::shared_ptr<Tensor> x, const std::vector<int64_t>& s,
                              const std::vector<int64_t>& dims, const std::string& norm) {
    check_norm(norm, "ifftn");
    std::vector<int64_t> shape = x->shape;
    int64_t ndim = (int64_t)shape.size();
    if (ndim < 1 || shape.back() != 2)
        throw std::runtime_error("[litetorch Error] fft::ifftn: input must be complex (trailing dim 2)");
    std::vector<int64_t> cshape(shape.begin(), shape.end() - 1);
    int64_t cndim = (int64_t)cshape.size();
    if (cndim == 0)
        throw std::runtime_error("[litetorch Error] fft::ifftn: input must have at least 1 complex dim");
    std::vector<int64_t> adims = dims;
    if (adims.empty()) {
        for (int64_t d = 0; d < cndim; ++d) adims.push_back(d);
    } else {
        for (size_t i = 0; i < adims.size(); ++i) adims[i] = norm_dim(adims[i], cndim, "ifftn");
    }
    if (!s.empty() && s.size() != adims.size())
        throw std::runtime_error("[litetorch Error] fft::ifftn: s and dims must match in length");
    std::vector<double> cur = to_doubles(x, "ifftn");
    for (size_t di = 0; di < adims.size(); ++di) {
        int64_t d = adims[di];
        int64_t n = s.empty() ? cshape[(size_t)d] : s[di];
        if (n <= 0)
            throw std::runtime_error("[litetorch Error] fft::ifftn: n must be positive");
        double scale = norm_scale(norm, n, true);
        cur = complex_fft_along_dim(cur, cshape, d, n, true, scale, n);
        cshape[(size_t)d] = n;
    }
    std::vector<double> rout(cur.size() / 2);
    for (size_t i = 0; i < rout.size(); ++i) rout[i] = cur[2 * i];
    auto out = make_tensor(cshape, out_dtype(x));
    write_doubles(out, rout);
    return out;
}

std::shared_ptr<Tensor> roll_shift(std::shared_ptr<Tensor> x, const std::vector<int64_t>& dims,
                                   bool inverse, const char* op) {
    std::vector<double> in = to_doubles(x, op);
    std::vector<int64_t> shape = x->shape;
    int64_t ndim = (int64_t)shape.size();
    std::vector<int64_t> adims = dims;
    if (adims.empty()) {
        for (int64_t d = 0; d < ndim; ++d) adims.push_back(d);
    } else {
        for (size_t i = 0; i < adims.size(); ++i) adims[i] = norm_dim(adims[i], ndim, op);
    }
    std::vector<int64_t> shifts((size_t)ndim, 0);
    for (int64_t d : adims) {
        int64_t n = shape[(size_t)d];
        shifts[(size_t)d] = inverse ? (n + 1) / 2 : n / 2;
    }
    std::vector<int64_t> strides((size_t)ndim);
    int64_t st = 1;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        strides[(size_t)d] = st;
        st *= shape[(size_t)d];
    }
    std::vector<double> out(in.size());
    std::vector<int64_t> idx((size_t)ndim), sidx((size_t)ndim);
    for (size_t flat = 0; flat < in.size(); ++flat) {
        size_t rem = flat;
        for (int64_t d = 0; d < ndim; ++d) {
            idx[(size_t)d] = (int64_t)(rem / (size_t)strides[(size_t)d]);
            rem %= (size_t)strides[(size_t)d];
        }
        for (int64_t d = 0; d < ndim; ++d) {
            int64_t n = shape[(size_t)d];
            sidx[(size_t)d] = (idx[(size_t)d] - shifts[(size_t)d]) % n;
            if (sidx[(size_t)d] < 0) sidx[(size_t)d] += n;
        }
        size_t sflat = 0;
        for (int64_t d = 0; d < ndim; ++d)
            sflat += (size_t)sidx[(size_t)d] * (size_t)strides[(size_t)d];
        out[flat] = in[sflat];
    }
    auto t = make_tensor(shape, out_dtype(x));
    write_doubles(t, out);
    return t;
}

std::shared_ptr<Tensor> fftshift(std::shared_ptr<Tensor> x, const std::vector<int64_t>& dims) {
    return roll_shift(x, dims, false, "fftshift");
}

std::shared_ptr<Tensor> ifftshift(std::shared_ptr<Tensor> x, const std::vector<int64_t>& dims) {
    return roll_shift(x, dims, true, "ifftshift");
}

std::shared_ptr<Tensor> fftfreq(int64_t n, double d) {
    if (n <= 0)
        throw std::runtime_error("[litetorch Error] fft::fftfreq: n must be positive");
    if (d == 0.0)
        throw std::runtime_error("[litetorch Error] fft::fftfreq: d must be nonzero");
    double val = 1.0 / ((double)n * d);
    int64_t npos = (n - 1) / 2 + 1;
    std::vector<double> out((size_t)n);
    for (int64_t i = 0; i < npos; ++i) out[(size_t)i] = (double)i * val;
    for (int64_t i = npos; i < n; ++i) out[(size_t)i] = (double)(i - n) * val;
    auto t = make_tensor({n}, DataType::FP64);
    write_doubles(t, out);
    return t;
}

}
}
