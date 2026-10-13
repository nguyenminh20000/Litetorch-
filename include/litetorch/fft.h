#ifndef LITETORCH_FFT_H
#define LITETORCH_FFT_H

#include "litetorch/tensor.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace litetorch {
namespace fft {

std::shared_ptr<Tensor> fft(std::shared_ptr<Tensor> x, int64_t n = -1, int64_t dim = -1,
                            const std::string& norm = "backward");
std::shared_ptr<Tensor> ifft(std::shared_ptr<Tensor> x, int64_t n = -1, int64_t dim = -1,
                             const std::string& norm = "backward");
std::shared_ptr<Tensor> rfft(std::shared_ptr<Tensor> x, int64_t n = -1, int64_t dim = -1,
                             const std::string& norm = "backward");
std::shared_ptr<Tensor> irfft(std::shared_ptr<Tensor> x, int64_t n = -1, int64_t dim = -1,
                              const std::string& norm = "backward");
std::shared_ptr<Tensor> fftn(std::shared_ptr<Tensor> x, const std::vector<int64_t>& s,
                             const std::vector<int64_t>& dims, const std::string& norm = "backward");
std::shared_ptr<Tensor> ifftn(std::shared_ptr<Tensor> x, const std::vector<int64_t>& s,
                              const std::vector<int64_t>& dims, const std::string& norm = "backward");
std::shared_ptr<Tensor> fftshift(std::shared_ptr<Tensor> x, const std::vector<int64_t>& dims);
std::shared_ptr<Tensor> ifftshift(std::shared_ptr<Tensor> x, const std::vector<int64_t>& dims);
std::shared_ptr<Tensor> fftfreq(int64_t n, double d = 1.0);

}
}

#endif
