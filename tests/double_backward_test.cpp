#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace litetorch;

int main() {
    auto x = Tensor::from_vector({3.0f}, {1}, Device(DeviceType::CPU), true);
    auto y = Ops::mul(x, x);
    
    y->backward(nullptr, true);
    
    assert(x->grad != nullptr);
    std::cout << "First derivative dy/dx: " << x->grad->item() << " (Expected: 6.0)" << std::endl;
    if (std::abs(x->grad->item() - 6.0f) >= 1e-5f) {
        std::cerr << "First derivative mismatch!" << std::endl;
        return 1;
    }
    
    auto grad = x->grad;
    x->zero_grad();
    grad->backward(nullptr);
    
    assert(x->grad != nullptr);
    std::cout << "Second derivative d2y/dx2: " << x->grad->item() << " (Expected: 2.0)" << std::endl;
    if (std::abs(x->grad->item() - 2.0f) >= 1e-5f) {
        std::cerr << "Second derivative mismatch!" << std::endl;
        return 1;
    }
    
    std::cout << "SUCCESS: Double backward verified!" << std::endl;
    return 0;
}
