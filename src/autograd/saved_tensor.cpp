#include "litetorch/autograd.h"
#include "litetorch/tensor.h"

namespace litetorch {

SavedTensor::SavedTensor(std::shared_ptr<Tensor> p) : ptr(p) {}

SavedTensor& SavedTensor::operator=(const std::shared_ptr<Tensor>& p) {
    ptr = p;
    return *this;
}

SavedTensor::operator std::shared_ptr<Tensor>() const {
    return ptr.lock();
}

std::shared_ptr<Tensor> SavedTensor::operator->() const {
    return ptr.lock();
}

bool SavedTensor::operator==(std::nullptr_t) const {
    return ptr.expired();
}

bool SavedTensor::operator!=(std::nullptr_t) const {
    return !ptr.expired();
}

bool SavedTensor::operator!() const {
    return ptr.expired();
}

SavedTensor::operator bool() const {
    return !ptr.expired();
}

}
