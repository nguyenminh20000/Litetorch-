#include "litetorch/data.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace litetorch {
namespace data {

TensorDataset::TensorDataset(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y) : x(x), y(y) {}

size_t TensorDataset::size() {
    return x->shape[0];
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> TensorDataset::get(size_t index) {
    if (index >= size()) {
        throw std::out_of_range("Dataset index out of bounds");
    }

    std::vector<int64_t> item_shape_x(x->shape.begin() + 1, x->shape.end());
    if (item_shape_x.empty()) item_shape_x = {1};
    auto item_x = Tensor::create(item_shape_x, Device(DeviceType::CPU, 0), false, x->dtype);
    size_t elem_sz_x = x->storage->element_size();
    char* x_src = (char*)x->data_ptr() + index * x->strides[0] * elem_sz_x;
    std::memcpy(item_x->data_ptr(), x_src, item_x->numel() * elem_sz_x);
    if (x->device.type == DeviceType::GPU) {
        item_x = item_x->to(x->device);
    }

    std::vector<int64_t> item_shape_y(y->shape.begin() + 1, y->shape.end());
    if (item_shape_y.empty()) item_shape_y = {1};
    auto item_y = Tensor::create(item_shape_y, Device(DeviceType::CPU, 0), false, y->dtype);
    size_t elem_sz_y = y->storage->element_size();
    char* y_src = (char*)y->data_ptr() + index * y->strides[0] * elem_sz_y;
    std::memcpy(item_y->data_ptr(), y_src, item_y->numel() * elem_sz_y);
    if (y->device.type == DeviceType::GPU) {
        item_y = item_y->to(y->device);
    }

    return {item_x, item_y};
}

}
}
