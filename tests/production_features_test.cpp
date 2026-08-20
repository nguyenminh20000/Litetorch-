#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/optim.h"
#include "litetorch/amp.h"
#include "litetorch/fsdp.h"
#include "litetorch/quantization.h"
#include "litetorch/data.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace litetorch;

void test_datatype_and_cast() {
    std::cout << "Testing DataType and Cast..." << std::endl;
    auto t = Tensor::from_vector({1.5f, -2.5f, 0.0f, 10.25f}, {4});
    
    auto t_fp16 = t->cast(DataType::FP16);
    assert(t_fp16->dtype == DataType::FP16);
    
    auto t_fp32 = t_fp16->cast(DataType::FP32);
    assert(t_fp32->dtype == DataType::FP32);
    
    std::vector<float> vals = t_fp32->to_vector();
    assert(std::abs(vals[0] - 1.5f) < 1e-3);
    assert(std::abs(vals[1] - -2.5f) < 1e-3);
    assert(std::abs(vals[2] - 0.0f) < 1e-3);
    assert(std::abs(vals[3] - 10.25f) < 1e-3);

    auto t_int8 = t->cast(DataType::INT8);
    assert(t_int8->dtype == DataType::INT8);

    auto t_int8_fp32 = t_int8->cast(DataType::FP32);
    std::vector<float> i8_vals = t_int8_fp32->to_vector();
    assert(i8_vals[0] == 1.0f);
    assert(i8_vals[1] == -2.0f);
    assert(i8_vals[2] == 0.0f);
    assert(i8_vals[3] == 10.0f);

    auto t_fp64 = t->cast(DataType::FP64);
    assert(t_fp64->dtype == DataType::FP64);
    auto t_fp64_fp32 = t_fp64->cast(DataType::FP32);
    std::vector<float> f64_vals = t_fp64_fp32->to_vector();
    assert(std::abs(f64_vals[0] - 1.5f) < 1e-5);

    auto t_int16 = t->cast(DataType::INT16);
    assert(t_int16->dtype == DataType::INT16);
    auto t_int16_fp32 = t_int16->cast(DataType::FP32);
    std::vector<float> i16_vals = t_int16_fp32->to_vector();
    assert(i16_vals[0] == 1.0f);

    auto t_int4 = t->cast(DataType::INT4);
    assert(t_int4->dtype == DataType::INT4);
    auto t_int4_fp32 = t_int4->cast(DataType::FP32);
    std::vector<float> i4_vals = t_int4_fp32->to_vector();
    assert(i4_vals[0] == 1.0f);

    auto t_bf16 = t->cast(DataType::BF16);
    assert(t_bf16->dtype == DataType::BF16);
    auto t_bf16_fp32 = t_bf16->cast(DataType::FP32);
    std::vector<float> bf16_vals = t_bf16_fp32->to_vector();
    assert(std::abs(bf16_vals[0] - 1.5f) < 1e-2);
    
    std::cout << "DataType and Cast tests passed!" << std::endl;
}

void test_amp_and_gradscaler() {
    std::cout << "Testing AMP and GradScaler..." << std::endl;
    auto input = Tensor::from_vector({1.0f, 2.0f}, {1, 2}, Device(DeviceType::CPU, 0), true);
    auto weight = Tensor::from_vector({0.5f, -0.5f, 1.0f, 2.0f}, {2, 2}, Device(DeviceType::CPU, 0), true);
    
    auto input_h = Ops::cast(input, DataType::FP16);
    auto weight_h = Ops::cast(weight, DataType::FP16);
    
    auto w_t = weight_h->transpose(0, 1);
    auto out_h = Ops::matmul(input_h, w_t);
    auto out = Ops::cast(out_h, DataType::FP32);
    
    auto target = Tensor::from_vector({0.5f, 4.5f}, {1, 2});
    auto loss = Ops::mse_loss(out, target);
    
    amp::GradScaler scaler(1024.0f);
    auto scaled_loss = scaler.scale(loss);
    
    scaled_loss->backward();
    
    std::vector<std::shared_ptr<Tensor>> params = { input, weight };
    optim::SGD opt(params, 0.1f);
    
    float old_w = weight->to_vector()[0];
    scaler.step(opt);
    scaler.update();
    
    float new_w = weight->to_vector()[0];
    assert(new_w != old_w);
    
    std::cout << "AMP and GradScaler tests passed!" << std::endl;
}

void test_fsdp() {
    std::cout << "Testing FSDP..." << std::endl;
    auto linear = std::make_shared<nn::Linear>(4, 4, false);
    
    distributed::FullyShardedDataParallel fsdp(linear);
    
    auto p = linear->weight;
    assert(p->shape[0] == 16); 
    
    fsdp.gather_parameters();
    assert(p->shape[0] == 4);
    assert(p->shape[1] == 4);
    
    fsdp.shard_parameters();
    assert(p->shape[0] == 16);
    
    std::cout << "FSDP tests passed!" << std::endl;
}

void test_quantization() {
    std::cout << "Testing Weight-only Quantization..." << std::endl;
    auto linear = std::make_shared<nn::Linear>(4, 4, false);
    std::vector<float> orig_w = {
        0.5f, -0.2f, 0.8f, -0.1f,
        1.2f, 0.0f, -0.5f, 0.3f,
        -0.9f, 0.4f, 0.1f, -0.2f,
        0.6f, -0.7f, 0.3f, 0.5f
    };
    std::copy(orig_w.begin(), orig_w.end(), linear->weight->data_ptr());
    
    quantization::quantize_linear(linear);
    
    assert(linear->weight->dtype == DataType::INT8);
    assert(linear->scales != nullptr);
    assert(linear->scales->shape[0] == 4);
    
    auto input = Tensor::from_vector({1.0f, 1.0f, 1.0f, 1.0f}, {1, 4});
    auto out = linear->forward(input);
    
    std::vector<float> out_vals = out->to_vector();
    assert(out_vals.size() == 4);
    
    std::cout << "Weight-only Quantization tests passed!" << std::endl;
}

void test_qat_and_calibration() {
    std::cout << "Testing QAT and Calibration..." << std::endl;

    auto calibrator = std::make_shared<quantization::Calibrator>();
    auto t1 = Tensor::from_vector({-2.5f, 1.0f, 3.2f, -0.5f}, {1, 4});
    calibrator->collect("layer1_act", t1);

    float expected_scale = 3.2f / 127.0f;
    float scale = calibrator->get_scale("layer1_act", 8);
    assert(std::abs(scale - expected_scale) < 1e-5f);

    auto linear = std::make_shared<nn::Linear>(4, 4, false);
    std::vector<float> orig_w = {
        0.5f, -0.2f, 0.8f, -0.1f,
        1.2f, 0.0f, -0.5f, 0.3f,
        -0.9f, 0.4f, 0.1f, -0.2f,
        0.6f, -0.7f, 0.3f, 0.5f
    };
    std::copy(orig_w.begin(), orig_w.end(), linear->weight->data_ptr());

    auto qat_linear = std::make_shared<quantization::QATLinear>(linear, 0.01f, 0.01f, 8);
    qat_linear->qat_enabled = true;
    qat_linear->train();

    auto input = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4});
    input->requires_grad = true;

    auto out = qat_linear->forward(input);
    assert(out->requires_grad);

    auto loss = Ops::sum(out);
    loss->backward();

    assert(input->grad != nullptr);
    assert(qat_linear->weight->grad != nullptr);

    std::cout << "QAT and Calibration tests passed!" << std::endl;
}

void test_autocast() {
    std::cout << "Testing AutocastGuard..." << std::endl;
    auto a = Tensor::from_vector({1.5f, 2.5f}, {1, 2});
    auto b = Tensor::from_vector({2.0f, 3.0f}, {2, 1});
    
    assert(a->dtype == DataType::FP32);
    assert(b->dtype == DataType::FP32);
    
    {
        amp::AutocastGuard guard(true, DataType::FP16);
        auto out = Ops::matmul(a, b);
        assert(out->dtype == DataType::FP16);
    }
    
    {
        amp::AutocastGuard guard(true, DataType::BF16);
        auto out = Ops::matmul(a, b);
        assert(out->dtype == DataType::BF16);
    }
    
    std::cout << "AutocastGuard tests passed!" << std::endl;
}

void test_tensor_parallelism() {
    std::cout << "Testing ColumnParallelLinear and RowParallelLinear..." << std::endl;
    auto col_linear = std::make_shared<nn::ColumnParallelLinear>(4, 8, true);
    auto row_linear = std::make_shared<nn::RowParallelLinear>(4, 8, true);
    
    auto input = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4});
    
    auto out_col = col_linear->forward(input);
    assert(out_col->shape.size() == 2);
    assert(out_col->shape[0] == 1);
    assert(out_col->shape[1] == 8);
    
    auto out_row = row_linear->forward(input);
    assert(out_row->shape.size() == 2);
    assert(out_row->shape[0] == 1);
    assert(out_row->shape[1] == 8);
    
    std::cout << "ColumnParallelLinear and RowParallelLinear tests passed!" << std::endl;
}

void test_dataloader_checkpoint() {
    std::cout << "Testing DataLoader Checkpointing..." << std::endl;

    std::vector<float> x_data(100);
    std::vector<float> y_data(100);
    for (int i = 0; i < 100; ++i) {
        x_data[i] = static_cast<float>(i);
        y_data[i] = static_cast<float>(i * 10);
    }
    auto x_tensor = Tensor::from_vector(x_data, {100, 1});
    auto y_tensor = Tensor::from_vector(y_data, {100, 1});

    auto dataset = std::make_shared<data::TensorDataset>(x_tensor, y_tensor);
    data::DataLoader loader(dataset, 10, true, Device(DeviceType::CPU, 0), 2);

    std::shared_ptr<Tensor> bx1, by1;
    std::shared_ptr<Tensor> bx2, by2;
    std::shared_ptr<Tensor> bx3, by3;

    assert(loader.next(bx1, by1));
    assert(loader.next(bx2, by2));
    assert(loader.next(bx3, by3));

    loader.save_state("dataloader_checkpoint.bin");

    std::shared_ptr<Tensor> bx4, by4;
    std::shared_ptr<Tensor> bx5, by5;
    assert(loader.next(bx4, by4));
    assert(loader.next(bx5, by5));

    std::vector<float> x4_expected = bx4->to_vector();
    std::vector<float> x5_expected = bx5->to_vector();

    loader.load_state("dataloader_checkpoint.bin");

    std::shared_ptr<Tensor> bx4_new, by4_new;
    std::shared_ptr<Tensor> bx5_new, by5_new;
    assert(loader.next(bx4_new, by4_new));
    assert(loader.next(bx5_new, by5_new));

    std::vector<float> x4_actual = bx4_new->to_vector();
    std::vector<float> x5_actual = bx5_new->to_vector();

    std::cout << "Expected x4: ";
    for (float v : x4_expected) std::cout << v << " ";
    std::cout << "\nActual x4:   ";
    for (float v : x4_actual) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "Expected x5: ";
    for (float v : x5_expected) std::cout << v << " ";
    std::cout << "\nActual x5:   ";
    for (float v : x5_actual) std::cout << v << " ";
    std::cout << "\n";

    assert(x4_expected.size() == x4_actual.size());
    assert(x5_expected.size() == x5_actual.size());
    for (size_t i = 0; i < x4_expected.size(); ++i) {
        assert(x4_expected[i] == x4_actual[i]);
    }
    for (size_t i = 0; i < x5_expected.size(); ++i) {
        assert(x5_expected[i] == x5_actual[i]);
    }

    std::cout << "DataLoader Checkpointing tests passed!" << std::endl;
}

int main() {
    test_datatype_and_cast();
    test_amp_and_gradscaler();
    test_fsdp();
    test_quantization();
    test_qat_and_calibration();
    test_autocast();
    test_tensor_parallelism();
    test_dataloader_checkpoint();
    std::cout << "All Production Features Tests Passed Successfully!" << std::endl;
    return 0;
}
