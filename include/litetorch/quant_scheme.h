#ifndef LITETORCH_QUANT_SCHEME_H
#define LITETORCH_QUANT_SCHEME_H

#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include "litetorch/quantization.h"
#include "litetorch/serialization.h"
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace litetorch {
namespace quantization {

struct QParams {
    std::shared_ptr<Tensor> scales;
    std::shared_ptr<Tensor> zero_points;
    float scale = 1.0f;
    float zero_point = 0.0f;
    int bits = 8;
    int block_size = 0;
};

struct QuantizedTensor {
    std::shared_ptr<Tensor> data;
    QParams qparams;
    std::string scheme_name;
    std::map<std::string, std::string> scheme_params;
    std::vector<int64_t> logical_shape;
    DataType logical_dtype = DataType::FP32;
    bool packed = false;
    std::string nibble_order;
};

struct QuantScheme {
    std::string name;
    std::map<std::string, std::string> params;
    std::function<QParams(std::shared_ptr<Tensor>, const Calibrator*, const std::string&, const std::map<std::string, std::string>&)> calibrate;
    std::function<QuantizedTensor(std::shared_ptr<Tensor>, const QParams&, const std::map<std::string, std::string>&)> quantize;
    std::function<std::shared_ptr<Tensor>(const QuantizedTensor&)> dequantize;
    std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>, const QParams&, const std::map<std::string, std::string>&)> fake_quantize;
};

void register_scheme(const QuantScheme& scheme);
void unregister_scheme(const std::string& name);
bool has_scheme(const std::string& name);
QuantScheme get_scheme(const std::string& name);
std::vector<std::string> list_schemes();
void register_builtin_schemes();

std::shared_ptr<Tensor> scheme_fake_quant(const QuantScheme& scheme, std::shared_ptr<Tensor> x, const QParams& qp, const std::map<std::string, std::string>& params);

class SchemeQuantizedLinear : public nn::Module {
public:
    QuantizedTensor record;
    std::shared_ptr<Tensor> bias;
    bool dynamic_activation = false;

    SchemeQuantizedLinear(const QuantizedTensor& rec, std::shared_ptr<Tensor> b);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

class SchemeQATLinear : public nn::Module {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::string scheme_name;
    std::map<std::string, std::string> scheme_params;
    QParams weight_qparams;
    QParams act_qparams;
    bool act_calibrated = false;
    bool qat_enabled = true;

    SchemeQATLinear(std::shared_ptr<nn::Linear> linear, const std::string& scheme_name_, const std::map<std::string, std::string>& params = {});
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(const Device& device) override;
};

std::shared_ptr<nn::Module> quantize_model(std::shared_ptr<nn::Module> model, const std::string& scheme_name, const std::map<std::string, std::string>& params = {});
std::shared_ptr<nn::Module> quantize_ptq(std::shared_ptr<PreparedModel> prepared, const std::string& scheme_name, const std::map<std::string, std::string>& params = {});

void save_quantized(std::shared_ptr<nn::Module> model, const std::string& dir);
std::shared_ptr<nn::Module> load_quantized(std::shared_ptr<nn::Module> model_template, const std::string& dir);

}
}

#endif
