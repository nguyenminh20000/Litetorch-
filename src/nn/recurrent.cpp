#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <cmath>
#include <random>
#include <stdexcept>

namespace litetorch {

extern std::mt19937& get_generator();

namespace nn {

static void uniform_fill(const std::shared_ptr<Tensor>& t, float bound) {
    std::mt19937& gen = get_generator();
    std::uniform_real_distribution<float> dis(-bound, bound);
    float* d = t->data_ptr();
    for (size_t i = 0; i < t->numel(); ++i) d[i] = dis(gen);
}

static void cpu_only(const std::shared_ptr<Tensor>& t, const char* who) {
    if (t->device.type != DeviceType::CPU)
        throw std::runtime_error(std::string("[litetorch Error] ") + who +
                                 ": only CPU supported (cuDNN RNN GPU path not implemented)");
}

static std::shared_ptr<Tensor> step_input(const std::shared_ptr<Tensor>& x, int64_t t) {
    return Ops::narrow(x, 0, t, 1)->reshape({x->shape[1], x->shape[2]});
}

RNNCell::RNNCell(int input_size, int hidden_size, bool bias, const std::string& nonlinearity)
    : input_size(input_size), hidden_size(hidden_size), has_bias(bias), nonlinearity(nonlinearity) {
    if (nonlinearity != "tanh" && nonlinearity != "relu")
        throw std::runtime_error("[litetorch Error] RNNCell: nonlinearity must be tanh or relu");
    float k = 1.0f / std::sqrt(static_cast<float>(hidden_size));
    weight_ih = Tensor::create({hidden_size, input_size}, Device(DeviceType::CPU, 0), true);
    weight_hh = Tensor::create({hidden_size, hidden_size}, Device(DeviceType::CPU, 0), true);
    uniform_fill(weight_ih, k);
    uniform_fill(weight_hh, k);
    register_parameter("weight_ih", weight_ih);
    register_parameter("weight_hh", weight_hh);
    if (has_bias) {
        bias_ih = Tensor::create({hidden_size}, Device(DeviceType::CPU, 0), true);
        bias_hh = Tensor::create({hidden_size}, Device(DeviceType::CPU, 0), true);
        uniform_fill(bias_ih, k);
        uniform_fill(bias_hh, k);
        register_parameter("bias_ih", bias_ih);
        register_parameter("bias_hh", bias_hh);
    }
}

std::shared_ptr<Tensor> RNNCell::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> hx) {
    cpu_only(input, "RNNCell");
    auto out = Ops::add(Ops::matmul(input, Ops::t(weight_ih)), Ops::matmul(hx, Ops::t(weight_hh)));
    if (has_bias) out = Ops::add(Ops::add(out, bias_ih), bias_hh);
    return nonlinearity == "tanh" ? Ops::tanh(out) : Ops::relu(out);
}

std::shared_ptr<Tensor> RNNCell::forward(std::shared_ptr<Tensor> input) {
    auto hx = Tensor::zeros({input->shape[0], hidden_size}, input->device, false, input->dtype);
    return forward(input, hx);
}

std::vector<std::shared_ptr<Tensor>> RNNCell::parameters() { return Module::parameters(); }

void RNNCell::to(const Device& device) {
    weight_ih = weight_ih->to(device);
    weight_hh = weight_hh->to(device);
    if (bias_ih) bias_ih = bias_ih->to(device);
    if (bias_hh) bias_hh = bias_hh->to(device);
}

LSTMCell::LSTMCell(int input_size, int hidden_size, bool bias)
    : input_size(input_size), hidden_size(hidden_size), has_bias(bias) {
    float k = 1.0f / std::sqrt(static_cast<float>(hidden_size));
    weight_ih = Tensor::create({4 * hidden_size, input_size}, Device(DeviceType::CPU, 0), true);
    weight_hh = Tensor::create({4 * hidden_size, hidden_size}, Device(DeviceType::CPU, 0), true);
    uniform_fill(weight_ih, k);
    uniform_fill(weight_hh, k);
    register_parameter("weight_ih", weight_ih);
    register_parameter("weight_hh", weight_hh);
    if (has_bias) {
        bias_ih = Tensor::create({4 * hidden_size}, Device(DeviceType::CPU, 0), true);
        bias_hh = Tensor::create({4 * hidden_size}, Device(DeviceType::CPU, 0), true);
        uniform_fill(bias_ih, k);
        uniform_fill(bias_hh, k);
        register_parameter("bias_ih", bias_ih);
        register_parameter("bias_hh", bias_hh);
    }
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> LSTMCell::forward(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> hx, std::shared_ptr<Tensor> cx) {
    cpu_only(input, "LSTMCell");
    auto gates = Ops::add(Ops::matmul(input, Ops::t(weight_ih)), Ops::matmul(hx, Ops::t(weight_hh)));
    if (has_bias) gates = Ops::add(Ops::add(gates, bias_ih), bias_hh);
    auto i = Ops::sigmoid(Ops::narrow(gates, 1, 0, hidden_size));
    auto f = Ops::sigmoid(Ops::narrow(gates, 1, hidden_size, hidden_size));
    auto g = Ops::tanh(Ops::narrow(gates, 1, 2 * hidden_size, hidden_size));
    auto o = Ops::sigmoid(Ops::narrow(gates, 1, 3 * hidden_size, hidden_size));
    auto cy = Ops::add(Ops::mul(f, cx), Ops::mul(i, g));
    auto hy = Ops::mul(o, Ops::tanh(cy));
    return {hy, cy};
}

std::shared_ptr<Tensor> LSTMCell::forward(std::shared_ptr<Tensor> input) {
    auto hx = Tensor::zeros({input->shape[0], hidden_size}, input->device, false, input->dtype);
    auto cx = Tensor::zeros({input->shape[0], hidden_size}, input->device, false, input->dtype);
    return forward(input, hx, cx).first;
}

std::vector<std::shared_ptr<Tensor>> LSTMCell::parameters() { return Module::parameters(); }

void LSTMCell::to(const Device& device) {
    weight_ih = weight_ih->to(device);
    weight_hh = weight_hh->to(device);
    if (bias_ih) bias_ih = bias_ih->to(device);
    if (bias_hh) bias_hh = bias_hh->to(device);
}

GRUCell::GRUCell(int input_size, int hidden_size, bool bias)
    : input_size(input_size), hidden_size(hidden_size), has_bias(bias) {
    float k = 1.0f / std::sqrt(static_cast<float>(hidden_size));
    weight_ih = Tensor::create({3 * hidden_size, input_size}, Device(DeviceType::CPU, 0), true);
    weight_hh = Tensor::create({3 * hidden_size, hidden_size}, Device(DeviceType::CPU, 0), true);
    uniform_fill(weight_ih, k);
    uniform_fill(weight_hh, k);
    register_parameter("weight_ih", weight_ih);
    register_parameter("weight_hh", weight_hh);
    if (has_bias) {
        bias_ih = Tensor::create({3 * hidden_size}, Device(DeviceType::CPU, 0), true);
        bias_hh = Tensor::create({3 * hidden_size}, Device(DeviceType::CPU, 0), true);
        uniform_fill(bias_ih, k);
        uniform_fill(bias_hh, k);
        register_parameter("bias_ih", bias_ih);
        register_parameter("bias_hh", bias_hh);
    }
}

std::shared_ptr<Tensor> GRUCell::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> hx) {
    cpu_only(input, "GRUCell");
    auto gi = Ops::matmul(input, Ops::t(weight_ih));
    auto gh = Ops::matmul(hx, Ops::t(weight_hh));
    if (has_bias) {
        gi = Ops::add(gi, bias_ih);
        gh = Ops::add(gh, bias_hh);
    }
    auto r = Ops::sigmoid(Ops::add(Ops::narrow(gi, 1, 0, hidden_size), Ops::narrow(gh, 1, 0, hidden_size)));
    auto z = Ops::sigmoid(Ops::add(Ops::narrow(gi, 1, hidden_size, hidden_size), Ops::narrow(gh, 1, hidden_size, hidden_size)));
    auto n = Ops::tanh(Ops::add(Ops::narrow(gi, 1, 2 * hidden_size, hidden_size),
                                Ops::mul(r, Ops::narrow(gh, 1, 2 * hidden_size, hidden_size))));
    auto one = Tensor::create({1}, input->device, false, input->dtype);
    one->data_ptr()[0] = 1.0f;
    return Ops::add(Ops::mul(Ops::sub(one, z), n), Ops::mul(z, hx));
}

std::shared_ptr<Tensor> GRUCell::forward(std::shared_ptr<Tensor> input) {
    auto hx = Tensor::zeros({input->shape[0], hidden_size}, input->device, false, input->dtype);
    return forward(input, hx);
}

std::vector<std::shared_ptr<Tensor>> GRUCell::parameters() { return Module::parameters(); }

void GRUCell::to(const Device& device) {
    weight_ih = weight_ih->to(device);
    weight_hh = weight_hh->to(device);
    if (bias_ih) bias_ih = bias_ih->to(device);
    if (bias_hh) bias_hh = bias_hh->to(device);
}

RNN::RNN(int input_size, int hidden_size, int num_layers, const std::string& nonlinearity,
         bool bias, bool batch_first, float dropout, bool bidirectional)
    : input_size(input_size), hidden_size(hidden_size), num_layers(num_layers), has_bias(bias),
      batch_first(batch_first), dropout_p(dropout), bidirectional(bidirectional), nonlinearity(nonlinearity) {
    int dirs = bidirectional ? 2 : 1;
    for (int l = 0; l < num_layers; ++l) {
        for (int d = 0; d < dirs; ++d) {
            int in_s = (l == 0) ? input_size : hidden_size * dirs;
            auto cell = std::make_shared<RNNCell>(in_s, hidden_size, bias, nonlinearity);
            cells.push_back(cell);
            add_module("cells." + std::to_string(l * dirs + d), cell);
        }
    }
    if (dropout > 0.0f) {
        this->dropout = std::make_shared<Dropout>(dropout);
        add_module("dropout", this->dropout);
    }
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> RNN::forward_full(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> h0) {
    cpu_only(input, "RNN");
    auto x = batch_first ? Ops::permute(input, {1, 0, 2}) : input;
    int64_t S = x->shape[0], B = x->shape[1];
    int dirs = bidirectional ? 2 : 1;
    std::vector<std::shared_ptr<Tensor>> hiddens;
    for (int i = 0; i < num_layers * dirs; ++i) {
        if (h0) hiddens.push_back(Ops::narrow(h0, 0, i, 1)->reshape({B, (int64_t)hidden_size}));
        else hiddens.push_back(Tensor::zeros({B, (int64_t)hidden_size}, x->device, false, x->dtype));
    }
    std::vector<std::shared_ptr<Tensor>> hn;
    for (int l = 0; l < num_layers; ++l) {
        std::vector<std::shared_ptr<Tensor>> dir_outs;
        for (int d = 0; d < dirs; ++d) {
            auto cell = cells[l * dirs + d];
            auto h = hiddens[l * dirs + d];
            std::vector<std::shared_ptr<Tensor>> steps;
            for (int t = 0; t < S; ++t) {
                int64_t tt = (d == 0) ? t : (S - 1 - t);
                h = cell->forward(step_input(x, tt), h);
                steps.push_back(h);
            }
            if (d == 1) {
                std::vector<std::shared_ptr<Tensor>> rev(steps.rbegin(), steps.rend());
                steps.swap(rev);
            }
            hn.push_back(h);
            dir_outs.push_back(Ops::stack(steps, 0));
        }
        x = (dirs == 2) ? Ops::cat(dir_outs, 2) : dir_outs[0];
        if (dropout && l < num_layers - 1) x = dropout->forward(x);
    }
    auto out = batch_first ? Ops::permute(x, {1, 0, 2}) : x;
    return {out, Ops::stack(hn, 0)};
}

std::shared_ptr<Tensor> RNN::forward(std::shared_ptr<Tensor> input) {
    return forward_full(input, nullptr).first;
}

std::vector<std::shared_ptr<Tensor>> RNN::parameters() { return Module::parameters(); }

void RNN::to(const Device& device) {
    for (auto& c : cells) c->to(device);
}

void RNN::train() {
    Module::train();
    for (auto& c : cells) c->train();
    if (dropout) dropout->train();
}

void RNN::eval() {
    Module::eval();
    for (auto& c : cells) c->eval();
    if (dropout) dropout->eval();
}

std::vector<std::shared_ptr<Module>> RNN::children() {
    std::vector<std::shared_ptr<Module>> ch;
    for (auto& c : cells) ch.push_back(c);
    if (dropout) ch.push_back(dropout);
    return ch;
}

LSTM::LSTM(int input_size, int hidden_size, int num_layers, bool bias,
           bool batch_first, float dropout, bool bidirectional)
    : input_size(input_size), hidden_size(hidden_size), num_layers(num_layers), has_bias(bias),
      batch_first(batch_first), dropout_p(dropout), bidirectional(bidirectional) {
    int dirs = bidirectional ? 2 : 1;
    for (int l = 0; l < num_layers; ++l) {
        for (int d = 0; d < dirs; ++d) {
            int in_s = (l == 0) ? input_size : hidden_size * dirs;
            auto cell = std::make_shared<LSTMCell>(in_s, hidden_size, bias);
            cells.push_back(cell);
            add_module("cells." + std::to_string(l * dirs + d), cell);
        }
    }
    if (dropout > 0.0f) {
        this->dropout = std::make_shared<Dropout>(dropout);
        add_module("dropout", this->dropout);
    }
}

std::pair<std::shared_ptr<Tensor>, std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> LSTM::forward_full(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> h0, std::shared_ptr<Tensor> c0) {
    cpu_only(input, "LSTM");
    auto x = batch_first ? Ops::permute(input, {1, 0, 2}) : input;
    int64_t S = x->shape[0], B = x->shape[1];
    int dirs = bidirectional ? 2 : 1;
    std::vector<std::shared_ptr<Tensor>> hs, cs;
    for (int i = 0; i < num_layers * dirs; ++i) {
        hs.push_back(h0 ? Ops::narrow(h0, 0, i, 1)->reshape({B, (int64_t)hidden_size})
                        : Tensor::zeros({B, (int64_t)hidden_size}, x->device, false, x->dtype));
        cs.push_back(c0 ? Ops::narrow(c0, 0, i, 1)->reshape({B, (int64_t)hidden_size})
                        : Tensor::zeros({B, (int64_t)hidden_size}, x->device, false, x->dtype));
    }
    std::vector<std::shared_ptr<Tensor>> hn, cn;
    for (int l = 0; l < num_layers; ++l) {
        std::vector<std::shared_ptr<Tensor>> dir_outs;
        for (int d = 0; d < dirs; ++d) {
            auto cell = cells[l * dirs + d];
            auto h = hs[l * dirs + d], c = cs[l * dirs + d];
            std::vector<std::shared_ptr<Tensor>> steps;
            for (int t = 0; t < S; ++t) {
                int64_t tt = (d == 0) ? t : (S - 1 - t);
                auto r = cell->forward(step_input(x, tt), h, c);
                h = r.first;
                c = r.second;
                steps.push_back(h);
            }
            if (d == 1) {
                std::vector<std::shared_ptr<Tensor>> rev(steps.rbegin(), steps.rend());
                steps.swap(rev);
            }
            hn.push_back(h);
            cn.push_back(c);
            dir_outs.push_back(Ops::stack(steps, 0));
        }
        x = (dirs == 2) ? Ops::cat(dir_outs, 2) : dir_outs[0];
        if (dropout && l < num_layers - 1) x = dropout->forward(x);
    }
    auto out = batch_first ? Ops::permute(x, {1, 0, 2}) : x;
    return {out, {Ops::stack(hn, 0), Ops::stack(cn, 0)}};
}

std::shared_ptr<Tensor> LSTM::forward(std::shared_ptr<Tensor> input) {
    return forward_full(input, nullptr, nullptr).first;
}

std::vector<std::shared_ptr<Tensor>> LSTM::parameters() { return Module::parameters(); }

void LSTM::to(const Device& device) {
    for (auto& c : cells) c->to(device);
}

void LSTM::train() {
    Module::train();
    for (auto& c : cells) c->train();
    if (dropout) dropout->train();
}

void LSTM::eval() {
    Module::eval();
    for (auto& c : cells) c->eval();
    if (dropout) dropout->eval();
}

std::vector<std::shared_ptr<Module>> LSTM::children() {
    std::vector<std::shared_ptr<Module>> ch;
    for (auto& c : cells) ch.push_back(c);
    if (dropout) ch.push_back(dropout);
    return ch;
}

GRU::GRU(int input_size, int hidden_size, int num_layers, bool bias,
         bool batch_first, float dropout, bool bidirectional)
    : input_size(input_size), hidden_size(hidden_size), num_layers(num_layers), has_bias(bias),
      batch_first(batch_first), dropout_p(dropout), bidirectional(bidirectional) {
    int dirs = bidirectional ? 2 : 1;
    for (int l = 0; l < num_layers; ++l) {
        for (int d = 0; d < dirs; ++d) {
            int in_s = (l == 0) ? input_size : hidden_size * dirs;
            auto cell = std::make_shared<GRUCell>(in_s, hidden_size, bias);
            cells.push_back(cell);
            add_module("cells." + std::to_string(l * dirs + d), cell);
        }
    }
    if (dropout > 0.0f) {
        this->dropout = std::make_shared<Dropout>(dropout);
        add_module("dropout", this->dropout);
    }
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> GRU::forward_full(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> h0) {
    cpu_only(input, "GRU");
    auto x = batch_first ? Ops::permute(input, {1, 0, 2}) : input;
    int64_t S = x->shape[0], B = x->shape[1];
    int dirs = bidirectional ? 2 : 1;
    std::vector<std::shared_ptr<Tensor>> hiddens;
    for (int i = 0; i < num_layers * dirs; ++i) {
        if (h0) hiddens.push_back(Ops::narrow(h0, 0, i, 1)->reshape({B, (int64_t)hidden_size}));
        else hiddens.push_back(Tensor::zeros({B, (int64_t)hidden_size}, x->device, false, x->dtype));
    }
    std::vector<std::shared_ptr<Tensor>> hn;
    for (int l = 0; l < num_layers; ++l) {
        std::vector<std::shared_ptr<Tensor>> dir_outs;
        for (int d = 0; d < dirs; ++d) {
            auto cell = cells[l * dirs + d];
            auto h = hiddens[l * dirs + d];
            std::vector<std::shared_ptr<Tensor>> steps;
            for (int t = 0; t < S; ++t) {
                int64_t tt = (d == 0) ? t : (S - 1 - t);
                h = cell->forward(step_input(x, tt), h);
                steps.push_back(h);
            }
            if (d == 1) {
                std::vector<std::shared_ptr<Tensor>> rev(steps.rbegin(), steps.rend());
                steps.swap(rev);
            }
            hn.push_back(h);
            dir_outs.push_back(Ops::stack(steps, 0));
        }
        x = (dirs == 2) ? Ops::cat(dir_outs, 2) : dir_outs[0];
        if (dropout && l < num_layers - 1) x = dropout->forward(x);
    }
    auto out = batch_first ? Ops::permute(x, {1, 0, 2}) : x;
    return {out, Ops::stack(hn, 0)};
}

std::shared_ptr<Tensor> GRU::forward(std::shared_ptr<Tensor> input) {
    return forward_full(input, nullptr).first;
}

std::vector<std::shared_ptr<Tensor>> GRU::parameters() { return Module::parameters(); }

void GRU::to(const Device& device) {
    for (auto& c : cells) c->to(device);
}

void GRU::train() {
    Module::train();
    for (auto& c : cells) c->train();
    if (dropout) dropout->train();
}

void GRU::eval() {
    Module::eval();
    for (auto& c : cells) c->eval();
    if (dropout) dropout->eval();
}

std::vector<std::shared_ptr<Module>> GRU::children() {
    std::vector<std::shared_ptr<Module>> ch;
    for (auto& c : cells) ch.push_back(c);
    if (dropout) ch.push_back(dropout);
    return ch;
}

TransformerEncoderLayer::TransformerEncoderLayer(int d_model, int nhead, int dim_feedforward,
                                                 float dropout, const std::string& activation, bool batch_first)
    : d_model(d_model), nhead(nhead), dim_feedforward(dim_feedforward), dropout_p(dropout),
      activation(activation), batch_first(batch_first) {
    if (activation != "relu" && activation != "gelu")
        throw std::runtime_error("[litetorch Error] TransformerEncoderLayer: activation must be relu or gelu");
    self_attn = std::make_shared<MultiHeadAttention>(d_model, nhead, dropout, true, -1, -1, batch_first);
    linear1 = std::make_shared<Linear>(d_model, dim_feedforward);
    linear2 = std::make_shared<Linear>(dim_feedforward, d_model);
    norm1 = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model});
    norm2 = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model});
    dropout1 = std::make_shared<Dropout>(dropout);
    dropout2 = std::make_shared<Dropout>(dropout);
    dropout3 = std::make_shared<Dropout>(dropout);
    add_module("self_attn", self_attn);
    add_module("linear1", linear1);
    add_module("linear2", linear2);
    add_module("norm1", norm1);
    add_module("norm2", norm2);
    add_module("dropout1", dropout1);
    add_module("dropout2", dropout2);
    add_module("dropout3", dropout3);
}

std::shared_ptr<Tensor> TransformerEncoderLayer::forward(std::shared_ptr<Tensor> src,
                                                         std::shared_ptr<Tensor> src_mask,
                                                         std::shared_ptr<Tensor> src_key_padding_mask) {
    auto x = src;
    auto a = self_attn->forward(x, x, x, src_mask, src_key_padding_mask, false).first;
    x = norm1->forward(Ops::add(x, dropout1->forward(a)));
    auto f = linear2->forward(dropout2->forward(
        activation == "relu" ? Ops::relu(linear1->forward(x)) : Ops::gelu(linear1->forward(x))));
    return norm2->forward(Ops::add(x, dropout3->forward(f)));
}

std::shared_ptr<Tensor> TransformerEncoderLayer::forward(std::shared_ptr<Tensor> input) {
    return forward(input, nullptr, nullptr);
}

std::vector<std::shared_ptr<Tensor>> TransformerEncoderLayer::parameters() { return Module::parameters(); }

void TransformerEncoderLayer::to(const Device& device) {
    self_attn->to(device);
    linear1->to(device);
    linear2->to(device);
    norm1->to(device);
    norm2->to(device);
}

void TransformerEncoderLayer::train() {
    Module::train();
    self_attn->train();
    dropout1->train();
    dropout2->train();
    dropout3->train();
}

void TransformerEncoderLayer::eval() {
    Module::eval();
    self_attn->eval();
    dropout1->eval();
    dropout2->eval();
    dropout3->eval();
}

std::vector<std::shared_ptr<Module>> TransformerEncoderLayer::children() {
    return {self_attn, linear1, linear2, norm1, norm2, dropout1, dropout2, dropout3};
}

TransformerEncoder::TransformerEncoder(std::shared_ptr<TransformerEncoderLayer> encoder_layer,
                                       int num_layers, std::shared_ptr<LayerNorm> norm)
    : norm(norm) {
    layers = std::make_shared<ModuleList>();
    for (int i = 0; i < num_layers; ++i) {
        auto layer = std::make_shared<TransformerEncoderLayer>(
            encoder_layer->d_model, encoder_layer->nhead, encoder_layer->dim_feedforward,
            encoder_layer->dropout_p, encoder_layer->activation, encoder_layer->batch_first);
        layers->append(layer);
    }
    add_module("layers", layers);
    if (norm) add_module("norm", norm);
}

std::shared_ptr<Tensor> TransformerEncoder::forward(std::shared_ptr<Tensor> src,
                                                    std::shared_ptr<Tensor> mask,
                                                    std::shared_ptr<Tensor> src_key_padding_mask) {
    auto x = src;
    for (auto& m : layers->modules) {
        auto layer = std::static_pointer_cast<TransformerEncoderLayer>(m);
        x = layer->forward(x, mask, src_key_padding_mask);
    }
    if (norm) x = norm->forward(x);
    return x;
}

std::shared_ptr<Tensor> TransformerEncoder::forward(std::shared_ptr<Tensor> input) {
    return forward(input, nullptr, nullptr);
}

std::vector<std::shared_ptr<Tensor>> TransformerEncoder::parameters() { return Module::parameters(); }

void TransformerEncoder::to(const Device& device) {
    layers->to(device);
    if (norm) norm->to(device);
}

void TransformerEncoder::train() {
    Module::train();
    layers->train();
    if (norm) norm->train();
}

void TransformerEncoder::eval() {
    Module::eval();
    layers->eval();
    if (norm) norm->eval();
}

std::vector<std::shared_ptr<Module>> TransformerEncoder::children() {
    std::vector<std::shared_ptr<Module>> ch = {layers};
    if (norm) ch.push_back(norm);
    return ch;
}

EmbeddingBag::EmbeddingBag(int num_embeddings, int embedding_dim, const std::string& mode)
    : num_embeddings(num_embeddings), embedding_dim(embedding_dim), mode(mode) {
    if (mode != "sum" && mode != "mean" && mode != "max")
        throw std::runtime_error("[litetorch Error] EmbeddingBag: mode must be sum, mean or max");
    weight = Tensor::create({num_embeddings, embedding_dim}, Device(DeviceType::CPU, 0), true);
    uniform_fill(weight, 1.0f);
    register_parameter("weight", weight);
}

std::shared_ptr<Tensor> EmbeddingBag::forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> offsets) {
    cpu_only(input, "EmbeddingBag");
    int64_t nbags = offsets->shape[0];
    int64_t nidx = input->shape[0];
    auto offv = offsets->to_vector();
    auto idxv = input->to_vector();
    std::vector<std::shared_ptr<Tensor>> bags;
    for (int64_t b = 0; b < nbags; ++b) {
        int64_t start = static_cast<int64_t>(offv[b]);
        int64_t end = (b + 1 < nbags) ? static_cast<int64_t>(offv[b + 1]) : nidx;
        if (start < 0 || end > nidx || start > end)
            throw std::runtime_error("[litetorch Error] EmbeddingBag: offsets out of range");
        auto acc = Tensor::zeros({embedding_dim}, weight->device, false, weight->dtype);
        float* ad = acc->data_ptr();
        const float* wd = weight->data_ptr();
        int64_t count = end - start;
        if (mode == "max") {
            for (int64_t j = 0; j < embedding_dim; ++j) ad[j] = -1e30f;
            for (int64_t k = start; k < end; ++k) {
                int64_t r = static_cast<int64_t>(idxv[k]);
                for (int64_t j = 0; j < embedding_dim; ++j)
                    if (wd[r * embedding_dim + j] > ad[j]) ad[j] = wd[r * embedding_dim + j];
            }
            if (count == 0)
                for (int64_t j = 0; j < embedding_dim; ++j) ad[j] = 0.0f;
        } else {
            for (int64_t k = start; k < end; ++k) {
                int64_t r = static_cast<int64_t>(idxv[k]);
                if (r < 0 || r >= num_embeddings)
                    throw std::runtime_error("[litetorch Error] EmbeddingBag: index out of range");
                for (int64_t j = 0; j < embedding_dim; ++j) ad[j] += wd[r * embedding_dim + j];
            }
            if (mode == "mean" && count > 0)
                for (int64_t j = 0; j < embedding_dim; ++j) ad[j] /= static_cast<float>(count);
        }
        bags.push_back(acc);
    }
    if (bags.empty()) return Tensor::zeros({0, embedding_dim}, weight->device, false, weight->dtype);
    return Ops::stack(bags, 0);
}

std::shared_ptr<Tensor> EmbeddingBag::forward(std::shared_ptr<Tensor> input) {
    auto offsets = Tensor::zeros({1}, input->device, false, DataType::INT32);
    return forward(input, offsets);
}

std::vector<std::shared_ptr<Tensor>> EmbeddingBag::parameters() { return Module::parameters(); }

void EmbeddingBag::to(const Device& device) {
    weight = weight->to(device);
}

}
}
